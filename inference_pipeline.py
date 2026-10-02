#!/usr/bin/env python3
"""
Inference worker pipeline for ClinicalWhisper v5.0.

Pipeline stages:
  1. Transcription & Diarization (MOSS)
  2. HIPAA PII Scrubbing (OpenMED)
  3. Acoustic feature extraction (WavLM/eGeMAPSv02 via OpenSMILE)
  4. Structured transcript formatting (Interviewer/Subject role detection)
  5. LLM clinical scoring (Llama-3 via MLX)

All processing is 100% local — no data ever leaves this machine.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import re
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import Optional

try:
    import nltk

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
except ImportError:
    nltk = None

try:
    import av
    import numpy as np
    import soundfile as sf
except ImportError:
    av = None
    np = None
    sf = None

# Everything downstream (MOSS, OpenSMILE) expects 16 kHz mono.
TARGET_SR = 16000
# Level targets for the transcription copy only.
TARGET_RMS_DBFS = -20.0
TARGET_PEAK_DBFS = -1.5
# Used only to turn a token count into an approximate progress fraction.
# Measured ~9 output tokens per second of conversational audio (timestamps and
# speaker tags included); overlapping windows add ~10%.
MOSS_TOKENS_PER_SECOND = 10.0
# Speakers whose voice features are extracted at the same time. Each worker
# holds one speaker's audio, so this also bounds the extra memory.
ACOUSTIC_WORKERS = min(4, max(1, (os.cpu_count() or 2) // 2))
# Status of an analysis saved before scoring finished (see score_job).
SCORING_PENDING = "scoring_pending"
# Files transcribed together are grouped up to about this much audio, or this
# many files (see InferencePipeline.iter_transcribed).
POOL_AUDIO_SECONDS = 2 * 3600
POOL_MAX_FILES = 32
# Resolution of the |amplitude| histogram used for the streaming percentile.
AMPLITUDE_BINS = 4096
# Samples per block when re-reading for normalisation (~1 s of audio).
DECODE_BLOCK = 16384

import crash_diagnostics
from version import __version__
from clinical_safeguards import (
    RESEARCH_USE_NOTICE,
    assess_quality,
    filevault_on,
    level_stats,
    score_reliability,
)
from cw_config import resolve_path

log = logging.getLogger("ClinicalWhisper")


def _overall_acoustics(wav_path: Path) -> dict:
    """Whole-file OpenSMILE features; runs on a worker thread during transcription."""
    from acoustic_features import AcousticExtractor

    extractor = AcousticExtractor()
    if not extractor.is_available():
        return {}
    return extractor.process_audio_file(str(wav_path))


# Checked once per process: whether outputs land on an encrypted disk.
_FILEVAULT = filevault_on()

# "[first_name_2]", "[city_1]", "[REDACTED]": one word each in the transcript.
_MASK_TOKEN = re.compile(r"\[[A-Za-z_]+(?:_\d+)?\]")

_SCRATCH_ROOT = Path(tempfile.gettempdir()) / "clinicalwhisper"


def _scratch_dir() -> Path:
    """This process's folder for decoded audio copies.

    One folder per process, so a stale-copy sweep never touches files another
    running instance (the app and a batch run, say) is still using.
    """
    path = _SCRATCH_ROOT / str(os.getpid())
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def clear_stale_scratch() -> int:
    """Delete decoded audio left behind by processes that are no longer running.

    A job that is killed (force quit, crash, power loss) never reaches its
    cleanup, and its 16 kHz copies carry identifiable voices. Returns the
    number of folders removed.
    """
    removed = 0
    if not _SCRATCH_ROOT.is_dir():
        return removed
    for entry in _SCRATCH_ROOT.iterdir():
        if not entry.is_dir() or not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            os.kill(int(entry.name), 0)  # still running: leave it alone
            continue
        except ProcessLookupError:
            pass
        except PermissionError:
            continue  # another user's live process
        shutil.rmtree(entry, ignore_errors=True)
        removed += 1
    if removed:
        log.info("Removed decoded audio left by %d interrupted run(s).", removed)
    return removed


def _guide_text(cfg: dict):
    """The study's interview guide, if one is configured (roles.guide_path)."""
    path = (cfg.get("roles") or {}).get("guide_path")
    if not path:
        return None
    try:
        return Path(resolve_path(path)).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        log.warning("Interview guide not readable (%s); roles use the other evidence.", exc)
        return None


def _scoring_gate(quality: dict, cfg: dict) -> str:
    """Why clinical scores shouldn't be made for this recording, or "".

    Version 1 rated a 30-second roll call as "consistent with anhedonic
    presentation". Scores need enough of the participant, clearly recorded.
    """
    sc = cfg.get("llm_scoring", {})
    min_speech = float(sc.get("min_participant_speech_s", 180))
    min_snr = float(sc.get("min_snr_db", 15))
    speech = quality.get("participant_speech_s") or 0.0
    snr = quality.get("snr_db")
    if speech < min_speech:
        return (f"Clinical scores were not made: {speech / 60:.1f} min of participant speech, "
                f"and scores need at least {min_speech / 60:.0f} min.")
    if isinstance(snr, (int, float)) and snr < min_snr:
        return (f"Clinical scores were not made: background noise is close to the voice "
                f"(SNR {snr:.0f} dB, below {min_snr:.0f} dB).")
    return ""


def _english(lang: dict) -> bool:
    """English, or too little text to tell (treated as before)."""
    import language
    return not lang or lang.get("code") == "und" or language.is_english(lang)


def _masker_for(lang: dict) -> dict:
    """Which masking model to use for a recording's language, or stop.

    English uses the bundled English model. Anything else needs the Languages
    add-on, and only for the languages its model was trained on: the English
    model run on Spanish or Chinese text misses most names, so writing that
    transcript would quietly break the de-identification promise.
    """
    import addons
    import language

    if language.is_english(lang) or lang.get("code") == "und":
        return {}
    name = lang.get("name", lang.get("code"))
    if lang.get("code") not in addons.MASKABLE:
        raise RuntimeError(
            f"This recording is in {name}. ClinicalWhisper can't mask names in {name} yet, "
            "so no transcript was written.")
    cache = addons.languages_cache()
    if cache is None:
        raise RuntimeError(
            f"This recording is in {name} (or mixes it with English). Masking names outside "
            "English needs the ClinicalWhisper Languages add-on, so no transcript was written. "
            "Install the add-on and process the file again.")
    import pii_scrubber
    from openmed.core.pii_i18n import SUPPORTED_LANGUAGES
    code = lang["code"] if lang["code"] in SUPPORTED_LANGUAGES else "en"
    return {"model_name": addons.LANGUAGES_MODEL, "lang": code, "cache_dir": str(cache)} \
        if pii_scrubber.deidentify is not None else {}


import kintsugi_dam  # noqa: E402  (after the module's own helpers it imports)


def _clinical_review(segments: list[dict], roles: dict, cfg: dict) -> dict:
    """Keyword screen for passages to review (see review_flags.py)."""
    review_cfg = cfg.get("review_flags", {}) or {}
    if review_cfg.get("enabled", True) is False:
        return {}
    import review_flags
    try:
        return review_flags.find(segments, roles, review_cfg.get("extra_terms"))
    except Exception as exc:  # noqa: BLE001 - a screen failure must not lose the run
        log.warning("Clinical review screen failed: %s", exc)
        return {"items": [], "counts": {}, "note": f"Screen failed: {exc}"}


def _scoring_installed(cfg: dict) -> bool:
    """Whether the clinical scoring model is on this Mac (see addons.py)."""
    import addons
    model = cfg.get("llm_scoring", {}).get("mlx_model", addons.SCORING_MODEL)
    return addons.scoring_available(model)


def _free_ram():
    """Force garbage collection to release memory."""
    gc.collect()


class InferencePipeline:
    """Loads models on-demand per job and unloads after each stage."""

    def __init__(self, cfg: dict, progress_cb=None, should_cancel=None):
        """
        Args:
            progress_cb: ``fn(stage: str, fraction: float | None, detail: str)``
                called as work proceeds, so the UI can show more than a frozen
                status line during the minutes-long transcription stage.
            should_cancel: ``fn() -> bool`` polled during long stages.
        """
        self.cfg = cfg
        self.progress_cb = progress_cb
        self.should_cancel = should_cancel
        self._acoustic_pool: Optional[ThreadPoolExecutor] = None
        try:
            clear_stale_scratch()
        except OSError as exc:  # never block a run over housekeeping
            log.warning("Could not clear stale audio copies: %s", exc)
        log.info("ClinicalWhisper %s pipeline initialized", __version__)

    def _emit(self, stage: str, fraction=None, detail: str = "") -> None:
        if stage != getattr(self, "_last_stage", None):
            # Breadcrumb for crash reports: a native crash or OOM kill leaves
            # no traceback, so the last recorded stage is the only evidence.
            self._last_stage = stage
            crash_diagnostics.mark_stage(stage, detail)
        if self.progress_cb is not None:
            try:
                self.progress_cb(stage, fraction, detail)
            except Exception:  # progress reporting must never break a job
                pass

    def _archive_audio(self, job_id: str, source_path: Path, original_filename: str) -> str:
        processed_dir = Path(resolve_path(self.cfg.get("processed_folder", "./Processed")))
        processed_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(original_filename).name
        destination = processed_dir / f"{job_id}_{safe_name}"
        shutil.move(str(source_path), str(destination))
        return str(destination)

    @staticmethod
    def _decode_stream(source_path: Path):
        """Yield float32 mono blocks at 16 kHz, without holding the whole file.

        PyAV ships its own ffmpeg libraries inside the wheel, so this works in a
        packaged .app with nothing installed on the host.

        Yielding rather than returning one array matters on real recordings: a
        two-hour interview is ~460 MB as float32, and building it with
        ``np.concatenate`` briefly doubled that.
        """
        resampler = av.audio.resampler.AudioResampler(
            format="s16", layout="mono", rate=TARGET_SR
        )
        produced = False

        with av.open(str(source_path)) as container:
            if not container.streams.audio:
                raise ValueError(f"No audio track found in {source_path.name}")
            stream = container.streams.audio[0]
            for frame in container.decode(stream):
                for resampled in resampler.resample(frame):
                    block = resampled.to_ndarray().reshape(-1)
                    if block.size:
                        produced = True
                        yield block.astype(np.float32) / 32768.0
            for resampled in resampler.resample(None):
                block = resampled.to_ndarray().reshape(-1)
                if block.size:
                    produced = True
                    yield block.astype(np.float32) / 32768.0

        if not produced:
            raise ValueError(f"Decoded no audio from {source_path.name}")

    @staticmethod
    def _asr_gain(sum_squares: float, count: int, hist: "np.ndarray") -> float:
        """Gain that lifts quiet speech to TARGET_RMS_DBFS for the ASR copy.

        The peak reference is the 99.9th percentile taken from a histogram of
        |x| accumulated during decoding, not the absolute maximum: one door slam
        has a crest factor high enough to hold an entire recording ~10 dB below
        target, which is exactly the quiet audio MOSS returns an empty
        transcript on. The few samples above the ceiling are clipped instead.

        Using a histogram keeps this O(1) in memory — an exact percentile would
        need every sample resident, which is what this change is removing.
        """
        if count == 0:
            return 1.0

        rms = float(np.sqrt(sum_squares / count))
        if rms <= 0.0:
            return 1.0

        total = hist.sum()
        if total <= 0:
            return 1.0
        cutoff = 0.999 * total
        cumulative = np.cumsum(hist)
        idx = int(np.searchsorted(cumulative, cutoff))
        idx = min(idx, AMPLITUDE_BINS - 1)
        # Upper edge of the bin, so the estimate never under-reports the peak.
        peak_ref = (idx + 1) / AMPLITUDE_BINS
        if peak_ref <= 0.0:
            return 1.0

        return min(
            (10.0 ** (TARGET_RMS_DBFS / 20.0)) / rms,
            (10.0 ** (TARGET_PEAK_DBFS / 20.0)) / peak_ref,
        )

    def _preprocess_audio(self, source_path: Path, tag: str = "",
                          edits=None) -> tuple[Path, Path, dict]:
        """Produce the two 16 kHz mono WAVs the pipeline needs.

        Returns ``(asr_wav, acoustic_wav, level_stats)``, the last being the
        original recording's level and clipping for the quality check:

        * ``asr_wav`` is level-normalised. Clinical recordings are often very
          quiet (the pilot recordings here average about -35 dBFS), and MOSS
          emits an immediate EOS — an empty transcript — on faint audio.
        * ``acoustic_wav`` keeps the original gain, because OpenSMILE's loudness
          and VTA features are amplitude-dependent and normalisation would
          invalidate them.

        Both are written block by block, so peak memory is a few MB whatever the
        length of the recording.
        """
        if av is None or sf is None or np is None:
            raise RuntimeError(
                "Audio decoding requires PyAV, soundfile and numpy. "
                "Reinstall dependencies with `uv pip install -e .`"
            )

        tmp_dir = _scratch_dir()
        # Prefixed with the job id: two recordings with the same name (P01/session.m4a,
        # P02/session.m4a, or interview.m4a beside interview.mp3) otherwise share
        # these paths, and in a batch one file is transcribed from the other's audio.
        prefix = f"{tag}_" if tag else ""
        acoustic_wav = tmp_dir / f"{prefix}{source_path.stem}_16k.wav"
        asr_wav = tmp_dir / f"{prefix}{source_path.stem}_16k_norm.wav"

        # Pass 1: decode straight to disk at original gain, accumulating the
        # statistics the ASR copy needs.
        log.info("Decoding audio to 16kHz mono...")
        sum_squares = 0.0
        count = 0
        clipped = 0
        hist = np.zeros(AMPLITUDE_BINS, dtype=np.int64)

        with sf.SoundFile(str(acoustic_wav), mode="w", samplerate=TARGET_SR,
                          channels=1, subtype="PCM_16") as out:
            blocks = self._decode_stream(source_path)
            if edits is not None:
                # Only the stretches the user kept (see audio_edits.py).
                import audio_edits
                blocks = audio_edits.apply(blocks, edits, TARGET_SR)
            for block in blocks:
                out.write(block)
                sum_squares += float(np.dot(block, block))
                count += block.size
                magnitudes = np.minimum(np.abs(block), 0.999999)
                clipped += int(np.count_nonzero(magnitudes >= 0.999))
                hist += np.bincount(
                    (magnitudes * AMPLITUDE_BINS).astype(np.int32),
                    minlength=AMPLITUDE_BINS,
                )[:AMPLITUDE_BINS]

        # Pass 2: re-read and apply a single gain, again block by block.
        gain = self._asr_gain(sum_squares, count, hist)
        log.info("Building level-normalised copy for transcription...")
        with sf.SoundFile(str(acoustic_wav)) as src, \
             sf.SoundFile(str(asr_wav), mode="w", samplerate=TARGET_SR,
                          channels=1, subtype="PCM_16") as out:
            while True:
                block = src.read(DECODE_BLOCK, dtype="float32")
                if not len(block):
                    break
                out.write(np.clip(block * gain, -1.0, 1.0))

        del hist
        _free_ram()

        return asr_wav, acoustic_wav, level_stats(sum_squares, count, clipped, TARGET_SR)

    def _extract_speaker_acoustics(self, extractor, wav_path: Path, segments: list[dict]) -> dict:
        """Extract acoustic features per speaker using streaming reads (low RAM)."""
        if not extractor or sf is None or np is None:
            return {}

        try:
            with sf.SoundFile(str(wav_path)) as audio_file:
                sr = audio_file.samplerate
                speaker_chunks: dict[str, list[np.ndarray]] = {}

                for seg in segments:
                    speaker = seg.get("speaker", "Speaker 1")
                    start_sample = int(seg.get("start", 0.0) * sr)
                    end_sample = int(seg.get("end", 0.0) * sr)
                    num_frames = end_sample - start_sample

                    if num_frames <= 0:
                        continue

                    audio_file.seek(start_sample)
                    # float32 holds 16-bit audio exactly, so features are unchanged
                    # and a long file's speech takes half the memory of float64.
                    chunk = audio_file.read(num_frames, dtype="float32")

                    if len(chunk) > 0:
                        speaker_chunks.setdefault(speaker, []).append(chunk)

            # Join each speaker's pieces, releasing them as we go so peak memory
            # is one copy of the speech rather than two.
            speaker_audio = {spk: np.concatenate(speaker_chunks.pop(spk))
                             for spk in list(speaker_chunks)}
            workers = min(len(speaker_audio), ACOUSTIC_WORKERS)

            def _one(item):
                speaker, audio = item
                # A separate OpenSMILE instance per thread: instances are not
                # documented as thread-safe. The computation is the same either
                # way; the library releases Python's lock while it runs, so
                # speakers are extracted side by side.
                ex = extractor if workers <= 1 else type(extractor)()
                return speaker, ex.process_audio_segment(audio, sr)

            if workers <= 1:
                results = [_one(item) for item in speaker_audio.items()]
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    results = list(pool.map(_one, speaker_audio.items()))
            del speaker_audio
            _free_ram()

            return dict(results)
        except Exception as e:
            log.warning("Failed to extract per-speaker acoustics: %s", e)
            return {}

    @staticmethod
    def _compute_statistics(transcript: str, segments: list[dict]) -> dict:
        words = transcript.split()
        # NLTK keeps "Dr." and "D.C." from inflating the count; the naive
        # punctuation tally is only a fallback when punkt is unavailable.
        if nltk is not None:
            try:
                sentence_count = len(nltk.tokenize.sent_tokenize(transcript))
            except Exception:
                sentence_count = sum(1 for c in transcript if c in ".!?")
        else:
            sentence_count = sum(1 for c in transcript if c in ".!?")
        duration_seconds = 0.0
        if segments:
            duration_seconds = max(float(seg["end"]) for seg in segments)

        # Masked identifiers stay in the text as one tag per word, so the count
        # above matches what the person actually said. Analyses that would
        # rather not count them get the second figure.
        masked = sum(1 for w in words if _MASK_TOKEN.fullmatch(w.strip(".,;:!?\"'()")))
        return {
            "word_count": len(words),
            "word_count_excluding_masked": len(words) - masked,
            "masked_word_count": masked,
            "character_count": len(transcript),
            "sentence_count": sentence_count,
            "estimated_minutes": round(duration_seconds / 60.0, 2) if duration_seconds > 0 else round(len(words) / 150.0, 2),
            "duration_seconds": round(duration_seconds, 2),
        }

    def process_job(self, job: dict) -> str:
        """Run both halves for one file and write `[job_id]_analysis.json`.

        Kept for single-file and CLI use. A batch should call
        :meth:`transcribe_job` for every file, then :meth:`release_transcriber`,
        then :meth:`score_job` for every file — that way the transcription and
        scoring models are never resident at the same time.
        """
        state = self.transcribe_job(job)
        return self.score_job(state)

    def release_transcriber(self) -> None:
        """Free the transcription model (~1.8 GB at float16)."""
        try:
            import moss_diarizer
            moss_diarizer.unload_models()
        except Exception as exc:  # pragma: no cover - best effort
            log.debug("Could not release MOSS: %s", exc)
        _free_ram()

    def release_scorer(self) -> None:
        """Free the clinical scoring model (~4.5 GB for Llama-3-8B at 4-bit)."""
        try:
            import llm_clinical_scorer
            llm_clinical_scorer.unload_models()
        except Exception as exc:  # pragma: no cover - best effort
            log.debug("Could not release the scoring model: %s", exc)
        _free_ram()

    def release_all(self) -> None:
        self.release_transcriber()
        self.release_scorer()
        if self._acoustic_pool is not None:
            self._acoustic_pool.shutdown(wait=True)
            self._acoustic_pool = None

    def transcribe_job(self, job: dict) -> dict:
        """First half for one file: decode, transcribe, de-identify, acoustics.

        Returns the intermediate state that :meth:`score_job` consumes. Nothing
        here touches the scoring model.
        """
        result = self.transcribe_jobs([job])[0]
        if isinstance(result, Exception):
            raise result
        return result

    def transcribe_jobs(self, jobs: list[dict]) -> list:
        """First half for many files. See :meth:`iter_transcribed`.

        Returns, per job, the state for :meth:`score_job` or the exception that
        file raised — one bad file does not sink the batch.
        """
        results: list = [None] * len(jobs)
        for group in self.iter_transcribed(jobs):
            for index, outcome in group:
                results[index] = outcome
        return results

    def iter_transcribed(self, jobs: list[dict]):
        """Transcribe files in groups, yielding one list of
        ``(job index, state or exception)`` per group.

        Files in a group share decoding batches (see
        ``moss_windowed.transcribe_many``), so a folder of short interviews is
        decoded at batch speed rather than one file at a time. Groups are capped
        at about :data:`POOL_AUDIO_SECONDS` of audio: every started file holds
        two temporary WAVs on disk until it is finished, so starting all 220
        files of a multi-day batch at once would need ~60 GB of scratch space,
        and a crash would lose every result. Yielding per group lets a caller
        save each file's output as soon as it exists, and swap models between
        groups.
        """
        group: list[tuple[int, dict]] = []
        failed: list[tuple[int, Exception]] = []
        group_audio = 0.0
        for i, job in enumerate(jobs):
            try:
                prep = self._start_job(job)
            except Exception as e:
                failed.append((i, e))
                continue
            group.append((i, prep))
            group_audio += prep["audio_seconds"] or 0.0
            if group_audio >= POOL_AUDIO_SECONDS or len(group) >= POOL_MAX_FILES:
                yield failed + list(self._transcribe_group(group))
                group, failed, group_audio = [], [], 0.0
        if group or failed:
            yield failed + (list(self._transcribe_group(group)) if group else [])

    def _transcribe_group(self, started: list[tuple[int, dict]]):
        try:
            moss = self._make_moss()
            total_audio = sum(p["audio_seconds"] or 0 for _, p in started)

            # The bar is an estimate, not a countdown — the token count next
            # to it is the honest number.
            def _moss_progress(tokens: int, _audio_seconds=None) -> None:
                if total_audio:
                    est = max(1.0, total_audio * MOSS_TOKENS_PER_SECOND)
                    self._emit("Transcribing", min(0.99, tokens / est), f"{tokens} tokens")
                else:
                    self._emit("Transcribing", None, f"{tokens} tokens")

            self._emit("Transcribing", 0.0, f"{len(started)} file(s)")
            all_segments = moss.process_files(
                [str(p["asr_wav"]) for _, p in started],
                progress_cb=_moss_progress,
                should_cancel=self.should_cancel,
            )
            self._emit("Transcribing", 1.0, f"{sum(len(x) for x in all_segments)} segments")
            del moss
            _free_ram()
        except Exception as e:
            for i, prep in started:
                self._cleanup(prep)
                yield i, e
            return

        for (i, prep), segments in zip(started, all_segments):
            try:
                if not segments:
                    # The model emits nothing on silent or near-silent audio.
                    raise RuntimeError(
                        "MOSS returned an empty transcript — the audio may be "
                        "silent, too quiet, or not speech."
                    )
                yield i, self._finish_job(prep, segments)
            except Exception as e:
                self._cleanup(prep)
                yield i, e

    def _make_moss(self):
        from moss_diarizer import MOSSDiarizer

        moss_cfg = self.cfg.get("moss", {})
        moss = MOSSDiarizer(
            model_name=moss_cfg.get("model", "OpenMOSS-Team/MOSS-Transcribe-Diarize"),
            device=moss_cfg.get("device"),
            dtype=moss_cfg.get("dtype"),
            hotwords=moss_cfg.get("hotwords"),
            **{
                k: v for k, v in moss_cfg.items()
                if k in (
                    "backend", "window_seconds", "window_overlap_seconds",
                    "batch_size", "speaker_similarity", "speaker_merge_similarity",
                    "speaker_split_similarity", "num_speakers",
                ) and v is not None
            },
        )
        # A failure here is fatal. Continuing with an empty segment list yields
        # a well-formed analysis JSON full of zeros and nulls, which is
        # indistinguishable from a genuine result.
        if not moss.is_available:
            raise RuntimeError(
                "MOSS transcription model could not be loaded. Check that "
                "moss-transcribe-diarize is installed and the model weights "
                "downloaded."
            )
        return moss

    def _start_job(self, job: dict) -> dict:
        """Validate and decode one file, and start its whole-file acoustics."""
        file_path = Path(job["file_path"]).expanduser()
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        log.info("Job %s: Processing %s", job["job_id"], file_path.name)

        # Decoding is a hard requirement: MOSS needs level-normalised audio and
        # OpenSMILE needs 16 kHz mono PCM. Falling back to the raw file here
        # used to produce silently-empty analyses.
        import audio_edits
        edits = audio_edits.Edits.from_dict(job.get("audio_edits"))
        asr_wav, acoustic_wav, audio_stats = self._preprocess_audio(
            file_path, tag=job["job_id"], edits=edits)

        # Whole-file acoustics need only the audio, not the transcript, and use
        # the CPU while transcription uses the GPU — so start them now instead
        # of after transcription and de-identification.
        # One shared worker: a per-file pool ran OpenSMILE on every file of a
        # group at once, competing with the decoder for CPU.
        if self._acoustic_pool is None:
            self._acoustic_pool = ThreadPoolExecutor(max_workers=1)
        from moss_diarizer import _audio_duration

        return {
            "job": job,
            "file_path": file_path,
            "asr_wav": asr_wav,
            "acoustic_wav": acoustic_wav,
            "audio_stats": audio_stats,
            "time_map": audio_edits.TimeMap.for_edits(edits),
            "overall_future": self._acoustic_pool.submit(_overall_acoustics, acoustic_wav),
            "audio_seconds": _audio_duration(str(asr_wav)),
        }

    @staticmethod
    def _cleanup(prep: dict) -> None:
        # The acoustics worker may still be reading acoustic_wav.
        future = prep["overall_future"]
        if not future.cancel():
            wait([future])
        for wav in (prep["asr_wav"], prep["acoustic_wav"]):
            if wav.exists():
                os.unlink(str(wav))

    def _finish_job(self, prep: dict, segments: list[dict]) -> dict:
        """De-identify, compute statistics and acoustics for one transcribed file."""
        job = prep["job"]
        file_path = prep["file_path"]
        acoustic_wav = prep["acoustic_wav"]
        # Non-fatal stage failures are recorded here and surfaced in the
        # output payload, so a degraded run is never mistaken for a clean one.
        warnings: list[str] = []

        # ── HIPAA Scrubbing (OpenMED) ──
        # Fatal when enabled: the app claims Safe Harbor de-identification,
        # so it must not emit a transcript that was never scrubbed.
        pii_cfg = self.cfg.get("pii_scrubbing", {})
        deid_summary: dict = {"enabled": False}
        # The language decides which masker can run (see language.py).
        import language
        lang = language.detect(" ".join(seg.get("text", "") for seg in segments))
        log.info("Job %s: language %s (%s)", job.get("job_id"), lang["name"], lang["code"])
        if pii_cfg.get("enabled", True):
            from pii_scrubber import PIIScrubber
            scrubber = PIIScrubber(
                confidence_threshold=pii_cfg.get("confidence_threshold", 0.7),
                strict=pii_cfg.get("strict", True),
                **_masker_for(lang),
            )
            if not scrubber.is_available:
                raise RuntimeError(
                    "PII scrubbing is enabled but the openmed package is not "
                    "installed. Install it, or set pii_scrubbing.enabled: false."
                )
            self._emit("De-identifying", None, f"{len(segments)} segments")
            # The original words are kept in memory, only long enough to find
            # where the masked names are spoken (audio_deid.py).
            raw_segments = [dict(seg) for seg in segments]
            segments = scrubber.scrub_segments(segments)
            deid_summary = scrubber.summary()
            del scrubber
            _free_ram()
        else:
            log.warning("PII scrubbing is DISABLED — transcript retains identifiers.")

        transcript = " ".join([seg.get("text", "") for seg in segments]).strip()
        stats = self._compute_statistics(transcript, segments)
        # Each masked word stays in the text as one tag ("Sarah Johnson" ->
        # "[first_name_1] [last_name_1]"), so word counts include masked names.
        stats["deidentification"] = deid_summary
        stats["language"] = lang

        # ── Acoustic extraction ──
        overall_acoustics = {}
        speaker_acoustics = {}
        try:
            from acoustic_features import AcousticExtractor
            extractor = AcousticExtractor()
            if extractor.is_available():
                log.info("Extracting acoustic features...")
                self._emit("Acoustics", None, "")
                # Original-gain audio: loudness/VTA features are amplitude-dependent.
                overall_acoustics = prep["overall_future"].result()
                speaker_acoustics = self._extract_speaker_acoustics(
                    extractor, acoustic_wav, segments
                )
            else:
                msg = "OpenSMILE is unavailable — no acoustic features extracted."
                log.error(msg)
                warnings.append(msg)
            del extractor
            _free_ram()
        except Exception as e:
            msg = f"Acoustic extraction failed: {e}"
            log.error(msg)
            warnings.append(msg)
        audio_stats = dict(prep.get("audio_stats") or {})
        from snr import estimate_snr
        audio_stats["snr_db"] = estimate_snr(str(acoustic_wav), segments)
        # Voice embeddings per speaker, in memory only: they recognise
        # remembered staff voices and let a confirmed moderator be remembered.
        voices = None
        try:
            import voice_library
            voices = voice_library.embeddings_for(segments, str(acoustic_wav))
        except Exception as exc:  # noqa: BLE001 - roles fall back to text evidence
            log.info("No voice embeddings for role matching: %s", exc)
        # Kintsugi's voice model (optional add-on), per speaker while the audio
        # is still here; the participant's result is picked once roles are known.
        voice_model = None
        try:
            import kintsugi_dam
            voice_model = kintsugi_dam.per_speaker(str(acoustic_wav), segments)
        except Exception as exc:  # noqa: BLE001 - an add-on must not lose the run
            log.warning("Kintsugi voice model failed: %s", exc)
            warnings.append(f"Kintsugi voice model failed: {exc}")
        # Praat voice measures (optional add-on, senselab / Bridge2AI definitions).
        praat = None
        try:
            import praat_measures
            praat = praat_measures.per_speaker(str(acoustic_wav), segments)
        except Exception as exc:  # noqa: BLE001 - an add-on must not lose the run
            log.warning("Praat measures failed: %s", exc)
        audio_deid_result = None
        if (self.cfg.get("audio_deid") or {}).get("enabled") and pii_cfg.get("enabled", True):
            audio_deid_result = self._silence_names(raw_segments, segments, acoustic_wav, job)
        raw_segments = None
        self._cleanup(prep)
        # Measured on the edited audio above; reported in the recording's own
        # time from here on, so timestamps match the file the user has.
        if prep.get("time_map") is not None:
            segments = prep["time_map"].segments(segments)

        return {
            "job": job,
            "job_id": job["job_id"],
            "file_path": file_path,
            "original_filename": job.get("original_filename", file_path.name),
            "segments": segments,
            "transcript": transcript,
            "stats": stats,
            "overall_acoustics": overall_acoustics,
            "speaker_acoustics": speaker_acoustics,
            "audio_stats": audio_stats,
            "voices": voices,
            "voice_model": voice_model,
            "praat": praat,
            "audio_deid": audio_deid_result,
            "warnings": warnings,
        }

    def _silence_names(self, raw_segments, segments, wav_path, job) -> dict:
        """Write a copy of the processed audio with every masked word silenced."""
        import audio_deid

        self._emit("Silencing names in the audio", None, "")
        try:
            found = audio_deid.silence_spans(raw_segments, segments, str(wav_path))
            out_dir = Path(resolve_path(self.cfg.get("pipeline", {}).get(
                "analysis_output_folder", self.cfg.get("output_folder", "./Output"))))
            stem = Path(job.get("original_filename") or wav_path).stem
            out = audio_deid.write(str(wav_path), found["spans"],
                                   out_dir / f"{job['job_id']}_{stem}_names_silenced.wav")
            return {"path": str(out), "silenced_spans": len(found["spans"]),
                    "segments_aligned": found["segments_aligned"],
                    "segments_silenced_whole": found["segments_silenced_whole"],
                    "aligner": "word-level" if audio_deid.available() else "whole segments",
                    "note": ("Names the transcript masker caught are silent in this copy; names it "
                             "missed are not. Listen before sharing. 16 kHz mono, and if parts were "
                             "left out before processing, those parts are not in it.")}
        except Exception as exc:  # noqa: BLE001 - the transcript must not be lost over this
            log.warning("Silencing names in the audio failed: %s", exc)
            return {"error": str(exc)}

    def score_job(self, state: dict) -> str:
        """Second half: role detection, clinical scoring, write the analysis.

        Nothing here touches the transcription model, so a batch can free it
        before this runs.
        """
        job = state["job"]
        job_id = state["job_id"]
        file_path = state["file_path"]
        original_filename = state["original_filename"]
        segments = state["segments"]
        transcript = state["transcript"]
        stats = state["stats"]
        overall_acoustics = state["overall_acoustics"]
        speaker_acoustics = state["speaker_acoustics"]
        warnings = state["warnings"]

        # ── Stage 4: Structured Transcript (role detection + formatting) ──
        notes: list[str] = []
        structured_transcript, speaker_roles, speaker_stats = "", {}, {}
        assignment: dict = {"mode": "interview", "uncertain": False}
        try:
            import speaker_roles as roles_mod
            import voice_library
            from transcript_formatter import compute_speaker_stats, format_structured_transcript

            assignment = roles_mod.assign(segments, _guide_text(self.cfg),
                                          state.get("voices"), voice_library.load())
            speaker_roles = assignment["roles"]
            structured_transcript = format_structured_transcript(segments, speaker_roles)
            speaker_stats = compute_speaker_stats(segments, speaker_roles)
            log.info("Job %s: %s, roles %s%s", job_id, assignment["mode"], speaker_roles,
                     " (uncertain)" if assignment["uncertain"] else "")
        except Exception as exc:
            log.warning("Job %s: structured transcript failed: %s", job_id, exc)
            warnings.append(f"Structured transcript failed: {exc}")
        group = assignment.get("mode") == "group"
        roles_uncertain = bool(assignment.get("uncertain"))

        # ── Stage 5a: Acoustic Context Serialization ──
        acoustic_context = ""
        try:
            from acoustic_context import build_acoustic_prompt_context
            acoustic_context = build_acoustic_prompt_context(
                overall_acoustics, speaker_acoustics
            )
        except Exception as exc:
            log.warning("Job %s: acoustic context serialization failed: %s", job_id, exc)

        # ── Provenance ──
        # Recorded per job so a result stays reproducible even if a model repo
        # is updated in place later.
        try:
            import provenance
            provenance_record = provenance.build(self.cfg)
        except Exception as exc:  # never fail a job over bookkeeping
            log.debug("Provenance unavailable: %s", exc)
            provenance_record = {}

        # ── Assemble output payload ──
        pipeline_cfg = self.cfg.get("pipeline", {})
        output_dir = Path(
            resolve_path(
                pipeline_cfg.get(
                    "analysis_output_folder", self.cfg.get("output_folder", "./Output")
                )
            )
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{job_id}_analysis.json"

        from timing_features import speaker_timing, subject_speaker

        # Pauses and latency on the edited timeline: on the original one a
        # skipped stretch would count as one enormous pause.
        import audio_edits
        edits = audio_edits.Edits.from_dict(job.get("audio_edits"))
        time_map = audio_edits.TimeMap.for_edits(edits)
        timing_segments = time_map.edited_segments(segments) if time_map else segments
        if edits is not None:
            notes.append(edits.describe())
        timing = speaker_timing(timing_segments)
        # No single participant in a group, and none to report while the roles
        # are uncertain: per-speaker measures stay in per_speaker.
        import elaboration
        lang = stats.get("language") or {}
        english = _english(lang)
        timing_payload = {
            # Words per answer after positive / neutral / negative questions.
            "elaboration": (elaboration.measure(segments, speaker_roles or {})
                            if english and not group and not roles_uncertain else None),
            "subject_speaker": (None if group or roles_uncertain
                                else subject_speaker(timing, speaker_roles or {})),
            "per_speaker": timing,
        }
        quality = assess_quality(
            segments, timing_payload["subject_speaker"], state.get("audio_stats"),
            expected_speakers=self.cfg.get("moss", {}).get("num_speakers"),
        )
        if roles_uncertain:
            quality["flags"].append({"code": "roles_uncertain", "message": (
                "It isn't clear who is interviewing. " + assignment.get("why", "") +
                " Participant measures and clinical scores wait until the roles are "
                "confirmed in the Speakers panel.")})
        if group:
            # Measures are per speaker in a group; "no participant" doesn't apply.
            quality["flags"] = [f for f in quality["flags"] if f["code"] not in
                                ("no_participant", "little_participant_speech")]
            notes.append("Group recording: moderators and participants are listed separately, "
                         "with measures for each speaker in timing_features.per_speaker. "
                         "Clinical scores are for one-to-one interviews and were not run.")
        lang = stats.get("language") or {}
        english = _english(lang)
        clinical_review = _clinical_review(segments, speaker_roles, self.cfg)
        if not english:
            # The keyword lists are English; an empty list would read as "nothing found".
            clinical_review = {"items": [], "counts": {}, "note": (
                f"The review keywords are English-only, so this {lang.get('name')} recording "
                "wasn't screened. The transcript needs to be read.")}
            for per in timing.values():
                per["filler_rate"] = None  # the filler list is English
            notes.append(f"Language: {lang.get('name')}. Filler counts and the review "
                         "keyword screen are English-only and were not run.")

        def _payload(llm_scoring: dict, reliability: dict, status: str = "") -> dict:
            return {
                "job_id": job_id,
                # Free-text study identifiers, so a results CSV can be grouped by
                # participant and session rather than by filename alone.
                "participant_id": job.get("participant_id", ""),
                "session_label": job.get("session_label", ""),
                # Criterion measure captured at recording time (e.g. "SHAPS 34").
                # Free text so it is not locked to one instrument. This is what makes
                # a later validity analysis possible — it cannot be retrofitted onto
                # audio that was collected without it.
                "criterion_score": job.get("criterion_score", ""),
                "status": status or ("completed_with_warnings" if warnings else "completed"),
                "warnings": warnings,
                # Deliberate choices worth recording, which are not problems.
                "notes": notes,
                "intended_use": RESEARCH_USE_NOTICE,
                # Input problems that make the measures less trustworthy.
                "quality": quality,
                # Measured test-retest reliability of each LLM score at this run count.
                "score_reliability": reliability,
                "storage": {"filevault": _FILEVAULT},
                "pipeline_version": __version__,
                "provenance": provenance_record,
                "source_audio": {
                    "original_filename": original_filename,
                    "stored_path": str(file_path),
                },
                "language": stats.get("language"),
                # Copy of the audio with masked names silenced, when asked for.
                "audio_deid": state.get("audio_deid"),
                # Stretches left out before processing; null when none.
                "audio_edits": edits.as_dict() if edits else None,
                "statistics": stats,
                "overall_acoustics": overall_acoustics,
                "speaker_acoustics": speaker_acoustics,
                # Deterministic pause / rate / latency measures from segment timing.
                "timing_features": timing_payload,
                "speaker_roles": speaker_roles,
                # How the roles were decided: mode, certainty, and the evidence.
                "speaker_assignment": {k: assignment.get(k) for k in
                                       ("mode", "uncertain", "why", "evidence")},
                "speaker_stats": speaker_stats,
                "structured_transcript": structured_transcript,
                # Keyword screen for passages a clinician should read; not scored.
                "clinical_review": clinical_review,
                # Praat measures (senselab definitions), if the add-on is installed.
                "praat_measures": ({"per_speaker": state.get("praat"),
                                    "subject": (state.get("praat") or {}).get(timing_payload["subject_speaker"]),
                                    "definitions": "senselab / Bridge2AI-Voice (Praat via parselmouth)"}
                                   if state.get("praat") is not None else None),
                # Kintsugi's open voice model, if installed: per speaker, and the
                # participant's estimate (research only).
                "kintsugi": ({"per_speaker": state.get("voice_model"),
                              **(kintsugi_dam.for_subject(state.get("voice_model"),
                                                          timing_payload["subject_speaker"]) or {})}
                             if state.get("voice_model") is not None else None),
                "llm_clinical_scoring": llm_scoring,
                "segments": segments,
                "transcript": transcript,
            }


        # Checkpoint: the transcript, timing and voice features are saved before
        # scoring starts, so a crash, Stop or sleep during a long scoring stage
        # does not throw them away. The batch command resumes scoring from here.
        llm_enabled = self.cfg.get("llm_scoring", {}).get("enabled", True)
        if llm_enabled and structured_transcript and (group or roles_uncertain):
            llm_enabled = False
        if llm_enabled and structured_transcript and not english:
            llm_enabled = False
            notes.append(f"Clinical scores were built and checked on English interviews, so "
                         f"they weren't run on this {lang.get('name')} recording.")
        # Only asked when scores are wanted and nothing above already ruled them out.
        gate = _scoring_gate(quality, self.cfg) if (llm_enabled and structured_transcript) else ""
        if gate:
            llm_enabled = False
            notes.append(gate)
        scoring_missing = bool(structured_transcript and llm_enabled
                               and not _scoring_installed(self.cfg))
        if scoring_missing:
            # Without the add-on the scorer would fail every run and fall back
            # to placeholder numbers that look like real scores.
            llm_enabled = False
            warnings.append("Clinical scoring isn't installed on this Mac, so the "
                            "scores are blank. Install the ClinicalWhisper Scoring add-on "
                            "to get them.")
        if structured_transcript and llm_enabled:
            output_path.write_text(json.dumps(
                _payload({}, {}, status=SCORING_PENDING), indent=2, default=str), encoding="utf-8")
            os.chmod(output_path, 0o600)

        # ── Stage 5b: LLM Clinical Scoring (MLX) ──
        llm_scoring = {}
        if structured_transcript and llm_enabled:
            try:
                from llm_clinical_scorer import score_transcript
                log.info("Job %s: running LLM clinical scoring...", job_id)
                self._emit("Clinical scoring", None, "")
                llm_scoring = score_transcript(
                    structured_transcript, acoustic_context, self.cfg,
                    progress=lambda frac, detail: self._emit("Clinical scoring", frac, detail),
                    should_cancel=self.should_cancel,
                )
                log.info("Job %s: LLM scoring complete", job_id)
            except Exception as exc:
                log.warning("Job %s: LLM clinical scoring failed: %s", job_id, exc)
                warnings.append(f"LLM clinical scoring failed: {exc}")
        elif not llm_enabled and not scoring_missing and english and not gate:
            # Asked for deliberately (the app's "transcribe only", the batch
            # command's --transcribe-only), this is a choice, not a fault: it
            # must not mark an otherwise clean run as "completed with warnings".
            # A group or unconfirmed roles already explain themselves above.
            msg = "Clinical scoring was skipped; the scores are blank."
            log.info("Job %s: %s", job_id, msg)
            if self.cfg.get("llm_scoring", {}).get("skipped_by_request"):
                notes.append(msg)
            elif not (group or roles_uncertain):
                warnings.append(msg)


        # Reliability at the number of runs actually averaged per chunk.
        meta = llm_scoring.get("_meta") or {}
        if meta.get("error"):
            # Every run failed: the scores are missing, and the result says so.
            warnings.append(f"Clinical scoring failed: {meta['error']}")
        if llm_scoring and not meta.get("error"):
            import consistency
            subject_t = (timing_payload.get("per_speaker") or {}).get(
                timing_payload.get("subject_speaker")) or {}
            quality["flags"].extend(consistency.check(
                " ".join([llm_scoring.get("summary") or ""] + list(llm_scoring.get("key_observations") or [])),
                subject_t))
        runs = meta.get("samples_per_window") or self.cfg.get("llm_scoring", {}).get("samples", 1)
        if meta.get("runs_used") and meta.get("windows"):
            runs = max(1, min(runs, meta["runs_used"] // meta["windows"]))
        reliability = score_reliability(runs) if llm_scoring else {}
        if meta.get("runs_failed"):
            quality["flags"].append({
                "code": "scoring_runs_failed",
                "message": (f"{meta['runs_failed']} of {meta['runs_expected']} scoring runs "
                            f"could not be used; reliability is shown for {runs} run(s) per chunk."),
            })

        payload = _payload(llm_scoring, reliability)

        output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        os.chmod(output_path, 0o600)

        # Retention: the transcript is de-identified but the source audio is
        # not — it carries identifiable voices and spoken names. Deleting it
        # here is the only way the finished analysis contains no PHI.
        retention = str(self.cfg.get("audio_retention", "archive")).lower()
        if retention == "delete":
            try:
                file_path.unlink()
                payload["source_audio"]["archived_path"] = None
                payload["source_audio"]["retention"] = "deleted"
                log.info("Job %s: source audio deleted (audio_retention: delete)", job_id)
            except OSError as exc:
                log.warning("Job %s: could not delete source audio: %s", job_id, exc)
                payload["source_audio"]["retention"] = "delete_failed"
        elif retention == "keep":
            # Leave the file where it is: batch runs over a study's originals.
            payload["source_audio"]["archived_path"] = None
            payload["source_audio"]["retention"] = "kept"
        else:
            archived_path = self._archive_audio(job_id, file_path, original_filename)
            payload["source_audio"]["archived_path"] = archived_path
            payload["source_audio"]["retention"] = "archived"
        output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

        log.info("Job %s: wrote %s", job_id, output_path)
        return str(output_path)
