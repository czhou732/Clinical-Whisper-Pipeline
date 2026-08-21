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
import shutil
import tempfile
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
MOSS_TOKENS_PER_SECOND = 5.0
# Resolution of the |amplitude| histogram used for the streaming percentile.
AMPLITUDE_BINS = 4096
# Samples per block when re-reading for normalisation (~1 s of audio).
DECODE_BLOCK = 16384

from cw_config import resolve_path

log = logging.getLogger("ClinicalWhisper")


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
        log.info("InferencePipeline v5.1 initialized (MOSS + OpenMED + MLX LLM)")

    def _emit(self, stage: str, fraction=None, detail: str = "") -> None:
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

    def _preprocess_audio(self, source_path: Path) -> tuple[Path, Path]:
        """Produce the two 16 kHz mono WAVs the pipeline needs.

        Returns ``(asr_wav, acoustic_wav)``:

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

        tmp_dir = Path(tempfile.gettempdir())
        acoustic_wav = tmp_dir / f"{source_path.stem}_16k.wav"
        asr_wav = tmp_dir / f"{source_path.stem}_16k_norm.wav"

        # Pass 1: decode straight to disk at original gain, accumulating the
        # statistics the ASR copy needs.
        log.info("Decoding audio to 16kHz mono...")
        sum_squares = 0.0
        count = 0
        hist = np.zeros(AMPLITUDE_BINS, dtype=np.int64)

        with sf.SoundFile(str(acoustic_wav), mode="w", samplerate=TARGET_SR,
                          channels=1, subtype="PCM_16") as out:
            for block in self._decode_stream(source_path):
                out.write(block)
                sum_squares += float(np.dot(block, block))
                count += block.size
                magnitudes = np.minimum(np.abs(block), 0.999999)
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

        return asr_wav, acoustic_wav

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
                    chunk = audio_file.read(num_frames)

                    if len(chunk) > 0:
                        speaker_chunks.setdefault(speaker, []).append(chunk)

            speaker_acoustics = {}
            for speaker, chunks in speaker_chunks.items():
                concatenated = np.concatenate(chunks)
                metrics = extractor.process_audio_segment(concatenated, sr)
                speaker_acoustics[speaker] = metrics
                del concatenated
            del speaker_chunks
            _free_ram()

            return speaker_acoustics
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

        return {
            "word_count": len(words),
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

    def transcribe_job(self, job: dict) -> dict:
        """First half: decode, transcribe, de-identify, acoustics.

        Returns the intermediate state that :meth:`score_job` consumes. Nothing
        here touches the scoring model.
        """
        job_id = job["job_id"]
        file_path = Path(job["file_path"]).expanduser()
        original_filename = job.get("original_filename", file_path.name)

        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        log.info("Job %s: Processing %s", job_id, file_path.name)

        # ── Preprocess ──
        # Decoding is a hard requirement: MOSS needs level-normalised audio and
        # OpenSMILE needs 16 kHz mono PCM. Falling back to the raw file here
        # used to produce silently-empty analyses.
        asr_wav, acoustic_wav = self._preprocess_audio(file_path)
        temp_wavs = [asr_wav, acoustic_wav]
        # Non-fatal stage failures are recorded here and surfaced in the
        # output payload, so a degraded run is never mistaken for a clean one.
        warnings: list[str] = []

        try:
            # ── Stage 1: Transcription & Diarization (MOSS) ──
            # A failure here is fatal. Continuing with an empty segment list
            # yields a well-formed analysis JSON full of zeros and nulls, which
            # is indistinguishable from a genuine result.
            from moss_diarizer import MOSSDiarizer
            moss = MOSSDiarizer(
                model_name=self.cfg.get("moss", {}).get(
                    "model", "OpenMOSS-Team/MOSS-Transcribe-Diarize"
                ),
                device=self.cfg.get("moss", {}).get("device"),
                dtype=self.cfg.get("moss", {}).get("dtype"),
                hotwords=self.cfg.get("moss", {}).get("hotwords"),
            )
            if not moss.is_available:
                raise RuntimeError(
                    "MOSS transcription model could not be loaded. Check that "
                    "moss-transcribe-diarize is installed and the model weights "
                    "downloaded."
                )
            # Rough: measured ~4 tokens per second of speech-dense audio, so
            # 5 keeps a typical file from pinning at 99% while silence-heavy
            # recordings finish early. The bar is an estimate, not a countdown —
            # the token count next to it is the honest number.
            def _moss_progress(tokens: int, audio_seconds) -> None:
                if audio_seconds:
                    est = max(1.0, audio_seconds * MOSS_TOKENS_PER_SECOND)
                    self._emit("Transcribing", min(0.99, tokens / est),
                               f"{tokens} tokens")
                else:
                    self._emit("Transcribing", None, f"{tokens} tokens")

            self._emit("Transcribing", 0.0, "starting")
            segments = moss.process_file(
                str(asr_wav),
                progress_cb=_moss_progress,
                should_cancel=self.should_cancel,
            )
            self._emit("Transcribing", 1.0, f"{len(segments)} segments")
            del moss
            _free_ram()

            # ── Stage 2: HIPAA Scrubbing (OpenMED) ──
            # Also fatal when enabled: the app claims Safe Harbor de-identification,
            # so it must not emit a transcript that was never scrubbed.
            pii_cfg = self.cfg.get("pii_scrubbing", {})
            if pii_cfg.get("enabled", True):
                from pii_scrubber import PIIScrubber
                scrubber = PIIScrubber(
                    confidence_threshold=pii_cfg.get("confidence_threshold", 0.7),
                    strict=pii_cfg.get("strict", True),
                )
                if not scrubber.is_available:
                    raise RuntimeError(
                        "PII scrubbing is enabled but the openmed package is not "
                        "installed. Install it, or set pii_scrubbing.enabled: false."
                    )
                self._emit("De-identifying", None, f"{len(segments)} segments")
                segments = scrubber.scrub_segments(segments)
                del scrubber
                _free_ram()
            else:
                log.warning("PII scrubbing is DISABLED — transcript retains identifiers.")
        except Exception:
            for wav in temp_wavs:
                if wav.exists():
                    os.unlink(str(wav))
            raise

        # Reconstruct Transcript
        transcript = " ".join([seg.get("text", "") for seg in segments]).strip()
        stats = self._compute_statistics(transcript, segments)

        # ── Stage 3: Acoustic extraction ──
        overall_acoustics = {}
        speaker_acoustics = {}
        try:
            from acoustic_features import AcousticExtractor
            extractor = AcousticExtractor()
            if extractor.is_available():
                log.info("Extracting acoustic features...")
                self._emit("Acoustics", None, "")
                # Original-gain audio: loudness/VTA features are amplitude-dependent.
                overall_acoustics = extractor.process_audio_file(str(acoustic_wav))
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

        for wav in temp_wavs:
            if wav.exists():
                os.unlink(str(wav))

        return {
            "job": job,
            "job_id": job_id,
            "file_path": file_path,
            "original_filename": original_filename,
            "segments": segments,
            "transcript": transcript,
            "stats": stats,
            "overall_acoustics": overall_acoustics,
            "speaker_acoustics": speaker_acoustics,
            "warnings": warnings,
        }

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
        structured_result = {}
        try:
            from transcript_formatter import process_segments
            structured_result = process_segments(segments)
            log.info("Job %s: structured transcript — roles: %s",
                     job_id, structured_result.get("roles", {}))
        except Exception as exc:
            log.warning("Job %s: structured transcript failed: %s", job_id, exc)
            warnings.append(f"Structured transcript failed: {exc}")

        structured_transcript = structured_result.get("structured_transcript", "")
        speaker_roles = structured_result.get("roles", {})
        speaker_stats = structured_result.get("speaker_stats", {})

        # ── Stage 5a: Acoustic Context Serialization ──
        acoustic_context = ""
        try:
            from acoustic_context import build_acoustic_prompt_context
            acoustic_context = build_acoustic_prompt_context(
                overall_acoustics, speaker_acoustics
            )
        except Exception as exc:
            log.warning("Job %s: acoustic context serialization failed: %s", job_id, exc)

        # ── Stage 5b: LLM Clinical Scoring (MLX) ──
        llm_scoring = {}
        llm_enabled = self.cfg.get("llm_scoring", {}).get("enabled", True)
        if structured_transcript and llm_enabled:
            try:
                from llm_clinical_scorer import score_transcript
                log.info("Job %s: running LLM clinical scoring...", job_id)
                self._emit("Clinical scoring", None, "")
                llm_scoring = score_transcript(
                    structured_transcript, acoustic_context, self.cfg
                )
                log.info("Job %s: LLM scoring complete", job_id)
            except Exception as exc:
                log.warning("Job %s: LLM clinical scoring failed: %s", job_id, exc)
                warnings.append(f"LLM clinical scoring failed: {exc}")
        elif not llm_enabled:
            msg = "llm_scoring.enabled is false — clinical scores are blank."
            log.warning("Job %s: %s", job_id, msg)
            warnings.append(msg)

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

        payload = {
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
            "status": "completed_with_warnings" if warnings else "completed",
            "warnings": warnings,
            "pipeline_version": "5.1",
            "provenance": provenance_record,
            "source_audio": {
                "original_filename": original_filename,
                "stored_path": str(file_path),
            },
            "statistics": stats,
            "overall_acoustics": overall_acoustics,
            "speaker_acoustics": speaker_acoustics,
            "speaker_roles": speaker_roles,
            "speaker_stats": speaker_stats,
            "structured_transcript": structured_transcript,
            "llm_clinical_scoring": llm_scoring,
            "segments": segments,
            "transcript": transcript,
        }

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
        else:
            archived_path = self._archive_audio(job_id, file_path, original_filename)
            payload["source_audio"]["archived_path"] = archived_path
            payload["source_audio"]["retention"] = "archived"
        output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

        log.info("Job %s: wrote %s", job_id, output_path)
        return str(output_path)
