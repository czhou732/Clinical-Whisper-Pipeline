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

from cw_config import resolve_path

log = logging.getLogger("ClinicalWhisper")


def _free_ram():
    """Force garbage collection to release memory."""
    gc.collect()


class InferencePipeline:
    """Loads models on-demand per job and unloads after each stage."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        log.info("InferencePipeline v5.0 initialized (MOSS + OpenMED + MLX LLM)")

    def _archive_audio(self, job_id: str, source_path: Path, original_filename: str) -> str:
        processed_dir = Path(resolve_path(self.cfg.get("processed_folder", "./Processed")))
        processed_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(original_filename).name
        destination = processed_dir / f"{job_id}_{safe_name}"
        shutil.move(str(source_path), str(destination))
        return str(destination)

    @staticmethod
    def _decode_to_mono16k(source_path: Path) -> "np.ndarray":
        """Decode any container to float32 mono at 16 kHz using PyAV.

        PyAV ships its own ffmpeg libraries inside the wheel, so this works in a
        packaged .app with nothing installed on the host. Shelling out to an
        external `ffmpeg` binary was the last thing forcing users through a
        Homebrew install.
        """
        resampler = av.audio.resampler.AudioResampler(
            format="s16", layout="mono", rate=TARGET_SR
        )
        blocks: list[np.ndarray] = []

        with av.open(str(source_path)) as container:
            if not container.streams.audio:
                raise ValueError(f"No audio track found in {source_path.name}")
            stream = container.streams.audio[0]
            for frame in container.decode(stream):
                for resampled in resampler.resample(frame):
                    blocks.append(resampled.to_ndarray().reshape(-1))
            # Flush the resampler's internal buffer.
            for resampled in resampler.resample(None):
                blocks.append(resampled.to_ndarray().reshape(-1))

        if not blocks:
            raise ValueError(f"Decoded no audio from {source_path.name}")

        samples = np.concatenate(blocks).astype(np.float32) / 32768.0
        return samples

    @staticmethod
    def _normalise_for_asr(samples: "np.ndarray") -> "np.ndarray":
        """Bring quiet speech up to a consistent level for the ASR model.

        Replaces ffmpeg's `loudnorm` filter with RMS normalisation to
        TARGET_RMS_DBFS, backed off so the peak stays under TARGET_PEAK_DBFS.
        Only the transcription copy is touched.
        """
        rms = float(np.sqrt(np.mean(np.square(samples))))
        if rms <= 0.0:
            return samples

        # Reference the 99.9th percentile rather than the absolute maximum. A
        # single door slam or mic bump has a crest factor high enough to hold
        # the whole recording ~10 dB below target, which is exactly the quiet
        # audio MOSS returns an empty transcript on. The few samples above the
        # ceiling are clipped instead.
        peak_ref = float(np.percentile(np.abs(samples), 99.9))
        if peak_ref <= 0.0:
            peak_ref = float(np.max(np.abs(samples)))
        if peak_ref <= 0.0:
            return samples

        gain = min(
            (10.0 ** (TARGET_RMS_DBFS / 20.0)) / rms,
            (10.0 ** (TARGET_PEAK_DBFS / 20.0)) / peak_ref,
        )

        return np.clip(samples * gain, -1.0, 1.0).astype(np.float32)

    def _preprocess_audio(self, source_path: Path) -> tuple[Path, Path]:
        """Produce the two 16 kHz mono WAVs the pipeline needs.

        Returns ``(asr_wav, acoustic_wav)``:

        * ``asr_wav`` is level-normalised. Clinical recordings are often very
          quiet (the pilot recordings here average about -35 dBFS), and MOSS
          emits an immediate EOS — an empty transcript — on faint audio.
        * ``acoustic_wav`` keeps the original gain, because OpenSMILE's loudness
          and VTA features are amplitude-dependent and normalisation would
          invalidate them.
        """
        if av is None or sf is None or np is None:
            raise RuntimeError(
                "Audio decoding requires PyAV, soundfile and numpy. "
                "Reinstall dependencies with `uv pip install -e .`"
            )

        tmp_dir = Path(tempfile.gettempdir())
        acoustic_wav = tmp_dir / f"{source_path.stem}_16k.wav"
        asr_wav = tmp_dir / f"{source_path.stem}_16k_norm.wav"

        log.info("Decoding audio to 16kHz mono...")
        samples = self._decode_to_mono16k(source_path)
        sf.write(str(acoustic_wav), samples, TARGET_SR, subtype="PCM_16")

        log.info("Building level-normalised copy for transcription...")
        sf.write(str(asr_wav), self._normalise_for_asr(samples), TARGET_SR, subtype="PCM_16")
        del samples
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
        """
        Process one queue job and write `[job_id]_analysis.json`.
        Returns: Path to the JSON output file.
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
            )
            if not moss.is_available:
                raise RuntimeError(
                    "MOSS transcription model could not be loaded. Check that "
                    "moss-transcribe-diarize is installed and the model weights "
                    "downloaded."
                )
            segments = moss.process_file(str(asr_wav))
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
            "status": "completed_with_warnings" if warnings else "completed",
            "warnings": warnings,
            "pipeline_version": "5.0",
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

        archived_path = self._archive_audio(job_id, file_path, original_filename)
        payload["source_audio"]["archived_path"] = archived_path
        output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

        log.info("Job %s: wrote %s", job_id, output_path)
        return str(output_path)
