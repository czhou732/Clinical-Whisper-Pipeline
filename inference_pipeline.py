#!/usr/bin/env python3
"""
Inference worker pipeline for ClinicalWhisper v4.0.

Pipeline stages:
  1. Transcription (MLX Whisper — Apple Silicon native)
  2. Speaker diarization (Pyannote)
  3. Sentiment analysis (RoBERTa transformer)
  4. Acoustic feature extraction (eGeMAPSv02 via OpenSMILE)
  5. Structured transcript formatting (Interviewer/Subject role detection)
  6. LLM clinical scoring + question detection (Ollama, local)
  7. LLM embedding extraction (HuggingFace, local)

All processing is 100% local — no data ever leaves this machine.
"""

from __future__ import annotations

import gc
import json
import logging
import nltk
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

try:
    import numpy as np
    import soundfile as sf
except ImportError:
    np = None
    sf = None

from cw_config import resolve_path
from sentiment_analyzer import analyze_per_speaker, analyze_sentiment, _get_segment_sentiment_score

# Ensure NLTK punkt tokenizer is available
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

log = logging.getLogger("ClinicalWhisper")


def _free_ram():
    """Force garbage collection to release memory."""
    gc.collect()


class InferencePipeline:
    """Loads models on-demand per job and unloads after each stage."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model_name = cfg.get("model", "small.en")
        self.mlx_model = cfg.get("mlx_model", f"mlx-community/whisper-{self.model_name}")
        # Models are NOT loaded at init — loaded per-job, unloaded after each stage
        log.info("InferencePipeline initialized (MLX Whisper: %s)", self.mlx_model)

    def _transcribe(self, audio_path: str) -> dict:
        """Transcribe locally. MLX Whisper on Apple Silicon, falls back to openai-whisper."""
        try:
            import mlx_whisper
            log.info("   📥 Transcribing with MLX Whisper (%s)...", self.mlx_model)
            return mlx_whisper.transcribe(audio_path, path_or_hf_repo=self.mlx_model, word_timestamps=True)
        except ImportError:
            import whisper
            log.info("   📥 Transcribing with openai-whisper (%s)...", self.model_name)
            model = whisper.load_model(self.model_name)
            result = model.transcribe(audio_path, word_timestamps=True, fp16=False)
            del model
            _free_ram()
            return result

    def _load_diarizer(self):
        """Load diarizer on demand. Returns diarizer or None."""
        diar_cfg = self.cfg.get("diarization", {})
        if not diar_cfg.get("enabled", False):
            return None

        try:
            from speaker_diarizer import SpeakerDiarizer
            diarizer = SpeakerDiarizer(
                hf_token=diar_cfg.get("hf_token"),
                min_speakers=diar_cfg.get("min_speakers", 2),
                max_speakers=diar_cfg.get("max_speakers", 6),
            )
            return diarizer
        except Exception as exc:
            log.warning("Diarization unavailable, continuing without it: %s", exc)
            return None

    def _unload_diarizer(self, diarizer):
        """Unload diarizer and free RAM."""
        if diarizer is None:
            return
        del diarizer
        _free_ram()
        log.info("   📤 Diarizer unloaded")

    def _load_acoustic_extractor(self):
        """Load acoustic extractor on demand."""
        if not self.cfg.get("acoustic_features", {}).get("enabled", False):
            return None
        from acoustic_features import AcousticExtractor
        extractor = AcousticExtractor()
        return extractor if extractor.is_available() else None

    def _build_segments(
        self, whisper_result: dict, merged_segments: Optional[list[dict]]
    ) -> list[dict]:
        segments: list[dict] = []

        if merged_segments:
            source_segments = merged_segments
        else:
            source_segments = [
                {
                    "start": s.get("start", 0.0),
                    "end": s.get("end", 0.0),
                    "speaker": "Speaker 1",
                    "text": (s.get("text") or "").strip(),
                }
                for s in whisper_result.get("segments", [])
            ]

        for seg in source_segments:
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            segments.append(
                {
                    "start": round(float(seg.get("start", 0.0)), 3),
                    "end": round(float(seg.get("end", 0.0)), 3),
                    "speaker": seg.get("speaker", "Speaker 1"),
                    "text": text,
                    "sentiment": _get_segment_sentiment_score(text),
                }
            )

        return segments

    @staticmethod
    def _compute_statistics(transcript: str, segments: list[dict]) -> dict:
        words = transcript.split()
        sentence_count = len(nltk.tokenize.sent_tokenize(transcript))
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

    def _archive_audio(self, job_id: str, source_path: Path, original_filename: str) -> str:
        processed_dir = Path(resolve_path(self.cfg.get("processed_folder", "./Processed")))
        processed_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(original_filename).name
        destination = processed_dir / f"{job_id}_{safe_name}"
        shutil.move(str(source_path), str(destination))
        return str(destination)

    def _preprocess_audio(self, source_path: Path) -> Path:
        """Convert audio to 16kHz mono WAV for Whisper & OpenSMILE."""
        tmp_wav = Path(tempfile.gettempdir()) / f"{source_path.stem}_16k.wav"
        cmd = [
            "ffmpeg", "-y", "-i", str(source_path),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            str(tmp_wav)
        ]
        log.info("Preprocessing audio to 16kHz mono WAV...")
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return tmp_wav

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
            gc.collect()

            return speaker_acoustics
        except Exception as e:
            log.warning("Failed to extract per-speaker acoustics: %s", e)
            return {}

    def process_job(self, job: dict) -> str:
        """
        Process one queue job and write `[job_id]_analysis.json`.

        Models are loaded on-demand per stage and unloaded after each stage
        to minimize peak RAM usage.

        Returns:
            Path to the JSON output file.
        """
        job_id = job["job_id"]
        file_path = Path(job["file_path"]).expanduser()
        original_filename = job.get("original_filename", file_path.name)

        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        log.info("Job %s: transcribing %s", job_id, file_path.name)

        # ── Preprocess ──
        wav_path = file_path
        tmp_wav = None
        try:
            tmp_wav = self._preprocess_audio(file_path)
            wav_path = tmp_wav
        except Exception as e:
            log.warning("Preprocessing failed, using original file: %s", e)

        # ── Stage 1: Transcription (MLX Whisper — native Apple Silicon) ──
        whisper_result = self._transcribe(str(wav_path))
        transcript = (whisper_result.get("text") or "").strip()

        # ── Stage 2: Diarization (load → diarize → unload) ──
        diarizer = self._load_diarizer()
        merged_segments = None
        speaker_texts = None
        if diarizer:
            try:
                diar_result = diarizer.diarize(str(wav_path))
                merged_segments = diarizer.merge_with_transcript(whisper_result, diar_result)
                speaker_texts = diarizer.group_by_speaker(merged_segments)
            except Exception as exc:
                log.warning("Job %s: diarization failed: %s", job_id, exc)
            finally:
                self._unload_diarizer(diarizer)

        segments = self._build_segments(whisper_result, merged_segments)
        # Free whisper_result now that segments are built
        whisper_result = None

        # ── Stage 3: Sentiment (Transformer — local RoBERTa, no unload needed) ──
        sentiment = analyze_sentiment(transcript)
        if speaker_texts:
            sentiment["speaker_sentiments"] = analyze_per_speaker(speaker_texts)
        else:
            sentiment["speaker_sentiments"] = {}

        stats = self._compute_statistics(transcript, segments)

        # ── Stage 4: Acoustic extraction (load → extract → unload) ──
        overall_acoustics = {}
        speaker_acoustics = {}
        extractor = self._load_acoustic_extractor()
        if extractor:
            log.info("Extracting acoustic features...")
            overall_acoustics = extractor.process_audio_file(str(wav_path))
            speaker_acoustics = self._extract_speaker_acoustics(extractor, wav_path, segments)
            del extractor
            _free_ram()

        if tmp_wav and tmp_wav.exists():
            os.unlink(str(tmp_wav))

        # ── Stage 5: Structured Transcript (role detection + formatting) ──
        structured_result = {}
        try:
            from transcript_formatter import process_segments
            structured_result = process_segments(segments)
            log.info("Job %s: structured transcript — roles: %s",
                     job_id, structured_result.get("roles", {}))
        except Exception as exc:
            log.warning("Job %s: structured transcript failed: %s", job_id, exc)

        structured_transcript = structured_result.get("structured_transcript", "")
        speaker_roles = structured_result.get("roles", {})
        speaker_stats = structured_result.get("speaker_stats", {})

        # ── Stage 6a: Acoustic Context Serialization ──
        acoustic_context = ""
        try:
            from acoustic_context import build_acoustic_prompt_context
            acoustic_context = build_acoustic_prompt_context(
                overall_acoustics, speaker_acoustics
            )
        except Exception as exc:
            log.warning("Job %s: acoustic context serialization failed: %s", job_id, exc)

        # ── Stage 6b: LLM Clinical Scoring (Ollama — local) ──
        llm_scoring = {}
        llm_cfg = self.cfg.get("llm_scoring", {})
        if llm_cfg.get("enabled", False) and structured_transcript:
            try:
                from llm_clinical_scorer import score_transcript
                log.info("Job %s: running LLM clinical scoring...", job_id)
                llm_scoring = score_transcript(
                    structured_transcript, acoustic_context, self.cfg
                )
                log.info("Job %s: LLM scoring complete", job_id)
            except Exception as exc:
                log.warning("Job %s: LLM clinical scoring failed: %s", job_id, exc)

        # ── Stage 6c: Clinical Question Detection (Ollama — local) ──
        probe_results = {}
        qd_cfg = self.cfg.get("question_detection", {})
        if qd_cfg.get("enabled", False) and segments and speaker_roles:
            try:
                from question_detector import (
                    detect_probes, tag_segments_with_probes,
                    summarize_probe_coverage
                )
                log.info("Job %s: detecting clinical probes...", job_id)
                probe_tags = detect_probes(segments, speaker_roles, self.cfg)
                if probe_tags:
                    segments = tag_segments_with_probes(segments, probe_tags)
                    probe_results = {
                        "probes": probe_tags,
                        "coverage": summarize_probe_coverage(segments),
                    }
                log.info("Job %s: detected %d clinical probes", job_id, len(probe_tags))
            except Exception as exc:
                log.warning("Job %s: question detection failed: %s", job_id, exc)

        # ── Stage 7: LLM Embedding Extraction (HuggingFace — local) ──
        embeddings_path = ""
        emb_cfg = self.cfg.get("embeddings", {})
        if emb_cfg.get("enabled", False) and segments:
            try:
                from llm_embeddings import (
                    extract_segment_embeddings, export_embeddings, unload_model
                )
                emb_model = emb_cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2")
                pooling = emb_cfg.get("pooling", "mean")
                fmt = emb_cfg.get("export_format", "csv")

                log.info("Job %s: extracting embeddings (%s)...", job_id, emb_model)
                emb_array = extract_segment_embeddings(
                    segments, model_name=emb_model, pooling=pooling
                )
                pipeline_cfg_emb = self.cfg.get("pipeline", {})
                emb_output_dir = Path(
                    resolve_path(
                        pipeline_cfg_emb.get(
                            "analysis_output_folder",
                            self.cfg.get("output_folder", "./Output")
                        )
                    )
                )
                emb_out = emb_output_dir / f"{job_id}_embeddings.{fmt}"
                export_embeddings(emb_array, str(emb_out), format=fmt)
                embeddings_path = str(emb_out)
                unload_model(emb_model)
                _free_ram()
                log.info("Job %s: embeddings exported to %s", job_id, emb_out.name)
            except Exception as exc:
                log.warning("Job %s: embedding extraction failed: %s", job_id, exc)

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
            "status": "completed",
            "model": self.model_name,
            "pipeline_version": "4.0",
            "source_audio": {
                "original_filename": original_filename,
                "stored_path": str(file_path),
            },
            "statistics": stats,
            "overall_sentiment": sentiment.get("overall", {}),
            "overall_acoustics": overall_acoustics,
            "tones": sentiment.get("tones", []),
            "critical_moments": sentiment.get("critical_moments", []),
            "speaker_sentiments": sentiment.get("speaker_sentiments", {}),
            "speaker_acoustics": speaker_acoustics,
            "speaker_roles": speaker_roles,
            "speaker_stats": speaker_stats,
            "structured_transcript": structured_transcript,
            "llm_clinical_scoring": llm_scoring,
            "clinical_probes": probe_results,
            "embeddings_file": embeddings_path,
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
