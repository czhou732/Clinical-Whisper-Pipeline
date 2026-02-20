#!/usr/bin/env python3
"""
Speaker Diarization Module for ClinicalWhisper
Identifies who spoke when using pyannote.audio, then merges
speaker labels with Whisper transcript segments.

Requires:
  - pyannote.audio (pip install pyannote.audio)
  - A HuggingFace token with access to pyannote/speaker-diarization-3.1
    Set via HF_TOKEN env var or in config.yaml
"""

import os
import logging
from typing import Optional

log = logging.getLogger("ClinicalWhisper")


class SpeakerDiarizer:
    """Speaker diarization using pyannote.audio."""

    def __init__(self, hf_token: Optional[str] = None, min_speakers: int = 2, max_speakers: int = 6):
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.pipeline = None
        self._init_pipeline()

    def _init_pipeline(self):
        """Load the pyannote speaker-diarization pipeline."""
        if not self.hf_token:
            raise ValueError(
                "HuggingFace token required for speaker diarization.\n"
                "  1. Get a token at: https://huggingface.co/settings/tokens\n"
                "  2. Accept terms at: https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "  3. Set HF_TOKEN in your environment or config.yaml"
            )

        from pyannote.audio import Pipeline
        import torch

        log.info("Loading speaker diarization model...")
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=self.hf_token,
        )

        # Use MPS (Apple Silicon GPU) if available, else CPU
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.pipeline.to(torch.device("mps"))
            log.info("✅ Diarization model loaded (Apple Silicon GPU)")
        elif torch.cuda.is_available():
            self.pipeline.to(torch.device("cuda"))
            log.info("✅ Diarization model loaded (CUDA GPU)")
        else:
            log.info("✅ Diarization model loaded (CPU)")

    def diarize(self, audio_path: str) -> dict:
        """
        Run speaker diarization on an audio file.

        Returns:
            {
                "speakers": ["Speaker 1", "Speaker 2", ...],
                "segments": [{"start": float, "end": float, "speaker": str}, ...]
            }
        """
        log.info("   🔊 Identifying speakers in %s", os.path.basename(audio_path))

        diarization = self.pipeline(
            audio_path,
            min_speakers=self.min_speakers if self.min_speakers > 1 else None,
            max_speakers=self.max_speakers,
        )

        segments = []
        speakers = set()
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # Rename "SPEAKER_00" → "Speaker 1"
            friendly = f"Speaker {int(speaker.split('_')[-1]) + 1}" if "SPEAKER_" in speaker else speaker
            segments.append({"start": turn.start, "end": turn.end, "speaker": friendly})
            speakers.add(friendly)

        log.info("   🗣️  Found %d speakers across %d segments", len(speakers), len(segments))
        return {"speakers": sorted(speakers), "segments": segments}

    def merge_with_transcript(self, whisper_result: dict, diarization: dict) -> list[dict]:
        """
        Attach speaker labels to Whisper transcript segments by matching
        each transcript segment's midpoint to the closest diarization segment.

        Returns a list of dicts: [{"speaker", "start", "end", "text"}, ...]
        """
        diar_segs = diarization["segments"]
        merged = []

        for seg in whisper_result.get("segments", []):
            mid = (seg["start"] + seg["end"]) / 2
            speaker = self._speaker_at(mid, diar_segs) or "Unknown"
            merged.append({
                "speaker": speaker,
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
            })

        return merged

    @staticmethod
    def _speaker_at(time: float, diar_segments: list[dict]) -> Optional[str]:
        """Find which speaker is active at a given timestamp."""
        for seg in diar_segments:
            if seg["start"] <= time <= seg["end"]:
                return seg["speaker"]
        return None

    @staticmethod
    def group_by_speaker(merged_segments: list[dict]) -> dict[str, str]:
        """
        Group all text by speaker.

        Returns: {"Speaker 1": "all their text ...", "Speaker 2": "...", ...}
        """
        groups: dict[str, list[str]] = {}
        for seg in merged_segments:
            groups.setdefault(seg["speaker"], []).append(seg["text"])
        return {spk: " ".join(texts) for spk, texts in groups.items()}

    @staticmethod
    def format_transcript_with_speakers(merged_segments: list[dict]) -> str:
        """Format the transcript with speaker labels and timestamps."""
        lines = []
        current_speaker = None
        current_texts: list[str] = []

        def _flush():
            if current_speaker and current_texts:
                lines.append(f"\n**{current_speaker}:** {' '.join(current_texts)}")

        for seg in merged_segments:
            if seg["speaker"] != current_speaker:
                _flush()
                current_speaker = seg["speaker"]
                current_texts = []
            current_texts.append(seg["text"])

        _flush()
        return "\n".join(lines)

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        h, rem = divmod(int(seconds), 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
