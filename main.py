#!/usr/bin/env python3
"""
ClinicalWhisper — Privacy-first local transcription pipeline
with automated sentiment analysis.

Watches the Input folder for audio files, transcribes locally via
MLX Whisper (Apple Silicon native), runs sentiment analysis, and
saves rich Markdown notes to Obsidian.

All processing is 100% local — no data ever leaves this machine.
"""

import gc
import sys
import time
import os
import shutil
import logging
import yaml

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from sentiment_analyzer import analyze_sentiment, analyze_per_speaker, format_sentiment_markdown

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ClinicalWhisper")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


def load_config() -> dict:
    """Load configuration from config.yaml with sensible defaults."""
    defaults = {
        "model": "small.en",
        "mlx_model": "mlx-community/whisper-small.en",
        "input_folder": "./Input",
        "processed_folder": "./Processed",
        "output_folder": "./Output",
        "audio_extensions": [".m4a", ".mp3", ".wav", ".mp4"],
        "sentiment": {"enabled": True, "sentences_per_segment": 5},
        "statistics": {"enabled": True},
        "skip_already_processed": True,
        "diarization": {
            "enabled": False,
            "hf_token": "",
            "min_speakers": 2,
            "max_speakers": 6,
        },
    }

    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            user_cfg = yaml.safe_load(f) or {}
        for key, value in user_cfg.items():
            if isinstance(value, dict) and key in defaults and isinstance(defaults[key], dict):
                defaults[key].update(value)
            else:
                defaults[key] = value
        log.info("Loaded config from %s", CONFIG_PATH)
    else:
        log.warning("No config.yaml found — using defaults")

    return defaults


CFG = load_config()
INPUT_FOLDER = CFG["input_folder"]
PROCESSED_FOLDER = CFG["processed_folder"]
OUTPUT_FOLDER = CFG["output_folder"]
MODEL_TYPE = CFG["model"]
MLX_MODEL = CFG.get("mlx_model", f"mlx-community/whisper-{MODEL_TYPE}")
AUDIO_EXTENSIONS = tuple(CFG["audio_extensions"])


# ---------------------------------------------------------------------------
# MLX Whisper transcription (stateless — no model object to manage)
# ---------------------------------------------------------------------------

def _transcribe(audio_path: str) -> dict:
    """Transcribe audio locally. Uses MLX Whisper on Apple Silicon, falls back to openai-whisper."""
    try:
        import mlx_whisper
        return mlx_whisper.transcribe(audio_path, path_or_hf_repo=MLX_MODEL)
    except ImportError:
        import whisper
        model = whisper.load_model(MODEL_TYPE)
        result = model.transcribe(audio_path, fp16=False)
        del model
        gc.collect()
        return result


# ---------------------------------------------------------------------------
# Speaker diarization (on-demand)
# ---------------------------------------------------------------------------

def _load_diarizer():
    """Load speaker diarizer on demand. Returns diarizer or None."""
    diar_cfg = CFG.get("diarization", {})
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
    except Exception as e:
        log.warning("⚠️  Diarization unavailable: %s — falling back to per-segment", e)
        return None


def _unload_diarizer(diarizer):
    """Explicitly unload diarizer and free RAM."""
    if diarizer is None:
        return
    del diarizer
    gc.collect()
    log.info("   📤 Diarizer unloaded")


# ---------------------------------------------------------------------------
# Transcript statistics
# ---------------------------------------------------------------------------

def compute_statistics(text: str) -> str:
    """Return a Markdown statistics block for the transcript."""
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    sentence_count = sum(1 for c in text if c in ".!?")
    est_minutes = word_count / 150

    lines = []
    lines.append("\n---\n")
    lines.append("## 📈 Transcript Statistics\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Words | {word_count:,} |")
    lines.append(f"| Characters | {char_count:,} |")
    lines.append(f"| Sentences (approx.) | {sentence_count} |")
    lines.append(f"| Est. Duration | ~{est_minutes:.1f} min |")
    lines.append("")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# File handler
# ---------------------------------------------------------------------------

class WhisperHandler(FileSystemEventHandler):
    def __init__(self):
        self._processed_set: set[str] = set()

    def on_created(self, event):
        self.process_file(event)

    def on_moved(self, event):
        if not event.is_directory:
            class MockEvent:
                is_directory = False
                src_path = event.dest_path
            self.process_file(MockEvent())

    def _already_processed(self, name_without_ext: str) -> bool:
        """Check if a note already exists in the output folder."""
        if not CFG.get("skip_already_processed", True):
            return False
        dest = os.path.join(OUTPUT_FOLDER, f"{name_without_ext}.md")
        return os.path.exists(dest) or name_without_ext in self._processed_set

    def process_file(self, event):
        if event.is_directory:
            return
        filename = event.src_path
        if not filename.endswith(AUDIO_EXTENSIONS):
            return

        time.sleep(2)  # let the file finish writing
        if not os.path.exists(filename):
            return

        base_name = os.path.basename(filename)
        name_without_ext = os.path.splitext(base_name)[0]

        if self._already_processed(name_without_ext):
            log.info("⏭️  Skipping (already processed): %s", base_name)
            return

        log.info("🎧 New file detected: %s", base_name)

        try:
            # ── Stage 1: Transcribe (MLX Whisper — native Apple Silicon) ──
            log.info("   Transcribing with MLX Whisper (%s)...", MLX_MODEL)
            t0 = time.time()
            result = _transcribe(filename)
            elapsed = time.time() - t0
            log.info("   ✅ Transcription complete (%.1fs)", elapsed)
            transcript = result["text"]

            # ── Stage 2: Diarization (load → diarize → unload) ──
            merged_segments = None
            speaker_texts = None
            diarizer = _load_diarizer()

            if diarizer:
                try:
                    diar_result = diarizer.diarize(filename)
                    merged_segments = diarizer.merge_with_transcript(result, diar_result)
                    speaker_texts = diarizer.group_by_speaker(merged_segments)
                    log.info("   🗣️  Speakers: %s", ", ".join(diar_result["speakers"]))
                except Exception as e:
                    log.warning("   ⚠️  Diarization failed: %s — continuing without", e)
                finally:
                    _unload_diarizer(diarizer)

            result = None  # free Whisper result

            # ── Stage 3: Build note content ──
            note_content = f"# 🎙️ Transcript: {name_without_ext}\n"
            note_content += f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n"
            note_content += f"**Tags:** #transcribed #clinical-whisper #sentiment\n"
            note_content += f"**Model:** `{MLX_MODEL}`\n"

            if speaker_texts:
                speakers_list = ", ".join(speaker_texts.keys())
                note_content += f"**Speakers:** {speakers_list}\n"

            note_content += "---\n\n"

            if merged_segments:
                from speaker_diarizer import SpeakerDiarizer
                note_content += SpeakerDiarizer.format_transcript_with_speakers(merged_segments)
            else:
                note_content += transcript

            # ── Stage 4: Statistics ──
            if CFG.get("statistics", {}).get("enabled", True):
                note_content += compute_statistics(transcript)

            # ── Stage 5: Sentiment analysis ──
            if CFG.get("sentiment", {}).get("enabled", True):
                log.info("   🧠 Running sentiment analysis...")
                sentiment = analyze_sentiment(transcript)

                if speaker_texts:
                    log.info("   👥 Analyzing per-speaker sentiment...")
                    sentiment["speaker_sentiments"] = analyze_per_speaker(speaker_texts)

                note_content += format_sentiment_markdown(sentiment)
                log.info(
                    "   📊 Sentiment: %s (%d/10)",
                    sentiment["overall"]["label"],
                    sentiment["overall"]["score"],
                )

            # ── Stage 6: Save ──
            os.makedirs(OUTPUT_FOLDER, exist_ok=True)
            destination_path = os.path.join(OUTPUT_FOLDER, f"{name_without_ext}.md")
            with open(destination_path, "w") as f:
                f.write(note_content)
            log.info("   💾 Note saved: %s", destination_path)

            # ── Stage 7: Archive audio ──
            if os.path.exists(filename):
                os.makedirs(PROCESSED_FOLDER, exist_ok=True)
                shutil.move(filename, os.path.join(PROCESSED_FOLDER, base_name))
                log.info("   📁 Audio archived to %s", PROCESSED_FOLDER)

            self._processed_set.add(name_without_ext)

        except Exception:
            log.exception("❌ Error processing %s", base_name)
        finally:
            gc.collect()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    log.info("🚀 ClinicalWhisper started (MLX Whisper: %s, Python: %s)",
             MLX_MODEL, sys.executable)
    log.info("   Backend: Apple MLX (native Apple Silicon, 4-10x faster)")
    log.info("👀 Watching '%s' for audio files...", INPUT_FOLDER)
    log.info("📂 Notes  → %s", OUTPUT_FOLDER)
    log.info("📁 Archive → %s", PROCESSED_FOLDER)

    event_handler = WhisperHandler()
    observer = Observer()
    observer.schedule(event_handler, INPUT_FOLDER, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("🛑 Shutting down...")
        observer.stop()
    observer.join()
    log.info("Done.")