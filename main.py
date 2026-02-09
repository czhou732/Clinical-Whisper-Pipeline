#!/usr/bin/env python3
"""
ClinicalWhisper — Privacy-first local transcription pipeline
with automated sentiment analysis.

Watches the Input folder for audio files, transcribes locally via
OpenAI Whisper, runs sentiment analysis, and saves rich Markdown
notes to Obsidian.
"""

import sys
import time
import os
import shutil
import logging
import yaml

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import whisper
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from sentiment_analyzer import analyze_sentiment, format_sentiment_markdown

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
        "model": "medium.en",
        "input_folder": "./Input",
        "processed_folder": "./Processed",
        "output_folder": "./Output",
        "audio_extensions": [".m4a", ".mp3", ".wav", ".mp4"],
        "sentiment": {"enabled": True, "sentences_per_segment": 5},
        "statistics": {"enabled": True},
        "skip_already_processed": True,
    }

    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            user_cfg = yaml.safe_load(f) or {}
        # Merge — user values override defaults
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
AUDIO_EXTENSIONS = tuple(CFG["audio_extensions"])

# ---------------------------------------------------------------------------
# Transcript statistics
# ---------------------------------------------------------------------------

def compute_statistics(text: str) -> str:
    """Return a Markdown statistics block for the transcript."""
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    sentence_count = sum(1 for c in text if c in ".!?")
    # Rough speaking-rate estimate: ~150 words per minute
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
    def __init__(self, model):
        self.model = model
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

        # Duplicate protection
        if self._already_processed(name_without_ext):
            log.info("⏭️  Skipping (already processed): %s", base_name)
            return

        log.info("🎧 New file detected: %s", base_name)

        try:
            # 1. Transcribe
            log.info("   Transcribing with Whisper (%s)...", MODEL_TYPE)
            t0 = time.time()
            result = self.model.transcribe(filename)
            elapsed = time.time() - t0
            log.info("   ✅ Transcription complete (%.1fs)", elapsed)

            transcript = result["text"]

            # 2. Build note content
            note_content = f"# 🎙️ Transcript: {name_without_ext}\n"
            note_content += f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n"
            note_content += f"**Tags:** #transcribed #clinical-whisper #sentiment\n"
            note_content += f"**Model:** `{MODEL_TYPE}`\n"
            note_content += "---\n\n"
            note_content += transcript

            # 3. Transcript statistics
            if CFG.get("statistics", {}).get("enabled", True):
                note_content += compute_statistics(transcript)

            # 4. Sentiment analysis
            if CFG.get("sentiment", {}).get("enabled", True):
                log.info("   🧠 Running sentiment analysis...")
                sentiment = analyze_sentiment(transcript)
                note_content += format_sentiment_markdown(sentiment)
                log.info(
                    "   📊 Sentiment: %s (%d/10)",
                    sentiment["overall"]["label"],
                    sentiment["overall"]["score"],
                )

            # 5. Save to output
            os.makedirs(OUTPUT_FOLDER, exist_ok=True)
            destination_path = os.path.join(OUTPUT_FOLDER, f"{name_without_ext}.md")
            with open(destination_path, "w") as f:
                f.write(note_content)
            log.info("   💾 Note saved: %s", destination_path)

            # 6. Archive audio
            if os.path.exists(filename):
                os.makedirs(PROCESSED_FOLDER, exist_ok=True)
                shutil.move(filename, os.path.join(PROCESSED_FOLDER, base_name))
                log.info("   📁 Audio archived to %s", PROCESSED_FOLDER)

            self._processed_set.add(name_without_ext)

        except Exception:
            log.exception("❌ Error processing %s", base_name)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    log.info("🚀 Loading Whisper model '%s'  (Python: %s)", MODEL_TYPE, sys.executable)
    model = whisper.load_model(MODEL_TYPE)
    log.info("👀 Watching '%s' for audio files...", INPUT_FOLDER)
    log.info("📂 Notes  → %s", OUTPUT_FOLDER)
    log.info("📁 Archive → %s", PROCESSED_FOLDER)

    event_handler = WhisperHandler(model)
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