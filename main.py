#!/usr/bin/env python3
"""
ClinicalWhisper V2 — The Ultimate Clinical Engine
Privacy-first local processing pipeline:
1. MOSS Transcription & Diarization
2. OpenMED PII Scrubbing
3. SSL & OpenSMILE Acoustic Analysis
4. MLX Clinical LLM Scoring
"""

import gc
import sys
import time
import os
import shutil
import logging
import yaml
import json

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Lazy imports for the ML pipeline
from moss_diarizer import MOSSDiarizer
from pii_scrubber import PIIScrubber
from acoustic_features import AcousticExtractor
from llm_clinical_scorer import score_transcript, _format_scoring_output

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
    defaults = {
        "model": "small.en",
        "mlx_model": "mlx-community/whisper-small.en-mlx",
        "input_folder": "./Input",
        "processed_folder": "./Processed",
        "output_folder": "./Output",
        "audio_extensions": [".m4a", ".mp3", ".wav", ".mp4"],
        "skip_already_processed": True,
        "diarization": {
            "enabled": True, # Enabled by default for V2 (MOSS)
            "model": "OpenMOSS-Team/MOSS-Transcribe-Diarize"
        },
        "pii_scrubbing": {
            "enabled": True
        },
        "acoustic_features": {
            "enabled": True
        },
        "llm_scoring": {
            "enabled": True,
            "mlx_model": "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
        }
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
AUDIO_EXTENSIONS = tuple(CFG["audio_extensions"])

# ---------------------------------------------------------------------------
# Global Pipeline Singletons (Loaded on startup or lazily)
# ---------------------------------------------------------------------------
PIPELINE = {
    "moss": None,
    "scrubber": None,
    "acoustics": None
}

def init_pipeline():
    if CFG["diarization"]["enabled"]:
        PIPELINE["moss"] = MOSSDiarizer(model_name=CFG["diarization"]["model"])
    if CFG["pii_scrubbing"]["enabled"]:
        PIPELINE["scrubber"] = PIIScrubber()
    if CFG["acoustic_features"]["enabled"]:
        PIPELINE["acoustics"] = AcousticExtractor()

# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------
def format_transcript_markdown(segments: list[dict]) -> str:
    lines = []
    for seg in segments:
        lines.append(f"**[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['speaker']}:** {seg['text']}")
    return "\n\n".join(lines)

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

    def process_file(self, event):
        if event.is_directory:
            return
        filename = event.src_path
        if not filename.endswith(AUDIO_EXTENSIONS):
            return

        time.sleep(2)
        if not os.path.exists(filename):
            return

        base_name = os.path.basename(filename)
        name_without_ext = os.path.splitext(base_name)[0]

        if CFG.get("skip_already_processed", True):
            if os.path.exists(os.path.join(OUTPUT_FOLDER, f"{name_without_ext}.md")) or name_without_ext in self._processed_set:
                return

        log.info("==================================================")
        log.info("🎧 New file detected: %s", base_name)

        try:
            # ── Stage 1: Transcribe & Diarize (MOSS) ──
            segments = []
            if PIPELINE["moss"] and PIPELINE["moss"].is_available:
                log.info("   🎙️ Running MOSS Transcription & Diarization...")
                segments = PIPELINE["moss"].process_file(filename)
            else:
                log.warning("   ⚠️ MOSS disabled or unavailable. Falling back to MLX Whisper (no diarization).")
                import mlx_whisper
                res = mlx_whisper.transcribe(filename, path_or_hf_repo=CFG.get("mlx_model", "mlx-community/whisper-small.en-mlx"))
                segments = [{"start": 0.0, "end": 0.0, "speaker": "Speaker 1", "text": res["text"]}]

            # ── Stage 2: PII Scrubbing (OpenMED) ──
            if PIPELINE["scrubber"] and PIPELINE["scrubber"].is_available:
                log.info("   🛡️ Scrubbing PII from transcript (OpenMED)...")
                segments = PIPELINE["scrubber"].scrub_segments(segments)
                
            formatted_text = format_transcript_markdown(segments)

            # ── Stage 3: Acoustic Analysis (OpenSMILE eGeMAPSv02) ──
            acoustic_data = {}
            acoustic_json = "No acoustic data extracted."
            if PIPELINE["acoustics"] and PIPELINE["acoustics"].is_available():
                log.info("   🎶 Extracting Acoustic Features (OpenSMILE)...")
                acoustic_data = PIPELINE["acoustics"].process_audio_file(filename)
                acoustic_json = json.dumps(acoustic_data, indent=2)

            # ── Stage 4: Clinical LLM Scoring ──
            scoring_result = {}
            if CFG["llm_scoring"]["enabled"]:
                log.info("   🧠 Running Clinical LLM Scoring via MLX...")
                scoring_result = score_transcript(
                    structured_transcript=formatted_text,
                    acoustic_context=acoustic_json,
                    config=CFG
                )

            # ── Stage 5: Build & Save Markdown Note ──
            note_content = f"# 🎙️ Clinical Assessment: {name_without_ext}\n"
            note_content += f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n"
            note_content += f"**Tags:** #clinical-whisper #v2 #assessment\n\n"
            
            note_content += "## 📝 Transcript\n"
            note_content += formatted_text + "\n\n"
            
            if acoustic_data:
                note_content += "## 🎶 Acoustic Features\n"
                note_content += f"- **Pitch (F0) CV:** {acoustic_data.get('pitch_cv', 'N/A')}\n"
                note_content += f"- **Loudness CV:** {acoustic_data.get('loudness_cv', 'N/A')}\n"
                note_content += f"- **Jitter:** {acoustic_data.get('jitter', 'N/A')}\n"
                note_content += f"- **Shimmer:** {acoustic_data.get('shimmer', 'N/A')}\n"
                note_content += f"- **VTA Score:** {acoustic_data.get('vta', 'N/A')}\n"

            if scoring_result:
                note_content += _format_scoring_output(scoring_result) + "\n"

            os.makedirs(OUTPUT_FOLDER, exist_ok=True)
            destination_path = os.path.join(OUTPUT_FOLDER, f"{name_without_ext}.md")
            with open(destination_path, "w") as f:
                f.write(note_content)
            log.info("   💾 Note saved: %s", destination_path)

            # ── Stage 6: Archive audio ──
            if os.path.exists(filename):
                os.makedirs(PROCESSED_FOLDER, exist_ok=True)
                shutil.move(filename, os.path.join(PROCESSED_FOLDER, base_name))
                log.info("   📁 Audio archived to %s", PROCESSED_FOLDER)

            self._processed_set.add(name_without_ext)
            log.info("==================================================")

        except Exception:
            log.exception("❌ Error processing %s", base_name)
        finally:
            gc.collect()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    log.info("🚀 ClinicalWhisper V2 Engine Initializing...")
    init_pipeline()
    log.info("👀 Watching '%s' for audio files...", INPUT_FOLDER)

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