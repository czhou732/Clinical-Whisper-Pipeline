# Clinical Whisper Pipeline

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/ChengdongPeter/Clinical-Whisper)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**An offline-first, HIPAA-compliant clinical intelligence pipeline for automated transcription, speaker diarization, sentiment analysis, and acoustic feature extraction.**

A privacy-first data processing pipeline designed for clinical and research transcription — no data ever leaves the local machine.

## Overview

This tool uses **MLX Whisper** (Apple Silicon native, 4-10x faster) with automatic fallback to **OpenAI Whisper** on other platforms. It provides local, air-gapped transcription to ensure **100% HIPAA/GDPR compliance** by processing all data on-device.

## Features

- **Zero-Touch Automation:** Uses `watchdog` to monitor input directories — drop a file, get a transcript.
- **Privacy-First:** All processing is 100% local. No API calls, no cloud, no network.
- **MLX Whisper (Apple Silicon):** Native Apple Silicon acceleration via MLX for 4-10x faster transcription. Automatic fallback to OpenAI Whisper on Linux/Windows.
- **Speaker Diarization:** Per-speaker identification and labeling using `pyannote.audio`.
- **Sentiment Analysis:** Per-transcript and per-speaker sentiment scoring (1–10), emotional tone detection, segment-by-segment breakdown, and critical moment identification.
- **Acoustic Feature Extraction:** OpenSMILE eGeMAPSv02 features with streaming reads for low RAM usage.
- **Obsidian Integration:** Automatically formats transcripts into Markdown with metadata tags.
- **Meeting Intelligence:** Ollama-powered local LLM summaries of transcripts (optional).
- **Plaud Integration:** Auto-ingests recordings from Plaud Note devices via USB or export folder.
- **External Configuration:** All settings in `config.yaml` — no need to edit Python code.
- **Transcript Statistics:** Word count, sentence count, and estimated duration.
- **Duplicate Protection:** Skips files that have already been transcribed.
- **Archive Management:** Auto-sorts processed audio files.

## Prerequisites

- Python 3.9+
- FFmpeg (`brew install ffmpeg`)
- Apple Silicon Mac recommended (for MLX acceleration)

## Installation

```bash
git clone https://github.com/czhou732/Clinical-Whisper-Pipeline.git
cd Clinical-Whisper-Pipeline
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# For Apple Silicon (recommended — 4-10x faster):
pip install mlx-whisper
```

## Configuration

Edit `config.yaml` to customize:

```yaml
model: "small.en"                        # Whisper model size
mlx_model: "mlx-community/whisper-small.en"  # MLX model (Apple Silicon)
output_folder: "/path/to/obsidian/inbox"

sentiment:
  enabled: true
diarization:
  enabled: false                         # Requires pyannote.audio + HF token
acoustic_features:
  enabled: false                         # Requires opensmile
statistics:
  enabled: true
skip_already_processed: true
```

## Usage

```bash
source venv/bin/activate
python main.py
```

Drop audio files (`.m4a`, `.mp3`, `.wav`, `.mp4`) into the `Input/` folder.
Transcripts with sentiment analysis will appear in your configured output folder.

### Running as a daemon (macOS)

```bash
bash scripts/start_pipeline.sh       # Start pipeline + Plaud connector
bash scripts/start_pipeline.sh --stop  # Stop all
```

## Output Format

Each transcript note includes:

1. **Header** — title, date, tags, model used, speakers (if diarized)
2. **Transcript** — full text with optional per-speaker labels
3. **📈 Statistics** — word count, sentences, estimated duration
4. **📊 Sentiment Analysis** — overall score, emotional tones, segment breakdown, critical moments

## Architecture

```
Plaud/Audio File → Input/ → Whisper (MLX/OpenAI) → Diarization → Sentiment → Acoustics → Markdown → Obsidian
```

- **Transcription backend:** MLX Whisper on Apple Silicon, OpenAI Whisper elsewhere
- **Models load on-demand** per file and unload after each stage (low idle RAM)
- **Single-worker design** optimized for daily transcription workflows

## Author

**Chengdong Zhou** — Undergraduate Researcher, USC
*Focus: Computational Neuroscience & Clinical AI*
