# Clinical Whisper Pipeline 

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/ChengdongPeter/Clinical-Whisper)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**An offline-first, HIPAA-compliant clinical intelligence pipeline for automated speaker diarization and suicide risk assessment.**

An automated, privacy-first data processing pipeline designed for clinical psychology research and sensitive interview transcription.

## Overview
This tool leverages OpenAI's **Whisper** model to provide local, air-gapped transcription of audio files. It is designed to replace cloud-based services (like Otter.ai) to ensure **100% HIPAA/GDPR data compliance** by processing all data locally on the machine.

## Features
- **Zero-Touch Automation:** Utilizes `watchdog` to monitor input directories.
- **Privacy-First:** No data leaves the local device.
- **Obsidian Integration:** Automatically formats transcripts into Markdown with metadata tags.
- **Archive Management:** Auto-sorts processed audio files.
- **🆕 Sentiment Analysis:** Automatic per-transcript sentiment scoring (1–10), emotional tone detection, segment-by-segment breakdown, and critical moment identification — powered by TextBlob.
- **🆕 Transcript Statistics:** Word count, sentence count, and estimated duration for every transcript.
- **🆕 External Configuration:** All settings in `config.yaml` — no need to edit Python code.
- **🆕 Duplicate Protection:** Skips files that have already been transcribed.
- **🆕 Structured Logging:** Timestamped logs with severity levels for easier debugging.

## Prerequisites
- Python 3.9+
- FFmpeg (`brew install ffmpeg`)

## Installation
```bash
git clone https://github.com/czhou732/Clinical-Whisper-Pipeline.git
cd Clinical-Whisper-Pipeline
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` to customize:
```yaml
model: "medium.en"           # Whisper model size
output_folder: "/path/to/obsidian/inbox"
sentiment:
  enabled: true
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

## Output Format

Each transcript note includes:
1. **Header** — title, date, tags, model used
2. **Transcript** — full text from Whisper
3. **📈 Statistics** — word count, sentences, estimated duration
4. **📊 Sentiment Analysis** — overall score, emotional tones, segment breakdown, critical moments

## Author
**Chengdong Zhou** - Undergraduate Researcher, USC
*Focus: Affective Neuroscience & Clinical Psychology*
