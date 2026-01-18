# Clinical Whisper Pipeline 

An automated, privacy-first data processing pipeline designed for clinical psychology research and sensitive interview transcription.

## Overview
This tool leverages OpenAI's **Whisper** model to provide local, air-gapped transcription of audio files. It is designed to replace cloud-based services (like Otter.ai) to ensure **100% HIPAA/GDPR data compliance** by processing all data locally on the machine.

## Features
- **Zero-Touch Automation:** Utilizes `watchdog` to monitor input directories.
- **Privacy-First:** No data leaves the local device.
- **Obsidian Integration:** Automatically formats transcripts into Markdown with metadata tags.
- **Archive Management:** Auto-sorts processed audio files.

## Prerequisites
- Python 3.9+
- FFmpeg (`brew install ffmpeg`)

## Installation (The Professional Way)
1. Clone the repo:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Clinical-Whisper-Pipeline.git](https://github.com/YOUR_USERNAME/Clinical-Whisper-Pipeline.git)
   cd Clinical-Whisper-Pipeline

## Author
**Chengdong Zhou** - Undergraduate Researcher, USC
*Focus: Affective Neuroscience & Clinical Psychology*
