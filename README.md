# ClinicalWhisper

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/ChengdongPeter/Clinical-Whisper)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20559786.svg)](https://doi.org/10.5281/zenodo.20559786)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-10.64898%2F2026.06.08.728970-b31b1b)](https://doi.org/10.64898/2026.06.08.728970)

**A privacy-first, on-device multimodal framework for anhedonia classification.**

ClinicalWhisper v5.0 processes clinical audio recordings entirely on the local machine. Audio, transcripts, and results never leave the device.

---

## What's New in v5.0

- 🗣️ **MOSS-Transcribe-Diarize 0.9B** — joint transcription and diarization in a single pass, replacing Whisper + Pyannote. Public and ungated; no Hugging Face token needed.
- 🛡️ **OpenMED PII scrubbing** — local HIPAA Safe Harbor de-identification of the transcript (names, dates, addresses, phone numbers, MRNs).
- 🧠 **Native MLX LLM scoring** — Llama-3-8B-Instruct runs on Apple Silicon via `mlx-lm`. No Ollama, no external services.
- 🖥️ **Native desktop app** — a standalone `.app`; results render in the window and save where you choose.

### Previous releases

- **v1.0** — Initial release (Whisper + Pyannote baseline)
- **v2.0** — Introduced RoBERTa sentiment analysis
- **v3.0** — Integrated OpenSMILE (VTA Zhou Index)
- **v4.0** — Added local LLM inference via Ollama

---

## Architecture Overview

| Category | Details |
|---|---|
| **Transcription & Diarization** | MOSS-Transcribe-Diarize 0.9B (MPS on Apple Silicon, CUDA, or CPU) |
| **HIPAA de-identification** | OpenMED `OpenMed-PII-SuperClinical-Small-44M-v1`, mask method |
| **Acoustic features** | OpenSMILE eGeMAPSv02 — pitch, loudness, jitter, shimmer, VTA |
| **LLM clinical scoring** | Llama-3-8B scores hesitancy, affect flatness, engagement, elaboration, psychomotor indicators (0–10) |
| **Batch processing** | Directory → CSV pipeline with per-file error handling |
| **Longitudinal tracking** | Cross-session trend detection via linear regression |

## The Pipeline

```
Audio File (.m4a/.mp3/.wav/.mp4)
    │
    ▼  ffmpeg → 16 kHz mono WAV, plus a loudness-normalised copy
┌──────────────────────────────────────────────────────────────────┐
│                      ClinicalWhisper v5.0                        │
│                                                                  │
│  Stage 1: Transcribe & Diarize   (MOSS, normalised audio)        │
│  Stage 2: PII Scrubbing          (OpenMED, HIPAA Safe Harbor)    │
│  Stage 3: Acoustic Analysis      (OpenSMILE, original gain)      │
│  Stage 4: Role Detection         (Interviewer / Subject)         │
│  Stage 5: LLM Clinical Scoring   (Llama-3-8B via mlx_lm)         │
│                          │                                       │
│                          ▼                                       │
│              JSON Analysis Report  +  Summary CSV                │
└──────────────────────────────────────────────────────────────────┘
```

Transcription runs on a loudness-normalised copy because clinical recordings are
often faint, and MOSS returns an empty transcript on very quiet audio. Acoustic
features are extracted from the **original-gain** audio, since loudness and VTA
are amplitude-dependent and normalisation would invalidate them.

---

## Requirements

| | |
|---|---|
| **Hardware** | Apple Silicon Mac (M1 or newer) for the MLX path. Check with `uname -m` — it must print `arm64`. Intel Macs, Windows, and Linux fall back to HuggingFace transformers on CUDA or CPU, which is substantially slower. |
| **ffmpeg** | Required. `brew install ffmpeg` on macOS, or [ffmpeg.org](https://ffmpeg.org/download.html). |
| **Disk / network** | ~6.4 GB of model weights download to `~/.cache` on first run: MOSS ~1.7 GB, Llama-3-8B 4-bit ~4.5 GB, OpenMED ~0.2 GB. No Hugging Face account or token is required. |

To skip the download entirely on a target machine, see
[Offline handoff](#offline-handoff-usb-drive) below.

---

## Desktop App (Recommended)

Build the `.app` and DMG:

```bash
./build_dmg.sh
```

Mount `ClinicalWhisper.dmg`, drag **ClinicalWhisper** into Applications, then
right-click → **Open** the first time (macOS blocks unsigned apps opened by
double-click).

Drop an audio file into the window and press Process. Stage-by-stage progress
appears in the console panel, results render in the window, and **Save CSV…**
writes the summary wherever you choose.

Files are written to `~/Documents/ClinicalWhisper/`:

```
~/Documents/ClinicalWhisper/
├── Input/        # uploaded audio
├── Output/       # <name>_analysis.json and <name>_summary.csv
└── Processed/    # archived audio after a successful run
```

Override that location with `CLINICALWHISPER_DATA_DIR`. To customise the
pipeline, copy `config.example.yaml` to
`~/Documents/ClinicalWhisper/config.yaml` — the app prefers it over the bundled
defaults.

### Run from source instead

```bash
uv run python main_app.py
```

---

## Command Line Usage

### Setup

```bash
git clone https://github.com/czhou732/ClinicalWhisper.git
cd ClinicalWhisper
uv venv && source .venv/bin/activate
uv pip install -e .
uv pip install "moss-transcribe-diarize @ git+https://github.com/OpenMOSS/MOSS-Transcribe-Diarize.git"
```

### Batch processing

```bash
uv run python batch_processor.py --input ./Input --output ./Output/summary.csv --config config.example.yaml
```

One CSV row per file: word count, duration, the six clinical scores, and the
acoustic features (VTA, pitch mean/CV, loudness mean/CV, jitter, shimmer).

### Configuration

Edit `config.example.yaml`:

```yaml
moss:
  model: "OpenMOSS-Team/MOSS-Transcribe-Diarize"
  device: "auto"        # auto | mps | cuda | cpu

pii_scrubbing:
  enabled: true
  confidence_threshold: 0.7
  strict: true          # abort rather than emit an unscrubbed transcript

llm_scoring:
  enabled: true
  mlx_model: "mlx-community/Meta-Llama-3-8B-Instruct-4bit"   # Apple Silicon
  hf_model: "NousResearch/Meta-Llama-3-8B-Instruct"          # everywhere else
```

Transcription and PII scrubbing failures abort the job rather than emitting a
partial result — an analysis file full of zeros is indistinguishable from a real
one, so the pipeline refuses to produce one.

---

## Offline handoff (USB drive)

To hand the app to a collaborator whose machine has never run it, build a bundle
with the DMG, every model weight, and ffmpeg:

```bash
./scripts/prepare_offline_bundle.sh /Volumes/USB_DRIVE/ClinicalWhisper
```

Run this on a machine that has already processed at least one file, so the model
caches are populated. The recipient runs `install.sh` from the folder, then
mounts the DMG. No downloads, no account, works with Wi-Fi off.

The bundle is roughly **9 GB** (1 GB app + ~7.7 GB of weights), so use a 16 GB or
larger drive formatted APFS or exFAT — FAT32 cannot hold the individual model
files.

---

## Citation

If you use ClinicalWhisper v5.0 in your research, please cite our [bioRxiv preprint](https://doi.org/10.64898/2026.06.08.728970):

```bibtex
@article{zhou2026clinicalwhisper,
  title={Privacy-first, on-device multimodal framework for anhedonia classification},
  author={Zhou, Chengdong and others},
  journal={bioRxiv},
  year={2026},
  doi={10.64898/2026.06.08.728970}
}
```
