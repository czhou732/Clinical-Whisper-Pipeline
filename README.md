# ClinicalWhisper V2

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/ChengdongPeter/Clinical-Whisper)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20559786.svg)](https://doi.org/10.5281/zenodo.20559786)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-10.64898%2F2026.06.08.728970-b31b1b)](https://doi.org/10.64898/2026.06.08.728970)

**A privacy-first, on-device multimodal framework for anhedonia classification.**

ClinicalWhisper V2 processes clinical audio recordings entirely on the local machine. No data ever leaves the device, making it 100% compliant with HIPAA/GDPR-regulated environments.

---

## What's New in V2 (The Computational Psychiatry Upgrade)

- 🗣️ **MOSS-Transcribe-Diarize** — End-to-end multimodal diarization replacing Whisper + Pyannote, highly robust in multi-speaker scenarios.
- 🛡️ **OpenMED PII Scrubbing** — Local-first healthcare AI for clinical entity recognition and HIPAA PII de-identification.
- 🧠 **Native MLX LLM Scoring** — Llama-3-8B-Instruct runs natively on Apple Silicon via `mlx-lm`. **No external software (like Ollama) required!** Models are cached automatically on first run.
- 🎶 **Deep SSL Acoustic Embeddings** — Added WavLM/HuBERT embeddings alongside OpenSMILE to map temporal acoustic trajectories.
- 🤖 **RL Mechanistic Modeling** — Engineered to output topic-segmented features for reinforcement learning (RL) parameter estimation ($\delta, \alpha, \rho, \gamma$) to model reward processing in anhedonia.

### Previous releases

- **v1.0** — Initial release (Whisper + Pyannote baseline)
- **v2.0** — Introduced RoBERTa sentiment analysis
- **v3.0** — Integrated OpenSMILE (VTA Zhou Index)
- **v4.0** — Added local LLM inference via Ollama

---

## Architecture Overview

| Category | Details |
|---|---|
| **Transcription & Diarization** | MOSS-Transcribe-Diarize-0.9B on Apple Silicon MPS |
| **HIPAA Compliance** | OpenMED Entity Recognition for automated PII masking |
| **Acoustic Features** | OpenSMILE (pitch, loudness, VTA) + WavLM Deep SSL embeddings |
| **LLM Clinical Scoring** | Llama-3-8B (via `mlx-lm`) scores engagement, elaboration, psychomotor indicators (0–10) natively |
| **Batch Processing** | Directory → CSV pipeline with per-file error handling |
| **Longitudinal Tracking** | Cross-session trend detection via linear regression |

## The Pipeline

```
Audio File (.m4a/.mp3/.wav/.mp4)
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│                      ClinicalWhisper V2                          │
│                                                                  │
│  Stage 1: Transcribe & Diarize (MOSS-Transcribe-Diarize-0.9B)    │
│  Stage 2: PII Scrubbing (OpenMED HIPAA de-identification)        │
│  Stage 3: Acoustic Analysis (WavLM SSL + OpenSMILE eGeMAPS)      │
│  Stage 4: LLM Clinical Scoring (Llama-3-8B via mlx_lm)           │
│                                                                  │
│  └──────┬──────┘  └──────┬───────┘  └──────┬───────┘            │
│         └────────────────┼─────────────────┘                    │
│                          ▼                                      │
│                 JSON Analysis Report                            │
└──────────────────────────────────────────────────────────────────┘
```

---

## Getting Started

### Prerequisites

- macOS (Apple Silicon M1/M2/M3 highly recommended for MLX acceleration)
- `uv` (Python package manager)

### Setup

```bash
git clone https://github.com/czhou732/ClinicalWhisper.git
cd ClinicalWhisper
uv venv
source .venv/bin/activate

# Install all dependencies (MOSS, OpenMED, MLX, Transformers, Torchaudio)
uv pip install -e .
```

> **Note on LLMs:** You **do not** need to install Ollama or any external software. The V2 architecture uses `mlx_lm` to run Llama-3-8B natively on your Apple Silicon chip. Models are automatically downloaded from HuggingFace and cached on your first run.

---

## macOS Desktop App (Recommended)

We provide a beautiful, native desktop interface powered by a secure local backend. 

1. Ensure dependencies are installed (see above).
2. Build the app by running: `bash build_dmg.sh`
3. Open the resulting `ClinicalWhisper.dmg` and drag the app into your Applications folder.

No external cloud servers are used. Everything stays on your machine.

---

## Command Line Usage

### Batch Processing

To process a directory of clinical interviews and generate an aggregated CSV of all multimodal biomarkers:

```bash
python batch_processor.py --input ./Input --output ./Output/summary.csv
```

### Advanced Configuration

Edit `config.example.yaml` to toggle pipeline stages:

```yaml
pipeline:
  run_diarization: true
  run_pii_scrubbing: true
  run_acoustic_features: true
  run_llm_scoring: true

llm:
  model_id: "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
  backend: "mlx"

acoustic:
  extract_wavlm: true
```

---

## Citation

If you use ClinicalWhisper V2 in your research, please cite our [bioRxiv preprint](https://doi.org/10.64898/2026.06.08.728970):

```bibtex
@article{zhou2026clinicalwhisper,
  title={Privacy-first, on-device multimodal framework for anhedonia classification},
  author={Zhou, Chengdong and others},
  journal={bioRxiv},
  year={2026},
  doi={10.64898/2026.06.08.728970}
}
```
