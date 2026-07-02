---
title: Clinical Whisper
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
---
# ClinicalWhisper

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/ChengdongPeter/Clinical-Whisper)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20559786.svg)](https://doi.org/10.5281/zenodo.20559786)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-10.64898%2F2026.06.08.728970-b31b1b)](https://doi.org/10.64898/2026.06.08.728970)

**A privacy-first clinical speech analysis and screening pipeline — transcription, diarization, sentiment, acoustic biomarkers, LLM-powered clinical scoring, and question detection. 100% on-device.**

ClinicalWhisper processes clinical audio recordings entirely on the local machine. No data ever leaves the device, making it suitable for HIPAA/GDPR-regulated environments.

---

## What's New in v4.0

- 🧠 **LLM clinical scoring** — Ollama-powered structured assessment: hesitancy, affect flatness, engagement, elaboration, psychomotor indicators (0–10 scale)
- 🏷️ **Clinical question detection** — Automatically categorizes interviewer probes (positive affect, negative affect, PTSD screen, suicidal ideation, etc.) with coverage analysis
- 🎤 **Structured transcripts** — Auto-detects Interviewer vs Subject roles and formats `[MM:SS] Interviewer: ...` transcripts
- 🔊 **Acoustic context injection** — Serializes prosodic features into natural language for LLM prompt augmentation
- 📐 **LLM embedding extraction** — Sentence-transformer hidden-state vectors per segment, exportable as CSV/NumPy for downstream ML
- 🏗️ **7-stage pipeline** — Expanded from 4 stages with config-gated LLM features

### Previous releases

<details>
<summary>v3.0 — Transformer Sentiment</summary>

- 🧠 **Transformer-based sentiment** — Replaced TextBlob with `cardiffnlp/twitter-roberta-base-sentiment-latest`
- 🧪 **Full test suite** — 20+ pytest tests covering sentiment, acoustics, pipeline integration, and the Zhou Index
- 📊 **Batch processing** — Process an entire directory of audio files into a single summary CSV
- 📈 **Longitudinal tracking** — Track patient metrics across sessions with automatic trend detection
- 📦 **Modern packaging** — `pyproject.toml` with `pip install -e .` support
- 🔄 **GitHub Actions CI** — Automated testing on every push and pull request
- 🔧 **NLTK sentence tokenization** — Handles abbreviations and edge cases that regex-based splitting misses
- 📋 **CSV/SPSS export** — Flatten analysis JSONs into R/SPSS-compatible tabular data

</details>

---

## Features

| Category | Details |
|---|---|
| **Transcription** | MLX Whisper on Apple Silicon (4–10× faster), OpenAI Whisper fallback |
| **Speaker Diarization** | pyannote.audio — per-speaker identification and labeling |
| **Structured Transcripts** | Auto-detects Interviewer/Subject roles, merges consecutive turns |
| **Sentiment Analysis** | Transformer-based scoring (1–10), tone detection, critical moment identification |
| **Acoustic Features** | OpenSMILE eGeMAPSv02 — pitch, loudness, jitter, shimmer, Zhou Index (VTA) |
| **LLM Clinical Scoring** | Hesitancy, affect flatness, engagement, elaboration, psychomotor (0–10) via Ollama |
| **Question Detection** | 10-category clinical probe classification with coverage analysis |
| **LLM Embeddings** | Sentence-transformer feature vectors for downstream ML classifiers |
| **Batch Processing** | Directory → CSV pipeline with per-file error handling |
| **Longitudinal Tracking** | Cross-session trend detection via linear regression |
| **Meeting Intelligence** | Ollama-powered local LLM summaries (optional) |
| **Plaud Integration** | Auto-ingest from Plaud Note devices via USB or export folder |
| **Privacy** | 100% local processing — zero network calls, zero cloud dependencies |
| **Output** | Markdown (Obsidian-compatible), JSON analysis, CSV export, embeddings |

---

## Architecture

```
Audio File (.m4a/.mp3/.wav/.mp4)
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│                      ClinicalWhisper v4.0                        │
│                                                                  │
│  Stage 1: Preprocess ──▶ MLX Whisper Transcription                │
│  Stage 2: Speaker Diarization (pyannote.audio)                    │
│  Stage 3: Sentiment Analysis (RoBERTa transformer)                │
│  Stage 4: Acoustic Features (OpenSMILE eGeMAPSv02)                │
│  Stage 5: Structured Transcript (Interviewer/Subject detection)   │
│                                                                  │
│         ┌─────────────────┼─────────────────┐                    │
│         ▼                 ▼                 ▼                    │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │ LLM Clinical│  │ Question     │  │ Embedding    │            │
│  │ Scoring     │  │ Detection    │  │ Extraction   │            │
│  │ (Ollama)    │  │ (Ollama)     │  │ (HuggingFace)│            │
│  └──────┬──────┘  └──────┬───────┘  └──────┬───────┘            │
│         └────────────────┼─────────────────┘                    │
│                          ▼                                      │
│               JSON + Markdown + Embeddings                       │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────┐
│ Research Tools        │
│ • Batch Processing    │
│ • CSV/SPSS Export     │
│ • Longitudinal Track  │
└───────────────────────┘
```

---

## Installation

### Prerequisites

- Python 3.9+
- FFmpeg (`brew install ffmpeg`)
- Apple Silicon Mac recommended (for MLX acceleration)

### Setup

```bash
git clone https://github.com/czhou732/ClinicalWhisper.git
cd ClinicalWhisper
python3 -m venv venv
source venv/bin/activate

# Install with pyproject.toml (editable mode)
pip install -e .

# For Apple Silicon (recommended — 4–10× faster):
pip install mlx-whisper

# Download NLTK data (one-time)
python -c "import nltk; nltk.download('punkt_tab')"
```

### Optional Dependencies

```bash
# Speaker diarization (requires HuggingFace token)
pip install pyannote.audio
export HF_TOKEN="your_token_here"

# Acoustic features
pip install opensmile soundfile

# LLM clinical analysis (v4.0)
# Install Ollama from https://ollama.ai, then:
ollama pull qwen2:7b

# Meeting intelligence
ollama pull gpt-oss:20b
```

---

## Quick Start

### Single-file processing (watcher mode)

```bash
source venv/bin/activate
python main.py
```

Drop audio files (`.m4a`, `.mp3`, `.wav`, `.mp4`) into the `Input/` folder. Transcripts with full analysis will appear in your configured output folder.

### Async API mode

```bash
# Start the API server
python api.py

# Submit a job
curl -X POST http://127.0.0.1:8000/jobs \
  -F "file=@recording.m4a"
```

### Running as a daemon (macOS)

```bash
bash scripts/start_pipeline.sh        # Start pipeline + Plaud connector
bash scripts/start_pipeline.sh --stop  # Stop all
```

---

## Batch Processing

Process an entire directory of audio files at once:

```bash
python batch_processor.py --input ./audio_dir --output results.csv
```

With a custom config:

```bash
python batch_processor.py \
  --input /path/to/recordings \
  --output /path/to/results.csv \
  --config config.yaml
```

Output CSV columns:

| Column | Description |
|---|---|
| `filename` | Source audio filename |
| `word_count` | Total words in transcript |
| `duration_minutes` | Audio duration |
| `sentiment_score` | Overall sentiment (1–10) |
| `sentiment_label` | Positive / Neutral / Negative |
| `vta` | Zhou Index (vocal anhedonia) |
| `pitch_mean_st` | Mean F0 in semitones |
| `pitch_cv` | Pitch coefficient of variation |
| `loudness_mean_db` | Mean loudness in dB |
| `loudness_cv` | Loudness coefficient of variation |
| `jitter` | Pitch perturbation |
| `shimmer` | Amplitude perturbation |

---

## Research Tools

### CSV Export (R/SPSS compatible)

Flatten analysis JSONs into a single CSV for statistical analysis:

```bash
python export_features.py --input ./Output --output features.csv
```

The output includes 21 columns covering demographics, sentiment, acoustics, and metadata — ready for direct import into R, SPSS, or pandas.

### Longitudinal Tracking

Track a patient's metrics across multiple sessions:

```bash
python longitudinal.py --patient P001 --input ./Output
```

Detects trends (improving / stable / declining) in VTA, pitch variability, loudness variability, sentiment score, and word count using linear regression.

---

## Configuration

All settings live in `config.yaml`:

```yaml
model: "small.en"                        # Whisper model size
mlx_model: "mlx-community/whisper-small.en-mlx"

sentiment:
  enabled: true
  sentences_per_segment: 5

diarization:
  enabled: true                          # Requires pyannote.audio + HF token

acoustic_features:
  enabled: true                          # Requires opensmile

# v4.0 LLM features (requires Ollama)
llm_scoring:
  enabled: false                         # Enable for clinical scoring
  ollama_model: "qwen2:7b"

question_detection:
  enabled: false                         # Enable for probe categorization
  ollama_model: "qwen2:7b"

embeddings:
  enabled: false                         # Enable for ML feature export
  model: "sentence-transformers/all-MiniLM-L6-v2"
  export_format: "csv"                   # or "npy"
```

See [config.yaml](config.yaml) for all options including Plaud integration, Meeting Intelligence, and pipeline settings.

---

## Testing

```bash
# Run the full test suite
pytest

# Run with verbose output
pytest -v

# Run a specific test module
pytest tests/test_sentiment.py -v
```

The test suite covers:
- Sentiment analysis (transformer model loading, scoring, edge cases)
- Acoustic feature extraction and VTA computation
- Batch processing and CSV export
- Longitudinal trend detection
- Configuration loading

---

## Output Format

Each processed file produces:

1. **Markdown note** — Obsidian-compatible with YAML frontmatter, full transcript, speaker labels, sentiment breakdown, and acoustic metrics
2. **JSON analysis** (`*_analysis.json`) — Machine-readable output with all extracted features

JSON structure:

```json
{
  "job_id": "batch_a1b2c3d4e5f6",
  "model": "small.en",
  "pipeline_version": "4.0",
  "statistics": {
    "word_count": 1247,
    "duration_seconds": 423.5
  },
  "overall_sentiment": {
    "label": "Neutral",
    "score": 6,
    "polarity": 0.112
  },
  "overall_acoustics": {
    "pitch_mean_st": 42.3,
    "pitch_cv": 0.187,
    "loudness_mean_db": 65.2,
    "vta": 3.41
  },
  "speaker_roles": {
    "Speaker 1": "Interviewer",
    "Speaker 2": "Subject"
  },
  "structured_transcript": "[00:00 - 00:15] Interviewer: How have you been?\n...",
  "llm_clinical_scoring": {
    "hesitancy_score": 6,
    "affect_flatness": 7,
    "engagement_level": 4,
    "elaboration_positive": 3,
    "elaboration_negative": 8,
    "psychomotor_indicators": 5,
    "key_observations": ["Subject shows reduced prosodic variation..."],
    "clinical_impression": "The subject demonstrates..."
  },
  "clinical_probes": {
    "probes": [{"text": "How have you been?", "category": "rapport_building"}],
    "coverage": {"categories_covered": 7, "coverage_pct": 70}
  },
  "embeddings_file": "batch_a1b2c3d4e5f6_embeddings.csv",
  "segments": [...],
  "speaker_acoustics": {...}
}
```

---

## The Zhou Index

ClinicalWhisper computes a custom **Vocal Tone Anhedonia (VTA)** metric — the Zhou Index:

```
V_anh = −log(CV_F0 × CV_Energy)
```

Where:
- **CV_F0** = coefficient of variation of fundamental frequency (pitch variability)
- **CV_Energy** = coefficient of variation of loudness/energy

Higher VTA values indicate flatter, more monotone speech — a potential biomarker for anhedonia and depressive states. The metric leverages OpenSMILE's eGeMAPSv02 feature set, which provides normalized standard deviation (stddevNorm) as the coefficient of variation.

---

## Author

**Chengdong (Peter) Zhou** — USC Dornsife · NIMH Summer Intern (ETPB)
*Computational Psychiatry · Clinical AI · Vocal Biomarkers*

- GitHub: [@czhou732](https://github.com/czhou732)
- ORCID: [coming soon]

**Co-authors:** Meihui (Lily) Wu · Yiming (Dora) Xiang · Laurent Itti

---

## 📄 Publication

This pipeline is described in:

> Zhou, C., Wu, M., Xiang, Y., & Itti, L. (2026). Dopaminergic Vocal Biomarkers of Reward Processing in Clinical Speech. *bioRxiv*. DOI: [10.64898/2026.06.08.728970](https://doi.org/10.64898/2026.06.08.728970)

---

## Citation

If you use ClinicalWhisper in your research, please cite the paper:

```bibtex
@article{zhou2026dopaminergic,
  title     = {Dopaminergic Vocal Biomarkers of Reward Processing in Clinical Speech},
  author    = {Zhou, Chengdong and Wu, Meihui and Xiang, Yiming and Itti, Laurent},
  journal   = {bioRxiv},
  year      = {2026},
  doi       = {10.64898/2026.06.08.728970}
}
```

To cite the software specifically:

```bibtex
@software{zhou2026clinicalwhisper,
  author    = {Zhou, Chengdong},
  title     = {ClinicalWhisper: Privacy-First Clinical Speech Analysis Pipeline},
  year      = {2026},
  publisher = {GitHub / Zenodo},
  doi       = {10.5281/zenodo.20559786},
  url       = {https://github.com/czhou732/Clinical-Whisper-Pipeline}
}
```

---

## License

[MIT](LICENSE)
