# ClinicalWhisper

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/ChengdongPeter/Clinical-Whisper)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20559786.svg)](https://doi.org/10.5281/zenodo.20559786)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-10.64898%2F2026.06.08.728970-b31b1b)](https://doi.org/10.64898/2026.06.08.728970)

**A privacy-first, on-device multimodal framework for anhedonia classification.**

ClinicalWhisper v5.1 processes clinical audio recordings entirely on the local machine. Audio, transcripts, and results never leave the device.

---

## What's New in v5.0

- 🗣️ **MOSS-Transcribe-Diarize 0.9B** — joint transcription and diarization in a single pass, replacing Whisper + Pyannote. Public and ungated; no Hugging Face token needed.
- 🛡️ **OpenMED PII scrubbing** — local HIPAA Safe Harbor de-identification of the transcript (names, dates, addresses, phone numbers, MRNs).
- 🧠 **Native MLX LLM scoring** — Llama-3-8B-Instruct runs on Apple Silicon via `mlx-lm`. No Ollama, no external services.
- 🖥️ **Native desktop app** — a standalone `.app`; results render in the window and save where you choose.

### Previous releases

- **v1.0** — Initial release (Whisper + Pyannote baseline)
- **v2.0** — Introduced RoBERTa sentiment analysis (removed from the pipeline in v5.0)
- **v3.0** — Integrated OpenSMILE (VTA Zhou Index)
- **v4.0** — Added local LLM inference via Ollama

---

## Architecture Overview

| Category | Details |
|---|---|
| **Transcription & Diarization** | MOSS-Transcribe-Diarize 0.9B (MPS on Apple Silicon, CUDA, or CPU) |
| **HIPAA de-identification** | OpenMED `OpenMed-PII-SuperClinical-Small-44M-v1`, mask method |
| **Acoustic features** | OpenSMILE eGeMAPSv02 — pitch, loudness, jitter, shimmer, VTA |
| **LLM clinical scoring** | Llama-3-8B scores hesitancy, affect flatness, engagement, elaboration, psychomotor indicators (0–10). **Not yet validated against any clinical instrument — see Status below.** |
| **Batch processing** | Directory → CSV pipeline with per-file error handling |

## The Pipeline

```
Audio File (.m4a/.mp3/.wav/.mp4)
    │
    ▼  PyAV → 16 kHz mono WAV, plus a level-normalised copy
┌──────────────────────────────────────────────────────────────────┐
│                      ClinicalWhisper v5.0                        │
│                                                                  │
│  Stage 1: Transcribe & Diarize   (MOSS, normalised audio)        │
│  Stage 2: PII Scrubbing          (OpenMED, HIPAA Safe Harbor)    │
│  Stage 3: Acoustic Analysis      (OpenSMILE, original gain)      │
│  Stage 4: Role Detection         (Interviewer / Subject)         │
│  Stage 5: LLM Clinical Scoring   (Llama-3-8B via mlx_lm, windowed) │
│                          │                                       │
│                          ▼                                       │
│              JSON Analysis Report  +  Summary CSV                │
└──────────────────────────────────────────────────────────────────┘
```

Transcription runs on a level-normalised copy because clinical recordings are
often faint, and MOSS returns an empty transcript on very quiet audio. Acoustic
features are extracted from the **original-gain** audio, since loudness and VTA
are amplitude-dependent and normalisation would invalidate them.

---

## Status of the measures

Two families of numbers come out of this pipeline, and they do not carry the same
evidentiary weight.

**Acoustic features** (`pitch_mean_st`, `pitch_cv`, `loudness_cv`, `jitter`,
`shimmer`) are eGeMAPSv02 via OpenSMILE — a published, standardised parameter set,
so the values are comparable to the affective-computing literature.

**The six clinical scores are unvalidated, and their reliability is low.** They
are the output of a local LLM reading the transcript. There is no correlation with
PHQ-9, SHAPS, HAM-D or any other instrument, and no inter-rater reliability against
clinicians.

Reliability *has* now been measured — see `evals/reports/reliability.md`. Across 9
clips from 5 separate recordings, scored 5 times each with sampling on, using
one-way random-effects ICC(1,1) (Shrout & Fleiss, 1979):

| | ICC(1,1) | 95% CI | MDC95 | |
|---|---|---|---|---|
| psychomotor_indicators | 0.22 | -0.01 – 0.62 | 3.7 | poor |
| hesitancy_score | 0.29 | 0.03 – 0.68 | 4.5 | poor |
| engagement_level | 0.37 | 0.09 – 0.74 | 3.6 | poor |
| affect_flatness | 0.50 | 0.21 – 0.82 | 3.8 | moderate |
| elaboration_negative | 0.53 | 0.24 – 0.83 | 4.0 | moderate |
| elaboration_positive | 0.60 | 0.31 – 0.86 | 2.9 | moderate |

No dimension reaches the conventional "good" threshold of 0.75, and every
confidence interval is wide because n = 9 is well below the ~30 usually
recommended for an ICC study. MDC95 is the smallest change exceeding measurement
error: on a 0–10 scale, two recordings must differ by **2.9–4.5 points** before
the gap is distinguishable from noise.

The shipped default is greedy decoding, so in normal use the same file returns the
same score every time — this is not run-to-run instability in the app. What a low
ICC means is that the score is one draw from a wide distribution rather than a
stable estimate, and will move under small changes to prompt, transcript, or model
version. Averaging several samples (`llm_scoring.samples: 5`) reduces that spread.

These should be described as automated interview features whose agreement with
clinical judgement has not been established — not as a validated instrument, and
not as a diagnosis. Correlating them against a clinical scale would currently be
limited by their own reliability rather than by the construct.

The eval suite (`evals/`) is a **regression test**, not a validation study: it
checks that the scorer still behaves as it did before a code change, using
synthetic vignettes with author-assigned expected ranges. Its pass rate says
nothing about clinical accuracy.

**VTA** is a derived index (`-ln(CV_F0 x CV_Energy)`) with no external
validation. Higher values mean *less* prosodic variability, so a high VTA is the
anhedonia-relevant direction. Its bands (2.4 / 4.6) are derived from the pitch
and loudness CV ranges so the index agrees with its own components — they are
internally consistent, not empirically calibrated, and should be replaced with
percentiles from a real corpus before being reported.

---

## Requirements

| | |
|---|---|
| **Hardware** | Apple Silicon Mac (M1 or newer) for the MLX path. Check with `uname -m` — it must print `arm64`. Intel Macs, Windows, and Linux fall back to HuggingFace transformers on CUDA or CPU, which is substantially slower. |
| **ffmpeg** | Not required. Audio is decoded through PyAV, which ships its own ffmpeg libraries. |
| **Disk / network** | The DMG is self-contained: model weights are inside the app, so nothing downloads and no Hugging Face account is needed. Running **from source** instead pulls ~6.4 GB into `~/.cache` on first use. |

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
git clone https://github.com/czhou732/Clinical-Whisper-Pipeline.git
cd Clinical-Whisper-Pipeline
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
  device: "auto"          # auto | mps | cuda | cpu
  dtype: "auto"           # float16 on Apple Silicon: ~2.4x faster, same output
  hotwords: []            # drug/scale names the transcriber should favour

pii_scrubbing:
  enabled: true
  confidence_threshold: 0.7
  strict: true            # abort rather than emit an unscrubbed transcript

llm_scoring:
  enabled: true
  mlx_model: "mlx-community/Meta-Llama-3-8B-Instruct-4bit"   # Apple Silicon
  hf_model: "NousResearch/Meta-Llama-3-8B-Instruct"          # everywhere else
  window_words: 2800      # long interviews are windowed, not truncated
  transcript_scope: "dialogue"   # or subject_only
  samples: 1              # >1 reports mean/SD instead of a point estimate

audio_retention: "archive"   # or "delete" to leave no identifiable audio
keep_models_loaded: false    # release model memory when a batch finishes
```

### Memory

A batch runs in two phases — transcribe every file, release the transcription
model, then score every file — so the 1.7 GB transcriber and the 4.9 GB scorer
are never resident together. Peak model memory is ~4.9 GB rather than ~6.6 GB,
and both are released when the batch ends unless `keep_models_loaded` is set.

Audio is decoded block by block straight to disk, so a two-hour recording costs
a few MB rather than the ~300 MB it would take to hold as one array.

### Scoring long interviews

Transcripts longer than one context window are split on turn boundaries, scored
window by window, and averaged. `_meta.coverage` records how much was scored and
`_meta.per_window_scores` keeps the individual results, so a score that drifted
across an interview is visible rather than averaged away.

This replaces the previous behaviour, which kept the first 3000 words and
dropped the rest — a 60-minute interview was scored on its first third while the
output looked complete.

### Not currently in the pipeline

`longitudinal.py`, `question_detector.py`, `llm_embeddings.py` and
`sentiment_analyzer.py` are present in the repository but are **not imported by
the pipeline**. They are available as standalone tools; nothing in the app or the
batch processor calls them.

### Reproducibility

Every analysis carries a `provenance` block: resolved model commit hashes, the
scoring settings used, package versions, platform, and the source commit when
run from a checkout. A result stays reconstructible even if a model repo is
updated in place later.

Transcription and PII scrubbing failures abort the job rather than emitting a
partial result — an analysis file full of zeros is indistinguishable from a real
one, so the pipeline refuses to produce one.

---


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
