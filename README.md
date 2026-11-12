# ClinicalWhisper

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/ChengdongPeter/Clinical-Whisper)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20559786.svg)](https://doi.org/10.5281/zenodo.20559786)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-10.64898%2F2026.06.08.728970-b31b1b)](https://doi.org/10.64898/2026.06.08.728970)

**A privacy-first, local pipeline for clinical interview transcription and research analysis.**

ClinicalWhisper v5.3 processes clinical audio recordings entirely on the local machine. Audio, transcripts, and results never leave the device during processing.

---

## Current release: v5.3

- 🗣️ **MOSS-Transcribe-Diarize 0.9B** — joint transcription and diarization in a single pass, replacing Whisper + Pyannote. Public and ungated; no Hugging Face token needed.
- 🛡️ **OpenMED PII scrubbing** — local identifier masking, followed by rule-based safety-net checks. Automatic masking still requires human review before data are shared.
- 🧠 **Scorer v2** — four evidence-backed 0–3 research ratings run through MLX on Apple Silicon or Transformers elsewhere. No Ollama or external inference service is used.
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
| **LLM research scoring** | Llama-3-8B rates anhedonia content, depressed-mood content, emotional-language flatness, and engagement (0–3, with transcript evidence). **These experimental scores are not clinically validated — see Status below.** |
| **Batch processing** | Directory → CSV pipeline with per-file error handling |

## The Pipeline

```
Audio File (.wav/.mp3/.m4a/.mp4/.ogg/.opus/.flac/.aac)
    │
    ▼  PyAV → 16 kHz mono WAV, plus a level-normalised copy
┌──────────────────────────────────────────────────────────────────┐
│                      ClinicalWhisper v5.3                        │
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

**The four scorer-v2 outputs are experimental research features.** A local LLM
reads the transcript and assigns 0–3 ratings for `anhedonia_content`,
`depressed_mood_content`, `affect_flatness`, and `engagement_level`. Every reported
rating requires a quotation from the participant; unsupported ratings remain
missing rather than being replaced with a default number. Agreement with PHQ-9,
SHAPS, HAM-D, SCID, clinician ratings, or any other clinical reference has not been
established.

Test-retest reliability has been measured on 13 short test transcripts (12
synthetic vignettes and one synthetic two-voice interview), each scored five times
with sampling enabled. See `evals/reports/reliability_v2.md` and its raw JSON:

| scorer-v2 output | transcripts with complete evidence | ICC(1,1) | 95% CI | reliability of five-run mean |
|---|---:|---:|---:|---:|
| `anhedonia_content` | 8 / 13 | 0.956 | 0.89–0.99 | 0.99 |
| `depressed_mood_content` | 5 / 13 | 0.826 | 0.55–0.98 | 0.96 |
| `affect_flatness` | 9 / 13 | 0.309 | 0.05–0.70 | 0.69 |
| `engagement_level` | 11 / 13 | 0.693 | 0.46–0.89 | 0.92 |

The synthetic examples are short and deliberately clear, so these estimates may
be optimistic for real interviews. Missing evidence also reduced the usable sample
for each rating. In particular, `affect_flatness` remains unreliable even after
five runs. Reliability is repeatability, not clinical accuracy or validity.

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
| **Disk / network** | The base DMG contains the transcription and masking models. Clinical scoring is a separate ~5 GB Scoring add-on. Running **from source** downloads the configured models into the local cache on first use. |

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

The base app transcribes and masks English interviews. Experimental clinical
scores require the separate **ClinicalWhisper Scoring** add-on. Language masking,
name silencing, Praat, and Kintsugi features are also distributed as optional
add-ons; install them with **Install an add-on…** at the bottom of the app.

The packaged app writes files to `~/ClinicalWhisper/`, outside Documents so that
common iCloud Documents settings do not automatically sync study data:

```
~/ClinicalWhisper/
├── Output/       # <name>_analysis.json, summary CSVs, and optional exports
└── config.yaml   # optional user configuration
```

The desktop app reads selected recordings in place; it does not move or copy the
source audio. Override the data location with `CLINICALWHISPER_DATA_DIR`. To
customise the pipeline, copy `config.example.yaml` to
`~/ClinicalWhisper/config.yaml` — the app prefers it over the bundled
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

One CSV row per file includes identifiers and quality flags, word count and
duration, the four scorer-v2 outputs, measured timing features, and whole-recording
and participant acoustic features. The complete analysis, transcript evidence,
provenance, warnings, and per-window details remain in the per-recording JSON.

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
  confidence_threshold: 0.5
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

## Contributing

Contributions are welcome. Open an issue first to describe the bug or the change
you have in mind, then send a pull request against `main`. Add or update tests
for the change and run `pytest` before opening the request; CI runs the same
suite on Python 3.11 and 3.12. Use synthetic or public audio (for example, the
AMI meeting corpus) in tests and examples, never clinical recordings or study
data.

## Support and problem reports

For usage questions or reproducible bugs, open a
[GitHub issue](https://github.com/czhou732/Clinical-Whisper-Pipeline/issues).
Include the ClinicalWhisper version, operating system, exact command or app steps,
and the error message. The desktop app can create a sanitized diagnostics archive
with **Save diagnostics for a bug report**; it excludes audio and transcripts and
removes file names. That archive can also be sent to Peter Zhou at
`czhou732@usc.edu`.

Do not attach clinical audio, raw transcripts, identifiers, credentials, model
weights, or other private study data to a public issue. Report suspected security
or privacy problems privately by email rather than in the public tracker.

---

## Citation

If you use ClinicalWhisper in your research, please cite the associated
[bioRxiv preprint](https://doi.org/10.64898/2026.06.08.728970):

```bibtex
@article{zhou2026clinicalwhisper,
  title={Cross-Modal Benchmarking of Acoustic Prosody and Ventral Striatal BOLD for Depression-Related Anhedonia Classification: A Pre-Registered Study with the ClinicalWhisper Pipeline},
  author={Zhou, Chengdong and Wu, Muqing and Xiang, Yi and Itti, Laurent},
  journal={bioRxiv},
  year={2026},
  doi={10.64898/2026.06.08.728970}
}
```
