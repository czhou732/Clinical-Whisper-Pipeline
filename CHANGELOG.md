# Changelog

Dates are those of the tagged commits. Release notes for each version are on the
[releases page](https://github.com/czhou732/Clinical-Whisper-Pipeline/releases).

## 5.3.0 (2026-11-11)

- Failed and partial LLM scoring is reported in each JSON record and batch CSV row (#1).
- The scorer's `--model` option overrides the active MLX or Transformers model (#2).
- English and Chinese READMEs updated for 5.3 (#3, #4).
- `openmed` pinned below 2.0, the version the masking evaluations used; PyAV and SciPy
  declared as dependencies. CI passes on Python 3.11 and 3.12.

## 5.2.0 (2026-10-01)

- Measured masking recall and a rule-based safety net after OpenMED.
- De-identification benchmark against Presidio, with the scripts in `evals/masking/`.

## 5.0 (2026-07-26)

- MOSS-Transcribe-Diarize replaces Whisper and pyannote for transcription and speaker
  separation.
- OpenMED identifier masking on by default.
- Local LLM scoring through MLX; Ollama removed. RoBERTa sentiment removed.
- New desktop interface.

## 4.0.0 (2026-06-17)

- Local LLM clinical scoring through Ollama, with structured Interviewer and Subject
  transcripts.

## 3.0.0 (2026-06-11)

- Transformer sentiment model, batch processing, longitudinal tracking and CSV export.
- pytest suite and GitHub Actions CI.

## 2.1-preprint (2026-06-05)

- Version tagged for the bioRxiv preprint.

## 1.0.0 (2026-03-25)

- First release: Whisper transcription and pyannote speaker separation. Archived on
  Zenodo on 2026-06-05 (doi:10.5281/zenodo.20559786).
