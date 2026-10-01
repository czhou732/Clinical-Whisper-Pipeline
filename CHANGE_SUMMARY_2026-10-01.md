# Change Summary

- **Date:** October 1, 2026
- **Author:** Tony Yu
- **Project:** USC PRAXIS ClinicalWhisper

## Overview

This document summarizes the repository changes made during the dependency investigation and the follow-up fix for hidden LLM scoring failures. The work addresses two independent issues:

1. MP3 decoding dependencies were used at runtime but were not all declared as direct project dependencies.
2. LLM scoring could return failure metadata with default numeric scores, while the overall analysis was still presented as successfully completed.

## 1. Audio Decoding Dependency Fix

### Problem

Processing `HospitalBeep.mp3` initially failed with the following runtime message:

> Audio decoding requires PyAV, soundfile and numpy. Reinstall dependencies with uv pip install -e .

`soundfile` was already declared, but `av` (PyAV) and `numpy` were not direct project dependencies. Their presence therefore depended on transitive dependencies or the state of an individual developer's environment.

### Changes

- Added `av` and `numpy` to the main dependency list in `pyproject.toml`.
- Added `av` and `numpy` to `requirements.txt` for consistency with the project metadata.
- Regenerated `uv.lock` so that PyAV and NumPy are represented as direct dependencies of the `clinicalwhisper` package, with platform- and Python-version-appropriate resolutions.

### Files

- `pyproject.toml`
- `requirements.txt`
- `uv.lock`

### Environment Verification

- Installed the project in editable mode in `.venv`.
- Installed the MOSS transcription/diarization package and its model assets.
- Confirmed that the active virtual environment can import `av`, `soundfile`, and `numpy`.
- Confirmed that PyAV can decode `Input/HospitalBeep.mp3` successfully.
- Confirmed that the pipeline's audio preprocessing step can create temporary 16 kHz mono PCM WAV files and clean them up afterward.

The later empty transcription result for `HospitalBeep.mp3` was not an audio-decoding failure. The MOSS model loaded and ran, but the recording did not produce recognizable speech segments.

## 2. LLM Scoring Failure Visibility

### Problem

The LLM scorer may handle generation failures internally and return default clinical scores together with an `_meta.error` or `_meta.errors` field. Because no exception reaches the pipeline in that case, the analysis could previously be marked as completed without clearly exposing that the clinical scoring stage failed or only partially covered the transcript.

This behavior is especially important for long interviews, where some scoring windows may succeed while others fail.

### Pipeline Changes

Added `InferencePipeline._assess_llm_scoring()` to classify scorer output as one of the following states:

- `not_run`: no scoring result was produced.
- `disabled`: LLM scoring was disabled by configuration.
- `completed`: scoring completed without reported errors.
- `partial`: one or more scoring runs failed, or transcript coverage was below 100%.
- `failed`: the scorer returned a top-level error or raised an exception.

The JSON analysis report now includes:

- `llm_scoring_status`
- `llm_scoring_coverage`

Returned scorer errors and incomplete coverage are also added to the report's `warnings` list. Because the pipeline already derives its overall status from warnings, affected analyses are now reported as `completed_with_warnings` instead of appearing fully successful.

### Batch CSV Changes

The batch summary row now includes:

- `analysis_status`
- `llm_scoring_status`
- `llm_scoring_coverage`
- `warnings`

For compatibility with existing analysis JSON files, the batch processor also infers LLM status and coverage from legacy `_meta.error`, `_meta.errors`, and `_meta.coverage` fields when the new top-level fields are absent.

### Files

- `inference_pipeline.py`
- `batch_processor.py`

## 3. Tests Added

### Pipeline Tests

Extended `tests/test_pipeline.py` with coverage for:

- Total LLM scoring failure.
- Partial scoring failure and coverage reporting.
- Successful scoring with full coverage.
- End-to-end `score_job()` behavior when the scorer returns failure metadata instead of raising an exception.

### Batch Processor Tests

Added `tests/test_batch_processor.py` with coverage for:

- Exporting failure status, coverage, overall analysis status, and warnings to the CSV row.
- Correctly classifying legacy partial-scoring results that only contain `_meta` fields.

### Test Result

The targeted test suite completed successfully:

```text
12 passed in 1.53s
```

The checks covered `tests/test_pipeline.py` and `tests/test_batch_processor.py`. This was a targeted test run, not the repository's entire test suite.

## 4. Files Changed by This Work

| File | Purpose |
| --- | --- |
| `pyproject.toml` | Declares PyAV and NumPy as direct runtime dependencies. |
| `requirements.txt` | Keeps the alternative dependency list consistent. |
| `uv.lock` | Locks the newly declared direct dependencies. |
| `inference_pipeline.py` | Detects and reports failed or partial LLM scoring. |
| `batch_processor.py` | Exposes analysis and LLM scoring health in summary CSV output. |
| `tests/test_pipeline.py` | Tests scoring-status classification and pipeline integration. |
| `tests/test_batch_processor.py` | Tests CSV reporting and legacy-result compatibility. |
| `CHANGE_SUMMARY_2026-10-01.md` | Documents the changes, validation, and scope. |

## 5. Unrelated Working-Tree Items

The following items were already present in the working tree and were not intentionally changed as part of this work:

- `Clinical_Whisper_Blueprint.pdf`
- `.idea/`

They should be reviewed separately and excluded from a focused pull request unless they are intentionally needed.

## 6. Pull Request Scope Recommendation

The dependency fix and the LLM scoring-status fix solve separate problems. For a clean public contribution, consider submitting them as separate commits or separate pull requests:

1. Declare the audio decoding dependencies and update the lock file.
2. Surface failed and partial LLM scoring in JSON and batch CSV outputs, with tests.

No commit has been created as part of this work.
