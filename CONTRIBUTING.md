# Contributing to ClinicalWhisper

Thank you for helping improve ClinicalWhisper. It is research software for clinical
interview audio, so every contribution has to keep participant data private.

## Reporting a bug or asking a question

Open a [GitHub issue](https://github.com/czhou732/Clinical-Whisper-Pipeline/issues).
Include the ClinicalWhisper version, your operating system, the exact command or app
steps, and the error message. The desktop app's **Save diagnostics for a bug report**
creates an archive without audio, transcripts or file names that you can attach.

Never attach clinical audio, raw transcripts, identifiers, credentials, model weights
or other study data to an issue or pull request. Report suspected security or privacy
problems privately to Peter Zhou (`czhou732@usc.edu`) instead of in the public tracker.

## Proposing a change

1. Open an issue first describing the bug or the change, so the approach can be agreed
   before you write code.
2. Fork the repository and create a branch from `main`.
3. Set up a development environment (Python 3.11 or 3.12):
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -e .
   pip install pytest
   ```
4. Add or update tests for your change and run `pytest`. CI runs the same suite on
   Python 3.11 and 3.12, and a pull request must pass it.
5. Use synthetic or public audio (for example the AMI meeting corpus) in tests,
   examples and evaluations.
6. Write commit messages in the Conventional Commits style (`fix(scoring): ...`,
   `feat(masking): ...`, `docs: ...`).
7. Open a pull request against `main` that links the issue. If the change affects a
   reported number, rerun the script in `evals/` that produced it and update the report.

## Masking and clinical scores

Changes to identifier masking, role labelling or the clinical scores need extra care:

- Masking must stay on by default, and a job must still stop rather than write an
  unmasked transcript when masking fails.
- Report masking recall from `evals/masking/` before and after the change.
- The clinical scores are research measures, not clinical assessments. Do not add
  wording or outputs that present them as diagnoses.

## Maintainers and decisions

ClinicalWhisper is maintained by Chengdong (Peter) Zhou and Jinghao (Tony) Yu at the
University of Southern California. Maintainers review pull requests, usually within
two weeks, and decide on changes to the pipeline's scope and defaults. Contributors
are credited in the release notes.
