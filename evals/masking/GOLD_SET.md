# Real-transcript gold set for masking (iLab)

The synthetic sets show the masker's ceiling on clean sentences. Real speech
has disfluency, recognition errors, nicknames and identifiers split across
segments. This protocol measures masking on consented real recordings without
any identifier leaving the machine they are stored on.

## Before starting

- **Check the IRB approval covers it.** The recordings must come from a
  protocol whose consent allows the research team to read the raw transcript
  and use it to evaluate de-identification software. If that is not clearly
  covered, file an amendment first. Team members without approval for that
  study do not take part.
- Never use DAIC-WOZ, Yale or UP-26-00343 data for this; they have their
  own terms.

## Steps (all on the iLab machine)

1. **Pick ~10 recordings, about 60 minutes in total:** a mix of one-to-one
   interviews and group sessions, with at least two speakers who mention other
   people by name.
2. **Raw transcripts.** Run ClinicalWhisper with masking switched off
   (`pii_scrubbing.enabled: false` in a copy of the config), and write one
   segment per line as `SPEAKER: text`. These files contain identifiers:
   keep them in the protocol's secure folder.
3. **Two annotators, independently.** Each copies the raw files and marks
   every identifier as `{{type:text}}` (types and rules are in
   `gold_from_markup.py`). If unsure, mark it.
4. **Agreement.** `python evals/masking/gold_from_markup.py agree A/ B/`.
   Report exact and overlap F1. Overlap F1 under 0.85 means the guidelines
   need tightening before going on.
5. **Adjudicate.** A third person (or both together) resolves every
   difference into one folder.
6. **Build and score.**
   ```
   python evals/masking/gold_from_markup.py build ADJ/ --out gold.jsonl
   python evals/masking/evaluate.py --set /abs/path/gold.jsonl --whole --label "CW 5.3"
   python evals/masking/evaluate.py --set /abs/path/gold.jsonl --whole --no-safety-net --label "OpenMed alone"
   ```
   For Presidio, see `presidio_baseline.py` (it needs its own environment).
7. **Share only the numbers.** With `--whole`, the JSON summary holds counts
   only (no quoted sentences). Misses are reviewed on iLab with `--show-misses`
   (printed to the screen, never saved),
   and patterns ("nicknames after 'my'") are described in words, never quoted.

## What is reported

Recall overall and by identifier type, over-masking, inter-annotator F1, and
the number of recordings, minutes and identifiers. Because the safety-net rules
will be improved using what the gold set shows, freeze it after the first
scoring and set aside about a third of it, unread, for a final check.
