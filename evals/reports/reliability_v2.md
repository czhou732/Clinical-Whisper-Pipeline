# Clinical scorer version 2: reliability

13 transcripts (12 synthetic vignettes from clinical_golden_dataset.jsonl and one
synthetic two-voice interview), each scored 5 times with sampling on
(temperature 0.7). ICC(1,1), one-way random effects (Shrout & Fleiss 1979), on
transcripts where all 5 runs found evidence for the score. Raw runs:
reliability_v2.json. Script: evals/reliability_v2.py.

| score | transcripts used | runs with no evidence | ICC(1,1) | 95% CI | mean of 5 runs |
|---|---|---|---|---|---|
| anhedonia_content | 8 / 13 | 12 / 65 | 0.956 | 0.89 - 0.99 | 0.99 |
| depressed_mood_content | 5 / 13 | 28 / 65 | 0.826 | 0.55 - 0.98 | 0.96 |
| affect_flatness | 9 / 13 | 10 / 65 | 0.309 | 0.05 - 0.70 | 0.69 |
| engagement_level | 11 / 13 | 2 / 65 | 0.693 | 0.46 - 0.89 | 0.92 |

Direction on the vignettes (mean score by category): interest/pleasure is 3.0 for
the anhedonia vignettes and 0.4 for engaged controls; engagement is highest for
engaged controls (2.0).

Read with care:
- Synthetic, short (15-130 words) and written to be clear-cut: real interviews
  will be noisier, so these ICCs are likely optimistic.
- Version 1 was measured on 9 real recordings, so the two versions are not yet
  compared on the same material.
- Reliability is not accuracy. Agreement with PHQ or SCID ratings still has to
  be measured on rated recordings (DAIC-WOZ dev split, Yale).
- Flat emotional language stays unreliable (0.69 with 5 runs) and is shown as such.
