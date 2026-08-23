# Scorer reliability on real recordings

Clips: 9 from separate source recordings  |  runs per clip: [5]  |  temperature 0.7

| dimension | mean | range | within-clip SD | between-clip SD | ratio | ICC(1,1) | |
|---|---|---|---|---|---|---|---|
| hesitancy_score | 4.53 | 1–9 | 1.279 | 1.178 | 0.92 | 0.288 | poor |
| affect_flatness | 4.47 | 1–9 | 1.079 | 1.405 | 1.3 | 0.501 | moderate |
| engagement_level | 5.51 | 3–8 | 1.127 | 1.08 | 0.96 | 0.366 | poor |
| elaboration_positive | 3.47 | 1–8 | 0.833 | 1.279 | 1.54 | 0.596 | moderate |
| elaboration_negative | 5.47 | 2–9 | 1.172 | 1.569 | 1.34 | 0.533 | moderate |
| psychomotor_indicators | 4.47 | 2–6 | 1.106 | 0.874 | 0.79 | 0.217 | poor |

## Smallest detectable difference

How far apart two recordings must score before the gap exceeds measurement noise (1.96 x sqrt(2) x within-clip SD), on a 0–10 scale:

- **hesitancy_score**: 3.5 points
- **affect_flatness**: 3.0 points
- **engagement_level**: 3.1 points
- **elaboration_positive**: 2.3 points
- **elaboration_negative**: 3.2 points
- **psychomotor_indicators**: 3.1 points

## How to read this

`ratio` is between-clip SD over within-clip SD. Above ~2 means a score separates recordings well clear of its own noise floor. Every dimension here sits between 0.8 and 1.6, so differences between recordings are roughly the same size as the noise from re-scoring one recording.

**This is measured with sampling on (temperature 0.7). The shipped default is greedy, so in normal use the same file always returns the same score.** What these numbers describe is not run-to-run flakiness in the app — it is how sharp the underlying judgement is. A low ICC means the greedy answer is one draw from a wide distribution rather than a stable estimate, so it will move under small changes to the prompt, the transcript, or the model.

Cronbach's alpha across the six dimensions: 0.071. Alpha assumes the items measure one construct; these six are meant to be distinct, so a low value indicates they are not redundant rather than that anything is broken.

Reliability only. This says nothing about agreement with a clinical instrument — that is validity, and it needs criterion scores collected at recording time.

## Dimensions that move together

Spearman rho >= 0.6 across per-clip means:

- hesitancy_score vs engagement_level: -0.83
- hesitancy_score vs affect_flatness: 0.82
- engagement_level vs elaboration_positive: 0.82
- affect_flatness vs engagement_level: -0.76
- elaboration_positive vs elaboration_negative: 0.63
- hesitancy_score vs psychomotor_indicators: 0.62
- engagement_level vs elaboration_negative: 0.61
- affect_flatness vs psychomotor_indicators: 0.6