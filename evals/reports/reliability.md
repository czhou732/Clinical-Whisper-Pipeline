# Scorer reliability on real recordings

9 clips from separate source recordings, scored 5 times each with sampling on (temperature 0.7).

One-way random-effects ICC(1,1), per Shrout & Fleiss (1979): the repeated scores are exchangeable draws from one stochastic process, not a fixed panel of identifiable raters.

| dimension | mean | range | SEM | SD between | ratio | ICC(1,1) | 95% CI | |
|---|---|---|---|---|---|---|---|---|
| hesitancy_score | 4.53 | 1–9 | 1.606 | 1.022 | 0.64 | 0.288 | 0.03 – 0.68 | poor |
| affect_flatness | 4.47 | 1–9 | 1.358 | 1.361 | 1.0 | 0.501 | 0.21 – 0.82 | moderate |
| engagement_level | 5.51 | 3–8 | 1.3 | 0.987 | 0.76 | 0.366 | 0.09 – 0.74 | poor |
| elaboration_positive | 3.47 | 1–8 | 1.049 | 1.273 | 1.21 | 0.596 | 0.31 – 0.86 | moderate |
| elaboration_negative | 5.47 | 2–9 | 1.438 | 1.535 | 1.07 | 0.533 | 0.24 – 0.83 | moderate |
| psychomotor_indicators | 4.47 | 2–6 | 1.342 | 0.707 | 0.53 | 0.217 | -0.01 – 0.62 | poor |

## Definitions

- **SEM** — standard error of measurement, sqrt(MSW) from the one-way ANOVA. The pooled within-target SD.
- **SD between** — between-target variance component, sqrt((MSB - MSW) / k), which excludes measurement error.
- **ratio** — SD between / SEM. Above ~2 means a score separates recordings well clear of its own noise floor.
- **MDC95** — smallest detectable change, 1.96 * sqrt(2) * SEM: the gap two recordings must show before it exceeds measurement error.

| dimension | MDC95 (0–10 scale) |
|---|---|
| hesitancy_score | 4.45 |
| affect_flatness | 3.76 |
| engagement_level | 3.6 |
| elaboration_positive | 2.91 |
| elaboration_negative | 3.99 |
| psychomotor_indicators | 3.72 |

## Limitations

- **n = ? targets** is well below the ~30 usually recommended for an ICC study, which is why the confidence intervals above are very wide. Treat the point estimates as indicative.
- Clips were drawn two per source recording, so they are **clustered** rather than fully independent; the between-target component is likely overstated.
- Measured with sampling on. The shipped default is greedy decoding, so in normal use the same file returns the same score. A low ICC does not mean the app is unstable — it means the single score is one draw from a wide distribution, and will move under small changes to prompt, transcript or model version.

Cronbach's alpha across the six dimensions: 0.071. Alpha assumes the items measure one construct; these six are meant to be distinct, so a low value indicates they are not redundant rather than that anything is wrong.

Reliability only. This says nothing about agreement with a clinical instrument — that is validity, and needs criterion scores collected at recording time.

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