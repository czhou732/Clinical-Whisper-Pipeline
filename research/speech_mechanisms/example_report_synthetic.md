> **Synthetic data** (`research/speech_mechanisms/synthetic.py`, n = 150, seed 11): item 8 drives motor slowing, item 1 drives blunted positive reactivity, item 2 drives neither. Section 4 compares against the same transcripts with ±150 ms boundary jitter, standing in for automatic timing error. It shows the analysis recovers a known mechanism — and what it cannot recover: `pause_rate_pw` follows elaboration (item 1) because pause counts depend on answer length, and `timing_sigma` / `beta_log_latency` do not survive timing error. Those are therefore secondary outcomes.

# Speech mechanisms: timing vs reward reactivity

150 interviews, 150 with PHQ-8 items; 2250 positive-prompt responses of 6750 labelled.

## 1. Discriminant validity — standardized coefficients (items entered together)

| parameter | n | item 1 (anhedonia) | item 2 (mood) | item 8 (psychomotor) | strongest |
|---|---|---|---|---|---|
| timing_mu | 150 | +0.08 [-0.05, +0.21] | -0.01 [-0.14, +0.13] | +0.62 [+0.49, +0.74] | item8 |
| timing_sigma | 150 | -0.01 [-0.20, +0.17] | -0.14 [-0.31, +0.03] | +0.02 [-0.14, +0.18] | item2 |
| timing_tau | 150 | +0.17 [+0.04, +0.29] | -0.06 [-0.19, +0.07] | +0.66 [+0.54, +0.79] | item8 |
| pause_log_mean | 150 | +0.10 [-0.03, +0.22] | +0.01 [-0.12, +0.15] | +0.66 [+0.54, +0.79] | item8 |
| pause_log_sd | 150 | +0.03 [-0.14, +0.18] | -0.12 [-0.30, +0.06] | +0.45 [+0.30, +0.61] | item8 |
| pause_rate_pw | 150 | -0.63 [-0.77, -0.50] | +0.09 [-0.04, +0.22] | +0.40 [+0.25, +0.55] | item1 |
| speech_rate_wps | 150 | -0.11 [-0.22, +0.01] | +0.03 [-0.08, +0.14] | -0.75 [-0.86, -0.65] | item8 |
| filler_rate | 150 | +0.07 [-0.08, +0.21] | -0.03 [-0.19, +0.12] | +0.45 [+0.28, +0.60] | item8 |
| beta_words | 150 | -0.77 [-0.88, -0.66] | +0.07 [-0.05, +0.21] | +0.08 [-0.05, +0.21] | item1 |
| beta_speech_rate_wps | 150 | -0.12 [-0.32, +0.08] | +0.02 [-0.18, +0.23] | -0.02 [-0.20, +0.16] | item1 |
| beta_log_latency | 150 | +0.02 [-0.16, +0.20] | +0.01 [-0.19, +0.20] | +0.13 [-0.04, +0.30] | item8 |
| beta_filler_rate | 150 | -0.06 [-0.23, +0.11] | -0.08 [-0.26, +0.10] | +0.10 [-0.06, +0.27] | item8 |
| mean_response_words | 150 | -0.76 [-0.87, -0.65] | +0.05 [-0.07, +0.18] | +0.12 [+0.01, +0.24] | item1 |

## 2. Reward-reactivity test (pre-registered H2)

`feature ~ positive * item1 + item2 + (1 | participant)`; key term `positive:item1`.

| feature | positive × item1 | 95% CI | p |
|---|---|---|---|
| words | -5.064 | [-5.340, -4.787] | 0.0000 |
| speech_rate_wps | -0.006 | [-0.015, +0.002] | 0.1348 |
| log_latency | +0.008 | [-0.012, +0.028] | 0.4438 |
| filler_rate | -0.048 | [-0.174, +0.078] | 0.4521 |

## 3. Which mechanism predicts symptoms? (nested CV, Spearman rho)

**Target: item8** (n = 150)

| features | mean rho | 95% of repeats |
|---|---|---|
| timing (A) | +0.748 | [+0.562, +0.867] |
| reactivity (B) | +0.114 | [-0.171, +0.401] |
| A + B | +0.732 | [+0.536, +0.853] |

| comparison | mean diff | corrected t | p |
|---|---|---|---|
| timing (A) vs reactivity (B) | +0.634 | +7.98 | 0.0000 |
| timing (A) vs A + B | +0.016 | +1.05 | 0.2945 |
| reactivity (B) vs A + B | -0.618 | -7.92 | 0.0000 |

**Target: item1** (n = 150)

| features | mean rho | 95% of repeats |
|---|---|---|
| timing (A) | +0.652 | [+0.435, +0.812] |
| reactivity (B) | +0.707 | [+0.526, +0.826] |
| A + B | +0.713 | [+0.549, +0.845] |

| comparison | mean diff | corrected t | p |
|---|---|---|---|
| timing (A) vs reactivity (B) | -0.055 | -1.12 | 0.2660 |
| timing (A) vs A + B | -0.061 | -2.13 | 0.0355 |
| reactivity (B) vs A + B | -0.006 | -0.22 | 0.8259 |

**Target: total** (n = 150)

| features | mean rho | 95% of repeats |
|---|---|---|
| timing (A) | +0.634 | [+0.414, +0.803] |
| reactivity (B) | +0.440 | [+0.201, +0.665] |
| A + B | +0.668 | [+0.440, +0.821] |

| comparison | mean diff | corrected t | p |
|---|---|---|---|
| timing (A) vs reactivity (B) | +0.194 | +2.58 | 0.0113 |
| timing (A) vs A + B | -0.034 | -1.50 | 0.1378 |
| reactivity (B) vs A + B | -0.228 | -3.42 | 0.0009 |

## 4. Measurement agreement: human-timed vs ClinicalWhisper

| parameter | n | ICC(2,1) |
|---|---|---|
| timing_mu | 150 | 0.800 |
| timing_sigma | 150 | 0.129 |
| timing_tau | 150 | 0.909 |
| pause_log_mean | 150 | 0.856 |
| pause_log_sd | 150 | 0.602 |
| pause_rate_pw | 150 | 0.680 |
| speech_rate_wps | 150 | 0.998 |
| filler_rate | 150 | 1.000 |
| beta_words | 150 | 1.000 |
| beta_speech_rate_wps | 150 | 0.796 |
| beta_log_latency | 150 | 0.055 |
| beta_filler_rate | 150 | 1.000 |
