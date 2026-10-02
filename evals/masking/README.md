# Masking recall

Labelled, shareable sentences with invented identifiers (make_set.py), scored by
evaluate.py through ClinicalWhisper's own masker. An identifier counts as caught
when all of it ends up inside tags; over-masking is ordinary words lost to tags.

| masker | tuning set (400 sentences, 600 ids) | held-out set (300 sentences, 475 ids) |
|---|---|---|
| 5.2: OpenMED, threshold 0.7 | 94.8% caught, 2.5% over | 90.5% caught, 1.7% over |
| OpenMED, threshold 0.5 | 96.3%, 2.7% | 95.8%, 3.3% |
| **5.3: OpenMED 0.5 + safety-net rules** | **98.8%, 2.7%** | **98.9%, 3.3%** |

Held-out first names: 95.3% -> 100%. Organisations: 48% -> 94%. Name recall was
the same for every name origin tested (Anglo, Hispanic, Chinese, Indian, Arabic,
African, Vietnamese/Korean, names that are ordinary words).

Honesty notes:
- The rules (pii_rules.py) were written while looking at the tuning set; the
  held-out set was written afterwards with different phrasing. One fix (spoken
  emails that OpenMED half-tags) was made after seeing held-out results, so the
  held-out email figure is no longer untouched; a fresh set should confirm it.
- Synthetic sentences are cleaner than real transcripts. The real miss that
  started this ("my sister Elena", OpenMED 0.61 in a long segment) is caught by
  the relationship rule.
- Still missed: brand-name workplaces with no cue word ("at Kroger"), and
  OpenMED masks some ordinary words ("lowercase" as sexuality).
