# Masking recall

## Against other tools (English, replication set)

`held_out_2.jsonl`: 300 sentences written on Oct 2 2026 after every rule and
fix, before any tool was run on it. Same scorer for every tool.

| masker | recall | over-masking | organisations | phone numbers | names that are ordinary words |
|---|---|---|---|---|---|
| Microsoft Presidio 2.2.364 (spaCy en_core_web_lg, defaults) | 74.3% | 2.7% | 20% | 24% | 67% |
| OpenMED alone (threshold 0.5) | 89.0% | 3.3% | 48% | 32% | 100% |
| **ClinicalWhisper 5.3 (OpenMED + safety net)** | **98.0%** | 3.3% | 88% | 96% | 100% |

Presidio also caught fewer African (85%) and Vietnamese/Korean (78%) names than
Anglo ones (100%); OpenMED and ClinicalWhisper caught every name origin. On the
earlier held-out set: Presidio 81.5% (8.1% over), ClinicalWhisper 98.9% (3.3%).
Still missed by ClinicalWhisper: a brand-name workplace with no cue word
("at Kroger"), a street after "on" ("Ocean Drive"), one 7-digit number.
Presidio runs in its own environment (`presidio_baseline.py`); it is a
comparison, not part of the app.

## Earlier results

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

## Outside English (multilingual.py)

Short interview sentences in Chinese, Japanese, Korean, Hindi and Spanish, in
each language's own script, with invented names and real cities. Scored by
character, since Chinese and Japanese do not separate words with spaces. The
held-out set was written after the language rules, with new names and sentence
frames, and half its cities are smaller places left out of the rules' city
lists. Recall on the held-out set, names and places together:

| masker | zh | ja | ko | hi | es |
|---|---|---|---|---|---|
| Languages add-on (OpenMED multilingual v2, 8-bit), threshold 0.3 | 29% | 54% | 35% | 14% | 88% |
| + language rules (pii_rules_intl.py) | 64% | 81% | 75% | 83% | 88% |
| **+ second name check (name_sweep.py, scoring model)** | **80%** | **89%** | **79%** | **91%** | **100%** |

Replication set (HELD_OUT_2, written Oct 2 2026 after every fix below, new
names and frames, run once):

| masker | zh | ja | ko | hi | es |
|---|---|---|---|---|---|
| Languages add-on alone, threshold 0.3 | 54% | 43% | 32% | 0% | 92% |
| + language rules | 79% | 76% | 57% | 64% | 92% |
| **+ second name check** | **91%** | **81%** | **74%** | **78%** | **100%** |

Over-masking up to 8% (Spanish). Across both sets, the honest range outside
English is 74-100%; the app's warning quotes it.

Over-masking stays at or below 5% of other characters. English, for
comparison, is 98.9% on its own held-out set.

Honesty notes:
- The language rules reach 94-99% on the sentences they were written from
  (multilingual_tuning_rules.json); the held-out numbers above are the ones
  to quote.
- After seeing held-out results: the threshold went from 0.5 to 0.3 (0.3 is
  10-25 points better everywhere), two small rule fixes were made (Korean
  role words like 선생님 were masked as names; a Chinese surname touching a tag
  was skipped), and two bugs in the second check were fixed (text after
  Llama's end-of-turn marker hid the last name it listed; a name the masker
  had tagged only in part was left half-exposed). The replication set below
  was written after all of these.
- The second check is measured one sentence at a time. The app sends 15
  lines at a time, and a name repeated in a recording is masked everywhere
  once found, so real recall should be at or above these figures. An earlier
  15-line run gave 92-100%, but this set reuses a small pool of names, which
  inflates that figure.
- What is still missed is mostly Chinese, Japanese and Korean person names
  with no cue around them ("上个星期朱琳来看我了"): neither OpenMED nor
  Llama-3-8B reliably recognises them in a short sentence. The app therefore
  warns on every non-English transcript to read it before sharing.
- The language is detected from the transcript; mixed-language speech uses
  the main language's rules plus the English ones.

## Real transcripts

The synthetic sets are a ceiling. GOLD_SET.md is the protocol for a gold set
of consented real recordings on iLab (two annotators, agreement, adjudication,
whole-recording scoring with `--whole`, numbers only leave the machine).
