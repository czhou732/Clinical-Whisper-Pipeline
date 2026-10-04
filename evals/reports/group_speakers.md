# Speaker labels in group discussions: late joiners and crosstalk

AMI test meetings (single distant microphone, audio rebuilt from annotated utterances).
Late-joiner versions: one participant's speech removed before 7 min, so they join
mid-window (late_joiner_prepare.py). MOSS transcripts made once with the current
pipeline (300 s windows); speaker_check.py and overlap_detector.py run on top.

Chosen on EN2002a-c: merge labels at voice similarity 0.6, move a segment when
another label matches it better by 0.1, split below 0.4. Overlap threshold 0.3.

## Speaker error (DER confusion, 0.25 s collar) and word error by speaker (cpWER)

| meeting | set | confusion before | after | cpWER before | after | labels before / after (with 30 s+ talk) / true people |
|---|---|---|---|---|---|---|
| EN2002a late joiner | tuning | 0.049 | 0.029 | 0.336 | 0.314 | 8 / 8 (4) / 4 |
| EN2002a | tuning | 0.039 | 0.022 | 0.305 | 0.285 | 7 / 7 (4) / 4 |
| EN2002b late joiner | tuning | 0.038 | 0.021 | 0.286 | 0.267 | 10 / 9 (4) / 4 |
| EN2002b | tuning | 0.032 | 0.019 | 0.272 | 0.255 | 9 / 9 (4) / 4 |
| EN2002c late joiner | tuning | 0.005 | 0.005 | 0.210 | 0.211 | 8 / 8 (3) / 3 |
| EN2002c | tuning | 0.005 | 0.003 | 0.205 | 0.205 | 8 / 8 (3) / 3 |
| ES2004a late joiner | test | 0.131 | 0.025 | 0.421 | 0.240 | 9 / 8 (4) / 4 |
| ES2004a | test | 0.015 | 0.015 | 0.217 | 0.217 | 6 / 6 (4) / 4 |
| ES2004b late joiner | test | 0.121 | 0.010 | 0.357 | 0.164 | 9 / 9 (4) / 4 |
| ES2004b | test | 0.068 | 0.012 | 0.248 | 0.159 | 7 / 7 (4) / 4 |

## Crosstalk detection (frames where two or more people talk, AMI annotation)

| meetings | precision | recall | F1 |
|---|---|---|---|
| EN2002a-c (tuning) | 0.85 | 0.74 | 0.79 |
| ES2004a-b (test) | 0.81 | 0.59 | 0.68 |

Labels with under 30 s of talk are mostly a backchannel ("yeah") given its own
label; the app flags labels under 10 s as possibly one person split in two.

Read with care: rebuilt audio has silence where nobody was annotated, which is
easier than a raw recording; five meetings, one joiner each.

## Known limit: a joiner who says very little

On a real 77-min focus group (UP-25-00970, checked locally, nothing shared), every
regular speaker kept one label, but a late joiner with under 20 s of clean speech
(most of it over others) stayed under a participant's label: too little voice to
split on. A per-segment "voice doesn't match" flag was tried on AMI and dropped:
13-30% precision at any threshold, so it would mostly raise false alarms. The
fix is moving a single turn to another speaker by hand (not yet in the app).
