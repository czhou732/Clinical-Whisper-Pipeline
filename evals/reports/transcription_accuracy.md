# Transcription accuracy against human ground truth (AMI)

Meetings: EN2002a, EN2002b, EN2002c, ES2004a, ES2004b (AMI test split, single distant microphone). Reference: AMI human transcripts and speaker labels. Audio rebuilt from the annotated utterances, so unannotated stretches are silence — slightly easier than a raw recording.

Pooled over meetings, weighted by reference words (error rates) or speech time (DER, approximated by words). Lower is better except filler recall/precision.

| configuration | cpWER | WER verbatim | WER clean | filler recall | filler precision | DER | missed | false alarm | confusion | speakers (hyp/ref) | x real time |
|---|---|---|---|---|---|---|---|---|---|---|---|
| single pass (full context) | 0.379 | 0.384 | 0.377 | 0.500 | 0.668 | 0.298 | 0.266 | 0.002 | 0.030 | 4/4, 5/4, 7/3, 4/4, 4/4 | 4.2x |
| windows 300 s / 30 s overlap | 0.249 | 0.258 | 0.251 | 0.611 | 0.651 | 0.169 | 0.136 | 0.003 | 0.031 | 7/4, 9/4, 8/3, 6/4, 7/4 | 20.9x |
| windows 240 s / 30 s overlap | 0.307 | 0.272 | 0.266 | 0.599 | 0.650 | 0.211 | 0.146 | 0.009 | 0.056 | 5/4, 8/4, 9/3, 8/4, 10/4 | 26.2x |
| windows 180 s / 30 s overlap | 0.304 | 0.261 | 0.254 | 0.602 | 0.656 | 0.198 | 0.138 | 0.003 | 0.057 | 8/4, 11/4, 5/3, 8/4, 14/4 | 29.7x |
| windows 300 s / 15 s overlap | 0.289 | 0.261 | 0.254 | 0.619 | 0.665 | 0.192 | 0.136 | 0.003 | 0.054 | 5/4, 7/4, 6/3, 6/4, 9/4 | 22.6x |
| windows 300 s + known speaker count | 0.243 | 0.258 | 0.251 | 0.611 | 0.651 | 0.167 | 0.136 | 0.003 | 0.028 | 4/4, 4/4, 3/3, 4/4, 4/4 | 18.9x |

## Notes

- Speaker-linking thresholds (voice model): link 0.5, merge 0.65, within-window split 0.75. Tuned with `tune_linking.py` on EN2002a-c (one group of four people); ES2004a-b (different people) were held out and scored only at the chosen setting. Held-out cpWER at the chosen setting was within 0.2 points of the best any setting achieved there.
- Word error with windows is within half a point of the single pass or better on every meeting; the single pass degrades with length (it collapsed on the 48-min EN2002c). Remaining windowed error is mostly speaker attribution in short multi-party meetings.
- Known speaker count (`moss.num_speakers`, batch `--speakers`): extra labels are folded into the given number by voice. Every meeting then has the right count; cpWER and DER improve slightly because the extra labels carried little speech. Word error is unchanged: this fixes who, not what.
- 'missed' is high in every configuration because AMI has much overlapping speech and MOSS assigns one speaker at a time.
- Configurations and meetings evaluated on one M2 Max; x real time depends on machine load.

Per-meeting results: `results.json` in the run directory (not committed).
