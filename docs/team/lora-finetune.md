# Fine-tuning MOSS on meeting audio (team task)

Goal: lower word and speaker errors on multi-speaker recordings by adapting
the transcription model (MOSS-Transcribe-Diarize) to meeting audio with small
LoRA adapters, without retraining the whole model.

Why: the best published AMI single-distant-mic systems (DM-ASR, 2026: cpWER
21.3%) were trained on 1,600–2,900 hours of meeting audio. ClinicalWhisper's
untuned MOSS is around 24% cpWER on our (easier) 5-meeting setup. Matching the
best numbers without training on meeting data is unlikely.

## Ground rules

- **Public or self-recorded consented audio only.** Never DAIC-WOZ, Yale, or any
  IRB study data (UP-26-00343, UP-25-00970).
- **Check every licence before downloading** and record it in the table below.
- **Tune on dev, report test once.** Nothing is chosen by looking at test
  scores (our speaker-linking thresholds were once tuned on test meetings;
  don't repeat that).
- **It has to run on a Mac afterwards.** The adapter is merged and converted to
  MLX; the model must still fit an M1 with 16 GB.

## The model

From its `config.json`: a Whisper-style audio encoder (24 layers, d_model 1024,
80 mel bins) feeding a Qwen3 decoder (28 layers, hidden 1024, about 0.6 B
parameters). The decoder writes speaker-tagged, timestamped text, so training
targets are that same text format (see `moss_diarizer.py` for how output is
parsed).

Start with LoRA on the decoder's attention and MLP projections only, encoder
frozen. Rank 16, alpha 32, dropout 0.05, learning rate 1e-4 with cosine decay,
300 s windows to match inference (`moss_windowed.py`).

## Data (fill in before downloading)

| Corpus | Hours | Licence | Use |
|---|---|---|---|
| AMI, train split (SDM and IHM) | ~70 | CC BY 4.0 | train |
| AMI, dev split | ~9 | CC BY 4.0 | tuning |
| ICSI meetings | ~70 | check | train |
| NOTSOFAR-1 | check | check | train |
| CHiME-6 / CHiME-8 | check | data licence agreement; check terms | train |

## Steps and owners

1. **Data prep (1 person):** convert each corpus to 300 s windows with target
   text in MOSS's output format. Script lives in `train/prepare_<corpus>.py`.
   Unit-test the format against MOSS's own parser.
2. **Training (1–2 people, CARC GPU):** PEFT LoRA via Transformers with
   `trust_remote_code=True`. Log every run's config, seed and data hash
   (see the lab's reproducibility rules).
3. **Export:** merge the adapter, convert to MLX, and load it through
   `moss_mlx.py`. Check the merged model produces identical text in PyTorch and
   MLX on 5 dev windows.
4. **Evaluation:** full AMI test set, original single-distant-mic audio,
   MeetEval cpWER and tcpWER, DER with no collar (`score.py --collar 0`).
   Compare against untuned MOSS and against Whisper large-v3 in the same harness.

## When it counts as a win

At least 2 points lower cpWER than untuned MOSS on the AMI test set, no
worse on the lab's own consented interview recordings, and no slower than
1.2× today's transcription time on an M2.
