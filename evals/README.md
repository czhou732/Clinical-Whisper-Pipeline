# ClinicalWhisper Evaluations

Reproducible evaluation of the **LLM clinical scorer** (`llm_clinical_scorer.py`) — the six
0–10 dimensions that appear in the v5 results and in the bioRxiv preprint.

```bash
# Model-free: dataset schema + assertion logic (<1s, runs anywhere, CI path)
python evals/run_clinical_evals.py --self-test

# Full evaluation against the local LLM (Apple Silicon + mlx_lm)
python evals/run_clinical_evals.py --report evals/reports/latest.json
```

## What is gated

The pre-commit hook runs this eval and blocks the commit if fewer than **85%** of assertions
hold. That threshold is inherited from the v3 sentiment gate this replaces — the target
changed, not the bar.

Before v5.0 the gate scored `sentiment_analyzer.py`
(`cardiffnlp/twitter-roberta-base-sentiment-latest`). The v5 overhaul removed sentiment analysis
from the pipeline: `inference_pipeline.py` no longer imports it and `batch_processor.py` emits
the six LLM clinical scores instead of `sentiment_score`/`sentiment_label`. That gate sat at
72.7% and blocked every commit, failing hardest on clinical masking (20%) and anhedonia — the
cases a Twitter-trained valence classifier structurally cannot read. Patients saying
"I'm fine, everything's okay" scored 8–10/10 positive.

## The golden dataset

`clinical_golden_dataset.jsonl`, one JSON object per case:

| field | meaning |
|---|---|
| `id`, `category`, `description` | identity and clinical framing |
| `segments` | diarized turns (`start`, `end`, `speaker`, `text`) |
| `acoustics` | `overall` + `by_speaker` eGeMAPS features (`pitch_cv`, `vta`, `jitter`, …) |
| `expect` | inclusive `[lo, hi]` bands per dimension |
| `relations` | ordinal constraints, e.g. `["elaboration_negative", ">", "elaboration_positive"]` |
| `rationale` | why those bands are clinically defensible |

Inputs are built through the **real pipeline helpers** — `transcript_formatter.process_segments`
and `acoustic_context.build_acoustic_prompt_context` — so the eval exercises the v5 path from
diarized segments onward, not hand-written prompt strings. Role classification, turn merging and
timestamp formatting are all in the loop.

### Band methodology

Three deliberate choices, each of which affects how the numbers should be read:

1. **Bands, not point targets.** An LLM's absolute calibration on a 0–10 scale is the noisy
   axis. Bands are set from the clinical rationale in each record *before* the model is run, and
   are not retuned to make the gate pass.
2. **Partial assertions.** A case asserts only the dimensions for which it provides clear
   evidence. `edge_01` (a four-turn stalled session) asserts engagement and elaboration but not
   `affect_flatness` — there is too little speech to judge it fairly.
3. **Ordinal relations carry the clinical contrasts.** The anhedonia signature is
   `elaboration_negative > elaboration_positive` despite lexically positive content ("my favorite
   movie", "love cooking"); the recovery control asserts the reverse. Relative ordering is more
   stable than absolute scores and is what actually distinguishes the categories.

### Categories

| category | what it tests |
|---|---|
| `clinical_masking` | "I'm fine" deflection — low elaboration in *both* valences |
| `anhedonia` | loss of pleasure described in lexically positive words |
| `engaged_control` | expressive, high-elaboration subjects that must **not** read as clinical |
| `hesitancy_psychomotor` | fillers and restarts vs. response latency (read from timestamps) |
| `flat_affect` | monosyllabic floor case |
| `edge_case` | degenerate input — must return valid JSON without inventing clinical detail |

The controls are load-bearing. Without them a scorer that pathologises everything would post a
high score; `ctrl_01` and `ctrl_02` are what make the number mean something.

## Failure handling

`score_transcript` returns `DEFAULT_SCORES` with `_meta.error` when generation fails, and sets
`_parse_error` on unparseable output. The runner treats both as hard case failures and counts
that case's assertions as failed — a dead model must not shrink the denominator into a pass by
returning a placid 5/10 across the board.

## Caching

Raw scorer output is cached in `evals/.cache/` (gitignored), keyed by
`sha256(model + full prompt)`. Routine commits are instant. Editing the prompt template in
`llm_clinical_scorer.py`, switching models, or changing a case invalidates the affected entries,
so the gate re-scores exactly when the thing it gates changes. `--no-cache` forces a full re-run.

Runs are deterministic: `call_local_lm` uses `mlx_lm.generate` with the default greedy sampler,
so a re-run on the same model reproduces the same scores.

## Environments

| environment | path |
|---|---|
| `.venv` (3.13, has `mlx_lm`) | full eval |
| `venv` (3.14, no `mlx_lm`) | self-test fallback |
| GitHub Actions (Ubuntu) | `--self-test` only — a hosted runner cannot host an 8B model |

The hook prefers `.venv`, falls back to `venv`, and passes `--fallback-self-test` so a machine
that physically cannot load the model is warned rather than permanently blocked. Run the full
eval on a machine with the model before relying on the gate.

## Reports

`--report PATH` writes a JSON artifact — model id, per-case scores, per-assertion outcomes,
timings — suitable for citation in the preprint's reproducibility section.
