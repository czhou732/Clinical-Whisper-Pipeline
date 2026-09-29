"""Synthetic interviews with a *known* mechanism, for validating the analysis.

Each synthetic participant has PHQ items 1, 2 and 8 (correlated through a
shared severity factor, as in real data) and two latent traits:

* **motor slowing** driven by item 8 — raises the ex-Gaussian latency mean and
  tail, slows speech, lengthens pauses;
* **reward sensitivity** driven by item 1 (negatively) — sets how much more a
  person elaborates when a prompt is positive.

Item 2 drives neither. An analysis that cannot recover exactly this mapping
from the generated transcripts should not be trusted on real ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.speech_mechanisms.transcripts import INTERVIEWER, PARTICIPANT

PROMPTS = {
    "positive": ["when was the last time you felt really happy",
                 "what are you most proud of in your life",
                 "tell me about a good memory"],
    "neutral": ["where are you from originally", "what do you do for work",
                "how would you describe your living situation"],
    "negative": ["when was the last time you argued with someone",
                 "is there anything you regret", "what has been difficult lately"],
}
VALENCE_MAP = {p: v for v, ps in PROMPTS.items() for p in ps}


def make_cohort(n: int = 150, prompts_per_valence: int = 15, seed: int = 0,
                motor_effect: float = 1.0, reward_effect: float = 1.0):
    """Return ``(transcripts, scores)`` like the DAIC loaders produce."""
    rng = np.random.default_rng(seed)
    severity = rng.normal(size=n)

    def item():
        return np.clip(np.round(1.2 + 0.8 * severity + rng.normal(0, 0.9, n)), 0, 3)

    item1, item2, item8 = item(), item(), item()
    z = lambda v: (v - v.mean()) / v.std()  # noqa: E731
    motor = motor_effect * 0.8 * z(item8) + 0.6 * rng.normal(size=n)
    reward = -reward_effect * 0.8 * z(item1) + 0.6 * rng.normal(size=n)

    transcripts, rows = {}, []
    for i in range(n):
        pid = f"p{i:03d}"
        rows.append({"participant": pid, "item1": item1[i], "item2": item2[i],
                     "item8": item8[i], "total": item1[i] + item2[i] + item8[i]})
        order = [v for v in PROMPTS for _ in range(prompts_per_valence)]
        rng.shuffle(order)
        t, segs = 0.0, []
        for valence in order:
            prompt = PROMPTS[valence][rng.integers(len(PROMPTS[valence]))]
            segs.append({"role": INTERVIEWER, "start": t, "end": t + 2.5, "text": prompt + "?"})
            t += 2.5
            mu = 0.6 + 0.12 * motor[i]
            tau = max(0.08, 0.45 + 0.2 * motor[i])
            t += max(0.02, rng.normal(mu, 0.15) + rng.exponential(tau))
            lift = max(0.0, 12 + 8 * reward[i]) if valence == "positive" else 0.0
            words = max(3, int(rng.normal(22 + lift, 5)))
            rate = max(1.0, 2.6 - 0.25 * motor[i] + rng.normal(0, 0.15))
            chunks = max(1, words // 8)
            per_chunk = max(1, words // chunks)
            for c in range(chunks):
                dur = per_chunk / rate
                filler = "um " if rng.random() < 0.15 + 0.05 * max(motor[i], 0) else ""
                segs.append({"role": PARTICIPANT, "start": t, "end": t + dur,
                             "text": filler + " ".join(["word"] * per_chunk)})
                t += dur
                if c < chunks - 1:
                    t += float(np.exp(rng.normal(np.log(0.6) + 0.2 * motor[i], 0.5)))
            t += 1.0
        transcripts[pid] = segs
    return transcripts, pd.DataFrame(rows).set_index("participant")
