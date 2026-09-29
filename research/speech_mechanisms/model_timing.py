"""Model A — speech timing: how fast and how steadily a person responds.

Per participant, from their prompt-response pairs:

* **Response latency ~ ex-Gaussian(mu, sigma, tau)**, fit by maximum
  likelihood. The ex-Gaussian is the standard decomposition of response times:
  ``mu``/``sigma`` describe the typical, Gaussian part (initiation speed and its
  spread); ``tau`` the exponential slow tail (occasional long delays). Motor or
  processing slowing is predicted to raise ``mu``; lapses of attention or
  effort to raise ``tau``.
* **Within-answer pauses ~ log-normal**, summarised by the mean and SD of
  log pause duration, plus pauses per 100 words. Per 100 words, not per
  minute: a per-minute rate rises with how much a person says, and on
  synthetic data it tracked the elaboration (anhedonia) item instead of the
  motor one.
* **Speech rate** (words per second of speech) and **filler rate** (per 100 words).

At least :data:`MIN_LATENCIES` valid latencies are required for the
ex-Gaussian; below that its parameters are reported as missing rather than
estimated from too little data.
"""

from __future__ import annotations

import math
import statistics

import numpy as np
from scipy import stats

MIN_LATENCIES = 15


def fit_exgauss(latencies: list[float]) -> dict[str, float]:
    """Maximum-likelihood ex-Gaussian fit: ``{"mu", "sigma", "tau"}``."""
    x = np.asarray(latencies, dtype=float)
    # Method-of-moments start (Lacouture & Cousineau, 2008), then scipy MLE.
    m, s = x.mean(), x.std(ddof=1)
    skew = max(float(stats.skew(x)), 0.05)
    tau0 = s * (skew / 2) ** (1 / 3)
    sigma0 = math.sqrt(max(s ** 2 - tau0 ** 2, (0.1 * s) ** 2))
    k, loc, scale = stats.exponnorm.fit(x, tau0 / sigma0, loc=m - tau0, scale=sigma0)
    return {"mu": float(loc), "sigma": float(scale), "tau": float(k * scale)}


def participant_timing(pairs: list[dict]) -> dict:
    latencies = [p["latency_s"] for p in pairs if p["latency_valid"]]
    pauses = [d for p in pairs for d in p["pauses_s"]]
    speech = sum(p["speech_s"] for p in pairs)
    words = sum(p["words"] for p in pairs)
    fillers = sum(p["fillers"] for p in pairs)

    params = {"n_latencies": len(latencies), "timing_mu": None, "timing_sigma": None,
              "timing_tau": None}
    if len(latencies) >= MIN_LATENCIES:
        fit = fit_exgauss(latencies)
        params.update(timing_mu=fit["mu"], timing_sigma=fit["sigma"], timing_tau=fit["tau"])
    logs = [math.log(d) for d in pauses]
    params.update({
        "pause_log_mean": statistics.fmean(logs) if logs else None,
        "pause_log_sd": statistics.stdev(logs) if len(logs) > 1 else None,
        "pause_rate_pw": 100 * len(pauses) / words if words else None,
        "speech_rate_wps": words / speech if speech > 0 else None,
        "filler_rate": 100 * fillers / words if words else None,
        # Reported so count-based measures can be read against answer length.
        "mean_response_words": words / len(pairs) if pairs else None,
    })
    return params


TIMING_PARAMS = ["timing_mu", "timing_sigma", "timing_tau", "pause_log_mean",
                 "pause_log_sd", "pause_rate_pw", "speech_rate_wps", "filler_rate"]
# Reported alongside, but never a Model A predictor: answer length measures
# elaboration, which is what the reward mechanism changes. Including it let the
# timing model predict the anhedonia item on synthetic data.
CONTEXT_PARAMS = ["mean_response_words"]
