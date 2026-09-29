"""Model B — reward reactivity: how much a person's speech lifts for positive prompts.

For each response feature, a mixed model with a random slope per participant:

    feature ~ positive + (1 + positive | participant)

``positive`` is 1 for positive-valence prompts and 0 for neutral or negative
ones. Each participant's slope (fixed effect plus their random-effect
prediction) is their **vocal reward sensitivity** ``beta`` for that feature —
how much more they elaborate, or how much faster they respond, when the
question invites something good. Anhedonia is predicted to *shrink* beta.

Features: ``words`` (elaboration), ``speech_rate_wps``, ``log_latency``,
``filler_rate``. Prosodic features are used too when the pairs carry them
(``f0_mean_st``, ``loudness_mean``), which requires the audio.

The inferential test of the same idea, with symptoms in the model, is
:func:`reactivity_test` (``feature ~ positive * item1 + item2 + (1|participant)``).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

REACTIVITY_FEATURES = ["words", "speech_rate_wps", "log_latency", "filler_rate",
                       "f0_mean_st", "loudness_mean"]


def pairs_frame(pairs_by_participant: dict[str, list[dict]]) -> pd.DataFrame:
    rows = []
    for pid, pairs in pairs_by_participant.items():
        for p in pairs:
            if p["valence"] is None:
                continue
            rows.append({
                "participant": pid,
                "positive": 1.0 if p["valence"] == "positive" else 0.0,
                "words": p["words"],
                "speech_rate_wps": p["speech_rate_wps"],
                "log_latency": math.log(p["latency_s"] + 0.05) if p["latency_valid"] else None,
                "filler_rate": 100 * p["fillers"] / p["words"] if p["words"] else None,
                "f0_mean_st": p.get("f0_mean_st"),
                "loudness_mean": p.get("loudness_mean"),
            })
    return pd.DataFrame(rows)


def participant_reactivity(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-participant beta for every available feature (one column each)."""
    import statsmodels.formula.api as smf

    out = pd.DataFrame(index=sorted(frame["participant"].unique()))
    for feature in REACTIVITY_FEATURES:
        if feature not in frame or frame[feature].isna().all():
            continue
        data = frame.dropna(subset=[feature])
        # Within-person centering: subtract each participant's own mean, so the
        # slope can only reflect how a person *changes* with valence. Without
        # it, correlated random effects let each slope borrow from that
        # person's overall level — on synthetic data where reactivity tracked
        # item 1 only, uncentered betas predicted the motor item (item 8) as
        # well as the timing model did.
        within = data[feature] - data.groupby("participant")[feature].transform("mean")
        sd = within.std()
        if not sd or np.isnan(sd):
            continue
        data = data.assign(y=within / sd)
        fit = smf.mixedlm("y ~ positive", data, groups=data["participant"],
                          re_formula="~positive").fit(reml=True)
        slope = fit.params["positive"]
        out[f"beta_{feature}"] = [
            slope + fit.random_effects[pid].get("positive", 0.0) if pid in fit.random_effects else np.nan
            for pid in out.index
        ]
    return out


def reactivity_test(frame: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Pre-registered H2: ``positive:item1`` per feature, controlling item 2.

    ``scores`` is indexed by participant with columns ``item1`` and ``item2``.
    """
    import statsmodels.formula.api as smf

    extra = ["sex"] if "sex" in scores and scores["sex"].nunique() > 1 else []
    data = frame.join(scores[["item1", "item2"] + extra], on="participant", how="inner")
    covariate = " + sex" if extra else ""
    rows = []
    for feature in REACTIVITY_FEATURES:
        if feature not in data or data[feature].isna().all():
            continue
        d = data.dropna(subset=[feature])
        fit = smf.mixedlm(f"{feature} ~ positive * item1 + item2{covariate}", d,
                          groups=d["participant"]).fit(reml=True)
        low, high = fit.conf_int().loc["positive:item1"]
        rows.append({"feature": feature, "interaction": fit.params["positive:item1"],
                     "ci_low": low, "ci_high": high, "p": fit.pvalues["positive:item1"]})
    return pd.DataFrame(rows)
