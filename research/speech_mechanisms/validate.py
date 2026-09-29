"""The three validation tests for the timing-vs-reactivity question.

1. :func:`discriminant` — which symptom does each parameter track?
   ``parameter ~ item1 + item2 + item8`` (all z-scored), so each coefficient is
   that item's association *controlling for the other two*, with percentile
   bootstrap CIs. Pre-registered expectations: timing parameters load on item 8
   (psychomotor), reactivity betas on item 1 (anhedonia), and neither on item 2
   (mood) once the others are in the model.

2. :func:`compare_models` — which mechanism predicts symptoms out of sample?
   Ridge regression with the penalty tuned in an inner CV loop (nested CV),
   repeated k-fold, scored by Spearman rho between predicted and observed.
   Models are compared on identical splits with the Nadeau & Bengio (2003)
   corrected resampled t-test, which accounts for overlapping training sets
   (the naive paired t-test across folds is badly anticonservative).

3. :func:`agreement` — is the automatic measurement the same as the human one?
   ICC(2,1), two-way random effects, absolute agreement (Shrout & Fleiss,
   1979), between parameters from human-timed transcripts and from
   ClinicalWhisper's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

ITEMS = ["item1", "item2", "item8"]


def _z(x: pd.Series) -> pd.Series:
    return (x - x.mean()) / x.std(ddof=1)


def discriminant(params: pd.DataFrame, scores: pd.DataFrame, n_boot: int = 2000,
                 seed: int = 0) -> pd.DataFrame:
    """Standardized coefficients of items 1, 2, 8 for each parameter.

    If ``scores`` has a ``sex`` column it enters as a covariate (F0 and speech
    rate differ by sex); its coefficient is not reported.
    """
    rng = np.random.default_rng(seed)
    covariates = [c for c in ("sex",) if c in scores and scores[c].nunique() > 1]
    data = params.join(scores[ITEMS + covariates], how="inner")
    rows = []
    for param in params.columns:
        d = data[[param] + ITEMS + covariates].dropna()
        if len(d) < 10 or d[param].std() == 0:
            continue
        y = _z(d[param]).to_numpy()
        X = np.column_stack([np.ones(len(d))] + [_z(d[i]).to_numpy() for i in ITEMS]
                            + [d[c].to_numpy(dtype=float) for c in covariates])
        beta = np.linalg.lstsq(X, y, rcond=None)[0][1:1 + len(ITEMS)]
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(d), len(d))
            Xb, yb = X[idx], y[idx]
            if np.linalg.matrix_rank(Xb) < X.shape[1]:
                continue
            boots.append(np.linalg.lstsq(Xb, yb, rcond=None)[0][1:1 + len(ITEMS)])
        lo, hi = np.percentile(np.array(boots), [2.5, 97.5], axis=0)
        row = {"parameter": param, "n": len(d)}
        for j, item in enumerate(ITEMS):
            row[f"{item}_beta"] = beta[j]
            row[f"{item}_ci"] = (lo[j], hi[j])
        row["strongest"] = ITEMS[int(np.argmax(np.abs(beta)))]
        rows.append(row)
    return pd.DataFrame(rows)


def _nested_ridge_predictions(X: np.ndarray, y: np.ndarray, train: np.ndarray,
                              test: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # RidgeCV tunes the penalty by efficient leave-one-out on the training fold
    # only — the inner loop of the nested CV. Nothing from the test fold is seen.
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12)))
    model.fit(X[train], y[train])
    return model.predict(X[test])


def compare_models(feature_sets: dict[str, pd.DataFrame], target: pd.Series,
                   k: int = 5, repeats: int = 20, seed: int = 0) -> dict:
    """Out-of-sample Spearman rho per feature set, and pairwise comparisons."""
    from sklearn.model_selection import KFold

    common = target.dropna().index
    for frame in feature_sets.values():
        common = common.intersection(frame.dropna().index)
    y = target.loc[common].to_numpy(dtype=float)
    n = len(common)
    scores: dict[str, list[float]] = {name: [] for name in feature_sets}
    for r in range(repeats):
        for train, test in KFold(k, shuffle=True, random_state=seed + r).split(np.arange(n)):
            for name, frame in feature_sets.items():
                X = frame.loc[common].to_numpy(dtype=float)
                pred = _nested_ridge_predictions(X, y, train, test)
                rho = stats.spearmanr(pred, y[test]).statistic if np.std(pred) > 0 else 0.0
                scores[name].append(0.0 if np.isnan(rho) else rho)

    n_test = n // k
    correction = 1 / (k * repeats) + n_test / (n - n_test)
    names = list(feature_sets)
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            diff = np.array(scores[a]) - np.array(scores[b])
            var = diff.var(ddof=1)
            t = diff.mean() / np.sqrt(correction * var) if var > 0 else 0.0
            p = 2 * stats.t.sf(abs(t), df=len(diff) - 1)
            pairs.append({"a": a, "b": b, "mean_diff": diff.mean(), "t": t, "p": p})
    return {
        "n": n,
        "rho": {name: (float(np.mean(v)), float(np.percentile(v, 2.5)),
                       float(np.percentile(v, 97.5))) for name, v in scores.items()},
        "comparisons": pd.DataFrame(pairs),
    }


def icc_2_1(a: np.ndarray, b: np.ndarray) -> float:
    """ICC(2,1), two raters, absolute agreement (Shrout & Fleiss, 1979)."""
    x = np.column_stack([a, b]).astype(float)
    n, k = x.shape
    grand = x.mean()
    ms_r = k * ((x.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_c = n * ((x.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    ss_e = ((x - x.mean(axis=1, keepdims=True) - x.mean(axis=0, keepdims=True) + grand) ** 2).sum()
    ms_e = ss_e / ((n - 1) * (k - 1))
    return float((ms_r - ms_e) / (ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n))


def agreement(human: pd.DataFrame, automatic: pd.DataFrame) -> pd.DataFrame:
    """Per-parameter ICC(2,1) between human-timed and automatic transcripts."""
    rows = []
    for param in human.columns.intersection(automatic.columns):
        d = pd.concat([human[param], automatic[param]], axis=1, join="inner").dropna()
        if len(d) >= 5:
            rows.append({"parameter": param, "n": len(d),
                         "icc_2_1": icc_2_1(d.iloc[:, 0].to_numpy(), d.iloc[:, 1].to_numpy())})
    return pd.DataFrame(rows)
