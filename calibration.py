"""Turning a continuous score into decisions, with room to say "can't tell".

Lessons taken from Kintsugi's released tuning code: report bands of a real
instrument (e.g. PHQ-9 >= 10), choose the cut-offs on a validation set and
report a test set once, and allow an *indeterminate* zone between two
thresholds so borderline cases are not forced into a call. These are small,
dependency-free versions for ClinicalWhisper's own scores once rated data
exists (PHQ- or SCID-rated recordings), and for checking Kintsugi's model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def auc(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """Area under the ROC curve (Mann-Whitney, ties averaged), or None with one class only."""
    from scipy.stats import rankdata

    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if not n_pos or not n_neg:
        return None
    ranks = rankdata(s)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@dataclass(frozen=True)
class Band:
    """Below ``low``: negative. Above ``high``: positive. Between: indeterminate."""

    low: float
    high: float

    def apply(self, scores: np.ndarray) -> np.ndarray:
        """1 positive, 0 negative, -1 indeterminate."""
        s = np.asarray(scores, float)
        return np.where(s > self.high, 1, np.where(s <= self.low, 0, -1))


def evaluate(band: Band, scores: np.ndarray, labels: np.ndarray) -> dict:
    """Sensitivity and specificity on decided cases, and the share left undecided."""
    call = band.apply(scores)
    y = np.asarray(labels, bool)
    decided = call >= 0
    tp = int(((call == 1) & y).sum())
    fn = int(((call == 0) & y).sum())
    tn = int(((call == 0) & ~y).sum())
    fp = int(((call == 1) & ~y).sum())
    return {
        "n": int(len(y)),
        "sensitivity": round(tp / (tp + fn), 3) if tp + fn else None,
        "specificity": round(tn / (tn + fp), 3) if tn + fp else None,
        "indeterminate": round(1 - decided.mean(), 3),
    }


def fit_band(scores: np.ndarray, labels: np.ndarray, budget: float = 0.2,
             grid: int = 200) -> Band:
    """Two thresholds leaving at most ``budget`` undecided, balancing sensitivity and specificity.

    Maximises the smaller of the two (so neither is sacrificed), over a grid
    of score quantiles. ``budget=0`` gives one threshold (low == high).
    """
    s = np.asarray(scores, float)
    qs = np.quantile(s, np.linspace(0, 1, grid + 1))
    best, best_val = Band(float(np.median(s)), float(np.median(s))), -1.0
    for i, low in enumerate(qs):
        for high in qs[i:]:
            if np.mean((s > low) & (s <= high)) > budget + 1e-9:
                break
            m = evaluate(Band(float(low), float(high)), s, labels)
            if m["sensitivity"] is None or m["specificity"] is None:
                continue
            val = min(m["sensitivity"], m["specificity"])
            if val > best_val:
                best, best_val = Band(float(low), float(high)), val
    return best


def fit_ordinal(scores: np.ndarray, levels: np.ndarray) -> list[float]:
    """Cut-offs mapping a score onto ordered levels (0..K), maximising macro-F1.

    Greedy, one cut-off at a time from the bottom, each chosen from score
    quantiles; enough for a handful of severity bands.
    """
    s, y = np.asarray(scores, float), np.asarray(levels, int)
    k = int(y.max())
    cuts: list[float] = []
    for level in range(1, k + 1):
        lo = cuts[-1] if cuts else -np.inf
        cands = np.quantile(s[s > lo], np.linspace(0.02, 0.98, 97)) if np.any(s > lo) else []
        best, best_f1 = None, -1.0
        for c in cands:
            pred = np.searchsorted(np.array(cuts + [c]), s, side="left")
            f1 = _macro_f1(np.minimum(pred, level), np.minimum(y, level))
            if f1 > best_f1:
                best, best_f1 = float(c), f1
        cuts.append(best if best is not None else lo)
    return cuts


def _macro_f1(pred: np.ndarray, true: np.ndarray) -> float:
    f1s = []
    for c in np.unique(true):
        tp = np.sum((pred == c) & (true == c))
        fp = np.sum((pred == c) & (true != c))
        fn = np.sum((pred != c) & (true == c))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return float(np.mean(f1s))
