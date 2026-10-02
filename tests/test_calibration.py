"""Calibration tools: AUC, an indeterminate band, ordinal cut-offs."""

import numpy as np

import calibration as c


def test_auc():
    assert c.auc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0
    assert c.auc([1, 1, 1, 1], [0, 1, 0, 1]) == 0.5
    assert c.auc([1, 2], [1, 1]) is None


def test_band_with_budget_beats_a_single_threshold_on_decided_cases():
    rng = np.random.default_rng(0)
    y = rng.random(2000) < 0.3
    s = rng.normal(y * 1.0, 1.0)
    one = c.fit_band(s, y, budget=0.0)
    two = c.fit_band(s, y, budget=0.3)
    m1, m2 = c.evaluate(one, s, y), c.evaluate(two, s, y)
    assert m1["indeterminate"] == 0.0 and m2["indeterminate"] <= 0.31
    assert min(m2["sensitivity"], m2["specificity"]) > min(m1["sensitivity"], m1["specificity"])


def test_ordinal_cutoffs_recover_levels():
    rng = np.random.default_rng(1)
    levels = rng.integers(0, 3, 3000)
    s = levels + rng.normal(0, 0.3, 3000)
    cuts = c.fit_ordinal(s, levels)
    assert len(cuts) == 2 and 0.2 < cuts[0] < 0.8 and 1.2 < cuts[1] < 1.8
