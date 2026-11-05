"""The mechanism analysis must recover a mechanism planted in synthetic data.

Required by the pre-registration: validate on synthetic data before any real
dataset is touched.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")
pytest.importorskip("sklearn")

from research.speech_mechanisms import analysis, turns  # noqa: E402
from research.speech_mechanisms.model_timing import fit_exgauss  # noqa: E402
from research.speech_mechanisms.synthetic import VALENCE_MAP, make_cohort  # noqa: E402
from research.speech_mechanisms.transcripts import load_daic, prompt_inventory  # noqa: E402
from research.speech_mechanisms.validate import compare_models, discriminant, icc_2_1  # noqa: E402


@pytest.fixture(scope="module")
def cohort():
    transcripts, scores = make_cohort(n=150, seed=3)
    timing, reactivity, frame = analysis.parameters(transcripts, VALENCE_MAP)
    return transcripts, scores, timing, reactivity, frame


def test_exgauss_recovers_known_parameters():
    rng = np.random.default_rng(0)
    x = rng.normal(0.6, 0.15, 4000) + rng.exponential(0.5, 4000)
    fit = fit_exgauss(list(x))
    assert fit["mu"] == pytest.approx(0.6, abs=0.05)
    assert fit["sigma"] == pytest.approx(0.15, abs=0.04)
    assert fit["tau"] == pytest.approx(0.5, abs=0.05)


def test_timing_parameters_track_psychomotor_not_anhedonia(cohort):
    _, scores, timing, _, _ = cohort
    table = discriminant(timing[["timing_tau", "speech_rate_wps", "pause_log_mean"]], scores,
                         n_boot=300).set_index("parameter")
    assert (table["strongest"] == "item8").all()


def test_reactivity_tracks_anhedonia_not_psychomotor(cohort):
    _, scores, _, reactivity, _ = cohort
    table = discriminant(reactivity[["beta_words"]], scores, n_boot=300).set_index("parameter")
    row = table.loc["beta_words"]
    assert row["strongest"] == "item1" and row["item1_beta"] < 0
    assert row["item1_ci"][1] < 0  # CI excludes zero


def test_model_comparison_picks_the_generating_mechanism(cohort):
    _, scores, timing, reactivity, _ = cohort
    sets = {"A": timing.dropna(axis=1, how="all"), "B": reactivity.dropna(axis=1, how="all")}
    on_item8 = compare_models(sets, scores["item8"], repeats=5)
    on_item1 = compare_models(sets, scores["item1"], repeats=5)
    assert on_item8["rho"]["A"][0] > on_item8["rho"]["B"][0]
    assert on_item1["rho"]["B"][0] > on_item1["rho"]["A"][0]


def test_null_cohort_finds_no_mechanism():
    transcripts, scores = make_cohort(n=150, seed=4, motor_effect=0.0, reward_effect=0.0)
    timing, reactivity, _ = analysis.parameters(transcripts, VALENCE_MAP)
    table = discriminant(timing[["timing_tau"]].join(reactivity[["beta_words"]]), scores,
                         n_boot=300).set_index("parameter")
    for param in ("timing_tau", "beta_words"):
        lo, hi = table.loc[param, "item8_ci"] if param == "timing_tau" else table.loc[param, "item1_ci"]
        assert lo < 0 < hi


def test_reactivity_test_detects_blunting(cohort):
    _, scores, _, _, frame = cohort
    from research.speech_mechanisms.model_reactivity import reactivity_test

    table = reactivity_test(frame, scores).set_index("feature")
    assert table.loc["words", "interaction"] < 0 and table.loc["words", "p"] < 0.01


def test_icc_agreement():
    rng = np.random.default_rng(1)
    truth = rng.normal(size=200)
    assert icc_2_1(truth, truth + rng.normal(0, 0.1, 200)) > 0.95
    assert icc_2_1(truth, rng.normal(size=200)) < 0.2


def test_daic_transcript_format(tmp_path):
    path = tmp_path / "300_TRANSCRIPT.csv"
    path.write_text(
        "start_time\tstop_time\tspeaker\tvalue\n"
        "36.588\t39.668\tEllie\twhen was the last time you felt really happy\n"
        "40.668\t42.100\tParticipant\tum last summer\n"
        "42.600\t44.000\tParticipant\twith my family\n"
        "45.000\t46.500\tEllie\twhere are you from originally\n"
        "47.900\t48.800\tParticipant\tlos angeles\n"
    )
    segs = load_daic(path)
    pairs = turns.pairs(segs, VALENCE_MAP)
    assert [p["valence"] for p in pairs] == ["positive", "neutral"]
    assert pairs[0]["latency_s"] == pytest.approx(1.0) and pairs[0]["pauses_s"] == [pytest.approx(0.5)]
    assert prompt_inventory({"300": segs})[0][1] == 1


def test_scores_loader_maps_daic_columns(tmp_path):
    path = tmp_path / "train_split.csv"
    pd.DataFrame({"Participant_ID": [300], "PHQ8_Score": [10], "PHQ8_NoInterest": [2],
                  "PHQ8_Depressed": [1], "PHQ8_Moving": [3]}).to_csv(path, index=False)
    s = analysis.load_scores(path)
    assert s.loc["300", "item1"] == 2 and s.loc["300", "item8"] == 3 and s.loc["300", "total"] == 10
