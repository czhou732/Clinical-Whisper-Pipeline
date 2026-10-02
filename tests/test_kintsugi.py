"""Kintsugi's voice model add-on: weight folding, levels, participant choice."""

import numpy as np
import pytest
import torch

import kintsugi_dam as k


def test_lora_is_folded_into_the_base_weight():
    a, b = torch.randn(4, 6), torch.randn(5, 4)
    state = {
        "backbone.audio.backbone.base_model.model.l.q.base_layer.weight": torch.zeros(5, 6),
        "backbone.audio.backbone.base_model.model.l.q.base_layer.bias": torch.ones(5),
        "backbone.audio.backbone.base_model.model.l.q.lora_A.default.weight": a,
        "backbone.audio.backbone.base_model.model.l.q.lora_B.default.weight": b,
        "backbone.audio.backbone.base_model.model.conv1.original_module.weight": torch.zeros(2),
        "backbone.audio.backbone.base_model.model.conv1.modules_to_save.default.weight": torch.ones(2),
    }
    out = k._merge(state, "backbone.audio.backbone.")
    assert set(out) == {"l.q.weight", "l.q.bias", "conv1.weight"}
    assert torch.allclose(out["l.q.weight"], k._LORA_SCALE * (b @ a))
    assert torch.equal(out["conv1.weight"], torch.ones(2))


def test_levels_follow_their_thresholds():
    assert k._level("depression", -1.0) == 0
    assert k._level("depression", -0.5) == 1
    assert k._level("depression", 0.1) == 2
    assert k._level("anxiety", 0.2) == 3


def test_subject_choice_and_skips():
    per = {"S02": {"depression": {"level": 1}, "speech_s": 400.0}}
    assert k.for_subject(per, "S02")["depression"]["level"] == 1
    assert "No single participant" in k.for_subject(per, None)["skipped"]
    assert "under 30 s" in k.for_subject(per, "S01")["skipped"]
    assert k.for_subject(None, "S02") is None


@pytest.mark.skipif(not k.available(), reason="Kintsugi checkpoint not on this machine")
def test_real_checkpoint_loads_strictly_and_scores():
    rng = np.random.default_rng(0)
    out = k.score_audio((0.05 * rng.standard_normal(16000 * 35)).astype(np.float32))
    assert set(out) == {"depression", "anxiety"}
    assert out["depression"]["label"].startswith(("none", "mild", "severe"))
