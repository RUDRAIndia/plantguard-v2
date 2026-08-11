"""Tests for src/evaluate/inference.py: checkpoint_weights_path's phase2->
phase1->raise fallback (no model needed), and a real load_trained_model
round trip proving saved weights are actually the ones loaded back.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src import config
from src.evaluate import inference
from src.models import build

MODEL_NAME = "MobileNetV2"


@pytest.fixture()
def checkpoint_env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHECKPOINT_DIR", tmp_path / "checkpoints")
    return config.CHECKPOINT_DIR / MODEL_NAME


def test_checkpoint_weights_path_prefers_phase2(checkpoint_env):
    checkpoint_env.mkdir(parents=True)
    (checkpoint_env / "phase1.weights.h5").write_bytes(b"\x00")
    (checkpoint_env / "phase2.weights.h5").write_bytes(b"\x00")

    result = inference.checkpoint_weights_path(MODEL_NAME)
    assert result.name == "phase2.weights.h5"


def test_checkpoint_weights_path_falls_back_to_phase1(checkpoint_env):
    checkpoint_env.mkdir(parents=True)
    (checkpoint_env / "phase1.weights.h5").write_bytes(b"\x00")

    result = inference.checkpoint_weights_path(MODEL_NAME)
    assert result.name == "phase1.weights.h5"


def test_checkpoint_weights_path_raises_when_neither_exists(checkpoint_env):
    with pytest.raises(FileNotFoundError, match="No phase1/phase2 checkpoint"):
        inference.checkpoint_weights_path(MODEL_NAME)


def test_load_trained_model_round_trips_real_weights(checkpoint_env):
    checkpoint_env.mkdir(parents=True)
    original = build.build_model(MODEL_NAME)
    original.save_weights(checkpoint_env / "phase2.weights.h5")

    loaded = inference.load_trained_model(MODEL_NAME)

    for original_weight, loaded_weight in zip(original.get_weights(), loaded.get_weights()):
        np.testing.assert_array_equal(original_weight, loaded_weight)
