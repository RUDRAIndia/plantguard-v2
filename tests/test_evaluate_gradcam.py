"""Tests for src/evaluate/gradcam.py: select_indices()'s pure sampling logic
(no model), plus one real-model check that compute_heatmap actually produces
a valid, correctly-shaped, [0, 1]-normalized heatmap end to end.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src import config
from src.evaluate import gradcam
from src.models import build

MODEL_NAME = "MobileNetV2"


def test_select_indices_respects_mask():
    mask = np.array([True, False, True, True, False, True])
    selected = gradcam.select_indices(mask, n=10, seed=0)
    assert set(selected) == {0, 2, 3, 5}


def test_select_indices_caps_at_n():
    mask = np.array([True] * 20)
    selected = gradcam.select_indices(mask, n=5, seed=0)
    assert len(selected) == 5
    assert len(set(selected)) == 5


def test_select_indices_deterministic_given_same_seed():
    mask = np.array([True] * 30)
    first = gradcam.select_indices(mask, n=10, seed=123)
    second = gradcam.select_indices(mask, n=10, seed=123)
    assert first == second


def test_select_indices_differs_across_seeds():
    mask = np.array([True] * 30)
    a = gradcam.select_indices(mask, n=10, seed=1)
    b = gradcam.select_indices(mask, n=10, seed=2)
    assert a != b


def test_select_indices_returns_fewer_than_n_when_mask_has_fewer_true():
    mask = np.array([True, False, False, True])
    selected = gradcam.select_indices(mask, n=10, seed=0)
    assert len(selected) == 2


@pytest.fixture(scope="module")
def real_model():
    return build.build_model(MODEL_NAME)


def test_last_conv_layer_finds_a_4d_output_layer(real_model):
    backbone = build.get_backbone(real_model)
    layer = gradcam._last_conv_layer(backbone)
    assert len(layer.output.shape) == 4


def test_compute_heatmap_shape_and_range(real_model):
    grad_model = gradcam.build_grad_model(real_model)
    rng = np.random.default_rng(0)
    model_input = rng.uniform(-1.0, 1.0, size=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3)).astype("float32")

    heatmap = gradcam.compute_heatmap(real_model, grad_model, model_input, target_class_idx=0)

    assert heatmap.ndim == 2
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-5
