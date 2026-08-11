"""Tests for src/evaluate/gradcam.py: select_indices()'s pure sampling logic
(no model), plus one real-model check that compute_heatmap actually produces
a valid, correctly-shaped, [0, 1]-normalized heatmap end to end.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from PIL import Image

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


def _write_synthetic_images(directory: Path, n: int) -> list:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        path = directory / f"img{i}.jpg"
        Image.new("RGB", (60, 45), color=((i * 40) % 256, (i * 80) % 256, (i * 120) % 256)).save(path)
        paths.append(str(path))
    return paths


def test_generate_gradcam_samples_wires_plantdoc_failures_correctly(real_model, tmp_path, monkeypatch):
    """End-to-end, real-model check of item 5's mechanism: given a known
    correct/incorrect split for both the PlantVillage-test-shaped and
    PlantDoc-shaped inputs, generate_gradcam_samples must (a) pick exactly
    the mispredicted PlantDoc indices, never a correct one, (b) attribute
    each sample's true/predicted class names using config.PLANTVILLAGE_
    CLASS_NAMES at the right index (never a PlantDoc-local index -- same
    concern as the label-mapping audit), and (c) actually write a heatmap
    PNG to disk for every sample claimed in its return value. This does not
    (and cannot, without the real trained checkpoint and real PlantDoc
    images) prove the heatmaps land on leaves rather than background --
    only that the sampling/attribution/file-writing plumbing feeding that
    visual judgment is correct.
    """
    monkeypatch.setattr(config, "EVAL_FIGURES_DIR", tmp_path / "figures")

    test_paths = _write_synthetic_images(tmp_path / "test", 4)
    test_true = np.array([0, 0, 1, 1])
    test_pred = np.array([0, 1, 1, 0])  # idx 1, 3 wrong

    plantdoc_paths = _write_synthetic_images(tmp_path / "plantdoc", 3)
    plantdoc_true = np.array([0, 1, 2])
    plantdoc_pred = np.array([0, 1, 0])  # only idx 2 wrong

    result = gradcam.generate_gradcam_samples(
        real_model, MODEL_NAME, test_true, test_pred, test_paths, plantdoc_true, plantdoc_pred, plantdoc_paths
    )

    assert result["num_correct_sampled"] == 2
    assert result["num_incorrect_sampled"] == 2
    assert result["num_plantdoc_failures_sampled"] == 1
    assert result["num_samples"] == 5

    plantdoc_samples = [s for s in result["samples"] if s["group"] == "plantdoc_failure"]
    assert len(plantdoc_samples) == 1
    sample = plantdoc_samples[0]
    assert sample["source_path"] == plantdoc_paths[2]
    assert sample["true_class"] == config.PLANTVILLAGE_CLASS_NAMES[2]
    assert sample["predicted_class"] == config.PLANTVILLAGE_CLASS_NAMES[0]

    for s in result["samples"]:
        assert Path(s["figure_path"]).is_file()
