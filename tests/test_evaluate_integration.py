"""Integration test: proves src/evaluate/metrics.py, calibration.py, and
ood.py actually wire together correctly against a REAL (untrained,
just-initialized) MobileNetV2 model and the synthetic PlantVillage-shaped
fixture from tests/conftest.py — the riskiest new plumbing (tf.data
pipelines built from arbitrary paths, Grad-CAM-adjacent gradient-free
inference, JSON round-tripping) exercised for real rather than only
unit-tested in isolation. Never asserts on prediction *quality* (the model
is untrained, so accuracy/ECE numbers are meaningless) — only on shape,
type, and internal consistency.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from PIL import Image

from src import config
from src.evaluate import calibration, metrics, ood
from src.models import build

MODEL_NAME = "MobileNetV2"


@pytest.fixture(scope="module")
def eval_env(synthetic_dataset, tmp_path_factory, monkeypatch_module_scoped):
    # EVAL_FIGURES_DIR/CHECKPOINT_DIR/RESULTS_JSON_PATH are resolved once at
    # config.py import time from the real ARTIFACTS_DIR — patching
    # config.ARTIFACTS_DIR (done by the synthetic_dataset fixture) does NOT
    # follow through to them, so each needs its own explicit patch (same
    # reasoning as conftest.py's DATASET_PROVENANCE_PATH patch).
    figures_dir = tmp_path_factory.mktemp("figures")
    negatives_dir = tmp_path_factory.mktemp("negatives")
    android_metadata_path = tmp_path_factory.mktemp("android") / "model_metadata.json"

    monkeypatch_module_scoped.setattr(config, "EVAL_FIGURES_DIR", figures_dir)
    monkeypatch_module_scoped.setattr(config, "NEGATIVES_DIR", negatives_dir)
    monkeypatch_module_scoped.setattr(config, "ANDROID_MODEL_METADATA_PATH", android_metadata_path)

    for category in ("buildings", "forest"):
        category_dir = negatives_dir / category
        category_dir.mkdir(parents=True)
        for i in range(6):
            Image.new("RGB", (64, 64), color=(i * 20, i * 10, i * 5)).save(category_dir / f"img{i}.jpg")

    android_metadata_path.write_text(
        json.dumps({"schema_version": 1, "placeholder": True, "confidence_threshold": 0.6, "class_names": []}),
        encoding="utf-8",
    )

    model = build.build_model(MODEL_NAME)
    return synthetic_dataset, model


def test_evaluate_test_set_runs_end_to_end(eval_env):
    _, model = eval_env
    class_names = tuple(config.PLANTVILLAGE_CLASS_NAMES)

    result, y_true, y_prob, paths = metrics.evaluate_test_set(model, MODEL_NAME, class_names)

    assert result["num_test_samples"] == len(y_true) == len(paths)
    assert y_prob.shape == (len(y_true), len(class_names))
    np.testing.assert_allclose(y_prob.sum(axis=1), 1.0, atol=1e-4)  # softmax output
    assert Path(result["confusion_matrix_figure"]).is_file()
    assert set(y_true) <= set(range(len(class_names)))


def test_calibration_runs_end_to_end_reusing_test_predictions(eval_env):
    _, model = eval_env
    class_names = tuple(config.PLANTVILLAGE_CLASS_NAMES)
    _, test_y_true, test_y_prob, _ = metrics.evaluate_test_set(model, MODEL_NAME, class_names)

    result = calibration.run_calibration(model, MODEL_NAME, test_y_true, test_y_prob)

    assert 0.0 <= result["test_ece_before"] <= 1.0
    assert 0.0 <= result["test_ece_after"] <= 1.0
    assert config.TEMPERATURE_GRID_MIN <= result["temperature"] <= config.TEMPERATURE_GRID_MAX
    assert Path(result["reliability_diagram_figure"]).is_file()


def test_ood_tuning_runs_end_to_end_and_updates_android_metadata(eval_env):
    _, model = eval_env

    result = ood.tune_ood_rejection(model, MODEL_NAME)

    assert 0.0 <= result["chosen_threshold"] <= 1.0
    assert result["num_negative_samples"] == 12
    updated = json.loads(config.ANDROID_MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    assert updated["confidence_threshold"] == result["chosen_threshold"]
    assert "confidence_threshold_source" in updated
    assert updated["placeholder"] is True  # untouched field, per CLAUDE.md rule 13
    # No real PlantDoc data was supplied -- the report-only PlantDoc arm
    # must not appear at all (never a null/placeholder stand-in for it).
    assert "plantdoc_at_chosen_threshold" not in result
    assert "plantdoc_safety_note" not in result
    assert len(result["population_comparison_at_chosen_threshold"]) == 2


def test_ood_tuning_reports_a_plantdoc_arm_without_retuning_the_threshold(eval_env):
    """Feeds synthetic-but-real (softmax-normalized) PlantDoc-shaped
    predictions straight in as arrays -- src/evaluate/ood.py's new PlantDoc
    arm only needs y_true/y_pred/y_prob, never real PlantDoc images, so this
    exercises the real wiring without needing a real PlantDoc download
    (CLAUDE.md rule 10).
    """
    _, model = eval_env
    num_classes = len(config.PLANTVILLAGE_CLASS_NAMES)
    rng = np.random.default_rng(0)
    n = 24
    plantdoc_y_true = rng.integers(0, num_classes, size=n)
    logits = rng.normal(size=(n, num_classes))
    plantdoc_y_prob = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    plantdoc_y_pred = plantdoc_y_prob.argmax(axis=1)

    baseline = ood.tune_ood_rejection(model, MODEL_NAME)
    result = ood.tune_ood_rejection(
        model,
        MODEL_NAME,
        plantdoc_y_true=plantdoc_y_true,
        plantdoc_y_pred=plantdoc_y_pred,
        plantdoc_y_prob=plantdoc_y_prob,
    )

    # Supplying PlantDoc data must never change the chosen threshold itself
    # (it's tuned on validation + negatives only -- PlantDoc is report-only).
    assert result["chosen_threshold"] == baseline["chosen_threshold"]

    plantdoc_arm = result["plantdoc_at_chosen_threshold"]
    assert plantdoc_arm["num_samples"] == n
    assert 0.0 <= plantdoc_arm["rejection_rate"] <= 1.0
    assert 0.0 <= plantdoc_arm["mean_confidence"] <= 1.0
    assert isinstance(result["plantdoc_safety_note"], str) and len(result["plantdoc_safety_note"]) > 0

    populations = {p["population"] for p in result["population_comparison_at_chosen_threshold"]}
    assert populations == {
        "validation_in_distribution",
        "intel_negatives_out_of_distribution",
        "plantdoc_external_field_photos",
    }
