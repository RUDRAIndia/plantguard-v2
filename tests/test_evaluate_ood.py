"""Tests for src/evaluate/ood.py's pure numeric core: choose_threshold(),
_population_summary(), and _plantdoc_safety_note() against synthetic
max-softmax-confidence distributions — no model.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.evaluate import ood


def test_choose_threshold_separates_well_separated_distributions():
    rng = np.random.default_rng(0)
    val_confidence = rng.uniform(0.85, 1.0, size=500)  # genuine leaves: high confidence
    negative_confidence = rng.uniform(0.0, 0.4, size=500)  # OOD: low confidence

    result = ood.choose_threshold(val_confidence, negative_confidence)

    assert 0.4 <= result["threshold"] <= 0.85
    assert result["rejection_rate_on_negatives"] > 0.95
    assert result["false_rejection_rate_on_genuine_leaves"] < 0.05
    assert result["j_statistic"] > 0.9


def test_choose_threshold_handles_overlapping_distributions_without_crashing():
    rng = np.random.default_rng(1)
    val_confidence = rng.uniform(0.3, 0.9, size=300)
    negative_confidence = rng.uniform(0.1, 0.7, size=300)

    result = ood.choose_threshold(val_confidence, negative_confidence)
    assert 0.0 <= result["threshold"] <= 1.0
    assert 0.0 <= result["rejection_rate_on_negatives"] <= 1.0
    assert 0.0 <= result["false_rejection_rate_on_genuine_leaves"] <= 1.0


def test_choose_threshold_never_rejects_everything_when_negatives_are_all_confident():
    # Degenerate case: negatives look just as confident as genuine leaves.
    # The best achievable operating point should not force away all
    # genuine-leaf acceptance just to reject a handful of negatives.
    val_confidence = np.full(100, 0.9)
    negative_confidence = np.full(100, 0.9)

    result = ood.choose_threshold(val_confidence, negative_confidence)
    # Every threshold yields identical rejection/false-rejection rates here
    # (J == 0 everywhere) — the search must still return a valid threshold.
    assert 0.0 <= result["threshold"] <= 1.0


def test_population_summary_reports_accepted_accuracy_when_labels_given():
    confidence = np.array([0.99, 0.99, 0.10, 0.99])
    y_true = np.array([0, 1, 2, 1])
    y_pred = np.array([0, 0, 2, 1])  # index 1 wrong, but still accepted

    summary = ood._population_summary("val", confidence, threshold=0.5, y_true=y_true, y_pred=y_pred)

    assert summary["num_samples"] == 4
    assert summary["num_accepted"] == 3  # the 0.10-confidence sample is rejected
    assert summary["rejection_rate"] == pytest.approx(0.25)
    assert summary["accuracy_on_accepted"] == pytest.approx(2 / 3)  # 2 of the 3 accepted are correct
    assert summary["mean_confidence"] == pytest.approx(np.mean(confidence))
    assert summary["median_confidence"] == pytest.approx(np.median(confidence))


def test_population_summary_omits_accuracy_when_no_labels_given():
    confidence = np.array([0.2, 0.3, 0.4])

    summary = ood._population_summary("intel_negatives", confidence, threshold=0.98)

    assert summary["accuracy_on_accepted"] is None
    assert summary["num_accepted"] == 0
    assert summary["rejection_rate"] == pytest.approx(1.0)


def test_population_summary_accuracy_is_none_when_nothing_is_accepted():
    confidence = np.array([0.1, 0.2])
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])

    summary = ood._population_summary("plantdoc", confidence, threshold=0.98, y_true=y_true, y_pred=y_pred)

    assert summary["num_accepted"] == 0
    assert summary["accuracy_on_accepted"] is None


def test_plantdoc_safety_note_names_the_harm_case_when_gate_provides_no_filtering():
    summary = {
        "num_samples": 100,
        "num_accepted": 40,
        "rejection_rate": 0.60,
        "accuracy_on_accepted": 0.15,
    }
    note = ood._plantdoc_safety_note(summary, overall_plantdoc_accuracy=0.1928, threshold=0.98)

    assert "NOT" in note
    assert "harm case" in note


def test_plantdoc_safety_note_names_safe_but_useless_when_everything_is_rejected():
    summary = {"num_samples": 50, "num_accepted": 0, "rejection_rate": 1.0, "accuracy_on_accepted": None}
    note = ood._plantdoc_safety_note(summary, overall_plantdoc_accuracy=0.1928, threshold=0.98)

    assert "safe" in note
    assert "unusable" in note


def test_plantdoc_safety_note_names_partial_protection_when_accepted_accuracy_beats_overall():
    summary = {
        "num_samples": 100,
        "num_accepted": 30,
        "rejection_rate": 0.70,
        "accuracy_on_accepted": 0.55,
    }
    note = ood._plantdoc_safety_note(summary, overall_plantdoc_accuracy=0.1928, threshold=0.98)

    assert "partial" in note


def test_tune_ood_rejection_rejects_a_partial_plantdoc_argument_set():
    with pytest.raises(ValueError, match="together or not at all"):
        ood.tune_ood_rejection(
            model=None,
            model_name="MobileNetV2",
            plantdoc_y_true=np.array([0]),
            plantdoc_y_pred=None,
            plantdoc_y_prob=None,
        )
