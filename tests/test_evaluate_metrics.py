"""Tests for src/evaluate/metrics.py's pure numeric core: compute_metrics()
and its _top_confused_pairs() helper, checked against sklearn ground truth
and hand-verified confusion counts on a small fixed example — no model, no
file I/O.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from sklearn.metrics import f1_score

from src.evaluate import metrics

CLASS_NAMES = ("A", "B", "C", "D")
# Every class appears at least once in Y_TRUE, avoiding sklearn's
# zero-division edge case for a cleaner ground-truth comparison.
Y_TRUE = np.array([0, 0, 0, 1, 1, 2, 2, 2, 3, 0, 1, 2, 3, 3])
Y_PRED_LABELS = np.array([0, 1, 0, 1, 1, 2, 0, 2, 3, 0, 1, 1, 3, 2])


def _one_hot_probs(labels: np.ndarray, num_classes: int) -> np.ndarray:
    return np.eye(num_classes)[labels].astype("float64")


def test_compute_metrics_accuracy_and_f1_match_sklearn():
    probs = _one_hot_probs(Y_PRED_LABELS, len(CLASS_NAMES))
    result = metrics.compute_metrics(Y_TRUE, probs, CLASS_NAMES)

    expected_accuracy = float(np.mean(Y_PRED_LABELS == Y_TRUE))
    expected_macro_f1 = f1_score(Y_TRUE, Y_PRED_LABELS, average="macro")
    expected_weighted_f1 = f1_score(Y_TRUE, Y_PRED_LABELS, average="weighted")

    assert result["accuracy"] == pytest.approx(expected_accuracy)
    assert result["macro_f1"] == pytest.approx(expected_macro_f1, abs=1e-6)
    assert result["weighted_f1"] == pytest.approx(expected_weighted_f1, abs=1e-6)
    assert result["num_test_samples"] == len(Y_TRUE)


def test_compute_metrics_per_class_and_worst_class_recall():
    probs = _one_hot_probs(Y_PRED_LABELS, len(CLASS_NAMES))
    result = metrics.compute_metrics(Y_TRUE, probs, CLASS_NAMES)

    per_class = {c["class_name"]: c for c in result["per_class_metrics"]}
    assert set(per_class) == set(CLASS_NAMES)

    # Class 0: true=[0,0,0,0] (indices 0,1,2,9) pred=[0,1,0,0] -> recall 3/4.
    assert per_class["A"]["recall"] == pytest.approx(0.75)
    assert per_class["A"]["support"] == 4

    worst = min(per_class.values(), key=lambda c: c["recall"])
    assert result["worst_class_recall"]["class_name"] == worst["class_name"]
    assert result["worst_class_recall"]["recall"] == pytest.approx(worst["recall"])


def test_compute_metrics_returns_confusion_matrix_for_caller_to_consume():
    probs = _one_hot_probs(Y_PRED_LABELS, len(CLASS_NAMES))
    result = metrics.compute_metrics(Y_TRUE, probs, CLASS_NAMES)
    confusion = result["_confusion_matrix"]
    assert confusion.shape == (4, 4)
    assert confusion.sum() == len(Y_TRUE)


def test_top_confused_pairs_ranks_by_count_and_excludes_diagonal():
    # 3 classes; true=0 gets predicted as 1 three times (most-confused pair),
    # true=1 gets predicted as 2 once. Diagonal (correct) entries must never
    # appear in the output.
    confusion = np.array(
        [
            [2, 3, 0],
            [0, 5, 1],
            [0, 0, 4],
        ]
    )
    class_names = ("X", "Y", "Z")

    pairs = metrics._top_confused_pairs(confusion, class_names, n=10)

    assert all(p["true_class"] != p["predicted_class"] for p in pairs)
    assert pairs[0] == {"true_class": "X", "predicted_class": "Y", "count": 3, "rate_of_true_class": pytest.approx(0.6)}
    assert pairs[1]["true_class"] == "Y" and pairs[1]["predicted_class"] == "Z" and pairs[1]["count"] == 1
    assert len(pairs) == 2  # only 2 nonzero off-diagonal cells exist


def test_top_confused_pairs_respects_n_limit():
    confusion = np.array([[0, 5, 4], [3, 0, 2], [1, 6, 0]])
    class_names = ("X", "Y", "Z")

    pairs = metrics._top_confused_pairs(confusion, class_names, n=2)
    assert len(pairs) == 2
    assert [p["count"] for p in pairs] == sorted([p["count"] for p in pairs], reverse=True)
