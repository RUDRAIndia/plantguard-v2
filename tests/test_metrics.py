"""Tests for src/models/metrics.py: proves MacroF1Score matches sklearn's
f1_score(average='macro') and PerClassRecall matches sklearn's
recall_score(average=None) on a small fixed example, then checks the
stateful accumulate/reset_state contract every Keras Metric needs to behave
correctly across epochs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import tensorflow as tf
from sklearn.metrics import f1_score, recall_score

from src.models.metrics import MacroF1Score, PerClassRecall

NUM_CLASSES = 4
# Every one of the 4 classes appears at least once in Y_TRUE, so sklearn's
# recall/F1 never hits its zero-division edge case (a class absent from
# y_true) — keeps this a clean, unambiguous numeric comparison.
Y_TRUE = np.array([0, 0, 0, 1, 1, 2, 2, 2, 3, 0, 1, 2, 3, 3])
Y_PRED_LABELS = np.array([0, 1, 0, 1, 1, 2, 0, 2, 3, 0, 1, 1, 3, 2])


def _one_hot_probabilities(labels: np.ndarray) -> np.ndarray:
    return np.eye(NUM_CLASSES)[labels].astype("float32")


def test_macro_f1_matches_sklearn():
    expected = f1_score(Y_TRUE, Y_PRED_LABELS, average="macro")

    metric = MacroF1Score(num_classes=NUM_CLASSES)
    metric.update_state(tf.constant(Y_TRUE), tf.constant(_one_hot_probabilities(Y_PRED_LABELS)))

    assert float(metric.result()) == pytest.approx(expected, abs=1e-5)


def test_per_class_recall_matches_sklearn():
    expected = recall_score(Y_TRUE, Y_PRED_LABELS, average=None)

    metric = PerClassRecall(num_classes=NUM_CLASSES)
    metric.update_state(tf.constant(Y_TRUE), tf.constant(_one_hot_probabilities(Y_PRED_LABELS)))

    np.testing.assert_allclose(metric.result().numpy(), expected, atol=1e-5)


def test_macro_f1_accumulates_across_batches_matching_sklearn_on_the_full_set():
    # Splitting the same fixed example into two update_state() calls (as
    # model.fit() would across mini-batches) must give the same result as
    # one call on the whole thing — proves accumulation happens over a
    # running confusion matrix, not by averaging per-batch scores (which
    # would silently misweight a smaller final batch).
    expected = f1_score(Y_TRUE, Y_PRED_LABELS, average="macro")
    midpoint = len(Y_TRUE) // 2

    metric = MacroF1Score(num_classes=NUM_CLASSES)
    metric.update_state(
        tf.constant(Y_TRUE[:midpoint]), tf.constant(_one_hot_probabilities(Y_PRED_LABELS[:midpoint]))
    )
    metric.update_state(
        tf.constant(Y_TRUE[midpoint:]), tf.constant(_one_hot_probabilities(Y_PRED_LABELS[midpoint:]))
    )

    assert float(metric.result()) == pytest.approx(expected, abs=1e-5)


def test_reset_state_clears_accumulated_confusion():
    metric = MacroF1Score(num_classes=NUM_CLASSES)
    metric.update_state(tf.constant(Y_TRUE), tf.constant(_one_hot_probabilities(Y_PRED_LABELS)))
    assert float(metric.result()) > 0.0

    metric.reset_state()

    # A perfect-prediction batch after reset must score 1.0 — if reset_state
    # left stale confusion-matrix entries behind, this would be pulled down
    # by the previous (imperfect) batch's contribution.
    metric.update_state(tf.constant(Y_TRUE), tf.constant(_one_hot_probabilities(Y_TRUE)))
    assert float(metric.result()) == pytest.approx(1.0, abs=1e-5)
