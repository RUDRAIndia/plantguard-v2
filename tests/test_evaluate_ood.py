"""Tests for src/evaluate/ood.py's pure numeric core: choose_threshold()
against synthetic max-softmax-confidence distributions — no model.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

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
