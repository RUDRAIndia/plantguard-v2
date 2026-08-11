"""Tests for src/evaluate/calibration.py's pure numeric core: ECE binning,
the softmax(log(p)/T) temperature-scaling identity, and the temperature grid
search — all against synthetic (y_true, probs) arrays, no model.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.evaluate import calibration


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def test_ece_is_small_for_approximately_calibrated_predictions():
    # 100 samples, each predicted with confidence exactly matching its own
    # per-bin empirical accuracy: 50 at confidence 0.9 with 90% correct, 50
    # at confidence 0.5 (=1/2, matching a 2-way tie) with 50% correct.
    rng = np.random.default_rng(0)
    y_true, probs = [], []
    for _ in range(50):
        correct = rng.random() < 0.9
        p_top = 0.9
        y_true.append(0 if correct else 1)
        probs.append([p_top, 1 - p_top] if correct else [1 - p_top, p_top])
    for _ in range(50):
        # confidence 0.5 on a binary problem is already "perfectly
        # calibrated" in the sense that any accuracy in a wide bin around
        # 0.5 contributes ~0 to a coarse-binned ECE; use a genuinely
        # 50/50 coin instead.
        label = int(rng.random() < 0.5)
        y_true.append(label)
        probs.append([0.5, 0.5])

    result = calibration.expected_calibration_error(np.array(y_true), np.array(probs), num_bins=10)
    assert result["ece"] < 0.15  # loose bound: this is a stochastic draw, not an exact identity


def test_ece_bins_are_disjoint_and_cover_all_samples():
    rng = np.random.default_rng(1)
    probs = rng.dirichlet(np.ones(3), size=40)
    y_true = rng.integers(0, 3, size=40)

    result = calibration.expected_calibration_error(y_true, probs, num_bins=5)
    total_counted = sum(b["count"] for b in result["bins"])
    assert total_counted == 40
    assert 0.0 <= result["ece"] <= 1.0


def test_apply_temperature_matches_direct_logits_softmax():
    """The core identity this module's docstring derives: softmax(log(p)/T)
    computed from probs p=softmax(z) must equal softmax(z/T) computed
    directly from the true logits z, for any temperature T.
    """
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(20, 5)) * 3.0
    probs = _softmax(logits)

    for temperature in (0.5, 1.0, 2.0, 4.0):
        expected = _softmax(logits / temperature)
        actual = calibration.apply_temperature(probs, temperature)
        np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_apply_temperature_at_one_is_identity():
    rng = np.random.default_rng(3)
    probs = rng.dirichlet(np.ones(6), size=15)
    scaled = calibration.apply_temperature(probs, 1.0)
    np.testing.assert_allclose(scaled, probs, atol=1e-6)


def test_fit_temperature_softens_an_overconfident_model():
    """An artificially overconfident model (always near-100% on the wrong
    answer half the time) should be fit with T > 1 (softening) rather than
    T < 1 (sharpening), since sharpening an already-overconfident,
    frequently-wrong model can only increase validation NLL.
    """
    rng = np.random.default_rng(4)
    n, num_classes = 200, 4
    y_true = rng.integers(0, num_classes, size=n)

    logits = rng.normal(scale=0.3, size=(n, num_classes))
    # Make the model near-certain about ITS OWN (frequently wrong) guess,
    # not necessarily the true label — this is what makes it overconfident
    # rather than just "confidently correct".
    guessed = rng.integers(0, num_classes, size=n)
    logits[np.arange(n), guessed] += 8.0
    probs = _softmax(logits)

    fit_result = calibration.fit_temperature(y_true, probs)
    assert fit_result["temperature"] > 1.0
    assert fit_result["val_nll"] == pytest.approx(
        calibration._nll(y_true, calibration.apply_temperature(probs, fit_result["temperature"])), abs=1e-9
    )


def test_fit_temperature_beats_or_matches_no_scaling_on_nll():
    rng = np.random.default_rng(5)
    n, num_classes = 100, 3
    y_true = rng.integers(0, num_classes, size=n)
    logits = rng.normal(size=(n, num_classes)) * 2.0
    probs = _softmax(logits)

    fit_result = calibration.fit_temperature(y_true, probs)
    nll_at_one = calibration._nll(y_true, probs)
    assert fit_result["val_nll"] <= nll_at_one + 1e-9
