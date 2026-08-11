"""Tests for src/evaluate/robustness.py's corruption ops: shape/dtype/range
preservation, and gaussian_noise's stateless-seed determinism — pure tensor
ops, no model.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import tensorflow as tf

from src import config
from src.evaluate import robustness

IMAGE_SHAPE = (config.IMAGE_SIZE, config.IMAGE_SIZE, 3)


def _sample_image(seed: int) -> tf.Tensor:
    rng = np.random.default_rng(seed)
    return tf.constant(rng.uniform(0.0, 1.0, size=IMAGE_SHAPE).astype("float32"))


@pytest.mark.parametrize("corruption", config.ROBUSTNESS_CORRUPTIONS)
@pytest.mark.parametrize("severity", config.ROBUSTNESS_SEVERITIES)
def test_corruption_preserves_shape_and_range(corruption, severity):
    image = _sample_image(0)
    seed = tf.constant([config.SEED, 0], dtype=tf.int32)

    corrupted = robustness.apply_corruption(image, corruption, severity, seed)

    assert corrupted.shape == IMAGE_SHAPE
    assert corrupted.dtype == tf.float32
    assert float(tf.reduce_min(corrupted)) >= -1e-5
    assert float(tf.reduce_max(corrupted)) <= 1.0 + 1e-5


def test_apply_corruption_rejects_unknown_corruption_name():
    image = _sample_image(0)
    seed = tf.constant([0, 0], dtype=tf.int32)
    with pytest.raises(ValueError, match="Unknown corruption"):
        robustness.apply_corruption(image, "not_a_real_corruption", 1, seed)


def test_gaussian_noise_is_deterministic_given_the_same_seed():
    image = _sample_image(1)
    seed = tf.constant([42, 7], dtype=tf.int32)

    first = robustness.gaussian_noise(image, severity=2, seed=seed)
    second = robustness.gaussian_noise(image, severity=2, seed=seed)

    np.testing.assert_array_equal(first.numpy(), second.numpy())


def test_gaussian_noise_differs_across_seeds():
    image = _sample_image(1)
    a = robustness.gaussian_noise(image, severity=2, seed=tf.constant([42, 1], dtype=tf.int32))
    b = robustness.gaussian_noise(image, severity=2, seed=tf.constant([42, 2], dtype=tf.int32))
    assert not np.array_equal(a.numpy(), b.numpy())


def test_higher_severity_noise_deviates_more_from_original():
    image = _sample_image(2)
    seed = tf.constant([1, 1], dtype=tf.int32)

    mild = robustness.gaussian_noise(image, severity=1, seed=seed)
    strong = robustness.gaussian_noise(image, severity=3, seed=seed)

    mild_delta = float(tf.reduce_mean(tf.abs(mild - image)))
    strong_delta = float(tf.reduce_mean(tf.abs(strong - image)))
    assert strong_delta > mild_delta


def test_brightness_up_and_down_move_in_opposite_directions():
    image = tf.fill(IMAGE_SHAPE, 0.5)
    up = robustness.brightness_up(image, severity=2)
    down = robustness.brightness_down(image, severity=2)
    assert float(tf.reduce_mean(up)) > 0.5
    assert float(tf.reduce_mean(down)) < 0.5


def test_blur_reduces_local_variance_on_a_noisy_image():
    image = _sample_image(3)  # uniform random noise — high local variance
    blurred = robustness.blur(image, severity=3)
    assert float(tf.math.reduce_variance(blurred)) < float(tf.math.reduce_variance(image))


def test_severity_params_cover_every_corruption_and_severity():
    for corruption in config.ROBUSTNESS_CORRUPTIONS:
        for severity in config.ROBUSTNESS_SEVERITIES:
            params = robustness._severity_params(corruption, severity)
            assert "value" in params
