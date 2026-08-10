"""Tests for src/models/finetune.py's DifferentialLRModel.train_step,
specifically its data-unpacking: src/models/phases.py's run_phase2 passes
class_weight to fit(), which Keras converts into a per-sample sample_weight
and hands train_step a 3-tuple (x, y, sample_weight) — a bare `x, y = data`
crashes there (ValueError: too many values to unpack), which is exactly
what killed the first real Kaggle run at the start of phase 2. These tests
prove both that the 3-tuple case no longer crashes and, more importantly,
that the sample weights it carries actually reach the loss rather than
being silently dropped (which would disable class weighting for phase 2
only, with no error).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from tensorflow import keras

from src.models import build, finetune

MODEL_NAME = "MobileNetV2"


class _RecordingLoss(keras.losses.Loss):
    """Wraps sparse_categorical_crossentropy but records whatever
    sample_weight it was actually called with, so a test can assert on it
    directly instead of inferring it indirectly from loss/gradient values.
    """

    def __init__(self):
        super().__init__(name="recording_loss")
        self.received_sample_weight = "NEVER_CALLED"

    def call(self, y_true, y_pred):
        return keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

    def __call__(self, y_true, y_pred, sample_weight=None):
        self.received_sample_weight = sample_weight
        return super().__call__(y_true, y_pred, sample_weight)


@pytest.fixture(scope="module")
def wrapped_model():
    model = build.build_model(MODEL_NAME)
    build.unfreeze_top_blocks(model, MODEL_NAME)
    return finetune.wrap_for_differential_lr(model, backbone_lr=1e-4)


def test_train_step_handles_3_tuple_from_class_weight_without_crashing(wrapped_model):
    x = np.random.rand(4, 224, 224, 3).astype("float32")
    y = np.array([0, 1, 0, 1])

    wrapped_model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    # class_weight is what turns fit()'s data tuple into (x, y, sample_weight)
    # — this must not raise "too many values to unpack".
    wrapped_model.fit(x, y, epochs=1, batch_size=4, verbose=0, class_weight={0: 1.0, 1: 5.0})


def test_train_step_still_handles_2_tuple_without_class_weight(wrapped_model):
    x = np.random.rand(4, 224, 224, 3).astype("float32")
    y = np.array([0, 1, 0, 1])

    wrapped_model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    wrapped_model.fit(x, y, epochs=1, batch_size=4, verbose=0)


def test_class_weight_sample_weights_actually_reach_the_loss(wrapped_model):
    """The critical assertion: not just "doesn't crash", but the exact
    per-sample weights class_weight implies are what compute_loss actually
    receives -- verified with shuffle=False so batch order is deterministic
    and the recorded tensor can be compared element-for-element against
    class_weight applied to the (unshuffled) labels.
    """
    x = np.random.rand(8, 224, 224, 3).astype("float32")
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    class_weight = {0: 1.0, 1: 5.0}

    recording_loss = _RecordingLoss()
    wrapped_model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=recording_loss,
        metrics=["accuracy"],
        run_eagerly=True,  # so the recorded sample_weight is a concrete, inspectable tensor
    )

    wrapped_model.fit(
        x, y, epochs=1, batch_size=8, verbose=0, class_weight=class_weight, shuffle=False
    )

    received = recording_loss.received_sample_weight
    assert not isinstance(received, str), "compute_loss was never called with a sample_weight at all"
    expected = np.where(y == 1, class_weight[1], class_weight[0]).astype("float32")
    np.testing.assert_allclose(received.numpy(), expected)


def test_no_class_weight_means_no_sample_weight(wrapped_model):
    x = np.random.rand(4, 224, 224, 3).astype("float32")
    y = np.array([0, 1, 0, 1])

    recording_loss = _RecordingLoss()
    wrapped_model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=recording_loss,
        metrics=["accuracy"],
        run_eagerly=True,
    )

    wrapped_model.fit(x, y, epochs=1, batch_size=4, verbose=0)

    assert recording_loss.received_sample_weight is None
