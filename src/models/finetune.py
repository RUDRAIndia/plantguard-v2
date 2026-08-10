"""Phase-2 fine-tuning support: a differential-learning-rate model wrapper
and a callback that keeps the backbone's learning rate synced to it.

Plain Keras optimizers apply one global learning rate to every trainable
variable — there's no "parameter group" concept like PyTorch's. Scaling
gradients instead of the learning rate is NOT an equivalent substitute for
an adaptive optimizer like Adam: Adam normalizes each update by that
variable's own running gradient magnitude (v_hat), so multiplying the
gradient by a constant factor c leaves the update almost unchanged (the c
cancels between the numerator and sqrt(v_hat) ~ c terms) instead of shrinking
it by c. Getting a genuinely smaller backbone step therefore requires a
second optimizer instance with its own, smaller learning rate — this module
implements that via a custom train_step, the pattern Keras' own
"Customizing what happens in fit()" guide uses for multiple optimizers.
"""

import sys
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config  # noqa: E402
from src.models import build  # noqa: E402


class DifferentialLRModel(keras.Model):
    """A drop-in replacement for a plain functional keras.Model that routes
    gradients for backbone variables through `self.backbone_optimizer`
    (set by the caller after construction, before compile/fit) and
    everything else through `self.optimizer` (the normal one, set by
    compile()). Reuses the exact same layers/weights as the model it wraps
    — constructing a Model from another model's `.input`/`.output` shares
    the underlying variables rather than copying them, so this is a
    behavior upgrade, not a retrain-from-scratch.
    """

    def __init__(self, inputs, outputs, backbone_variable_ids: set, **kwargs):
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)
        self._backbone_variable_ids = backbone_variable_ids
        self.backbone_optimizer = None  # caller must set before fit()

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(y=y, y_pred=y_pred)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        head_grads_and_vars, backbone_grads_and_vars = [], []
        for var, grad in zip(trainable_vars, gradients):
            if grad is None:
                continue
            if id(var) in self._backbone_variable_ids:
                backbone_grads_and_vars.append((grad, var))
            else:
                head_grads_and_vars.append((grad, var))

        self.optimizer.apply_gradients(head_grads_and_vars)
        if backbone_grads_and_vars:
            if self.backbone_optimizer is None:
                raise RuntimeError(
                    "DifferentialLRModel has unfrozen backbone variables but "
                    "backbone_optimizer was never set — call site must assign "
                    "it before fit()."
                )
            self.backbone_optimizer.apply_gradients(backbone_grads_and_vars)

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}


def wrap_for_differential_lr(model: keras.Model, backbone_lr: float) -> DifferentialLRModel:
    """Rebuilds `model` (already unfrozen via build.unfreeze_top_blocks) as
    a DifferentialLRModel sharing its exact layers/weights, with a fresh
    Adam backbone_optimizer at `backbone_lr`. Caller still owns compiling
    the returned model with the head optimizer/loss/metrics as normal —
    only the backbone side is pre-wired here.
    """
    backbone = build.get_backbone(model)
    backbone_variable_ids = {id(v) for v in backbone.trainable_variables}

    wrapped = DifferentialLRModel(
        model.input, model.output, backbone_variable_ids, name=model.name
    )
    wrapped.backbone_optimizer = keras.optimizers.Adam(learning_rate=backbone_lr)
    return wrapped


class SyncBackboneLR(keras.callbacks.Callback):
    """Keeps backbone_optimizer's learning rate at
    head_optimizer.learning_rate * config.BACKBONE_LR_FACTOR every epoch —
    including after a ReduceLROnPlateau reduction. Must be listed AFTER
    ReduceLROnPlateau in the callbacks passed to fit() (Keras runs
    callbacks in list order), so it re-derives the backbone rate from the
    head rate ReduceLROnPlateau just possibly changed, rather than the
    other way around: ReduceLROnPlateau only ever adjusts
    `model.optimizer`, so without this callback the backbone side would
    silently stop tracking it after the first LR reduction.
    """

    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        if self.model.backbone_optimizer is None:
            return
        head_lr = float(self.model.optimizer.learning_rate)
        self.model.backbone_optimizer.learning_rate.assign(head_lr * config.BACKBONE_LR_FACTOR)
