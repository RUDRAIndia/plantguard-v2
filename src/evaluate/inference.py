"""Shared helpers every src/evaluate/* module reuses instead of re-deriving
its own model-loading or batched-inference plumbing: locating a model's real
(non-smoke) trained checkpoint, loading it back into a keras.Model, building
a deterministic backbone-preprocessed pipeline for an arbitrary (paths,
labels) list, and running a model over a tf.data.Dataset to collect
predictions.
"""

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config  # noqa: E402
from src.data import pipeline  # noqa: E402
from src.models import build  # noqa: E402


def checkpoint_weights_path(model_name: str) -> Path:
    """The final-phase weights file for `model_name`'s real (non-smoke)
    training run: phase 2 (fine-tuned) if it exists, else phase 1 (frozen
    backbone only) — never falls back further than that, since a model with
    neither has not actually been trained.
    """
    checkpoint_dir = config.CHECKPOINT_DIR / model_name
    phase2 = checkpoint_dir / "phase2.weights.h5"
    if phase2.is_file():
        return phase2
    phase1 = checkpoint_dir / "phase1.weights.h5"
    if phase1.is_file():
        return phase1
    raise FileNotFoundError(
        f"No phase1/phase2 checkpoint found under {checkpoint_dir}. Train "
        f"'{model_name}' first (src/train.py) before evaluating it."
    )


def load_trained_model(model_name: str) -> tf.keras.Model:
    """Builds `model_name`'s architecture (src/models/build.py) and loads
    its real trained checkpoint weights. Never used for --smoke checkpoints
    — those live in a "{model_name}_smoke" directory this never looks at.
    """
    model = build.build_model(model_name)
    model.load_weights(checkpoint_weights_path(model_name))
    return model


def build_eval_pipeline(paths: list, labels: list, model_name: str) -> tf.data.Dataset:
    """Deterministic (no shuffle, no augmentation), `model_name`-preprocessed
    pipeline for an arbitrary (paths, labels) list — mirrors
    src/data/pipeline.py's val/test branch exactly, so external/robustness/
    calibration/OOD evaluation all see input distributed identically to how
    the model was validated. Iteration order is guaranteed to match `paths`'
    order (no shuffle), so callers can align predictions back to file paths
    by position.
    """
    preprocess_fn = pipeline._resolve_preprocess_fn(model_name)
    return pipeline._build_pipeline(
        paths,
        labels,
        training=False,
        batch_size=config.BATCH_SIZE,
        preprocess_fn=preprocess_fn,
        shuffle=False,
        cache_prefix=None,
    )


def predict_dataset(model: tf.keras.Model, dataset: tf.data.Dataset) -> tuple:
    """Runs `model` over every batch of `dataset` (already backbone-
    preprocessed, as build_eval_pipeline/build_datasets produce) and returns
    (y_true, y_prob) as numpy arrays: integer labels and the full softmax
    output, concatenated in dataset iteration order.
    """
    y_true_batches, y_prob_batches = [], []
    for images, labels in dataset:
        probs = model(images, training=False)
        y_true_batches.append(labels.numpy())
        y_prob_batches.append(np.asarray(probs))

    if not y_true_batches:
        raise RuntimeError("Dataset produced zero batches — nothing to evaluate.")
    return np.concatenate(y_true_batches), np.concatenate(y_prob_batches)


def predict_paths(model: tf.keras.Model, paths: list, model_name: str) -> np.ndarray:
    """Same as predict_dataset, but for inputs with no meaningful label
    (e.g. OOD negatives) — dummy labels of -1 are threaded through purely so
    build_eval_pipeline's (paths, labels) contract is satisfied; only y_prob
    is returned.
    """
    dummy_labels = [-1] * len(paths)
    dataset = build_eval_pipeline(paths, dummy_labels, model_name)
    _, y_prob = predict_dataset(model, dataset)
    return y_prob
