"""Small shared pieces of src/train.py's two-phase loop: the metrics list
every compile() call uses, the EarlyStopping/ReduceLROnPlateau pair (both
monitoring validation macro-F1 — CLAUDE.md rule 4: accuracy is misleading
under a 36:1 class imbalance), and a Keras History -> plain-JSON converter.
"""

import sys
from pathlib import Path

import numpy as np
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config  # noqa: E402
from src.models import metrics  # noqa: E402


def compile_metrics() -> list:
    return [
        metrics.MacroF1Score(config.NUM_CLASSES),
        metrics.PerClassRecall(config.NUM_CLASSES),
        "accuracy",
    ]


def early_stopping_and_reduce_lr() -> list:
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_macro_f1",
            mode="max",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_macro_f1",
            mode="max",
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=config.REDUCE_LR_MIN_LR,
        ),
    ]


def json_safe(value):
    """Recursively converts a Keras History.history-shaped structure
    (floats, and — for src.models.metrics.PerClassRecall — per-epoch numpy
    arrays) into plain JSON-serializable Python types.
    """
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value
