"""The two training phases src/train.py drives: phase 1 trains a frozen-
backbone classifier head; phase 2 unfreezes the backbone's last blocks
(src/models/build.py) and fine-tunes head and backbone at different
learning rates (src/models/finetune.py) — the old project used the same
rate for both phases, which pushes large updates through the just-unfrozen
pretrained ImageNet features and damages them. Both phases checkpoint every
epoch (src/models/checkpoint.py) and resume automatically if a checkpoint
already exists for their phase.
"""

import json
import sys
import time
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config  # noqa: E402
from src.models import build, checkpoint, finetune, training_utils  # noqa: E402


def run_phase1(
    model_name: str,
    model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    class_weights: dict,
    epochs: int,
    checkpoint_dir: Path,
) -> tuple:
    """Frozen-backbone, head-only training. Returns (history_or_None,
    wall_seconds) — history is None if phase 1 was already complete on
    entry (resumed run, nothing left to do).
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    complete_marker = checkpoint_dir / "phase1_complete.json"

    if complete_marker.is_file():
        model.load_weights(checkpoint_dir / "phase1.weights.h5")
        print(f"[train] '{model_name}' phase 1 already complete — loaded final weights, skipping.")
        return None, 0.0

    initial_epoch = 0
    state = checkpoint.read_state(checkpoint_dir)
    if state is not None and state["phase"] == 1:
        model.load_weights(state["weights_path"])
        initial_epoch = state["epoch"] + 1
        print(f"[train] Resuming '{model_name}' phase 1 from epoch {initial_epoch}.")

    build.freeze_backbone(model)
    model.compile(
        optimizer=keras.optimizers.Adam(config.HEAD_LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=training_utils.compile_metrics(),
    )
    callbacks = training_utils.early_stopping_and_reduce_lr() + [
        checkpoint.EpochCheckpointCallback(1, checkpoint_dir)
    ]

    start = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        initial_epoch=initial_epoch,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )
    wall_seconds = time.time() - start

    model.save_weights(checkpoint_dir / "phase1.weights.h5")
    complete_marker.write_text(json.dumps({"epochs_ran": len(history.epoch)}), encoding="utf-8")
    return history, wall_seconds


def run_phase2(
    model_name: str,
    model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    class_weights: dict,
    epochs: int,
    checkpoint_dir: Path,
) -> tuple:
    """Unfreezes the top blocks and fine-tunes at a
    config.BACKBONE_LR_FACTOR-lower backbone learning rate
    (src/models/finetune.py). Returns (finetune_model, history_or_None,
    wall_seconds).
    """
    complete_marker = checkpoint_dir / "phase2_complete.json"

    build.unfreeze_top_blocks(model, model_name)
    finetune_model = finetune.wrap_for_differential_lr(
        model, backbone_lr=config.HEAD_LEARNING_RATE * config.BACKBONE_LR_FACTOR
    )
    finetune_model.compile(
        optimizer=keras.optimizers.Adam(config.HEAD_LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=training_utils.compile_metrics(),
    )

    if complete_marker.is_file():
        finetune_model.load_weights(checkpoint_dir / "phase2.weights.h5")
        print(f"[train] '{model_name}' phase 2 already complete — loaded final weights, skipping.")
        return finetune_model, None, 0.0

    initial_epoch = 0
    state = checkpoint.read_state(checkpoint_dir)
    if state is not None and state["phase"] == 2:
        finetune_model.load_weights(state["weights_path"])
        initial_epoch = state["epoch"] + 1
        print(f"[train] Resuming '{model_name}' phase 2 from epoch {initial_epoch}.")

    # SyncBackboneLR must run after ReduceLROnPlateau (list order) so it
    # re-derives the backbone rate from whatever the head rate just became.
    callbacks = training_utils.early_stopping_and_reduce_lr() + [
        finetune.SyncBackboneLR(),
        checkpoint.EpochCheckpointCallback(2, checkpoint_dir),
    ]

    start = time.time()
    history = finetune_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        initial_epoch=initial_epoch,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )
    wall_seconds = time.time() - start

    finetune_model.save_weights(checkpoint_dir / "phase2.weights.h5")
    complete_marker.write_text(json.dumps({"epochs_ran": len(history.epoch)}), encoding="utf-8")
    return finetune_model, history, wall_seconds
