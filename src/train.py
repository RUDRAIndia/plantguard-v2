"""Two-phase transfer-learning training entrypoint. Kaggle Notebooks (two
Tesla T4s) is the primary training environment; Google Colab is a fallback
usable only while its free tier happens to attach a GPU. See
src/models/phases.py for what each phase actually does and
src/models/checkpoint.py for the per-epoch in-session checkpoint mechanism.

**Cross-session persistence on Kaggle.** /kaggle/working does not survive a
session restart or the ~9-12 hour session time limit, and real training is
roughly 4 GPU-hours across several sessions — so on Kaggle (config.IS_KAGGLE,
unless --no-persist or --smoke), this file pushes each model's checkpoint to
a private Kaggle Dataset after phase 1, after phase 2, and once more with
history/manifest included at the very end, and restores it back into
config.CHECKPOINT_DIR at startup before anything else runs. See
src/models/kaggle_persist.py's module docstring for the full design; the
short version is that a fresh session resuming a model in progress ends up
with exactly the same config.CHECKPOINT_DIR contents an interrupted-but-not-
restarted session would have had, so src/models/checkpoint.py's existing
resume logic (state.json / phaseN_complete.json) runs completely unchanged
on top of it — a fresh session therefore loses at most the training done
since the last phase boundary, never an entire session's work. This never
fires for --smoke runs (their checkpoints are throwaway synthetic-fixture
runs that must never touch the real persisted dataset) or off Kaggle
(Colab/local have no /kaggle/working problem to solve).

Never reads the test split anywhere in this file — CLAUDE.md rule 2
reserves it for final reporting, once, at the very end of the whole
project: _load_datasets() discards pipeline.build_datasets()'s test_ds
without ever iterating it.

--smoke runs config.SMOKE_EPOCHS_PER_PHASE short epoch(s) per phase on a
synthetic on-disk dataset (src/data/smoke_dataset.py) capped at
config.SMOKE_MAX_IMAGES images (CLAUDE.md rule 10) — never the real
pipeline/splits.json, so it's runnable on a machine that has never
downloaded PlantVillage.
"""

import argparse
import json
import sys
from pathlib import Path

import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402
from src.data import pipeline, smoke_dataset, split  # noqa: E402
from src.models import build, kaggle_persist, phases, training_utils  # noqa: E402


def _load_datasets(model_name: str, smoke: bool) -> tuple:
    """Returns (train_ds, val_ds, class_weights, phase1_epochs,
    phase2_epochs). Never returns or touches a test split: the smoke
    fixture has none, and the real path deliberately discards
    pipeline.build_datasets()'s test_ds without ever reading from it.
    """
    if smoke:
        train_paths, train_labels, val_paths, val_labels = smoke_dataset.build_smoke_dataset()
        preprocess_fn = pipeline._resolve_preprocess_fn(model_name)
        train_ds = pipeline._build_pipeline(
            train_paths,
            train_labels,
            training=True,
            batch_size=config.BATCH_SIZE,
            preprocess_fn=preprocess_fn,
            cache_prefix=None,
        )
        val_ds = pipeline._build_pipeline(
            val_paths,
            val_labels,
            training=False,
            batch_size=config.BATCH_SIZE,
            preprocess_fn=preprocess_fn,
            cache_prefix=None,
        )
        # A SMOKE_NUM_CLASSES-class synthetic fixture can't support
        # "balanced" weights over all 38 real classes — the other classes
        # have zero smoke images, and pipeline.compute_class_weights()
        # fails loudly rather than fake a weight for them (CLAUDE.md rule
        # 1). Smoke mode trains unweighted; it's a pipeline plumbing
        # check, not a real training signal.
        return train_ds, val_ds, None, config.SMOKE_EPOCHS_PER_PHASE, config.SMOKE_EPOCHS_PER_PHASE

    train_ds, val_ds, _test_ds, class_weights = pipeline.build_datasets(model_name)
    return train_ds, val_ds, class_weights, config.PHASE1_EPOCHS, config.PHASE2_EPOCHS


def _build_manifest(
    model_name: str,
    smoke: bool,
    class_weights: dict,
    phase1_epochs: int,
    phase2_epochs: int,
    phase1_wall_seconds: float,
    phase2_wall_seconds: float,
    history_path: Path,
) -> dict:
    gpu_devices = [device.name for device in tf.config.list_physical_devices("GPU")]
    build_info = tf.sysconfig.get_build_info()

    return {
        "model_name": model_name,
        "smoke": smoke,
        "seed": config.SEED,
        "git_commit_hash": config.get_git_commit_hash(),
        "split_input_hash": None if smoke else split.compute_split_input_hash(),
        "class_weights": class_weights,
        "hyperparameters": {
            "image_size": config.IMAGE_SIZE,
            "batch_size": config.BATCH_SIZE,
            "dropout_rate": config.DROPOUT_RATE,
            "head_learning_rate": config.HEAD_LEARNING_RATE,
            "backbone_lr_factor": config.BACKBONE_LR_FACTOR,
            "unfreeze_blocks": config.UNFREEZE_BLOCKS[model_name],
            "phase1_epochs_configured": phase1_epochs,
            "phase2_epochs_configured": phase2_epochs,
            "early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
            "reduce_lr_factor": config.REDUCE_LR_FACTOR,
            "reduce_lr_patience": config.REDUCE_LR_PATIENCE,
            "reduce_lr_min_lr": config.REDUCE_LR_MIN_LR,
        },
        "wall_clock_seconds": {"phase1": phase1_wall_seconds, "phase2": phase2_wall_seconds},
        "tensorflow_version": tf.__version__,
        "gpu_devices": gpu_devices,
        "gpu_build_info": {
            "cuda_version": build_info.get("cuda_version"),
            "cudnn_version": build_info.get("cudnn_version"),
        },
        "history_path": str(history_path),
    }


def run_training(model_name: str, smoke: bool, persist: bool = True) -> dict:
    if model_name not in config.CANDIDATE_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. Choices: {sorted(config.CANDIDATE_MODELS)}."
        )

    # Persistence is a Kaggle-specific answer to a Kaggle-specific problem
    # (/kaggle/working not surviving a session restart) — never attempted
    # off Kaggle, and never for --smoke's throwaway synthetic-fixture runs
    # regardless of `persist` (CLAUDE.md rule 10 concern extended to a new
    # failure mode: a smoke run must never touch the real persisted
    # dataset).
    persist_enabled = persist and not smoke and config.IS_KAGGLE

    checkpoint_dir = config.CHECKPOINT_DIR / (model_name + ("_smoke" if smoke else ""))
    if persist_enabled:
        kaggle_persist.restore_checkpoint(model_name, checkpoint_dir)
        kaggle_persist.restore_artifacts(model_name)

    train_ds, val_ds, class_weights, phase1_epochs, phase2_epochs = _load_datasets(model_name, smoke)

    model = build.build_model(model_name)

    phase1_history, phase1_wall_seconds = phases.run_phase1(
        model_name, model, train_ds, val_ds, class_weights, phase1_epochs, checkpoint_dir
    )
    if persist_enabled and phase1_history is not None:
        kaggle_persist.push_checkpoint(model_name, checkpoint_dir)

    _finetune_model, phase2_history, phase2_wall_seconds = phases.run_phase2(
        model_name, model, train_ds, val_ds, class_weights, phase2_epochs, checkpoint_dir
    )
    if persist_enabled and phase2_history is not None:
        kaggle_persist.push_checkpoint(model_name, checkpoint_dir)

    history = {
        "phase1": training_utils.json_safe(phase1_history.history) if phase1_history is not None else None,
        "phase2": training_utils.json_safe(phase2_history.history) if phase2_history is not None else None,
    }
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    history_path = config.ARTIFACTS_DIR / f"history_{model_name}.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    manifest = _build_manifest(
        model_name,
        smoke,
        class_weights,
        phase1_epochs,
        phase2_epochs,
        phase1_wall_seconds,
        phase2_wall_seconds,
        history_path,
    )
    manifest_path = config.ARTIFACTS_DIR / f"train_manifest_{model_name}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if persist_enabled:
        kaggle_persist.push_checkpoint(model_name, checkpoint_dir, include_artifacts=True)

    print(f"[train] Wrote {history_path} and {manifest_path}.")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-phase transfer-learning training for one PlantGuard candidate model."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=config.CANDIDATE_MODELS,
        help="Which config.CANDIDATE_MODELS backbone to train.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            f"Run {config.SMOKE_EPOCHS_PER_PHASE} short epoch(s) per phase on a synthetic "
            f"<= {config.SMOKE_MAX_IMAGES}-image dataset instead of the real one "
            "(CLAUDE.md rule 10: local runs are smoke tests only)."
        ),
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help=(
            "Disable pushing/restoring checkpoints via the Kaggle API "
            "(src/models/kaggle_persist.py). Has no effect off Kaggle or "
            "on --smoke runs, which never persist regardless."
        ),
    )
    args = parser.parse_args()

    run_training(args.model, args.smoke, persist=not args.no_persist)


if __name__ == "__main__":
    main()
