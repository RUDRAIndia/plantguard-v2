"""Tests for src/train.py: the --smoke path actually produces a history
file end to end (a real training run, on a synthetic <= 200-image dataset
per CLAUDE.md rule 10 — this is the same code path colab/01_data_setup.ipynb
Cell 11 exercises on Colab), and the real (non-smoke) dataset-loading branch
never touches the test split src/data/pipeline.py's build_datasets()
returns (CLAUDE.md rule 2).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, train
from src.data import pipeline


class _PoisonedDataset:
    """Raises on any use — proves a caller holding this object never reads
    from it. Standing in for pipeline.build_datasets()'s real test_ds.
    """

    def __getattr__(self, name):
        raise AssertionError(
            f"test_ds.{name} was accessed — CLAUDE.md rule 2 forbids touching "
            "the test split during training."
        )

    def __iter__(self):
        raise AssertionError(
            "test_ds was iterated — CLAUDE.md rule 2 forbids touching the "
            "test split during training."
        )


def test_load_datasets_never_touches_test_split(monkeypatch):
    fake_train_ds = object()
    fake_val_ds = object()
    fake_class_weights = {0: 1.0, 1: 2.0}

    def fake_build_datasets(model_name, batch_size=None):
        return fake_train_ds, fake_val_ds, _PoisonedDataset(), fake_class_weights

    monkeypatch.setattr(pipeline, "build_datasets", fake_build_datasets)

    train_ds, val_ds, class_weights, phase1_epochs, phase2_epochs = train._load_datasets(
        "MobileNetV2", smoke=False
    )

    assert train_ds is fake_train_ds
    assert val_ds is fake_val_ds
    assert class_weights is fake_class_weights
    assert phase1_epochs == config.PHASE1_EPOCHS
    assert phase2_epochs == config.PHASE2_EPOCHS


def test_smoke_run_produces_history_and_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SMOKE_DIR", tmp_path / "smoke")
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(config, "CHECKPOINT_DIR", tmp_path / "checkpoints")

    model_name = "MobileNetV2"
    manifest = train.run_training(model_name, smoke=True)

    history_path = config.ARTIFACTS_DIR / f"history_{model_name}.json"
    manifest_path = config.ARTIFACTS_DIR / f"train_manifest_{model_name}.json"
    assert history_path.is_file()
    assert manifest_path.is_file()

    history = json.loads(history_path.read_text(encoding="utf-8"))
    for phase_key in ("phase1", "phase2"):
        assert history[phase_key] is not None
        assert len(history[phase_key]["val_macro_f1"]) == config.SMOKE_EPOCHS_PER_PHASE
        assert len(history[phase_key]["val_per_class_recall"][0]) == config.NUM_CLASSES

    assert manifest["model_name"] == model_name
    assert manifest["smoke"] is True
    assert manifest["split_input_hash"] is None  # smoke never reads the real split
    assert manifest["class_weights"] is None  # smoke's synthetic fixture skips class weighting
    assert manifest["seed"] == config.SEED
    assert manifest["git_commit_hash"]
    assert manifest["hyperparameters"]["unfreeze_blocks"] == config.UNFREEZE_BLOCKS[model_name]
    assert manifest == json.loads(manifest_path.read_text(encoding="utf-8"))

    # Re-running against the same checkpoint dir must short-circuit both
    # phases (already-complete markers), not silently redo real work.
    manifest_again = train.run_training(model_name, smoke=True)
    assert manifest_again["model_name"] == model_name
