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
from src.models import kaggle_persist


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


class _FakeHistory:
    def __init__(self, val_macro_f1):
        self.history = {"val_macro_f1": [val_macro_f1]}
        self.epoch = [0]


def _stub_run_training_internals(monkeypatch, tmp_path):
    """Replaces everything expensive/real (real datasets, a real backbone,
    real Keras fit() calls, the real split-input-hash lookup) with cheap
    stand-ins, so these tests can exercise run_training()'s persistence
    *gating* logic (--no-persist / smoke / IS_KAGGLE) without needing real
    PlantVillage data or a downloaded backbone.
    """
    monkeypatch.setattr(
        train, "_load_datasets", lambda model_name, smoke: (object(), object(), {0: 1.0}, 1, 1)
    )
    monkeypatch.setattr(train.build, "build_model", lambda model_name: object())
    monkeypatch.setattr(
        train.phases,
        "run_phase1",
        lambda model_name, model, train_ds, val_ds, class_weights, epochs, checkpoint_dir: (
            _FakeHistory(0.1),
            1.0,
        ),
    )
    monkeypatch.setattr(
        train.phases,
        "run_phase2",
        lambda model_name, model, train_ds, val_ds, class_weights, epochs, checkpoint_dir: (
            model,
            _FakeHistory(0.2),
            1.0,
        ),
    )
    monkeypatch.setattr(train.split, "compute_split_input_hash", lambda: "fake-split-hash")
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(config, "CHECKPOINT_DIR", tmp_path / "checkpoints")


def _poison_kaggle_persist(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("kaggle_persist function called when persistence should not fire")

    monkeypatch.setattr(kaggle_persist, "restore_checkpoint", _boom)
    monkeypatch.setattr(kaggle_persist, "restore_artifacts", _boom)
    monkeypatch.setattr(kaggle_persist, "push_checkpoint", _boom)


def test_persistence_never_fires_for_smoke_runs_even_on_kaggle(tmp_path, monkeypatch):
    _stub_run_training_internals(monkeypatch, tmp_path)
    monkeypatch.setattr(config, "IS_KAGGLE", True)
    _poison_kaggle_persist(monkeypatch)

    train.run_training("MobileNetV2", smoke=True, persist=True)  # must not raise


def test_persistence_never_fires_off_kaggle(tmp_path, monkeypatch):
    _stub_run_training_internals(monkeypatch, tmp_path)
    monkeypatch.setattr(config, "IS_KAGGLE", False)
    _poison_kaggle_persist(monkeypatch)

    train.run_training("MobileNetV2", smoke=False, persist=True)  # must not raise


def test_no_persist_flag_disables_persistence_on_kaggle(tmp_path, monkeypatch):
    _stub_run_training_internals(monkeypatch, tmp_path)
    monkeypatch.setattr(config, "IS_KAGGLE", True)
    _poison_kaggle_persist(monkeypatch)

    train.run_training("MobileNetV2", smoke=False, persist=False)  # must not raise


def test_real_kaggle_run_restores_on_startup_and_pushes_after_each_phase(tmp_path, monkeypatch):
    _stub_run_training_internals(monkeypatch, tmp_path)
    monkeypatch.setattr(config, "IS_KAGGLE", True)

    calls = []
    monkeypatch.setattr(
        kaggle_persist, "restore_checkpoint", lambda model_name, checkpoint_dir: calls.append("restore_checkpoint")
    )
    monkeypatch.setattr(kaggle_persist, "restore_artifacts", lambda model_name: calls.append("restore_artifacts"))

    def _fake_push(model_name, checkpoint_dir, include_artifacts=False):
        calls.append(("push", include_artifacts))

    monkeypatch.setattr(kaggle_persist, "push_checkpoint", _fake_push)

    train.run_training("MobileNetV2", smoke=False, persist=True)

    assert calls == [
        "restore_checkpoint",
        "restore_artifacts",
        ("push", False),  # after phase 1
        ("push", False),  # after phase 2
        ("push", True),  # final push with history/manifest
    ]


def test_cli_no_persist_flag_reaches_run_training(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        train,
        "run_training",
        lambda model_name, smoke, persist=True: captured.update(
            model_name=model_name, smoke=smoke, persist=persist
        ),
    )
    monkeypatch.setattr(sys, "argv", ["train.py", "--model", "MobileNetV2", "--smoke", "--no-persist"])

    train.main()

    assert captured == {"model_name": "MobileNetV2", "smoke": True, "persist": False}
