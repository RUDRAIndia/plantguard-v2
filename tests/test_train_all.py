"""Tests for src/train_all.py: the multi-model runner must skip models that
are already complete (locally or, on Kaggle, cross-session), isolate each
model's failure so the rest still run, and its exit code must never lie
about whether every model actually finished.
"""

import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src import config, train_all
from src.models import kaggle_persist_report

_FAKE_MODELS = ("ModelA", "ModelB", "ModelC")


@pytest.fixture(autouse=True)
def fake_candidate_models(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CANDIDATE_MODELS", _FAKE_MODELS)
    monkeypatch.setattr(config, "CHECKPOINT_DIR", tmp_path / "checkpoints")
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(config, "IS_KAGGLE", False)


def _model_name_from_cmd(cmd) -> str:
    return cmd[cmd.index("--model") + 1]


def _write_history(model_name: str, best_f1: float):
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    history_path = config.ARTIFACTS_DIR / f"history_{model_name}.json"
    history_path.write_text(
        json.dumps({"phase1": {"val_macro_f1": [best_f1]}, "phase2": None}), encoding="utf-8"
    )


def _mark_locally_complete(model_name: str, smoke: bool = False):
    checkpoint_dir = config.CHECKPOINT_DIR / (model_name + ("_smoke" if smoke else ""))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "phase1_complete.json").write_text("{}", encoding="utf-8")
    (checkpoint_dir / "phase2_complete.json").write_text("{}", encoding="utf-8")


def test_is_locally_complete_requires_both_phase_markers():
    monkeypatch_dir = config.CHECKPOINT_DIR / "ModelA"
    monkeypatch_dir.mkdir(parents=True)
    assert train_all._is_locally_complete("ModelA", smoke=False) is False

    (monkeypatch_dir / "phase1_complete.json").write_text("{}", encoding="utf-8")
    assert train_all._is_locally_complete("ModelA", smoke=False) is False  # phase 2 still missing

    (monkeypatch_dir / "phase2_complete.json").write_text("{}", encoding="utf-8")
    assert train_all._is_locally_complete("ModelA", smoke=False) is True


def test_skips_a_locally_complete_model_without_running_it(monkeypatch):
    _mark_locally_complete("ModelA")
    _write_history("ModelA", 0.75)

    calls = []
    monkeypatch.setattr(
        train_all.subprocess,
        "run",
        lambda *a, **k: calls.append(a) or types.SimpleNamespace(returncode=0, stderr=""),
    )
    monkeypatch.setattr(sys, "argv", ["train_all.py", "--no-persist"])

    with pytest.raises(SystemExit) as exc_info:
        train_all.main()

    assert exc_info.value.code == 0
    ran_models = {_model_name_from_cmd(c[0]) for c in calls}
    assert "ModelA" not in ran_models
    assert ran_models == {"ModelB", "ModelC"}


def test_skips_a_cross_session_complete_model_on_kaggle(monkeypatch):
    monkeypatch.setattr(config, "IS_KAGGLE", True)
    monkeypatch.setattr(
        kaggle_persist_report,
        "report_progress",
        lambda: {
            "ModelA": {"checkpoint_exists": True, "phase_reached": "phase 2 complete", "best_val_macro_f1": 0.9},
            "ModelB": {"checkpoint_exists": False, "phase_reached": "not started", "best_val_macro_f1": None},
            "ModelC": {"checkpoint_exists": False, "phase_reached": "not started", "best_val_macro_f1": None},
        },
    )

    calls = []
    monkeypatch.setattr(
        train_all.subprocess,
        "run",
        lambda *a, **k: calls.append(a) or types.SimpleNamespace(returncode=0, stderr=""),
    )
    monkeypatch.setattr(sys, "argv", ["train_all.py"])

    with pytest.raises(SystemExit) as exc_info:
        train_all.main()

    assert exc_info.value.code == 0
    ran_models = {_model_name_from_cmd(c[0]) for c in calls}
    assert ran_models == {"ModelB", "ModelC"}  # ModelA skipped via persisted state, not local files


def test_one_models_failure_does_not_stop_the_others(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_all.py", "--no-persist"])

    # _run_one_model reads best-val-macro-F1 from the local history file the
    # real subprocess would have written -- simulate that for the two
    # successful models.
    def _run_with_history(cmd, stdout=None, stderr=None, text=True):
        model_name = _model_name_from_cmd(cmd)
        if model_name == "ModelB":
            return types.SimpleNamespace(returncode=1, stderr="Traceback...\nRuntimeError: boom")
        _write_history(model_name, {"ModelA": 0.5, "ModelC": 0.6}[model_name])
        return types.SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(train_all.subprocess, "run", _run_with_history)

    with pytest.raises(SystemExit) as exc_info:
        train_all.main()

    assert exc_info.value.code == 1  # ModelB failed -- must not be reported as a clean run


def test_only_flag_restricts_to_the_given_subset(monkeypatch):
    calls = []
    monkeypatch.setattr(
        train_all.subprocess,
        "run",
        lambda *a, **k: calls.append(a) or types.SimpleNamespace(returncode=0, stderr=""),
    )
    monkeypatch.setattr(sys, "argv", ["train_all.py", "--no-persist", "--only", "ModelA", "ModelC"])

    with pytest.raises(SystemExit) as exc_info:
        train_all.main()

    assert exc_info.value.code == 0
    ran_models = {_model_name_from_cmd(c[0]) for c in calls}
    assert ran_models == {"ModelA", "ModelC"}


def test_smoke_and_no_persist_flags_are_passed_through_to_each_subprocess(monkeypatch):
    calls = []
    monkeypatch.setattr(
        train_all.subprocess,
        "run",
        lambda *a, **k: calls.append(a) or types.SimpleNamespace(returncode=0, stderr=""),
    )
    monkeypatch.setattr(sys, "argv", ["train_all.py", "--smoke", "--no-persist", "--only", "ModelA"])

    with pytest.raises(SystemExit):
        train_all.main()

    (cmd,) = calls[0]
    assert "--smoke" in cmd
    assert "--no-persist" in cmd


def test_exit_code_is_zero_when_every_model_completes(monkeypatch):
    def _run(cmd, stdout=None, stderr=None, text=True):
        _write_history(_model_name_from_cmd(cmd), 0.42)
        return types.SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(train_all.subprocess, "run", _run)
    monkeypatch.setattr(sys, "argv", ["train_all.py", "--no-persist"])

    with pytest.raises(SystemExit) as exc_info:
        train_all.main()

    assert exc_info.value.code == 0
