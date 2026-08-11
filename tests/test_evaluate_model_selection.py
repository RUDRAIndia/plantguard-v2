"""Tests for src/evaluate/model_selection.py against a small fake pair of
config.CANDIDATE_MODELS entries, with fake history_*.json / checkpoint files
written to a tmp_path — proves ranking/selection reads exactly the recorded
val_macro_f1 (never test-set-derived) and never silently invents a value for
a missing artifact.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src import config
from src.evaluate import model_selection

# Two real, cheap-to-build backbones — real names are required since
# build.build_model's constructor table is keyed by them.
FAKE_CANDIDATES = ("MobileNetV2", "MobileNetV3Small")


@pytest.fixture()
def selection_env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CANDIDATE_MODELS", FAKE_CANDIDATES)
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(config, "CHECKPOINT_DIR", tmp_path / "checkpoints")
    config.ARTIFACTS_DIR.mkdir(parents=True)

    val_macro_f1_by_model = {"MobileNetV2": 0.55, "MobileNetV3Small": 0.70}
    for model_name, best_f1 in val_macro_f1_by_model.items():
        history = {
            "phase1": {"val_macro_f1": [best_f1 - 0.1]},
            "phase2": {"val_macro_f1": [best_f1 - 0.05, best_f1]},
        }
        (config.ARTIFACTS_DIR / f"history_{model_name}.json").write_text(json.dumps(history), encoding="utf-8")

        checkpoint_dir = config.CHECKPOINT_DIR / model_name
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "phase2.weights.h5").write_bytes(b"\x00" * 1024)

    return val_macro_f1_by_model


def test_select_best_model_picks_the_highest_val_macro_f1(selection_env):
    result = model_selection.select_best_model()
    assert result["selected_model"] == "MobileNetV3Small"
    assert result["ranking"][0]["model_name"] == "MobileNetV3Small"
    assert result["ranking"][0]["val_macro_f1"] == pytest.approx(0.70)
    assert result["ranking"][1]["model_name"] == "MobileNetV2"


def test_rank_candidates_records_param_count_and_file_size(selection_env):
    ranking = model_selection.rank_candidates()
    for entry in ranking:
        assert entry["param_count"] > 0
        assert entry["checkpoint_file_size_bytes"] == 1024


def test_rank_candidates_fails_loudly_when_history_missing(selection_env):
    (config.ARTIFACTS_DIR / "history_MobileNetV2.json").unlink()
    with pytest.raises(FileNotFoundError, match="history_MobileNetV2"):
        model_selection.rank_candidates()


def test_checkpoint_file_size_fails_loudly_when_no_checkpoint_exists(selection_env):
    checkpoint_dir = config.CHECKPOINT_DIR / "MobileNetV2"
    (checkpoint_dir / "phase2.weights.h5").unlink()
    with pytest.raises(FileNotFoundError, match="No phase1/phase2 checkpoint"):
        model_selection.rank_candidates()
