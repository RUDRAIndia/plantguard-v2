"""Tests for src/evaluate/model_selection.py against a small fake pair of
config.CANDIDATE_MODELS entries. val_macro_f1 is no longer read from a
history log — _compute_val_macro_f1 (the real validation-pass logic, tested
separately via the real TF pipeline in tests/test_evaluate_integration.py)
is stubbed out here so these tests target the orchestration: restoring each
checkpoint, ranking/selecting, and the checkpoint-mtime-keyed cache in
results.json — not the TF plumbing underneath it.
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src import config
from src.evaluate import model_selection

# Two real, cheap-to-build backbones — real names are required since
# build.build_model's constructor table (via inference.load_trained_model,
# stubbed below) is keyed by them.
FAKE_CANDIDATES = ("MobileNetV2", "MobileNetV3Small")


class _FakeModel:
    def __init__(self, param_count):
        self._param_count = param_count

    def count_params(self):
        return self._param_count


@pytest.fixture()
def selection_env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CANDIDATE_MODELS", FAKE_CANDIDATES)
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(config, "CHECKPOINT_DIR", tmp_path / "checkpoints")
    monkeypatch.setattr(config, "RESULTS_JSON_PATH", tmp_path / "artifacts" / "results.json")
    config.ARTIFACTS_DIR.mkdir(parents=True)

    restore_calls = []
    monkeypatch.setattr(
        model_selection.kaggle_persist,
        "restore_checkpoint",
        lambda model_name, checkpoint_dir: restore_calls.append(model_name) or False,
    )

    val_macro_f1_by_model = {"MobileNetV2": 0.55, "MobileNetV3Small": 0.70}
    param_count_by_model = {"MobileNetV2": 1000, "MobileNetV3Small": 2000}

    monkeypatch.setattr(
        model_selection.inference,
        "load_trained_model",
        lambda model_name: _FakeModel(param_count_by_model[model_name]),
    )
    monkeypatch.setattr(
        model_selection,
        "_compute_val_macro_f1",
        lambda model, model_name: (val_macro_f1_by_model[model_name], 42),
    )

    for model_name in FAKE_CANDIDATES:
        checkpoint_dir = config.CHECKPOINT_DIR / model_name
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "phase2.weights.h5").write_bytes(b"\x00" * 1024)

    return {
        "val_macro_f1_by_model": val_macro_f1_by_model,
        "param_count_by_model": param_count_by_model,
        "restore_calls": restore_calls,
    }


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
        assert entry["num_val_samples"] == 42


def test_rank_candidates_restores_checkpoint_for_every_candidate(selection_env):
    model_selection.rank_candidates()
    assert selection_env["restore_calls"] == list(FAKE_CANDIDATES)


def test_rank_candidates_fails_loudly_when_checkpoint_missing_even_after_restore(selection_env):
    checkpoint_dir = config.CHECKPOINT_DIR / "MobileNetV2"
    (checkpoint_dir / "phase2.weights.h5").unlink()
    with pytest.raises(FileNotFoundError, match="No phase1/phase2 checkpoint"):
        model_selection.rank_candidates()


def _write_cache(ranking: list) -> None:
    config.RESULTS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.RESULTS_JSON_PATH.write_text(
        json.dumps({"model_selection": {"ranking": ranking}}), encoding="utf-8"
    )


def test_cache_is_reused_when_checkpoint_is_unchanged(selection_env, monkeypatch):
    first_ranking = model_selection.rank_candidates()
    _write_cache(first_ranking)

    def _poison(model, model_name):
        raise AssertionError(f"_compute_val_macro_f1 was called for {model_name} despite a fresh cache")

    monkeypatch.setattr(model_selection, "_compute_val_macro_f1", _poison)

    second_ranking = model_selection.rank_candidates()
    assert second_ranking == sorted(first_ranking, key=lambda r: r["val_macro_f1"], reverse=True)


def test_cache_entry_missing_required_keys_is_ignored(selection_env):
    _write_cache([{"model_name": "MobileNetV2"}, {"model_name": "MobileNetV3Small"}])  # incomplete rows

    # Must recompute rather than raise/return the incomplete cached rows.
    ranking = model_selection.rank_candidates()
    assert {entry["model_name"] for entry in ranking} == set(FAKE_CANDIDATES)
    for entry in ranking:
        assert "val_macro_f1" in entry and "param_count" in entry


def test_stale_cache_is_recomputed_when_checkpoint_is_newer(selection_env):
    first_ranking = model_selection.rank_candidates()
    _write_cache(first_ranking)

    # Simulate MobileNetV2 having been retrained after the cache was
    # written: bump its checkpoint's mtime into the future.
    weights_path = config.CHECKPOINT_DIR / "MobileNetV2" / "phase2.weights.h5"
    future = time.time() + 3600
    os.utime(weights_path, (future, future))

    selection_env["val_macro_f1_by_model"]["MobileNetV2"] = 0.99  # "retrained" value
    second_ranking = model_selection.rank_candidates()

    entries = {entry["model_name"]: entry for entry in second_ranking}
    assert entries["MobileNetV2"]["val_macro_f1"] == pytest.approx(0.99)  # recomputed, not the stale 0.55
    assert entries["MobileNetV3Small"]["val_macro_f1"] == pytest.approx(0.70)  # unchanged, still cache-eligible


def test_missing_results_json_means_no_cache_and_no_crash(selection_env):
    assert not config.RESULTS_JSON_PATH.exists()
    ranking = model_selection.rank_candidates()  # must recompute cleanly, not raise
    assert len(ranking) == len(FAKE_CANDIDATES)


def test_corrupt_results_json_is_treated_as_no_cache(selection_env):
    config.RESULTS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.RESULTS_JSON_PATH.write_text("{not valid json", encoding="utf-8")
    ranking = model_selection.rank_candidates()  # must recompute cleanly, not raise
    assert len(ranking) == len(FAKE_CANDIDATES)
