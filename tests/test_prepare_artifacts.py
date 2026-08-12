"""Tests for src/data/prepare_artifacts.py's ensure_data_artifacts(): the
skip-vs-regenerate decision, and that it never weakens the split provenance
check src/data/pipeline.py's load_splits() applies at training time (a
restored or already-local splits.json is only trusted if its recorded
split_input_hash matches a fresh split.compute_split_inputs() recomputation
— never just "the file exists").

Deliberately does NOT reuse tests/conftest.py's module-scoped
`synthetic_dataset` fixture (which runs the real dedupe.py/split_report.py
against real synthetic images) — the decision logic under test here only
needs a real split.compute_split_inputs()/hash_split_inputs() round trip
(which needs nothing but a real dataset_provenance.json and the real
dedupe.py/phash_cluster.py/split.py source files already on disk, not a
real PlantVillage-shaped image tree), so a lightweight per-test fixture
keeps these tests isolated from each other and fast.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src import config
from src.data import prepare_artifacts, split


@pytest.fixture
def isolated_artifacts(monkeypatch, tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    monkeypatch.setattr(config, "ARTIFACTS_DIR", artifacts_dir)

    provenance_path = tmp_path / "dataset_provenance.json"
    provenance_path.write_text(
        json.dumps(
            {
                "plantvillage": {
                    "source": "test-stub",
                    "observed_image_count": 100,
                    "observed_class_count": 3,
                    "sha256_of_sorted_relative_paths": "stub-hash",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "DATASET_PROVENANCE_PATH", provenance_path)
    return artifacts_dir


def _write_stub_data_prep_files(artifacts_dir: Path, *, split_input_hash: str) -> None:
    """Writes every _DATA_PREP_FILENAMES entry as a content-free stub except
    splits.json, whose manifest.split_input_hash is the one field
    _local_data_prep_artifacts_valid() actually inspects.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for name in prepare_artifacts._DATA_PREP_FILENAMES:
        if name == "splits.json":
            continue
        (artifacts_dir / name).write_text("stub", encoding="utf-8")

    manifest = {"split_input_hash": split_input_hash}
    (artifacts_dir / "splits.json").write_text(
        json.dumps({"manifest": manifest, "splits": {"train": [], "val": [], "test": []}}),
        encoding="utf-8",
    )


def _current_split_input_hash() -> str:
    return split.hash_split_inputs(split.compute_split_inputs())


@pytest.fixture
def spy_mains(monkeypatch):
    calls: list[str] = []
    for name, module in (
        ("dedupe", prepare_artifacts.dedupe),
        ("split_report", prepare_artifacts.split_report),
        ("inventory", prepare_artifacts.inventory),
        ("mapping_report", prepare_artifacts.mapping_report),
    ):
        monkeypatch.setattr(module, "main", lambda name=name: calls.append(name))
    return calls


def test_skips_regeneration_when_local_artifacts_are_complete_and_valid(
    isolated_artifacts, spy_mains, monkeypatch
):
    monkeypatch.setattr(config, "IS_KAGGLE", False)
    _write_stub_data_prep_files(isolated_artifacts, split_input_hash=_current_split_input_hash())

    prepare_artifacts.ensure_data_artifacts()

    assert spy_mains == []


def test_regenerates_when_a_data_prep_file_is_missing(isolated_artifacts, spy_mains, monkeypatch):
    monkeypatch.setattr(config, "IS_KAGGLE", False)
    # Nothing written at all -- isolated_artifacts dir doesn't even exist yet.

    prepare_artifacts.ensure_data_artifacts()

    assert spy_mains == ["dedupe", "split_report", "inventory", "mapping_report"]


def test_regenerates_when_split_input_hash_is_stale(isolated_artifacts, spy_mains, monkeypatch):
    """The core provenance guard: every file is present, but the recorded
    split_input_hash doesn't match a fresh recomputation -- this must still
    force full regeneration, proving a complete-looking artifacts/ isn't
    trusted just because the files exist (CLAUDE.md's split-provenance
    guarantee must never be weakened by this restore-and-skip fast path).
    """
    monkeypatch.setattr(config, "IS_KAGGLE", False)
    _write_stub_data_prep_files(isolated_artifacts, split_input_hash="deliberately-stale-hash")

    prepare_artifacts.ensure_data_artifacts()

    assert spy_mains == ["dedupe", "split_report", "inventory", "mapping_report"]


def test_attempts_restore_only_when_is_kaggle(isolated_artifacts, spy_mains, monkeypatch):
    restore_calls = []
    monkeypatch.setattr(
        prepare_artifacts.kaggle_persist_artifacts, "restore_data_artifacts", lambda: restore_calls.append(1) or False
    )

    monkeypatch.setattr(config, "IS_KAGGLE", False)
    prepare_artifacts.ensure_data_artifacts()
    assert restore_calls == []

    monkeypatch.setattr(config, "IS_KAGGLE", True)
    prepare_artifacts.ensure_data_artifacts()
    assert restore_calls == [1]


def test_restored_valid_files_skip_regeneration(isolated_artifacts, spy_mains, monkeypatch):
    """End-to-end restore-then-validate flow: restore_data_artifacts() is
    mocked to simulate a real restore's side effect (writing files into
    ARTIFACTS_DIR), and a valid restored splits.json must be enough to skip
    regeneration without ever touching the real Kaggle API.
    """
    monkeypatch.setattr(config, "IS_KAGGLE", True)

    def fake_restore():
        _write_stub_data_prep_files(isolated_artifacts, split_input_hash=_current_split_input_hash())
        return True

    monkeypatch.setattr(prepare_artifacts.kaggle_persist_artifacts, "restore_data_artifacts", fake_restore)

    prepare_artifacts.ensure_data_artifacts()

    assert spy_mains == []


def test_restored_stale_files_still_trigger_regeneration(isolated_artifacts, spy_mains, monkeypatch):
    monkeypatch.setattr(config, "IS_KAGGLE", True)

    def fake_restore():
        _write_stub_data_prep_files(isolated_artifacts, split_input_hash="stale-from-a-different-code-version")
        return True

    monkeypatch.setattr(prepare_artifacts.kaggle_persist_artifacts, "restore_data_artifacts", fake_restore)

    prepare_artifacts.ensure_data_artifacts()

    assert spy_mains == ["dedupe", "split_report", "inventory", "mapping_report"]
