"""Tests for src/data/split.py's grouped/stratified allocation, run against
the synthetic dataset built by the `synthetic_dataset` fixture in
tests/conftest.py (see that file's docstring for why synthetic, not real,
data).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data import split

# Acceptance tolerance for this test's proportion check — a test-assertion
# threshold, not a pipeline hyperparameter, so it lives here rather than in
# src/config.py. 0.10 (10 percentage points) comfortably accommodates the
# synthetic fixture's small per-class group counts while still catching a
# genuinely broken stratification (e.g. one split starved to near-zero).
PROPORTION_TOLERANCE = 0.10


def test_no_group_spans_splits(synthetic_dataset):
    image_to_group = synthetic_dataset["image_to_group"]
    splits = synthetic_dataset["splits"]

    group_to_splits_seen: dict[str, set[str]] = {}
    for split_name, paths in splits.items():
        for path in paths:
            group_id = image_to_group[path]
            group_to_splits_seen.setdefault(group_id, set()).add(split_name)

    spanning = {gid: seen for gid, seen in group_to_splits_seen.items() if len(seen) > 1}
    assert not spanning, f"group_id(s) spanning multiple splits: {spanning}"


def test_no_image_in_two_splits(synthetic_dataset):
    splits = synthetic_dataset["splits"]
    all_paths = [path for paths in splits.values() for path in paths]
    assert len(all_paths) == len(set(all_paths)), "an image path appears more than once across splits"
    assert len(all_paths) == synthetic_dataset["total_images"]


def test_proportions_within_tolerance(synthetic_dataset):
    splits = synthetic_dataset["splits"]
    total = sum(len(paths) for paths in splits.values())
    target = {
        "train": config.TRAIN_FRACTION,
        "val": config.VAL_FRACTION,
        "test": config.TEST_FRACTION,
    }
    for split_name, fraction_target in target.items():
        achieved = len(splits[split_name]) / total
        assert abs(achieved - fraction_target) <= PROPORTION_TOLERANCE, (
            f"achieved '{split_name}' fraction {achieved:.3f} is more than "
            f"{PROPORTION_TOLERANCE:.0%} away from target {fraction_target:.3f}"
        )


def test_split_deterministic_for_same_seed(synthetic_dataset):
    image_to_group = synthetic_dataset["image_to_group"]
    first = split.build_split(image_to_group)
    second = split.build_split(image_to_group)
    assert first == second, "re-running build_split with the same seed/input produced a different split"


def _write_fake_provenance(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "plantvillage": {
                    "observed_image_count": 100,
                    "observed_class_count": 3,
                    "sha256_of_sorted_relative_paths": "deadbeef",
                }
            }
        ),
        encoding="utf-8",
    )


def _fake_module_paths(tmp_path: Path) -> tuple:
    module_paths = tuple(tmp_path / name for name in split._SPLIT_PRODUCING_MODULE_NAMES)
    for path in module_paths:
        path.write_text("# original content\n", encoding="utf-8")
    return module_paths


def test_unrelated_module_edit_does_not_change_split_input_hash(tmp_path, monkeypatch):
    provenance_path = tmp_path / "dataset_provenance.json"
    _write_fake_provenance(provenance_path)
    monkeypatch.setattr(config, "DATASET_PROVENANCE_PATH", provenance_path)
    module_paths = _fake_module_paths(tmp_path)

    unrelated = tmp_path / "sanity.py"
    unrelated.write_text("# unrelated module, not part of the split\n", encoding="utf-8")

    hash_before = split.compute_split_input_hash(module_paths)
    unrelated.write_text("# unrelated module, EDITED\n", encoding="utf-8")
    hash_after = split.compute_split_input_hash(module_paths)

    assert hash_before == hash_after, "editing a module outside module_paths must never change the hash"


def test_seed_change_invalidates_split_input_hash(tmp_path, monkeypatch):
    provenance_path = tmp_path / "dataset_provenance.json"
    _write_fake_provenance(provenance_path)
    monkeypatch.setattr(config, "DATASET_PROVENANCE_PATH", provenance_path)
    module_paths = _fake_module_paths(tmp_path)

    hash_before = split.compute_split_input_hash(module_paths)
    monkeypatch.setattr(config, "SEED", config.SEED + 1)
    hash_after = split.compute_split_input_hash(module_paths)

    assert hash_before != hash_after, "changing config.SEED must invalidate the split input hash"


def test_hamming_threshold_change_invalidates_split_input_hash(tmp_path, monkeypatch):
    provenance_path = tmp_path / "dataset_provenance.json"
    _write_fake_provenance(provenance_path)
    monkeypatch.setattr(config, "DATASET_PROVENANCE_PATH", provenance_path)
    module_paths = _fake_module_paths(tmp_path)

    hash_before = split.compute_split_input_hash(module_paths)
    monkeypatch.setattr(config, "DEDUPE_HAMMING_THRESHOLD", config.DEDUPE_HAMMING_THRESHOLD + 1)
    hash_after = split.compute_split_input_hash(module_paths)

    assert hash_before != hash_after, (
        "changing config.DEDUPE_HAMMING_THRESHOLD must invalidate the split input hash"
    )


def test_split_producing_module_content_change_invalidates_split_input_hash(tmp_path, monkeypatch):
    provenance_path = tmp_path / "dataset_provenance.json"
    _write_fake_provenance(provenance_path)
    monkeypatch.setattr(config, "DATASET_PROVENANCE_PATH", provenance_path)
    module_paths = _fake_module_paths(tmp_path)

    hash_before = split.compute_split_input_hash(module_paths)
    module_paths[0].write_text("# EDITED content\n", encoding="utf-8")
    hash_after = split.compute_split_input_hash(module_paths)

    assert hash_before != hash_after, (
        "editing one of the actual split-producing modules must invalidate the split input hash"
    )
