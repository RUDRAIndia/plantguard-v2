"""Tests for src/data/split.py's grouped/stratified allocation, run against
the synthetic dataset built by the `synthetic_dataset` fixture in
tests/conftest.py (see that file's docstring for why synthetic, not real,
data).
"""

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
