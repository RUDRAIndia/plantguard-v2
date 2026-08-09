"""Builds a leakage-free train/val/test split of PlantVillage color.

Reads artifacts/image_groups.json (written by src/data/dedupe.py, which
MUST run first) rather than raw images: splitting happens at the GROUP
level, not the image level, so that every photo of the same physical leaf
(a duplicate/near-duplicate cluster) lands entirely inside one split. A
naive per-image random split would put near-duplicates of one leaf into
both train and test, inflating test accuracy on memorization rather than
generalization — this is CLAUDE.md's "near-duplicate leakage" failure mode.

Within that grouping constraint, the split is also stratified per class:
each class's groups are allocated to try to hit CLAUDE.md's configured
70/15/15 (config.TRAIN_FRACTION/VAL_FRACTION/TEST_FRACTION) as closely as
the grouping allows. A class with very few duplicate-groups literally
cannot be split into exact proportions — see _allocate_class for the
explicit feasibility check and how the split degrades gracefully (never
silently) when it can't.

Determinism: allocation is seeded per-class from config.SEED, independent
of iteration/processing order, so the same input (config.SEED +
image_groups.json) always produces byte-identical output regardless of
dict/set iteration order (which Python's hash randomization can otherwise
vary between runs) — every collection that feeds randomness or output
order is explicitly sorted first.
"""

import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config  # noqa: E402

SPLIT_NAMES = ("train", "val", "test")


def load_image_groups() -> dict[str, str]:
    """Reads artifacts/image_groups.json (relative_path -> group_id)."""
    path = config.ARTIFACTS_DIR / "image_groups.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run src/data/dedupe.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def class_of(relative_path: str) -> str:
    """Class is the first path component, by PlantVillage's own
    <class>/<file> directory convention (same convention used throughout
    src/data/dedupe.py and src/data/inventory.py).
    """
    return relative_path.split("/")[0]


def _build_class_groups(image_to_group: dict[str, str]) -> dict[str, dict[str, int]]:
    """Returns {class_name: {group_id: image_count}}. A group can only
    belong to one class since src/data/dedupe.py never merges clusters
    across class boundaries.
    """
    class_groups: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for path, group_id in image_to_group.items():
        class_groups[class_of(path)][group_id] += 1
    return class_groups


def _allocate_class(class_name: str, groups: dict[str, int]) -> dict[str, str]:
    """Returns {group_id: split_name} for one class's groups.

    Raises immediately, naming the class, if it has fewer than 3 distinct
    groups — with fewer than 3 groups it is mathematically impossible to
    give all three splits at least one image of this class, since a group
    can't be torn apart to cover more than one split (CLAUDE.md rule 1:
    fail loudly rather than produce a silently broken split).

    Otherwise: reserves the 3 smallest groups (least distortion) to
    guarantee every split gets >= 1 group, then greedily assigns the
    remaining groups (in a per-class-seeded deterministic shuffle) to
    whichever split currently has the largest deficit against its target
    count, so achieved proportions land as close to config.TRAIN_FRACTION /
    VAL_FRACTION / TEST_FRACTION as the group sizes allow.
    """
    if len(groups) < 3:
        raise RuntimeError(
            f"Class '{class_name}' has only {len(groups)} duplicate-group(s) "
            f"after dedupe — cannot place at least one group in each of "
            f"train/val/test. Groups: {sorted(groups)}. This class needs "
            "more distinct physical leaves before a 3-way grouped split is "
            "possible."
        )

    total = sum(groups.values())
    target = {
        "test": round(total * config.TEST_FRACTION),
        "val": round(total * config.VAL_FRACTION),
    }
    target["train"] = total - target["test"] - target["val"]

    ordered = sorted(groups.items(), key=lambda kv: (kv[1], kv[0]))  # (size, group_id) ascending
    reserved, remaining = ordered[:3], ordered[3:]

    assignment: dict[str, str] = {}
    current = {"train": 0, "val": 0, "test": 0}

    # Spend the 3 smallest groups on guaranteeing coverage — least possible
    # distortion of the achieved proportions.
    for (group_id, size), split_name in zip(reserved, ("test", "val", "train")):
        assignment[group_id] = split_name
        current[split_name] += size

    rng = random.Random(f"{config.SEED}:{class_name}")
    remaining_sorted = sorted(remaining, key=lambda kv: kv[0])  # deterministic pre-shuffle order
    rng.shuffle(remaining_sorted)

    for group_id, size in remaining_sorted:
        # Relative deficit (fraction of target still unmet), not absolute —
        # absolute deficit systematically favors train (whose target is by
        # far the largest number), starving val/test of groups whenever a
        # tie-breaking or close call comes up. Relative deficit balances
        # fairly across splits with very different target sizes.
        split_name = max(SPLIT_NAMES, key=lambda s: (target[s] - current[s]) / max(target[s], 1))
        assignment[group_id] = split_name
        current[split_name] += size

    return assignment


def build_split(image_to_group: dict[str, str]) -> dict[str, list[str]]:
    """Pure, deterministic core: image_to_group -> {"train": [...], "val":
    [...], "test": [...]} (each list sorted). Validates the result before
    returning it.
    """
    class_groups = _build_class_groups(image_to_group)

    group_to_split: dict[str, str] = {}
    for class_name in sorted(class_groups):
        group_to_split.update(_allocate_class(class_name, class_groups[class_name]))

    splits: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    for path, group_id in image_to_group.items():
        splits[group_to_split[group_id]].append(path)
    for split_name in splits:
        splits[split_name].sort()

    _validate_split(splits, image_to_group)
    return splits


def _validate_split(splits: dict[str, list[str]], image_to_group: dict[str, str]) -> None:
    """Re-derives every fact from `splits` itself (not from intermediate
    bookkeeping) and asserts loudly on the first violation found, naming the
    offending image/group/class.
    """
    path_to_split: dict[str, str] = {}
    for split_name, paths in splits.items():
        for path in paths:
            if path in path_to_split:
                raise AssertionError(
                    f"Image '{path}' appears in both '{path_to_split[path]}' "
                    f"and '{split_name}' splits."
                )
            path_to_split[path] = split_name

    if set(path_to_split) != set(image_to_group):
        missing = sorted(set(image_to_group) - set(path_to_split))
        extra = sorted(set(path_to_split) - set(image_to_group))
        raise AssertionError(
            f"Split image set doesn't match input image set. Missing: "
            f"{missing[:10]}. Extra: {extra[:10]}."
        )

    total_in_splits = sum(len(paths) for paths in splits.values())
    if total_in_splits != len(image_to_group):
        raise AssertionError(
            f"Total images across splits ({total_in_splits}) != input total "
            f"({len(image_to_group)})."
        )

    group_splits_seen: dict[str, set] = defaultdict(set)
    for path, split_name in path_to_split.items():
        group_splits_seen[image_to_group[path]].add(split_name)
    spanning = {gid: sorted(s) for gid, s in group_splits_seen.items() if len(s) > 1}
    if spanning:
        raise AssertionError(f"{len(spanning)} group_id(s) span multiple splits: {spanning}.")

    class_splits_seen: dict[str, set] = defaultdict(set)
    for path, split_name in path_to_split.items():
        class_splits_seen[class_of(path)].add(split_name)
    missing_entirely = sorted(set(config.PLANTVILLAGE_CLASS_NAMES) - set(class_splits_seen))
    if missing_entirely:
        raise AssertionError(f"Classes with zero images in the split entirely: {missing_entirely}.")
    incomplete = {
        cls: sorted(set(SPLIT_NAMES) - seen)
        for cls, seen in class_splits_seen.items()
        if seen != set(SPLIT_NAMES)
    }
    if incomplete:
        raise AssertionError(f"Classes missing from at least one split: {incomplete}.")


# The only modules whose logic actually determines build_split's output —
# NOT the whole repo. A commit to any other file (e.g. sanity.py, a
# pipeline.py cache tweak) must never invalidate an otherwise-still-valid
# split; see compute_split_inputs below and src/data/pipeline.py's
# load_splits, which recomputes and compares this at load time.
_SPLIT_PRODUCING_MODULE_NAMES = ("dedupe.py", "phash_cluster.py", "split.py")


def _default_split_producing_module_paths() -> tuple:
    directory = Path(__file__).resolve().parent
    return tuple(directory / name for name in _SPLIT_PRODUCING_MODULE_NAMES)


def compute_split_inputs(module_paths: tuple = None) -> dict:
    """Returns a dict of exactly the inputs that determine build_split's
    output, broken out by name (not just one opaque combined hash) so a
    mismatch can be diagnosed field-by-field later:
      - config.SEED, TRAIN_FRACTION, VAL_FRACTION, TEST_FRACTION,
        PHASH_SIZE, DEDUPE_HAMMING_THRESHOLD
      - the recorded PlantVillage dataset provenance (image/class counts
        and the sha256 of its sorted relative file paths) — i.e. which
        files exist, never *when* or *how* they were downloaded
      - the sha256 of each module in `module_paths` (default:
        dedupe.py, phash_cluster.py, split.py — the only code that
        actually produces the split). Overridable so tests can point this
        at throwaway files instead of hashing/mutating real repo source.

    Raises if PlantVillage's provenance hasn't been recorded yet
    (src/data/download.py must run first) — this hash is meaningless
    without knowing which dataset it was computed against.
    """
    if module_paths is None:
        module_paths = _default_split_producing_module_paths()

    provenance_path = config.DATASET_PROVENANCE_PATH
    if not provenance_path.is_file():
        raise FileNotFoundError(
            f"{provenance_path} not found. Run src/data/download.py first — "
            "the split input hash needs PlantVillage's recorded provenance."
        )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if "plantvillage" not in provenance:
        raise RuntimeError(
            f"{provenance_path} has no 'plantvillage' entry. Run "
            "src/data/download.py first."
        )
    pv_entry = provenance["plantvillage"]

    return {
        "seed": config.SEED,
        "train_fraction": config.TRAIN_FRACTION,
        "val_fraction": config.VAL_FRACTION,
        "test_fraction": config.TEST_FRACTION,
        "phash_size": config.PHASH_SIZE,
        "dedupe_hamming_threshold": config.DEDUPE_HAMMING_THRESHOLD,
        "plantvillage_provenance": {
            "image_count": pv_entry["observed_image_count"],
            "class_count": pv_entry["observed_class_count"],
            "sha256_of_sorted_relative_paths": pv_entry["sha256_of_sorted_relative_paths"],
        },
        "module_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in module_paths
        },
    }


def hash_split_inputs(split_inputs: dict) -> str:
    """Pure: canonical-JSON-then-sha256 of a compute_split_inputs() dict, so
    the same inputs always hash identically regardless of dict key order."""
    return hashlib.sha256(json.dumps(split_inputs, sort_keys=True).encode("utf-8")).hexdigest()


def compute_split_input_hash(module_paths: tuple = None) -> str:
    """Convenience: hash_split_inputs(compute_split_inputs(module_paths))."""
    return hash_split_inputs(compute_split_inputs(module_paths))
