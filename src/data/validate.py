"""Validation checks shared by src/data/download.py (and reusable by any
other script that needs to sanity-check a downloaded dataset directory).
Every check here is a hard pass/fail against config.py constants — never a
fuzzy or weakened check (CLAUDE.md rule 1).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config  # noqa: E402


def count_images(directory: Path) -> int:
    """Counts only image files (by extension, case-insensitive) — never
    directories, never non-image files.
    """
    if not directory.is_dir():
        return 0
    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in config.IMAGE_EXTENSIONS
    )


def validate_plantvillage_dir(path: Path) -> tuple:
    """Returns (is_valid, detail). Image count must fall inside
    config.PLANTVILLAGE_EXPECTED_IMAGE_COUNT_RANGE; class folders must match
    config.PLANTVILLAGE_CLASS_NAMES exactly (hard failure, never weakened).
    """
    if not path.is_dir():
        return False, f"{path} does not exist."

    image_count = count_images(path)
    class_dirs = sorted(p.name for p in path.iterdir() if p.is_dir())
    lo, hi = config.PLANTVILLAGE_EXPECTED_IMAGE_COUNT_RANGE
    expected_classes = sorted(config.PLANTVILLAGE_CLASS_NAMES)

    count_ok = lo <= image_count <= hi
    classes_ok = class_dirs == expected_classes

    if count_ok and classes_ok:
        return True, f"{image_count} images across {len(class_dirs)} classes"

    detail = (
        f"found {image_count} images (expected {lo}-{hi}) across "
        f"{len(class_dirs)} class folders (expected exactly {config.NUM_CLASSES})"
    )
    if not classes_ok:
        missing = sorted(set(expected_classes) - set(class_dirs))
        extra = sorted(set(class_dirs) - set(expected_classes))
        detail += f"; missing classes: {missing}; unexpected classes: {extra}"
    return False, detail


def validate_plantdoc_dirs(train_dir: Path, test_dir: Path) -> tuple:
    """Returns (is_valid, detail). Image count (train+test combined) must
    fall inside config.PLANTDOC_EXPECTED_IMAGE_COUNT_RANGE.
    """
    if not train_dir.is_dir() or not test_dir.is_dir():
        return False, f"{train_dir} and/or {test_dir} do not exist."

    image_count = count_images(train_dir) + count_images(test_dir)
    lo, hi = config.PLANTDOC_EXPECTED_IMAGE_COUNT_RANGE
    if lo <= image_count <= hi:
        return True, f"{image_count} images across train/ and test/"
    return False, f"found {image_count} images across train/ and test/, expected {lo}-{hi}"
