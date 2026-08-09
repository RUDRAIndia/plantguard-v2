"""Tests for src/data/fetch.py's post-extraction directory resolution."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import fetch


def _touch_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def test_resolves_through_a_doubled_wrapper_directory(tmp_path):
    # Reproduces Kaggle's intel-image-classification real layout:
    # seg_train/seg_train/<category>/*.jpg.
    outer = tmp_path / "seg_train"
    inner = outer / "seg_train"
    for category in ("buildings", "forest", "glacier", "mountain", "sea", "street"):
        _touch_image(inner / category / "1.jpg")

    resolved = fetch._resolve_through_wrapper_dirs(outer)
    assert resolved == inner


def test_resolves_through_multiple_nested_wrapper_levels(tmp_path):
    root = tmp_path / "a"
    leaf = root / "b" / "c"
    _touch_image(leaf / "category_one" / "1.jpg")
    _touch_image(leaf / "category_two" / "1.jpg")

    resolved = fetch._resolve_through_wrapper_dirs(root)
    assert resolved == leaf


def test_stops_immediately_when_already_at_the_category_level(tmp_path):
    root = tmp_path / "already_resolved"
    _touch_image(root / "buildings" / "1.jpg")
    _touch_image(root / "forest" / "1.jpg")

    resolved = fetch._resolve_through_wrapper_dirs(root)
    assert resolved == root


def test_stops_at_a_directory_that_has_images_directly_inside(tmp_path):
    root = tmp_path / "flat_category"
    _touch_image(root / "1.jpg")
    _touch_image(root / "2.jpg")

    resolved = fetch._resolve_through_wrapper_dirs(root)
    assert resolved == root
