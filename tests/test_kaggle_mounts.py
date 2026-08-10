"""Tests for the Kaggle-mount acquisition paths in src/data/download.py and
src/data/negatives.py: both datasets are simulated as read-only notebook
inputs under tmp_path (config.IS_KAGGLE monkeypatched True), proving
validation runs for real against the mount, provenance is recorded with a
"kaggle-input-mount:" source, and — the whole point of a read-only mount —
nothing is ever written into it.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src import config
from src.data import download, fetch, negatives, validate


def _populate_plantvillage_mount(root: Path, images_per_class: int) -> Path:
    for class_name in config.PLANTVILLAGE_CLASS_NAMES:
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        for i in range(images_per_class):
            (class_dir / f"img{i}.jpg").write_bytes(b"x")
    return root


def _populate_negatives_mount(root: Path, images_per_category: int) -> Path:
    # Real doubled nesting: seg_train/seg_train/<category>/...
    inner = root / "seg_train"
    for category in config.NEGATIVES_EXPECTED_CATEGORIES:
        category_dir = inner / category
        category_dir.mkdir(parents=True)
        for i in range(images_per_category):
            (category_dir / f"img{i}.jpg").write_bytes(b"x")
    return root


@pytest.fixture
def kaggle_env(tmp_path, monkeypatch):
    """Points every Kaggle-relevant config path at tmp_path, so real
    filesystem I/O happens against a throwaway fixture rather than a real
    mount or the repo's working tree.
    """
    monkeypatch.setattr(config, "IS_KAGGLE", True)
    monkeypatch.setattr(config, "IS_COLAB", False)

    data_root = tmp_path / "working" / "data"
    monkeypatch.setattr(config, "DATA_ROOT", data_root)
    monkeypatch.setattr(config, "NEGATIVES_DIR", data_root / "negatives")
    monkeypatch.setattr(config, "NEGATIVES_TAR", None)
    monkeypatch.setattr(config, "PLANTVILLAGE_COLOR_TAR", None)

    artifacts_dir = tmp_path / "working" / "artifacts"
    monkeypatch.setattr(config, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(config, "DATASET_PROVENANCE_PATH", artifacts_dir / "dataset_provenance.json")

    return tmp_path


def test_download_plantvillage_validates_kaggle_mount_without_writing_to_it(kaggle_env, monkeypatch):
    lo, _hi = config.PLANTVILLAGE_EXPECTED_IMAGE_COUNT_RANGE
    images_per_class = (lo // config.NUM_CLASSES) + 2
    mount = _populate_plantvillage_mount(kaggle_env / "mount" / "color", images_per_class)
    monkeypatch.setattr(config, "KAGGLE_PLANTVILLAGE_COLOR_DIR", mount)
    monkeypatch.setattr(config, "PLANTVILLAGE_COLOR_DIR", mount)

    before = {p.name: sorted(q.name for q in p.iterdir()) for p in mount.iterdir()}

    result = download.download_plantvillage()

    assert result == mount
    after = {p.name: sorted(q.name for q in p.iterdir()) for p in mount.iterdir()}
    assert before == after  # nothing added, removed, or moved inside the mount

    provenance = json.loads(config.DATASET_PROVENANCE_PATH.read_text(encoding="utf-8"))
    assert provenance["plantvillage"]["source"] == f"kaggle-input-mount:{mount}"


def test_download_plantvillage_raises_loudly_on_a_broken_kaggle_mount(kaggle_env, monkeypatch):
    # Only 2 of the 38 real classes -- a deliberately broken mount.
    mount = kaggle_env / "mount" / "color"
    for class_name in list(config.PLANTVILLAGE_CLASS_NAMES)[:2]:
        class_dir = mount / class_name
        class_dir.mkdir(parents=True)
        (class_dir / "img0.jpg").write_bytes(b"x")
    monkeypatch.setattr(config, "KAGGLE_PLANTVILLAGE_COLOR_DIR", mount)
    monkeypatch.setattr(config, "PLANTVILLAGE_COLOR_DIR", mount)

    with pytest.raises(RuntimeError, match="Re-attach"):
        download.download_plantvillage()


def test_resolve_negatives_mount_raises_loudly_when_not_attached(kaggle_env, monkeypatch):
    monkeypatch.setattr(config, "KAGGLE_NEGATIVES_SEG_TRAIN_DIR", kaggle_env / "not_attached" / "seg_train")

    with pytest.raises(RuntimeError, match="Re-attach"):
        fetch.resolve_negatives_mount()


def test_plantdoc_guard_allows_kaggle_since_it_still_git_clones_there(kaggle_env):
    # PlantDoc is NOT a Kaggle input (only PlantVillage and negatives are),
    # so download_plantdoc() must still be permitted to run on Kaggle
    # (it git-clones to config.PLANTDOC_DIR, which resolves under
    # /kaggle/working) -- this only checks the guard, not a real clone.
    download._require_colab_or_kaggle("PlantDoc")  # must not raise


def test_download_negatives_resolves_kaggle_mount_and_subsamples_without_writing_to_it(kaggle_env, monkeypatch):
    per_category = (config.NEGATIVES_TARGET_COUNT // len(config.NEGATIVES_EXPECTED_CATEGORIES)) + 5
    mount = _populate_negatives_mount(kaggle_env / "mount" / "seg_train", per_category)
    monkeypatch.setattr(config, "KAGGLE_NEGATIVES_SEG_TRAIN_DIR", mount)

    inner = mount / "seg_train"
    before = {p.name: len(list(p.iterdir())) for p in inner.iterdir()}

    result = negatives.download_negatives()

    assert result == config.NEGATIVES_DIR
    assert validate.count_images(config.NEGATIVES_DIR) == config.NEGATIVES_TARGET_COUNT

    after = {p.name: len(list(p.iterdir())) for p in inner.iterdir()}
    assert before == after  # the mount's category directories are untouched

    provenance = json.loads(config.DATASET_PROVENANCE_PATH.read_text(encoding="utf-8"))
    assert provenance["negatives"]["source"] == f"kaggle-input-mount:{mount}"
