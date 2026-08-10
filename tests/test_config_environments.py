"""Tests for src/config.py's three-way environment detection (IS_KAGGLE,
IS_COLAB, local) and the paths that derive from it. Each test reloads
src.config under a patched signal (a fake /kaggle/input, a fake
"google.colab" module) so the module's own top-level detection code runs
for real, then reloads it back to the real environment's values in a
`finally` — src.config is a single shared module object every other test
file also imports, so leaving it mutated would leak into unrelated tests.
"""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config as config


def _reload_with_fake_kaggle_input(monkeypatch, present: bool):
    real_is_dir = Path.is_dir
    kaggle_input = Path("/kaggle/input")

    def fake_is_dir(self):
        if self == kaggle_input:
            return present
        return real_is_dir(self)

    monkeypatch.setattr(Path, "is_dir", fake_is_dir)
    importlib.reload(config)


def test_is_kaggle_detected_by_input_mount_presence(monkeypatch):
    try:
        _reload_with_fake_kaggle_input(monkeypatch, present=True)

        assert config.IS_KAGGLE is True
        assert config.IS_COLAB is False
        assert config.DATA_ROOT == Path("/kaggle/working/data")
        assert config.DRIVE_DATA_ROOT is None
        assert config.PLANTVILLAGE_COLOR_DIR == config.KAGGLE_PLANTVILLAGE_COLOR_DIR
        assert config.PLANTVILLAGE_COLOR_DIR == Path(
            "/kaggle/input/datasets/abdallahalidev/plantvillage-dataset/color"
        )
        assert config.KAGGLE_NEGATIVES_SEG_TRAIN_DIR == Path(
            "/kaggle/input/datasets/puneet6060/intel-image-classification/seg_train"
        )
        assert config.CHECKPOINT_DIR == Path("/kaggle/working/data/checkpoints")
        # No Drive on Kaggle -> nothing to persist a tar copy of.
        assert config.PLANTVILLAGE_COLOR_TAR is None
        assert config.PLANTDOC_TAR is None
        assert config.NEGATIVES_TAR is None
        assert config.DRIVE_PROVENANCE_PATH is None
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_kaggle_input_mount_takes_precedence_over_colab_signal(monkeypatch):
    """Even if "google.colab" were somehow importable, a present
    /kaggle/input must still win — src/config.py documents this exact
    precedence (the more specific, more reliable signal is checked first).
    """
    monkeypatch.setitem(sys.modules, "google.colab", object())
    try:
        _reload_with_fake_kaggle_input(monkeypatch, present=True)
        assert config.IS_KAGGLE is True
        assert config.IS_COLAB is False
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_no_kaggle_input_falls_back_to_local_when_colab_signal_absent(monkeypatch):
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    try:
        _reload_with_fake_kaggle_input(monkeypatch, present=False)

        assert config.IS_KAGGLE is False
        assert config.IS_COLAB is False
        assert config.DATA_ROOT == config.REPO_ROOT / "data"
        assert config.DRIVE_DATA_ROOT is None
        assert config.PLANTVILLAGE_COLOR_DIR == config.PLANTVILLAGE_DIR / "color"
        assert config.KAGGLE_PLANTVILLAGE_COLOR_DIR is None
        assert config.KAGGLE_NEGATIVES_SEG_TRAIN_DIR is None
    finally:
        monkeypatch.undo()
        importlib.reload(config)
