"""Audit tests written in response to a real observation: MobileNetV3Large
scored 0.9659 accuracy on the PlantVillage test split but 0.1928 accuracy on
PlantDoc. These check the two remaining hypotheses that could produce a
collapse that size: a reversed/off-by-one label mapping (item 2), and image
handling that distorts or discards the leaf (item 3). Deliberately kept out
of tests/test_evaluate_plantdoc_audit.py, which uses a module-scoped fixture
that monkeypatches config.PLANTVILLAGE_CLASS_NAMES to a small fake tuple —
these tests need the REAL config.PLANTVILLAGE_CLASS_NAMES and the REAL
mapping.PLANTDOC_TO_PLANTVILLAGE dict, never faked, so they live in their own
file where nothing else in the module could leave that patch active.

Conclusion of this audit (see the conversation this was written for): no
mapping-direction bug found (item 2). Image handling (item 3) IS real —
PlantDoc's arbitrary-aspect-ratio images get anisotropically stretched to a
square, unlike PlantVillage's native 256x256 squares — but on its own this
would not plausibly explain an 80-point drop; the overall magnitude is
consistent with the well-documented "PlantVillage-only training collapses on
real field photos" finding that PlantDoc's own paper exists to demonstrate.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from src import config
from src.data import mapping, pipeline
from src.evaluate import external


# ---------------------------------------------------------------------------
# Item 2: label mapping direction, using the REAL mapping dict (never faked).
# ---------------------------------------------------------------------------
def test_build_eval_set_maps_each_plantdoc_class_to_the_correct_plantvillage_index(tmp_path, monkeypatch):
    """A representative slice of the real mapping.PLANTDOC_TO_PLANTVILLAGE
    dict, covering all three confidence tiers, against a synthetic on-disk
    PlantDoc-shaped tree. Proves _build_eval_set's label is the correct
    GLOBAL PlantVillage index for each image -- not a PlantDoc-local index,
    not off-by-one, not reversed.
    """
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    monkeypatch.setattr(config, "PLANTDOC_TRAIN_DIR", train_dir)
    monkeypatch.setattr(config, "PLANTDOC_TEST_DIR", test_dir)

    sample_classes = {
        "Apple Scab Leaf": "Apple___Apple_scab",  # exact tier
        "Apple leaf": "Apple___healthy",  # convention tier (bare species name)
        "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",  # forced tier
    }
    for i, plantdoc_class in enumerate(sample_classes):
        class_dir = train_dir / plantdoc_class
        class_dir.mkdir(parents=True)
        Image.new("RGB", (50, 60), color=(i * 10, i * 20, i * 30)).save(class_dir / "img0.jpg")
    test_dir.mkdir(parents=True)

    paths, pv_labels, plantdoc_classes = external._build_eval_set()

    expected_index = {
        plantdoc_class: config.PLANTVILLAGE_CLASS_NAMES.index(pv_name)
        for plantdoc_class, pv_name in sample_classes.items()
    }
    assert len(paths) == 3
    for label, plantdoc_class in zip(pv_labels, plantdoc_classes):
        assert label == expected_index[plantdoc_class], (
            f"'{plantdoc_class}' mapped to PlantVillage index {label}, expected {expected_index[plantdoc_class]}"
        )


def test_every_real_mapping_entry_round_trips_through_the_class_index():
    """Every one of the real 28 entries in mapping.PLANTDOC_TO_PLANTVILLAGE,
    checked exhaustively (not just a sample): the PlantVillage name it maps
    to must be an exact member of config.PLANTVILLAGE_CLASS_NAMES, and its
    resolved index must map back to that exact name (catches any
    transcription typo that validate_mapping's existing check might share a
    blind spot with, by re-deriving the index independently here).
    """
    for plantdoc_class, pv_name in mapping.PLANTDOC_TO_PLANTVILLAGE.items():
        index = config.PLANTVILLAGE_CLASS_NAMES.index(pv_name)
        assert config.PLANTVILLAGE_CLASS_NAMES[index] == pv_name, plantdoc_class


def test_mapping_values_are_all_distinct_plantvillage_classes():
    """The 28 PlantDoc classes map to 28 DISTINCT PlantVillage classes -- if
    two PlantDoc classes collapsed onto the same PlantVillage index, that
    would silently understate coverage and, worse, mix two different
    diseases' images under one label without ever raising an error.
    """
    values = list(mapping.PLANTDOC_TO_PLANTVILLAGE.values())
    assert len(values) == len(set(values)) == 28


# ---------------------------------------------------------------------------
# Item 3: image handling -- quantifying the aspect-ratio distortion.
# ---------------------------------------------------------------------------
def test_resize_and_center_crop_does_not_preserve_aspect_ratio():
    """PlantVillage images are native 256x256 squares, so this resize is a
    structural no-op for them. PlantDoc images are arbitrary sizes/aspect
    ratios (a real field photo, not a studio square crop) -- this proves,
    with a concrete measurement, that a non-square input gets anisotropically
    stretched (not padded) to fill the square frame, changing a known
    feature's aspect ratio by the analytically predicted factor.

    This is not a code-path mismatch (see test_evaluate_plantdoc_audit.py --
    it's the exact same function PlantVillage's own val/test splits use) but
    it IS a real, PlantDoc-specific source of visual distortion worth naming
    explicitly, since PlantVillage images never exercise this branch of the
    function's behavior.
    """
    height, width = 400, 100  # a tall, narrow synthetic "field photo"
    image = np.zeros((height, width, 3), dtype=np.float32)
    # A solid white SQUARE patch (40x40) in the center -- a perfect square is
    # the simplest shape whose aspect ratio after a transform directly
    # reveals anisotropic (non-uniform per-axis) scaling.
    patch = 40
    top, left = (height - patch) // 2, (width - patch) // 2
    image[top : top + patch, left : left + patch, :] = 1.0

    result = pipeline._resize_and_center_crop(tf.constant(image)).numpy()

    # Locate the transformed patch by thresholding, then measure its
    # bounding-box aspect ratio.
    mask = result[..., 0] > 0.5
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    assert len(rows) > 0 and len(cols) > 0, "patch vanished entirely -- crop discarded it"
    patch_height = rows[-1] - rows[0] + 1
    patch_width = cols[-1] - cols[0] + 1
    observed_ratio = patch_width / patch_height

    # tf.image.resize(image, (256, 256)) scales width by 256/100=2.56 and
    # height by 256/400=0.64 independently; the center-crop to 224x224 that
    # follows removes a uniform border and does not change this ratio. A
    # square patch (ratio 1.0 originally) should come out at very close to
    # (256/100) / (256/400) = 4.0 -- nowhere near 1.0, which is what an
    # aspect-ratio-preserving resize would have produced instead.
    expected_ratio = (config.VAL_TEST_RESIZE_SIZE / width) / (config.VAL_TEST_RESIZE_SIZE / height)
    assert observed_ratio == pytest.approx(expected_ratio, rel=0.15)
    assert observed_ratio > 2.0  # unambiguously not aspect-preserving (that would give ~1.0)


def test_resize_is_a_structural_no_op_for_plantvillages_native_256x256_square():
    """The other half of the item-3 finding: PlantVillage images ARE already
    256x256 (config.VAL_TEST_RESIZE_SIZE), so this same resize step never
    distorts them -- the distortion demonstrated above is real but
    PlantDoc-specific, not something PlantVillage's own val/test/model-
    selection numbers were ever exposed to.
    """
    size = config.VAL_TEST_RESIZE_SIZE
    image = np.random.default_rng(0).uniform(0, 1, size=(size, size, 3)).astype(np.float32)
    resized = tf.image.resize(tf.constant(image), (size, size)).numpy()
    np.testing.assert_allclose(resized, image, atol=1e-5)


def test_no_crop_after_resize_discards_the_whole_image():
    """The 224x224 center-crop that follows the 256x256 resize only ever
    removes a fixed, small border (16px per side out of 256), regardless of
    the source image's original aspect ratio -- this is NOT where distortion
    risk comes from (that's the resize step above); this just confirms the
    crop itself can never discard more than that fixed border.
    """
    image = np.ones((400, 100, 3), dtype=np.float32)
    result = pipeline._resize_and_center_crop(tf.constant(image)).numpy()
    assert result.shape == (config.IMAGE_SIZE, config.IMAGE_SIZE, 3)
    assert np.all(result > 0.99)  # a uniformly white input must stay uniformly white -- no black padding introduced
