"""Tests for src/data/pipeline.py, src/data/augment.py, and
src/data/negatives.py's pure subsampling logic, run against the synthetic
dataset built by the `synthetic_dataset` fixture in tests/conftest.py.
"""

import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import tensorflow as tf

from src import config
from src.data import augment, negatives, pipeline

MODEL_NAME = "MobileNetV2"


@pytest.fixture(scope="module")
def pipeline_env(synthetic_dataset, tmp_path_factory, monkeypatch_module_scoped):
    """Points the train-side disk cache at a throwaway tmp dir instead of
    the real repo's config.TRAIN_DECODE_CACHE_DIR, so running these tests
    never writes cache files into the working tree.
    """
    monkeypatch_module_scoped.setattr(
        config, "TRAIN_DECODE_CACHE_DIR", tmp_path_factory.mktemp("train_cache")
    )
    return synthetic_dataset


def _dataset_to_list(dataset: tf.data.Dataset) -> list:
    result = []
    for images, labels in dataset:
        for image, label in zip(images, labels):
            result.append((image.numpy().tobytes(), int(label)))
    return result


def test_val_and_test_pipelines_deterministic(pipeline_env):
    _, val_ds, test_ds, _ = pipeline.build_datasets(MODEL_NAME, batch_size=4)
    for ds in (val_ds, test_ds):
        first_pass = _dataset_to_list(ds)
        second_pass = _dataset_to_list(ds)
        assert first_pass == second_pass
        assert len(first_pass) > 0


def test_train_and_val_label_distributions_match_split_file(pipeline_env):
    train_ds, val_ds, _, _ = pipeline.build_datasets(MODEL_NAME, batch_size=4)
    class_index = {name: i for i, name in enumerate(config.PLANTVILLAGE_CLASS_NAMES)}
    splits = pipeline_env["splits"]

    for split_name, ds in (("train", train_ds), ("val", val_ds)):
        expected = Counter(class_index[path.split("/")[0]] for path in splits[split_name])
        actual = Counter()
        for _, labels in ds:
            for label in labels:
                actual[int(label)] += 1
        assert actual == expected, f"label distribution mismatch for '{split_name}'"


def test_class_weights_finite_positive_and_sum_sensibly(pipeline_env):
    splits = pipeline_env["splits"]
    weights = pipeline.compute_class_weights(splits["train"])

    assert all(math.isfinite(w) and w > 0 for w in weights.values())

    class_index = {name: i for i, name in enumerate(config.PLANTVILLAGE_CLASS_NAMES)}
    counts = Counter(path.split("/")[0] for path in splits["train"])
    total = sum(counts.values())
    weighted_sum = sum(weights[class_index[cls]] * n for cls, n in counts.items())
    assert abs(weighted_sum - total) < 1e-6


def test_augmentation_changes_pixels_but_val_pipeline_is_static(pipeline_env):
    color_dir = pipeline_env["color_dir"]
    sample_path = next(color_dir.rglob("*.jpg"))
    decoded = pipeline._decode(tf.constant(str(sample_path)))
    size = (config.IMAGE_SIZE, config.IMAGE_SIZE)

    augmented_a = augment.apply(decoded, size)
    augmented_b = augment.apply(decoded, size)
    assert not bool(tf.reduce_all(tf.equal(augmented_a, augmented_b))), (
        "two independent calls to augment.apply() on the same image produced "
        "identical pixels — augmentation isn't actually randomizing"
    )

    baseline = tf.image.resize(decoded, size)
    assert not bool(tf.reduce_all(tf.equal(augmented_a, baseline))), (
        "augment.apply() left the image unchanged from a plain deterministic resize"
    )

    val_a = pipeline._resize_and_center_crop(decoded)
    val_b = pipeline._resize_and_center_crop(decoded)
    assert bool(tf.reduce_all(tf.equal(val_a, val_b))), (
        "the deterministic val/test path produced different pixels across two calls"
    )


def test_negatives_deterministic_subsample(tmp_path):
    for category, count in (("a", 5), ("b", 5), ("c", 5)):
        class_dir = tmp_path / category
        class_dir.mkdir()
        for i in range(count):
            (class_dir / f"img{i}.jpg").write_bytes(b"x")

    class_dirs = [tmp_path / name for name in ("a", "b", "c")]

    first = negatives._deterministic_subsample(class_dirs, target_count=9, seed=config.SEED)
    second = negatives._deterministic_subsample(class_dirs, target_count=9, seed=config.SEED)

    assert len(first) == 9
    assert len(set(first)) == 9
    assert first == second

    with pytest.raises(RuntimeError):
        negatives._deterministic_subsample(class_dirs, target_count=100, seed=config.SEED)
