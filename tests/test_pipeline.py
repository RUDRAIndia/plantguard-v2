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
    """Points the train- and val-side disk caches at throwaway tmp dirs
    instead of the real repo's config.TRAIN_DECODE_CACHE_DIR /
    VAL_DECODE_CACHE_DIR, so running these tests never writes cache files
    into the working tree.
    """
    monkeypatch_module_scoped.setattr(
        config, "TRAIN_DECODE_CACHE_DIR", tmp_path_factory.mktemp("train_cache")
    )
    monkeypatch_module_scoped.setattr(
        config, "VAL_DECODE_CACHE_DIR", tmp_path_factory.mktemp("val_cache")
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
    # Same order the real pipeline uses: decode (uint8) -> _to_float -> the
    # rest — feeding raw uint8 straight into augment.apply()/resize would
    # implicitly cast (255 -> 255.0) rather than rescale (255 -> 1.0),
    # silently testing the wrong numeric range.
    decoded = pipeline._to_float(pipeline._decode(tf.constant(str(sample_path))))
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


def test_decode_yields_uint8_not_float32(pipeline_env):
    color_dir = pipeline_env["color_dir"]
    sample_path = next(color_dir.rglob("*.jpg"))
    decoded = pipeline._decode(tf.constant(str(sample_path)))
    assert decoded.dtype == tf.uint8


def test_cached_element_dtype_is_uint8_and_matches_measured_file_size(pipeline_env, tmp_path):
    """The whole point of this fix: the on-disk cache must store uint8, not
    float32 (~4x smaller). Checked two ways — the dtype flowing through the
    exact decode->cache steps _build_pipeline uses, and the real cache
    file's measured byte size on disk against the projection formula
    _assert_cache_will_fit uses (CLAUDE.md: measure, don't assume).
    """
    color_dir = pipeline_env["color_dir"]
    sample_paths = sorted(str(p) for p in color_dir.rglob("*.jpg"))[:6]
    labels = list(range(len(sample_paths)))

    ds = tf.data.Dataset.from_tensor_slices((sample_paths, labels))
    ds = ds.map(lambda p, y: (pipeline._decode(p), y))
    cache_prefix = tmp_path / "cache" / "probe"
    cache_prefix.parent.mkdir(parents=True)
    ds = ds.cache(str(cache_prefix))

    assert ds.element_spec[0].dtype == tf.uint8

    for _ in ds:  # force the cache to actually materialize on disk
        pass

    data_files = list(cache_prefix.parent.glob(f"{cache_prefix.name}.data-*"))
    assert len(data_files) == 1, f"expected exactly one cache data file, found {data_files}"
    measured_bytes = data_files[0].stat().st_size

    projected_bytes = pipeline._projected_uint8_cache_bytes(sample_paths)
    # Real measurement (probed separately): tf.data's per-record framing
    # overhead on a uint8 cache is well under 1% of the raw pixel bytes, not
    # a multiple of it — this would fail hard if the cache were still
    # storing float32 (measured_bytes would be ~4x projected_bytes).
    assert projected_bytes <= measured_bytes <= projected_bytes * 1.05


def test_post_cache_pipeline_yields_float32_in_unit_range(pipeline_env):
    """Confirms _to_float's move to after the cache/shuffle didn't change
    what callers actually receive: build_datasets() with preprocess_fn
    effectively disabled (via a real backbone, but reading the pre-
    preprocess stage isn't exposed publicly, so this instead checks
    _build_pipeline directly with preprocess_fn=None — build_
    visualization_datasets' own documented contract) for both the train
    and val/test code paths.
    """
    splits = pipeline_env["splits"]
    train_paths, train_labels = pipeline._paths_and_labels(splits["train"][:4])
    val_paths, val_labels = pipeline._paths_and_labels(splits["val"][:4])

    train_ds = pipeline._build_pipeline(
        train_paths, train_labels, training=True, batch_size=4, cache_prefix=None
    )
    val_ds = pipeline._build_pipeline(
        val_paths, val_labels, training=False, batch_size=4, cache_prefix=None
    )

    for ds in (train_ds, val_ds):
        images, _ = next(iter(ds))
        assert images.dtype == tf.float32
        assert float(tf.reduce_min(images)) >= 0.0
        assert float(tf.reduce_max(images)) <= 1.0


def test_assert_cache_will_fit_raises_loudly_when_disk_is_too_small(pipeline_env, tmp_path, monkeypatch):
    import shutil as shutil_module

    color_dir = pipeline_env["color_dir"]
    sample_paths = sorted(str(p) for p in color_dir.rglob("*.jpg"))[:4]
    cache_prefix = tmp_path / "cache" / "train"

    fake_usage = shutil_module.disk_usage(tmp_path)._replace(free=1)  # ~0 bytes free
    monkeypatch.setattr(pipeline.shutil, "disk_usage", lambda path: fake_usage)

    with pytest.raises(RuntimeError, match="Refusing to build the on-disk decode cache"):
        pipeline._assert_cache_will_fit(sample_paths, cache_prefix)


def test_assert_cache_will_fit_passes_with_ample_disk_space(pipeline_env, tmp_path):
    color_dir = pipeline_env["color_dir"]
    sample_paths = sorted(str(p) for p in color_dir.rglob("*.jpg"))[:4]
    cache_prefix = tmp_path / "cache" / "train"

    pipeline._assert_cache_will_fit(sample_paths, cache_prefix)  # must not raise


def test_train_decode_cache_enabled_false_writes_no_cache_files(pipeline_env, tmp_path, monkeypatch):
    # A dedicated, guaranteed-empty cache dir for this test -- pipeline_env's
    # TRAIN_DECODE_CACHE_DIR is module-scoped and shared with earlier tests
    # that legitimately do cache into it, so reusing it here would just be
    # checking pre-existing files rather than this test's own run.
    monkeypatch.setattr(config, "TRAIN_DECODE_CACHE_DIR", tmp_path / "empty_train_cache")
    monkeypatch.setattr(config, "TRAIN_DECODE_CACHE_ENABLED", False)

    train_ds, _val_ds, _test_ds, _ = pipeline.build_datasets(MODEL_NAME, batch_size=4)
    for _ in train_ds:  # fully consume -- a cache, if any, would be written by now
        pass

    assert not config.TRAIN_DECODE_CACHE_DIR.exists(), (
        "TRAIN_DECODE_CACHE_ENABLED=False must skip the train cache entirely"
    )


def test_sample_paths_deterministic_and_bounded():
    paths = [f"ClassA/img{i}.jpg" for i in range(50)]

    first = pipeline._sample_paths(paths, n=10, seed=config.SEED)
    second = pipeline._sample_paths(paths, n=10, seed=config.SEED)

    assert len(first) == 10
    assert len(set(first)) == 10
    assert set(first) <= set(paths)
    assert first == second

    with pytest.raises(ValueError):
        pipeline._sample_paths(paths, n=51, seed=config.SEED)


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
