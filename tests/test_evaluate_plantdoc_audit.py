"""Audit test written in response to a real observation: MobileNetV3Large
scored 0.9659 accuracy on the PlantVillage test split but 0.1928 accuracy on
PlantDoc. A collapse that size is exactly what a preprocessing mismatch would
produce, so this proves — BY CONSTRUCTION, not by re-reading the code — that
PlantDoc's evaluation pipeline and PlantVillage's own val/test pipeline are
not just similar but the literal same code path.

This uses the synthetic_dataset fixture from tests/conftest.py, which
monkeypatches config.PLANTVILLAGE_CLASS_NAMES to a small fake 3-class tuple
for the whole module's duration (module-scoped fixture) — kept in its own
file, separate from tests/test_evaluate_plantdoc_mapping_and_resize_audit.py,
specifically so that patch can never leak into tests that need the real
config.PLANTVILLAGE_CLASS_NAMES / mapping.PLANTDOC_TO_PLANTVILLAGE (a
cross-file leak is impossible — module-scoped fixtures are scoped per test
file — but a same-file leak across an entire module's test run is exactly
the trap this split avoids).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import tensorflow as tf

from src import config
from src.data import pipeline
from src.evaluate import inference


@pytest.fixture(scope="module")
def pipeline_env(synthetic_dataset, tmp_path_factory, monkeypatch_module_scoped):
    # Own tmp cache dirs, same reasoning as tests/test_pipeline.py's
    # pipeline_env: never write into the real repo's cache directories.
    monkeypatch_module_scoped.setattr(
        config, "TRAIN_DECODE_CACHE_DIR", tmp_path_factory.mktemp("train_cache")
    )
    monkeypatch_module_scoped.setattr(
        config, "VAL_DECODE_CACHE_DIR", tmp_path_factory.mktemp("val_cache")
    )
    return synthetic_dataset


@pytest.mark.parametrize("model_name", config.CANDIDATE_MODELS)
def test_plantdoc_eval_pipeline_is_bit_identical_to_the_val_pipeline(pipeline_env, model_name):
    """external.py's PlantDoc evaluation calls inference.build_eval_pipeline;
    src/train.py's validation (and src/evaluate/model_selection.py's
    recomputed val macro-F1) both go through pipeline.build_datasets' val
    branch. This feeds the SAME real on-disk images through both real public
    entrypoints and asserts the preprocessed tensors are bit-for-bit
    identical — for every config.CANDIDATE_MODELS backbone, including
    MobileNetV3Large, the model under audit.
    """
    splits = pipeline_env["splits"]
    val_paths, val_labels = pipeline._paths_and_labels(splits["val"])

    _, val_ds, _, _ = pipeline.build_datasets(model_name, batch_size=4)
    plantdoc_style_ds = inference.build_eval_pipeline(val_paths, val_labels, model_name)

    val_images = tf.concat([images for images, _ in val_ds], axis=0).numpy()
    val_labels_seen = tf.concat([labels for _, labels in val_ds], axis=0).numpy()
    plantdoc_images = tf.concat([images for images, _ in plantdoc_style_ds], axis=0).numpy()
    plantdoc_labels_seen = tf.concat([labels for _, labels in plantdoc_style_ds], axis=0).numpy()

    np.testing.assert_array_equal(val_labels_seen, plantdoc_labels_seen)  # same order, prerequisite for the next check
    np.testing.assert_array_equal(val_images, plantdoc_images)


def test_build_eval_pipeline_resolves_preprocessing_the_same_way_build_datasets_does():
    """Even more directly: both call the exact same private resolver."""
    assert inference.pipeline._resolve_preprocess_fn is pipeline._resolve_preprocess_fn
    for model_name in config.CANDIDATE_MODELS:
        # Resolving twice for the same model_name must yield the same
        # underlying function object (import-cached), not two different
        # wrappers that happen to look alike.
        fn_a = pipeline._resolve_preprocess_fn(model_name)
        fn_b = pipeline._resolve_preprocess_fn(model_name)
        assert fn_a is fn_b
