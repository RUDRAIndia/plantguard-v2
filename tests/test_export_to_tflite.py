"""Tests for src/export/to_tflite.py against a real (untrained) MobileNetV2
checkpoint and the synthetic PlantVillage-shaped fixture from
tests/conftest.py — exercises real code paths (real checkpoint save/load,
real tf.lite.TFLiteConverter INT8/float16 conversion) rather than mocking
them, since the conversion mechanics (graph-wrapping for quantization,
uint8-vs-float32 I/O) are exactly what's risky here. Never asserts on
prediction *quality* (the model is untrained) — only on shape, dtype, and
control-flow correctness (deploy vs. raise, staged vs. untouched).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from tensorflow import keras

from src import config
from src.export import metadata, to_tflite, verify_tflite


@pytest.fixture
def fresh_export_targets(export_env, tmp_path, monkeypatch):
    """Per-test (function-scoped) mutable export targets: a fresh
    TFLITE_OUTPUT_DIR, fresh Android asset paths seeded with sentinel
    placeholder content, and a fresh results.json referencing export_env's
    (module-scoped, read-only) checkpoint. Never the real
    android/app/src/main/assets/. A fresh copy per test, unlike export_env's
    checkpoint, because src.export.to_tflite.export() mutates all four of
    these paths -- sharing them across tests would make each test after the
    first see whatever the previous export() call already wrote.
    """
    tflite_output_dir = tmp_path / "tflite_staging"
    android_dir = tmp_path / "android_assets"
    android_dir.mkdir(parents=True)
    android_metadata_path = android_dir / "model_metadata.json"
    android_tflite_path = android_dir / "model.tflite"
    results_json_path = tmp_path / "results.json"

    monkeypatch.setattr(config, "TFLITE_OUTPUT_DIR", tflite_output_dir)
    monkeypatch.setattr(config, "ANDROID_MODEL_METADATA_PATH", android_metadata_path)
    monkeypatch.setattr(config, "ANDROID_MODEL_TFLITE_PATH", android_tflite_path)
    monkeypatch.setattr(config, "RESULTS_JSON_PATH", results_json_path)
    # A small representative-dataset size so conversion stays fast against
    # the tiny synthetic train split (which has far fewer than the real 200).
    monkeypatch.setitem(config.TFLITE_CONFIG, "representative_dataset_size", 8)
    monkeypatch.setattr(config, "TFLITE_LATENCY_WARMUP_RUNS", 1)
    monkeypatch.setattr(config, "TFLITE_LATENCY_MEASURED_RUNS", 3)

    placeholder_metadata = {
        "schema_version": 1,
        "placeholder": True,
        "confidence_threshold": 0.6,
        "class_names": [],
    }
    android_metadata_path.write_text(json.dumps(placeholder_metadata), encoding="utf-8")
    android_tflite_path.write_bytes(b"PLACEHOLDER")

    results = {
        "selected_model": export_env["model_name"],
        "git_commit_hash": "a" * 40,
        "model_selection": {
            "ranking": [
                {
                    "model_name": export_env["model_name"],
                    "val_macro_f1": 0.95,
                    "num_val_samples": 1,
                    "param_count": export_env["param_count"],
                    "checkpoint_file_size_bytes": export_env["checkpoint_file_size_bytes"],
                    "checkpoint_mtime": export_env["checkpoint_mtime"],
                }
            ]
        },
        "calibration": {"temperature": 1.36},
        "ood_rejection": {"chosen_threshold": 0.98},
    }
    results_json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    return {
        "tflite_output_dir": tflite_output_dir,
        "android_metadata_path": android_metadata_path,
        "android_tflite_path": android_tflite_path,
        "results_json_path": results_json_path,
        "placeholder_metadata_text": android_metadata_path.read_text(encoding="utf-8"),
        "placeholder_tflite_bytes": b"PLACEHOLDER",
    }


def test_select_representative_paths_draws_only_from_train_split(synthetic_dataset):
    splits = synthetic_dataset["splits"]
    train_paths = splits["train"]
    val_test_paths = set(splits["val"]) | set(splits["test"])

    selected = to_tflite._select_representative_paths(train_paths, n=5, seed=config.SEED)

    assert len(selected) == 5
    assert set(selected) <= set(train_paths)
    assert not (set(selected) & val_test_paths)

    selected_again = to_tflite._select_representative_paths(train_paths, n=5, seed=config.SEED)
    assert selected == selected_again  # deterministic given the same seed


def test_wrap_with_preprocessing_shapes_and_softmax(export_env):
    model = export_env["model"]
    model_name = export_env["model_name"]

    export_model = to_tflite._wrap_with_preprocessing(model, model_name)

    assert export_model.input_shape == (None,) + config.IMAGE_SHAPE
    assert export_model.output_shape == (None, config.NUM_CLASSES)

    raw_batch = np.random.default_rng(0).uniform(0, 255, size=(1,) + config.IMAGE_SHAPE).astype(np.float32)
    probs = np.asarray(export_model(raw_batch, training=False))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-4)


def test_convert_int8_produces_uint8_io_contract(export_env, synthetic_dataset):
    export_model = to_tflite._wrap_with_preprocessing(export_env["model"], export_env["model_name"])
    representative_paths = to_tflite._select_representative_paths(
        synthetic_dataset["splits"]["train"], n=4, seed=config.SEED
    )

    tflite_bytes = to_tflite._convert_int8(export_model, to_tflite._representative_dataset(representative_paths))
    to_tflite._assert_uint8_io_contract(tflite_bytes, config.NUM_CLASSES)  # must not raise

    interpreter = verify_tflite.load_interpreter(tflite_bytes)
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    assert input_detail["dtype"] == np.uint8
    assert output_detail["dtype"] == np.uint8
    assert tuple(output_detail["shape"]) == (1, config.NUM_CLASSES)


def test_convert_float16_keeps_float32_io(export_env):
    export_model = to_tflite._wrap_with_preprocessing(export_env["model"], export_env["model_name"])

    tflite_bytes = to_tflite._convert_float16(export_model)

    interpreter = verify_tflite.load_interpreter(tflite_bytes)
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    assert input_detail["dtype"] == np.float32
    assert output_detail["dtype"] == np.float32


def test_assert_class_names_integrity_fires_on_reordered_class_list(export_env):
    broken = list(reversed(config.PLANTVILLAGE_CLASS_NAMES))
    with pytest.raises(RuntimeError, match="class_names"):
        metadata.assert_class_names_integrity(broken, export_env["model"])


def test_assert_class_names_integrity_fires_on_wrong_output_dimension():
    wrong_model = keras.Sequential([keras.layers.Input(shape=(4,)), keras.layers.Dense(2)])
    with pytest.raises(RuntimeError, match="output dimension"):
        metadata.assert_class_names_integrity(config.PLANTVILLAGE_CLASS_NAMES, wrong_model)


def test_export_deploys_when_drop_within_tolerance(fresh_export_targets, monkeypatch):
    monkeypatch.setattr(config, "TFLITE_MAX_MACRO_F1_DROP", 1.0)  # guarantee within tolerance

    result = to_tflite.export()

    assert result["deployment_outcome"] == "deployed"
    assert result["int8"]["deployed_to_android"] is True
    assert result["float16"] is None

    deployed_bytes = fresh_export_targets["android_tflite_path"].read_bytes()
    assert deployed_bytes != fresh_export_targets["placeholder_tflite_bytes"]

    written_metadata = json.loads(fresh_export_targets["android_metadata_path"].read_text(encoding="utf-8"))
    assert written_metadata["placeholder"] is False
    assert written_metadata["class_names"] == list(config.PLANTVILLAGE_CLASS_NAMES)
    assert written_metadata["confidence_threshold"] == 0.98
    assert written_metadata["calibration_temperature"] == 1.36
    assert written_metadata["architecture"] == "MobileNetV2"
    assert written_metadata["quantization"] == "int8"
    assert len(written_metadata["git_commit_hash"]) == 40


def test_export_raises_and_leaves_placeholder_when_drop_exceeds_tolerance(fresh_export_targets, monkeypatch):
    monkeypatch.setattr(config, "TFLITE_MAX_MACRO_F1_DROP", -1.0)  # guarantee out of tolerance

    with pytest.raises(RuntimeError, match="macro-F1 drop"):
        to_tflite.export()

    assert fresh_export_targets["android_tflite_path"].read_bytes() == fresh_export_targets["placeholder_tflite_bytes"]
    assert (
        fresh_export_targets["android_metadata_path"].read_text(encoding="utf-8")
        == fresh_export_targets["placeholder_metadata_text"]
    )
    assert (fresh_export_targets["tflite_output_dir"] / config.TFLITE_FLOAT16_FILENAME).is_file()
    assert (fresh_export_targets["tflite_output_dir"] / config.TFLITE_INT8_FILENAME).is_file()

    results = json.loads(fresh_export_targets["results_json_path"].read_text(encoding="utf-8"))
    export_section = results["export"]
    assert export_section["deployment_outcome"] == "raised_for_human_decision"
    assert export_section["float16"] is not None
    assert export_section["int8"]["within_tolerance"] is False
    assert export_section["int8"]["deployed_to_android"] is False
    assert export_section["float16"]["deployed_to_android"] is False


def test_export_results_json_round_trips_full_export_schema(fresh_export_targets, monkeypatch):
    monkeypatch.setattr(config, "TFLITE_MAX_MACRO_F1_DROP", 1.0)

    to_tflite.export()

    results = json.loads(fresh_export_targets["results_json_path"].read_text(encoding="utf-8"))
    export_section = results["export"]

    assert export_section["selected_model"] == "MobileNetV2"
    assert len(export_section["git_commit_hash"]) == 40
    assert export_section["representative_dataset"]["size"] == 8
    assert export_section["representative_dataset"]["source_split"] == "train"
    assert export_section["representative_dataset"]["seed"] == config.SEED

    int8 = export_section["int8"]
    assert int8["file_size_mb"] > 0
    assert 0.0 <= int8["val_macro_f1_quantized"] <= 1.0
    assert int8["latency_ms"]["num_measured_runs"] == config.TFLITE_LATENCY_MEASURED_RUNS
    assert int8["latency_ms"]["min_ms"] <= int8["latency_ms"]["mean_ms"] <= int8["latency_ms"]["max_ms"]

    # Other results.json keys the fixture seeded must still be present/untouched.
    assert results["selected_model"] == "MobileNetV2"
    assert results["calibration"]["temperature"] == 1.36


def test_export_raises_before_deploy_when_results_json_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_JSON_PATH", tmp_path / "does_not_exist.json")
    with pytest.raises(FileNotFoundError):
        to_tflite.export()
