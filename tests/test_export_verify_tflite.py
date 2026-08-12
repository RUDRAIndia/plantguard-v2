"""Tests for src/export/verify_tflite.py: the pure macro-F1 computation
against hand-built arrays, and the TFLite-interpreter-driven pieces
(dequantization, latency measurement, verify_int8 orchestration) against a
real (untrained) MobileNetV2 checkpoint converted with
src/export/to_tflite.py's real converter recipes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from sklearn.metrics import f1_score

from src import config
from src.data import pipeline
from src.export import to_tflite, verify_tflite


def test_macro_f1_from_probs_matches_manual_sklearn_computation():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_prob = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.2, 0.7, 0.1],  # wrong: predicts class 1
            [0.1, 0.8, 0.1],
            [0.05, 0.9, 0.05],
            [0.1, 0.1, 0.8],
            [0.3, 0.3, 0.4],
        ]
    )
    expected = f1_score(y_true, y_prob.argmax(axis=1), labels=[0, 1, 2], average="macro", zero_division=0)

    result = verify_tflite.macro_f1_from_probs(y_true, y_prob, num_classes=3)

    assert result == pytest.approx(expected)


@pytest.fixture
def int8_tflite_bytes(export_env, synthetic_dataset, monkeypatch):
    """Real INT8 conversion of export_env's real (untrained) checkpoint,
    built once per test via the module-scoped export_env/synthetic_dataset
    fixtures -- read-only from here on, so safe to share across tests.
    """
    monkeypatch.setitem(config.TFLITE_CONFIG, "representative_dataset_size", 4)
    export_model = to_tflite._wrap_with_preprocessing(export_env["model"], export_env["model_name"])
    representative_paths = to_tflite._select_representative_paths(
        synthetic_dataset["splits"]["train"], n=4, seed=config.SEED
    )
    return to_tflite._convert_int8(export_model, to_tflite._representative_dataset(representative_paths))


def test_predict_tflite_dequantizes_using_tensor_own_scale_zero_point(int8_tflite_bytes, synthetic_dataset):
    interpreter = verify_tflite.load_interpreter(int8_tflite_bytes)
    val_relative_paths = synthetic_dataset["splits"]["val"][:2]
    val_paths, val_labels = pipeline._paths_and_labels(val_relative_paths)

    y_true, y_prob = verify_tflite.predict_tflite(interpreter, val_paths, val_labels)

    assert y_true.shape == (2,)
    assert y_prob.shape == (2, config.NUM_CLASSES)
    assert np.all(y_prob >= 0.0)

    # Manually reproduce the dequantization for the first sample and confirm
    # it matches -- proves the formula uses the interpreter's OWN
    # scale/zero-point, never a hardcoded one.
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    scale, zero_point = output_detail["quantization"]

    image_batch, _ = next(iter(verify_tflite._raw_pixel_dataset(val_paths[:1], val_labels[:1])))
    prepared = verify_tflite._prepare_input(image_batch.numpy(), input_detail)
    interpreter.set_tensor(input_detail["index"], prepared)
    interpreter.invoke()
    raw_output = interpreter.get_tensor(output_detail["index"])
    manual = (raw_output.astype(np.float32) - zero_point) * scale

    np.testing.assert_allclose(y_prob[0], manual[0])


def test_measure_latency_returns_consistent_positive_stats(int8_tflite_bytes):
    interpreter = verify_tflite.load_interpreter(int8_tflite_bytes)
    sample_image = np.zeros((1,) + config.IMAGE_SHAPE, dtype=np.float32)

    result = verify_tflite.measure_latency(interpreter, sample_image, warmup_runs=2, measured_runs=5)

    assert result["num_warmup_runs"] == 2
    assert result["num_measured_runs"] == 5
    assert 0.0 < result["min_ms"] <= result["mean_ms"] <= result["max_ms"]
    assert result["std_ms"] >= 0.0


def test_verify_int8_computes_drop_against_supplied_float_baseline(int8_tflite_bytes, synthetic_dataset, monkeypatch):
    monkeypatch.setattr(config, "TFLITE_LATENCY_WARMUP_RUNS", 1)
    monkeypatch.setattr(config, "TFLITE_LATENCY_MEASURED_RUNS", 2)

    result = verify_tflite.verify_int8(
        int8_tflite_bytes, synthetic_dataset["splits"]["val"], "MobileNetV2", val_macro_f1_float=0.95
    )

    assert result["val_macro_f1_drop"] == pytest.approx(0.95 - result["val_macro_f1_quantized"])
    assert result["within_tolerance"] == (result["val_macro_f1_drop"] <= config.TFLITE_MAX_MACRO_F1_DROP)
    assert result["num_val_samples"] == len(synthetic_dataset["splits"]["val"])
    assert result["quantization"] == "int8"


def test_verify_int8_raises_on_non_uint8_input(synthetic_dataset, export_env, monkeypatch):
    monkeypatch.setattr(config, "TFLITE_LATENCY_WARMUP_RUNS", 1)
    monkeypatch.setattr(config, "TFLITE_LATENCY_MEASURED_RUNS", 2)
    export_model = to_tflite._wrap_with_preprocessing(export_env["model"], export_env["model_name"])
    float16_bytes = to_tflite._convert_float16(export_model)

    with pytest.raises(RuntimeError, match="uint8"):
        verify_tflite.verify_int8(float16_bytes, synthetic_dataset["splits"]["val"], "MobileNetV2", 0.95)
