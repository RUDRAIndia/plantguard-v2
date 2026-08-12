"""Verifies an exported LiteRT (TFLite) model's predictions match the source
Keras model within tolerance, and that embedded metadata is present and
correct before the model is considered ready for the Android app.

**What "the source Keras model" means here.** Rather than re-running the
float Keras model over the validation split a second time in this module,
the float baseline (`val_macro_f1_float`) is passed in by the caller —
already computed, from the same validation split, by
`src/evaluate/model_selection.py`'s real recompute-from-checkpoint pass, and
recorded in `artifacts/results.json`. This avoids a redundant full
validation-set forward pass and matches how the number is actually reported:
"the float and quantized macro-F1 side by side," not "the quantized macro-F1
against a number this module happens to compute again."

**Raw-pixel evaluation.** `src/export/to_tflite.py` bakes each backbone's
`preprocess_input` INTO the exported graph (see that module's docstring for
why), so the exported `.tflite`'s own input tensor expects raw `[0, 255]`
pixel values — exactly what the Android app feeds it (no client-side
normalization, per `android/README.md`). This module's validation-set
pipeline therefore builds the *same* deterministic resize-then-center-crop,
no-backbone-preprocessing images `src/export/to_tflite.py`'s representative
dataset uses, never the backbone-preprocessed pipeline
`src/evaluate/inference.py` uses for the float model.

**Dequantization never uses a hardcoded formula** — every quantized
tensor's scale/zero-point is read from the interpreter itself
(`get_input_details()`/`get_output_details()`'s `"quantization"` field) at
call time, exactly mirroring how the Android app's
`PlantClassifier.kt`/`ImagePreprocessing.kt` do it on-device.
"""

import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config  # noqa: E402
from src.data import pipeline  # noqa: E402


def load_interpreter(tflite_source) -> tf.lite.Interpreter:
    """Accepts either a Path (reads the file) or raw bytes — a single
    chokepoint so orchestration code and tests build interpreters
    identically.
    """
    tflite_bytes = Path(tflite_source).read_bytes() if isinstance(tflite_source, (str, Path)) else tflite_source
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    return interpreter


def _assert_input_output_dtype(interpreter: tf.lite.Interpreter, expected_dtype) -> None:
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    if input_detail["dtype"] != expected_dtype or output_detail["dtype"] != expected_dtype:
        raise RuntimeError(
            f"Expected {expected_dtype} input AND output tensors, got "
            f"input={input_detail['dtype']} output={output_detail['dtype']}. The converter "
            "did not produce the expected I/O contract for this quantization scheme."
        )


def _prepare_input(image_float_0_255: np.ndarray, input_detail: dict) -> np.ndarray:
    """Converts a raw [0, 255] float32 image batch to whatever dtype the
    interpreter's input tensor actually expects. For a quantized (uint8)
    tensor, uses THIS tensor's own scale/zero-point
    (`quantized = real / scale + zero_point`, the inverse of TFLite's own
    dequantization formula) — never an assumed scale of 1.0, even though
    that's what a raw-pixel-range representative dataset happens to produce.
    """
    dtype = input_detail["dtype"]
    if dtype == np.uint8:
        scale, zero_point = input_detail["quantization"]
        quantized = np.round(image_float_0_255 / scale + zero_point)
        return np.clip(quantized, 0, 255).astype(np.uint8)
    return image_float_0_255.astype(dtype)


def _dequantize_output(raw_output: np.ndarray, output_detail: dict) -> np.ndarray:
    dtype = output_detail["dtype"]
    if dtype == np.uint8:
        scale, zero_point = output_detail["quantization"]
        return (raw_output.astype(np.float32) - zero_point) * scale
    return raw_output.astype(np.float32)


def _raw_pixel_dataset(paths: list, labels: list) -> tf.data.Dataset:
    """Deterministic resize+center-crop, no backbone preprocessing (that's
    baked into the exported graph instead), no augmentation, batch size 1 —
    raw [0, 255] float32 pixel values, one image per element to match a
    TFLite interpreter's fixed single-image input (the same one-image-at-a-
    time contract Android itself uses).
    """
    ds = pipeline._build_pipeline(
        paths, labels, training=False, batch_size=1, preprocess_fn=None, shuffle=False, cache_prefix=None
    )
    return ds.map(lambda x, y: (x * 255.0, y))


def predict_tflite(interpreter: tf.lite.Interpreter, paths: list, labels: list) -> tuple:
    """Runs `interpreter` over every (path, label) pair, one image at a
    time, and returns (y_true, y_prob) numpy arrays in `paths`' order —
    mirrors src/evaluate/inference.py's predict_dataset() contract, but
    against a TFLite interpreter instead of a Keras model.
    """
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    y_true, y_prob = [], []
    for image_batch, label_batch in _raw_pixel_dataset(paths, labels):
        prepared = _prepare_input(image_batch.numpy(), input_detail)
        interpreter.set_tensor(input_detail["index"], prepared)
        interpreter.invoke()
        raw_output = interpreter.get_tensor(output_detail["index"])
        y_prob.append(_dequantize_output(raw_output, output_detail)[0])
        y_true.append(int(label_batch.numpy()[0]))

    if not y_true:
        raise RuntimeError("No images to evaluate -- the given val split produced zero samples.")
    return np.array(y_true), np.array(y_prob)


def macro_f1_from_probs(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int) -> float:
    y_pred = y_prob.argmax(axis=1)
    return float(f1_score(y_true, y_pred, labels=list(range(num_classes)), average="macro", zero_division=0))


def measure_latency(interpreter: tf.lite.Interpreter, sample_image: np.ndarray, warmup_runs: int, measured_runs: int) -> dict:
    """Single-image CPU inference latency: `warmup_runs` untimed invocations
    (interpreter/cache warmup) discarded, then `measured_runs` invocations
    individually timed via time.perf_counter() around each invoke() call —
    the same single-image, wall-clock-around-invoke pattern
    android/README.md documents for the on-device measurement.
    """
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    prepared = _prepare_input(sample_image, input_detail)

    for _ in range(warmup_runs):
        interpreter.set_tensor(input_detail["index"], prepared)
        interpreter.invoke()
        interpreter.get_tensor(output_detail["index"])

    durations_ms = []
    for _ in range(measured_runs):
        start = time.perf_counter()
        interpreter.set_tensor(input_detail["index"], prepared)
        interpreter.invoke()
        interpreter.get_tensor(output_detail["index"])
        durations_ms.append((time.perf_counter() - start) * 1000.0)

    durations = np.array(durations_ms)
    return {
        "num_warmup_runs": warmup_runs,
        "num_measured_runs": measured_runs,
        "mean_ms": float(durations.mean()),
        "median_ms": float(np.median(durations)),
        "std_ms": float(durations.std()),
        "min_ms": float(durations.min()),
        "max_ms": float(durations.max()),
    }


def _verify(tflite_path: Path, val_relative_paths: list, model_name: str, val_macro_f1_float: float, *, quantization: str, expected_dtype) -> dict:
    print(f"[verify_tflite] Evaluating {quantization} export for '{model_name}' on {len(val_relative_paths)} validation images...")
    interpreter = load_interpreter(tflite_path)
    _assert_input_output_dtype(interpreter, expected_dtype)

    val_paths, val_labels = pipeline._paths_and_labels(val_relative_paths)
    y_true, y_prob = predict_tflite(interpreter, val_paths, val_labels)
    macro_f1 = macro_f1_from_probs(y_true, y_prob, config.NUM_CLASSES)
    drop = val_macro_f1_float - macro_f1

    sample_image, _ = next(iter(_raw_pixel_dataset(val_paths[:1], val_labels[:1])))
    latency = measure_latency(
        interpreter, sample_image.numpy(), config.TFLITE_LATENCY_WARMUP_RUNS, config.TFLITE_LATENCY_MEASURED_RUNS
    )

    result = {
        "quantization": quantization,
        "val_macro_f1_float": val_macro_f1_float,
        "val_macro_f1_quantized": macro_f1,
        "val_macro_f1_drop": drop,
        "num_val_samples": int(len(y_true)),
        "max_allowed_drop": config.TFLITE_MAX_MACRO_F1_DROP,
        "within_tolerance": drop <= config.TFLITE_MAX_MACRO_F1_DROP,
        "latency_ms": latency,
    }
    print(
        f"[verify_tflite]   float={val_macro_f1_float:.4f} quantized={macro_f1:.4f} "
        f"drop={drop:.4f} (max allowed {config.TFLITE_MAX_MACRO_F1_DROP:.4f})"
    )
    return result


def verify_int8(tflite_path: Path, val_relative_paths: list, model_name: str, val_macro_f1_float: float) -> dict:
    return _verify(
        tflite_path, val_relative_paths, model_name, val_macro_f1_float, quantization="int8", expected_dtype=np.uint8
    )


def verify_float16(tflite_path: Path, val_relative_paths: list, model_name: str, val_macro_f1_float: float) -> dict:
    return _verify(
        tflite_path,
        val_relative_paths,
        model_name,
        val_macro_f1_float,
        quantization="float16",
        expected_dtype=np.float32,
    )
