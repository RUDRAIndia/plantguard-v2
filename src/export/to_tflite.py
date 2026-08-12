"""Converts a trained Keras 3 model to LiteRT (TFLite) with INT8 quantization
for on-device inference. Embeds metadata: architecture, class_names in exact
index order, image size, preprocessing, seed, git commit hash, val metrics.

The Android app (android/) was built ahead of this exporter against a
metadata contract this module must conform to: full-integer quantization
with uint8 input AND uint8 output (config.TFLITE_CONFIG), class_names copied
verbatim from config.PLANTVILLAGE_CLASS_NAMES in order, and a sibling
model_metadata.json (schema documented in android/README.md) carrying
image_size, confidence_threshold, and that same class_names array. The real
export must overwrite both android/app/src/main/assets/model.tflite and
model_metadata.json together — see scripts/generate_placeholder_tflite.py
for the placeholder that currently stands in for this module's output.

**Preprocessing must be baked into the exported graph.** src/models/build.py
does NOT include the backbone's preprocess_input in the Keras graph —
src/data/pipeline.py applies it externally, and for MobileNetV3 (unlike
EfficientNet, whose preprocess_input is close to a no-op) that's a real,
non-trivial rescale. Converting the trained model as-is would produce a
.tflite whose input tensor expects already-preprocessed values, breaking
Android's "raw uint8 pixels, no client-side normalization" contract
(model_metadata.json's input_format field). _wrap_with_preprocessing wraps
the trained model with a Lambda(preprocess_fn) ahead of it, so the
quantized graph's exposed input tensor corresponds to raw [0, 255] pixels —
exactly what android/app/src/main/java/.../ImagePreprocessing.kt feeds it.

**Stage, validate, then swap (CLAUDE.md rule 13).** Nothing under
android/app/src/main/assets/ is touched until the INT8 candidate's
validation-split macro-F1 drop (src/export/verify_tflite.py, against the
SAME split src/evaluate/model_selection.py already scored the float model
on) is measured and found within config.TFLITE_MAX_MACRO_F1_DROP. If it
isn't, a float16 comparison-only artifact is also built and staged (never
deployed — standard TFLite float16 quantization keeps float32 I/O tensors,
not drop-in compatible with the app's uint8-only Kotlin code), both
candidates are recorded in artifacts/results.json's "export" section, and
export() raises — the placeholder is left completely untouched and the
failure is loud by design, never a warning to scroll past.

Run (Kaggle, after src.evaluate.runner.run() has written results.json):
    python -m src.export.to_tflite
"""

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config  # noqa: E402
from src.data import pipeline  # noqa: E402
from src.evaluate import inference  # noqa: E402
from src.export import metadata, verify_tflite  # noqa: E402


def _wrap_with_preprocessing(trained_model: keras.Model, model_name: str) -> keras.Model:
    """Wraps `trained_model` with its backbone's preprocess_input baked into
    the graph as a Lambda layer ahead of it — see module docstring for why
    this is required before conversion, not optional.
    """
    preprocess_fn = pipeline._resolve_preprocess_fn(model_name)
    raw_input = keras.Input(shape=config.IMAGE_SHAPE, dtype=tf.float32, name="raw_pixels_0_255")
    preprocessed = keras.layers.Lambda(preprocess_fn, name="preprocess")(raw_input)
    outputs = trained_model(preprocessed)
    return keras.Model(raw_input, outputs, name=f"{model_name}_export")


def _select_representative_paths(train_relative_paths: list, n: int, seed: int) -> list:
    """Thin wrapper over pipeline._sample_paths, kept as its own function so
    tests can assert train-only provenance directly against the call site
    rather than by convention: the caller must pass splits["train"], never
    splits["val"]/["test"] (CLAUDE.md rule 2's spirit extended to
    quantization calibration, not just evaluation).
    """
    return pipeline._sample_paths(train_relative_paths, n, seed)


def _representative_dataset(representative_paths: list):
    """Returns a zero-arg generator function for
    converter.representative_dataset: deterministic resize+center-crop, no
    backbone preprocessing (baked into the graph instead), no augmentation,
    raw [0, 255] float32 range — reuses verify_tflite's raw-pixel pipeline
    so calibration and validation-set scoring build images identically.
    """
    absolute_paths, labels = pipeline._paths_and_labels(representative_paths)

    def generator():
        for image_batch, _ in verify_tflite._raw_pixel_dataset(absolute_paths, labels):
            yield [image_batch.numpy()]

    return generator


def _convert_int8(export_model: keras.Model, representative_dataset_fn) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_fn
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    return converter.convert()


def _convert_float16(export_model: keras.Model) -> bytes:
    """Standard weight-only float16 quantization — no representative dataset
    needed, and (empirically confirmed) keeps float32 input/output tensors,
    unlike the INT8 path. Comparison-only: see module docstring.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    return converter.convert()


def _assert_uint8_io_contract(tflite_bytes: bytes, num_classes: int) -> None:
    """Fails fast, before spending minutes on a full validation pass against
    a possibly-malformed interpreter (CLAUDE.md rule 1).
    """
    interpreter = verify_tflite.load_interpreter(tflite_bytes)
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    if input_detail["dtype"] != np.uint8 or output_detail["dtype"] != np.uint8:
        raise RuntimeError(
            f"INT8 conversion did not produce uint8 input/output tensors "
            f"(input={input_detail['dtype']}, output={output_detail['dtype']}) -- the Android "
            "app assumes raw uint8 input and a uint8 output it dequantizes itself."
        )
    if tuple(output_detail["shape"]) != (1, num_classes):
        raise RuntimeError(
            f"INT8 export output shape {tuple(output_detail['shape'])} != (1, {num_classes}) -- "
            "class_names in model_metadata.json would silently misalign with the model's output "
            "indices."
        )


def _write_staged_artifact(tflite_bytes: bytes, filename: str) -> Path:
    config.TFLITE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.TFLITE_OUTPUT_DIR / filename
    path.write_bytes(tflite_bytes)
    return path


def _deploy_to_android(int8_staged_path: Path, metadata_dict: dict) -> None:
    """The only function permitted to touch android/app/src/main/assets/ —
    only ever called after the staged INT8 candidate has been verified
    within tolerance (CLAUDE.md rule 13).
    """
    if not config.ANDROID_MODEL_METADATA_PATH.parent.is_dir():
        raise FileNotFoundError(
            f"{config.ANDROID_MODEL_METADATA_PATH.parent} does not exist -- expected the "
            "placeholder assets committed alongside android/ (see android/README.md)."
        )
    shutil.copy2(int8_staged_path, config.ANDROID_MODEL_TFLITE_PATH)
    config.ANDROID_MODEL_METADATA_PATH.write_text(json.dumps(metadata_dict, indent=2) + "\n", encoding="utf-8")
    print(f"[to_tflite] Deployed {int8_staged_path} -> {config.ANDROID_MODEL_TFLITE_PATH}")
    print(f"[to_tflite] Wrote {config.ANDROID_MODEL_METADATA_PATH}")


def _load_results() -> dict:
    if not config.RESULTS_JSON_PATH.is_file():
        raise FileNotFoundError(
            f"{config.RESULTS_JSON_PATH} not found. Run src.evaluate.runner first -- "
            "src/export/to_tflite.py reads the selected model, calibration temperature, and "
            "OOD threshold from it, never hardcoding them (CLAUDE.md rule 5)."
        )
    return json.loads(config.RESULTS_JSON_PATH.read_text(encoding="utf-8"))


def _ranking_entry(results: dict, model_name: str) -> dict:
    ranking = results.get("model_selection", {}).get("ranking", [])
    for entry in ranking:
        if entry.get("model_name") == model_name:
            return entry
    raise RuntimeError(
        f"No model_selection.ranking entry for '{model_name}' in {config.RESULTS_JSON_PATH} -- "
        "results.json is internally inconsistent (selected model not present in its own ranking)."
    )


def _write_results_export_section(results: dict, export_section: dict) -> None:
    results["export"] = export_section
    config.RESULTS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.RESULTS_JSON_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[to_tflite] Wrote export section to {config.RESULTS_JSON_PATH}")


def export(model_name: str = None) -> dict:
    """Top-level orchestration. `model_name` defaults to
    results["selected_model"] — overridable for testing or for exporting a
    non-default candidate. Returns the export section written into
    results.json; raises (after writing everything staged/recorded) if the
    INT8 drop exceeds config.TFLITE_MAX_MACRO_F1_DROP.
    """
    results = _load_results()
    selected_model_name = model_name or results["selected_model"]
    print(f"[to_tflite] Exporting '{selected_model_name}'...")

    ranking_entry = _ranking_entry(results, selected_model_name)

    trained_model = inference.load_trained_model(selected_model_name)
    metadata.assert_class_names_integrity(config.PLANTVILLAGE_CLASS_NAMES, trained_model)
    export_model = _wrap_with_preprocessing(trained_model, selected_model_name)

    splits, _ = pipeline.load_splits()
    representative_size = config.TFLITE_CONFIG["representative_dataset_size"]
    representative_paths = _select_representative_paths(splits["train"], representative_size, config.SEED)

    print(f"[to_tflite] Converting to INT8 with {len(representative_paths)} representative train images...")
    int8_bytes = _convert_int8(export_model, _representative_dataset(representative_paths))
    _assert_uint8_io_contract(int8_bytes, config.NUM_CLASSES)
    int8_path = _write_staged_artifact(int8_bytes, config.TFLITE_INT8_FILENAME)
    int8_file_size_mb = len(int8_bytes) / 1e6

    val_macro_f1_float = ranking_entry["val_macro_f1"]
    int8_verification = verify_tflite.verify_int8(int8_path, splits["val"], selected_model_name, val_macro_f1_float)

    export_section = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "git_commit_hash": config.get_git_commit_hash(),
        "evaluated_at_git_commit_hash": results["git_commit_hash"],
        "selected_model": selected_model_name,
        "checkpoint_path": str(inference.checkpoint_weights_path(selected_model_name)),
        "representative_dataset": {
            "size": len(representative_paths),
            "source_split": "train",
            "seed": config.SEED,
            "note": (
                "Sampled via src/data/pipeline._sample_paths from splits['train'] only -- "
                "never splits['val'] or splits['test']."
            ),
        },
        "int8": {
            **int8_verification,
            "tflite_path": str(int8_path),
            "file_size_mb": int8_file_size_mb,
            "deployed_to_android": False,
        },
        "float16": None,
        "deployment_outcome": None,
    }

    if int8_verification["within_tolerance"]:
        metadata_dict = metadata.build_metadata(
            model_name=selected_model_name,
            class_names=config.PLANTVILLAGE_CLASS_NAMES,
            trained_model=trained_model,
            results=results,
            ranking_entry=ranking_entry,
            quantization="int8",
            tflite_file_size_mb=int8_file_size_mb,
            latency=int8_verification["latency_ms"],
            val_macro_f1_float=val_macro_f1_float,
            val_macro_f1_quantized=int8_verification["val_macro_f1_quantized"],
            representative_dataset_size=len(representative_paths),
        )
        _deploy_to_android(int8_path, metadata_dict)
        export_section["int8"]["deployed_to_android"] = True
        export_section["deployment_outcome"] = "deployed"
        _write_results_export_section(results, export_section)
        return export_section

    print(
        f"[to_tflite] INT8 macro-F1 drop {int8_verification['val_macro_f1_drop']:.4f} exceeds "
        f"config.TFLITE_MAX_MACRO_F1_DROP ({config.TFLITE_MAX_MACRO_F1_DROP:.4f}) -- NOT deploying to "
        "android/assets/. Building a float16 comparison artifact instead."
    )
    float16_bytes = _convert_float16(export_model)
    float16_path = _write_staged_artifact(float16_bytes, config.TFLITE_FLOAT16_FILENAME)
    float16_file_size_mb = len(float16_bytes) / 1e6
    float16_verification = verify_tflite.verify_float16(
        float16_path, splits["val"], selected_model_name, val_macro_f1_float
    )
    export_section["float16"] = {
        **float16_verification,
        "tflite_path": str(float16_path),
        "file_size_mb": float16_file_size_mb,
        "deployed_to_android": False,
        "note": (
            "Standard TFLite float16 quantization keeps float32 input/output tensors -- NOT "
            "drop-in compatible with android/app/src/main/java/.../PlantClassifier.kt's "
            "uint8-only contract. Deploying this would require Kotlin changes (input packing, "
            "output parsing, no self-dequantization) -- out of scope here, a real follow-up."
        ),
    }
    export_section["deployment_outcome"] = "raised_for_human_decision"
    _write_results_export_section(results, export_section)

    raise RuntimeError(
        f"INT8 export's validation macro-F1 drop ({int8_verification['val_macro_f1_drop']:.4f}) exceeds "
        f"the {config.TFLITE_MAX_MACRO_F1_DROP:.4f} tolerance -- refusing to silently deploy a degraded "
        f"model. float={val_macro_f1_float:.4f} int8={int8_verification['val_macro_f1_quantized']:.4f} "
        f"float16={float16_verification['val_macro_f1_quantized']:.4f}. Both candidates are staged at "
        f"{config.TFLITE_OUTPUT_DIR} and recorded in {config.RESULTS_JSON_PATH}'s 'export' section -- "
        "android/app/src/main/assets/ was left untouched. A human must decide: accept the INT8 drop and "
        "deploy it manually, or invest in Kotlin changes for float16."
    )


def main() -> None:
    export()


if __name__ == "__main__":
    main()
