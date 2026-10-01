"""Converts a trained Keras 3 model to LiteRT (TFLite) for on-device
inference. Embeds metadata: architecture, class_names in exact index order,
image size, preprocessing, seed, git commit hash, val metrics.

**float16 is the deployed format; INT8 is comparison-only.** A real Kaggle
export run measured, on the same 8,146-image validation split
src/evaluate/model_selection.py already scored the float model on: INT8
macro-F1 0.5545 (a 0.4133 absolute drop from the float model's 0.9678 — the
known hard-swish/squeeze-excite INT8 quantization pathology for
MobileNetV3's architecture family, not something a larger
representative_dataset_size fixes), float16 macro-F1 0.9682 (effectively
lossless). Both formats are always built, verified, and recorded in
artifacts/results.json's "export" section — INT8's failure is itself a real
finding worth reporting — but only float16 is ever deployed:
android/app/src/main/java/.../PlantClassifier.kt now requires float32
input AND output tensors (config.ANDROID_TFLITE_IO_DTYPE) to match
float16's float32 I/O, so an INT8 candidate (uint8 I/O) is never even
eligible for deployment regardless of its own accuracy — see
_assert_deployable_for_android.

The Android app (android/) was built ahead of this exporter against a
metadata contract this module must conform to: class_names copied verbatim
from config.PLANTVILLAGE_CLASS_NAMES in order, and a sibling
model_metadata.json (schema documented in android/README.md) carrying
image_size, confidence_threshold, input_format, and that same class_names
array. The real export must overwrite both android/app/src/main/assets/
model.tflite and model_metadata.json together — see
scripts/generate_placeholder_tflite.py for the (now also float16) placeholder
that stands in for this module's output until a real export deploys.

**Preprocessing must be baked into the exported graph.** src/models/build.py
does NOT include the backbone's preprocess_input in the Keras graph —
src/data/pipeline.py applies it externally, and for MobileNetV3 (unlike
EfficientNet, whose preprocess_input is close to a no-op) that's a real,
non-trivial rescale. Converting the trained model as-is would produce a
.tflite whose input tensor expects already-preprocessed values, breaking
Android's "raw pixels, no client-side normalization" contract
(model_metadata.json's input_format field). _wrap_with_preprocessing wraps
the trained model with a Lambda(preprocess_fn) ahead of it, so the exported
graph's input tensor corresponds to raw [0, 255] pixels — exactly what
android/app/src/main/java/.../ImagePreprocessing.kt feeds it.

**Stage, validate, then swap (CLAUDE.md rule 13).** Nothing under
android/app/src/main/assets/ is touched until float16's validation-split
macro-F1 drop is measured and found within config.TFLITE_MAX_MACRO_F1_DROP
— staged first under artifacts/tflite/, deployed only after that check
passes. If it doesn't (hypothetically, for a future different selected
model), export() raises after writing both candidates' full verification to
results.json — the placeholder is left completely untouched and the
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
from src.export import metadata, results_io, verify_tflite  # noqa: E402


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


def _write_staged_artifact(tflite_bytes: bytes, filename: str) -> Path:
    config.TFLITE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.TFLITE_OUTPUT_DIR / filename
    path.write_bytes(tflite_bytes)
    return path


def _deploy_to_android(staged_path: Path, metadata_dict: dict) -> None:
    """The only function permitted to touch android/app/src/main/assets/ —
    only ever called after the staged candidate has been verified within
    tolerance (CLAUDE.md rule 13) AND confirmed deployable
    (verify_tflite.assert_deployable_for_android).
    """
    if not config.ANDROID_MODEL_METADATA_PATH.parent.is_dir():
        raise FileNotFoundError(
            f"{config.ANDROID_MODEL_METADATA_PATH.parent} does not exist -- expected the "
            "placeholder assets committed alongside android/ (see android/README.md)."
        )
    verify_tflite.assert_deployable_for_android(staged_path.read_bytes())
    shutil.copy2(staged_path, config.ANDROID_MODEL_TFLITE_PATH)
    config.ANDROID_MODEL_METADATA_PATH.write_text(json.dumps(metadata_dict, indent=2) + "\n", encoding="utf-8")
    print(f"[to_tflite] Deployed {staged_path} -> {config.ANDROID_MODEL_TFLITE_PATH}")
    print(f"[to_tflite] Wrote {config.ANDROID_MODEL_METADATA_PATH}")


def export(model_name: str = None) -> dict:
    """Top-level orchestration. `model_name` defaults to
    results["selected_model"] — overridable for testing or for exporting a
    non-default candidate. Always builds and verifies BOTH float16 (the
    deployed format) and INT8 (comparison-only — see module docstring for
    why INT8 is never deployable regardless of its own accuracy). Returns
    the export section written into results.json; raises (after writing
    everything staged/recorded) if float16's drop exceeds
    config.TFLITE_MAX_MACRO_F1_DROP, since then nothing is deployable.
    """
    results = results_io.load_results()
    selected_model_name = model_name or results["selected_model"]
    print(f"[to_tflite] Exporting '{selected_model_name}'...")

    ranking_entry = results_io.ranking_entry(results, selected_model_name)

    trained_model = inference.load_trained_model(selected_model_name)
    metadata.assert_class_names_integrity(config.PLANTVILLAGE_CLASS_NAMES, trained_model)
    export_model = _wrap_with_preprocessing(trained_model, selected_model_name)

    splits, _ = pipeline.load_splits()
    representative_size = config.TFLITE_CONFIG["representative_dataset_size"]
    representative_paths = _select_representative_paths(splits["train"], representative_size, config.SEED)
    val_macro_f1_float = ranking_entry["val_macro_f1"]

    print("[to_tflite] Converting to float16 (deployed format)...")
    float16_bytes = _convert_float16(export_model)
    verify_tflite.assert_io_contract(float16_bytes, np.float32, config.NUM_CLASSES, label="float16")
    float16_path = _write_staged_artifact(float16_bytes, config.TFLITE_FLOAT16_FILENAME)
    float16_file_size_mb = len(float16_bytes) / 1e6
    float16_verification = verify_tflite.verify_float16(
        float16_path, splits["val"], selected_model_name, val_macro_f1_float
    )

    print(f"[to_tflite] Converting to INT8 (comparison artifact) with {len(representative_paths)} representative train images...")
    int8_bytes = _convert_int8(export_model, _representative_dataset(representative_paths))
    verify_tflite.assert_io_contract(int8_bytes, np.uint8, config.NUM_CLASSES, label="INT8")
    int8_path = _write_staged_artifact(int8_bytes, config.TFLITE_INT8_FILENAME)
    int8_file_size_mb = len(int8_bytes) / 1e6
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
                "never splits['val'] or splits['test']. Used by the INT8 comparison artifact only "
                "-- float16 quantization needs no representative dataset."
            ),
        },
        "float16": {
            **float16_verification,
            "tflite_path": str(float16_path),
            "file_size_mb": float16_file_size_mb,
            "deployed_to_android": False,
        },
        "int8": {
            **int8_verification,
            "tflite_path": str(int8_path),
            "file_size_mb": int8_file_size_mb,
            "deployed_to_android": False,
            "note": (
                "Comparison artifact only, never deployed regardless of its own accuracy: full "
                "INT8 quantization keeps uint8 input/output tensors, but "
                "android/app/src/main/java/.../PlantClassifier.kt requires float32 "
                "(config.ANDROID_TFLITE_IO_DTYPE) to match the deployed float16 format."
            ),
        },
        "deployment_outcome": None,
    }

    if not float16_verification["within_tolerance"]:
        export_section["deployment_outcome"] = "raised_for_human_decision"
        results_io.write_export_section(results, export_section)
        raise RuntimeError(
            f"float16 export's validation macro-F1 drop ({float16_verification['val_macro_f1_drop']:.4f}) "
            f"exceeds the {config.TFLITE_MAX_MACRO_F1_DROP:.4f} tolerance -- refusing to deploy. "
            f"float={val_macro_f1_float:.4f} float16={float16_verification['val_macro_f1_quantized']:.4f} "
            f"int8={int8_verification['val_macro_f1_quantized']:.4f} (comparison only, never deployable). "
            f"Both candidates are staged at {config.TFLITE_OUTPUT_DIR} and recorded in "
            f"{config.RESULTS_JSON_PATH}'s 'export' section -- android/app/src/main/assets/ was left "
            "untouched."
        )

    metadata_dict = metadata.build_metadata(
        model_name=selected_model_name,
        class_names=config.PLANTVILLAGE_CLASS_NAMES,
        trained_model=trained_model,
        results=results,
        ranking_entry=ranking_entry,
        quantization="float16",
        tflite_file_size_mb=float16_file_size_mb,
        latency=float16_verification["latency_ms"],
        val_macro_f1_float=val_macro_f1_float,
        val_macro_f1_quantized=float16_verification["val_macro_f1_quantized"],
        representative_dataset_size=len(representative_paths),
    )
    _deploy_to_android(float16_path, metadata_dict)
    export_section["float16"]["deployed_to_android"] = True
    export_section["deployment_outcome"] = "deployed"
    results_io.write_export_section(results, export_section)
    return export_section


def main() -> None:
    export()


if __name__ == "__main__":
    main()
