"""Builds the real android/app/src/main/assets/model_metadata.json content
for the Day-9 export, and the class-name integrity check that guards it.
Split out from src/export/to_tflite.py to stay under CLAUDE.md rule 12's
~300-line guideline for that module's conversion/orchestration logic.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config  # noqa: E402

# Describes the ACTUAL .tflite input tensor's dtype/semantics for each
# quantization scheme this project can export — read by the Android app's
# human maintainers (Kotlin never parses this field), so it must always
# match reality, not just the deployed default. Preprocessing (the
# backbone's normalization) is baked into the exported graph either way
# (src/export/to_tflite.py's _wrap_with_preprocessing) — the app must never
# normalize or scale anything itself, only decode/resize/cast pixels.
_INPUT_FORMAT_BY_QUANTIZATION = {
    "float16": (
        "float32, RGB, HWC, raw 0-255 pixel values (NOT normalized to [0,1] or [-1,1]) -- the "
        "exported graph's raw_pixels_0_255 input applies the backbone's preprocess_input "
        "internally, so the app must not normalize or scale anything itself, just cast decoded "
        "pixel bytes to float32 as-is"
    ),
    "int8": (
        "uint8, RGB, HWC, raw 0-255 pixel values, no client-side normalization (the "
        ".tflite's own input-tensor quantization already absorbs whatever float "
        "preprocessing was used at training time)"
    ),
}


def assert_class_names_integrity(class_names, model: keras.Model) -> None:
    """Raises loudly (never a bare assert, which strips under python -O) on
    the single check the user called out as highest-stakes: a wrong class
    order makes every prediction silently wrong.
    """
    if list(class_names) != list(config.PLANTVILLAGE_CLASS_NAMES):
        raise RuntimeError(
            "class_names does not match config.PLANTVILLAGE_CLASS_NAMES exactly -- refusing "
            "to export. A wrong class order makes every prediction silently wrong."
        )
    if model.output_shape[-1] != len(class_names):
        raise RuntimeError(
            f"Model output dimension {model.output_shape[-1]} != {len(class_names)} class names -- "
            "this checkpoint may have been trained against a different class count than "
            "config.PLANTVILLAGE_CLASS_NAMES currently declares."
        )


def build_metadata(
    *,
    model_name: str,
    class_names,
    trained_model: keras.Model,
    results: dict,
    ranking_entry: dict,
    quantization: str,
    tflite_file_size_mb: float,
    latency: dict,
    val_macro_f1_float: float,
    val_macro_f1_quantized: float,
    representative_dataset_size: int,
) -> dict:
    """Pure (no I/O): assembles the real model_metadata.json content from
    the same source of truth as training (config.py + results.json), and
    asserts class-list integrity before returning.
    """
    assert_class_names_integrity(class_names, trained_model)
    if quantization not in _INPUT_FORMAT_BY_QUANTIZATION:
        raise ValueError(
            f"No input_format description for quantization='{quantization}' -- known schemes: "
            f"{sorted(_INPUT_FORMAT_BY_QUANTIZATION)}. Add one to metadata.py's "
            "_INPUT_FORMAT_BY_QUANTIZATION rather than guess; the app's maintainers read this "
            "field to know what dtype/range the app must feed the model."
        )

    module_path, function_name = config.PREPROCESSING_ENTRYPOINTS[model_name]
    export_commit = config.get_git_commit_hash()
    evaluated_commit = results["git_commit_hash"]

    return {
        "schema_version": 1,
        "placeholder": False,
        "architecture": model_name,
        "image_size": config.IMAGE_SIZE,
        "input_format": _INPUT_FORMAT_BY_QUANTIZATION[quantization],
        "preprocessing_entrypoint": f"{module_path}.{function_name}",
        "quantization": quantization,
        "confidence_threshold": results["ood_rejection"]["chosen_threshold"],
        "confidence_threshold_source": (
            f"Read from results.json['ood_rejection']['chosen_threshold'] (tuned by "
            f"src/evaluate/ood.py at commit {evaluated_commit}); this model.tflite exported "
            f"at commit {export_commit}."
        ),
        "calibration_temperature": results["calibration"]["temperature"],
        "class_names_source": (
            "Verbatim copy of src/config.py:PLANTVILLAGE_CLASS_NAMES (alphabetically sorted "
            "= index order). Kotlin cannot import Python, so this array is the Android-side "
            "copy of that source of truth. If config.py's tuple ever changes order, "
            "membership, or count, this array must be re-synced in the same commit "
            "(CLAUDE.md rule 7)."
        ),
        "class_names": list(class_names),
        "seed": config.SEED,
        "git_commit_hash": export_commit,
        "evaluated_at_git_commit_hash": evaluated_commit,
        "val_macro_f1_float": val_macro_f1_float,
        "val_macro_f1_quantized": val_macro_f1_quantized,
        "val_macro_f1_drop": val_macro_f1_float - val_macro_f1_quantized,
        "param_count": ranking_entry["param_count"],
        "checkpoint_file_size_mb": ranking_entry["checkpoint_file_size_bytes"] / 1e6,
        "tflite_file_size_mb": tflite_file_size_mb,
        "inference_latency_ms": latency,
        "representative_dataset_size": representative_dataset_size,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "notes": (
            f"Real trained {model_name} export ({quantization} quantization), replacing the "
            "placeholder. See artifacts/results.json's 'export' section for the full "
            "verification report (float vs. quantized macro-F1, latency)."
        ),
    }
