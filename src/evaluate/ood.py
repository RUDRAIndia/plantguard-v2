"""Out-of-distribution (OOD) rejection: tunes a max-softmax confidence
threshold below which the app should refuse to name a disease at all,
rather than force a confident wrong prediction onto a photo that isn't even
a leaf. Tuned on the validation split (in-distribution) plus the Intel
negatives set (src/data/negatives.py, out-of-distribution) — never the test
split (CLAUDE.md rule 2).

The chosen threshold is written into config.ANDROID_MODEL_METADATA_PATH's
confidence_threshold field so the Android app already reads a real, tuned
value the moment src/export/to_tflite.py's Day-9 real export replaces the
placeholder model.tflite — see that file's docstring and android/README.md
for the metadata contract this must not otherwise disturb.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config  # noqa: E402
from src.data import pipeline  # noqa: E402
from src.evaluate import inference  # noqa: E402


def _negatives_paths() -> list:
    if not config.NEGATIVES_DIR.is_dir():
        raise FileNotFoundError(
            f"{config.NEGATIVES_DIR} does not exist. Run src/data/negatives.py first."
        )
    paths = sorted(
        str(p)
        for category_dir in config.NEGATIVES_DIR.iterdir()
        if category_dir.is_dir()
        for p in category_dir.iterdir()
        if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS
    )
    if not paths:
        raise RuntimeError(f"No negative images found under {config.NEGATIVES_DIR}.")
    return paths


def choose_threshold(val_confidence: np.ndarray, negative_confidence: np.ndarray) -> dict:
    """Pure numeric core: grid search over config.OOD_THRESHOLD_GRID_STEP
    increments for the threshold maximizing Youden's J statistic —
    rejection_rate on negatives (true positive: OOD correctly rejected)
    minus false_rejection_rate on val (false positive: a genuine leaf
    wrongly rejected) — the standard way to pick a single operating point
    for a binary accept/reject decision without an externally-imposed target
    rate.
    """
    # np.linspace (exact endpoints) rather than np.arange (float
    # accumulation drift) — see calibration.fit_temperature's identical fix
    # for why that drift matters for a caller checking grid bounds.
    step = config.OOD_THRESHOLD_GRID_STEP
    num_steps = round(1.0 / step) + 1
    grid = np.linspace(0.0, 1.0, num_steps)

    best = None
    for threshold in grid:
        rejection_rate = float(np.mean(negative_confidence < threshold))
        false_rejection_rate = float(np.mean(val_confidence < threshold))
        j_statistic = rejection_rate - false_rejection_rate
        if best is None or j_statistic > best["j_statistic"]:
            best = {
                "threshold": float(threshold),
                "rejection_rate_on_negatives": rejection_rate,
                "false_rejection_rate_on_genuine_leaves": false_rejection_rate,
                "j_statistic": j_statistic,
            }
    return best


def _update_android_metadata(threshold: float) -> Path:
    """Updates only confidence_threshold (plus a small provenance note) in
    the existing android/app/src/main/assets/model_metadata.json — every
    other field (class_names, image_size, placeholder, notes) is left
    exactly as-is, since model.tflite itself is still the Day-9 placeholder
    (CLAUDE.md rule 13: never overwrite more than the validated piece).
    """
    path = config.ANDROID_MODEL_METADATA_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist — expected the placeholder model_metadata.json "
            "committed alongside android/ (see android/README.md)."
        )
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["confidence_threshold"] = threshold
    metadata["confidence_threshold_source"] = (
        "Tuned by src/evaluate/ood.py on the validation split + Intel negatives "
        f"(git commit {config.get_git_commit_hash()})."
    )
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path


def tune_ood_rejection(model, model_name: str) -> dict:
    splits, _ = pipeline.load_splits()
    val_paths, val_labels = pipeline._paths_and_labels(splits["val"])
    val_ds = inference.build_eval_pipeline(val_paths, val_labels, model_name)
    val_y_true, val_y_prob = inference.predict_dataset(model, val_ds)
    val_confidence = val_y_prob.max(axis=1)
    val_y_pred = val_y_prob.argmax(axis=1)

    negative_paths = _negatives_paths()
    negative_y_prob = inference.predict_paths(model, negative_paths, model_name)
    negative_confidence = negative_y_prob.max(axis=1)

    chosen = choose_threshold(val_confidence, negative_confidence)
    threshold = chosen["threshold"]

    accepted_mask = val_confidence >= threshold
    num_accepted = int(accepted_mask.sum())
    accuracy_on_accepted = (
        float(np.mean(val_y_pred[accepted_mask] == val_y_true[accepted_mask])) if num_accepted > 0 else None
    )

    android_metadata_path = _update_android_metadata(threshold)

    return {
        "chosen_threshold": threshold,
        "tuned_on": "validation split (in-distribution) + Intel negatives (out-of-distribution)",
        "rejection_rate_on_negatives": chosen["rejection_rate_on_negatives"],
        "false_rejection_rate_on_genuine_leaves": chosen["false_rejection_rate_on_genuine_leaves"],
        "num_val_samples": int(len(val_y_true)),
        "num_negative_samples": int(len(negative_paths)),
        "num_val_accepted_at_threshold": num_accepted,
        "accuracy_on_accepted_val_samples": accuracy_on_accepted,
        "android_model_metadata_path": str(android_metadata_path),
    }
