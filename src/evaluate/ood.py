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

**PlantDoc arm (report-only, never tuned on).** PlantDoc — real field
photos, the closest available proxy for what the app will actually see —
is scored against the already-chosen threshold, never used to choose it:
choose_threshold() below only ever sees val_confidence/negative_confidence.
This exists because the threshold was tuned against PlantVillage validation
images (studio leaf photos) and Intel negatives (ordinary non-leaf scenes),
neither of which resembles a real, imperfect field photo of an actual leaf
— PlantDoc scoring only 19% raw accuracy while validation scores ~99% means
it was an open question whether the OOD gate even engages on PlantDoc-like
input, and if it does, whether it engages *correctly*. A threshold that
rejects nearly all PlantDoc images is a real (if disappointing) safety
property: no confident wrong diagnosis reaches a user, at the cost of the
app being nearly unusable on real photos. A threshold that accepts PlantDoc
images at a rate no better than PlantDoc's already-low unfiltered accuracy
means the gate provides no actual protection — a farmer whose photo is
accepted gets a confident diagnosis that is wrong about as often as if the
gate didn't exist at all. tune_ood_rejection() computes both real
possibilities from actual predictions and states plainly, in
plantdoc_safety_note, which one is true — never a value the code assumes.
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


def _population_summary(
    population: str,
    confidence: np.ndarray,
    threshold: float,
    y_true: np.ndarray = None,
    y_pred: np.ndarray = None,
) -> dict:
    """One population's behavior at an already-chosen threshold: how many
    of its samples are rejected/accepted, and (only when ground-truth
    labels are meaningful for this population — never true for the Intel
    negatives, which aren't leaves at all) accuracy on the accepted subset.
    mean/median confidence are reported regardless, so populations can be
    compared even where accuracy can't be (e.g. negatives vs. leaves).
    """
    accepted_mask = confidence >= threshold
    num_accepted = int(accepted_mask.sum())
    if y_true is not None and y_pred is not None:
        accuracy_on_accepted = (
            float(np.mean(y_pred[accepted_mask] == y_true[accepted_mask])) if num_accepted > 0 else None
        )
    else:
        accuracy_on_accepted = None
    return {
        "population": population,
        "num_samples": int(len(confidence)),
        "rejection_rate": float(np.mean(confidence < threshold)),
        "num_accepted": num_accepted,
        "accuracy_on_accepted": accuracy_on_accepted,
        "mean_confidence": float(np.mean(confidence)),
        "median_confidence": float(np.median(confidence)),
    }


def _plantdoc_safety_note(plantdoc_summary: dict, overall_plantdoc_accuracy: float, threshold: float) -> str:
    """States plainly, from the real computed numbers (never a hand-waved
    judgment call), which of the two safety-relevant possibilities is true:
    the gate rejects nearly everything (safe but useless), the gate accepts
    a meaningful share at no better than PlantDoc's own already-low
    unfiltered accuracy (the harm case — the gate is not protecting users),
    or the gate is providing some real, if partial, filtering. The "is this
    accuracy actually filtering for correctness" comparison is against
    PlantDoc's own overall unfiltered accuracy — a real, already-computed
    number for this exact model and dataset — rather than an arbitrary
    percentage cutoff, so the verdict never depends on a chosen constant.
    """
    num_samples = plantdoc_summary["num_samples"]
    num_accepted = plantdoc_summary["num_accepted"]
    rejection_rate = plantdoc_summary["rejection_rate"]
    accuracy_on_accepted = plantdoc_summary["accuracy_on_accepted"]

    if num_accepted == 0:
        return (
            f"At threshold {threshold:.2f}, the OOD gate rejects all {num_samples} PlantDoc "
            f"images ({rejection_rate:.1%} rejection rate) — the app would refuse to diagnose "
            "every real field photo in this set rather than name a disease. This is safe (no "
            "confident wrong diagnosis reaches a user) but means the app is effectively unusable "
            "on real field photos as tested; this limitation must be reported plainly, not "
            "glossed as a success."
        )

    acceptance_rate = num_accepted / num_samples
    if accuracy_on_accepted <= overall_plantdoc_accuracy:
        return (
            f"At threshold {threshold:.2f}, {acceptance_rate:.1%} of PlantDoc images "
            f"({num_accepted}/{num_samples}) are accepted, and accuracy on those accepted images "
            f"({accuracy_on_accepted:.1%}) is no better than PlantDoc's overall unfiltered "
            f"accuracy ({overall_plantdoc_accuracy:.1%}) — the confidence gate provides no real "
            "filtering for correctness on real field photos. The rejection threshold is NOT "
            "protecting users: a farmer whose photo passes the gate receives a confident "
            "diagnosis that is wrong about as often as an unfiltered prediction would be. This is "
            "the harm case and must be reported as such, not softened."
        )

    return (
        f"At threshold {threshold:.2f}, {acceptance_rate:.1%} of PlantDoc images "
        f"({num_accepted}/{num_samples}) are accepted, with accuracy on those accepted images at "
        f"{accuracy_on_accepted:.1%} — meaningfully above PlantDoc's overall unfiltered accuracy "
        f"({overall_plantdoc_accuracy:.1%}), so the gate is providing some real filtering for "
        "correctness. This is still well below PlantVillage validation-split performance and "
        "should be reported as a partial, not complete, safeguard."
    )


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


def tune_ood_rejection(
    model,
    model_name: str,
    plantdoc_y_true: np.ndarray = None,
    plantdoc_y_pred: np.ndarray = None,
    plantdoc_y_prob: np.ndarray = None,
) -> dict:
    """Tunes the threshold on validation + Intel negatives only, then (when
    PlantDoc predictions are supplied — src/evaluate/runner.py always
    supplies them from the same Step 3 external.evaluate_plantdoc() run;
    the integration test in tests/test_evaluate_integration.py omits them,
    since it has no real PlantDoc data) reports, but never tunes on, how
    that already-chosen threshold behaves on real field photos. Raises if
    only some of plantdoc_y_true/plantdoc_y_pred/plantdoc_y_prob are given
    — a partial set would silently skip the PlantDoc safety arm instead of
    failing loudly (CLAUDE.md rule 1).
    """
    plantdoc_args = (plantdoc_y_true, plantdoc_y_pred, plantdoc_y_prob)
    if any(a is not None for a in plantdoc_args) and not all(a is not None for a in plantdoc_args):
        raise ValueError(
            "tune_ood_rejection() requires plantdoc_y_true, plantdoc_y_pred, and plantdoc_y_prob "
            "together or not at all."
        )

    splits, _ = pipeline.load_splits()
    val_paths, val_labels = pipeline._paths_and_labels(splits["val"])
    val_ds = inference.build_eval_pipeline(val_paths, val_labels, model_name)
    val_y_true, val_y_prob = inference.predict_dataset(model, val_ds)
    val_confidence = val_y_prob.max(axis=1)
    val_y_pred = val_y_prob.argmax(axis=1)

    negative_paths = _negatives_paths()
    negative_y_prob = inference.predict_paths(model, negative_paths, model_name)
    negative_confidence = negative_y_prob.max(axis=1)

    # Threshold is chosen from validation + Intel negatives ONLY. PlantDoc
    # never reaches choose_threshold() — see module docstring.
    chosen = choose_threshold(val_confidence, negative_confidence)
    threshold = chosen["threshold"]

    val_summary = _population_summary(
        "validation_in_distribution", val_confidence, threshold, val_y_true, val_y_pred
    )
    negatives_summary = _population_summary("intel_negatives_out_of_distribution", negative_confidence, threshold)
    population_comparison = [val_summary, negatives_summary]

    result = {
        "chosen_threshold": threshold,
        "tuned_on": "validation split (in-distribution) + Intel negatives (out-of-distribution)",
        "rejection_rate_on_negatives": chosen["rejection_rate_on_negatives"],
        "false_rejection_rate_on_genuine_leaves": chosen["false_rejection_rate_on_genuine_leaves"],
        "num_val_samples": val_summary["num_samples"],
        "num_negative_samples": negatives_summary["num_samples"],
        "num_val_accepted_at_threshold": val_summary["num_accepted"],
        "accuracy_on_accepted_val_samples": val_summary["accuracy_on_accepted"],
    }

    if plantdoc_y_true is not None:
        plantdoc_confidence = plantdoc_y_prob.max(axis=1)
        plantdoc_summary = _population_summary(
            "plantdoc_external_field_photos", plantdoc_confidence, threshold, plantdoc_y_true, plantdoc_y_pred
        )
        population_comparison.append(plantdoc_summary)

        overall_plantdoc_accuracy = float(np.mean(plantdoc_y_pred == plantdoc_y_true))
        result["plantdoc_at_chosen_threshold"] = plantdoc_summary
        result["plantdoc_overall_accuracy_unfiltered"] = overall_plantdoc_accuracy
        result["plantdoc_safety_note"] = _plantdoc_safety_note(plantdoc_summary, overall_plantdoc_accuracy, threshold)

    result["population_comparison_at_chosen_threshold"] = population_comparison

    android_metadata_path = _update_android_metadata(threshold)
    result["android_model_metadata_path"] = str(android_metadata_path)

    return result
