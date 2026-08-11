"""Single entrypoint for the Day-8 full evaluation suite: runs model
selection, PlantVillage test-set evaluation, PlantDoc external evaluation,
calibration, out-of-distribution rejection tuning, robustness, and Grad-CAM,
in that order, and writes everything into artifacts/results.json — the one
file every report/README/figure number is read from (CLAUDE.md rule 5).

The PlantVillage test split is touched exactly once for its own evaluation
(Step 2) — CLAUDE.md rule 2 — and its resulting (y_true, y_prob, paths) are
threaded into calibration and Grad-CAM rather than re-running inference over
it again. Robustness (Step 6) re-reads the same test split paths/labels but
runs genuinely different, corrupted inputs through the model; no step here
selects a model, tunes a threshold, or makes any decision using the test
split's clean predictions beyond what Step 2 already reported.

Run (Kaggle, once every config.CANDIDATE_MODELS entry has a real
train_manifest_*/history_*.json from src/train_all.py):
    python -m src.evaluate.runner
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config  # noqa: E402
from src.evaluate import calibration, external, gradcam, inference, metrics, model_selection, ood, robustness  # noqa: E402
from src.models import training_utils  # noqa: E402


def run() -> dict:
    print("[runner] Step 1/7: model selection (validation macro-F1 only)...")
    selection = model_selection.select_best_model()
    selected_model_name = selection["selected_model"]
    print(f"[runner]   selected: {selected_model_name}")

    model = inference.load_trained_model(selected_model_name)

    print("[runner] Step 2/7: PlantVillage test-set evaluation (touching the test split once)...")
    test_metrics, test_y_true, test_y_prob, test_paths = metrics.evaluate_test_set(
        model, selected_model_name, config.PLANTVILLAGE_CLASS_NAMES
    )
    test_y_pred = test_y_prob.argmax(axis=1)
    print(f"[runner]   accuracy={test_metrics['accuracy']:.4f} macro_f1={test_metrics['macro_f1']:.4f}")

    print("[runner] Step 3/7: PlantDoc external evaluation...")
    plantdoc_results, plantdoc_y_true, plantdoc_y_pred, plantdoc_paths = external.evaluate_plantdoc(
        model, selected_model_name
    )
    print(f"[runner]   PlantDoc accuracy={plantdoc_results['overall']['accuracy']}")

    print("[runner] Step 4/7: calibration (ECE + temperature scaling)...")
    calibration_results = calibration.run_calibration(model, selected_model_name, test_y_true, test_y_prob)
    print(
        f"[runner]   ECE before={calibration_results['test_ece_before']:.4f} "
        f"after={calibration_results['test_ece_after']:.4f} (T={calibration_results['temperature']:.2f})"
    )

    print("[runner] Step 5/7: out-of-distribution rejection tuning...")
    ood_results = ood.tune_ood_rejection(model, selected_model_name)
    print(f"[runner]   threshold={ood_results['chosen_threshold']:.2f}")

    print("[runner] Step 6/7: robustness under corruption...")
    robustness_results = robustness.evaluate_robustness(model, selected_model_name)

    print("[runner] Step 7/7: Grad-CAM sampling...")
    gradcam_results = gradcam.generate_gradcam_samples(
        model,
        selected_model_name,
        test_y_true,
        test_y_pred,
        test_paths,
        plantdoc_y_true,
        plantdoc_y_pred,
        plantdoc_paths,
    )

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit_hash": config.get_git_commit_hash(),
        "selected_model": selected_model_name,
        "model_selection": selection,
        "test_evaluation": test_metrics,
        "external_evaluation_plantdoc": plantdoc_results,
        "calibration": calibration_results,
        "ood_rejection": ood_results,
        "robustness": robustness_results,
        "gradcam": gradcam_results,
    }
    results = training_utils.json_safe(results)

    config.RESULTS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.RESULTS_JSON_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[runner] Wrote {config.RESULTS_JSON_PATH}")
    return results


if __name__ == "__main__":
    run()
