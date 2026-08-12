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

Step 3's PlantDoc predictions (y_true/y_pred/y_prob) are also threaded into
Step 5: the OOD rejection threshold is tuned on validation + Intel negatives
only (src/evaluate/ood.py never lets PlantDoc into that search), but is then
scored against PlantDoc — report only, never retuned — since PlantDoc is
the closest available proxy for what the app will actually see, and the
threshold's behavior there was previously unmeasured.

Run (Kaggle, once every config.CANDIDATE_MODELS entry has a real trained
checkpoint from src/train_all.py — either locally in config.CHECKPOINT_DIR
or restorable from it, since src/evaluate/model_selection.py restores and
recomputes each candidate's validation macro-F1 directly from its
checkpoint, never from a training-log history file):
    python -m src.evaluate.runner

On Kaggle, once results.json is written, run() pushes it — along with
artifacts/figures/ and the dedupe/split/inventory/mapping outputs — to the
same Kaggle-persisted dataset src/models/kaggle_persist.py already uses for
checkpoints (src/models/kaggle_persist_artifacts.py), so a session recycle
never destroys them. See that module's docstring for the push/restore
design and src/data/prepare_artifacts.py for where restore happens.

**That push failing must never fail this whole run.** By the time it runs,
every number in results.json has already been computed and durably written
to disk — the push is a best-effort attempt to also durably persist it
*across sessions*, not a precondition for the evaluation itself having
succeeded. A real Kaggle run once reported "Version N failed to run" purely
because this push raised an uncaught exception at the very end, even though
results.json was already complete and correct; _push_artifacts_or_report_loudly()
below catches that instead of letting it propagate, and reports it loudly —
worded distinctly from an evaluation failure — rather than silently.
"""

import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config  # noqa: E402
from src.evaluate import calibration, external, gradcam, inference, metrics, model_selection, ood, robustness  # noqa: E402
from src.models import kaggle_persist_artifacts, training_utils  # noqa: E402


def _print_plantdoc_summary(plantdoc_results: dict) -> None:
    """Surfaces macro-F1 and the exact/convention/forced confidence-tier
    breakdown on the console — both are already computed and written into
    results.json by external.py, but only accuracy used to be printed here.
    A tier breakdown that's far worse for convention/forced than exact is
    informative about the hand-written mapping itself, not the model, so
    it belongs in plain view during the run, not only in the JSON.
    """
    overall = plantdoc_results["overall"]
    print(
        f"[runner]   overall: accuracy={overall['accuracy']:.4f} macro_f1={overall['macro_f1']:.4f} "
        f"({overall['num_images']} images, {plantdoc_results['num_covered_classes']}/"
        f"{plantdoc_results['num_plantvillage_classes']} PlantVillage classes covered)"
    )
    for tier, tier_result in plantdoc_results["mapping_confidence_tier_breakdown"].items():
        accuracy = tier_result["accuracy"]
        macro_f1 = tier_result["macro_f1"]
        accuracy_str = f"{accuracy:.4f}" if accuracy is not None else "n/a"
        macro_f1_str = f"{macro_f1:.4f}" if macro_f1 is not None else "n/a"
        print(
            f"[runner]     {tier:<10} accuracy={accuracy_str} macro_f1={macro_f1_str} "
            f"({tier_result['num_images']} images)"
        )


def _push_artifacts_or_report_loudly() -> None:
    """Best-effort cross-session persistence of artifacts/ — see module
    docstring for why a failure here must be caught rather than propagated,
    and why it's still reported loudly and distinctly from an evaluation
    failure rather than silently (CLAUDE.md rule 1: not-persisted-and-you-
    don't-know-it is its own kind of silent failure).
    """
    try:
        kaggle_persist_artifacts.push_data_artifacts()
    except Exception as exc:
        banner = "!" * 78
        print(f"[runner] {banner}")
        print(
            "[runner] ARTIFACTS PERSISTENCE FAILED -- this is NOT an evaluation failure. "
            f"Every number above is real and was already written to {config.RESULTS_JSON_PATH} "
            f"before this push was attempted. It was NOT pushed to the Kaggle-persisted "
            f"'{config.KAGGLE_PERSIST_DATASET_SLUG}' dataset, so results.json, artifacts/figures/, "
            "and this run's dedupe/split/inventory/mapping outputs will be LOST if this "
            "session ends without a successful push -- before the session recycles, retry "
            "manually: python -c "
            "\"from src.models import kaggle_persist_artifacts as k; k.push_data_artifacts()\""
        )
        print(f"[runner] {type(exc).__name__}: {exc}")
        traceback.print_exc()
        print(f"[runner] {banner}")


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
    plantdoc_results, plantdoc_y_true, plantdoc_y_pred, plantdoc_y_prob, plantdoc_paths = external.evaluate_plantdoc(
        model, selected_model_name
    )
    _print_plantdoc_summary(plantdoc_results)

    print("[runner] Step 4/7: calibration (ECE + temperature scaling)...")
    calibration_results = calibration.run_calibration(model, selected_model_name, test_y_true, test_y_prob)
    print(
        f"[runner]   ECE before={calibration_results['test_ece_before']:.4f} "
        f"after={calibration_results['test_ece_after']:.4f} (T={calibration_results['temperature']:.2f})"
    )

    print("[runner] Step 5/7: out-of-distribution rejection tuning...")
    ood_results = ood.tune_ood_rejection(
        model,
        selected_model_name,
        plantdoc_y_true=plantdoc_y_true,
        plantdoc_y_pred=plantdoc_y_pred,
        plantdoc_y_prob=plantdoc_y_prob,
    )
    print(f"[runner]   threshold={ood_results['chosen_threshold']:.2f}")
    plantdoc_arm = ood_results["plantdoc_at_chosen_threshold"]
    print(
        f"[runner]   plantdoc: rejection_rate={plantdoc_arm['rejection_rate']:.4f} "
        f"num_accepted={plantdoc_arm['num_accepted']}/{plantdoc_arm['num_samples']} "
        f"accuracy_on_accepted={plantdoc_arm['accuracy_on_accepted']} "
        f"confident_wrong={plantdoc_arm['num_confident_wrong']}/{plantdoc_arm['num_samples']} "
        f"({plantdoc_arm['pct_confident_wrong']:.1%})"
    )
    print(f"[runner]   {ood_results['plantdoc_safety_note']}")

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

    if config.IS_KAGGLE:
        _push_artifacts_or_report_loudly()

    return results


if __name__ == "__main__":
    run()
