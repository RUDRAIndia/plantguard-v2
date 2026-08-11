"""Multi-model training runner: invokes src/train.py once per
config.CANDIDATE_MODELS backbone as a fresh subprocess (process isolation --
one model's TensorFlow/GPU state can never leak into the next), skips any
model whose training is already complete, and never lets one model's
failure stop the rest.

Written after a real Kaggle run lost MobileNetV3Large, EfficientNetB0, and
EfficientNetV2B0 entirely: the ad-hoc notebook loop used
subprocess.run(..., check=True), so a persistence-layer error on
MobileNetV2's *final* push (weights already safely uploaded -- see
src/models/kaggle_persist.py's retry fix for that bug) raised
CalledProcessError and aborted every model queued after it, even though
MobileNetV2 itself had trained fine. This script exists so that mistake
can't repeat: each model's failure is caught and reported on its own, and
the closing summary makes a partial run impossible to mistake for a
complete one.

"Already complete" is checked two ways, cheapest first: the local
checkpoint dir's phase1_complete.json/phase2_complete.json markers (see
src/models/phases.py), then -- once, for the whole run, not per model --
the cross-session Kaggle-persisted state (src/models/kaggle_persist_report.py),
since a fresh Kaggle session's local checkpoint dir starts out empty even
for a model that already finished in an earlier session.

Run (Kaggle/Colab, real training, every config.CANDIDATE_MODELS entry):
    python -m src.train_all
Run a subset:
    python -m src.train_all --only MobileNetV2 EfficientNetV2B0
Exercise the runner itself without real data/GPU time (CLAUDE.md rule 10 --
local runs are smoke tests only):
    python -m src.train_all --smoke --no-persist
"""

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402
from src.models import kaggle_persist_report  # noqa: E402

_STATUS_COMPLETE = "COMPLETE"
_STATUS_SKIPPED = "SKIPPED"
_STATUS_FAILED = "FAILED"


def _checkpoint_dir(model_name: str, smoke: bool) -> Path:
    return config.CHECKPOINT_DIR / (model_name + ("_smoke" if smoke else ""))


def _is_locally_complete(model_name: str, smoke: bool) -> bool:
    checkpoint_dir = _checkpoint_dir(model_name, smoke)
    return (checkpoint_dir / "phase1_complete.json").is_file() and (
        checkpoint_dir / "phase2_complete.json"
    ).is_file()


def _best_val_macro_f1_from_local_history(model_name: str):
    history_path = config.ARTIFACTS_DIR / f"history_{model_name}.json"
    if not history_path.is_file():
        return None
    history = json.loads(history_path.read_text(encoding="utf-8"))
    return kaggle_persist_report._best_val_macro_f1(history)


def _fetch_persisted_report(smoke: bool, persist: bool) -> dict:
    """Cross-session view of what's already been pushed to Kaggle -- see
    module docstring. Only meaningful under the same conditions
    src/train.py itself gates persistence on; returns {} otherwise (every
    model then falls back to the local-only completeness check).
    """
    if smoke or not persist or not config.IS_KAGGLE:
        return {}
    try:
        return kaggle_persist_report.report_progress()
    except Exception as exc:
        print(
            f"[train_all] Could not check cross-session persisted state ({exc}) -- "
            "treating every model as not-yet-complete rather than guessing."
        )
        return {}


def _run_one_model(model_name: str, smoke: bool, persist: bool) -> dict:
    """Runs src/train.py for one model in a fresh subprocess. Never raises
    -- any failure is caught here, logged loudly with the subprocess's own
    traceback, and reported as FAILED instead of propagating and aborting
    the other models (the exact brittleness this script exists to fix).
    """
    cmd = [sys.executable, "-m", "src.train", "--model", model_name]
    if smoke:
        cmd.append("--smoke")
    if not persist:
        cmd.append("--no-persist")

    print(f"\n{'=' * 70}\n[train_all] Starting {model_name}: {' '.join(cmd)}\n{'=' * 70}")

    try:
        # stdout is inherited (live training progress in the notebook),
        # stderr is captured -- Python prints an uncaught exception's full
        # traceback to stderr, so this is exactly the text to show under
        # the FAILED banner below without having to scroll back through
        # potentially hours of interleaved epoch logs to find it.
        result = subprocess.run(cmd, stdout=None, stderr=subprocess.PIPE, text=True)
    except Exception as exc:
        print(f"[train_all] *** FAILED to launch {model_name}: {exc} ***")
        print(traceback.format_exc())
        return {"model_name": model_name, "status": _STATUS_FAILED, "best_val_macro_f1": None}

    if result.returncode != 0:
        print(f"\n[train_all] *** {model_name} FAILED (exit code {result.returncode}) ***")
        if result.stderr:
            print(f"[train_all] {model_name}'s traceback:\n{result.stderr}")
        return {"model_name": model_name, "status": _STATUS_FAILED, "best_val_macro_f1": None}

    print(f"[train_all] {model_name} finished successfully.")
    return {
        "model_name": model_name,
        "status": _STATUS_COMPLETE,
        "best_val_macro_f1": _best_val_macro_f1_from_local_history(model_name),
    }


def _print_summary(results: list) -> bool:
    """Prints the summary table. Returns whether every model is accounted
    for as complete or already-skipped (used for the process exit code) --
    the one thing this run must never let the caller mistake for "every
    candidate model finished."
    """
    print(f"\n{'=' * 70}\n[train_all] Summary\n{'=' * 70}")
    header = f"{'Model':<20}{'Status':<12}{'Best val macro-F1':<20}"
    print(header)
    print("-" * len(header))
    all_ok = True
    for r in results:
        f1 = r["best_val_macro_f1"]
        f1_str = f"{f1:.4f}" if f1 is not None else "n/a"
        print(f"{r['model_name']:<20}{r['status']:<12}{f1_str:<20}")
        if r["status"] == _STATUS_FAILED:
            all_ok = False
    print("-" * len(header))
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Trains every config.CANDIDATE_MODELS backbone, isolating each in its own "
            "subprocess so one model's failure never stops the rest."
        )
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=config.CANDIDATE_MODELS,
        metavar="MODEL",
        help="Train only these models instead of all of config.CANDIDATE_MODELS.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Pass --smoke through to every model (see src/train.py). Also disables the cross-session persisted-state check.",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Pass --no-persist through to every model, and skip the cross-session persisted-state check.",
    )
    args = parser.parse_args()

    models = list(args.only) if args.only else list(config.CANDIDATE_MODELS)
    persist = not args.no_persist

    persisted_report = _fetch_persisted_report(args.smoke, persist)

    results = []
    for model_name in models:
        already_complete = _is_locally_complete(model_name, args.smoke) or (
            persisted_report.get(model_name, {}).get("phase_reached") == "phase 2 complete"
        )
        if already_complete:
            print(f"[train_all] {model_name} is already complete -- skipping.")
            best_f1 = _best_val_macro_f1_from_local_history(model_name)
            if best_f1 is None:
                best_f1 = persisted_report.get(model_name, {}).get("best_val_macro_f1")
            results.append({"model_name": model_name, "status": _STATUS_SKIPPED, "best_val_macro_f1": best_f1})
            continue

        try:
            results.append(_run_one_model(model_name, args.smoke, persist))
        except Exception:
            # Defense in depth: _run_one_model is designed to never raise,
            # but if it somehow does, this model must not take the rest of
            # the run down with it either.
            print(f"[train_all] *** Unexpected error running {model_name} ***")
            print(traceback.format_exc())
            results.append({"model_name": model_name, "status": _STATUS_FAILED, "best_val_macro_f1": None})

    all_ok = _print_summary(results)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
