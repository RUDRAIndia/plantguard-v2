"""Cross-session training-progress report, reading whatever
src/models/kaggle_persist.py has pushed to the private
config.KAGGLE_PERSIST_DATASET_SLUG Kaggle Dataset so far — a session's local
config.CHECKPOINT_DIR can be wiped at any time (see that module's
docstring), so this is the durable view of "how far did each model
actually get." Split out from kaggle_persist.py to keep that module under
CLAUDE.md rule 12's ~300-line guideline; reuses its authenticated-API and
staging helpers rather than duplicating them.
"""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config  # noqa: E402
from src.models import kaggle_persist  # noqa: E402


def _best_val_macro_f1(history: dict):
    values = []
    for phase_key in ("phase1", "phase2"):
        phase_history = history.get(phase_key)
        if phase_history and phase_history.get("val_macro_f1"):
            values.extend(phase_history["val_macro_f1"])
    return max(values) if values else None


def report_progress() -> dict:
    """For every config.CANDIDATE_MODELS entry, reports whether a persisted
    checkpoint exists, how far training got, and the best validation
    macro-F1 reached so far (read from the persisted history JSON, once one
    has been pushed). Returns {model_name: {"checkpoint_exists",
    "phase_reached", "best_val_macro_f1"}}.
    """
    report = {
        model_name: {
            "checkpoint_exists": False,
            "phase_reached": "not started",
            "best_val_macro_f1": None,
        }
        for model_name in config.CANDIDATE_MODELS
    }

    api = kaggle_persist._authenticated_api()
    dataset_ref = kaggle_persist._dataset_ref(api)
    if not kaggle_persist._dataset_exists(api, dataset_ref):
        return report

    listing = api.dataset_list_files(dataset_ref)
    file_names = {f.name for f in (listing.dataset_files or [])} if listing is not None else set()

    staging = kaggle_persist._fresh_staging_dir()
    for model_name in config.CANDIDATE_MODELS:
        prefix = model_name + kaggle_persist._SEPARATOR
        model_files = {name[len(prefix):] for name in file_names if name.startswith(prefix)}
        if not model_files:
            continue

        if "phase2_complete.json" in model_files:
            phase_reached = "phase 2 complete"
        elif "phase2.weights.h5" in model_files:
            phase_reached = "phase 2 in progress"
        elif "phase1_complete.json" in model_files:
            phase_reached = "phase 1 complete"
        elif "phase1.weights.h5" in model_files:
            phase_reached = "phase 1 in progress"
        else:
            phase_reached = "checkpoint files present, phase unclear"

        report[model_name]["checkpoint_exists"] = True
        report[model_name]["phase_reached"] = phase_reached

        history_filename = f"history_{model_name}.json"
        if history_filename in model_files:
            remote_name = prefix + history_filename
            try:
                api.dataset_download_file(dataset_ref, remote_name, path=str(staging), quiet=True, force=True)
                history = json.loads((staging / remote_name).read_text(encoding="utf-8"))
                report[model_name]["best_val_macro_f1"] = _best_val_macro_f1(history)
            except Exception as exc:
                # Observability only, not a decision input -- one model's
                # history being unreadable shouldn't blank out the other 4.
                print(f"[kaggle_persist_report] Could not read {remote_name}: {exc}")

    shutil.rmtree(staging, ignore_errors=True)
    return report
