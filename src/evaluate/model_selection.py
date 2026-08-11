"""Model selection: ranks every config.CANDIDATE_MODELS entry by validation
macro-F1 only (CLAUDE.md rule 2 — the test set is never touched here).

val_macro_f1 is RECOMPUTED from each candidate's real, persisted checkpoint
— restored from the Kaggle-persisted dataset if not already present locally
(src/models/kaggle_persist.py's restore_checkpoint, with its existing retry
logic) — by loading the model and running it once over the real validation
split. It is never read out of artifacts/history_<model>.json. Two real
incidents motivate this:
  1. A fresh Kaggle session starts with an empty /kaggle/working; the
     history JSONs only ever lived on local disk, so model_selection used to
     fail outright with FileNotFoundError the moment a session restarted.
  2. Two of those history files (history_MobileNetV2.json,
     history_MobileNetV3Small.json) were separately overwritten with
     {"phase1": null, "phase2": null} stubs when those models were re-run in
     a later session with both phases already complete — restoring them
     would not have fixed anything, since the restored file would itself be
     wrong (see src/train.py's run_training for the fix to the write path
     that caused this; this module's fix is independent of that one — it
     removes the read-a-log-file dependency entirely, which is the more
     defensible source of truth regardless of whether the log is intact).

The recomputed per-model result (val_macro_f1, param_count,
checkpoint_file_size_bytes, and the checkpoint's mtime at computation time)
is cached in this file's own previous artifacts/results.json output — see
_load_previous_ranking_cache/_cached_entry_if_fresh below — so a rerun of
the whole evaluation suite doesn't repeat four full validation-set forward
passes when nothing has changed. A cached entry is trusted only if its
recorded checkpoint_mtime is >= the checkpoint's actual current mtime;
anything newer (a checkpoint that was retrained since the cache was
written) forces a fresh recompute.

Writes nothing on its own; src/evaluate/runner.py folds the return value
into artifacts/results.json (which is also where the cache above is read
from, on the next run).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from sklearn.metrics import f1_score  # noqa: E402

from src import config  # noqa: E402
from src.data import pipeline  # noqa: E402
from src.evaluate import inference  # noqa: E402
from src.models import kaggle_persist  # noqa: E402

_REQUIRED_CACHE_KEYS = {"val_macro_f1", "param_count", "checkpoint_file_size_bytes", "checkpoint_mtime"}


def _load_previous_ranking_cache() -> dict:
    """Reads whatever ranking this module previously wrote into
    artifacts/results.json (if it exists) into a {model_name: entry} map.
    Returns {} for "no usable cache yet" — a missing file, unparseable JSON,
    or a results.json with no model_selection section are all just the
    normal first-run state, never an error.
    """
    if not config.RESULTS_JSON_PATH.is_file():
        return {}
    try:
        previous = json.loads(config.RESULTS_JSON_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    ranking = previous.get("model_selection", {}).get("ranking", [])
    return {entry["model_name"]: entry for entry in ranking if "model_name" in entry}


def _cached_entry_if_fresh(cache: dict, model_name: str, checkpoint_mtime: float) -> dict:
    entry = cache.get(model_name)
    if entry is None or not _REQUIRED_CACHE_KEYS <= set(entry):
        return None
    if entry["checkpoint_mtime"] >= checkpoint_mtime:
        return entry
    return None


def _compute_val_macro_f1(model, model_name: str) -> tuple:
    """Runs `model` once over the real validation split and returns
    (val_macro_f1, num_val_samples). Never the test split (CLAUDE.md rule 2)."""
    splits, _ = pipeline.load_splits()
    val_paths, val_labels = pipeline._paths_and_labels(splits["val"])
    val_ds = inference.build_eval_pipeline(val_paths, val_labels, model_name)
    y_true, y_prob = inference.predict_dataset(model, val_ds)
    y_pred = y_prob.argmax(axis=1)
    macro_f1 = float(
        f1_score(y_true, y_pred, labels=list(range(config.NUM_CLASSES)), average="macro", zero_division=0)
    )
    return macro_f1, int(len(y_true))


def _evaluate_candidate(model_name: str, cache: dict) -> dict:
    checkpoint_dir = config.CHECKPOINT_DIR / model_name
    kaggle_persist.restore_checkpoint(model_name, checkpoint_dir)
    weights_path = inference.checkpoint_weights_path(model_name)
    checkpoint_mtime = weights_path.stat().st_mtime

    cached = _cached_entry_if_fresh(cache, model_name, checkpoint_mtime)
    if cached is not None:
        print(f"[model_selection] '{model_name}': reusing cached val_macro_f1={cached['val_macro_f1']:.4f}.")
        return cached

    print(f"[model_selection] '{model_name}': recomputing val_macro_f1 from the real checkpoint...")
    model = inference.load_trained_model(model_name)
    val_macro_f1, num_val_samples = _compute_val_macro_f1(model, model_name)
    print(f"[model_selection]   -> val_macro_f1={val_macro_f1:.4f} ({num_val_samples} val samples).")

    return {
        "model_name": model_name,
        "val_macro_f1": val_macro_f1,
        "num_val_samples": num_val_samples,
        "param_count": int(model.count_params()),
        "checkpoint_file_size_bytes": weights_path.stat().st_size,
        "checkpoint_mtime": checkpoint_mtime,
    }


def rank_candidates() -> list:
    """Returns one dict per config.CANDIDATE_MODELS entry — the full
    ranking, not just the winner — sorted descending by val_macro_f1.
    """
    cache = _load_previous_ranking_cache()
    ranking = [_evaluate_candidate(model_name, cache) for model_name in config.CANDIDATE_MODELS]
    ranking.sort(key=lambda r: r["val_macro_f1"], reverse=True)
    return ranking


def select_best_model() -> dict:
    """CLAUDE.md rule 2: model selection uses validation macro-F1 only —
    this function never reads splits["test"] or anything test-derived.
    """
    ranking = rank_candidates()
    return {
        "ranking": ranking,
        "selected_model": ranking[0]["model_name"],
        "selection_metric": "val_macro_f1",
        "selection_note": (
            "Selected by highest validation macro-F1 alone (CLAUDE.md rule 2). "
            "val_macro_f1 is recomputed from each candidate's real, persisted "
            "checkpoint (restored from Kaggle if needed) run once over the "
            "validation split -- never read from a training-log history file, "
            "which a later resumed run can silently overwrite with a null "
            "stub (see src/train.py's history-write guard) and which does not "
            "survive a fresh Kaggle session on its own. param_count and "
            "checkpoint_file_size_bytes are reported alongside for a "
            "size/latency-aware deployment decision, but did not influence "
            "this selection."
        ),
    }
