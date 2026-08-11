"""Model selection: ranks every config.CANDIDATE_MODELS entry by validation
macro-F1 only (CLAUDE.md rule 2 — the test set is never touched here),
recording each candidate's parameter count and checkpoint file size
alongside the validation score so a deployment decision can weigh size and
latency, not accuracy alone. Writes nothing on its own; src/evaluate/runner.py
folds the return value into artifacts/results.json.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config  # noqa: E402
from src.evaluate import inference  # noqa: E402
from src.models import build, kaggle_persist_report  # noqa: E402


def _best_val_macro_f1(model_name: str) -> float:
    """Reads artifacts/history_{model_name}.json (written by src/train.py)
    and returns the best val_macro_f1 across both phases — reuses
    kaggle_persist_report's exact same extraction logic (already relied on
    by src/train_all.py) rather than re-deriving it.
    """
    history_path = config.ARTIFACTS_DIR / f"history_{model_name}.json"
    if not history_path.is_file():
        raise FileNotFoundError(
            f"{history_path} not found. Train '{model_name}' first "
            "(src/train.py) before running model selection."
        )
    history = json.loads(history_path.read_text(encoding="utf-8"))
    best = kaggle_persist_report._best_val_macro_f1(history)
    if best is None:
        raise RuntimeError(f"{history_path} has no val_macro_f1 values recorded.")
    return best


def _param_count(model_name: str) -> int:
    """Total parameter count of model_name's architecture. Built fresh
    (ImageNet-initialized, not the trained checkpoint) since parameter count
    is a pure function of architecture, identical regardless of which
    weights are loaded.
    """
    return build.build_model(model_name).count_params()


def _checkpoint_file_size_bytes(model_name: str) -> int:
    """Real on-disk size of the trained checkpoint — a stand-in for
    deployment size until src/export/to_tflite.py's real INT8 export exists
    (Day 9); the smaller, quantized .tflite will be meaningfully smaller
    than this, so this number is directional, not final.
    """
    return inference.checkpoint_weights_path(model_name).stat().st_size


def rank_candidates() -> list:
    """Returns one dict per config.CANDIDATE_MODELS entry — the full
    ranking, not just the winner — sorted descending by val_macro_f1.
    """
    ranking = []
    for model_name in config.CANDIDATE_MODELS:
        ranking.append(
            {
                "model_name": model_name,
                "val_macro_f1": _best_val_macro_f1(model_name),
                "param_count": _param_count(model_name),
                "checkpoint_file_size_bytes": _checkpoint_file_size_bytes(model_name),
            }
        )
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
            "param_count and checkpoint_file_size_bytes are reported alongside "
            "for a size/latency-aware deployment decision, but did not "
            "influence this selection."
        ),
    }
