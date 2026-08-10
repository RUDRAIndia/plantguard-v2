"""Prints, for every config.CANDIDATE_MODELS backbone, whether a persisted
checkpoint exists in the Kaggle Dataset src/models/kaggle_persist.py pushes
to and the best validation macro-F1 it's reached so far — a cross-session
progress view, since /kaggle/working itself can be wiped at any time (see
that module's docstring). Thin wrapper around
kaggle_persist_report.report_progress(); only ever meaningful on Kaggle,
where the Kaggle API can actually authenticate against the real dataset.

Run: python -m scripts.report_training_progress
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402
from src.models import kaggle_persist_report  # noqa: E402


def main() -> None:
    if not config.IS_KAGGLE:
        raise RuntimeError(
            "This report reads from the Kaggle API against a real, "
            f"authenticated account's '{config.KAGGLE_PERSIST_DATASET_SLUG}' "
            "dataset — only meaningful when run on Kaggle (config.IS_KAGGLE)."
        )

    report = kaggle_persist_report.report_progress()

    name_width = max(len(name) for name in config.CANDIDATE_MODELS)
    header = f"{'model'.ljust(name_width)}  {'checkpoint':<10}  {'phase reached':<24}  best val macro-F1"
    print(header)
    print("-" * len(header))
    for model_name in config.CANDIDATE_MODELS:
        entry = report[model_name]
        checkpoint = "yes" if entry["checkpoint_exists"] else "no"
        best_f1 = entry["best_val_macro_f1"]
        best_f1_str = f"{best_f1:.4f}" if best_f1 is not None else "n/a"
        print(f"{model_name.ljust(name_width)}  {checkpoint:<10}  {entry['phase_reached']:<24}  {best_f1_str}")


if __name__ == "__main__":
    main()
