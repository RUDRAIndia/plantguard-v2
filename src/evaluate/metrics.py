"""Computes primary classification metrics on the PlantVillage test split for
the model src/evaluate/model_selection.py selected: accuracy (secondary,
never reported alone), macro-F1, weighted F1, per-class precision/recall,
worst-class recall, and the config.NUM_TOP_CONFUSED_PAIRS most-confused
class pairs. A full 38x38 confusion matrix is never the headline — it is
saved only as a supplementary figure alongside the top-confusions table.

CLAUDE.md rule 2: this module reads the test split exactly once, via
src/data/pipeline.py's real test split (never re-derived here) — the caller
(src/evaluate/runner.py) is responsible for calling evaluate_test_set()
exactly once per run and reusing its (y_true, y_prob) return values for
anything else that needs test-set predictions (calibration, Grad-CAM
sampling), rather than re-running inference over the test split again.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend — this module only saves files
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config  # noqa: E402
from src.data import pipeline  # noqa: E402
from src.evaluate import inference  # noqa: E402


def _top_confused_pairs(confusion: np.ndarray, class_names: tuple, n: int) -> list:
    """The n largest off-diagonal (true, predicted) cells, ranked by raw
    count — the diagonal (correct predictions) is excluded by construction.
    `rate` is that count as a fraction of the true class's total test
    samples, so a pair isn't judged by raw count alone on an imbalanced set.
    """
    num_classes = confusion.shape[0]
    row_totals = confusion.sum(axis=1)
    pairs = []
    for true_idx in range(num_classes):
        for pred_idx in range(num_classes):
            if true_idx == pred_idx:
                continue
            count = int(confusion[true_idx, pred_idx])
            if count == 0:
                continue
            pairs.append(
                {
                    "true_class": class_names[true_idx],
                    "predicted_class": class_names[pred_idx],
                    "count": count,
                    "rate_of_true_class": round(count / row_totals[true_idx], 4)
                    if row_totals[true_idx] > 0
                    else 0.0,
                }
            )
    pairs.sort(key=lambda p: p["count"], reverse=True)
    return pairs[:n]


def _save_confusion_matrix_figure(confusion: np.ndarray, class_names: tuple, path: Path) -> None:
    """Supplementary figure only — CLAUDE.md's task brief is explicit that a
    raw 38x38 grid is unreadable as a headline result; this exists for
    completeness/audit, not as the primary evidence.
    """
    fig, ax = plt.subplots(figsize=(14, 14))
    im = ax.imshow(confusion, cmap="viridis")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=5)
    ax.set_yticklabels(class_names, fontsize=5)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("PlantVillage test set — full confusion matrix (supplementary)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, class_names: tuple) -> dict:
    """Pure numeric core (no model, no I/O beyond the confusion-matrix
    figure) — separated from evaluate_test_set below so it's directly
    testable against a small synthetic (y_true, y_prob).
    """
    num_classes = len(class_names)
    y_pred = np.argmax(y_prob, axis=1)
    labels = list(range(num_classes))

    accuracy = float(np.mean(y_pred == y_true))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    per_class = [
        {
            "class_name": class_names[i],
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(num_classes)
    ]
    worst = min(per_class, key=lambda c: c["recall"])

    confusion = confusion_matrix(y_true, y_pred, labels=labels)
    top_confusions = _top_confused_pairs(confusion, class_names, config.NUM_TOP_CONFUSED_PAIRS)

    return {
        "num_test_samples": int(len(y_true)),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_metrics": per_class,
        "worst_class_recall": {"class_name": worst["class_name"], "recall": worst["recall"]},
        "top_confused_pairs": top_confusions,
        "_confusion_matrix": confusion,  # consumed by evaluate_test_set, stripped before returning
    }


def evaluate_test_set(model, model_name: str, class_names: tuple) -> dict:
    """Touches the PlantVillage test split — see this module's docstring for
    why that must happen exactly once per evaluation run. Returns the metrics
    dict (JSON-safe) plus the raw (y_true, y_prob) arrays for reuse by
    calibration/Grad-CAM, and the test paths (same order as y_true/y_prob,
    src/data/pipeline.py's val/test branch never shuffles) for Grad-CAM
    sampling.
    """
    splits, _ = pipeline.load_splits()
    test_paths, test_labels = pipeline._paths_and_labels(splits["test"])
    test_ds = inference.build_eval_pipeline(test_paths, test_labels, model_name)
    y_true, y_prob = inference.predict_dataset(model, test_ds)

    metrics = compute_metrics(y_true, y_prob, class_names)
    confusion = metrics.pop("_confusion_matrix")

    figure_path = config.EVAL_FIGURES_DIR / "confusion_matrix.png"
    _save_confusion_matrix_figure(confusion, class_names, figure_path)
    metrics["confusion_matrix_figure"] = str(figure_path)

    return metrics, y_true, y_prob, test_paths
