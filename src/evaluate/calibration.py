"""Expected Calibration Error (ECE), reliability diagrams, and temperature
scaling. An uncalibrated confidence shown to a farmer is a real harm (a
wrong prediction reported at 98% confidence), not a cosmetic metric — this
module exists to measure and then correct exactly that.

Temperature is fit on the VALIDATION split only, never the test split
(CLAUDE.md rule 2); ECE is then reported on the test split before and after
applying that validation-fit temperature, so "after" numbers reflect genuine
generalization of the calibration fix, not calibration fit on the same data
it's evaluated on.

The model's final layer is Dense(..., activation="softmax") (src/models/
build.py) — there is no separate pre-softmax logits tensor to pull
temperature scaling's usual softmax(logits / T) formula from. This module
instead uses softmax(log(p) / T), which is exactly equivalent: for true
logits z, softmax(z) = p implies log(p) = z - logsumexp(z), a per-example
*scalar* shift of every class's logit by the same amount. Softmax is
invariant to adding a constant to every logit of one example
(softmax(z + c) == softmax(z) for scalar c), so softmax((log(p) + c) / T) ==
softmax(log(p) / T) for any such c — meaning softmax(log(p) / T) equals
softmax(z / T) exactly, with no approximation.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend — this module only saves files
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config  # noqa: E402
from src.data import pipeline  # noqa: E402
from src.evaluate import inference  # noqa: E402

_PROB_FLOOR = 1e-12  # guards log(0) for the rare exactly-zero softmax output


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, num_bins: int) -> dict:
    """Standard equal-width-bin ECE: confidence = max predicted probability,
    correctness = whether argmax matches y_true, weighted by each bin's
    population fraction. Pure function of arrays — no model, no I/O.
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == y_true).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    bins = []
    n = len(y_true)
    for i in range(num_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Last bin is closed on both ends so confidence == 1.0 is included.
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        count = int(in_bin.sum())
        if count == 0:
            bins.append({"lo": float(lo), "hi": float(hi), "count": 0, "accuracy": None, "confidence": None})
            continue
        bin_accuracy = float(correct[in_bin].mean())
        bin_confidence = float(confidences[in_bin].mean())
        weight = count / n
        ece += weight * abs(bin_accuracy - bin_confidence)
        bins.append(
            {"lo": float(lo), "hi": float(hi), "count": count, "accuracy": bin_accuracy, "confidence": bin_confidence}
        )

    return {"ece": float(ece), "num_bins": num_bins, "bins": bins}


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """softmax(log(p) / T) — see this module's docstring for why this is
    exactly equivalent to the usual logits-based temperature scaling here.
    """
    log_probs = np.log(np.clip(probs, _PROB_FLOOR, 1.0))
    scaled = log_probs / temperature
    scaled -= scaled.max(axis=1, keepdims=True)  # numerical stability only; softmax is shift-invariant
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def _nll(y_true: np.ndarray, probs: np.ndarray) -> float:
    true_class_probs = np.clip(probs[np.arange(len(y_true)), y_true], _PROB_FLOOR, 1.0)
    return float(-np.mean(np.log(true_class_probs)))


def fit_temperature(y_true: np.ndarray, probs: np.ndarray) -> dict:
    """Grid search over config.TEMPERATURE_GRID_MIN..MAX (step
    TEMPERATURE_GRID_STEP) for the temperature minimizing validation NLL —
    exact enough for this single scalar parameter, with no new dependency
    (see this module's docstring).
    """
    # np.linspace's endpoints are exact by construction (no float
    # accumulation drift the way repeated addition in np.arange would have),
    # which matters here: a grid point landing a few ULPs past
    # TEMPERATURE_GRID_MAX must never be mistaken for "outside the
    # configured range" by a caller checking grid.max() <= TEMPERATURE_GRID_MAX.
    num_steps = round((config.TEMPERATURE_GRID_MAX - config.TEMPERATURE_GRID_MIN) / config.TEMPERATURE_GRID_STEP) + 1
    grid = np.linspace(config.TEMPERATURE_GRID_MIN, config.TEMPERATURE_GRID_MAX, num_steps)
    losses = [_nll(y_true, apply_temperature(probs, t)) for t in grid]
    best_index = int(np.argmin(losses))
    return {"temperature": float(grid[best_index]), "val_nll": float(losses[best_index])}


def _save_reliability_diagram(ece_before: dict, ece_after: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, ece_result, title in ((axes[0], ece_before, "Before"), (axes[1], ece_after, "After")):
        centers = [(b["lo"] + b["hi"]) / 2 for b in ece_result["bins"]]
        accuracies = [b["accuracy"] if b["accuracy"] is not None else 0.0 for b in ece_result["bins"]]
        ax.bar(centers, accuracies, width=1.0 / ece_result["num_bins"], edgecolor="black", alpha=0.7, label="Accuracy")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
        ax.set_xlabel("Confidence")
        ax.set_title(f"{title} temperature scaling (ECE={ece_result['ece']:.4f})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_calibration(model, model_name: str, test_y_true: np.ndarray, test_y_prob: np.ndarray) -> dict:
    """Fits temperature on validation, reports test-set ECE before/after,
    and saves a two-panel reliability diagram. `test_y_true`/`test_y_prob`
    are reused from src/evaluate/metrics.py's already-computed test-set
    predictions — this function never re-runs inference over the test
    split.
    """
    splits, _ = pipeline.load_splits()
    val_paths, val_labels = pipeline._paths_and_labels(splits["val"])
    val_ds = inference.build_eval_pipeline(val_paths, val_labels, model_name)
    val_y_true, val_y_prob = inference.predict_dataset(model, val_ds)

    fit_result = fit_temperature(val_y_true, val_y_prob)
    temperature = fit_result["temperature"]

    ece_before = expected_calibration_error(test_y_true, test_y_prob, config.ECE_NUM_BINS)
    test_y_prob_scaled = apply_temperature(test_y_prob, temperature)
    ece_after = expected_calibration_error(test_y_true, test_y_prob_scaled, config.ECE_NUM_BINS)

    figure_path = config.EVAL_FIGURES_DIR / "reliability_diagram.png"
    _save_reliability_diagram(ece_before, ece_after, figure_path)

    return {
        "temperature": temperature,
        "temperature_fit_on": "validation split only (CLAUDE.md rule 2)",
        "val_nll_at_fitted_temperature": fit_result["val_nll"],
        "test_ece_before": ece_before["ece"],
        "test_ece_after": ece_after["ece"],
        "num_bins": config.ECE_NUM_BINS,
        "reliability_diagram_figure": str(figure_path),
    }
