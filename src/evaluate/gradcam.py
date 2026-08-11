"""Grad-CAM heatmaps for a sample of correct predictions, incorrect
predictions, and PlantDoc failures — the cheapest available evidence for
whether the model attends to the actual lesion or to shortcut features like
the studio background (CLAUDE.md's background-shortcut failure mode).
Deliberately includes failures, not just successes: an honest sample is the
point, not a curated highlight reel.

Reuses the (y_true, y_pred, paths) already computed by src/evaluate/metrics.py
(PlantVillage test set) and src/evaluate/external.py (PlantDoc) — this module
never re-runs inference over either split, it only re-decodes the small
handful of individual images actually sampled for visualization.
"""

import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend — this module only saves files
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config  # noqa: E402
from src.data import pipeline  # noqa: E402
from src.models import build  # noqa: E402


def select_indices(mask: np.ndarray, n: int, seed: int) -> list:
    """Deterministically picks up to n indices where `mask` is True, shuffled
    first (random.Random(seed)) so the sample isn't just "the first n in
    file order" (which would skew toward whichever class sorts first).
    Returns fewer than n if fewer than n qualify — never raises for a small
    failure count, which is itself informative (e.g. a very accurate class).
    """
    candidates = list(np.flatnonzero(mask))
    random.Random(seed).shuffle(candidates)
    return candidates[:n]


def _last_conv_layer(backbone: tf.keras.Model) -> tf.keras.layers.Layer:
    """The deepest layer in `backbone` with a 4D (batch, h, w, channels)
    output — the last spatial feature map before global-average pooling
    collapses it, i.e. exactly the layer Grad-CAM needs gradients through.
    Found structurally (by output rank), never a hardcoded per-architecture
    layer name, so it works for any of config.CANDIDATE_MODELS.
    """
    for layer in reversed(backbone.layers):
        try:
            rank = len(layer.output.shape)
        except AttributeError:
            continue
        if rank == 4:
            return layer
    raise RuntimeError(f"No 4D-output (conv) layer found in backbone '{backbone.name}' for Grad-CAM.")


def build_grad_model(model: tf.keras.Model) -> tf.keras.Model:
    """Keras 3's functional API cannot trace a new Model directly from the
    OUTER model's input through to a layer nested inside the backbone
    sub-model (attempting it raises "Output ... is not connected to
    inputs") — so this builds the conv-output extractor from the backbone's
    OWN input instead, stopping at the backbone's pooled output. The head
    (Dropout + Dense) is re-applied separately, inside the same
    GradientTape as the conv-layer extraction, by apply_head below —
    together they reproduce the outer model's full forward pass while still
    exposing the intermediate conv-layer tensor gradients need to flow
    through.
    """
    backbone = build.get_backbone(model)
    conv_layer = _last_conv_layer(backbone)
    return tf.keras.Model(inputs=backbone.input, outputs=[conv_layer.output, backbone.output])


def apply_head(model: tf.keras.Model, backbone_output: tf.Tensor) -> tf.Tensor:
    """Re-applies build.py's Dropout(head_dropout) + Dense(predictions) head
    to a backbone output tensor, at inference time (training=False, so
    Dropout is a no-op) — see build_grad_model's docstring for why this
    can't just be part of one nested Model instead.
    """
    dropout = model.get_layer("head_dropout")
    dense = model.get_layer("predictions")
    return dense(dropout(backbone_output, training=False), training=False)


def compute_heatmap(
    model: tf.keras.Model, grad_model: tf.keras.Model, model_input: np.ndarray, target_class_idx: int
) -> np.ndarray:
    """model_input is a single already-backbone-preprocessed image (H, W, 3).
    Returns a (h, w) heatmap normalized to [0, 1], h/w matching the target
    conv layer's spatial resolution (smaller than the input image — resized
    up at overlay time).
    """
    image_batch = tf.constant(model_input[np.newaxis, ...])
    with tf.GradientTape() as tape:
        conv_output, backbone_output = grad_model(image_batch)
        predictions = apply_head(model, backbone_output)
        class_score = predictions[:, target_class_idx]
    grads = tape.gradient(class_score, conv_output)
    if grads is None:
        raise RuntimeError("Grad-CAM gradient computation returned None — check grad_model's output wiring.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0.0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def _load_viewable_image(path: str) -> np.ndarray:
    """Float32 [0, 1], resized+center-cropped exactly like the deterministic
    val/test path (src/data/pipeline.py), but WITHOUT backbone-specific
    preprocessing — so it stays directly viewable regardless of which
    backbone made the prediction.
    """
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return pipeline._resize_and_center_crop(image).numpy()


def _load_model_input(viewable_image: np.ndarray, preprocess_fn) -> np.ndarray:
    return pipeline._finalize(tf.constant(viewable_image), preprocess_fn).numpy()


def _save_overlay(viewable_image: np.ndarray, heatmap: np.ndarray, path: Path, title: str) -> None:
    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], viewable_image.shape[:2]).numpy()[..., 0]
    colored = matplotlib.colormaps["jet"](heatmap_resized)[..., :3]
    overlay = np.clip(
        viewable_image * (1 - config.GRADCAM_OVERLAY_ALPHA) + colored * config.GRADCAM_OVERLAY_ALPHA, 0.0, 1.0
    )

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
    axes[0].imshow(viewable_image)
    axes[0].set_title("Original", fontsize=9)
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title("Grad-CAM", fontsize=9)
    axes[1].axis("off")
    fig.suptitle(title, fontsize=8)
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _process_group(
    model: tf.keras.Model,
    grad_model: tf.keras.Model,
    preprocess_fn,
    paths: list,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    indices: list,
    group_name: str,
    class_names: tuple,
    output_dir: Path,
) -> list:
    samples = []
    for rank, idx in enumerate(indices):
        viewable = _load_viewable_image(paths[idx])
        model_input = _load_model_input(viewable, preprocess_fn)
        true_class = class_names[int(y_true[idx])]
        predicted_class = class_names[int(y_pred[idx])]

        heatmap = compute_heatmap(model, grad_model, model_input, int(y_pred[idx]))
        figure_path = output_dir / f"{group_name}_{rank:02d}.png"
        _save_overlay(viewable, heatmap, figure_path, f"{group_name} — true: {true_class} / pred: {predicted_class}")

        samples.append(
            {
                "group": group_name,
                "source_path": paths[idx],
                "true_class": true_class,
                "predicted_class": predicted_class,
                "figure_path": str(figure_path),
            }
        )
    return samples


def generate_gradcam_samples(
    model: tf.keras.Model,
    model_name: str,
    test_true: np.ndarray,
    test_pred: np.ndarray,
    test_paths: list,
    plantdoc_true: np.ndarray,
    plantdoc_pred: np.ndarray,
    plantdoc_paths: list,
) -> dict:
    grad_model = build_grad_model(model)
    preprocess_fn = pipeline._resolve_preprocess_fn(model_name)
    output_dir = config.EVAL_FIGURES_DIR / "gradcam"

    correct_idx = select_indices(test_pred == test_true, config.GRADCAM_NUM_CORRECT_SAMPLES, config.SEED)
    incorrect_idx = select_indices(test_pred != test_true, config.GRADCAM_NUM_INCORRECT_SAMPLES, config.SEED + 1)
    plantdoc_failure_idx = select_indices(
        plantdoc_pred != plantdoc_true, config.GRADCAM_NUM_PLANTDOC_FAILURE_SAMPLES, config.SEED + 2
    )

    samples = []
    samples += _process_group(
        model, grad_model, preprocess_fn, test_paths, test_true, test_pred, correct_idx,
        "plantvillage_test_correct", config.PLANTVILLAGE_CLASS_NAMES, output_dir,
    )
    samples += _process_group(
        model, grad_model, preprocess_fn, test_paths, test_true, test_pred, incorrect_idx,
        "plantvillage_test_incorrect", config.PLANTVILLAGE_CLASS_NAMES, output_dir,
    )
    samples += _process_group(
        model, grad_model, preprocess_fn, plantdoc_paths, plantdoc_true, plantdoc_pred, plantdoc_failure_idx,
        "plantdoc_failure", config.PLANTVILLAGE_CLASS_NAMES, output_dir,
    )

    return {
        "num_samples": len(samples),
        "num_correct_sampled": len(correct_idx),
        "num_incorrect_sampled": len(incorrect_idx),
        "num_plantdoc_failures_sampled": len(plantdoc_failure_idx),
        "output_dir": str(output_dir),
        "samples": samples,
    }
