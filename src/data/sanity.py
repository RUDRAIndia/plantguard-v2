"""Visual sanity checks for src/data/pipeline.py: saves an 8x8 grid of
augmented train images and a 4x4 grid of val images, so a human can look at
what the model is actually fed rather than trusting the code alone
(CLAUDE.md rule 11's spirit — verify with real output, not just review).

Uses pipeline.build_visualization_datasets(), not build_datasets(): that
variant skips the backbone-specific preprocess_input step, so images stay
viewable float32 [0, 1] regardless of which of the 5 candidate backbones is
eventually chosen for training.
"""

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend — this script only saves files
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config  # noqa: E402
from src.data import pipeline  # noqa: E402


def _collect(dataset: tf.data.Dataset, n: int, batch_size: int) -> tuple:
    """Pulls only as many batches as needed to reach `n` images — never
    more. `dataset.take(num_batches_needed)` bounds how many batches are
    even requested from the pipeline; the early `return` the moment `n`
    images are collected is a second, redundant guard. Never materializes
    more than `n` images.
    """
    num_batches_needed = math.ceil(n / batch_size)
    images, labels = [], []
    for batch_images, batch_labels in dataset.take(num_batches_needed):
        for image, label in zip(batch_images, batch_labels):
            images.append(image)
            labels.append(int(label))
            if len(images) == n:
                return images, labels
    raise RuntimeError(
        f"Only collected {len(images)}/{n} images from {num_batches_needed} "
        f"batch(es) of size {batch_size} — dataset/batch_size mismatch."
    )


def _save_grid(images: list, labels: list, path: Path, rows: int, cols: int, title: str) -> None:
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 2.0))
    for ax, image, label in zip(axes.flat, images, labels):
        ax.imshow(tf.clip_by_value(image, 0.0, 1.0))
        ax.set_title(config.PLANTVILLAGE_CLASS_NAMES[label], fontsize=6)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_augmentation_grid(
    train_ds: tf.data.Dataset, batch_size: int, path: Path = None, n: int = 64
) -> None:
    """Saves an 8x8 grid of augmented train images to
    artifacts/augmentation_grid.png, titled with each image's class name.
    `batch_size` must match how `train_ds` was actually batched (e.g.
    build_visualization_datasets' returned n_train), so _collect knows how
    many batches are needed for `n` images.
    """
    path = path or (config.ARTIFACTS_DIR / "augmentation_grid.png")
    images, labels = _collect(train_ds, n, batch_size)
    _save_grid(images, labels, path, rows=8, cols=8, title="Train (augmented)")
    print(f"[sanity] Wrote {path} ({n} augmented train images).")


def save_validation_grid(
    val_ds: tf.data.Dataset, batch_size: int, path: Path = None, n: int = 16
) -> None:
    """Saves a 4x4 grid of val images to artifacts/validation_grid.png, to
    confirm by eye that no augmentation is applied there. `batch_size` must
    match how `val_ds` was actually batched (e.g.
    build_visualization_datasets' returned n_val).
    """
    path = path or (config.ARTIFACTS_DIR / "validation_grid.png")
    images, labels = _collect(val_ds, n, batch_size)
    _save_grid(images, labels, path, rows=4, cols=4, title="Validation (no augmentation)")
    print(f"[sanity] Wrote {path} ({n} validation images).")


def main() -> None:
    train_ds, val_ds, n_train, n_val = pipeline.build_visualization_datasets()
    save_augmentation_grid(train_ds, batch_size=n_train, n=n_train)
    save_validation_grid(val_ds, batch_size=n_val, n=n_val)


if __name__ == "__main__":
    main()
