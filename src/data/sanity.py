"""Visual sanity checks for src/data/pipeline.py: saves an 8x8 grid of
augmented train images and a 4x4 grid of val images, so a human can look at
what the model is actually fed rather than trusting the code alone
(CLAUDE.md rule 11's spirit — verify with real output, not just review).

Uses pipeline.build_visualization_datasets(), not build_datasets(): that
variant skips the backbone-specific preprocess_input step, so images stay
viewable float32 [0, 1] regardless of which of the 5 candidate backbones is
eventually chosen for training.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend — this script only saves files
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config  # noqa: E402
from src.data import pipeline  # noqa: E402


def _collect(dataset: tf.data.Dataset, n: int) -> tuple:
    images, labels = [], []
    for batch_images, batch_labels in dataset:
        for image, label in zip(batch_images, batch_labels):
            images.append(image)
            labels.append(int(label))
            if len(images) == n:
                return images, labels
    raise RuntimeError(f"Dataset exhausted before collecting {n} images (got {len(images)}).")


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


def save_augmentation_grid(train_ds: tf.data.Dataset, path: Path = None, n: int = 64) -> None:
    """Saves an 8x8 grid of augmented train images to
    artifacts/augmentation_grid.png, titled with each image's class name.
    """
    path = path or (config.ARTIFACTS_DIR / "augmentation_grid.png")
    images, labels = _collect(train_ds, n)
    _save_grid(images, labels, path, rows=8, cols=8, title="Train (augmented)")
    print(f"[sanity] Wrote {path} ({n} augmented train images).")


def save_validation_grid(val_ds: tf.data.Dataset, path: Path = None, n: int = 16) -> None:
    """Saves a 4x4 grid of val images to artifacts/validation_grid.png, to
    confirm by eye that no augmentation is applied there.
    """
    path = path or (config.ARTIFACTS_DIR / "validation_grid.png")
    images, labels = _collect(val_ds, n)
    _save_grid(images, labels, path, rows=4, cols=4, title="Validation (no augmentation)")
    print(f"[sanity] Wrote {path} ({n} validation images).")


def main() -> None:
    train_ds, val_ds = pipeline.build_visualization_datasets()
    save_augmentation_grid(train_ds)
    save_validation_grid(val_ds)


if __name__ == "__main__":
    main()
