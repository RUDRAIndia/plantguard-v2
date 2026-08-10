"""Builds a tiny, fully synthetic on-disk dataset under config.SMOKE_DIR for
src/train.py's --smoke runs (CLAUDE.md rule 10: local runs are capped at
config.SMOKE_MAX_IMAGES images). Deliberately independent of
src/data/split.py's real split machinery and artifacts/splits.json — a
smoke run must be exercisable on a bare machine that has never downloaded
PlantVillage, and this dataset has no test partition at all, so a smoke run
can never touch the real train/val/test split CLAUDE.md rule 2 reserves for
the end of the project.

Images use real class names (the first config.SMOKE_NUM_CLASSES entries of
config.PLANTVILLAGE_CLASS_NAMES, in their real global index order), so the
38-wide classifier head is exercised against genuine label positions rather
than a remapped 0..N range.
"""

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config  # noqa: E402


def _write_synthetic_image(path: Path, class_index: int, image_index: int) -> None:
    """Deterministic per (class_index, image_index) — same inputs always
    produce the same file, so re-running build_smoke_dataset() is a cheap
    no-op once the images already exist (see the is_file() guard below).
    """
    size = (config.VAL_TEST_RESIZE_SIZE, config.VAL_TEST_RESIZE_SIZE)
    color = (
        (class_index * 61) % 256,
        (image_index * 37) % 256,
        (class_index * 97 + image_index * 13) % 256,
    )
    image = Image.new("RGB", size, color=color)
    # Cheap deterministic texture so augmentation ops (crop/flip/blur/JPEG)
    # have more than a flat color field to act on — irrelevant to
    # correctness, only there so the real image-processing ops in
    # src/data/augment.py run against non-degenerate input.
    for x in range(0, size[0], 8):
        for y in range(0, size[1], 8):
            if (x + y + image_index) % 32 == 0:
                image.putpixel((x, y), (255, 255, 255))

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=90)


def build_smoke_dataset() -> tuple:
    """Writes (if not already present) config.SMOKE_NUM_CLASSES classes'
    worth of synthetic images under config.SMOKE_DIR and returns
    (train_paths, train_labels, val_paths, val_labels) — absolute path
    strings and integer labels (config.PLANTVILLAGE_CLASS_NAMES' index
    order), ready for src/data/pipeline.py's _build_pipeline().
    """
    class_names = config.PLANTVILLAGE_CLASS_NAMES[: config.SMOKE_NUM_CLASSES]

    train_paths, train_labels, val_paths, val_labels = [], [], [], []
    for class_index, class_name in enumerate(class_names):
        class_dir = config.SMOKE_DIR / class_name

        for i in range(config.SMOKE_TRAIN_IMAGES_PER_CLASS):
            path = class_dir / f"train_{i:03d}.jpg"
            if not path.is_file():
                _write_synthetic_image(path, class_index, i)
            train_paths.append(str(path))
            train_labels.append(class_index)

        for i in range(config.SMOKE_VAL_IMAGES_PER_CLASS):
            path = class_dir / f"val_{i:03d}.jpg"
            if not path.is_file():
                _write_synthetic_image(path, class_index, config.SMOKE_TRAIN_IMAGES_PER_CLASS + i)
            val_paths.append(str(path))
            val_labels.append(class_index)

    return train_paths, train_labels, val_paths, val_labels
