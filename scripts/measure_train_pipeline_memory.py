"""Regression check for the train-pipeline OOM fixed by capping
config.SHUFFLE_BUFFER_SIZE (see src/config.py's comment on that constant and
src/data/pipeline.py's module docstring).

Builds src/data/pipeline.py's real training pipeline (decode -> disk cache ->
shuffle -> augment -> batch, exactly what build_datasets() wires up) against
a throwaway set of real-resolution (config.IMAGE_SIZE) images, pulls exactly
one batch, and asserts resident memory growth stays under a stated budget.

CLAUDE.md rule 10 caps local runs at config.SMOKE_MAX_IMAGES — that's too
small to reproduce the original 22 GB blowup (buffer used to be capped at
50_000, i.e. effectively the full ~38k-image training split; here
min(len(paths), config.SHUFFLE_BUFFER_SIZE) is bounded by the smoke count
regardless of the config value). What this script actually guards against is
a future regression that reintroduces full materialization before the first
batch — e.g. caching after augmentation, or shuffling on decoded tensors
sized to the whole split — by asserting the measured growth for a known
image count stays within a small, explicit multiple of that count's expected
decoded-tensor footprint.

Run: .venv/Scripts/python.exe scripts/measure_train_pipeline_memory.py
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psutil
from PIL import Image

from src import config
from src.data import pipeline

NUM_IMAGES = config.SMOKE_MAX_IMAGES  # CLAUDE.md rule 10: local cap
FAKE_CLASS = "FakeSpecies___fake_disease"

# Decoded uint8 [IMAGE_SIZE, IMAGE_SIZE, 3] footprint per image — what the
# shuffle buffer actually holds now (decode is uint8, and the uint8->float32
# conversion happens after shuffle; see src/data/pipeline.py's module
# docstring and config.py's SHUFFLE_BUFFER_SIZE comment).
BYTES_PER_DECODED_IMAGE = config.IMAGE_SIZE * config.IMAGE_SIZE * 3
# Generous multiple of NUM_IMAGES worth of decoded tensors, to absorb TF's
# one-time graph-tracing/allocator overhead on the first pull without
# masking a real full-materialization regression (which would blow well
# past this on its own for any nontrivial image count).
BUDGET_BYTES = 25 * NUM_IMAGES * BYTES_PER_DECODED_IMAGE


def _make_dataset(root: Path) -> tuple:
    class_dir = root / FAKE_CLASS
    class_dir.mkdir(parents=True)
    paths, labels = [], []
    for i in range(NUM_IMAGES):
        color = ((i * 37) % 256, (i * 91) % 256, (i * 53) % 256)
        img = Image.new("RGB", (config.IMAGE_SIZE, config.IMAGE_SIZE), color=color)
        path = class_dir / f"img{i}.jpg"
        img.save(path)
        paths.append(str(path))
        labels.append(0)
    return paths, labels


def main() -> None:
    process = psutil.Process(os.getpid())

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        color_dir = tmp_path / "color"
        color_dir.mkdir()
        paths, labels = _make_dataset(color_dir)

        rss_before = process.memory_info().rss
        ds = pipeline._build_pipeline(
            paths,
            labels,
            training=True,
            batch_size=config.BATCH_SIZE,
            preprocess_fn=None,
            shuffle=True,
            cache_prefix=tmp_path / "cache" / "train",
        )
        images, labels_batch = next(iter(ds))
        rss_after = process.memory_info().rss

    growth_bytes = rss_after - rss_before
    growth_mb = growth_bytes / (1024 * 1024)
    budget_mb = BUDGET_BYTES / (1024 * 1024)

    print(f"[measure] images={NUM_IMAGES} shuffle_buffer={min(NUM_IMAGES, config.SHUFFLE_BUFFER_SIZE)}")
    print(f"[measure] first batch shape={tuple(images.shape)} labels={tuple(labels_batch.shape)}")
    print(f"[measure] RSS before={rss_before / (1024 * 1024):.1f} MB after={rss_after / (1024 * 1024):.1f} MB")
    print(f"[measure] growth={growth_mb:.1f} MB budget={budget_mb:.1f} MB")

    if growth_bytes > BUDGET_BYTES:
        raise AssertionError(
            f"Train pipeline RSS grew {growth_mb:.1f} MB pulling one batch from "
            f"{NUM_IMAGES} images — budget is {budget_mb:.1f} MB. This is the same "
            "shape of regression that used to OOM Colab: something upstream of the "
            "first batch is materializing far more than one shuffle-buffer's worth "
            "of decoded images."
        )
    print("[measure] OK — growth within budget.")


if __name__ == "__main__":
    main()
