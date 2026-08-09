"""Tests for src/data/sanity.py's _collect, which must never pull more
batches than are actually needed to reach n images — the OOM Cell 9 hit
was exactly this guarantee failing further upstream (see pipeline.py's
build_visualization_datasets), so this pins down the guarantee at the
_collect level directly, independent of the real image pipeline.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tensorflow as tf

from src.data import sanity


def test_collect_touches_at_most_the_needed_batches():
    batch_size = 4
    n = 10  # needs ceil(10 / 4) = 3 batches
    total_batches_available = 100  # far more than needed

    batches_pulled = []

    def generator():
        for i in range(total_batches_available):
            batches_pulled.append(i)
            images = tf.zeros([batch_size, 2, 2, 3], dtype=tf.float32)
            labels = tf.zeros([batch_size], dtype=tf.int32)
            yield images, labels

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(batch_size, 2, 2, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size,), dtype=tf.int32),
        ),
    )

    images, labels = sanity._collect(dataset, n, batch_size)

    assert len(images) == n
    assert len(labels) == n
    assert len(batches_pulled) <= math.ceil(n / batch_size), (
        f"_collect pulled {len(batches_pulled)} batch(es) but only "
        f"{math.ceil(n / batch_size)} were needed for n={n} at "
        f"batch_size={batch_size} — it is materializing more than necessary"
    )
