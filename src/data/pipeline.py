"""tf.data input pipeline: loading, augmentation, and preprocessing for
train/val/test splits. Preprocessing parameters are read from src/config.py
only, so they can be embedded verbatim in exported model metadata.

Never re-derives the split — always reads artifacts/splits.json (written by
src/data/split_report.py) and refuses to run against a split recorded under
a different git commit than the one currently checked out (CLAUDE.md rule 1:
fail loudly rather than silently train against a stale or unreviewed split).

Train and val/test are two structurally different pipelines, not the same
pipeline with an "augment or not" flag bolted on:
  - train: random-resized-crop from the native-resolution image, then the
    full stochastic stack in src/data/augment.py (see that module's
    docstring for which augmentation targets which shortcut).
  - val/test: deterministic resize-then-center-crop only, never any
    randomness.

Every image stays in float32 [0, 1] through decode/augment/crop; the final
per-backbone `preprocess_input` (config.PREPROCESSING_ENTRYPOINTS) expects
raw [0, 255] input, so `_finalize` rescales by 255 immediately before
applying it — skipping that step would silently feed every backbone
mis-scaled input.
"""

import importlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config  # noqa: E402
from src.data import augment  # noqa: E402

AUTOTUNE = tf.data.AUTOTUNE


def load_splits() -> tuple:
    """Reads artifacts/splits.json. Raises if it's missing, or if its
    manifest's git_commit_hash doesn't match the currently checked-out
    commit — the split must be regenerated (src/data/split_report.py) any
    time the repo has moved on, rather than silently training against a
    split from an unknown prior state.
    """
    path = config.ARTIFACTS_DIR / "splits.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run src/data/split_report.py first — "
            "src/data/pipeline.py never re-derives the split itself."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    manifest, splits = data["manifest"], data["splits"]

    current_commit = config.get_git_commit_hash()
    recorded_commit = manifest["git_commit_hash"]
    if recorded_commit != current_commit:
        raise RuntimeError(
            f"{path} was generated at commit {recorded_commit}, but the "
            f"currently checked-out commit is {current_commit}. Re-run "
            "src/data/split_report.py to regenerate the split before "
            "building the pipeline, rather than training against a split "
            "of unknown provenance relative to the current code."
        )
    return splits, manifest


def _paths_and_labels(relative_paths: list) -> tuple:
    """Turns ["ClassName/img.jpg", ...] into (absolute_path_strings,
    integer_labels), using config.PLANTVILLAGE_CLASS_NAMES' index order —
    the same order every checkpoint/export's class_names metadata must use
    (CLAUDE.md rule 7).
    """
    class_index = {name: i for i, name in enumerate(config.PLANTVILLAGE_CLASS_NAMES)}
    paths, labels = [], []
    for relative_path in relative_paths:
        class_name = relative_path.split("/")[0]
        paths.append(str(config.PLANTVILLAGE_COLOR_DIR / relative_path))
        labels.append(class_index[class_name])
    return paths, labels


def _sample_paths(relative_paths: list, n: int, seed: int) -> list:
    """Deterministically samples `n` paths from `relative_paths` via a
    cheap Python-level shuffle over path strings — no image ever touched,
    negligible memory regardless of split size. For callers that only need
    a small representative subset rather than the full split (e.g.
    build_visualization_datasets below), so they never have to route
    through a tf.data shuffle buffer sized for the whole split.
    """
    if n > len(relative_paths):
        raise ValueError(
            f"Requested {n} images but only {len(relative_paths)} are available."
        )
    ordered = sorted(relative_paths)  # deterministic pre-shuffle order
    rng = random.Random(seed)
    rng.shuffle(ordered)
    return ordered[:n]


def compute_class_weights(train_relative_paths: list) -> dict:
    """Standard "balanced" class weights — weight[c] = total / (num_classes
    * count[c]) — computed from the TRAIN split only. Returned as a plain
    dict {class_index: weight}; never applied to the dataset itself, so the
    caller decides how (e.g. passing it to model.fit(class_weight=...)).
    """
    counts = Counter(path.split("/")[0] for path in train_relative_paths)
    total = sum(counts.values())
    num_classes = len(config.PLANTVILLAGE_CLASS_NAMES)

    weights = {}
    for index, class_name in enumerate(config.PLANTVILLAGE_CLASS_NAMES):
        count = counts.get(class_name, 0)
        if count == 0:
            raise RuntimeError(
                f"Class '{class_name}' has zero images in the train split — "
                "cannot compute a class weight for it. This should be "
                "impossible given src/data/split.py's per-class assertions; "
                "re-run src/data/split_report.py and inspect artifacts/splits.json."
            )
        weights[index] = total / (num_classes * count)
    return weights


def _resolve_preprocess_fn(model_name: str):
    if model_name not in config.PREPROCESSING_ENTRYPOINTS:
        raise ValueError(
            f"No preprocessing entrypoint configured for model '{model_name}'. "
            f"Known models: {sorted(config.PREPROCESSING_ENTRYPOINTS)}."
        )
    module_path, function_name = config.PREPROCESSING_ENTRYPOINTS[model_name]
    try:
        module = importlib.import_module(module_path)
        return getattr(module, function_name)
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            f"Could not resolve preprocessing function '{function_name}' from "
            f"'{module_path}' for model '{model_name}': {exc}."
        ) from exc


def _decode(path: tf.Tensor) -> tf.Tensor:
    """Reads and decodes an image file to float32 [0, 1], native resolution
    (no resize yet). Uses tf.io.decode_image (not a JPEG-only decoder)
    because config.IMAGE_EXTENSIONS also allows .png.
    """
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    return tf.image.convert_image_dtype(image, tf.float32)


def _resize_and_center_crop(image: tf.Tensor) -> tf.Tensor:
    """Deterministic val/test path: resize to config.VAL_TEST_RESIZE_SIZE,
    then center-crop to config.IMAGE_SIZE. No randomness, ever.
    """
    resized = tf.image.resize(
        image, (config.VAL_TEST_RESIZE_SIZE, config.VAL_TEST_RESIZE_SIZE)
    )
    return tf.image.resize_with_crop_or_pad(resized, config.IMAGE_SIZE, config.IMAGE_SIZE)


def _finalize(image: tf.Tensor, preprocess_fn) -> tf.Tensor:
    """Rescales [0, 1] -> [0, 255] and applies the backbone-specific
    preprocess_input. See this module's docstring for why the *255 step is
    required (every keras.applications preprocess_input expects [0, 255]).
    """
    return preprocess_fn(image * 255.0)


def _build_pipeline(
    relative_paths: list,
    *,
    training: bool,
    batch_size: int,
    preprocess_fn=None,
    shuffle: bool = True,
    cache_prefix: Path = None,
) -> tf.data.Dataset:
    """`cache_prefix`, if given, caches right after decode (native
    resolution, pre-augmentation, pre-preprocess) — never after
    backbone-specific preprocessing, since that would bake in one model's
    normalization and silently serve mis-scaled cached images if a later
    call reuses the same prefix with a different model_name. `cache_prefix
    is None` means no caching at all: the right choice for a split/purpose
    that's only ever read once (see build_datasets' test split, and
    build_visualization_datasets below). Two callers must never share one
    cache_prefix — each gets its own.

    `shuffle=False` skips the shuffle buffer entirely; needed by
    build_visualization_datasets, whose already-small, pre-sampled input
    doesn't need it, and at full-split size the buffer would have to
    prefill with every decoded image before yielding anything.
    """
    paths, labels = _paths_and_labels(relative_paths)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, y: (_decode(p), y), num_parallel_calls=AUTOTUNE)

    if cache_prefix is not None:
        cache_prefix.parent.mkdir(parents=True, exist_ok=True)
        ds = ds.cache(str(cache_prefix))

    if training:
        if shuffle:
            buffer_size = min(len(paths), config.SHUFFLE_BUFFER_SIZE)
            ds = ds.shuffle(buffer_size, seed=config.SEED, reshuffle_each_iteration=True)
        size = (config.IMAGE_SIZE, config.IMAGE_SIZE)
        ds = ds.map(lambda x, y: (augment.apply(x, size), y), num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(lambda x, y: (_resize_and_center_crop(x), y), num_parallel_calls=AUTOTUNE)

    if preprocess_fn is not None:
        ds = ds.map(lambda x, y: (_finalize(x, preprocess_fn), y), num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size)
    return ds.prefetch(AUTOTUNE)


def build_datasets(model_name: str, batch_size: int = None) -> tuple:
    """The training-ready entrypoint: reads artifacts/splits.json, builds
    train/val/test tf.data.Dataset objects with the given backbone's
    preprocessing applied, and returns (train_ds, val_ds, test_ds,
    class_weights).

    Cache strategy differs per split, not uniformly: val is read once per
    epoch across many epochs of real training, so an on-disk decode cache
    (own prefix, distinct from train's) pays for itself repeatedly. Test is
    read exactly once, ever (CLAUDE.md rule 2) — a cache would only add
    lockfile-contention risk for zero benefit, so it gets none.
    """
    batch_size = batch_size or config.BATCH_SIZE
    splits, _ = load_splits()
    preprocess_fn = _resolve_preprocess_fn(model_name)

    train_ds = _build_pipeline(
        splits["train"],
        training=True,
        batch_size=batch_size,
        preprocess_fn=preprocess_fn,
        cache_prefix=config.TRAIN_DECODE_CACHE_DIR / "train",
    )
    val_ds = _build_pipeline(
        splits["val"],
        training=False,
        batch_size=batch_size,
        preprocess_fn=preprocess_fn,
        cache_prefix=config.VAL_DECODE_CACHE_DIR / "val",
    )
    test_ds = _build_pipeline(
        splits["test"],
        training=False,
        batch_size=batch_size,
        preprocess_fn=preprocess_fn,
        cache_prefix=None,
    )
    class_weights = compute_class_weights(splits["train"])
    return train_ds, val_ds, test_ds, class_weights


def build_visualization_datasets(n_train: int = 64, n_val: int = 16) -> tuple:
    """Same augmentation/preprocessing as build_datasets, but scoped to a
    small, cheaply-sampled subset of each split (via _sample_paths) rather
    than the full split — used only by src/data/sanity.py, which never
    needs more than a couple dozen images. Deliberately skips both the
    shuffle buffer and any caching: with only n images ever read, a
    config.SHUFFLE_BUFFER_SIZE-sized buffer of decoded full-split images
    would force tens of GB into RAM before yielding a single one, and a
    one-shot read gets no benefit from caching. Skips backbone
    preprocessing too, so images stay viewable float32 [0, 1] regardless of
    which backbone is eventually chosen for training.

    Returns (train_ds, val_ds, n_train, n_val) — the counts double as each
    dataset's batch size, so callers (sanity.py) know exactly how many
    images one batch holds without duplicating these defaults.
    """
    splits, _ = load_splits()
    train_paths = _sample_paths(splits["train"], n_train, config.SEED)
    val_paths = _sample_paths(splits["val"], n_val, config.SEED)

    train_ds = _build_pipeline(
        train_paths, training=True, batch_size=n_train, shuffle=False, cache_prefix=None
    )
    val_ds = _build_pipeline(val_paths, training=False, batch_size=n_val, cache_prefix=None)
    return train_ds, val_ds, n_train, n_val
