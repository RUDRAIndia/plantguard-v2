"""tf.data input pipeline: loading, augmentation, and preprocessing for
train/val/test splits. Preprocessing parameters are read from src/config.py
only, so they can be embedded verbatim in exported model metadata.

Never re-derives the split — always reads artifacts/splits.json (written by
src/data/split_report.py) and refuses to run if the recorded
split_input_hash doesn't match a fresh recomputation of
split.compute_split_inputs() (CLAUDE.md rule 1: fail loudly rather than
silently train against a stale or unreviewed split). Deliberately NOT a
repo-wide git commit comparison: split_input_hash covers only the seed,
split fractions, phash/dedupe params, PlantVillage's recorded provenance,
and the file contents of dedupe.py/phash_cluster.py/split.py — the only
things that can actually change build_split's output — so an unrelated
commit elsewhere (e.g. a pipeline.py cache tweak, a sanity.py fix) never
forces a needless ~20-minute re-dedupe on Colab.

Train and val/test are two structurally different pipelines, not the same
pipeline with an "augment or not" flag bolted on:
  - train: random-resized-crop from the native-resolution image, then the
    full stochastic stack in src/data/augment.py (see that module's
    docstring for which augmentation targets which shortcut).
  - val/test: deterministic resize-then-center-crop only, never any
    randomness.

`_decode` reads/decodes to uint8 [0, 255] at native resolution — no resize,
no float conversion yet. Any on-disk cache (config.TRAIN_DECODE_CACHE_ENABLED
/ VAL_DECODE_CACHE_DIR) sits immediately after decode, so it stores that
uint8 representation (~4x smaller than caching float32 would): PlantVillage's
real 38,013-image train split at native 256x256 resolution is ~7.5 GB as
uint8 versus ~29.9 GB as float32 — the latter would exhaust a 19.5 GB Kaggle
/kaggle/working quota partway through epoch 1 (see config.py's
TRAIN_DECODE_CACHE_ENABLED comment for the full numbers, including val's).
`_to_float` (uint8 -> float32 [0, 1]) always runs strictly *after* the cache
read (and, for train, after shuffle too — shuffling uint8 elements is
correspondingly ~4x cheaper in RAM than shuffling float32 ones), so it never
changes what pixel values reach augmentation or the deterministic val/test
crop: converting to float immediately at decode time versus later, after a
cache/shuffle round-trip that only reorders and stores/restores the same
uint8 bytes, is bit-identical either way — a pure function of the same
input, just computed later. The final per-backbone `preprocess_input`
(config.PREPROCESSING_ENTRYPOINTS) expects raw [0, 255] input, so `_finalize`
rescales the float32 [0, 1] image by 255 immediately before applying it —
skipping that step would silently feed every backbone mis-scaled input.
"""

import importlib
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import tensorflow as tf
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config  # noqa: E402
from src.data import augment, split  # noqa: E402

AUTOTUNE = tf.data.AUTOTUNE


def _diff_split_inputs(recorded: dict, current: dict) -> str:
    """Best-effort field-by-field diff of two compute_split_inputs() dicts,
    for naming exactly which input changed in a mismatch error. Handles one
    level of nesting (plantvillage_provenance, module_sha256) since that's
    the actual shape compute_split_inputs() produces.
    """
    differences = []
    for key in sorted(set(recorded) | set(current)):
        old_val, new_val = recorded.get(key), current.get(key)
        if isinstance(old_val, dict) or isinstance(new_val, dict):
            old_val, new_val = old_val or {}, new_val or {}
            for sub_key in sorted(set(old_val) | set(new_val)):
                if old_val.get(sub_key) != new_val.get(sub_key):
                    differences.append(
                        f"{key}.{sub_key} changed ({old_val.get(sub_key)!r} -> {new_val.get(sub_key)!r})"
                    )
        elif old_val != new_val:
            differences.append(f"{key} changed ({old_val!r} -> {new_val!r})")
    return "; ".join(differences) if differences else "an unspecified input changed"


def load_splits() -> tuple:
    """Reads artifacts/splits.json. Raises if it's missing, or if its
    recorded split_input_hash doesn't match a fresh recomputation of
    split.compute_split_inputs() — the split must be regenerated
    (src/data/split_report.py) any time one of those specific inputs has
    actually changed, rather than silently training against a split of
    unknown provenance. Does NOT compare git commit hashes — see this
    module's docstring for why that was too coarse.
    """
    path = config.ARTIFACTS_DIR / "splits.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run src/data/split_report.py first — "
            "src/data/pipeline.py never re-derives the split itself."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    manifest, splits = data["manifest"], data["splits"]

    recorded_hash = manifest.get("split_input_hash")
    if recorded_hash is None:
        raise RuntimeError(
            f"{path}'s manifest has no 'split_input_hash' — it was written "
            "before this guard existed. Re-run src/data/split_report.py to "
            "regenerate it."
        )

    current_inputs = split.compute_split_inputs()
    if split.hash_split_inputs(current_inputs) == recorded_hash:
        return splits, manifest

    recorded_inputs = manifest.get("split_inputs")
    if recorded_inputs is not None:
        raise RuntimeError(
            f"{path} is stale: {_diff_split_inputs(recorded_inputs, current_inputs)}. "
            "Re-run src/data/dedupe.py and src/data/split_report.py to "
            "regenerate the split."
        )
    raise RuntimeError(
        f"{path} is stale (split_input_hash mismatch), but its manifest has "
        "no 'split_inputs' breakdown to name which input changed. This hash "
        "covers: config.SEED/TRAIN_FRACTION/VAL_FRACTION/TEST_FRACTION/"
        "PHASH_SIZE/DEDUPE_HAMMING_THRESHOLD, PlantVillage's recorded "
        "dataset provenance, and the contents of dedupe.py, "
        "phash_cluster.py, and split.py. Re-run src/data/dedupe.py and "
        "src/data/split_report.py to regenerate the split."
    )


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
    """Reads and decodes an image file to uint8 [0, 255], native resolution
    — no resize, no float conversion (see _to_float below; this module's
    docstring covers why that step is deferred). Uses tf.io.decode_image
    (not a JPEG-only decoder) because config.IMAGE_EXTENSIONS also allows
    .png. decode_image's own default output dtype is uint8 for both.
    """
    raw = tf.io.read_file(path)
    return tf.io.decode_image(raw, channels=3, expand_animations=False)


def _to_float(image: tf.Tensor) -> tf.Tensor:
    """Converts a uint8 [0, 255] image (as produced by _decode, and as
    stored by the on-disk cache) to float32 [0, 1]. Deliberately its own
    step, not fused into _decode, so a cache can sit between the two and
    store the ~4x-smaller uint8 representation — see this module's
    docstring for why deferring this conversion doesn't change the pixel
    values augmentation or the val/test crop ultimately see.
    """
    return tf.image.convert_image_dtype(image, tf.float32)


def _nearest_existing_ancestor(path: Path) -> Path:
    current = path
    while not current.exists():
        if current.parent == current:
            raise RuntimeError(
                f"No existing ancestor directory found for {path} — cannot "
                "check free disk space before writing a cache there."
            )
        current = current.parent
    return current


def _projected_uint8_cache_bytes(paths: list) -> int:
    """Projects the on-disk cache size (uint8, one byte per channel) for the
    whole split, measured from the FIRST image's actual native resolution —
    read for real via PIL rather than assumed (CLAUDE.md rule 1: measure,
    don't assume PlantVillage is uniformly 256x256 just because that's
    documented elsewhere). An estimate, not an exact figure: a split with
    genuinely mixed resolutions would make this approximate, and tf.data's
    own on-disk cache format adds a small amount of per-record framing
    overhead on top of the raw pixel bytes this counts — see
    config.CACHE_DISK_SPACE_SAFETY_MARGIN.
    """
    with Image.open(paths[0]) as first_image:
        width, height = first_image.size
    return len(paths) * height * width * 3


def _assert_cache_will_fit(paths: list, cache_prefix: Path) -> None:
    """Refuses to start writing a decode cache that free disk space can't
    actually hold — CLAUDE.md rule 1: a training run that dies on a full
    disk 40 minutes into epoch 1 is exactly the late, expensive failure this
    guards against. Checked against the nearest existing ancestor directory
    of `cache_prefix` (its own directory doesn't exist yet the first time
    this runs).
    """
    projected_bytes = _projected_uint8_cache_bytes(paths)
    required_bytes = int(projected_bytes * config.CACHE_DISK_SPACE_SAFETY_MARGIN)
    existing_ancestor = _nearest_existing_ancestor(cache_prefix.parent)
    free_bytes = shutil.disk_usage(existing_ancestor).free

    if required_bytes > free_bytes:
        margin_pct = round((config.CACHE_DISK_SPACE_SAFETY_MARGIN - 1) * 100)
        raise RuntimeError(
            f"Refusing to build the on-disk decode cache at {cache_prefix}: "
            f"projected size is {projected_bytes / 1e9:.2f} GB for "
            f"{len(paths)} images ({required_bytes / 1e9:.2f} GB including a "
            f"{margin_pct}% safety margin), but only "
            f"{free_bytes / 1e9:.2f} GB is free at {existing_ancestor}. Set "
            "config.TRAIN_DECODE_CACHE_ENABLED = False (train split only) "
            "or free up disk space before training."
        )


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
    paths: list,
    labels: list,
    *,
    training: bool,
    batch_size: int,
    preprocess_fn=None,
    shuffle: bool = True,
    cache_prefix: Path = None,
) -> tf.data.Dataset:
    """Takes already-resolved absolute path strings + integer labels — not
    PlantVillage-relative paths — so this is reusable against any image
    root (PlantVillage's color/ dir via _paths_and_labels below, or e.g.
    src/train.py's --smoke path reading from config.SMOKE_DIR instead).
    Callers resolve their own (paths, labels) rather than this function
    assuming a single fixed root.

    `cache_prefix`, if given, caches right after decode — uint8, native
    resolution, pre-augmentation, pre-preprocess (this module's docstring
    covers why uint8 and why that's safe) — never after backbone-specific
    preprocessing, since that would bake in one model's normalization and
    silently serve mis-scaled cached images if a later call reuses the same
    prefix with a different model_name. Before writing anything, checks
    that free disk space can actually hold the projected cache size
    (_assert_cache_will_fit) — raises loudly rather than dying on a full
    disk partway through training. `cache_prefix is None` means no caching
    at all: the right choice for a split/purpose that's only ever read once
    (see build_datasets' test split, and build_visualization_datasets
    below) or when config.TRAIN_DECODE_CACHE_ENABLED is False. Two callers
    must never share one cache_prefix — each gets its own.

    `shuffle=False` skips the shuffle buffer entirely; needed by
    build_visualization_datasets, whose already-small, pre-sampled input
    doesn't need it, and at full-split size the buffer would have to
    prefill with every decoded image before yielding anything.
    """
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, y: (_decode(p), y), num_parallel_calls=AUTOTUNE)

    if cache_prefix is not None:
        _assert_cache_will_fit(paths, cache_prefix)
        cache_prefix.parent.mkdir(parents=True, exist_ok=True)
        ds = ds.cache(str(cache_prefix))

    if training:
        if shuffle:
            buffer_size = min(len(paths), config.SHUFFLE_BUFFER_SIZE)
            ds = ds.shuffle(buffer_size, seed=config.SEED, reshuffle_each_iteration=True)
        ds = ds.map(lambda x, y: (_to_float(x), y), num_parallel_calls=AUTOTUNE)
        size = (config.IMAGE_SIZE, config.IMAGE_SIZE)
        ds = ds.map(lambda x, y: (augment.apply(x, size), y), num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(lambda x, y: (_to_float(x), y), num_parallel_calls=AUTOTUNE)
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
    (own prefix, distinct from train's) pays for itself repeatedly and is
    always enabled. Train's cache is env-configurable
    (config.TRAIN_DECODE_CACHE_ENABLED — see its comment in src/config.py
    for the measured on-disk sizes this decision trades off). Test is read
    exactly once, ever (CLAUDE.md rule 2) — a cache would only add
    lockfile-contention risk for zero benefit, so it gets none.
    """
    batch_size = batch_size or config.BATCH_SIZE
    splits, _ = load_splits()
    preprocess_fn = _resolve_preprocess_fn(model_name)

    train_paths, train_labels = _paths_and_labels(splits["train"])
    val_paths, val_labels = _paths_and_labels(splits["val"])
    test_paths, test_labels = _paths_and_labels(splits["test"])

    train_cache_prefix = (
        config.TRAIN_DECODE_CACHE_DIR / "train" if config.TRAIN_DECODE_CACHE_ENABLED else None
    )
    train_ds = _build_pipeline(
        train_paths,
        train_labels,
        training=True,
        batch_size=batch_size,
        preprocess_fn=preprocess_fn,
        cache_prefix=train_cache_prefix,
    )
    val_ds = _build_pipeline(
        val_paths,
        val_labels,
        training=False,
        batch_size=batch_size,
        preprocess_fn=preprocess_fn,
        cache_prefix=config.VAL_DECODE_CACHE_DIR / "val",
    )
    test_ds = _build_pipeline(
        test_paths,
        test_labels,
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
    train_relative_paths = _sample_paths(splits["train"], n_train, config.SEED)
    val_relative_paths = _sample_paths(splits["val"], n_val, config.SEED)
    train_paths, train_labels = _paths_and_labels(train_relative_paths)
    val_paths, val_labels = _paths_and_labels(val_relative_paths)

    train_ds = _build_pipeline(
        train_paths, train_labels, training=True, batch_size=n_train, shuffle=False, cache_prefix=None
    )
    val_ds = _build_pipeline(
        val_paths, val_labels, training=False, batch_size=n_val, cache_prefix=None
    )
    return train_ds, val_ds, n_train, n_val
