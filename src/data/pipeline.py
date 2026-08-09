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
    cache_dir: Path = None,
) -> tf.data.Dataset:
    paths, labels = _paths_and_labels(relative_paths)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, y: (_decode(p), y), num_parallel_calls=AUTOTUNE)

    if training:
        cache_path = cache_dir or config.TRAIN_DECODE_CACHE_DIR
        cache_path.mkdir(parents=True, exist_ok=True)
        ds = ds.cache(str(cache_path / "train"))
        buffer_size = min(len(paths), config.SHUFFLE_BUFFER_SIZE)
        ds = ds.shuffle(buffer_size, seed=config.SEED, reshuffle_each_iteration=True)
        size = (config.IMAGE_SIZE, config.IMAGE_SIZE)
        ds = ds.map(lambda x, y: (augment.apply(x, size), y), num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(lambda x, y: (_resize_and_center_crop(x), y), num_parallel_calls=AUTOTUNE)

    if preprocess_fn is not None:
        ds = ds.map(lambda x, y: (_finalize(x, preprocess_fn), y), num_parallel_calls=AUTOTUNE)

    if not training:
        ds = ds.cache()

    ds = ds.batch(batch_size)
    return ds.prefetch(AUTOTUNE)


def build_datasets(model_name: str, batch_size: int = None) -> tuple:
    """The training-ready entrypoint: reads artifacts/splits.json, builds
    train/val/test tf.data.Dataset objects with the given backbone's
    preprocessing applied, and returns (train_ds, val_ds, test_ds,
    class_weights).
    """
    batch_size = batch_size or config.BATCH_SIZE
    splits, _ = load_splits()
    preprocess_fn = _resolve_preprocess_fn(model_name)

    train_ds = _build_pipeline(
        splits["train"], training=True, batch_size=batch_size, preprocess_fn=preprocess_fn
    )
    val_ds = _build_pipeline(
        splits["val"], training=False, batch_size=batch_size, preprocess_fn=preprocess_fn
    )
    test_ds = _build_pipeline(
        splits["test"], training=False, batch_size=batch_size, preprocess_fn=preprocess_fn
    )
    class_weights = compute_class_weights(splits["train"])
    return train_ds, val_ds, test_ds, class_weights


def build_visualization_datasets(batch_size: int = 64) -> tuple:
    """Same pipeline as build_datasets, but WITHOUT backbone preprocessing —
    images stay viewable float32 [0, 1] — used only by src/data/sanity.py so
    the saved grids look like real photos regardless of which backbone is
    eventually chosen.
    """
    splits, _ = load_splits()
    train_ds = _build_pipeline(splits["train"], training=True, batch_size=batch_size)
    val_ds = _build_pipeline(splits["val"], training=False, batch_size=batch_size)
    return train_ds, val_ds
