"""Evaluates the selected model's robustness to six corruption types a phone
camera actually produces (blur, Gaussian noise, brightness up/down,
rotation, JPEG re-compression), at three severities each, over the
PlantVillage test split. Reports macro-F1 per corruption per severity —
never plain accuracy alone (CLAUDE.md rule 4).

Reuses the same test split paths/labels src/evaluate/metrics.py already
evaluated once, but runs a SEPARATE forward pass per (corruption, severity)
combination — each is a genuinely different input distribution, not a
repeat of the same evaluation. This is part of the one final evaluation run
CLAUDE.md rule 2 reserves the test split for; no threshold, model, or
hyperparameter is chosen from any number computed here.
"""

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config  # noqa: E402
from src.data import augment, pipeline  # noqa: E402
from src.evaluate import inference  # noqa: E402

AUTOTUNE = tf.data.AUTOTUNE

_rotation_layers = {}


def _rotation_layer(degrees: float):
    """One shared, cached RandomRotation layer per distinct angle —
    factor=(f, f) (equal lo/hi) makes the sampled angle deterministic
    (always exactly f), the same trick src/data/augment.py's random rotation
    uses, just with a degenerate range instead of a real one.
    """
    if degrees not in _rotation_layers:
        factor = degrees / 360.0
        _rotation_layers[degrees] = tf.keras.layers.RandomRotation(factor=(factor, factor), fill_mode="reflect")
    return _rotation_layers[degrees]


def blur(image: tf.Tensor, severity: int) -> tf.Tensor:
    sigma = config.ROBUSTNESS_BLUR_SIGMA[severity]
    kernel_2d = augment._gaussian_kernel(config.ROBUSTNESS_BLUR_KERNEL_SIZE, sigma)
    kernel = tf.tile(kernel_2d[:, :, tf.newaxis, tf.newaxis], [1, 1, 3, 1])
    blurred = tf.nn.depthwise_conv2d(image[tf.newaxis, ...], kernel, strides=[1, 1, 1, 1], padding="SAME")
    return blurred[0]


def gaussian_noise(image: tf.Tensor, severity: int, seed: tf.Tensor) -> tf.Tensor:
    """Additive noise via stateless RNG keyed on (config.SEED + severity,
    per-image index), so repeated runs corrupt every image identically
    instead of drawing fresh noise each time.
    """
    stddev = config.ROBUSTNESS_GAUSSIAN_NOISE_STDDEV[severity]
    noise = tf.random.stateless_normal(tf.shape(image), seed=seed, stddev=stddev)
    return tf.clip_by_value(image + noise, 0.0, 1.0)


def brightness_up(image: tf.Tensor, severity: int) -> tf.Tensor:
    delta = config.ROBUSTNESS_BRIGHTNESS_UP_DELTA[severity]
    return tf.clip_by_value(tf.image.adjust_brightness(image, delta), 0.0, 1.0)


def brightness_down(image: tf.Tensor, severity: int) -> tf.Tensor:
    delta = config.ROBUSTNESS_BRIGHTNESS_DOWN_DELTA[severity]
    return tf.clip_by_value(tf.image.adjust_brightness(image, -delta), 0.0, 1.0)


def rotation(image: tf.Tensor, severity: int) -> tf.Tensor:
    degrees = config.ROBUSTNESS_ROTATION_DEGREES[severity]
    return _rotation_layer(degrees)(image, training=True)


def jpeg_compression(image: tf.Tensor, severity: int) -> tf.Tensor:
    quality = config.ROBUSTNESS_JPEG_QUALITY[severity]
    return tf.image.adjust_jpeg_quality(image, quality)


# Uniform (image, severity, seed) signature for every corruption, even the
# five that ignore `seed` (only gaussian_noise needs RNG state) — keeps
# apply_corruption's dispatch a single dict lookup rather than a branch per
# corruption needing a different call shape.
_CORRUPTION_FNS = {
    "blur": lambda image, severity, seed: blur(image, severity),
    "gaussian_noise": gaussian_noise,
    "brightness_up": lambda image, severity, seed: brightness_up(image, severity),
    "brightness_down": lambda image, severity, seed: brightness_down(image, severity),
    "rotation": lambda image, severity, seed: rotation(image, severity),
    "jpeg_compression": lambda image, severity, seed: jpeg_compression(image, severity),
}
assert set(_CORRUPTION_FNS) == set(config.ROBUSTNESS_CORRUPTIONS), (
    "_CORRUPTION_FNS must have exactly one entry per config.ROBUSTNESS_CORRUPTIONS."
)


def apply_corruption(image: tf.Tensor, corruption: str, severity: int, seed: tf.Tensor) -> tf.Tensor:
    if corruption not in _CORRUPTION_FNS:
        raise ValueError(f"Unknown corruption '{corruption}'. Known: {sorted(_CORRUPTION_FNS)}.")
    return _CORRUPTION_FNS[corruption](image, severity, seed)


def _severity_params(corruption: str, severity: int) -> dict:
    param_maps = {
        "blur": config.ROBUSTNESS_BLUR_SIGMA,
        "gaussian_noise": config.ROBUSTNESS_GAUSSIAN_NOISE_STDDEV,
        "brightness_up": config.ROBUSTNESS_BRIGHTNESS_UP_DELTA,
        "brightness_down": config.ROBUSTNESS_BRIGHTNESS_DOWN_DELTA,
        "rotation": config.ROBUSTNESS_ROTATION_DEGREES,
        "jpeg_compression": config.ROBUSTNESS_JPEG_QUALITY,
    }
    return {"value": param_maps[corruption][severity]}


def _build_corrupted_pipeline(
    paths: list, labels: list, model_name: str, corruption: str, severity: int
) -> tf.data.Dataset:
    """Mirrors src/data/pipeline.py's val/test branch (decode -> float ->
    resize+center-crop -> backbone preprocess), with one corruption step
    inserted between the deterministic crop and backbone preprocessing —
    every corrupted image still gets exactly the crop/preprocessing a clean
    test image would, isolating the corruption as the only variable.
    """
    preprocess_fn = pipeline._resolve_preprocess_fn(model_name)
    seed_base = config.SEED + severity

    ds = tf.data.Dataset.from_tensor_slices((paths, labels)).enumerate()
    ds = ds.map(lambda i, p_y: (i, pipeline._decode(p_y[0]), p_y[1]), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda i, x, y: (i, pipeline._to_float(x), y), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda i, x, y: (i, pipeline._resize_and_center_crop(x), y), num_parallel_calls=AUTOTUNE)
    ds = ds.map(
        lambda i, x, y: (
            apply_corruption(x, corruption, severity, seed=tf.stack([seed_base, tf.cast(i, tf.int32)])),
            y,
        ),
        num_parallel_calls=AUTOTUNE,
    )
    ds = ds.map(lambda x, y: (pipeline._finalize(x, preprocess_fn), y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(config.BATCH_SIZE)
    return ds.prefetch(AUTOTUNE)


def evaluate_robustness(model, model_name: str) -> dict:
    splits, _ = pipeline.load_splits()
    test_paths, test_labels = pipeline._paths_and_labels(splits["test"])

    results = {}
    for corruption in config.ROBUSTNESS_CORRUPTIONS:
        results[corruption] = {}
        for severity in config.ROBUSTNESS_SEVERITIES:
            ds = _build_corrupted_pipeline(test_paths, test_labels, model_name, corruption, severity)
            y_true, y_prob = inference.predict_dataset(model, ds)
            y_pred = np.argmax(y_prob, axis=1)
            macro_f1 = float(
                f1_score(y_true, y_pred, labels=list(range(config.NUM_CLASSES)), average="macro", zero_division=0)
            )
            results[corruption][str(severity)] = {"macro_f1": macro_f1, "params": _severity_params(corruption, severity)}
    return results
