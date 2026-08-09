"""Pure, per-image training augmentation ops. No file I/O, no dataset
assembly — src/data/pipeline.py orchestrates these against the actual
tf.data pipeline.

PlantVillage is a single detached leaf on a uniform lab background, shot at
fixed framing, lighting, and focus. A model trained on it can hit high
validation accuracy while actually keying off the *background/framing/
lighting*, not the lesion — and then collapse on PlantDoc's real field
photos (CLAUDE.md's "background-shortcut learning" failure mode). Every op
below exists to break one specific version of that shortcut, not as generic
regularization:

- `random_resized_crop` — randomizes how much of the frame the background
  vs. leaf occupies, breaking a fixed leaf-to-background size/position cue.
- `random_flip` (left-right, up-down) — breaks any canonical orientation the
  studio setup always shot leaves in.
- `random_rotation` — breaks a fixed camera/frame angle.
- `color_jitter` (brightness/contrast/saturation/hue) — breaks reliance on
  the lab's uniform, consistent lighting/white-balance as a proxy for "this
  is the lab domain" rather than the lesion's own color/texture.
- `random_gaussian_blur` — breaks reliance on the lab photos' uniform,
  tack-sharp focus (field photos are often less sharp).
- `random_jpeg_quality` — breaks reliance on the absence of compression
  artifacts in (likely high-quality) lab captures vs. real phone/field
  photos, via a genuine JPEG encode/decode round trip.
- `random_erasing` — occludes a random patch so the model can't rely on any
  single fixed location in frame always being visible.

`apply()` composes all of the above in one fixed order for the train
pipeline. All strengths/probabilities come from src/config.py — never
inline — per CLAUDE.md rule 6.
"""

import sys
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config  # noqa: E402

# Built once at module scope: RandomRotation carries no trainable weights,
# just RNG state, so a single shared instance is safe to reuse across every
# call inside a tf.data .map() (no state leaks between images/epochs).
_rotation_layer = keras.layers.RandomRotation(
    factor=config.AUGMENT_ROTATION_FACTOR, fill_mode="reflect"
)


def random_resized_crop(image: tf.Tensor, size: tuple) -> tf.Tensor:
    """Crops a random scale/aspect-ratio window of `image` (fractions of the
    original area/aspect, from config.AUGMENT_CROP_SCALE_RANGE /
    AUGMENT_CROP_RATIO_RANGE) and resizes it to `size`. Targets: fixed
    leaf-to-background framing — the studio shots always place the leaf at
    roughly the same scale/position, which a model could use as a shortcut
    instead of the lesion.
    """
    shape = tf.shape(image)
    height = tf.cast(shape[0], tf.float32)
    width = tf.cast(shape[1], tf.float32)
    area = height * width

    scale_lo, scale_hi = config.AUGMENT_CROP_SCALE_RANGE
    ratio_lo, ratio_hi = config.AUGMENT_CROP_RATIO_RANGE
    target_area = area * tf.random.uniform([], scale_lo, scale_hi)
    aspect = tf.exp(tf.random.uniform([], tf.math.log(ratio_lo), tf.math.log(ratio_hi)))

    crop_w = tf.minimum(tf.sqrt(target_area * aspect), width)
    crop_h = tf.minimum(tf.sqrt(target_area / aspect), height)
    crop_w_int = tf.cast(crop_w, tf.int32)
    crop_h_int = tf.cast(crop_h, tf.int32)

    offset_x = tf.random.uniform([], 0, width - crop_w + 1.0, dtype=tf.float32)
    offset_y = tf.random.uniform([], 0, height - crop_h + 1.0, dtype=tf.float32)

    cropped = tf.image.crop_to_bounding_box(
        image, tf.cast(offset_y, tf.int32), tf.cast(offset_x, tf.int32), crop_h_int, crop_w_int
    )
    return tf.image.resize(cropped, size)


def random_flip(image: tf.Tensor) -> tf.Tensor:
    """Independent random horizontal + vertical flip. Targets: any
    canonical up/down or left/right orientation tied to the studio setup.
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return image


def random_rotation(image: tf.Tensor) -> tf.Tensor:
    """Random rotation via the shared Keras layer. Targets: a fixed
    camera/frame angle across the lab photos.
    """
    return _rotation_layer(image, training=True)


def color_jitter(image: tf.Tensor) -> tf.Tensor:
    """Random brightness/contrast/saturation/hue, clipped back to [0, 1].
    Targets: the lab's uniform, consistent lighting/white-balance being
    learned as a proxy for "this is the lab domain" instead of the lesion's
    own color/texture.
    """
    image = tf.image.random_brightness(image, config.AUGMENT_BRIGHTNESS_MAX_DELTA)
    image = tf.image.random_contrast(image, *config.AUGMENT_CONTRAST_RANGE)
    image = tf.image.random_saturation(image, *config.AUGMENT_SATURATION_RANGE)
    image = tf.image.random_hue(image, config.AUGMENT_HUE_MAX_DELTA)
    return tf.clip_by_value(image, 0.0, 1.0)


def _gaussian_kernel(kernel_size: int, sigma: tf.Tensor) -> tf.Tensor:
    offset = kernel_size // 2
    x = tf.range(-offset, offset + 1, dtype=tf.float32)
    g = tf.exp(-(x**2) / (2.0 * sigma**2))
    g = g / tf.reduce_sum(g)
    return tf.tensordot(g, g, axes=0)


def random_gaussian_blur(image: tf.Tensor) -> tf.Tensor:
    """With probability config.AUGMENT_GAUSSIAN_BLUR_PROBABILITY, applies a
    real depthwise-conv2d Gaussian blur with a random sigma from
    config.AUGMENT_GAUSSIAN_BLUR_SIGMA_RANGE. Targets: reliance on the lab
    photos' uniform, tack-sharp focus — real field photos are often
    somewhat blurred (motion, focus, compression).
    """

    def _blur():
        sigma = tf.random.uniform([], *config.AUGMENT_GAUSSIAN_BLUR_SIGMA_RANGE)
        kernel_2d = _gaussian_kernel(config.AUGMENT_GAUSSIAN_BLUR_KERNEL_SIZE, sigma)
        kernel = kernel_2d[:, :, tf.newaxis, tf.newaxis]
        kernel = tf.tile(kernel, [1, 1, 3, 1])
        blurred = tf.nn.depthwise_conv2d(
            image[tf.newaxis, ...], kernel, strides=[1, 1, 1, 1], padding="SAME"
        )
        return blurred[0]

    return tf.cond(
        tf.random.uniform([]) < config.AUGMENT_GAUSSIAN_BLUR_PROBABILITY,
        _blur,
        lambda: image,
    )


def random_jpeg_quality(image: tf.Tensor) -> tf.Tensor:
    """Real JPEG encode/decode round trip at a random quality from
    config.AUGMENT_JPEG_QUALITY_RANGE. Targets: reliance on the absence of
    compression artifacts in (likely high-quality) lab captures, vs. real
    phone/field photos that are often re-compressed harder.
    """
    lo, hi = config.AUGMENT_JPEG_QUALITY_RANGE
    return tf.image.random_jpeg_quality(image, lo, hi)


def random_erasing(image: tf.Tensor) -> tf.Tensor:
    """With probability config.AUGMENT_RANDOM_ERASING_PROBABILITY, fills a
    random-area/aspect rectangle with uniform noise. Targets: reliance on
    any single fixed location in frame (background corner or otherwise)
    always being present/visible.
    """

    def _erase():
        shape = tf.shape(image)
        height, width = shape[0], shape[1]
        height_f, width_f = tf.cast(height, tf.float32), tf.cast(width, tf.float32)
        area = height_f * width_f

        area_lo, area_hi = config.AUGMENT_RANDOM_ERASING_AREA_RANGE
        aspect_lo, aspect_hi = config.AUGMENT_RANDOM_ERASING_ASPECT_RANGE
        target_area = area * tf.random.uniform([], area_lo, area_hi)
        aspect = tf.exp(tf.random.uniform([], tf.math.log(aspect_lo), tf.math.log(aspect_hi)))

        erase_h = tf.minimum(tf.cast(tf.sqrt(target_area * aspect), tf.int32), height)
        erase_w = tf.minimum(tf.cast(tf.sqrt(target_area / aspect), tf.int32), width)
        erase_h = tf.maximum(erase_h, 1)
        erase_w = tf.maximum(erase_w, 1)

        top = tf.random.uniform([], 0, height - erase_h + 1, dtype=tf.int32)
        left = tf.random.uniform([], 0, width - erase_w + 1, dtype=tf.int32)

        paddings = tf.stack(
            [
                tf.stack([top, height - top - erase_h]),
                tf.stack([left, width - left - erase_w]),
            ]
        )
        keep_mask = tf.pad(
            tf.zeros([erase_h, erase_w], dtype=tf.float32),
            paddings,
            constant_values=1.0,
        )
        keep_mask = keep_mask[:, :, tf.newaxis]
        noise = tf.random.uniform(shape, 0.0, 1.0)
        return image * keep_mask + noise * (1.0 - keep_mask)

    return tf.cond(
        tf.random.uniform([]) < config.AUGMENT_RANDOM_ERASING_PROBABILITY,
        _erase,
        lambda: image,
    )


def apply(image: tf.Tensor, size: tuple) -> tf.Tensor:
    """Composes every augmentation above, in a fixed order, for the train
    pipeline. `image` must already be float32 in [0, 1]; `size` is the
    final (height, width) the random-resized-crop step resizes to.
    """
    image = random_resized_crop(image, size)
    image = random_flip(image)
    image = random_rotation(image)
    image = color_jitter(image)
    image = random_gaussian_blur(image)
    image = random_jpeg_quality(image)
    image = random_erasing(image)
    return tf.clip_by_value(image, 0.0, 1.0)
