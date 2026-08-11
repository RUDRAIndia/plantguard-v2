"""One-off local tool: builds a tiny, never-trained Keras model with the
correct (224, 224, 3) input / (38,) output shape and converts it to a
fully-integer-quantized (uint8 in, uint8 out) .tflite file, matching the
metadata contract documented in android/README.md and
src/export/to_tflite.py's docstring.

Zero training: random weights, no dataset (CLAUDE.md rule 10 — this
machine never runs training, but constructing+converting an architecture
is not training). Output is committed to
android/app/src/main/assets/model.tflite so the Android app has a
structurally valid model to build and run against before the real trained
model exists on Day 9. Swapping in the real model later is just replacing
this file plus model_metadata.json.

Usage (repo root, with the project's .venv active):
    python scripts/generate_placeholder_tflite.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.config import IMAGE_SIZE, NUM_CLASSES, REPO_ROOT

OUTPUT_PATH = REPO_ROOT / "android" / "app" / "src" / "main" / "assets" / "model.tflite"
SEED = 42
REPRESENTATIVE_SAMPLES = 100


def build_placeholder_model() -> keras.Model:
    """A minimal CNN with the right I/O shape. Weights are whatever Keras
    randomly initializes them to — this model is never trained, it only
    exists to give the converter something real to quantize.
    """
    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=tf.float32, name="image")
    x = keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = keras.layers.Conv2D(8, 3, strides=2, activation="relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax", name="probabilities")(x)
    return keras.Model(inputs, outputs, name="placeholder_plantguard")


def representative_dataset():
    # Random synthetic images, not real data — this is only here so the
    # converter can observe an activation range to derive quantization
    # scale/zero-point from. This is exactly the role a real representative
    # dataset plays later in src/export/to_tflite.py, just with noise
    # instead of real leaf photos.
    rng = np.random.default_rng(SEED)
    for _ in range(REPRESENTATIVE_SAMPLES):
        img = rng.integers(0, 256, size=(1, IMAGE_SIZE, IMAGE_SIZE, 3)).astype(np.float32)
        yield [img]


def main() -> None:
    model = build_placeholder_model()

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_bytes(tflite_model)

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]
    print(f"Wrote {len(tflite_model)} bytes to {OUTPUT_PATH}")
    print(f"input:  dtype={in_detail['dtype']} shape={in_detail['shape']} quant={in_detail['quantization']}")
    print(f"output: dtype={out_detail['dtype']} shape={out_detail['shape']} quant={out_detail['quantization']}")

    assert in_detail["dtype"] == np.uint8 and out_detail["dtype"] == np.uint8, (
        "Converter did not produce full uint8 in/out despite the requested "
        "config — investigate before committing this file; the Android app "
        "assumes raw uint8 input and a uint8 output it dequantizes itself."
    )
    assert tuple(out_detail["shape"]) == (1, NUM_CLASSES), (
        f"Output shape {tuple(out_detail['shape'])} != (1, {NUM_CLASSES}) — "
        "class_names in model_metadata.json would silently misalign with "
        "the model's output indices."
    )


if __name__ == "__main__":
    main()
