"""One-off local tool: builds a tiny, never-trained Keras model with the
correct (224, 224, 3) input / (38,) output shape and converts it to a
float16-quantized (float32 in, float32 out) .tflite file, matching the
metadata contract documented in android/README.md and
src/export/to_tflite.py's docstring.

**Why float16, not full-integer (uint8).** A real Kaggle export of the
selected model (MobileNetV3Large) found full INT8 quantization collapses
validation macro-F1 by 0.41 absolute (0.9678 -> 0.5545) — the known
hard-swish/squeeze-excite INT8 quantization pathology for this architecture
family. float16 measured 0.9682 (effectively lossless), so it's what
src/export/to_tflite.py actually deploys, and android/app/src/main/java/
.../PlantClassifier.kt now requires float32 input/output tensors to match.
This placeholder must match that same contract or the app would crash on
its own startup dtype assertion before a real export ever replaces it.

Zero training: random weights, no dataset (CLAUDE.md rule 10 — this
machine never runs training, but constructing+converting an architecture
is not training). Output is committed to
android/app/src/main/assets/model.tflite so the Android app has a
structurally valid model to build and run against before the real trained
model is deployed. Swapping in the real model later is just replacing this
file plus model_metadata.json.

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


def build_placeholder_model() -> keras.Model:
    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=tf.float32, name="raw_pixels_0_255")
    x = keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = keras.layers.Conv2D(8, 3, strides=2, activation="relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax", name="probabilities")(x)
    return keras.Model(inputs, outputs, name="placeholder_plantguard")


def main() -> None:
    model = build_placeholder_model()

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_bytes(tflite_model)

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]
    print(f"Wrote {len(tflite_model)} bytes to {OUTPUT_PATH}")
    print(f"input:  dtype={in_detail['dtype']} shape={in_detail['shape']}")
    print(f"output: dtype={out_detail['dtype']} shape={out_detail['shape']}")

    assert in_detail["dtype"] == np.float32 and out_detail["dtype"] == np.float32, (
        "Converter did not produce float32 in/out despite the requested config -- investigate "
        "before committing this file; the Android app assumes float32 input and reads the "
        "output directly as float32 probabilities, no dequantization."
    )
    assert tuple(out_detail["shape"]) == (1, NUM_CLASSES), (
        f"Output shape {tuple(out_detail['shape'])} != (1, {NUM_CLASSES}) — "
        "class_names in model_metadata.json would silently misalign with "
        "the model's output indices."
    )


if __name__ == "__main__":
    main()
