"""Converts a trained Keras 3 model to LiteRT (TFLite) with INT8 quantization
for on-device inference. Embeds metadata: architecture, class_names in exact
index order, image size, preprocessing, seed, git commit hash, val metrics.

The Android app (android/) was built ahead of this exporter against a
metadata contract this module must conform to: full-integer quantization
with uint8 input AND uint8 output (config.TFLITE_CONFIG), class_names copied
verbatim from config.PLANTVILLAGE_CLASS_NAMES in order, and a sibling
model_metadata.json (schema documented in android/README.md) carrying
image_size, confidence_threshold, and that same class_names array. The real
export must overwrite both android/app/src/main/assets/model.tflite and
model_metadata.json together — see scripts/generate_placeholder_tflite.py
for the placeholder that currently stands in for this module's output.
"""
