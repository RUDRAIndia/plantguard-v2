"""Converts a trained Keras 3 model to LiteRT (TFLite) with INT8 quantization
for on-device inference. Embeds metadata: architecture, class_names in exact
index order, image size, preprocessing, seed, git commit hash, val metrics.
"""
