"""Training entrypoint, intended to run on Google Colab (T4 GPU) only.
Saves checkpoints with full metadata: architecture, class_names in exact
index order, image size, preprocessing, seed, git commit hash, val metrics.
Never touches the test set.
"""
