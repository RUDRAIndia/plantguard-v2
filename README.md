# PlantGuard v2

An on-device plant disease classifier: a TensorFlow/Keras 3 model trained on
PlantVillage, exported to LiteRT (TFLite) INT8, and served through a Kotlin +
Jetpack Compose + CameraX Android app (minSdk 24).

This is a from-scratch rebuild of a prior failed student project. See
[`CLAUDE.md`](./CLAUDE.md) for the non-negotiable rules that govern all work
in this repository — read it before writing any code here.

## Status

Repository skeleton only. No ML or app code has been written yet.

## Stack

- **Training**: Python 3.11, TensorFlow/Keras 3, run on Google Colab (free T4
  GPU) — never locally.
- **Export**: LiteRT (TFLite), INT8 quantized.
- **App**: Kotlin, Jetpack Compose, CameraX, minSdk 24.

## Data

- **PlantVillage** (Kaggle, 54,303 colour images, 38 classes): train/val/test.
- **PlantDoc** (~2,598 real field images): external test set only, never
  trained on.

## Layout

```
src/
  config.py        # all constants — no magic values live outside this file
  data/            # download, dedupe, split, input pipeline, negatives
  models/          # architecture definition
  train.py         # Colab-only training entrypoint
  evaluate/        # metrics, calibration, robustness, external eval, gradcam
  export/          # TFLite conversion and verification
tests/             # smoke tests (local, ≤200 images)
colab/             # Colab runner notebook(s)
artifacts/         # gitignored: trained models, figures, metrics JSONs
docs/
android/           # Kotlin/Compose app (not yet started)
```
