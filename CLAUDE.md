# CLAUDE.md — PlantGuard v2

This file is binding instructions for every future Claude Code session working
in this repository. PlantGuard v2 is a rebuild of a failed student project,
under a 14-day deadline, solo operator, Windows 11 + PowerShell, Intel
i5-1235U, 16 GB RAM, **no CUDA GPU**. The prior attempt failed at least in
part because of the mistakes these rules exist to prevent. Read this file
before writing any code, and re-read it if you're about to do something that
feels like a shortcut.

## Non-negotiable rules

1. **Fail loudly.** Never silently substitute a default value, never
   fuzzy/substring-match keys, never swallow exceptions to keep a pipeline
   "succeeding". If an assumption breaks, raise an exception — do not log a
   warning and continue.

2. **The test set is touched exactly once, at the end.** Never select a
   model, tune a threshold, or make any decision using the test set.
   Validation only, for everything until final reporting.

3. **Never compute R², RMSE, or any regression metric on classification
   outputs.** This is a classification problem end to end.

4. **Primary metrics are macro-F1, per-class recall, worst-class recall, and
   ECE.** Accuracy is secondary and must never be reported alone.

5. **Every number that appears in any report, README, or figure must be read
   from a metrics JSON produced by a real run.** Never hand-write, estimate,
   or carry over a number from memory or from a previous run's context.

6. **No magic values outside `src/config.py`.** All constants — paths, split
   ratios, image size, seeds, hyperparameters — live there and are imported,
   not restated. Assert `TRAIN + VAL + TEST == 1.0` and that class count
   `== 38` before any split is written or used.

7. **Every checkpoint and exported model saves metadata with it**:
   architecture, `class_names` in exact index order, image size,
   preprocessing, seed, git commit hash, and validation metrics. A model file
   without this metadata is not considered a valid artifact.

8. **No agrochemical dosages anywhere in this project.** Disease information
   is limited to: name, symptoms, generic management practice, and a
   "consult your local KVK / agricultural extension officer" line with an
   ICAR citation. Never specify a chemical, a dose, or an application
   schedule.

9. **Pin every dependency to an exact version in `requirements.txt`.** No
   ranges, no `~=`, no unpinned entries.

10. **Do not run training in this environment.** This machine has no CUDA
    GPU. Training runs on Google Colab (free T4), which clones this repo from
    GitHub. Locally you may only run smoke tests on a subset of at most 200
    images.

11. **After any code change, run the relevant test or script and paste the
    real terminal output.** Never claim something works without showing
    output from an actual run.

12. **Prefer small, single-purpose modules.** No file over ~300 lines. Split
    before you're forced to.

13. **Never delete or overwrite existing data before a validated replacement
    is fully in place.** Build the replacement to a temporary/staging
    location, validate it there, and only then swap it in for the original
    — the original must still exist, untouched, at the moment validation
    runs. This applies to datasets, checkpoints, exported models, and
    artifacts alike. (This rule exists because an earlier version of
    `src/data/download.py` deleted an existing, valid 54,305-image
    PlantVillage `color/` directory before validating its replacement, and
    the replacement run turned out to be redundant — the valid data was
    gone for nothing. See that module's docstring for the full incident.)

## Stack (locked)

- **Language/runtime**: Python 3.11.
- **Training**: TensorFlow / Keras 3, executed on Google Colab free T4 —
  never assume a local GPU.
- **On-device inference**: LiteRT (TFLite), INT8 quantized.
- **App**: Kotlin + Jetpack Compose + CameraX, `minSdk 24`.
- **Explicitly ruled out**: PyTorch, ONNX, Flutter, React Native, and any
  server/backend component. Do not propose any of these.

## Dataset

- **PlantVillage** (Kaggle, 54,303 colour images, 38 classes): source for
  train/val/test splits.
- **PlantDoc** (public, ~2,598 real field images): external test set only.
  Never trained on, never used for validation or model selection — evaluation
  against PlantDoc happens once, at the very end, alongside the PlantVillage
  test set.

## Data storage on Colab: local disk vs Drive

Google Drive, mounted via FUSE in a Colab session, writes and stats small
files at only a few dozen per second — extracting the ~54,000-image
PlantVillage `color/` set directly onto Drive takes about an hour, and
*validating* an existing Drive copy by walking it file-by-file is just as
slow. Because of this:

- **All per-image work (download, extraction, counting, validation,
  splitting, training reads) happens on the Colab VM's own local disk**
  (`/content/data`), never on the Drive FUSE mount.
- **Drive is cold storage only**: exactly one large tar file per dataset
  (e.g. `plantvillage_color.tar`), plus a small `dataset_provenance.json`
  sidecar. Nothing else ever gets written to Drive.
- **Tradeoff**: local disk is wiped whenever the Colab runtime recycles, so
  the Drive tar is what makes a session resumable — a later session copies
  the single tar file back to local disk and untars it there (fast, one
  sequential read) instead of re-downloading from Kaggle/git.
- The check for "is there already a usable copy on Drive?" must stay fast
  and walk-free (tar presence + size + the counts already recorded in
  `dataset_provenance.json`), never a full scan of extracted files over
  Drive — see rule 13 above and `src/data/download.py`'s module docstring.

## Known failure modes of PlantVillage

- **Background-shortcut learning.** Most images share uniform lab
  backgrounds. A model can learn to key off the background rather than the
  leaf/lesion, producing misleadingly high validation accuracy that collapses
  on real field images (this is exactly why PlantDoc exists as an external
  check).
- **Near-duplicate leakage across splits.** Many images are multiple photos
  of the same physical leaf taken moments apart. If these end up in different
  splits, validation/test metrics are inflated by memorization rather than
  generalization — this is why `src/data/dedupe.py` must run before
  `src/data/split.py`.
- **36:1 class imbalance.** Class sizes are highly uneven. Accuracy alone
  will hide poor performance on rare classes — this is why macro-F1,
  per-class recall, and worst-class recall are primary metrics, not
  accuracy.
- **12 of 38 classes are "healthy".** Healthy-vs-diseased is a much easier
  sub-problem than distinguishing between diseases, and can dominate
  aggregate metrics if not examined per-class.
