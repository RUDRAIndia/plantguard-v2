"""Single source of truth for all constants: paths, split ratios, class count,
image size, preprocessing params, seeds, and hyperparameters. No magic values
may appear anywhere else in this project — import them from here instead.
"""

import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------
# Three environments this project ever runs in:
#   - Kaggle Notebooks: training (two Tesla T4s). Detected by /kaggle/input
#     existing, NOT an environment variable — Kaggle doesn't reliably set one
#     the way "google.colab" is reliably importable, but /kaggle/input is
#     always mounted (read-only) on every Kaggle notebook session, including
#     ones with no datasets attached (it still exists, just empty), so its
#     presence is a solid environment signal.
#   - Google Colab: training fallback, only usable while a free-tier GPU is
#     actually attached (not guaranteed — see CLAUDE.md). "google.colab" is
#     only ever importable inside a Colab runtime.
#   - Local Windows dev machine: code and tests only, per CLAUDE.md rule 10
#     — no CUDA GPU, so never real training, only smoke tests on <=200
#     images.
# Checked in this order because a Kaggle notebook could theoretically also
# satisfy some Colab-like signal in the future; /kaggle/input is the more
# specific, more reliable check, so it's resolved first.
IS_KAGGLE = Path("/kaggle/input").is_dir()
IS_COLAB = (not IS_KAGGLE) and "google.colab" in sys.modules

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Repo root is derived from this file's own location, so it resolves
# correctly regardless of platform or where the repo happens to be cloned
# (e.g. C:\Projects\plantguard-v2 locally, /content/plantguard-v2 on Colab,
# /kaggle/working/plantguard-v2 on Kaggle).
REPO_ROOT = Path(__file__).resolve().parent.parent

ARTIFACTS_DIR = REPO_ROOT / "artifacts"

# DATA_ROOT is where all per-image work happens (extraction, counting,
# splitting, caches, checkpoints) and — critically — every path DATA_ROOT
# derives must be WRITABLE. On Kaggle specifically, /kaggle/input (where the
# real datasets are mounted) is READ-ONLY: any write attempted there fails,
# so DATA_ROOT must never point inside it. /kaggle/working is the one
# location Kaggle guarantees is writable for the session (see CHECKPOINT_DIR
# below for exactly how far that guarantee extends). On Colab, DATA_ROOT
# MUST be the VM's own local disk, never the Drive FUSE mount — Drive
# writes/stats small files at only a few dozen per second, so any per-file
# operation is unusably slow or effectively never completes there. See
# src/data/download.py's module docstring for the full cold-storage-vs-
# local-disk design and CLAUDE.md's Drive rule.
if IS_KAGGLE:
    DATA_ROOT = Path("/kaggle/working/data")
    # No Drive on Kaggle — PlantVillage and the negatives source are
    # mounted directly as read-only inputs (see KAGGLE_PLANTVILLAGE_COLOR_DIR
    # / KAGGLE_NEGATIVES_SEG_TRAIN_DIR below), so there is nothing to persist
    # a tar copy of. PlantDoc still git-clones fresh each session (small
    # enough, ~2,598 images, that re-cloning every session is cheap) rather
    # than needing its own persistence mechanism.
    DRIVE_DATA_ROOT = None
elif IS_COLAB:
    DATA_ROOT = Path("/content/data")
    # Drive is cold storage ONLY: a single tar per dataset, nothing else.
    # It exists purely so a session can resume without re-downloading, since
    # /content (DATA_ROOT above) is wiped whenever the Colab runtime recycles.
    DRIVE_DATA_ROOT = Path("/content/drive/MyDrive/plantguard-data")
else:
    # Local Windows dev machine: DATA_ROOT is only ever used to hold the
    # ≤200-image smoke-test subset (see SMOKE_MAX_IMAGES below). Full-size
    # dataset downloads must never be triggered here — see the IS_COLAB
    # guard in src/data/download.py. There is no Drive concept outside Colab.
    DATA_ROOT = REPO_ROOT / "data"
    DRIVE_DATA_ROOT = None

# Sidecar on Drive: a copy of DATASET_PROVENANCE_PATH (see below), written
# alongside each tar so a *fresh* Colab session — a fresh git clone, whose
# gitignored local artifacts/ has no history — can still read last session's
# observed counts without walking any files. This is what makes the fast,
# walk-free integrity check in src/data/download.py possible. None on
# Kaggle/local, where there is no Drive.
DRIVE_PROVENANCE_PATH = (DRIVE_DATA_ROOT / "dataset_provenance.json") if DRIVE_DATA_ROOT else None
PLANTVILLAGE_COLOR_TAR = (DRIVE_DATA_ROOT / "plantvillage_color.tar") if DRIVE_DATA_ROOT else None
PLANTDOC_TAR = (DRIVE_DATA_ROOT / "plantdoc.tar") if DRIVE_DATA_ROOT else None

# Coarse sanity floors for the fast Drive integrity check: catch an empty or
# obviously-truncated tar without opening/reading it. Deliberately loose —
# the authoritative check is the observed image/class counts recorded in
# dataset_provenance.json, not tar size.
MIN_PLANTVILLAGE_TAR_BYTES = 500_000_000
MIN_PLANTDOC_TAR_BYTES = 20_000_000

# Verified real paths on a running Kaggle notebook instance (per-file
# checked, not guessed): both datasets are attached as read-only notebook
# inputs, already extracted — no download, no zip, no tar. None outside
# Kaggle; only ever read, never written to (rule 13 / the read-only-mount
# constraint above) — src/data/download.py and src/data/negatives.py point
# at these directly instead of downloading.
if IS_KAGGLE:
    # The same mount also contains segmented/, grayscale/, and a wrapper
    # directory literally named "plantvillage dataset" — this is the
    # top-level color/ directory specifically, not the wrapper.
    KAGGLE_PLANTVILLAGE_COLOR_DIR = Path(
        "/kaggle/input/datasets/abdallahalidev/plantvillage-dataset/color"
    )
    # Doubled nesting (seg_train/seg_train/<category>/...), same quirk
    # src/data/fetch.py's _resolve_through_wrapper_dirs already handles for
    # a downloaded zip — src/data/negatives.py resolves it here too instead
    # of downloading.
    KAGGLE_NEGATIVES_SEG_TRAIN_DIR = Path(
        "/kaggle/input/datasets/puneet6060/intel-image-classification/seg_train"
    )
else:
    KAGGLE_PLANTVILLAGE_COLOR_DIR = None
    KAGGLE_NEGATIVES_SEG_TRAIN_DIR = None

PLANTVILLAGE_DIR = DATA_ROOT / "plantvillage"
# Only the "color" variant is ever used for training — never grayscale or
# segmented. On Kaggle this IS the read-only mounted directory itself (never
# a DATA_ROOT-derived, writable path) — see KAGGLE_PLANTVILLAGE_COLOR_DIR
# above. Elsewhere it's its own leaf directory under PLANTVILLAGE_DIR so a
# stray grayscale/segmented extraction can never be mistaken for it.
PLANTVILLAGE_COLOR_DIR = KAGGLE_PLANTVILLAGE_COLOR_DIR if IS_KAGGLE else PLANTVILLAGE_DIR / "color"

PLANTDOC_DIR = DATA_ROOT / "plantdoc"
PLANTDOC_TRAIN_DIR = PLANTDOC_DIR / "train"
PLANTDOC_TEST_DIR = PLANTDOC_DIR / "test"

SMOKE_DIR = REPO_ROOT / "data" / "smoke"

SPLIT_MANIFEST_DIR = DATA_ROOT / "splits"

# ---------------------------------------------------------------------------
# Dataset sources
# ---------------------------------------------------------------------------
# Kaggle slug for PlantVillage. abdallahalidev/plantvillage-dataset is used
# because it is the dataset that mirrors the original spMohanty/
# PlantVillage-Dataset GitHub release with the raw folder structure intact:
# it ships three top-level variants (color/, grayscale/, segmented/) rather
# than a single pre-selected/augmented set, so the "color" subtree can be
# unambiguously identified and extracted on its own. This project only ever
# uses that "color" subtree.
KAGGLE_DATASET_SLUG = "abdallahalidev/plantvillage-dataset"
# The canonical figure from the original PlantVillage paper is 54,303, but
# Kaggle mirrors differ by a handful of files (observed: 54,305 images on
# abdallahalidev/plantvillage-dataset as of this writing). This is a sanity
# RANGE, not a loosened check — anything outside it still raises, and the
# class-count check (see PLANTVILLAGE_CLASS_NAMES below) stays exact.
PLANTVILLAGE_EXPECTED_IMAGE_COUNT_RANGE = (54_000, 54_400)

# PlantDoc's classification split ("Cropped-PlantDoc") lives here. It is a
# live GitHub repo, not a frozen Kaggle release, so unlike PlantVillage its
# image count can drift by a handful of files over time. Canonical figure
# cited by the PlantDoc paper is ~2,598; this range gives it the same
# sanity-range treatment as PlantVillage above.
PLANTDOC_REPO_URL = "https://github.com/pratikkayal/PlantDoc-Dataset"
PLANTDOC_EXPECTED_IMAGE_COUNT_RANGE = (2_548, 2_648)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------
# Canonical, alphabetically sorted class names for PlantVillage color, i.e.
# the exact index order every checkpoint, split manifest, and export must
# use for class_names (CLAUDE.md rule 7). Hardcoded rather than re-derived
# from a live directory listing so the index order can never silently shift
# between runs; src/data/inventory.py asserts the actual downloaded folder
# set matches this list exactly.
PLANTVILLAGE_CLASS_NAMES = (
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
)
NUM_CLASSES = len(PLANTVILLAGE_CLASS_NAMES)
assert NUM_CLASSES == 38, (
    f"PLANTVILLAGE_CLASS_NAMES has {NUM_CLASSES} entries, expected 38. "
    "This list must never be edited without also re-verifying the Kaggle "
    "source directory listing."
)
assert list(PLANTVILLAGE_CLASS_NAMES) == sorted(PLANTVILLAGE_CLASS_NAMES), (
    "PLANTVILLAGE_CLASS_NAMES must be kept in alphabetically sorted order — "
    "this is the index order Keras' directory-based loaders produce, and "
    "every checkpoint/export's class_names metadata depends on it matching."
)

# A class is "healthy" iff the condition token after the literal "___"
# separator is exactly "healthy" — an exact-match check on PlantVillage's
# own `<species>___<condition>` naming convention, not a substring guess.
HEALTHY_CLASS_NAMES = tuple(
    name for name in PLANTVILLAGE_CLASS_NAMES if name.split("___")[-1] == "healthy"
)

# Below this many images, a per-class metric (recall, F1, ...) for that class
# is too noise-dominated to report as if it meant something — e.g. PlantDoc's
# "Tomato two spotted spider mites leaf" class has only 2 images. Used by
# src/data/mapping.py to flag which mapped PlantDoc classes can't actually
# support a per-class metric, so that fact is recorded in the artifact rather
# than discovered later when a report shows a suspiciously perfect/zero
# per-class score.
MIN_IMAGES_FOR_PER_CLASS_METRICS = 10

# ---------------------------------------------------------------------------
# Deduplication (src/data/dedupe.py)
# ---------------------------------------------------------------------------
# imagehash.phash's hash_size parameter: produces a hash_size x hash_size
# bit hash (64 bits at the default of 8). Spelled out explicitly here
# (rather than left as an implicit library default) so the bit-count is
# auditable and DEDUPE_HAMMING_THRESHOLD below can be sanity-checked against
# it.
PHASH_SIZE = 8

# Hamming distance threshold for treating two images as the same physical
# leaf. 0 means bit-identical; larger values catch near-duplicates such as
# rotations and slight crops, at the cost of being more likely to also catch
# two genuinely different leaves that happen to look similar. 5 is a
# conservative middle ground for a 64-bit phash.
DEDUPE_HAMMING_THRESHOLD = 5
assert 0 <= DEDUPE_HAMMING_THRESHOLD <= PHASH_SIZE * PHASH_SIZE, (
    f"DEDUPE_HAMMING_THRESHOLD ({DEDUPE_HAMMING_THRESHOLD}) must be between "
    f"0 and the phash bit count ({PHASH_SIZE * PHASH_SIZE}) inclusive."
)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Image / preprocessing
# ---------------------------------------------------------------------------
IMAGE_SIZE = 224
IMAGE_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

PREPROCESSING = {
    "resize": (IMAGE_SIZE, IMAGE_SIZE),
    "rescale": 1.0 / 255.0,
}

# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

_split_sum = TRAIN_FRACTION + VAL_FRACTION + TEST_FRACTION
assert abs(_split_sum - 1.0) < 1e-9, (
    f"TRAIN_FRACTION + VAL_FRACTION + TEST_FRACTION == {_split_sum}, "
    "expected 1.0 (CLAUDE.md rule 6)."
)

# ---------------------------------------------------------------------------
# Local smoke tests (CLAUDE.md rule 10: laptop may only run smoke tests on
# at most 200 images — never full training, never full dataset downloads).
# ---------------------------------------------------------------------------
SMOKE_MAX_IMAGES = 200

# ---------------------------------------------------------------------------
# tf.data pipeline (src/data/pipeline.py)
# ---------------------------------------------------------------------------
BATCH_SIZE = 32

# Canonical pre-crop size for the deterministic val/test path (resize then
# center-crop to IMAGE_SIZE) — PlantVillage's native resolution, so val/test
# never sees the aggressive scale/aspect jitter train's random-resized-crop
# applies. See src/data/pipeline.py's module docstring.
VAL_TEST_RESIZE_SIZE = 256

# Cap on the train shuffle buffer. src/data/pipeline.py's shuffle sits
# *after* decode and the disk cache (the correct tf.data order — see that
# module's docstring — cache must see the deterministic pre-shuffle order
# for reshuffle_each_iteration to actually reshuffle every epoch) but
# *before* the uint8->float32 conversion, so this buffer holds decoded uint8
# [native_resolution, native_resolution, 3] tensors (PlantVillage's native
# resolution is 256x256, i.e. VAL_TEST_RESIZE_SIZE below — NOT IMAGE_SIZE;
# there is no resize before this point), not path strings: ~0.19 MB each, so
# 2048 caps the buffer at ~0.4 GB. (Sizing this any larger than the full
# split would be the same OOM this project already hit once — see
# TRAIN_DECODE_CACHE_ENABLED's comment below for the disk-side version of
# that same mistake, caught before the first real Kaggle run.)
# split.py's per-class group allocation already shuffles group order within
# each class before writing splits.json, so the on-disk train split isn't
# sorted by class to begin with — a buffer far smaller than the full split
# still mixes classes well within each shuffled window.
SHUFFLE_BUFFER_SIZE = 2048

# Whether src/data/pipeline.py's build_datasets() gives the TRAIN split an
# on-disk decode cache at all — a real, environment-specific tradeoff
# (disk space vs. re-decoding JPEGs every epoch), unlike val's cache below,
# which is small enough to always be worth it. Measured cache sizes for the
# real 38,013-image train split, at native 256x256 resolution, uint8 (see
# src/data/pipeline.py's _decode / _to_float — the cache stores uint8,
# post-decode pre-augmentation; the float32 conversion happens strictly
# after the cache read, never before):
#   - True:  ~7.5 GB on disk. Comfortably fits Kaggle's 19.5 GB
#     /kaggle/working quota alongside checkpoints/artifacts.
#   - False: 0 GB (no train cache; every epoch re-decodes from
#     PLANTVILLAGE_COLOR_DIR, which is a fast local/mounted read on both
#     Kaggle and Colab — see src/data/pipeline.py's module docstring).
# Before this was fixed, the cache stored float32 at native resolution
# (~29.9 GB for train alone — worse than the ~22 GB a 224x224 assumption
# would suggest, since no resize happens before the cache point) and would
# have exhausted a 19.5 GB Kaggle quota partway through epoch 1.
TRAIN_DECODE_CACHE_ENABLED = True

# Safety margin applied on top of the raw projected pixel-byte count when
# src/data/pipeline.py checks free disk space before writing a cache (tf.data's
# own on-disk cache format adds a small amount of per-record framing
# overhead on top of raw pixel bytes, and leaving zero headroom on a
# multi-hour run is exactly the kind of near-miss this check exists to
# avoid). 1.1 == require 10% more free space than the raw projection.
CACHE_DISK_SPACE_SAFETY_MARGIN = 1.1

# Local-disk cache for decoded-but-unaugmented train images (never Drive —
# see the Drive-vs-local-disk rule above). Caching here, before augmentation,
# avoids re-decoding JPEGs every epoch while still letting shuffle order and
# every augmentation re-randomize fresh each epoch. Only used when
# TRAIN_DECODE_CACHE_ENABLED is True.
TRAIN_DECODE_CACHE_DIR = DATA_ROOT / "cache" / "train_decoded"

# Same idea for val: read once per epoch across many epochs of real
# training, so caching the decoded image pays for itself repeatedly — own
# directory so it can never collide with train's cache prefix. Always
# enabled (no toggle — at ~1.6 GB for the real 8,146-image val split, uint8,
# native 256x256 resolution, it's small enough to never be worth disabling).
# Test has no such repeat-read pattern (CLAUDE.md rule 2: touched exactly
# once, at the end), so it deliberately gets no disk cache at all — see
# src/data/pipeline.py's build_datasets.
VAL_DECODE_CACHE_DIR = DATA_ROOT / "cache" / "val_decoded"

# ---------------------------------------------------------------------------
# Augmentation (src/data/augment.py) — train split only, NEVER val/test.
# ---------------------------------------------------------------------------
# PlantVillage is a single detached leaf on a uniform lab background at fixed
# framing/lighting/focus. Every value below exists to break one specific
# shortcut a model could learn from that studio setup rather than the lesion
# itself — see src/data/augment.py's module docstring for the explicit
# op-to-shortcut pairing (CLAUDE.md's "Known failure modes" section).
# Minimum raised from 0.65 to 0.8 after visual inspection of the 64-image
# augmentation_grid.png sanity grid: roughly 6% of tiles (4/64) were zoomed
# in so far that no diagnostic feature remained in frame — a Corn healthy
# leaf reduced to plain venation stripes, two Orange Haunglongbing tiles
# showing only uniform green, and a Corn Common rust tile cropped to a bare
# leaf edge. PlantVillage images are single centred leaves at 256x256, so an
# aggressive minimum crop area routinely excises the lesion entirely,
# producing a disease label with no disease evidence in frame (label noise).
AUGMENT_CROP_SCALE_RANGE = (0.8, 1.0)
AUGMENT_CROP_RATIO_RANGE = (0.8, 1.25)
AUGMENT_ROTATION_FACTOR = 0.0833  # keras RandomRotation factor -> ~30 degrees
# Lowered from 0.25 to 0.15 after visual inspection of the 64-image
# augmentation_grid.png sanity grid: two tiles (a Blueberry healthy and a
# Tomato Early blight) were darkened to nearly black, with no leaf detail or
# lesion visible at all. tf.image.random_brightness draws its delta from
# [-AUGMENT_BRIGHTNESS_MAX_DELTA, +AUGMENT_BRIGHTNESS_MAX_DELTA], so this
# single symmetric constant is also each image's darkening floor; 0.15 still
# darkens meaningfully (real field photos are often shot in shade) without
# driving a dim lab image's already-low pixel values to near-black.
AUGMENT_BRIGHTNESS_MAX_DELTA = 0.15
AUGMENT_CONTRAST_RANGE = (0.7, 1.3)
AUGMENT_SATURATION_RANGE = (0.6, 1.4)
# Halved from 0.08 to 0.04 after the same sanity grid showed an Apple
# Apple_scab tile hue-shifted so far the leaf appeared blue. Hue jitter is
# the main defence against background-shortcut learning (decorrelating
# background/leaf colour from the label), but chlorosis and yellowing are
# themselves diagnostic for several classes — pushing hue too far destroys
# that genuine colour signal along with the shortcut, so this range trades
# some background decorrelation strength to keep real disease colour cues
# intact.
AUGMENT_HUE_MAX_DELTA = 0.04
AUGMENT_GAUSSIAN_BLUR_PROBABILITY = 0.3
AUGMENT_GAUSSIAN_BLUR_SIGMA_RANGE = (0.5, 1.5)
AUGMENT_GAUSSIAN_BLUR_KERNEL_SIZE = 5
AUGMENT_JPEG_QUALITY_RANGE = (30, 90)
AUGMENT_RANDOM_ERASING_PROBABILITY = 0.25
AUGMENT_RANDOM_ERASING_AREA_RANGE = (0.02, 0.15)
AUGMENT_RANDOM_ERASING_ASPECT_RANGE = (0.3, 3.3)

# ---------------------------------------------------------------------------
# Not-a-leaf negative set (src/data/negatives.py) — OOD rejection calibration
# only, never a training label, never one of the 38 classes.
# ---------------------------------------------------------------------------
# Kaggle mirror of Intel's "Natural Scenes" dataset: ~25k photos of ordinary,
# non-leaf scenes (buildings, forest, glacier, mountain, sea, street) —
# chosen because on Colab it reuses the same KaggleApi + zip-extraction
# machinery already in src/data/fetch.py rather than adding a new raw-URL
# download path, and it's about as visually unlike a studio leaf macro shot
# as an "ordinary photograph" dataset gets. On Kaggle this dataset is
# instead a read-only mounted input (KAGGLE_NEGATIVES_SEG_TRAIN_DIR above) —
# this slug is only used for the Colab-only KaggleApi download.
NEGATIVES_KAGGLE_SLUG = "puneet6060/intel-image-classification"
# Name of the directory inside the Kaggle archive (Colab path) / mount
# (Kaggle path, see KAGGLE_NEGATIVES_SEG_TRAIN_DIR) holding the labeled
# training images. Located the same defensive way as fetch.py's
# "color"-directory search (exact name match, raise if 0 or >1 found) since
# the exact nesting can't be verified without a live download.
NEGATIVES_SOURCE_SUBDIR_NAME = "seg_train"
# The six category directories the resolved seg_train subtree must contain
# directly (after src/data/fetch.py's wrapper-directory descent) — asserted
# exactly, not fuzzily, so a source-side layout/category change is caught
# immediately rather than silently subsampling from the wrong level again.
NEGATIVES_EXPECTED_CATEGORIES = ("buildings", "forest", "glacier", "mountain", "sea", "street")
# Always DATA_ROOT-relative (never the read-only Kaggle mount itself) — this
# is the destination src/data/negatives.py's deterministic subsample is
# copied into, so it must always be writable.
NEGATIVES_DIR = DATA_ROOT / "negatives"
NEGATIVES_TARGET_COUNT = 3_000
NEGATIVES_EXPECTED_IMAGE_COUNT_RANGE = (2_900, 3_100)
NEGATIVES_TAR = (DRIVE_DATA_ROOT / "negatives.tar") if DRIVE_DATA_ROOT else None
MIN_NEGATIVES_TAR_BYTES = 20_000_000

# ---------------------------------------------------------------------------
# Candidate model architectures
# ---------------------------------------------------------------------------
# keras.applications names for small, mobile-friendly backbones suitable for
# LiteRT INT8 export and CameraX-based inference on minSdk 24 devices. Final
# architecture selection happens later, on the validation set only
# (CLAUDE.md rule 2) — this list is just the candidate pool.
#
# EfficientNetB0 was dropped from this pool on 2026-08-11: its Kaggle
# training run failed with a 404 from the Kaggle API (src/models/
# kaggle_persist.py's checkpoint push/restore), and per src/train_all.py's
# module docstring — the exact incident that runner was written to survive
# — a persistence-layer 404 is not retried; the model was left out rather
# than spending further budget on it with 6 days left in the project. The
# comparison is exactly the remaining four: MobileNetV2, MobileNetV3Small,
# MobileNetV3Large, EfficientNetV2B0.
CANDIDATE_MODELS = (
    "MobileNetV2",
    "MobileNetV3Small",
    "MobileNetV3Large",
    "EfficientNetV2B0",
)

# ---------------------------------------------------------------------------
# Per-backbone preprocessing (src/data/pipeline.py)
# ---------------------------------------------------------------------------
# Each CANDIDATE_MODELS entry expects its own normalization (e.g.
# MobileNetV2 rescales to [-1, 1]; EfficientNet's preprocess_input is close
# to a no-op because the model itself embeds a Rescaling/Normalization
# layer) — using one normalization for all of them would silently feed
# several mis-scaled input. Stored as plain (module, function) name strings,
# not live imports, so this file stays importable without TensorFlow
# installed; src/data/pipeline.py resolves the callable via importlib.
PREPROCESSING_ENTRYPOINTS = {
    "MobileNetV2": ("keras.applications.mobilenet_v2", "preprocess_input"),
    "MobileNetV3Small": ("keras.applications.mobilenet_v3", "preprocess_input"),
    "MobileNetV3Large": ("keras.applications.mobilenet_v3", "preprocess_input"),
    "EfficientNetV2B0": ("keras.applications.efficientnet_v2", "preprocess_input"),
}
assert set(PREPROCESSING_ENTRYPOINTS) == set(CANDIDATE_MODELS), (
    "PREPROCESSING_ENTRYPOINTS must have exactly one entry per CANDIDATE_MODELS "
    f"entry. Missing: {set(CANDIDATE_MODELS) - set(PREPROCESSING_ENTRYPOINTS)}. "
    f"Unexpected: {set(PREPROCESSING_ENTRYPOINTS) - set(CANDIDATE_MODELS)}."
)

# ---------------------------------------------------------------------------
# Model architecture (src/models/build.py)
# ---------------------------------------------------------------------------
DROPOUT_RATE = 0.3

# Per-backbone count of "last N blocks" src/models/build.py's
# unfreeze_top_blocks() unfreezes for phase-2 fine-tuning. Block structure
# differs across CANDIDATE_MODELS (different depths, different naming
# conventions), so this is a per-backbone value, never a single shared
# constant — see BACKBONE_BLOCK_NAME_PATTERNS below for how each
# architecture's blocks are actually identified.
UNFREEZE_BLOCKS = {
    "MobileNetV2": 3,        # unfreezes inverted-residual blocks 14-16 of 16
    "MobileNetV3Small": 3,   # unfreezes blocks 8-10 of 10
    "MobileNetV3Large": 3,   # unfreezes blocks 12-14 of 14
    "EfficientNetV2B0": 2,   # unfreezes stages 5-6 of 6
}
assert set(UNFREEZE_BLOCKS) == set(CANDIDATE_MODELS), (
    "UNFREEZE_BLOCKS must have exactly one entry per CANDIDATE_MODELS entry."
)

# Regex matched against each backbone layer's name to recover its integer
# block/stage index, one pattern per architecture because
# keras.applications names blocks differently per family (verified against
# the actual layer names of each CANDIDATE_MODELS backbone,
# include_top=False, pooling="avg"):
#   - MobileNetV2:                "block_<N>_..."       N = 1..16
#   - MobileNetV3Small/Large:     "expanded_conv_<N>_..." N = 1..10 / 1..14
#   - EfficientNetV2B0:           "block<N><letter>_..."  N = stage 1..6
# The first block of every family (MobileNetV2/V3's block 0, EfficientNet's
# stage-less stem) doesn't match and is therefore never a candidate for
# unfreezing — correct, since "last N blocks" always means the deepest ones.
BACKBONE_BLOCK_NAME_PATTERNS = {
    "MobileNetV2": r"^block_(\d+)_",
    "MobileNetV3Small": r"^expanded_conv_(\d+)_",
    "MobileNetV3Large": r"^expanded_conv_(\d+)_",
    "EfficientNetV2B0": r"^block(\d+)[a-z]_",
}
assert set(BACKBONE_BLOCK_NAME_PATTERNS) == set(CANDIDATE_MODELS), (
    "BACKBONE_BLOCK_NAME_PATTERNS must have exactly one entry per "
    "CANDIDATE_MODELS entry."
)

# ---------------------------------------------------------------------------
# Training (src/train.py)
# ---------------------------------------------------------------------------
# Phase 1 (frozen backbone, head only) learning rate. Phase 2 keeps this
# same rate for the head and additionally trains the unfrozen backbone
# blocks at HEAD_LEARNING_RATE * BACKBONE_LR_FACTOR (src/models/finetune.py)
# — never the same rate for both, which would push large, ImageNet-feature-
# destroying updates through the just-unfrozen pretrained layers.
HEAD_LEARNING_RATE = 1e-3
BACKBONE_LR_FACTOR = 0.1

PHASE1_EPOCHS = 15
# Measured on the first real Kaggle run (T4, MobileNetV3Small — the
# smallest of CANDIDATE_MODELS, so the fastest per epoch and the most
# epoch-budget-generous case): 207 s/epoch. At that rate PHASE1_EPOCHS (15)
# is ~52 min; the original PHASE2_EPOCHS of 30 would have added another
# ~1h45 per model, ~2.5h per model / ~12.5 GPU-hours for all 5 models
# originally planned (before EfficientNetB0 was dropped — see
# CANDIDATE_MODELS' comment) — too much of an 8-day remaining budget that
# still has Android work ahead. 12 keeps phase 2 to ~41 min worst-case (no
# early stop), ~1.5h/model, ~7.75 GPU-hours across those 5 in the worst
# case, less in practice since EARLY_STOPPING_PATIENCE below usually
# triggers first — now ~6.2 GPU-hours worst-case across the 4 remaining
# candidates.
PHASE2_EPOCHS = 12

# Lowered from 5: at PHASE2_EPOCHS=12, a patience of 5 lets a plateaued run
# burn nearly half its remaining epoch budget before stopping. 3 stops a
# plateaued run sooner without meaningfully undercutting a still-improving
# one (REDUCE_LR_PATIENCE below fires first at 3 anyway, so an EarlyStopping
# patience of 3 mostly just means "stop if the LR drop didn't help either").
EARLY_STOPPING_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_PATIENCE = 3
REDUCE_LR_MIN_LR = 1e-6

# Every epoch's weights are saved here. On Colab this is Drive
# (DRIVE_DATA_ROOT), so a runtime disconnection never costs more than one
# epoch — Drive outlives the VM. On Kaggle there is no Drive, so this falls
# through to DATA_ROOT ("/kaggle/working/data/checkpoints"), which on its
# own does NOT survive a session restart or the ~9-12 hour session time
# limit — see src/models/kaggle_persist.py's module docstring for how Day 6
# closes that gap (pushing checkpoints to a private Kaggle Dataset via the
# Kaggle API, restored back into this directory at the start of a fresh
# session) and src/models/checkpoint.py's docstring for what "resume" means
# once a checkpoint is here, regardless of how it got here. Local runs also
# fall through to DATA_ROOT, but only ever hold smoke-test checkpoints
# (CLAUDE.md rule 10: no real training locally), and Kaggle-persistence
# never fires for them (src/train.py's --no-persist / smoke gating). Each
# model gets its own subdirectory (src/train.py: CHECKPOINT_DIR /
# model_name), so resuming one model can never pick up another's
# checkpoint.
CHECKPOINT_DIR = (DRIVE_DATA_ROOT / "checkpoints") if DRIVE_DATA_ROOT else (DATA_ROOT / "checkpoints")

# ---------------------------------------------------------------------------
# Cross-session checkpoint persistence on Kaggle (src/models/kaggle_persist.py)
# ---------------------------------------------------------------------------
# Private Kaggle Dataset every model's checkpoint + history/manifest JSONs
# get pushed to after each phase (and restored from at the start of a fresh
# session) — see that module's docstring for the full push/restore design.
# Created on first use if it doesn't already exist under the authenticated
# user's account.
KAGGLE_PERSIST_DATASET_SLUG = "plantguard-artifacts"

# Scratch folder pushes/restores are staged through before touching the
# real Kaggle Dataset — DATA_ROOT-relative so it's /kaggle/working/data/...
# on Kaggle (writable, gitignored, never the read-only input mount).
KAGGLE_PERSIST_STAGING_DIR = DATA_ROOT / "_kaggle_artifacts_staging"

# The flat, non-per-model files under ARTIFACTS_DIR that
# src/models/kaggle_persist_artifacts.py pushes/restores as a single bundle
# (under the "artifacts__" prefix in the same KAGGLE_PERSIST_DATASET_SLUG
# dataset) — everything a report/figure number is read from
# (RESULTS_JSON_PATH) plus the dedupe/split/inventory/mapping outputs that
# take ~20 minutes to regenerate (src/data/dedupe.py, split_report.py,
# inventory.py, mapping_report.py). EVAL_FIGURES_DIR is pushed/restored
# separately since its contents are a directory tree, not fixed filenames.
KAGGLE_PERSIST_ARTIFACT_FILENAMES = (
    "results.json",
    "image_groups.json",
    "dedupe_report.json",
    "dedupe_report.md",
    "splits.json",
    "split_report.md",
    "inventory.json",
    "inventory.md",
    "plantdoc_mapping.json",
)

# --smoke: two short epochs (CLAUDE.md rule 10 — local runs are smoke tests
# only) on a synthetic on-disk dataset built by src/data/smoke_dataset.py,
# never on the real splits.json (which doesn't exist on a machine that has
# never downloaded PlantVillage). SMOKE_NUM_CLASSES real class names are
# used (real indices into PLANTVILLAGE_CLASS_NAMES, so the 38-wide output
# head is exercised against genuine label positions) with a fixed per-class
# image count; the assertion below keeps the total under SMOKE_MAX_IMAGES
# automatically if either constant is ever changed.
SMOKE_EPOCHS_PER_PHASE = 1
SMOKE_NUM_CLASSES = 5
SMOKE_TRAIN_IMAGES_PER_CLASS = 32
SMOKE_VAL_IMAGES_PER_CLASS = 8
assert SMOKE_NUM_CLASSES * (SMOKE_TRAIN_IMAGES_PER_CLASS + SMOKE_VAL_IMAGES_PER_CLASS) <= SMOKE_MAX_IMAGES, (
    "Smoke dataset size exceeds CLAUDE.md rule 10's 200-image local cap."
)

# ---------------------------------------------------------------------------
# TFLite export
# ---------------------------------------------------------------------------
TFLITE_CONFIG = {
    "quantization": "int8",
    "optimizations": "DEFAULT",  # maps to tf.lite.Optimize.DEFAULT
    "representative_dataset_size": SMOKE_MAX_IMAGES,
    "input_dtype": "uint8",
    "output_dtype": "uint8",
}
TFLITE_OUTPUT_DIR = ARTIFACTS_DIR / "tflite"

# ---------------------------------------------------------------------------
# Evaluation (src/evaluate/) — Day 8 full evaluation suite
# ---------------------------------------------------------------------------
# Single JSON every report/README/figure number is read from (CLAUDE.md rule
# 5) — src/evaluate/runner.py is the only thing that writes it.
RESULTS_JSON_PATH = ARTIFACTS_DIR / "results.json"
EVAL_FIGURES_DIR = ARTIFACTS_DIR / "figures"

# The real, non-placeholder Android metadata contract (see
# src/export/to_tflite.py's docstring and android/README.md): class_names,
# image_size, confidence_threshold. src/evaluate/ood.py updates only the
# confidence_threshold field here once a real threshold is tuned, ahead of
# the Day-9 model.tflite swap — every other field (including "placeholder")
# is left untouched since the .tflite itself is still the placeholder.
ANDROID_MODEL_METADATA_PATH = REPO_ROOT / "android" / "app" / "src" / "main" / "assets" / "model_metadata.json"

# Test-set evaluation (src/evaluate/metrics.py). A 38x38 confusion matrix is
# unreadable as a headline number — this is how many off-diagonal (true,
# predicted) pairs get surfaced instead, ranked by raw count.
NUM_TOP_CONFUSED_PAIRS = 10

# Calibration (src/evaluate/calibration.py). ECE_NUM_BINS=15 is the standard
# choice from Guo et al. 2017 ("On Calibration of Modern Neural Networks").
# Temperature is fit by grid search (never a magic single value) over
# [TEMPERATURE_GRID_MIN, TEMPERATURE_GRID_MAX] in TEMPERATURE_GRID_STEP
# increments, minimizing validation NLL — a 1-parameter search, so grid
# search is exact enough and avoids adding a scipy.optimize dependency.
ECE_NUM_BINS = 15
TEMPERATURE_GRID_MIN = 0.5
TEMPERATURE_GRID_MAX = 5.0
TEMPERATURE_GRID_STEP = 0.01

# Out-of-distribution rejection (src/evaluate/ood.py). Max-softmax
# confidence thresholds tested 0.00..1.00 inclusive in this step size, when
# choosing the operating point that best separates validation (in-
# distribution) from the Intel negatives (out-of-distribution).
OOD_THRESHOLD_GRID_STEP = 0.01

# Robustness (src/evaluate/robustness.py): corruption types a phone camera
# actually produces, at 3 severities each. Every severity map below must
# cover exactly ROBUSTNESS_SEVERITIES — asserted below rather than assumed.
ROBUSTNESS_CORRUPTIONS = (
    "blur",
    "gaussian_noise",
    "brightness_up",
    "brightness_down",
    "rotation",
    "jpeg_compression",
)
ROBUSTNESS_SEVERITIES = (1, 2, 3)
# Gaussian blur: depthwise-conv sigma per severity (same op family as
# src/data/augment.py's random_gaussian_blur, but deterministic per severity
# rather than randomly sampled) and a fixed kernel size wide enough to
# actually resolve the largest sigma below.
ROBUSTNESS_BLUR_SIGMA = {1: 1.0, 2: 2.0, 3: 3.0}
ROBUSTNESS_BLUR_KERNEL_SIZE = 9
# Additive Gaussian noise stddev, on a float32 [0, 1] image.
ROBUSTNESS_GAUSSIAN_NOISE_STDDEV = {1: 0.05, 2: 0.10, 3: 0.20}
# tf.image.adjust_brightness delta, on a float32 [0, 1] image.
ROBUSTNESS_BRIGHTNESS_UP_DELTA = {1: 0.10, 2: 0.20, 3: 0.35}
ROBUSTNESS_BRIGHTNESS_DOWN_DELTA = {1: 0.10, 2: 0.20, 3: 0.35}
# Degrees, applied via a fixed-factor (lo == hi) keras.layers.RandomRotation
# — the same layer src/data/augment.py uses for random rotation, but with a
# degenerate range so the angle is deterministic per severity instead of
# sampled.
ROBUSTNESS_ROTATION_DEGREES = {1: 10.0, 2: 20.0, 3: 30.0}
# tf.image.adjust_jpeg_quality quality level (0-100); lower == more severe
# compression artifacting.
ROBUSTNESS_JPEG_QUALITY = {1: 50, 2: 30, 3: 10}
for _map in (
    ROBUSTNESS_BLUR_SIGMA,
    ROBUSTNESS_GAUSSIAN_NOISE_STDDEV,
    ROBUSTNESS_BRIGHTNESS_UP_DELTA,
    ROBUSTNESS_BRIGHTNESS_DOWN_DELTA,
    ROBUSTNESS_ROTATION_DEGREES,
    ROBUSTNESS_JPEG_QUALITY,
):
    assert set(_map) == set(ROBUSTNESS_SEVERITIES), (
        f"Every ROBUSTNESS_* severity map must have exactly one entry per "
        f"ROBUSTNESS_SEVERITIES {ROBUSTNESS_SEVERITIES}, got keys {sorted(_map)}."
    )
del _map

# Grad-CAM (src/evaluate/gradcam.py): sample counts, never a curated-only
# "successes" set — incorrect predictions and PlantDoc failures are sampled
# just as deliberately as correct ones (CLAUDE.md rule 1's spirit: an honest
# result, not a flattering one).
GRADCAM_NUM_CORRECT_SAMPLES = 8
GRADCAM_NUM_INCORRECT_SAMPLES = 8
GRADCAM_NUM_PLANTDOC_FAILURE_SAMPLES = 6
GRADCAM_OVERLAY_ALPHA = 0.4

# ---------------------------------------------------------------------------
# Dataset provenance
# ---------------------------------------------------------------------------
# Written by src/data/download.py after each successful download: source,
# timestamp, observed counts, and a sha256 of the sorted relative file-path
# list, so the exact dataset version behind any report is reproducible.
DATASET_PROVENANCE_PATH = ARTIFACTS_DIR / "dataset_provenance.json"


def get_git_commit_hash() -> str:
    """Returns the current git commit hash. Raises if not in a git repo or
    git is unavailable — every checkpoint/export needs a real commit hash
    (CLAUDE.md rule 7), never a placeholder.
    """
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()
