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
# "google.colab" is only ever importable inside a Colab runtime, so checking
# sys.modules is the standard, reliable way to tell Colab apart from a local
# machine without hardcoding either platform's paths.
IS_COLAB = "google.colab" in sys.modules

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Repo root is derived from this file's own location, so it resolves
# correctly regardless of platform or where the repo happens to be cloned
# (e.g. C:\Projects\plantguard-v2 locally vs /content/plantguard-v2 on Colab).
REPO_ROOT = Path(__file__).resolve().parent.parent

ARTIFACTS_DIR = REPO_ROOT / "artifacts"

# DATA_ROOT is where all per-image work happens: extraction, counting,
# splitting, training reads. On Colab this MUST be the VM's own local disk,
# never the Drive FUSE mount — Drive writes/stats small files at only a few
# dozen per second, so any per-file operation (extracting ~54k images,
# walking them to validate) is unusably slow or effectively never completes
# there. See src/data/download.py's module docstring for the full
# cold-storage-vs-local-disk design and CLAUDE.md's Drive rule.
if IS_COLAB:
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
# walk-free integrity check in src/data/download.py possible.
DRIVE_PROVENANCE_PATH = (DRIVE_DATA_ROOT / "dataset_provenance.json") if DRIVE_DATA_ROOT else None
PLANTVILLAGE_COLOR_TAR = (DRIVE_DATA_ROOT / "plantvillage_color.tar") if DRIVE_DATA_ROOT else None
PLANTDOC_TAR = (DRIVE_DATA_ROOT / "plantdoc.tar") if DRIVE_DATA_ROOT else None

# Coarse sanity floors for the fast Drive integrity check: catch an empty or
# obviously-truncated tar without opening/reading it. Deliberately loose —
# the authoritative check is the observed image/class counts recorded in
# dataset_provenance.json, not tar size.
MIN_PLANTVILLAGE_TAR_BYTES = 500_000_000
MIN_PLANTDOC_TAR_BYTES = 20_000_000

PLANTVILLAGE_DIR = DATA_ROOT / "plantvillage"
# Only the "color" variant is ever used for training — never grayscale or
# segmented. Kept as its own leaf directory so a stray grayscale/segmented
# extraction can never be mistaken for it.
PLANTVILLAGE_COLOR_DIR = PLANTVILLAGE_DIR / "color"

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

# Cap on the train shuffle buffer. Shuffling happens over lightweight path
# strings *before* decode (cheap even near the full training-set size), so
# this cap only matters as a safety ceiling, not a memory/quality tradeoff
# knob — the actual buffer used is min(len(train_paths), this).
SHUFFLE_BUFFER_SIZE = 50_000

# Local-disk cache for decoded-but-unaugmented train images (never Drive —
# see the Drive-vs-local-disk rule above). Caching here, before augmentation,
# avoids re-decoding JPEGs every epoch while still letting shuffle order and
# every augmentation re-randomize fresh each epoch.
TRAIN_DECODE_CACHE_DIR = DATA_ROOT / "cache" / "train_decoded"

# ---------------------------------------------------------------------------
# Augmentation (src/data/augment.py) — train split only, NEVER val/test.
# ---------------------------------------------------------------------------
# PlantVillage is a single detached leaf on a uniform lab background at fixed
# framing/lighting/focus. Every value below exists to break one specific
# shortcut a model could learn from that studio setup rather than the lesion
# itself — see src/data/augment.py's module docstring for the explicit
# op-to-shortcut pairing (CLAUDE.md's "Known failure modes" section).
AUGMENT_CROP_SCALE_RANGE = (0.65, 1.0)
AUGMENT_CROP_RATIO_RANGE = (0.8, 1.25)
AUGMENT_ROTATION_FACTOR = 0.0833  # keras RandomRotation factor -> ~30 degrees
AUGMENT_BRIGHTNESS_MAX_DELTA = 0.25
AUGMENT_CONTRAST_RANGE = (0.7, 1.3)
AUGMENT_SATURATION_RANGE = (0.6, 1.4)
AUGMENT_HUE_MAX_DELTA = 0.08
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
# chosen because it reuses the same KaggleApi + zip-extraction machinery
# already in src/data/fetch.py rather than adding a new raw-URL download
# path, and it's about as visually unlike a studio leaf macro shot as an
# "ordinary photograph" dataset gets.
NEGATIVES_KAGGLE_SLUG = "puneet6060/intel-image-classification"
# Name of the directory inside the Kaggle archive holding the labeled
# training images. Located the same defensive way as fetch.py's
# "color"-directory search (exact name match, raise if 0 or >1 found) since
# the exact nesting can't be verified without a live download.
NEGATIVES_SOURCE_SUBDIR_NAME = "seg_train"
# The six category directories the resolved seg_train subtree must contain
# directly (after src/data/fetch.py's wrapper-directory descent) — asserted
# exactly, not fuzzily, so a Kaggle-side layout/category change is caught
# immediately rather than silently subsampling from the wrong level again.
NEGATIVES_EXPECTED_CATEGORIES = ("buildings", "forest", "glacier", "mountain", "sea", "street")
NEGATIVES_DIR = DATA_ROOT / "negatives"
NEGATIVES_TARGET_COUNT = 3_000
NEGATIVES_EXPECTED_IMAGE_COUNT_RANGE = (2_900, 3_100)
NEGATIVES_TAR = (DRIVE_DATA_ROOT / "negatives.tar") if DRIVE_DATA_ROOT else None
MIN_NEGATIVES_TAR_BYTES = 20_000_000

# ---------------------------------------------------------------------------
# Candidate model architectures
# ---------------------------------------------------------------------------
# All five are keras.applications names for small, mobile-friendly backbones
# suitable for LiteRT INT8 export and CameraX-based inference on minSdk 24
# devices. Final architecture selection happens later, on the validation
# set only (CLAUDE.md rule 2) — this list is just the candidate pool.
CANDIDATE_MODELS = (
    "MobileNetV2",
    "MobileNetV3Small",
    "MobileNetV3Large",
    "EfficientNetB0",
    "EfficientNetV2B0",
)

# ---------------------------------------------------------------------------
# Per-backbone preprocessing (src/data/pipeline.py)
# ---------------------------------------------------------------------------
# Each of the 5 CANDIDATE_MODELS expects its own normalization (e.g.
# MobileNetV2 rescales to [-1, 1]; EfficientNet's preprocess_input is close
# to a no-op because the model itself embeds a Rescaling/Normalization
# layer) — using one normalization for all five would silently feed several
# of them mis-scaled input. Stored as plain (module, function) name strings,
# not live imports, so this file stays importable without TensorFlow
# installed; src/data/pipeline.py resolves the callable via importlib.
PREPROCESSING_ENTRYPOINTS = {
    "MobileNetV2": ("keras.applications.mobilenet_v2", "preprocess_input"),
    "MobileNetV3Small": ("keras.applications.mobilenet_v3", "preprocess_input"),
    "MobileNetV3Large": ("keras.applications.mobilenet_v3", "preprocess_input"),
    "EfficientNetB0": ("keras.applications.efficientnet", "preprocess_input"),
    "EfficientNetV2B0": ("keras.applications.efficientnet_v2", "preprocess_input"),
}
assert set(PREPROCESSING_ENTRYPOINTS) == set(CANDIDATE_MODELS), (
    "PREPROCESSING_ENTRYPOINTS must have exactly one entry per CANDIDATE_MODELS "
    f"entry. Missing: {set(CANDIDATE_MODELS) - set(PREPROCESSING_ENTRYPOINTS)}. "
    f"Unexpected: {set(PREPROCESSING_ENTRYPOINTS) - set(CANDIDATE_MODELS)}."
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
