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

if IS_COLAB:
    # Per project convention, all real data lives in this exact Drive path
    # and is only ever touched from a Colab session (never downloaded to the
    # local laptop).
    DATA_ROOT = Path("/content/drive/MyDrive/plantguard-data")
else:
    # Local Windows dev machine: DATA_ROOT is only ever used to hold the
    # ≤200-image smoke-test subset (see SMOKE_MAX_IMAGES below). Full-size
    # dataset downloads must never be triggered here — see the IS_COLAB
    # guard in src/data/download.py.
    DATA_ROOT = REPO_ROOT / "data"

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
PLANTVILLAGE_EXPECTED_IMAGE_COUNT = 54_303

# PlantDoc's classification split ("Cropped-PlantDoc") lives here. It is a
# live GitHub repo, not a frozen Kaggle release, so unlike PlantVillage its
# image count can drift by a handful of files over time — hence a tolerance
# band instead of an exact-match requirement.
PLANTDOC_REPO_URL = "https://github.com/pratikkayal/PlantDoc-Dataset"
PLANTDOC_EXPECTED_IMAGE_COUNT = 2_598
PLANTDOC_IMAGE_COUNT_TOLERANCE = 50

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
