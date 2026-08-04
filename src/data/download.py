"""Downloads PlantVillage (Kaggle) and PlantDoc (public) into raw data dirs.
Verifies expected image and class counts after download; fails loudly on mismatch.

Only ever runs on Colab — see the IS_COLAB guard in each download function.
Both downloads are idempotent: if valid data is already present, they skip.

Archives are downloaded/extracted directly into the Drive destination via a
staging subdirectory (never into /tmp), and that staging directory is only
ever removed after a successful post-move validation — so a validation
failure leaves whatever was downloaded on disk for inspection instead of
silently discarding it. Provenance recording lives in src/data/provenance.py.
"""

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config  # noqa: E402
from src.data import provenance  # noqa: E402


def _count_images(directory: Path) -> int:
    """Counts only image files (by extension, case-insensitive) — never
    directories, never non-image files.
    """
    if not directory.is_dir():
        return 0
    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in config.IMAGE_EXTENSIONS
    )


def _find_single_dir_named(root: Path, name: str) -> Path:
    """Finds exactly one directory under `root` whose basename == `name`
    (exact match, not a substring/fuzzy search), at any depth — this is
    what lets a single wrapper directory inside the archive (e.g.
    "plantvillage-dataset/color/" instead of "color/" at the top level) be
    found without failing. Raises if zero or more than one match is found,
    rather than guessing which one is correct.
    """
    matches = [p for p in root.rglob(name) if p.is_dir()]
    if len(matches) == 0:
        raise RuntimeError(
            f"No directory named exactly '{name}' found under {root}. "
            "The Kaggle archive layout may have changed — inspect it "
            "manually before proceeding."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Found {len(matches)} directories named exactly '{name}' under "
            f"{root}: {matches}. Ambiguous — refusing to guess which one is "
            "the correct color/ directory."
        )
    return matches[0]


def _require_colab(dataset_name: str) -> None:
    if not config.IS_COLAB:
        raise RuntimeError(
            f"Refusing to download {dataset_name}: this is not a Colab "
            "environment. Per project convention, real data is only ever "
            "downloaded into Google Drive from a Colab session — never onto "
            "the local laptop (CLAUDE.md rule 10 and the environment split "
            "in project instructions)."
        )


def _validate_plantvillage_dir(path: Path) -> tuple:
    """Returns (is_valid, detail). Image count must fall inside
    config.PLANTVILLAGE_EXPECTED_IMAGE_COUNT_RANGE; class folders must match
    config.PLANTVILLAGE_CLASS_NAMES exactly (hard failure, never weakened).
    """
    if not path.is_dir():
        return False, f"{path} does not exist."

    image_count = _count_images(path)
    class_dirs = sorted(p.name for p in path.iterdir() if p.is_dir())
    lo, hi = config.PLANTVILLAGE_EXPECTED_IMAGE_COUNT_RANGE
    expected_classes = sorted(config.PLANTVILLAGE_CLASS_NAMES)

    count_ok = lo <= image_count <= hi
    classes_ok = class_dirs == expected_classes

    if count_ok and classes_ok:
        return True, f"{image_count} images across {len(class_dirs)} classes"

    detail = (
        f"found {image_count} images (expected {lo}-{hi}) across "
        f"{len(class_dirs)} class folders (expected exactly {config.NUM_CLASSES})"
    )
    if not classes_ok:
        missing = sorted(set(expected_classes) - set(class_dirs))
        extra = sorted(set(class_dirs) - set(expected_classes))
        detail += f"; missing classes: {missing}; unexpected classes: {extra}"
    return False, detail


def _validate_plantdoc_dirs(train_dir: Path, test_dir: Path) -> tuple:
    """Returns (is_valid, detail). Image count (train+test combined) must
    fall inside config.PLANTDOC_EXPECTED_IMAGE_COUNT_RANGE.
    """
    if not train_dir.is_dir() or not test_dir.is_dir():
        return False, f"{train_dir} and/or {test_dir} do not exist."

    image_count = _count_images(train_dir) + _count_images(test_dir)
    lo, hi = config.PLANTDOC_EXPECTED_IMAGE_COUNT_RANGE
    if lo <= image_count <= hi:
        return True, f"{image_count} images across train/ and test/"
    return False, f"found {image_count} images across train/ and test/, expected {lo}-{hi}"


def download_plantvillage(force: bool = False) -> Path:
    """Downloads the PlantVillage color dataset from Kaggle into
    config.PLANTVILLAGE_COLOR_DIR. Idempotent: skips if the directory
    already passes validation. Extraction happens directly under Drive via
    a staging subdirectory — never /tmp — so a validation failure after
    extraction leaves the downloaded archive contents on disk instead of
    losing them to temp-directory cleanup.
    """
    _require_colab("PlantVillage")

    if not force and config.PLANTVILLAGE_COLOR_DIR.is_dir():
        valid, detail = _validate_plantvillage_dir(config.PLANTVILLAGE_COLOR_DIR)
        if valid:
            print(
                f"[download] PlantVillage already present and valid ({detail}) "
                "— skipping download."
            )
            return config.PLANTVILLAGE_COLOR_DIR
        if any(config.PLANTVILLAGE_COLOR_DIR.iterdir()):
            raise RuntimeError(
                f"{config.PLANTVILLAGE_COLOR_DIR} already exists but is "
                f"invalid: {detail}. Inspect and remove this directory "
                "manually before re-running, or pass force=True."
            )

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError(
            "The 'kaggle' package is not installed. Install it from "
            "requirements.txt before running this script."
        ) from exc

    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as exc:
        raise RuntimeError(
            "Kaggle authentication failed. Ensure kaggle.json has been "
            "uploaded and installed to ~/.kaggle/kaggle.json with mode 600 "
            "before running this script."
        ) from exc

    config.PLANTVILLAGE_DIR.mkdir(parents=True, exist_ok=True)
    staging_dir = config.PLANTVILLAGE_DIR / "_staging_download"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir()

    print(
        f"[download] Downloading Kaggle dataset '{config.KAGGLE_DATASET_SLUG}' "
        f"directly to {staging_dir} ..."
    )
    api.dataset_download_files(
        config.KAGGLE_DATASET_SLUG, path=str(staging_dir), unzip=True, quiet=False
    )

    color_dir = _find_single_dir_named(staging_dir, "color")
    if config.PLANTVILLAGE_COLOR_DIR.exists():
        shutil.rmtree(config.PLANTVILLAGE_COLOR_DIR)
    shutil.move(str(color_dir), str(config.PLANTVILLAGE_COLOR_DIR))

    valid, detail = _validate_plantvillage_dir(config.PLANTVILLAGE_COLOR_DIR)
    if not valid:
        raise RuntimeError(
            f"Downloaded PlantVillage color/ at {config.PLANTVILLAGE_COLOR_DIR} "
            f"failed validation: {detail}. The downloaded data has been left "
            f"in place (along with staging remnants at {staging_dir}) for "
            "inspection — re-running will not re-download it until this "
            "directory is fixed or removed."
        )

    shutil.rmtree(staging_dir, ignore_errors=True)

    image_count = _count_images(config.PLANTVILLAGE_COLOR_DIR)
    class_count = sum(1 for p in config.PLANTVILLAGE_COLOR_DIR.iterdir() if p.is_dir())
    provenance.record(
        "plantvillage",
        source=f"kaggle:{config.KAGGLE_DATASET_SLUG}",
        image_count=image_count,
        class_count=class_count,
        hash_base_dir=config.PLANTVILLAGE_COLOR_DIR,
    )

    print(
        f"[download] PlantVillage color/ ready at "
        f"{config.PLANTVILLAGE_COLOR_DIR} ({image_count} images)."
    )
    return config.PLANTVILLAGE_COLOR_DIR


def download_plantdoc(force: bool = False) -> Path:
    """Clones the PlantDoc classification dataset (train/ + test/) into
    config.PLANTDOC_DIR. Used as an external test set only — never trained
    or validated on. Cloned directly under Drive via a staging subdirectory
    — never /tmp — so a validation failure after cloning leaves the cloned
    contents on disk instead of losing them to temp-directory cleanup.
    """
    _require_colab("PlantDoc")

    if not force and config.PLANTDOC_DIR.is_dir():
        valid, detail = _validate_plantdoc_dirs(config.PLANTDOC_TRAIN_DIR, config.PLANTDOC_TEST_DIR)
        if valid:
            print(f"[download] PlantDoc already present and valid ({detail}) — skipping download.")
            return config.PLANTDOC_DIR
        if any(config.PLANTDOC_DIR.iterdir()):
            raise RuntimeError(
                f"{config.PLANTDOC_DIR} already exists but is invalid: {detail}. "
                "Inspect and remove this directory manually before re-running, "
                "or pass force=True."
            )

    config.PLANTDOC_DIR.mkdir(parents=True, exist_ok=True)
    staging_dir = config.PLANTDOC_DIR / "_staging_download"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)

    print(f"[download] Cloning '{config.PLANTDOC_REPO_URL}' directly to {staging_dir} ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", f"{config.PLANTDOC_REPO_URL}.git", str(staging_dir)],
        check=True,
    )
    commit_hash = subprocess.run(
        ["git", "-C", str(staging_dir), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    # git clone always places the repo root directly at staging_dir (unlike
    # a zip archive, there is no wrapper-directory case to handle here).
    train_src = staging_dir / "train"
    test_src = staging_dir / "test"
    if not train_src.is_dir() or not test_src.is_dir():
        raise RuntimeError(
            f"Expected 'train' and 'test' directories directly in cloned "
            f"repo at {staging_dir}, found: "
            f"{sorted(p.name for p in staging_dir.iterdir())}. The "
            "PlantDoc-Dataset repo layout may have changed. Staging left in "
            f"place at {staging_dir} for inspection."
        )

    for dest, src in ((config.PLANTDOC_TRAIN_DIR, train_src), (config.PLANTDOC_TEST_DIR, test_src)):
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(src), str(dest))

    valid, detail = _validate_plantdoc_dirs(config.PLANTDOC_TRAIN_DIR, config.PLANTDOC_TEST_DIR)
    if not valid:
        raise RuntimeError(
            f"Cloned PlantDoc at {config.PLANTDOC_DIR} failed validation: "
            f"{detail}. The downloaded data has been left in place (along "
            f"with staging remnants at {staging_dir}) for inspection — "
            "re-running will not re-clone it until this is fixed or removed."
        )

    shutil.rmtree(staging_dir, ignore_errors=True)

    image_count = _count_images(config.PLANTDOC_TRAIN_DIR) + _count_images(config.PLANTDOC_TEST_DIR)
    class_names = set()
    for split_dir in (config.PLANTDOC_TRAIN_DIR, config.PLANTDOC_TEST_DIR):
        class_names.update(p.name for p in split_dir.iterdir() if p.is_dir())

    provenance.record(
        "plantdoc",
        source=f"git:{config.PLANTDOC_REPO_URL}@{commit_hash}",
        image_count=image_count,
        class_count=len(class_names),
        hash_base_dir=config.PLANTDOC_DIR,
        hash_subdirs=[config.PLANTDOC_TRAIN_DIR, config.PLANTDOC_TEST_DIR],
    )

    print(f"[download] PlantDoc ready at {config.PLANTDOC_DIR} ({image_count} images).")
    return config.PLANTDOC_DIR


def main() -> None:
    download_plantvillage()
    download_plantdoc()


if __name__ == "__main__":
    main()
