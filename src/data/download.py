"""Downloads PlantVillage (Kaggle) and PlantDoc (public) into raw data dirs.
Verifies expected image and class counts after download; fails loudly on mismatch.

Only ever runs on Colab — see the IS_COLAB guard in each download function.
Both downloads are idempotent: if valid data is already present, they skip.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config  # noqa: E402


def _count_images(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in config.IMAGE_EXTENSIONS
    )


def _find_single_dir_named(root: Path, name: str) -> Path:
    """Finds exactly one directory under `root` whose basename == `name`
    (exact match, not a substring/fuzzy search). Raises if zero or more than
    one match is found, rather than guessing which one is correct.
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


def download_plantvillage(force: bool = False) -> Path:
    """Downloads the PlantVillage color dataset from Kaggle into
    config.PLANTVILLAGE_COLOR_DIR. Idempotent: skips if the directory
    already contains exactly config.NUM_CLASSES class folders and exactly
    config.PLANTVILLAGE_EXPECTED_IMAGE_COUNT images.
    """
    _require_colab("PlantVillage")

    if not force and config.PLANTVILLAGE_COLOR_DIR.is_dir():
        existing_count = _count_images(config.PLANTVILLAGE_COLOR_DIR)
        existing_classes = sorted(
            p.name for p in config.PLANTVILLAGE_COLOR_DIR.iterdir() if p.is_dir()
        )
        if (
            existing_count == config.PLANTVILLAGE_EXPECTED_IMAGE_COUNT
            and existing_classes == sorted(config.PLANTVILLAGE_CLASS_NAMES)
        ):
            print(
                f"[download] PlantVillage already present and valid "
                f"({existing_count} images, {len(existing_classes)} classes) "
                "— skipping download."
            )
            return config.PLANTVILLAGE_COLOR_DIR
        if existing_count > 0:
            raise RuntimeError(
                f"{config.PLANTVILLAGE_COLOR_DIR} already exists but is "
                f"invalid: found {existing_count} images across "
                f"{len(existing_classes)} classes, expected "
                f"{config.PLANTVILLAGE_EXPECTED_IMAGE_COUNT} images across "
                f"{config.NUM_CLASSES} classes. Inspect and remove this "
                "directory manually before re-running, or pass force=True."
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

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        print(f"[download] Downloading Kaggle dataset '{config.KAGGLE_DATASET_SLUG}'...")
        api.dataset_download_files(
            config.KAGGLE_DATASET_SLUG, path=str(tmp_path), unzip=True, quiet=False
        )

        color_dir = _find_single_dir_named(tmp_path, "color")
        color_image_count = _count_images(color_dir)
        if color_image_count != config.PLANTVILLAGE_EXPECTED_IMAGE_COUNT:
            raise RuntimeError(
                f"Downloaded PlantVillage color/ contains "
                f"{color_image_count} images, expected "
                f"{config.PLANTVILLAGE_EXPECTED_IMAGE_COUNT}. Refusing to "
                "proceed with a mismatched dataset."
            )

        config.PLANTVILLAGE_DIR.mkdir(parents=True, exist_ok=True)
        if config.PLANTVILLAGE_COLOR_DIR.exists():
            shutil.rmtree(config.PLANTVILLAGE_COLOR_DIR)
        shutil.move(str(color_dir), str(config.PLANTVILLAGE_COLOR_DIR))

    print(
        f"[download] PlantVillage color/ ready at "
        f"{config.PLANTVILLAGE_COLOR_DIR} ({config.PLANTVILLAGE_EXPECTED_IMAGE_COUNT} images)."
    )
    return config.PLANTVILLAGE_COLOR_DIR


def download_plantdoc(force: bool = False) -> Path:
    """Clones the PlantDoc classification dataset (train/ + test/) into
    config.PLANTDOC_DIR. Used as an external test set only — never trained
    or validated on. Idempotent within a tolerance band on total image
    count, since this is a live GitHub repo rather than a frozen release.
    """
    _require_colab("PlantDoc")

    if not force and config.PLANTDOC_DIR.is_dir():
        existing_count = _count_images(config.PLANTDOC_DIR)
        lower = config.PLANTDOC_EXPECTED_IMAGE_COUNT - config.PLANTDOC_IMAGE_COUNT_TOLERANCE
        upper = config.PLANTDOC_EXPECTED_IMAGE_COUNT + config.PLANTDOC_IMAGE_COUNT_TOLERANCE
        if lower <= existing_count <= upper:
            print(
                f"[download] PlantDoc already present and valid "
                f"({existing_count} images) — skipping download."
            )
            return config.PLANTDOC_DIR
        if existing_count > 0:
            raise RuntimeError(
                f"{config.PLANTDOC_DIR} already exists but is invalid: found "
                f"{existing_count} images, expected {config.PLANTDOC_EXPECTED_IMAGE_COUNT} "
                f"+/- {config.PLANTDOC_IMAGE_COUNT_TOLERANCE}. Inspect and remove this "
                "directory manually before re-running, or pass force=True."
            )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "PlantDoc-Dataset"
        print(f"[download] Cloning '{config.PLANTDOC_REPO_URL}'...")
        subprocess.run(
            ["git", "clone", "--depth", "1", f"{config.PLANTDOC_REPO_URL}.git", str(tmp_path)],
            check=True,
        )

        train_src = tmp_path / "train"
        test_src = tmp_path / "test"
        if not train_src.is_dir() or not test_src.is_dir():
            raise RuntimeError(
                f"Expected 'train' and 'test' directories in cloned repo at "
                f"{tmp_path}, found: {sorted(p.name for p in tmp_path.iterdir())}. "
                "The PlantDoc-Dataset repo layout may have changed."
            )

        total_count = _count_images(train_src) + _count_images(test_src)
        lower = config.PLANTDOC_EXPECTED_IMAGE_COUNT - config.PLANTDOC_IMAGE_COUNT_TOLERANCE
        upper = config.PLANTDOC_EXPECTED_IMAGE_COUNT + config.PLANTDOC_IMAGE_COUNT_TOLERANCE
        if not (lower <= total_count <= upper):
            raise RuntimeError(
                f"Cloned PlantDoc train+test contains {total_count} images, "
                f"expected {config.PLANTDOC_EXPECTED_IMAGE_COUNT} +/- "
                f"{config.PLANTDOC_IMAGE_COUNT_TOLERANCE}. Refusing to "
                "proceed with a mismatched dataset."
            )

        config.PLANTDOC_DIR.mkdir(parents=True, exist_ok=True)
        for name, src in (("train", train_src), ("test", test_src)):
            dest = config.PLANTDOC_DIR / name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(src), str(dest))

    print(f"[download] PlantDoc ready at {config.PLANTDOC_DIR} ({total_count} images).")
    return config.PLANTDOC_DIR


def main() -> None:
    download_plantvillage()
    download_plantdoc()


if __name__ == "__main__":
    main()
