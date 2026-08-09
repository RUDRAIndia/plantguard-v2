"""Acquires raw dataset archives onto local disk.

Pure "get bytes from Kaggle/git onto local disk" logic — validation and
Drive persistence live in src/data/download.py and src/data/drive_tar.py
respectively. See download.py's module docstring for the overall design.
"""

import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config  # noqa: E402


def _extract_color_only(zip_path: Path, extract_root: Path) -> Path:
    """Extracts only the members that live under a directory named exactly
    "color" inside the zip (at any depth, exact path-component match — the
    archive may wrap everything in a container directory), skipping
    grayscale/ and segmented/ entirely. This is what keeps local extraction
    to ~54k files instead of the ~162k the full archive contains.
    """
    with zipfile.ZipFile(zip_path) as zf:
        all_names = zf.namelist()
        color_roots = set()
        for name in all_names:
            parts = Path(name.replace("\\", "/")).parts
            for i, part in enumerate(parts):
                if part == "color":
                    color_roots.add("/".join(parts[: i + 1]))
                    break

        if len(color_roots) == 0:
            raise RuntimeError(
                f"No directory named exactly 'color' found inside {zip_path}. "
                "The Kaggle archive layout may have changed — inspect it "
                "manually before proceeding."
            )
        if len(color_roots) > 1:
            raise RuntimeError(
                f"Found {len(color_roots)} directories named exactly 'color' "
                f"inside {zip_path}: {sorted(color_roots)}. Ambiguous — "
                "refusing to guess which one is correct."
            )
        color_root = next(iter(color_roots))

        members = [
            name
            for name in all_names
            if name.replace("\\", "/") == color_root
            or name.replace("\\", "/").startswith(color_root + "/")
        ]
        print(
            f"[fetch] Extracting {len(members)} files under '{color_root}/' "
            "only — grayscale/ and segmented/ are skipped entirely."
        )
        zf.extractall(path=extract_root, members=members)

    return extract_root / color_root


def download_plantvillage_color_from_kaggle(staging_dir: Path) -> Path:
    """Downloads the Kaggle archive (zip, not auto-unzipped) to local disk
    and extracts only its color/ subtree. Returns the path to the extracted
    color/ directory, still inside staging — not yet validated or moved.
    """
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

    print(
        f"[fetch] Downloading Kaggle dataset '{config.KAGGLE_DATASET_SLUG}' "
        f"to local disk at {staging_dir} (zip only, not yet extracted) ..."
    )
    api.dataset_download_files(config.KAGGLE_DATASET_SLUG, path=str(staging_dir), unzip=False, quiet=False)

    zip_matches = list(staging_dir.glob("*.zip"))
    if len(zip_matches) != 1:
        raise RuntimeError(
            f"Expected exactly one .zip file in {staging_dir} after Kaggle "
            f"download, found {len(zip_matches)}: {zip_matches}."
        )
    zip_path = zip_matches[0]

    color_dir = _extract_color_only(zip_path, staging_dir)
    zip_path.unlink()
    return color_dir


def _extract_named_dir_only(zip_path: Path, extract_root: Path, dir_name: str) -> Path:
    """Extracts only the members that live under a directory named exactly
    `dir_name` inside the zip (at any depth — same "search, don't guess"
    approach as _extract_color_only, since a third-party Kaggle dataset's
    exact internal nesting can't be relied on without a live download).
    """
    with zipfile.ZipFile(zip_path) as zf:
        all_names = zf.namelist()
        matching_roots = set()
        for name in all_names:
            parts = Path(name.replace("\\", "/")).parts
            for i, part in enumerate(parts):
                if part == dir_name:
                    matching_roots.add("/".join(parts[: i + 1]))
                    break

        if len(matching_roots) == 0:
            raise RuntimeError(
                f"No directory named exactly '{dir_name}' found inside {zip_path}."
            )
        if len(matching_roots) == 1:
            matched_root = next(iter(matching_roots))
        else:
            # Some third-party Kaggle archives nest a directory inside
            # another of the exact same name (e.g. "seg_train/seg_train/...")
            # — a known real quirk, not a hypothetical. That's unambiguous
            # (the deepest one is always the actual leaf-category holder),
            # so only raise when the matches AREN'T a strict nesting chain.
            ordered = sorted(matching_roots, key=len)
            is_nesting_chain = all(
                deeper == shallow or deeper.startswith(shallow + "/")
                for shallow, deeper in zip(ordered, ordered[1:])
            )
            if not is_nesting_chain:
                raise RuntimeError(
                    f"Found {len(matching_roots)} directories named exactly "
                    f"'{dir_name}' inside {zip_path}: {sorted(matching_roots)}. "
                    "Ambiguous — refusing to guess which one is correct."
                )
            matched_root = ordered[-1]

        members = [
            name
            for name in all_names
            if name.replace("\\", "/") == matched_root
            or name.replace("\\", "/").startswith(matched_root + "/")
        ]
        print(f"[fetch] Extracting {len(members)} files under '{matched_root}/' only.")
        zf.extractall(path=extract_root, members=members)

    return extract_root / matched_root


def download_negatives_from_kaggle(staging_dir: Path) -> Path:
    """Downloads config.NEGATIVES_KAGGLE_SLUG (zip, not auto-unzipped) to
    local disk and extracts only the config.NEGATIVES_SOURCE_SUBDIR_NAME
    subtree. Returns the path to that extracted directory, still inside
    staging — not yet subsampled, validated, or moved.
    """
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

    print(
        f"[fetch] Downloading Kaggle dataset '{config.NEGATIVES_KAGGLE_SLUG}' "
        f"to local disk at {staging_dir} (zip only, not yet extracted) ..."
    )
    api.dataset_download_files(config.NEGATIVES_KAGGLE_SLUG, path=str(staging_dir), unzip=False, quiet=False)

    zip_matches = list(staging_dir.glob("*.zip"))
    if len(zip_matches) != 1:
        raise RuntimeError(
            f"Expected exactly one .zip file in {staging_dir} after Kaggle "
            f"download, found {len(zip_matches)}: {zip_matches}."
        )
    zip_path = zip_matches[0]

    source_dir = _extract_named_dir_only(zip_path, staging_dir, config.NEGATIVES_SOURCE_SUBDIR_NAME)
    zip_path.unlink()
    return source_dir


def clone_plantdoc_to(staging_dir: Path) -> str:
    """Clones the PlantDoc classification repo directly to local disk.
    Returns the cloned commit hash.
    """
    print(f"[fetch] Cloning '{config.PLANTDOC_REPO_URL}' to local disk at {staging_dir} ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", f"{config.PLANTDOC_REPO_URL}.git", str(staging_dir)],
        check=True,
    )
    result = subprocess.run(
        ["git", "-C", str(staging_dir), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()
