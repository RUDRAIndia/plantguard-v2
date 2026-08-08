"""Downloads PlantVillage (Kaggle) and PlantDoc (public) datasets.

Design (rewritten after real measurements on Colab — see git history for the
prior, broken design):

- **All per-image work happens on local disk** (config.DATA_ROOT, which is
  /content/data on Colab), never on the Drive FUSE mount. Drive writes and
  stats small files at only a few dozen per second — extracting ~54k color
  images directly onto Drive takes about an hour, and *validating* an
  existing Drive copy by walking it file-by-file is just as slow. That
  file-by-file walk was the actual cause of a prior idempotency bug: the
  "is valid data already here?" check itself took so long over Drive FUSE
  that it never reliably finished before something interrupted the session,
  so in practice the skip-if-valid branch never got a chance to fire and
  every session re-downloaded and re-extracted from scratch.
- **Drive is cold storage only**: after a successful local download+extract,
  each dataset is packed into a single tar (one big sequential write is
  fast; ~54,000 small ones are not) and copied to config.DRIVE_DATA_ROOT
  (see src/data/drive_tar.py). A later session prefers that tar — copying
  one large file off Drive and untarring locally is fast — and only falls
  back to Kaggle/git when the tar is absent. This is what makes sessions
  resumable despite local disk (/content) being wiped whenever the Colab
  runtime recycles.
- **The Drive-side integrity check is fast and walk-free**: tar presence + a
  coarse size floor, plus the observed image/class counts recorded in a
  Drive-side sidecar copy of dataset_provenance.json — never a full
  file-by-file scan over Drive.
- **The destination is never destroyed before its replacement is
  validated.** Every download/restore path writes into a
  `_staging_download` directory, validates the *staged* candidate, and only
  then removes the old destination (if any) and moves the candidate into
  place. A validation failure leaves the old data untouched and the
  candidate on disk for inspection. (A prior version of this module deleted
  the destination unconditionally before validating the replacement, which
  once destroyed a complete, valid 54,305-image color/ directory when the
  run after it turned out to be redundant — see CLAUDE.md's build/validate/
  swap rule.)

Only ever runs on Colab — see the IS_COLAB guard in each download function.
Kaggle/git acquisition lives in src/data/fetch.py, Drive persistence in
src/data/drive_tar.py, provenance recording in src/data/provenance.py.
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config  # noqa: E402
from src.data import drive_tar, fetch, provenance, validate  # noqa: E402


def _require_colab(dataset_name: str) -> None:
    if not config.IS_COLAB:
        raise RuntimeError(
            f"Refusing to download {dataset_name}: this is not a Colab "
            "environment. Per project convention, full dataset downloads "
            "only ever happen from a Colab session (CLAUDE.md rule 10) — "
            "the local laptop only ever holds the smoke-test subset."
        )


def download_plantvillage(force: bool = False) -> Path:
    """Ensures config.PLANTVILLAGE_COLOR_DIR (local disk) holds a valid
    PlantVillage color/ dataset. Idempotent: skips if already present and
    valid. Prefers restoring from the Drive tar over Kaggle when the tar
    passes the fast integrity check; only falls back to Kaggle when the tar
    is absent (or fails that check, or `force=True`). The destination is
    never touched until the candidate (freshly downloaded or tar-restored)
    has been validated in staging.
    """
    _require_colab("PlantVillage")

    if not force and config.PLANTVILLAGE_COLOR_DIR.is_dir():
        valid, detail = validate.validate_plantvillage_dir(config.PLANTVILLAGE_COLOR_DIR)
        if valid:
            print(
                f"[download] PlantVillage already present on local disk and "
                f"valid ({detail}) — skipping download."
            )
            return config.PLANTVILLAGE_COLOR_DIR
        if any(config.PLANTVILLAGE_COLOR_DIR.iterdir()):
            raise RuntimeError(
                f"{config.PLANTVILLAGE_COLOR_DIR} already exists but is "
                f"invalid: {detail}. Inspect and remove this directory "
                "manually before re-running, or pass force=True."
            )

    config.PLANTVILLAGE_DIR.mkdir(parents=True, exist_ok=True)
    staging_dir = config.PLANTVILLAGE_DIR / "_staging_download"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    tar_path = config.PLANTVILLAGE_COLOR_TAR
    restored_from_tar = not force and drive_tar.tar_integrity_ok(
        "plantvillage",
        tar_path,
        config.MIN_PLANTVILLAGE_TAR_BYTES,
        config.PLANTVILLAGE_EXPECTED_IMAGE_COUNT_RANGE,
        expected_class_count=config.NUM_CLASSES,
    )
    if restored_from_tar:
        print(f"[download] Valid tar found at {tar_path} — restoring from Drive cold storage (skipping Kaggle).")
        drive_tar.restore_tar_to(tar_path, staging_dir)
        candidate = staging_dir / "color"
        source = f"drive-tar:{tar_path}"
    else:
        candidate = fetch.download_plantvillage_color_from_kaggle(staging_dir)
        source = f"kaggle:{config.KAGGLE_DATASET_SLUG}"

    valid, detail = validate.validate_plantvillage_dir(candidate)
    if not valid:
        raise RuntimeError(
            f"{'Tar restored from' if restored_from_tar else 'Downloaded'} "
            f"PlantVillage color/ at {candidate} failed validation: {detail}. "
            f"Existing data at {config.PLANTVILLAGE_COLOR_DIR} (if any) was "
            f"left completely untouched — the candidate has been left on "
            "disk for inspection instead of being swapped in."
        )

    if config.PLANTVILLAGE_COLOR_DIR.exists():
        shutil.rmtree(config.PLANTVILLAGE_COLOR_DIR)
    shutil.move(str(candidate), str(config.PLANTVILLAGE_COLOR_DIR))

    image_count = validate.count_images(config.PLANTVILLAGE_COLOR_DIR)
    class_count = sum(1 for p in config.PLANTVILLAGE_COLOR_DIR.iterdir() if p.is_dir())
    provenance.record(
        "plantvillage",
        source=source,
        image_count=image_count,
        class_count=class_count,
        hash_base_dir=config.PLANTVILLAGE_COLOR_DIR,
    )

    if not restored_from_tar:
        drive_tar.persist_to_drive_tar(
            "plantvillage",
            tar_path,
            [(config.PLANTVILLAGE_COLOR_DIR, "color")],
            config.MIN_PLANTVILLAGE_TAR_BYTES,
        )

    shutil.rmtree(staging_dir, ignore_errors=True)

    print(
        f"[download] PlantVillage color/ ready at "
        f"{config.PLANTVILLAGE_COLOR_DIR} ({image_count} images)."
    )
    return config.PLANTVILLAGE_COLOR_DIR


def download_plantdoc(force: bool = False) -> Path:
    """Ensures config.PLANTDOC_TRAIN_DIR / config.PLANTDOC_TEST_DIR (local
    disk) hold a valid PlantDoc dataset. Used as an external test set only —
    never trained or validated on. Same tar-first, validate-before-swap
    design as download_plantvillage.
    """
    _require_colab("PlantDoc")

    if not force and config.PLANTDOC_DIR.is_dir():
        valid, detail = validate.validate_plantdoc_dirs(config.PLANTDOC_TRAIN_DIR, config.PLANTDOC_TEST_DIR)
        if valid:
            print(f"[download] PlantDoc already present on local disk and valid ({detail}) — skipping download.")
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
    staging_dir.mkdir(parents=True)

    tar_path = config.PLANTDOC_TAR
    restored_from_tar = not force and drive_tar.tar_integrity_ok(
        "plantdoc",
        tar_path,
        config.MIN_PLANTDOC_TAR_BYTES,
        config.PLANTDOC_EXPECTED_IMAGE_COUNT_RANGE,
    )
    if restored_from_tar:
        print(f"[download] Valid tar found at {tar_path} — restoring from Drive cold storage (skipping git clone).")
        drive_tar.restore_tar_to(tar_path, staging_dir)
        train_src, test_src = staging_dir / "train", staging_dir / "test"
        source = f"drive-tar:{tar_path}"
    else:
        commit_hash = fetch.clone_plantdoc_to(staging_dir)
        # git clone always places the repo root directly at staging_dir
        # (unlike a zip archive, there is no wrapper-directory case here).
        train_src, test_src = staging_dir / "train", staging_dir / "test"
        if not train_src.is_dir() or not test_src.is_dir():
            raise RuntimeError(
                f"Expected 'train' and 'test' directories directly in cloned "
                f"repo at {staging_dir}, found: "
                f"{sorted(p.name for p in staging_dir.iterdir())}. The "
                "PlantDoc-Dataset repo layout may have changed."
            )
        source = f"git:{config.PLANTDOC_REPO_URL}@{commit_hash}"

    valid, detail = validate.validate_plantdoc_dirs(train_src, test_src)
    if not valid:
        raise RuntimeError(
            f"{'Tar restored from' if restored_from_tar else 'Cloned'} "
            f"PlantDoc failed validation: {detail}. Existing data (if any) "
            f"at {config.PLANTDOC_TRAIN_DIR} / {config.PLANTDOC_TEST_DIR} "
            f"was left completely untouched — the candidates have been left "
            f"at {train_src} and {test_src} for inspection."
        )

    for dest, src in ((config.PLANTDOC_TRAIN_DIR, train_src), (config.PLANTDOC_TEST_DIR, test_src)):
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(src), str(dest))

    image_count = validate.count_images(config.PLANTDOC_TRAIN_DIR) + validate.count_images(config.PLANTDOC_TEST_DIR)
    class_names = set()
    for split_dir in (config.PLANTDOC_TRAIN_DIR, config.PLANTDOC_TEST_DIR):
        class_names.update(p.name for p in split_dir.iterdir() if p.is_dir())

    provenance.record(
        "plantdoc",
        source=source,
        image_count=image_count,
        class_count=len(class_names),
        hash_base_dir=config.PLANTDOC_DIR,
        hash_subdirs=[config.PLANTDOC_TRAIN_DIR, config.PLANTDOC_TEST_DIR],
    )

    if not restored_from_tar:
        drive_tar.persist_to_drive_tar(
            "plantdoc",
            tar_path,
            [(config.PLANTDOC_TRAIN_DIR, "train"), (config.PLANTDOC_TEST_DIR, "test")],
            config.MIN_PLANTDOC_TAR_BYTES,
        )

    shutil.rmtree(staging_dir, ignore_errors=True)

    print(f"[download] PlantDoc ready at {config.PLANTDOC_DIR} ({image_count} images).")
    return config.PLANTDOC_DIR


def main() -> None:
    download_plantvillage()
    download_plantdoc()


if __name__ == "__main__":
    main()
