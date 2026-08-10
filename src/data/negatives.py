"""Handles non-plant / out-of-distribution negative examples so the model
has a defined behavior for inputs that are not leaves at all, rather than
forcing a confident wrong prediction into one of the 38 classes.

Source: Kaggle's "Intel Image Classification" (puneet6060/intel-image-
classification, config.NEGATIVES_KAGGLE_SLUG) — ~25k photos of ordinary,
non-leaf scenes (buildings, forest, glacier, mountain, sea, street). Chosen
because on Colab it reuses the exact KaggleApi + zip-extraction machinery
already in src/data/fetch.py (no new download mechanism, same credentials
already set up for PlantVillage), and it is about as visually unlike a
studio leaf macro shot as an "ordinary photograph" dataset gets. On Kaggle
this dataset is instead a read-only mounted notebook input
(config.KAGGLE_NEGATIVES_SEG_TRAIN_DIR) — fetch.resolve_negatives_mount()
resolves it in place, no download, no KaggleApi call.

These images are NEVER part of the 38 PlantVillage classes and NEVER used
as training labels — they exist solely to calibrate an out-of-distribution
rejection threshold on the validation split, later. This module only
assembles and persists the set; calibration is a separate, later step.

The *destination* (config.NEGATIVES_DIR, the deterministic subsample this
module writes) always follows src/data/download.py's local-disk-first,
Drive-as-cold-storage pattern on Colab: local per-image work only, Drive
holds a single tar + provenance sidecar, and the destination is never
touched until a staged, validated candidate is ready to swap in (CLAUDE.md
rule 13). On Kaggle there is no Drive, so the persist-to-tar step is skipped
entirely (config.NEGATIVES_TAR is None) — NEGATIVES_DIR still resolves
under the writable /kaggle/working/data, just without cross-session
persistence beyond what CLAUDE.md's checkpoint docstring already describes
for that environment.
"""

import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config  # noqa: E402
from src.data import drive_tar, fetch, provenance, validate  # noqa: E402


def _require_colab_or_kaggle() -> None:
    if not (config.IS_COLAB or config.IS_KAGGLE):
        raise RuntimeError(
            "Refusing to assemble the negatives set: this is not a Colab or "
            "Kaggle environment. Per project convention, full dataset "
            "downloads only ever happen from a training session (CLAUDE.md "
            "rule 10)."
        )


def _deterministic_subsample(class_dirs: list, target_count: int, seed: int) -> list:
    """Pure function: given sorted category directories, deterministically
    picks `target_count` images spread as evenly as possible across them
    (any remainder goes to the first classes in sorted order), and returns
    their paths. Raises loudly, naming the category, if it doesn't have
    enough images to cover its allocation — never silently takes fewer.

    Determinism: each category's file list is sorted first, then shuffled
    with its own random.Random(f"{seed}:{category_name}") — same discipline
    as src/data/split.py, so re-running with the same seed/input always
    picks the same images regardless of filesystem iteration order.
    """
    class_dirs = sorted(class_dirs, key=lambda p: p.name)
    if not class_dirs:
        raise RuntimeError("No category directories given to subsample from.")

    num_classes = len(class_dirs)
    base, remainder = divmod(target_count, num_classes)
    selected = []
    for i, class_dir in enumerate(class_dirs):
        allocation = base + (1 if i < remainder else 0)
        files = sorted(
            p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS
        )
        if len(files) < allocation:
            raise RuntimeError(
                f"Category '{class_dir.name}' has only {len(files)} image(s), "
                f"needs {allocation} to meet its share of the {target_count}-image "
                "negative-set target."
            )
        rng = random.Random(f"{seed}:{class_dir.name}")
        rng.shuffle(files)
        selected.extend(files[:allocation])

    return selected


def download_negatives(force: bool = False) -> Path:
    """Ensures config.NEGATIVES_DIR holds a valid negatives set:
    config.NEGATIVES_TARGET_COUNT images deterministically subsampled from
    the source (Kaggle API download on Colab, the read-only mounted input on
    Kaggle), laid out as <original_category>/<file> for traceability
    (categories are never used as training labels). Idempotent,
    tar-first-on-Colab, validate-before-swap — same design as
    src/data/download.py's download_plantvillage. NEGATIVES_DIR itself is
    always a writable DATA_ROOT-relative destination, never the read-only
    Kaggle mount.
    """
    _require_colab_or_kaggle()

    if not force and config.NEGATIVES_DIR.is_dir():
        valid, detail = validate.validate_negatives_dir(
            config.NEGATIVES_DIR, config.NEGATIVES_EXPECTED_IMAGE_COUNT_RANGE
        )
        if valid:
            print(f"[negatives] Already present on local disk and valid ({detail}) — skipping download.")
            return config.NEGATIVES_DIR
        if any(config.NEGATIVES_DIR.iterdir()):
            raise RuntimeError(
                f"{config.NEGATIVES_DIR} already exists but is invalid: {detail}. "
                "Inspect and remove this directory manually before re-running, "
                "or pass force=True."
            )

    config.DATA_ROOT.mkdir(parents=True, exist_ok=True)
    staging_dir = config.DATA_ROOT / "_staging_download_negatives"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    tar_path = config.NEGATIVES_TAR
    restored_from_tar = not force and drive_tar.tar_integrity_ok(
        "negatives", tar_path, config.MIN_NEGATIVES_TAR_BYTES, config.NEGATIVES_EXPECTED_IMAGE_COUNT_RANGE
    )
    if restored_from_tar:
        print(f"[negatives] Valid tar found at {tar_path} — restoring from Drive cold storage (skipping Kaggle).")
        drive_tar.restore_tar_to(tar_path, staging_dir)
        candidate = staging_dir / "negatives"
        source = f"drive-tar:{tar_path}"
    else:
        if config.IS_KAGGLE:
            source_dir = fetch.resolve_negatives_mount()
            source_label = f"kaggle-input-mount:{config.KAGGLE_NEGATIVES_SEG_TRAIN_DIR}"
        else:
            source_dir = fetch.download_negatives_from_kaggle(staging_dir)
            source_label = f"kaggle:{config.NEGATIVES_KAGGLE_SLUG}"

        class_dirs = [p for p in source_dir.iterdir() if p.is_dir()]
        found_categories = sorted(p.name for p in class_dirs)
        expected_categories = sorted(config.NEGATIVES_EXPECTED_CATEGORIES)
        if found_categories != expected_categories:
            raise RuntimeError(
                f"Expected exactly the categories {expected_categories} directly "
                f"inside {source_dir}, found {found_categories}. The source "
                f"layout for '{config.NEGATIVES_KAGGLE_SLUG}' may have changed "
                "— inspect it manually before proceeding."
            )
        selected_files = _deterministic_subsample(class_dirs, config.NEGATIVES_TARGET_COUNT, config.SEED)

        candidate = staging_dir / "negatives"
        for src_file in selected_files:
            category = src_file.parent.name
            dest_file = candidate / category / src_file.name
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_file, dest_file)
        source = source_label

    valid, detail = validate.validate_negatives_dir(candidate, config.NEGATIVES_EXPECTED_IMAGE_COUNT_RANGE)
    if not valid:
        raise RuntimeError(
            f"{'Tar restored from' if restored_from_tar else 'Subsampled'} "
            f"negatives set at {candidate} failed validation: {detail}. "
            f"Existing data at {config.NEGATIVES_DIR} (if any) was left "
            f"completely untouched — the candidate has been left on disk "
            "for inspection instead of being swapped in."
        )

    if config.NEGATIVES_DIR.exists():
        shutil.rmtree(config.NEGATIVES_DIR)
    shutil.move(str(candidate), str(config.NEGATIVES_DIR))

    image_count = validate.count_images(config.NEGATIVES_DIR)
    category_count = sum(1 for p in config.NEGATIVES_DIR.iterdir() if p.is_dir())
    provenance.record(
        "negatives",
        source=source,
        image_count=image_count,
        class_count=category_count,
        hash_base_dir=config.NEGATIVES_DIR,
    )

    if not restored_from_tar and tar_path is not None:
        drive_tar.persist_to_drive_tar(
            "negatives", tar_path, [(config.NEGATIVES_DIR, "negatives")], config.MIN_NEGATIVES_TAR_BYTES
        )

    shutil.rmtree(staging_dir, ignore_errors=True)

    print(f"[negatives] Ready at {config.NEGATIVES_DIR} ({image_count} images across {category_count} categories).")
    return config.NEGATIVES_DIR


def main() -> None:
    download_negatives()


if __name__ == "__main__":
    main()
