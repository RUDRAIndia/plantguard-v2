"""Drive cold-storage helpers for src/data/download.py.

Drive is used only as cold storage for large, single-file tar archives —
never for per-image work (see download.py's module docstring for why: Drive
FUSE writes/stats small files at only a few dozen per second). This module
checks a Drive-side tar's fast, walk-free integrity signal, restores it to
local disk, and persists a freshly validated local dataset back to Drive as
a tar — always validating the new tar before it replaces anything on Drive
(CLAUDE.md's build/validate/swap rule).
"""

import json
import shutil
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config  # noqa: E402


def tar_integrity_ok(
    dataset_key: str,
    tar_path: Path,
    min_bytes: int,
    expected_count_range: tuple,
    expected_class_count: int | None = None,
) -> bool:
    """Fast pre-flight check for whether `tar_path` can be trusted as a
    stand-in for a fresh download — WITHOUT extracting it or walking any
    files over Drive (that per-file walk is exactly the FUSE bottleneck this
    design exists to avoid). Checks tar presence + a coarse size floor, plus
    the observed image/class counts recorded in the Drive-side sidecar copy
    of dataset_provenance.json.
    """
    if tar_path is None or not tar_path.is_file():
        return False
    if tar_path.stat().st_size < min_bytes:
        return False

    prov_path = config.DRIVE_PROVENANCE_PATH
    if prov_path is None or not prov_path.is_file():
        return False
    try:
        data = json.loads(prov_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False

    entry = data.get(dataset_key)
    if not isinstance(entry, dict):
        return False

    count = entry.get("observed_image_count")
    lo, hi = expected_count_range
    if not isinstance(count, int) or not (lo <= count <= hi):
        return False

    if expected_class_count is not None and entry.get("observed_class_count") != expected_class_count:
        return False

    return True


def restore_tar_to(tar_path: Path, dest_root: Path) -> None:
    """Copies `tar_path` from Drive to local disk and extracts it there.
    Copying a single large file is fast even over Drive FUSE (one
    sequential read); it's per-file metadata operations on Drive that are
    slow, and this does none of those.
    """
    local_tar = dest_root / tar_path.name
    print(f"[drive_tar] Copying {tar_path} to local disk at {local_tar} ...")
    shutil.copyfile(tar_path, local_tar)
    print(f"[drive_tar] Extracting {local_tar} locally ...")
    with tarfile.open(local_tar) as tf:
        tf.extractall(path=dest_root)
    local_tar.unlink()


def _cleanup_legacy_drive_staging(dataset_key: str) -> None:
    """One-off cleanup for the old design's leftover _staging_download
    directory on Drive (from when this project extracted per-image data
    directly under Drive — see download.py's module docstring). Only ever
    removes a directory literally named _staging_download, and only called
    after a validated tar has already been swapped in, so nothing depends on
    the old per-image Drive layout anymore.
    """
    legacy_staging = config.DRIVE_DATA_ROOT / dataset_key / "_staging_download"
    if legacy_staging.is_dir():
        print(
            f"[drive_tar] Removing leftover {legacy_staging} from Drive "
            f"(superseded by the {dataset_key} tar)."
        )
        shutil.rmtree(legacy_staging)


def persist_to_drive_tar(dataset_key: str, tar_path: Path, members: list, min_bytes: int) -> None:
    """Builds a tar of `members` ((local_dir, arcname) pairs) to a temporary
    file on Drive, verifies it is a readable, non-trivial tar, and only then
    swaps it in for `tar_path` — any existing tar is never removed until the
    replacement is confirmed valid. Also copies dataset_provenance.json to
    Drive as a sidecar, since a fresh Colab session's git clone never has
    last session's copy (artifacts/ is gitignored) and tar_integrity_ok
    depends on it being there.
    """
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_tar_path = tar_path.with_name(tar_path.name + ".tmp")
    if tmp_tar_path.exists():
        tmp_tar_path.unlink()

    print(f"[drive_tar] Writing {dataset_key} to a single tar at {tmp_tar_path} ...")
    with tarfile.open(tmp_tar_path, "w") as tf:
        for local_dir, arcname in members:
            tf.add(local_dir, arcname=arcname)

    if not tarfile.is_tarfile(tmp_tar_path) or tmp_tar_path.stat().st_size < min_bytes:
        raise RuntimeError(
            f"Freshly written tar at {tmp_tar_path} failed its own sanity "
            f"check (not a valid tar, or under {min_bytes} bytes) — "
            f"refusing to replace {tar_path}. The existing tar (if any) at "
            f"{tar_path} has been left untouched."
        )
    with tarfile.open(tmp_tar_path) as tf:
        member_file_count = sum(1 for m in tf.getmembers() if m.isfile())
    if member_file_count == 0:
        raise RuntimeError(
            f"Freshly written tar at {tmp_tar_path} contains zero files — "
            f"refusing to replace {tar_path}."
        )

    tmp_tar_path.replace(tar_path)
    shutil.copyfile(config.DATASET_PROVENANCE_PATH, config.DRIVE_PROVENANCE_PATH)
    _cleanup_legacy_drive_staging(dataset_key)

    print(
        f"[drive_tar] {dataset_key} persisted to Drive cold storage at "
        f"{tar_path} ({member_file_count} files)."
    )
