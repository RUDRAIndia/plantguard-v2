"""Records dataset provenance metadata to config.DATASET_PROVENANCE_PATH
after a successful download: source, timestamp, observed image/class
counts, and a sha256 of the sorted list of relative file paths — so the
exact dataset version behind any report is reproducible.
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config  # noqa: E402


def _sha256_of_relative_paths(base_dir: Path, subdirs: list | None = None) -> str:
    if subdirs is None:
        subdirs = [base_dir]
    relative_paths = []
    for d in subdirs:
        relative_paths.extend(
            str(p.relative_to(base_dir)).replace("\\", "/")
            for p in d.rglob("*")
            if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS
        )
    digest = hashlib.sha256("\n".join(sorted(relative_paths)).encode("utf-8"))
    return digest.hexdigest()


def record(
    dataset_key: str,
    *,
    source: str,
    image_count: int,
    class_count: int,
    hash_base_dir: Path,
    hash_subdirs: list | None = None,
) -> None:
    """Read-modify-writes config.DATASET_PROVENANCE_PATH, setting/replacing
    only the entry for `dataset_key` so PlantVillage and PlantDoc provenance
    can be recorded independently into the same file.
    """
    entry = {
        "source": source,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "observed_image_count": image_count,
        "observed_class_count": class_count,
        "sha256_of_sorted_relative_paths": _sha256_of_relative_paths(
            hash_base_dir, hash_subdirs
        ),
    }

    path = config.DATASET_PROVENANCE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    data[dataset_key] = entry
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
