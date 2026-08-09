"""Groups PlantVillage color images into duplicate clusters by perceptual
hash (imagehash.phash), so src/data/split.py can keep every photo of the
same physical leaf entirely inside one split.

This module NEVER deletes, moves, or renames any image file. "Deduplication"
here means grouping images for the split — every image stays exactly where
it is; the only output is a group_id assignment (artifacts/image_groups.json)
plus reports (artifacts/dedupe_report.json, artifacts/dedupe_report.md).

Clusters never span class boundaries: a class name is the "boundary_key"
passed to src/data/phash_cluster.py, so two images in different classes are
never merged even if their hashes are within config.DEDUPE_HAMMING_THRESHOLD
— instead that pair is recorded as a cross-class collision, since two
different classes' images looking near-identical is a labeling anomaly worth
surfacing, not a duplicate to fold together.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import imagehash
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config  # noqa: E402
from src.data import phash_cluster  # noqa: E402


def compute_hashes() -> dict[str, tuple[str, imagehash.ImageHash]]:
    """Returns {relative_path: (class_name, phash)} for every image under
    config.PLANTVILLAGE_COLOR_DIR. Raises if the directory is missing, empty,
    or any image can't be opened/hashed — a silently-skipped image would
    break the total-count reconciliation src/data/split.py asserts later
    (CLAUDE.md rule 1: fail loudly, never silently drop data).
    """
    if not config.PLANTVILLAGE_COLOR_DIR.is_dir():
        raise FileNotFoundError(
            f"{config.PLANTVILLAGE_COLOR_DIR} does not exist. Run "
            "src/data/download.py first."
        )

    hashes: dict[str, tuple[str, imagehash.ImageHash]] = {}
    for class_dir in sorted(config.PLANTVILLAGE_COLOR_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        for path in sorted(class_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in config.IMAGE_EXTENSIONS:
                continue
            relative_path = str(path.relative_to(config.PLANTVILLAGE_COLOR_DIR)).replace("\\", "/")
            try:
                with Image.open(path) as img:
                    phash = imagehash.phash(img, hash_size=config.PHASH_SIZE)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to open/hash {path}: {exc}. Every image under "
                    f"{config.PLANTVILLAGE_COLOR_DIR} must be hashable — "
                    "inspect and fix or remove this file rather than "
                    "silently skipping it."
                ) from exc
            hashes[relative_path] = (class_name, phash)

    if not hashes:
        raise RuntimeError(f"No images found under {config.PLANTVILLAGE_COLOR_DIR}.")

    return hashes


def _class_of(relative_path: str) -> str:
    return relative_path.split("/")[0]


def build_dedupe_report(
    image_hashes: dict[str, tuple[str, imagehash.ImageHash]],
    image_to_group: dict[str, str],
    collisions: list[dict],
) -> dict:
    groups_by_class: dict[str, set[str]] = defaultdict(set)
    images_by_group: dict[str, list[str]] = defaultdict(list)
    for path, group_id in image_to_group.items():
        images_by_group[group_id].append(path)
        groups_by_class[_class_of(path)].add(group_id)

    cluster_sizes = {group_id: len(members) for group_id, members in images_by_group.items()}
    duplicate_clusters = {gid: size for gid, size in cluster_sizes.items() if size > 1}

    per_class = {}
    for class_name in sorted(config.PLANTVILLAGE_CLASS_NAMES):
        class_group_ids = groups_by_class.get(class_name, set())
        class_num_images = sum(1 for path in image_to_group if _class_of(path) == class_name)
        class_duplicate_sizes = [cluster_sizes[gid] for gid in class_group_ids if cluster_sizes[gid] > 1]
        per_class[class_name] = {
            "num_images": class_num_images,
            "num_clusters": len(class_group_ids),
            "num_images_in_duplicate_clusters": sum(class_duplicate_sizes),
            "largest_cluster_size": max((cluster_sizes[gid] for gid in class_group_ids), default=0),
        }

    return {
        "total_images": len(image_to_group),
        "num_clusters": len(cluster_sizes),
        "num_images_in_duplicate_clusters": sum(duplicate_clusters.values()),
        "largest_cluster_size": max(cluster_sizes.values(), default=0),
        "hamming_threshold": config.DEDUPE_HAMMING_THRESHOLD,
        "per_class": per_class,
        "num_cross_class_collisions": len(collisions),
        "cross_class_collisions": sorted(
            collisions, key=lambda c: (c["item_a"], c["item_b"])
        ),
    }


def write_json(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# PlantVillage Deduplication Report",
        "",
        f"- Total images: **{report['total_images']}**",
        f"- Hamming distance threshold: **{report['hamming_threshold']}**",
        f"- Clusters (distinct physical leaves, singletons included): **{report['num_clusters']}**",
        f"- Images in a cluster of size > 1: **{report['num_images_in_duplicate_clusters']}**",
        f"- Largest cluster size: **{report['largest_cluster_size']}**",
        f"- Cross-class collisions: **{report['num_cross_class_collisions']}**",
        "",
        "## Per-class breakdown",
        "",
        "| Class | Images | Clusters | Images in dup. clusters | Largest cluster |",
        "|---|---|---|---|---|",
    ]
    for class_name, stats in report["per_class"].items():
        lines.append(
            f"| {class_name} | {stats['num_images']} | {stats['num_clusters']} | "
            f"{stats['num_images_in_duplicate_clusters']} | {stats['largest_cluster_size']} |"
        )

    if report["cross_class_collisions"]:
        lines += ["", "## Cross-class collisions", "", "| Image A | Class A | Image B | Class B | Hamming distance |", "|---|---|---|---|---|"]
        for c in report["cross_class_collisions"]:
            lines.append(
                f"| {c['item_a']} | {c['boundary_a']} | {c['item_b']} | {c['boundary_b']} | {c['hamming_distance']} |"
            )
    else:
        lines += ["", "## Cross-class collisions", "", "None found."]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    image_hashes = compute_hashes()
    clusterable = {path: (cls, h) for path, (cls, h) in image_hashes.items()}
    image_to_group, collisions = phash_cluster.cluster(clusterable, config.DEDUPE_HAMMING_THRESHOLD)

    report = build_dedupe_report(image_hashes, image_to_group, collisions)
    write_json(report, config.ARTIFACTS_DIR / "dedupe_report.json")
    write_markdown(report, config.ARTIFACTS_DIR / "dedupe_report.md")

    groups_path = config.ARTIFACTS_DIR / "image_groups.json"
    groups_path.parent.mkdir(parents=True, exist_ok=True)
    groups_path.write_text(
        json.dumps(dict(sorted(image_to_group.items())), indent=2), encoding="utf-8"
    )

    print(
        f"[dedupe] {report['total_images']} images -> {report['num_clusters']} clusters "
        f"({report['num_images_in_duplicate_clusters']} images in clusters of size > 1, "
        f"largest {report['largest_cluster_size']}). "
        f"{report['num_cross_class_collisions']} cross-class collision(s). "
        f"Wrote {groups_path}, {config.ARTIFACTS_DIR / 'dedupe_report.json'}, "
        f"{config.ARTIFACTS_DIR / 'dedupe_report.md'}."
    )


if __name__ == "__main__":
    main()
