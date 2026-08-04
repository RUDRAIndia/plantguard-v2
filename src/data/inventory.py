"""Scans the downloaded PlantVillage color dataset and produces
artifacts/inventory.json and artifacts/inventory.md: total image count,
per-class counts, class imbalance, healthy-class count, image dimension
distribution, and any corrupt/unreadable files. Fails loudly if the
discovered class set does not exactly match config.PLANTVILLAGE_CLASS_NAMES.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config  # noqa: E402

from PIL import Image, UnidentifiedImageError  # noqa: E402


def _scan_class(class_dir: Path) -> tuple[int, dict[str, int], list[dict]]:
    """Returns (image_count, {"WxH": count}, [corrupt file records])
    for a single class directory.
    """
    dimension_counts: dict[str, int] = {}
    corrupt: list[dict] = []
    count = 0

    for path in sorted(class_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in config.IMAGE_EXTENSIONS:
            continue
        count += 1
        try:
            with Image.open(path) as img:
                width, height = img.size
                img.verify()
        except (OSError, UnidentifiedImageError) as exc:
            corrupt.append({"path": str(path), "error": str(exc)})
            continue
        key = f"{width}x{height}"
        dimension_counts[key] = dimension_counts.get(key, 0) + 1

    return count, dimension_counts, corrupt


def scan_plantvillage() -> dict:
    if not config.PLANTVILLAGE_COLOR_DIR.is_dir():
        raise FileNotFoundError(
            f"{config.PLANTVILLAGE_COLOR_DIR} does not exist. Run "
            "src/data/download.py first."
        )

    discovered_classes = sorted(
        p.name for p in config.PLANTVILLAGE_COLOR_DIR.iterdir() if p.is_dir()
    )
    expected_classes = sorted(config.PLANTVILLAGE_CLASS_NAMES)
    if discovered_classes != expected_classes:
        missing = sorted(set(expected_classes) - set(discovered_classes))
        extra = sorted(set(discovered_classes) - set(expected_classes))
        raise AssertionError(
            f"Discovered {len(discovered_classes)} class folders, expected "
            f"exactly {config.NUM_CLASSES}. Missing: {missing}. "
            f"Unexpected: {extra}."
        )

    per_class_counts: dict[str, int] = {}
    dimension_totals: dict[str, int] = {}
    corrupt_files: list[dict] = []

    for class_name in discovered_classes:
        count, dims, corrupt = _scan_class(config.PLANTVILLAGE_COLOR_DIR / class_name)
        per_class_counts[class_name] = count
        for key, n in dims.items():
            dimension_totals[key] = dimension_totals.get(key, 0) + n
        corrupt_files.extend(corrupt)

    return {
        "discovered_classes": discovered_classes,
        "per_class_counts": per_class_counts,
        "dimension_totals": dimension_totals,
        "corrupt_files": corrupt_files,
    }


def build_report(scan: dict) -> dict:
    per_class_counts = scan["per_class_counts"]
    total_images = sum(per_class_counts.values())

    counts_ascending = sorted(per_class_counts.items(), key=lambda kv: kv[1])
    min_class_name, min_count = counts_ascending[0]
    max_class_name, max_count = counts_ascending[-1]

    class_list_sorted = [
        {"index": i, "class_name": name}
        for i, name in enumerate(config.PLANTVILLAGE_CLASS_NAMES)
    ]

    return {
        "dataset": "plantvillage_color",
        "total_images": total_images,
        "num_classes": len(per_class_counts),
        "class_counts_ascending": [
            {"class_name": name, "count": count} for name, count in counts_ascending
        ],
        "min_class": {"class_name": min_class_name, "count": min_count},
        "max_class": {"class_name": max_class_name, "count": max_count},
        "imbalance_ratio": round(max_count / min_count, 2),
        "num_healthy_classes": len(config.HEALTHY_CLASS_NAMES),
        "healthy_classes": list(config.HEALTHY_CLASS_NAMES),
        "image_dimension_distribution": dict(
            sorted(scan["dimension_totals"].items(), key=lambda kv: kv[1], reverse=True)
        ),
        "corrupt_files": scan["corrupt_files"],
        "num_corrupt_files": len(scan["corrupt_files"]),
        "class_list_sorted_with_index": class_list_sorted,
    }


def write_json(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# PlantVillage (color) Inventory",
        "",
        f"- Total images: **{report['total_images']}**",
        f"- Classes: **{report['num_classes']}**",
        f"- Healthy classes: **{report['num_healthy_classes']}**",
        f"- Min class: `{report['min_class']['class_name']}` "
        f"({report['min_class']['count']} images)",
        f"- Max class: `{report['max_class']['class_name']}` "
        f"({report['max_class']['count']} images)",
        f"- Imbalance ratio (max/min): **{report['imbalance_ratio']}**",
        f"- Corrupt/unreadable files: **{report['num_corrupt_files']}**",
        "",
        "## Per-class counts (ascending)",
        "",
        "| Class | Count |",
        "|---|---|",
    ]
    for entry in report["class_counts_ascending"]:
        lines.append(f"| {entry['class_name']} | {entry['count']} |")

    lines += ["", "## Image dimension distribution", "", "| Dimensions | Count |", "|---|---|"]
    for dims, count in report["image_dimension_distribution"].items():
        lines.append(f"| {dims} | {count} |")

    lines += ["", "## Class list (sorted, canonical index order)", "", "| Index | Class |", "|---|---|"]
    for entry in report["class_list_sorted_with_index"]:
        lines.append(f"| {entry['index']} | {entry['class_name']} |")

    if report["corrupt_files"]:
        lines += ["", "## Corrupt/unreadable files", ""]
        for record in report["corrupt_files"]:
            lines.append(f"- `{record['path']}`: {record['error']}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    scan = scan_plantvillage()
    report = build_report(scan)
    write_json(report, config.ARTIFACTS_DIR / "inventory.json")
    write_markdown(report, config.ARTIFACTS_DIR / "inventory.md")
    print(
        f"[inventory] {report['total_images']} images, "
        f"{report['num_classes']} classes, "
        f"imbalance ratio {report['imbalance_ratio']}, "
        f"{report['num_corrupt_files']} corrupt files. "
        f"Wrote {config.ARTIFACTS_DIR / 'inventory.json'} and "
        f"{config.ARTIFACTS_DIR / 'inventory.md'}."
    )


if __name__ == "__main__":
    main()
