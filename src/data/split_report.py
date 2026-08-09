"""Builds artifacts/splits.json and artifacts/split_report.md from
src/data/split.py's grouped/stratified allocation, and is the notebook-
facing entrypoint (mirrors src/data/mapping.py vs. mapping_report.py: the
core allocation algorithm stays a pure, testable module; this module owns
I/O and reporting).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config  # noqa: E402
from src.data import split  # noqa: E402


def _per_class_split_counts(splits: dict[str, list[str]]) -> dict[str, dict[str, int]]:
    counts = {cls: {name: 0 for name in split.SPLIT_NAMES} for cls in config.PLANTVILLAGE_CLASS_NAMES}
    for split_name, paths in splits.items():
        for path in paths:
            counts[split.class_of(path)][split_name] += 1
    return counts


def build_manifest(splits: dict[str, list[str]], image_to_group: dict[str, str]) -> dict:
    num_images = len(image_to_group)
    per_class_counts = _per_class_split_counts(splits)

    return {
        "seed": config.SEED,
        "num_images": num_images,
        "num_groups": len(set(image_to_group.values())),
        "target_fractions": {
            "train": config.TRAIN_FRACTION,
            "val": config.VAL_FRACTION,
            "test": config.TEST_FRACTION,
        },
        "achieved_overall_fractions": {
            name: round(len(splits[name]) / num_images, 4) for name in split.SPLIT_NAMES
        },
        "per_class_split_counts": per_class_counts,
        "git_commit_hash": config.get_git_commit_hash(),
    }


def write_splits_json(splits: dict[str, list[str]], manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"manifest": manifest, "splits": splits}, indent=2), encoding="utf-8"
    )


def write_markdown(manifest: dict, path: Path) -> None:
    achieved = manifest["achieved_overall_fractions"]
    target = manifest["target_fractions"]
    lines = [
        "# PlantVillage Train/Val/Test Split Report",
        "",
        f"- Seed: **{manifest['seed']}**",
        f"- Images: **{manifest['num_images']}**",
        f"- Duplicate groups: **{manifest['num_groups']}**",
        f"- Git commit: `{manifest['git_commit_hash']}`",
        f"- Target fractions: train {target['train']:.0%}, val {target['val']:.0%}, test {target['test']:.0%}",
        f"- Achieved overall fractions: train {achieved['train']:.2%}, "
        f"val {achieved['val']:.2%}, test {achieved['test']:.2%}",
        "",
        "## Per-class counts and achieved proportions",
        "",
        "| Class | Train | Val | Test | Total | Train % | Val % | Test % |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for class_name, counts in manifest["per_class_split_counts"].items():
        total = sum(counts.values())
        pct = {name: (counts[name] / total * 100 if total else 0.0) for name in split.SPLIT_NAMES}
        lines.append(
            f"| {class_name} | {counts['train']} | {counts['val']} | {counts['test']} | "
            f"{total} | {pct['train']:.1f}% | {pct['val']:.1f}% | {pct['test']:.1f}% |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    image_to_group = split.load_image_groups()
    splits = split.build_split(image_to_group)
    manifest = build_manifest(splits, image_to_group)

    write_splits_json(splits, manifest, config.ARTIFACTS_DIR / "splits.json")
    write_markdown(manifest, config.ARTIFACTS_DIR / "split_report.md")

    achieved = manifest["achieved_overall_fractions"]
    print(
        f"[split_report] {manifest['num_images']} images, "
        f"{manifest['num_groups']} groups -> train {len(splits['train'])} "
        f"({achieved['train']:.2%}), val {len(splits['val'])} ({achieved['val']:.2%}), "
        f"test {len(splits['test'])} ({achieved['test']:.2%}). "
        f"Wrote {config.ARTIFACTS_DIR / 'splits.json'}, "
        f"{config.ARTIFACTS_DIR / 'split_report.md'}."
    )


if __name__ == "__main__":
    main()
