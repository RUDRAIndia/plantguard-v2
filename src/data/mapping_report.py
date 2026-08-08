"""Builds artifacts/plantdoc_mapping.json from src/data/mapping.py's
hand-written class mapping and the real, on-disk PlantDoc image counts.

Every figure in the resulting JSON is self-describing: confidence tiers are
broken out (never collapsed into a bare "unmapped" count), coverage of the
38 PlantVillage classes is reported explicitly (mapping every PlantDoc class
is not the same thing as every PlantVillage class having external eval
data), and classes too small to support a per-class metric are flagged by
name rather than left for a reader to notice later (CLAUDE.md rule 5).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config  # noqa: E402
from src.data import mapping  # noqa: E402

# What was checked to back the "bare species name -> healthy" convention
# used throughout mapping.PLANTDOC_TO_PLANTVILLAGE, and what it said — see
# mapping.py's module docstring for the full citations. Carried into the
# JSON report itself so the convention isn't a silently-kept assumption.
HEALTHY_CONVENTION_VERIFICATION = {
    "status": "verified",
    "checked": [
        "PlantDoc paper: Singh et al., 'PlantDoc: A Dataset for Visual Plant "
        "Disease Detection', arXiv:1911.10317 (CoDS-COMAD 2020)",
        "pratikkayal/PlantDoc-Dataset GitHub repository (README + full file "
        "tree, fetched via the GitHub API)",
    ],
    "finding": (
        "The paper states the dataset has '27 classes (17-10, "
        "disease-healthy)' — 17 disease classes, 10 healthy classes — and "
        "its Figure 1 caption explicitly labels one example 'Blueberry "
        "Healthy' (Blueberry has no disease class in this dataset at all, "
        "and its only class folder is the bare-species-named 'Blueberry "
        "leaf'). Cross-checked against the live repository: exactly 10 of "
        "its 28 class folders carry no disease/pest qualifier in their "
        "name, matching the paper's '10 healthy classes' exactly."
    ),
    "caveat": (
        "The paper does not publish a full per-class table, so this is a "
        "count-and-example cross-check rather than an explicit blanket "
        "naming rule stated in the paper's text."
    ),
}


def _require_every_discovered_class_is_mapped(image_counts: dict[str, int]) -> None:
    """mapping.PLANTDOC_TO_PLANTVILLAGE is a hand-written, supposedly
    exhaustive dictionary — if the actual downloaded dataset contains a
    class directory this dict doesn't know about, that is a real gap that
    must stop the run (CLAUDE.md rule 1: fail loudly), not get quietly
    folded into an "unmapped images" bucket that looks like an intentional,
    already-accounted-for category.
    """
    unmapped = sorted(set(image_counts) - set(mapping.PLANTDOC_TO_PLANTVILLAGE))
    if unmapped:
        raise RuntimeError(
            f"Found {len(unmapped)} PlantDoc class director"
            f"{'y' if len(unmapped) == 1 else 'ies'} with no entry in "
            f"mapping.PLANTDOC_TO_PLANTVILLAGE: {unmapped}. This "
            "hand-written mapping must cover every class actually present "
            "in the downloaded dataset — add an entry (with a confidence "
            "tier and rationale) or confirm this is a stray/renamed folder "
            "before proceeding."
        )


def build_coverage(mapped_pv_classes: set) -> dict:
    """How many of the 38 PlantVillage classes have at least one PlantDoc
    image mapped to them — external-eval coverage is not the same thing as
    "the mapping has no unmapped PlantDoc classes"; a PlantVillage class can
    have zero PlantDoc representation even when every PlantDoc class is
    successfully mapped.
    """
    all_pv_classes = set(config.PLANTVILLAGE_CLASS_NAMES)
    covered = sorted(mapped_pv_classes & all_pv_classes)
    uncovered = sorted(all_pv_classes - mapped_pv_classes)
    return {
        "num_plantvillage_classes": len(all_pv_classes),
        "num_covered": len(covered),
        "num_uncovered": len(uncovered),
        "coverage_pct": round(100 * len(covered) / len(all_pv_classes), 2),
        "uncovered_plantvillage_classes": uncovered,
    }


def build_report(image_counts: dict[str, int]) -> dict:
    _require_every_discovered_class_is_mapped(image_counts)

    min_images = config.MIN_IMAGES_FOR_PER_CLASS_METRICS
    entries = []
    breakdown = {tier: {"num_classes": 0, "num_images": 0} for tier in mapping.CONFIDENCE_TIERS}
    below_min_images = []
    mapped_pv_classes = set()

    for plantdoc_class in sorted(image_counts):
        pv_name = mapping.PLANTDOC_TO_PLANTVILLAGE[plantdoc_class]
        confidence, rationale = mapping.MAPPING_METADATA[plantdoc_class]
        count = image_counts[plantdoc_class]
        pv_index = config.PLANTVILLAGE_CLASS_NAMES.index(pv_name)
        below_min = count < min_images

        entries.append(
            {
                "plantdoc_class": plantdoc_class,
                "plantvillage_class": pv_name,
                "plantvillage_index": pv_index,
                "confidence": confidence,
                "rationale": rationale,
                "image_count": count,
                "below_min_images_for_per_class_metrics": below_min,
            }
        )

        breakdown[confidence]["num_classes"] += 1
        breakdown[confidence]["num_images"] += count
        mapped_pv_classes.add(pv_name)
        if below_min:
            below_min_images.append(
                {"plantdoc_class": plantdoc_class, "plantvillage_class": pv_name, "image_count": count}
            )

    total_classes = len(entries)
    total_images = sum(e["image_count"] for e in entries)
    breakdown_image_sum = sum(tier["num_images"] for tier in breakdown.values())
    assert breakdown_image_sum == total_images, (
        f"Per-class image counts sum to {total_images}, but the confidence-"
        f"tier breakdown sums to {breakdown_image_sum} — these must match "
        "exactly (CLAUDE.md rule 1: fail loudly on a bookkeeping mismatch "
        "rather than report an inconsistent total)."
    )

    return {
        "num_plantdoc_classes": total_classes,
        "mapping_confidence_breakdown": {
            **breakdown,
            "total_classes": total_classes,
            "total_images": total_images,
        },
        "min_images_for_per_class_metrics": min_images,
        "classes_below_min_images_for_per_class_metrics": below_min_images,
        "healthy_convention_verification": HEALTHY_CONVENTION_VERIFICATION,
        "plantvillage_coverage": build_coverage(mapped_pv_classes),
        "mapping": entries,
    }


def main() -> None:
    mapping.validate_mapping()
    image_counts = mapping.count_plantdoc_images()
    report = build_report(image_counts)

    out_path = config.ARTIFACTS_DIR / "plantdoc_mapping.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    breakdown = report["mapping_confidence_breakdown"]
    coverage = report["plantvillage_coverage"]
    print(
        f"[mapping_report] {report['num_plantdoc_classes']} PlantDoc classes "
        f"({breakdown['exact']['num_classes']} exact, "
        f"{breakdown['convention']['num_classes']} convention, "
        f"{breakdown['forced']['num_classes']} forced), "
        f"{breakdown['total_images']} images total. "
        f"{len(report['classes_below_min_images_for_per_class_metrics'])} "
        f"class(es) below the {report['min_images_for_per_class_metrics']}-image "
        "per-class-metric floor. "
        f"PlantVillage coverage: {coverage['num_covered']}/"
        f"{coverage['num_plantvillage_classes']} classes "
        f"({coverage['coverage_pct']}%). "
        f"Wrote {out_path}."
    )


if __name__ == "__main__":
    main()
