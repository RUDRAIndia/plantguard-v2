"""Builds an explicit, hand-written PlantDoc-to-PlantVillage class mapping
and reports how many PlantDoc images are usable for external evaluation.

Per CLAUDE.md rule 1, this mapping is a literal dictionary written by a
human — there is no fuzzy/substring matching anywhere in this file. Where a
PlantDoc class name isn't an exact match for a PlantVillage disease, the
mapping was chosen by elimination against PlantVillage's known class list
for that species (documented per-entry in MAPPING_NOTES), never guessed
algorithmically. Produces artifacts/plantdoc_mapping.json.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config  # noqa: E402

# PlantDoc's "Cropped-PlantDoc" classification set has 28 class folders
# across its train/ split (27 in test/ — it lacks "Tomato two spotted
# spider mites leaf"). Per the PlantDoc paper, the 10 classes with no
# disease qualifier in their name (e.g. "Apple leaf", "grape leaf") are its
# healthy classes — that convention, not string similarity, is what backs
# every "generic name -> ___healthy" mapping below.
PLANTDOC_TO_PLANTVILLAGE = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato leaf": "Tomato___healthy",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape leaf black rot": "Grape___Black_rot",
    "grape leaf": "Grape___healthy",
}

# One-line rationale per entry, so a reader can audit the judgment calls
# without re-deriving them. "exact" = the PlantDoc name names the same
# disease as the PlantVillage class. "healthy-by-convention" = PlantDoc's
# generic species-only name, mapped per the paper's documented healthy-class
# convention. "only-option" = PlantVillage has exactly one disease class for
# that species, so the PlantDoc disease name maps to it by elimination even
# though the wording differs.
MAPPING_NOTES = {
    "Apple Scab Leaf": "exact",
    "Apple leaf": "healthy-by-convention",
    "Apple rust leaf": "exact (rust == Cedar_apple_rust, Apple's only rust class)",
    "Bell_pepper leaf spot": "only-option (Bacterial_spot is bell pepper's only disease class)",
    "Bell_pepper leaf": "healthy-by-convention",
    "Blueberry leaf": "healthy-by-convention (Blueberry has no disease classes at all)",
    "Cherry leaf": "healthy-by-convention",
    "Corn Gray leaf spot": "exact (Gray leaf spot == Cercospora leaf spot Gray leaf spot)",
    "Corn leaf blight": "only-option (Northern_Leaf_Blight is corn's only blight class)",
    "Corn rust leaf": "exact",
    "Peach leaf": "healthy-by-convention",
    "Potato leaf early blight": "exact",
    "Potato leaf late blight": "exact",
    "Raspberry leaf": "healthy-by-convention (Raspberry has no disease classes at all)",
    "Soyabean leaf": "healthy-by-convention (Soybean has no disease classes at all; spelling differs)",
    "Squash Powdery mildew leaf": "exact (Squash has no healthy class at all)",
    "Strawberry leaf": "healthy-by-convention (not Leaf_scorch)",
    "Tomato Early blight leaf": "exact",
    "Tomato Septoria leaf spot": "exact",
    "Tomato leaf bacterial spot": "exact",
    "Tomato leaf late blight": "exact",
    "Tomato leaf mosaic virus": "exact",
    "Tomato leaf yellow virus": "exact (yellow virus == Tomato_Yellow_Leaf_Curl_Virus)",
    "Tomato leaf": "healthy-by-convention",
    "Tomato mold leaf": "only-option (Leaf_Mold is tomato's only mold-related class)",
    "Tomato two spotted spider mites leaf": "exact",
    "grape leaf black rot": "exact",
    "grape leaf": "healthy-by-convention",
}

assert set(PLANTDOC_TO_PLANTVILLAGE) == set(MAPPING_NOTES), (
    "PLANTDOC_TO_PLANTVILLAGE and MAPPING_NOTES must have identical keys."
)


def validate_mapping() -> None:
    """Every non-null mapped value must be a literal, exact member of
    config.PLANTVILLAGE_CLASS_NAMES — this is an integrity check on a
    hand-written dict, not fuzzy matching.
    """
    valid_names = set(config.PLANTVILLAGE_CLASS_NAMES)
    for plantdoc_name, pv_name in PLANTDOC_TO_PLANTVILLAGE.items():
        if pv_name is not None and pv_name not in valid_names:
            raise AssertionError(
                f"PLANTDOC_TO_PLANTVILLAGE['{plantdoc_name}'] = '{pv_name}' "
                "is not an exact match for any name in "
                "config.PLANTVILLAGE_CLASS_NAMES. Fix the typo in the "
                "hand-written mapping — do not fuzzy-match it."
            )


def count_plantdoc_images() -> dict[str, int]:
    """Counts images per PlantDoc class across both its train/ and test/
    splits combined (we use all of PlantDoc as external eval — its own
    train/test split is irrelevant since we never train on it).
    """
    if not config.PLANTDOC_TRAIN_DIR.is_dir() or not config.PLANTDOC_TEST_DIR.is_dir():
        raise FileNotFoundError(
            f"{config.PLANTDOC_TRAIN_DIR} and/or {config.PLANTDOC_TEST_DIR} "
            "do not exist. Run src/data/download.py first."
        )

    counts: dict[str, int] = {}
    for split_dir in (config.PLANTDOC_TRAIN_DIR, config.PLANTDOC_TEST_DIR):
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            n = sum(
                1
                for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS
            )
            counts[class_dir.name] = counts.get(class_dir.name, 0) + n
    return counts


def build_report(image_counts: dict[str, int]) -> dict:
    entries = []
    usable_images = 0
    unusable_images = 0

    for plantdoc_class in sorted(image_counts):
        pv_name = PLANTDOC_TO_PLANTVILLAGE.get(plantdoc_class)
        note = MAPPING_NOTES.get(plantdoc_class, "unmapped: no dictionary entry")
        count = image_counts[plantdoc_class]
        pv_index = config.PLANTVILLAGE_CLASS_NAMES.index(pv_name) if pv_name else None

        entries.append(
            {
                "plantdoc_class": plantdoc_class,
                "plantvillage_class": pv_name,
                "plantvillage_index": pv_index,
                "note": note,
                "image_count": count,
            }
        )
        if pv_name is not None:
            usable_images += count
        else:
            unusable_images += count

    return {
        "num_plantdoc_classes": len(entries),
        "num_mapped_classes": sum(1 for e in entries if e["plantvillage_class"] is not None),
        "num_unmapped_classes": sum(1 for e in entries if e["plantvillage_class"] is None),
        "usable_images_for_external_eval": usable_images,
        "unusable_images": unusable_images,
        "mapping": entries,
    }


def main() -> None:
    validate_mapping()
    image_counts = count_plantdoc_images()
    report = build_report(image_counts)

    out_path = config.ARTIFACTS_DIR / "plantdoc_mapping.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"[mapping] {report['num_plantdoc_classes']} PlantDoc classes, "
        f"{report['num_mapped_classes']} mapped, "
        f"{report['num_unmapped_classes']} unmapped. "
        f"{report['usable_images_for_external_eval']} images usable for "
        f"external evaluation ({report['unusable_images']} unusable). "
        f"Wrote {out_path}."
    )


if __name__ == "__main__":
    main()
