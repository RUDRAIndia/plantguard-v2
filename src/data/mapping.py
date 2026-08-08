"""Hand-written PlantDoc-to-PlantVillage class mapping, plus the raw
per-class image counts it's built from.

Per CLAUDE.md rule 1, this mapping is a literal dictionary written by a
human — there is no fuzzy/substring matching anywhere in this file. Where a
PlantDoc class name isn't an exact match for a PlantVillage disease, the
mapping was chosen by elimination against PlantVillage's known class list
for that species, never guessed algorithmically. Every entry carries an
explicit `confidence` tier (see CONFIDENCE_TIERS below) so a report built
from this data never presents a judgment call as an unqualified fact.
Report generation (breakdowns, coverage, the JSON artifact) lives in
src/data/mapping_report.py.

Verification of the "bare species name means a healthy leaf" convention
(checked 2026-08-08 — literal citations, not an assumption): the
PlantDoc-Dataset GitHub README does not itself state a naming rule, but the
PlantDoc paper (Singh et al., "PlantDoc: A Dataset for Visual Plant Disease
Detection", arXiv:1911.10317 / CoDS-COMAD 2020) does two things that
corroborate it directly:
  1. It states the dataset has "27 classes (17-10, disease-healthy)" — i.e.
     exactly 17 disease classes and 10 healthy classes.
  2. Its Figure 1 caption explicitly labels one example image "Blueberry
     Healthy" — Blueberry has no disease class in this dataset at all, and
     its only PlantDoc folder is the bare-species-named "Blueberry leaf".
Cross-checked against the actual repository (pratikkayal/PlantDoc-Dataset,
default branch, full recursive tree fetched via the GitHub API on
2026-08-08): exactly 10 of its 28 class folders carry no disease/pest
qualifier in their name (Apple leaf, Bell_pepper leaf, Blueberry leaf,
Cherry leaf, Peach leaf, Raspberry leaf, Soyabean leaf, Strawberry leaf,
Tomato leaf, grape leaf) — this matches the paper's "10 healthy classes"
exactly, and the paper's own example directly identifies one of them
("Blueberry") as healthy. This is treated as VERIFIED (see
HEALTHY_CONVENTION_VERIFICATION in mapping_report.py), not merely assumed —
but note the paper does not provide a full per-class table, so this is a
count-and-example cross-check rather than an explicit blanket naming rule
stated in the text.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config  # noqa: E402

# PlantDoc's "Cropped-PlantDoc" classification set has 28 class folders
# across its train/ split (27 in test/ — it lacks "Tomato two spotted
# spider mites leaf"). See the module docstring for how the healthy-class
# convention behind every "generic name -> ___healthy" mapping below was
# checked, not assumed.
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

# The three confidence tiers a reader can trust at face value — no other
# string is a valid value (see validate_mapping):
#   "exact"      the PlantDoc name names the same disease as the
#                PlantVillage class (wording may differ, e.g.
#                "yellow virus" == "Tomato_Yellow_Leaf_Curl_Virus", but the
#                condition genuinely corresponds).
#   "convention" PlantDoc's generic species-only name, mapped to
#                ___healthy per the verified convention documented in the
#                module docstring.
#   "forced"     PlantVillage has exactly one disease class for that
#                species, so the PlantDoc name maps to it by elimination
#                even though PlantDoc's label is less specific than (or
#                worded differently from) the PlantVillage target — this is
#                the weakest tier and the one most likely to introduce
#                label noise into external-eval numbers.
CONFIDENCE_TIERS = ("exact", "convention", "forced")

# One entry per PLANTDOC_TO_PLANTVILLAGE key: the confidence tier plus a
# one-line rationale, so a reader can audit every judgment call without
# re-deriving it.
MAPPING_METADATA = {
    "Apple Scab Leaf": ("exact", "Scab is Apple's Apple_scab class."),
    "Apple leaf": ("convention", "Bare species name; Apple has other disease classes, so this is the leftover healthy one."),
    "Apple rust leaf": ("exact", "Rust == Cedar_apple_rust, Apple's only rust class."),
    "Bell_pepper leaf spot": ("forced", "Bacterial_spot is bell pepper's only disease class; PlantDoc's generic 'leaf spot' doesn't name the pathogen."),
    "Bell_pepper leaf": ("convention", "Bare species name; bell pepper has a disease class, so this is the leftover healthy one."),
    "Blueberry leaf": ("convention", "Bare species name; Blueberry has no disease classes at all in PlantVillage, so this can only be healthy."),
    "Cherry leaf": ("convention", "Bare species name; Cherry has other disease classes, so this is the leftover healthy one."),
    "Corn Gray leaf spot": ("exact", "Gray leaf spot == Cercospora_leaf_spot Gray_leaf_spot."),
    "Corn leaf blight": ("forced", "Northern_Leaf_Blight is corn's only blight class; PlantDoc's generic 'leaf blight' doesn't specify Northern vs. another blight."),
    "Corn rust leaf": ("exact", "Rust == Common_rust_, corn's only rust class."),
    "Peach leaf": ("convention", "Bare species name; Peach has a disease class, so this is the leftover healthy one."),
    "Potato leaf early blight": ("exact", "Names the same condition as Potato's Early_blight class."),
    "Potato leaf late blight": ("exact", "Names the same condition as Potato's Late_blight class."),
    "Raspberry leaf": ("convention", "Bare species name; Raspberry has no disease classes at all in PlantVillage, so this can only be healthy."),
    "Soyabean leaf": ("convention", "Bare species name (PlantDoc spelling); Soybean has no disease classes at all in PlantVillage, so this can only be healthy."),
    "Squash Powdery mildew leaf": ("exact", "Powdery mildew is named directly; Squash has no healthy class at all in PlantVillage."),
    "Strawberry leaf": ("convention", "Bare species name; Strawberry has a disease class (Leaf_scorch), so this is the leftover healthy one, not Leaf_scorch."),
    "Tomato Early blight leaf": ("exact", "Names the same condition as Tomato's Early_blight class."),
    "Tomato Septoria leaf spot": ("exact", "Names the same condition as Tomato's Septoria_leaf_spot class."),
    "Tomato leaf bacterial spot": ("exact", "Names the same condition as Tomato's Bacterial_spot class."),
    "Tomato leaf late blight": ("exact", "Names the same condition as Tomato's Late_blight class."),
    "Tomato leaf mosaic virus": ("exact", "Names the same condition as Tomato's Tomato_mosaic_virus class."),
    "Tomato leaf yellow virus": ("exact", "Yellow virus == Tomato_Yellow_Leaf_Curl_Virus."),
    "Tomato leaf": ("convention", "Bare species name; Tomato has many disease classes, so this is the leftover healthy one."),
    "Tomato mold leaf": ("forced", "Leaf_Mold is tomato's only mold-related class; PlantDoc's generic 'mold leaf' doesn't confirm it's specifically leaf mold rather than another mold."),
    "Tomato two spotted spider mites leaf": ("exact", "Names the same condition as Tomato's Spider_mites Two-spotted_spider_mite class."),
    "grape leaf black rot": ("exact", "Names the same condition as Grape's Black_rot class."),
    "grape leaf": ("convention", "Bare species name; Grape has other disease classes, so this is the leftover healthy one."),
}

assert set(PLANTDOC_TO_PLANTVILLAGE) == set(MAPPING_METADATA), (
    "PLANTDOC_TO_PLANTVILLAGE and MAPPING_METADATA must have identical keys."
)
for _plantdoc_name, (_confidence, _rationale) in MAPPING_METADATA.items():
    assert _confidence in CONFIDENCE_TIERS, (
        f"MAPPING_METADATA['{_plantdoc_name}'] has confidence '{_confidence}', "
        f"expected exactly one of {CONFIDENCE_TIERS}."
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
