"""Runs the project's headline external evaluation: PlantDoc (real field
images, never trained on, never used for model selection or threshold
tuning) against the model src/evaluate/model_selection.py selected. Uses
src/data/mapping.py's hand-written PlantDoc-to-PlantVillage class mapping,
regenerating artifacts/plantdoc_mapping.json fresh (src/data/mapping_report.py)
so every count here is a real computation over whatever PlantDoc copy is
actually on disk (CLAUDE.md rule 5), never a stale cached JSON.

PlantDoc only covers 28 of PlantVillage's 38 classes (73.68%) — accuracy and
macro-F1 are reported over those 28 mapped classes only, broken down by the
mapping's three confidence tiers (exact/convention/forced), with the 10
uncovered classes (including Orange___Haunglongbing_(Citrus_greening), the
largest PlantVillage training class) listed by name. Any mapped class whose
real PlantDoc image count falls below config.MIN_IMAGES_FOR_PER_CLASS_METRICS
is excluded from the per-class precision/recall table and named explicitly,
per CLAUDE.md rule 5's "say which was excluded."

Expect this number to land far below the PlantVillage test-set figure —
report it plainly, without softening (CLAUDE.md's explicit instruction);
this gap is the project's real contribution, not a defect to explain away.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config  # noqa: E402
from src.data import mapping, mapping_report  # noqa: E402
from src.evaluate import inference  # noqa: E402


def _build_eval_set() -> tuple:
    """Returns (paths, pv_labels, plantdoc_class_of_each_image) — every real
    image under config.PLANTDOC_TRAIN_DIR/TEST_DIR, labeled with its mapped
    PlantVillage class index (mapping.PLANTDOC_TO_PLANTVILLAGE), plus the
    originating PlantDoc class name so confidence-tier breakdowns can be
    computed per image.
    """
    mapping.validate_mapping()
    paths, pv_labels, plantdoc_classes = [], [], []
    for split_dir in (config.PLANTDOC_TRAIN_DIR, config.PLANTDOC_TEST_DIR):
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"{split_dir} does not exist. Run src/data/download.py first."
            )
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            pv_name = mapping.PLANTDOC_TO_PLANTVILLAGE[class_dir.name]
            pv_index = config.PLANTVILLAGE_CLASS_NAMES.index(pv_name)
            for image_path in sorted(
                p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS
            ):
                paths.append(str(image_path))
                pv_labels.append(pv_index)
                plantdoc_classes.append(class_dir.name)
    if not paths:
        raise RuntimeError(f"No PlantDoc images found under {config.PLANTDOC_DIR}.")
    return paths, pv_labels, plantdoc_classes


def _excluded_classes(mapping_report_data: dict) -> set:
    return {
        entry["plantvillage_class"]
        for entry in mapping_report_data["classes_below_min_images_for_per_class_metrics"]
    }


def _tier_of_each_image(plantdoc_classes: list) -> list:
    return [mapping.MAPPING_METADATA[name][0] for name in plantdoc_classes]


def _subset_metrics(y_true: np.ndarray, y_pred: np.ndarray, covered_labels: list) -> dict:
    accuracy = float(np.mean(y_pred == y_true)) if len(y_true) else None
    macro_f1 = (
        float(f1_score(y_true, y_pred, labels=covered_labels, average="macro", zero_division=0))
        if len(y_true)
        else None
    )
    return {"num_images": int(len(y_true)), "accuracy": accuracy, "macro_f1": macro_f1}


def _per_class_table(
    y_true: np.ndarray, y_pred: np.ndarray, covered_labels: list, class_names: tuple, excluded_classes: set
) -> tuple:
    """Per-class precision/recall/f1 for the covered 28 classes, excluding
    any class below config.MIN_IMAGES_FOR_PER_CLASS_METRICS real PlantDoc
    images. Returns (table, excluded_entries) — excluded_entries names
    exactly which classes were dropped and why, never silently.
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=covered_labels, average=None, zero_division=0
    )
    table, excluded_entries = [], []
    for i, label in enumerate(covered_labels):
        class_name = class_names[label]
        entry = {
            "class_name": class_name,
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        if class_name in excluded_classes:
            excluded_entries.append(entry)
        else:
            table.append(entry)
    return table, excluded_entries


def evaluate_plantdoc(model, model_name: str) -> tuple:
    """Returns (results_dict, y_true, y_pred, paths) — the predictions and
    paths are for src/evaluate/gradcam.py's PlantDoc-failure sampling.
    """
    mapping_report_data = mapping_report.build_report(mapping.count_plantdoc_images())
    report_path = config.ARTIFACTS_DIR / "plantdoc_mapping.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(mapping_report_data, indent=2), encoding="utf-8")

    paths, y_true_list, plantdoc_classes = _build_eval_set()
    y_true = np.array(y_true_list)
    dataset = inference.build_eval_pipeline(paths, y_true_list, model_name)
    y_true_check, y_prob = inference.predict_dataset(model, dataset)
    assert np.array_equal(y_true, y_true_check), "Prediction order drifted from the built eval set."
    y_pred = np.argmax(y_prob, axis=1)

    covered_pv_names = sorted(set(mapping.PLANTDOC_TO_PLANTVILLAGE.values()))
    covered_labels = sorted(config.PLANTVILLAGE_CLASS_NAMES.index(n) for n in covered_pv_names)
    uncovered = mapping_report_data["plantvillage_coverage"]["uncovered_plantvillage_classes"]
    excluded_classes = _excluded_classes(mapping_report_data)

    overall = _subset_metrics(y_true, y_pred, covered_labels)
    per_class_table, excluded_entries = _per_class_table(
        y_true, y_pred, covered_labels, config.PLANTVILLAGE_CLASS_NAMES, excluded_classes
    )

    tiers = _tier_of_each_image(plantdoc_classes)
    tier_breakdown = {}
    for tier in mapping.CONFIDENCE_TIERS:
        mask = np.array([t == tier for t in tiers])
        tier_breakdown[tier] = _subset_metrics(y_true[mask], y_pred[mask], covered_labels)

    results = {
        "num_plantvillage_classes": len(config.PLANTVILLAGE_CLASS_NAMES),
        "num_covered_classes": len(covered_labels),
        "coverage_pct": mapping_report_data["plantvillage_coverage"]["coverage_pct"],
        "uncovered_plantvillage_classes": uncovered,
        "note": (
            "PlantDoc covers 28 of 38 PlantVillage classes (73.68%). "
            f"{len(uncovered)} classes have zero PlantDoc coverage, including "
            "Orange___Haunglongbing_(Citrus_greening), the largest PlantVillage "
            "training class. Accuracy/macro-F1 below are computed over the "
            "covered classes only; performance here is expected to be, and "
            "typically is, substantially lower than the PlantVillage test-set "
            "figure — this gap is reported as-is."
        ),
        "overall": overall,
        "mapping_confidence_tier_breakdown": tier_breakdown,
        "per_class_metrics": per_class_table,
        "excluded_from_per_class_metrics": {
            "min_images_threshold": config.MIN_IMAGES_FOR_PER_CLASS_METRICS,
            "excluded_classes": excluded_entries,
        },
        "mapping_report_path": str(report_path),
    }
    return results, y_true, y_pred, paths
