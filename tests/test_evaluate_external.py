"""Tests for src/evaluate/external.py's pure numeric helpers: subset
metrics, per-class table construction with min-images exclusion, and
confidence-tier tagging — no PlantDoc download needed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.evaluate import external

CLASS_NAMES = ("Apple___healthy", "Apple___Apple_scab", "Grape___healthy")


def test_subset_metrics_matches_manual_accuracy():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    result = external._subset_metrics(y_true, y_pred, covered_labels=[0, 1, 2])
    assert result["num_images"] == 6
    assert result["accuracy"] == 4 / 6


def test_subset_metrics_handles_empty_subset():
    result = external._subset_metrics(np.array([]), np.array([]), covered_labels=[0, 1, 2])
    assert result["num_images"] == 0
    assert result["accuracy"] is None
    assert result["macro_f1"] is None


def test_per_class_table_excludes_flagged_classes():
    y_true = np.array([0, 0, 0, 1, 1, 2])
    y_pred = np.array([0, 0, 1, 1, 1, 2])
    excluded = {"Apple___Apple_scab"}  # class index 1

    table, excluded_entries = external._per_class_table(y_true, y_pred, [0, 1, 2], CLASS_NAMES, excluded)

    table_names = {row["class_name"] for row in table}
    assert "Apple___Apple_scab" not in table_names
    assert {"Apple___healthy", "Grape___healthy"} <= table_names
    assert len(excluded_entries) == 1
    assert excluded_entries[0]["class_name"] == "Apple___Apple_scab"


def test_per_class_table_includes_everything_when_nothing_excluded():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])
    table, excluded_entries = external._per_class_table(y_true, y_pred, [0, 1, 2], CLASS_NAMES, excluded_classes=set())
    assert len(table) == 3
    assert excluded_entries == []


def test_excluded_classes_reads_plantvillage_class_field():
    mapping_report_data = {
        "classes_below_min_images_for_per_class_metrics": [
            {"plantdoc_class": "Blueberry leaf", "plantvillage_class": "Blueberry___healthy", "image_count": 3}
        ]
    }
    assert external._excluded_classes(mapping_report_data) == {"Blueberry___healthy"}


def test_tier_of_each_image_looks_up_mapping_metadata():
    plantdoc_classes = ["Apple Scab Leaf", "Apple leaf", "grape leaf black rot"]
    tiers = external._tier_of_each_image(plantdoc_classes)
    assert tiers == ["exact", "convention", "exact"]
