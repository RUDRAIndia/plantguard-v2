"""Tests for android/app/src/main/assets/disease_info.json: the Android app
reads this file directly (DiseaseInfoRepository.kt), so it's validated here
in Python against the same class-name source of truth the rest of the
project uses (src/config.py:PLANTVILLAGE_CLASS_NAMES) -- no fuzzy matching,
fail loudly on any mismatch (CLAUDE.md rule 1). Also guards CLAUDE.md rule 8
(no agrochemical dosages/concentrations/schedules) with a regression check:
no entry may contain a number immediately followed by a dosage-style unit,
so a dosage can't creep back in unnoticed later.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config

DISEASE_INFO_PATH = (
    config.REPO_ROOT / "android" / "app" / "src" / "main" / "assets" / "disease_info.json"
)

REQUIRED_FIELDS = ("display_name", "symptoms", "management", "citation", "status")

# A number immediately followed (optionally after whitespace) by a
# dosage/concentration-style unit -- g/L, ml, mg/L, kg/ha, l/ha, ppm, mg, g,
# kg, or a percent sign/word. Deliberately broad: content is written to
# avoid ANY number+unit construction at all (even non-chemical ones, like
# leaf sizes or sun-hours), so there's no ambiguity to adjudicate here.
DOSAGE_PATTERN = re.compile(
    r"\d+(\.\d+)?\s*(%|percent|g/l|mg/l|kg/ha|l/ha|ppm|ml|mg|g|kg)\b",
    re.IGNORECASE,
)


def _load_disease_info() -> dict:
    return json.loads(DISEASE_INFO_PATH.read_text(encoding="utf-8"))


def test_asset_file_exists():
    assert DISEASE_INFO_PATH.is_file(), f"Missing {DISEASE_INFO_PATH}"


def test_all_38_class_keys_present_and_match_config_exactly():
    data = _load_disease_info()
    json_keys = set(data["entries"].keys())
    config_keys = set(config.PLANTVILLAGE_CLASS_NAMES)

    missing = config_keys - json_keys
    extra = json_keys - config_keys
    assert not missing, f"disease_info.json is missing entries for: {sorted(missing)}"
    assert not extra, (
        f"disease_info.json has entries for classes not in "
        f"config.PLANTVILLAGE_CLASS_NAMES (typo?): {sorted(extra)}"
    )
    assert len(json_keys) == 38


def test_every_entry_has_all_required_non_empty_fields():
    data = _load_disease_info()
    for class_name, entry in data["entries"].items():
        for field in REQUIRED_FIELDS:
            assert field in entry, f"{class_name} is missing field '{field}'"
            assert isinstance(entry[field], str) and entry[field].strip(), (
                f"{class_name}.{field} must be a non-empty string"
            )


def test_no_entry_contains_a_dosage_style_number_and_unit():
    data = _load_disease_info()
    violations = []
    for class_name, entry in data["entries"].items():
        for field in REQUIRED_FIELDS:
            match = DOSAGE_PATTERN.search(entry[field])
            if match:
                violations.append(f"{class_name}.{field}: {match.group(0)!r}")
    assert not violations, (
        "CLAUDE.md rule 8 violation -- found a number immediately followed by "
        f"a dosage/concentration-style unit: {violations}"
    )


def test_every_entry_is_verified():
    # Every shipped entry must be fully sourced -- an unsourced claim gets
    # dropped at write time (see disease_info.json's _readme), never kept
    # with a "pending" flag, since this text is shown directly to farmers
    # and a pending flag isn't visible to them. This test is the guard
    # against a future edit re-introducing that pattern.
    data = _load_disease_info()
    for class_name, entry in data["entries"].items():
        assert entry["status"] == "verified", (
            f"{class_name} has status {entry['status']!r}, not 'verified' -- "
            "every claim in a shipped entry must be sourced; drop the "
            "unsourced part instead of flagging it as pending."
        )
