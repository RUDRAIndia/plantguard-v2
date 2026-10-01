"""Reads/writes artifacts/results.json for src/export/to_tflite.py — split
out to keep that module's conversion/orchestration logic under CLAUDE.md
rule 12's ~300-line guideline.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config  # noqa: E402


def load_results() -> dict:
    if not config.RESULTS_JSON_PATH.is_file():
        raise FileNotFoundError(
            f"{config.RESULTS_JSON_PATH} not found. Run src.evaluate.runner first -- "
            "src/export/to_tflite.py reads the selected model, calibration temperature, and "
            "OOD threshold from it, never hardcoding them (CLAUDE.md rule 5)."
        )
    return json.loads(config.RESULTS_JSON_PATH.read_text(encoding="utf-8"))


def ranking_entry(results: dict, model_name: str) -> dict:
    ranking = results.get("model_selection", {}).get("ranking", [])
    for entry in ranking:
        if entry.get("model_name") == model_name:
            return entry
    raise RuntimeError(
        f"No model_selection.ranking entry for '{model_name}' in {config.RESULTS_JSON_PATH} -- "
        "results.json is internally inconsistent (selected model not present in its own ranking)."
    )


def write_export_section(results: dict, export_section: dict) -> None:
    results["export"] = export_section
    config.RESULTS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.RESULTS_JSON_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[to_tflite] Wrote export section to {config.RESULTS_JSON_PATH}")
