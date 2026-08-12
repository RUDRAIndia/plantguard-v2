"""Tests for src/evaluate/runner.py's console-output helpers. runner.run()
itself is a thin, expensive end-to-end orchestration (already exercised
module-by-module via the other tests/test_evaluate_*.py files); this covers
_print_plantdoc_summary (macro-F1 and the mapping-confidence tier breakdown
visible on the console during a real run, not only inside
artifacts/results.json) and _push_artifacts_or_report_loudly (a real Kaggle
run once reported the whole evaluation as failed purely because this push
raised uncaught at the very end, even though results.json was already
complete -- these tests are what would catch a regression of that fix).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import runner


def _fake_plantdoc_results():
    return {
        "num_covered_classes": 28,
        "num_plantvillage_classes": 38,
        "overall": {"num_images": 2598, "accuracy": 0.1928, "macro_f1": 0.15},
        "mapping_confidence_tier_breakdown": {
            "exact": {"num_images": 1500, "accuracy": 0.30, "macro_f1": 0.28},
            "convention": {"num_images": 800, "accuracy": 0.10, "macro_f1": 0.08},
            "forced": {"num_images": 298, "accuracy": 0.05, "macro_f1": 0.04},
        },
    }


def test_print_plantdoc_summary_reports_overall_accuracy_and_macro_f1(capsys):
    runner._print_plantdoc_summary(_fake_plantdoc_results())
    out = capsys.readouterr().out
    assert "accuracy=0.1928" in out
    assert "macro_f1=0.1500" in out
    assert "28/38" in out


def test_print_plantdoc_summary_reports_every_confidence_tier(capsys):
    runner._print_plantdoc_summary(_fake_plantdoc_results())
    out = capsys.readouterr().out
    for tier in ("exact", "convention", "forced"):
        assert tier in out


def test_print_plantdoc_summary_handles_a_tier_with_no_images():
    results = _fake_plantdoc_results()
    results["mapping_confidence_tier_breakdown"]["forced"] = {
        "num_images": 0,
        "accuracy": None,
        "macro_f1": None,
    }
    runner._print_plantdoc_summary(results)  # must not raise on None accuracy/macro_f1


def test_push_artifacts_or_report_loudly_swallows_a_push_failure_and_reports_it(monkeypatch, capsys):
    def _raise():
        raise RuntimeError("simulated Kaggle API failure")

    monkeypatch.setattr(runner.kaggle_persist_artifacts, "push_data_artifacts", _raise)

    runner._push_artifacts_or_report_loudly()  # must NOT raise -- this is the whole point of the fix

    out = capsys.readouterr().out
    assert "ARTIFACTS PERSISTENCE FAILED" in out
    assert "NOT an evaluation failure" in out
    assert "simulated Kaggle API failure" in out


def test_push_artifacts_or_report_loudly_is_quiet_on_success(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(runner.kaggle_persist_artifacts, "push_data_artifacts", lambda: calls.append(1))

    runner._push_artifacts_or_report_loudly()

    assert calls == [1]
    out = capsys.readouterr().out
    assert "ARTIFACTS PERSISTENCE FAILED" not in out
