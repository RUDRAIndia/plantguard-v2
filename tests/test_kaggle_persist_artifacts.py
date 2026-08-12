"""Tests for src/models/kaggle_persist_artifacts.py against the same fake
KaggleApi tests/test_kaggle_persist.py uses (tests/kaggle_api_fake.py) — a
real account/kaggle_secrets is only ever available on an actual Kaggle
notebook instance (see that module's docstring).
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests/ itself, for kaggle_api_fake

import pytest

from src import config
from src.models import kaggle_persist, kaggle_persist_artifacts, kaggle_retry
from kaggle_api_fake import FakeKaggleApi
from kaggle_api_fake import http_error as _http_error


@pytest.fixture
def fake_api(monkeypatch, tmp_path):
    api = FakeKaggleApi()
    monkeypatch.setattr(kaggle_persist, "_authenticated_api", lambda: api)
    monkeypatch.setattr(config, "KAGGLE_PERSIST_STAGING_DIR", tmp_path / "staging")
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(config, "EVAL_FIGURES_DIR", tmp_path / "artifacts" / "figures")
    return api


def _write_full_local_artifacts(skip: set = frozenset()) -> None:
    """Writes every config.KAGGLE_PERSIST_ARTIFACT_FILENAMES file plus a
    figures/ tree (including a gradcam/ subdirectory, exercising the "/" ->
    "__" flattening) under the currently-monkeypatched config.ARTIFACTS_DIR.
    `skip` names files to deliberately leave out, for incompleteness tests.
    """
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    for name in config.KAGGLE_PERSIST_ARTIFACT_FILENAMES:
        if name in skip:
            continue
        (config.ARTIFACTS_DIR / name).write_text(f"content of {name}", encoding="utf-8")

    if "figures" not in skip:
        config.EVAL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        (config.EVAL_FIGURES_DIR / "confusion_matrix.png").write_bytes(b"confusion-matrix-bytes")
        gradcam_dir = config.EVAL_FIGURES_DIR / "gradcam"
        gradcam_dir.mkdir(parents=True, exist_ok=True)
        (gradcam_dir / "plantvillage_test_correct_00.png").write_bytes(b"gradcam-bytes")


def test_push_data_artifacts_creates_dataset_on_first_use(fake_api):
    _write_full_local_artifacts()

    kaggle_persist_artifacts.push_data_artifacts()

    assert fake_api.create_calls == 1
    assert fake_api.version_calls == 0
    assert fake_api.files["artifacts__results.json"] == b"content of results.json"
    assert fake_api.files["artifacts__figures__confusion_matrix.png"] == b"confusion-matrix-bytes"
    assert fake_api.files["artifacts__figures__gradcam__plantvillage_test_correct_00.png"] == b"gradcam-bytes"


def test_push_data_artifacts_uses_version_on_subsequent_pushes(fake_api):
    _write_full_local_artifacts()
    kaggle_persist_artifacts.push_data_artifacts()

    (config.ARTIFACTS_DIR / "results.json").write_text("content of results.json v2", encoding="utf-8")
    kaggle_persist_artifacts.push_data_artifacts()

    assert fake_api.create_calls == 1
    assert fake_api.version_calls == 1
    assert fake_api.files["artifacts__results.json"] == b"content of results.json v2"


def test_push_data_artifacts_refuses_when_a_flat_file_is_missing(fake_api):
    _write_full_local_artifacts(skip={"plantdoc_mapping.json"})

    with pytest.raises(RuntimeError, match="plantdoc_mapping.json"):
        kaggle_persist_artifacts.push_data_artifacts()
    assert fake_api.create_calls == 0


def test_push_data_artifacts_refuses_when_figures_dir_is_missing(fake_api):
    _write_full_local_artifacts(skip={"figures"})

    with pytest.raises(RuntimeError, match="figures"):
        kaggle_persist_artifacts.push_data_artifacts()
    assert fake_api.create_calls == 0


def test_push_data_artifacts_and_push_checkpoint_never_clobber_each_other(fake_api):
    checkpoint_dir = config.KAGGLE_PERSIST_STAGING_DIR.parent / "ckpt"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "phase2.weights.h5").write_text("model-weights", encoding="utf-8")
    kaggle_persist.push_checkpoint("MobileNetV2", checkpoint_dir)

    _write_full_local_artifacts()
    kaggle_persist_artifacts.push_data_artifacts()

    assert fake_api.files["MobileNetV2__phase2.weights.h5"] == b"model-weights"
    assert fake_api.files["artifacts__results.json"] == b"content of results.json"

    # Pushing a checkpoint again must not remove the already-pushed
    # artifacts/ bundle, and vice versa (both share one flat namespace).
    (checkpoint_dir / "phase2.weights.h5").write_text("model-weights-v2", encoding="utf-8")
    kaggle_persist.push_checkpoint("MobileNetV2", checkpoint_dir)
    assert fake_api.files["artifacts__results.json"] == b"content of results.json"

    (config.ARTIFACTS_DIR / "results.json").write_text("content of results.json v2", encoding="utf-8")
    kaggle_persist_artifacts.push_data_artifacts()
    assert fake_api.files["MobileNetV2__phase2.weights.h5"] == b"model-weights-v2"


def test_push_data_artifacts_raises_a_safe_message_when_download_retries_exhaust(fake_api, monkeypatch):
    monkeypatch.setattr(kaggle_retry.time, "sleep", lambda seconds: None)

    checkpoint_dir = config.KAGGLE_PERSIST_STAGING_DIR.parent / "ckpt"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "phase1.weights.h5").write_text("already-safe", encoding="utf-8")
    kaggle_persist.push_checkpoint("MobileNetV2", checkpoint_dir)

    fake_api.raise_on_download = [_http_error(404)] * kaggle_retry.MAX_ATTEMPTS
    _write_full_local_artifacts()

    with pytest.raises(RuntimeError) as exc_info:
        kaggle_persist_artifacts.push_data_artifacts()

    message = str(exc_info.value)
    assert "uploaded NOTHING" in message
    assert "untouched and safe" in message
    assert fake_api.version_calls == 0
    assert fake_api.files["MobileNetV2__phase1.weights.h5"] == b"already-safe"
    assert "artifacts__results.json" not in fake_api.files


def test_push_data_artifacts_retries_transient_404_then_succeeds(fake_api, monkeypatch):
    monkeypatch.setattr(kaggle_retry.time, "sleep", lambda seconds: None)

    checkpoint_dir = config.KAGGLE_PERSIST_STAGING_DIR.parent / "ckpt"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "phase1.weights.h5").write_text("w", encoding="utf-8")
    kaggle_persist.push_checkpoint("MobileNetV2", checkpoint_dir)

    fake_api.raise_on_download = [_http_error(404), _http_error(404)]
    _write_full_local_artifacts()
    kaggle_persist_artifacts.push_data_artifacts()

    assert fake_api.files["artifacts__results.json"] == b"content of results.json"
    assert fake_api.raise_on_download == []


def test_restore_data_artifacts_returns_false_when_locally_complete(fake_api):
    _write_full_local_artifacts()

    restored = kaggle_persist_artifacts.restore_data_artifacts()

    assert restored is False
    assert fake_api.download_call_count == 0


def test_restore_data_artifacts_returns_false_when_no_dataset_exists_yet(fake_api):
    restored = kaggle_persist_artifacts.restore_data_artifacts()
    assert restored is False


def test_restore_data_artifacts_round_trips_flat_files_and_nested_figures(fake_api, tmp_path, monkeypatch):
    _write_full_local_artifacts()
    kaggle_persist_artifacts.push_data_artifacts()

    # A "fresh session" — different, empty ARTIFACTS_DIR/EVAL_FIGURES_DIR.
    fresh_artifacts_dir = tmp_path / "fresh_artifacts"
    monkeypatch.setattr(config, "ARTIFACTS_DIR", fresh_artifacts_dir)
    monkeypatch.setattr(config, "EVAL_FIGURES_DIR", fresh_artifacts_dir / "figures")

    restored = kaggle_persist_artifacts.restore_data_artifacts()

    assert restored is True
    assert (fresh_artifacts_dir / "results.json").read_text(encoding="utf-8") == "content of results.json"
    assert (fresh_artifacts_dir / "splits.json").read_text(encoding="utf-8") == "content of splits.json"
    assert (fresh_artifacts_dir / "figures" / "confusion_matrix.png").read_bytes() == b"confusion-matrix-bytes"
    assert (
        fresh_artifacts_dir / "figures" / "gradcam" / "plantvillage_test_correct_00.png"
    ).read_bytes() == b"gradcam-bytes"


def test_restore_data_artifacts_never_overwrites_a_locally_present_file(fake_api):
    _write_full_local_artifacts()
    kaggle_persist_artifacts.push_data_artifacts()

    # A fresh-looking but partially-populated session: results.json already
    # exists locally (with different, "more recent" content) but everything
    # else is missing.
    for name in config.KAGGLE_PERSIST_ARTIFACT_FILENAMES:
        (config.ARTIFACTS_DIR / name).unlink()
    shutil.rmtree(config.EVAL_FIGURES_DIR)
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.ARTIFACTS_DIR / "results.json").write_text("local in-session results", encoding="utf-8")

    restored = kaggle_persist_artifacts.restore_data_artifacts()

    assert restored is True
    assert (config.ARTIFACTS_DIR / "results.json").read_text(encoding="utf-8") == "local in-session results"
    assert (config.ARTIFACTS_DIR / "splits.json").read_text(encoding="utf-8") == "content of splits.json"


def test_restore_data_artifacts_raises_a_safe_message_when_download_retries_exhaust(fake_api, monkeypatch):
    monkeypatch.setattr(kaggle_retry.time, "sleep", lambda seconds: None)
    fake_api.exists = True
    fake_api.files["artifacts__results.json"] = b"persisted"
    fake_api.raise_on_download = [_http_error(503)] * kaggle_retry.MAX_ATTEMPTS

    with pytest.raises(RuntimeError, match="Nothing was restored"):
        kaggle_persist_artifacts.restore_data_artifacts()
