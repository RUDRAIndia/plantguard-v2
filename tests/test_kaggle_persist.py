"""Tests for src/models/kaggle_persist.py against a fake KaggleApi (a real
account/kaggle_secrets is only ever available on an actual Kaggle notebook
— see that module's docstring). The fake replicates the one behavior this
module's correctness actually depends on: dataset_create_version() is a
full-snapshot REPLACE, not a diff — whatever isn't in the uploaded folder
disappears from the new version. That's exactly what push_checkpoint()'s
pull-then-merge-then-push design exists to work around, and
test_pushing_one_model_never_deletes_another_already_persisted_model below
is the test that would catch a regression of that design.
"""

import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src import config
from src.models import kaggle_persist, kaggle_persist_report


class FakeKaggleApi:
    """In-memory stand-in for KaggleApi: a single flat filename -> bytes
    store, mutated only the way the real dataset_create_new/_version calls
    would (full replace on every call).
    """

    CONFIG_NAME_USER = "username"

    def __init__(self):
        self.config_values = {"username": "testuser"}
        self.exists = False
        self.files: dict[str, bytes] = {}
        self.create_calls = 0
        self.version_calls = 0
        self.next_result_status = "ok"
        self.next_result_error = None
        self.raise_on_create = None

    def authenticate(self):
        pass

    def dataset_list(self, mine=False, search=None):
        if self.exists:
            return [types.SimpleNamespace(ref=f"testuser/{search}")]
        return []

    def dataset_download_files(self, dataset, path=None, unzip=False, quiet=True, force=False):
        dest = Path(path)
        dest.mkdir(parents=True, exist_ok=True)
        for name, content in self.files.items():
            (dest / name).write_bytes(content)

    def dataset_download_file(self, dataset, file_name, path=None, force=False, quiet=True, licenses=None):
        if file_name not in self.files:
            raise RuntimeError(f"404: {file_name} not found in fake dataset")
        dest = Path(path)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / file_name).write_bytes(self.files[file_name])

    def dataset_list_files(self, dataset):
        return types.SimpleNamespace(dataset_files=[types.SimpleNamespace(name=n) for n in self.files])

    def _snapshot(self, folder):
        folder_path = Path(folder)
        return {
            f.name: f.read_bytes()
            for f in folder_path.iterdir()
            if f.is_file() and f.name != "dataset-metadata.json"
        }

    def dataset_create_new(self, folder, public=False, quiet=False, **kwargs):
        self.create_calls += 1
        if self.raise_on_create:
            raise self.raise_on_create
        self.files = self._snapshot(folder)  # full replace
        self.exists = True
        return types.SimpleNamespace(status=self.next_result_status, error=self.next_result_error)

    def dataset_create_version(self, folder, version_notes, quiet=False, **kwargs):
        self.version_calls += 1
        if self.raise_on_create:
            raise self.raise_on_create
        self.files = self._snapshot(folder)  # full replace, same as the real API
        return types.SimpleNamespace(status=self.next_result_status, error=self.next_result_error)


@pytest.fixture
def fake_api(monkeypatch, tmp_path):
    api = FakeKaggleApi()
    monkeypatch.setattr(kaggle_persist, "_authenticated_api", lambda: api)
    monkeypatch.setattr(config, "KAGGLE_PERSIST_STAGING_DIR", tmp_path / "staging")
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path / "artifacts")
    return api


def _make_checkpoint_dir(tmp_path, name, files: dict) -> Path:
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    for filename, content in files.items():
        (d / filename).write_text(content, encoding="utf-8")
    return d


def test_push_checkpoint_creates_dataset_on_first_use(fake_api, tmp_path):
    checkpoint_dir = _make_checkpoint_dir(
        tmp_path, "ckpt_a", {"phase1.weights.h5": "weights-a", "state.json": '{"phase": 1}'}
    )

    kaggle_persist.push_checkpoint("MobileNetV2", checkpoint_dir)

    assert fake_api.create_calls == 1
    assert fake_api.version_calls == 0
    assert fake_api.files["MobileNetV2__phase1.weights.h5"] == b"weights-a"
    assert fake_api.files["MobileNetV2__state.json"] == b'{"phase": 1}'


def test_push_checkpoint_uses_version_on_subsequent_pushes(fake_api, tmp_path):
    checkpoint_dir = _make_checkpoint_dir(tmp_path, "ckpt_a", {"phase1.weights.h5": "v1"})
    kaggle_persist.push_checkpoint("MobileNetV2", checkpoint_dir)

    (checkpoint_dir / "phase1.weights.h5").write_text("v2", encoding="utf-8")
    kaggle_persist.push_checkpoint("MobileNetV2", checkpoint_dir)

    assert fake_api.create_calls == 1
    assert fake_api.version_calls == 1
    assert fake_api.files["MobileNetV2__phase1.weights.h5"] == b"v2"


def test_pushing_one_model_never_deletes_another_already_persisted_model(fake_api, tmp_path):
    # Model A finishes and is pushed first.
    ckpt_a = _make_checkpoint_dir(tmp_path, "ckpt_a", {"phase2.weights.h5": "a-weights"})
    kaggle_persist.push_checkpoint("MobileNetV2", ckpt_a)

    # Model B's push must not wipe out model A's already-persisted files,
    # even though dataset_create_version() is a full-snapshot replace.
    ckpt_b = _make_checkpoint_dir(tmp_path, "ckpt_b", {"phase1.weights.h5": "b-weights"})
    kaggle_persist.push_checkpoint("EfficientNetB0", ckpt_b)

    assert fake_api.files["MobileNetV2__phase2.weights.h5"] == b"a-weights"
    assert fake_api.files["EfficientNetB0__phase1.weights.h5"] == b"b-weights"


def test_push_checkpoint_refuses_to_push_an_empty_directory(fake_api, tmp_path):
    empty_dir = tmp_path / "empty_ckpt"
    empty_dir.mkdir()
    with pytest.raises(RuntimeError, match="empty/missing checkpoint"):
        kaggle_persist.push_checkpoint("MobileNetV2", empty_dir)
    assert fake_api.create_calls == 0


def test_push_checkpoint_raises_loudly_on_api_error_status(fake_api, tmp_path):
    checkpoint_dir = _make_checkpoint_dir(tmp_path, "ckpt", {"phase1.weights.h5": "x"})
    fake_api.next_result_status = "error"
    fake_api.next_result_error = "quota exceeded"

    with pytest.raises(RuntimeError, match="quota exceeded"):
        kaggle_persist.push_checkpoint("MobileNetV2", checkpoint_dir)


def test_push_checkpoint_raises_loudly_when_the_api_call_itself_raises(fake_api, tmp_path):
    checkpoint_dir = _make_checkpoint_dir(tmp_path, "ckpt", {"phase1.weights.h5": "x"})
    fake_api.raise_on_create = ConnectionError("network is unreachable")

    with pytest.raises(RuntimeError, match="Failed to push"):
        kaggle_persist.push_checkpoint("MobileNetV2", checkpoint_dir)


def test_restore_checkpoint_never_overwrites_in_session_data(fake_api, tmp_path):
    checkpoint_dir = _make_checkpoint_dir(tmp_path, "ckpt", {"phase1.weights.h5": "in-session"})
    fake_api.exists = True
    fake_api.files["MobileNetV2__phase1.weights.h5"] = b"persisted"

    restored = kaggle_persist.restore_checkpoint("MobileNetV2", checkpoint_dir)

    assert restored is False
    assert (checkpoint_dir / "phase1.weights.h5").read_text(encoding="utf-8") == "in-session"


def test_restore_checkpoint_returns_false_when_no_dataset_exists_yet(fake_api, tmp_path):
    checkpoint_dir = tmp_path / "fresh_ckpt"
    assert kaggle_persist.restore_checkpoint("MobileNetV2", checkpoint_dir) is False


def test_restore_checkpoint_downloads_persisted_files_for_a_fresh_session(fake_api, tmp_path):
    fake_api.exists = True
    fake_api.files["MobileNetV2__phase1.weights.h5"] = b"persisted-weights"
    fake_api.files["MobileNetV2__state.json"] = b'{"phase": 1, "epoch": 4}'
    fake_api.files["MobileNetV2__phase1_complete.json"] = b'{"epochs_ran": 5}'
    # A different model's file must never leak into this model's restore.
    fake_api.files["EfficientNetB0__phase1.weights.h5"] = b"other-model"
    # history/manifest belong in artifacts/, not the checkpoint dir.
    fake_api.files["MobileNetV2__history_MobileNetV2.json"] = b"{}"

    checkpoint_dir = tmp_path / "fresh_ckpt"
    restored = kaggle_persist.restore_checkpoint("MobileNetV2", checkpoint_dir)

    assert restored is True
    assert (checkpoint_dir / "phase1.weights.h5").read_bytes() == b"persisted-weights"
    assert json.loads((checkpoint_dir / "state.json").read_text()) == {"phase": 1, "epoch": 4}
    assert (checkpoint_dir / "phase1_complete.json").is_file()
    assert not (checkpoint_dir / "EfficientNetB0__phase1.weights.h5").exists()
    assert not (checkpoint_dir / "history_MobileNetV2.json").exists()


def test_restore_artifacts_never_overwrites_local_files(fake_api, tmp_path):
    config.ARTIFACTS_DIR.mkdir(parents=True)
    history_path = config.ARTIFACTS_DIR / "history_MobileNetV2.json"
    history_path.write_text('{"local": true}', encoding="utf-8")
    fake_api.exists = True
    fake_api.files["MobileNetV2__history_MobileNetV2.json"] = b'{"persisted": true}'

    restored = kaggle_persist.restore_artifacts("MobileNetV2")

    assert restored is False
    assert json.loads(history_path.read_text()) == {"local": True}


def test_restore_artifacts_downloads_when_locally_absent(fake_api):
    fake_api.exists = True
    fake_api.files["MobileNetV2__history_MobileNetV2.json"] = b'{"phase1": {}}'
    fake_api.files["MobileNetV2__train_manifest_MobileNetV2.json"] = b'{"model_name": "MobileNetV2"}'

    restored = kaggle_persist.restore_artifacts("MobileNetV2")

    assert restored is True
    history = json.loads((config.ARTIFACTS_DIR / "history_MobileNetV2.json").read_text())
    assert history == {"phase1": {}}
    manifest = json.loads((config.ARTIFACTS_DIR / "train_manifest_MobileNetV2.json").read_text())
    assert manifest == {"model_name": "MobileNetV2"}


def test_report_progress_reflects_each_models_persisted_state(fake_api):
    fake_api.exists = True
    # MobileNetV2: fully done, with a history to read best macro-F1 from.
    fake_api.files["MobileNetV2__phase2_complete.json"] = b"{}"
    fake_api.files["MobileNetV2__phase2.weights.h5"] = b"w"
    fake_api.files["MobileNetV2__history_MobileNetV2.json"] = json.dumps(
        {"phase1": {"val_macro_f1": [0.1, 0.2]}, "phase2": {"val_macro_f1": [0.3, 0.5, 0.4]}}
    ).encode("utf-8")
    # EfficientNetB0: mid phase 1, no history yet.
    fake_api.files["EfficientNetB0__phase1.weights.h5"] = b"w"
    # Every other CANDIDATE_MODELS entry has nothing persisted at all.

    report = kaggle_persist_report.report_progress()

    assert report["MobileNetV2"]["checkpoint_exists"] is True
    assert report["MobileNetV2"]["phase_reached"] == "phase 2 complete"
    assert report["MobileNetV2"]["best_val_macro_f1"] == pytest.approx(0.5)

    assert report["EfficientNetB0"]["checkpoint_exists"] is True
    assert report["EfficientNetB0"]["phase_reached"] == "phase 1 in progress"
    assert report["EfficientNetB0"]["best_val_macro_f1"] is None

    untouched = set(config.CANDIDATE_MODELS) - {"MobileNetV2", "EfficientNetB0"}
    for model_name in untouched:
        assert report[model_name]["checkpoint_exists"] is False
        assert report[model_name]["phase_reached"] == "not started"


def test_report_progress_returns_all_not_started_when_no_dataset_exists_yet(fake_api):
    report = kaggle_persist_report.report_progress()
    assert all(not entry["checkpoint_exists"] for entry in report.values())
    assert all(entry["phase_reached"] == "not started" for entry in report.values())
