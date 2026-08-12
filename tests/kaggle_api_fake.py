"""Shared fake KaggleApi for tests/test_kaggle_persist.py and
tests/test_kaggle_persist_artifacts.py (a real account/kaggle_secrets is
only ever available on an actual Kaggle notebook instance). The fake
replicates the one behavior every push/restore function's correctness
actually depends on: dataset_create_version() is a full-snapshot REPLACE,
not a diff -- whatever isn't in the uploaded folder disappears from the new
version. That's exactly what push_checkpoint()'s and push_data_artifacts()'s
pull-then-merge-then-push design exists to work around.
"""

import types
from pathlib import Path

import requests


def http_error(status_code: int) -> requests.exceptions.HTTPError:
    """Builds a real requests.exceptions.HTTPError with a real .response,
    the same shape the real Kaggle API client raises (it's requests-based
    under the hood) -- not a hand-waved stand-in exception.
    """
    response = requests.models.Response()
    response.status_code = status_code
    return requests.exceptions.HTTPError(f"{status_code} Client Error", response=response)


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
        # Queue of exceptions dataset_download_files() raises, one per call,
        # before falling back to succeeding normally -- lets a test script
        # "fails N times then succeeds" or "always fails" (queue longer than
        # kaggle_retry.MAX_ATTEMPTS).
        self.raise_on_download: list = []
        self.download_call_count = 0

    def authenticate(self):
        pass

    def dataset_list(self, mine=False, search=None):
        if self.exists:
            return [types.SimpleNamespace(ref=f"testuser/{search}")]
        return []

    def dataset_download_files(self, dataset, path=None, unzip=False, quiet=True, force=False):
        self.download_call_count += 1
        if self.raise_on_download:
            raise self.raise_on_download.pop(0)
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
