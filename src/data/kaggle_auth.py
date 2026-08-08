"""Installs Kaggle API credentials on a Colab runtime from an uploaded
kaggle.json, so colab/01_data_setup.ipynb's credentials cell can be a thin
`kaggle_auth.install_credentials()` call with no logic of its own.

Kaggle has issued kaggle.json in two shapes depending on when/where the
token was generated:
  - legacy:  {"username": ..., "key": ...}
  - current: {"userName": ..., "apiToken": ...}
Both are accepted here and normalized to the legacy shape on disk, since
that's what the `kaggle` package's own config loader expects. The
normalized file is written to both locations the `kaggle` package may look
for it (older versions default to ~/.kaggle, some environments expect the
XDG-style ~/.config/kaggle), and KAGGLE_CONFIG_DIR is set explicitly so
there's no ambiguity about which one is actually used. Authentication is
proven with a real KaggleApi().authenticate() call before this function
returns, so a bad/expired token fails here — loudly — rather than partway
through src/data/download.py's Kaggle fetch.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config  # noqa: E402

# Colab runtimes always run as root, so hardcoding /root (rather than
# Path.home()) is safe here — this module is Colab-only (see
# _require_colab) and these are the exact two locations the `kaggle`
# package's config loader is known to check across versions.
_LEGACY_CONFIG_DIR = Path("/root/.kaggle")
_XDG_CONFIG_DIR = Path("/root/.config/kaggle")


def _require_colab() -> None:
    if not config.IS_COLAB:
        raise RuntimeError(
            "Refusing to install Kaggle credentials: this is not a Colab "
            "environment. Per project convention, this only ever runs from "
            "a Colab session (CLAUDE.md rule 10)."
        )


def _normalize(raw: dict) -> dict:
    """Returns {"username", "key"}, accepting either credential shape
    Kaggle has used. Raises — never guesses — on anything else.
    """
    if "username" in raw and "key" in raw:
        return {"username": raw["username"], "key": raw["key"]}
    if "userName" in raw and "apiToken" in raw:
        return {"username": raw["userName"], "key": raw["apiToken"]}
    raise ValueError(
        "kaggle.json has neither the legacy {'username', 'key'} shape nor "
        "the current {'userName', 'apiToken'} shape. Keys found: "
        f"{sorted(raw)}."
    )


def _write_credentials(creds: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / "kaggle.json"
    dest.write_text(json.dumps(creds), encoding="utf-8")
    os.chmod(dest, 0o600)


def install_credentials() -> None:
    """Prompts for a kaggle.json upload (Colab file-picker widget),
    normalizes and installs the credentials, and proves authentication
    succeeds. Never prints or logs the key/token — only the username, as
    confirmation.
    """
    _require_colab()

    from google.colab import files

    uploaded = files.upload()
    raw = json.loads(next(iter(uploaded.values())).decode("utf-8"))
    creds = _normalize(raw)

    _write_credentials(creds, _LEGACY_CONFIG_DIR)
    _write_credentials(creds, _XDG_CONFIG_DIR)

    os.environ["KAGGLE_USERNAME"] = creds["username"]
    os.environ["KAGGLE_KEY"] = creds["key"]
    os.environ["KAGGLE_CONFIG_DIR"] = str(_LEGACY_CONFIG_DIR)

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    print(f"[kaggle_auth] Authenticated as '{creds['username']}'.")
