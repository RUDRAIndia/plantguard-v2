"""Installs Kaggle API credentials, on Colab from an uploaded kaggle.json
and on Kaggle from the notebook's own attached Secret, so a notebook's
credentials cell can be a thin `kaggle_auth.install_credentials()` /
`install_credentials_kaggle()` call with no logic of its own. Both
src/data/fetch.py (dataset downloads) and src/models/kaggle_persist.py
(checkpoint push/restore, added for Day 6's cross-session persistence) rely
on whichever of these ran first in the session.

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
proven with a real KaggleApi().authenticate() call before either function
returns, so a bad/expired/missing credential fails here — loudly — rather
than partway through a download or a checkpoint push. Neither function ever
prints or logs the key/token itself — only the username, as confirmation.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config  # noqa: E402

# The two locations the `kaggle` package's config loader is known to check
# across versions. Path.home() rather than a hardcoded /root: Colab
# runtimes always run as root so this resolves the same either way there,
# but Kaggle's kernel user isn't guaranteed to be root the same way.
_LEGACY_CONFIG_DIR = Path.home() / ".kaggle"
_XDG_CONFIG_DIR = Path.home() / ".config" / "kaggle"


def _require_colab() -> None:
    if not config.IS_COLAB:
        raise RuntimeError(
            "Refusing to install Kaggle credentials: this is not a Colab "
            "environment. Per project convention, this only ever runs from "
            "a Colab session (CLAUDE.md rule 10)."
        )


def _require_kaggle() -> None:
    if not config.IS_KAGGLE:
        raise RuntimeError(
            "Refusing to install Kaggle credentials: this is not a Kaggle "
            "environment (config.IS_KAGGLE is False)."
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


def _install_and_verify(creds: dict) -> None:
    """Shared tail end of both install_credentials() and
    install_credentials_kaggle(): write to both known config locations, set
    the env vars the `kaggle` package also checks, and prove authentication
    with a real API call before returning.
    """
    _write_credentials(creds, _LEGACY_CONFIG_DIR)
    _write_credentials(creds, _XDG_CONFIG_DIR)

    os.environ["KAGGLE_USERNAME"] = creds["username"]
    os.environ["KAGGLE_KEY"] = creds["key"]
    os.environ["KAGGLE_CONFIG_DIR"] = str(_LEGACY_CONFIG_DIR)

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    print(f"[kaggle_auth] Authenticated as '{creds['username']}'.")


def install_credentials() -> None:
    """Colab path: prompts for a kaggle.json upload (Colab file-picker
    widget), normalizes and installs the credentials, and proves
    authentication succeeds.
    """
    _require_colab()

    from google.colab import files

    uploaded = files.upload()
    raw = json.loads(next(iter(uploaded.values())).decode("utf-8"))
    creds = _normalize(raw)
    _install_and_verify(creds)


def install_credentials_kaggle() -> None:
    """Kaggle path: reads KAGGLE_USERNAME / KAGGLE_KEY from the notebook's
    own attached Secrets (Add-ons -> Secrets in the Kaggle notebook editor —
    the standard, documented way to give a Kaggle kernel its own API
    credentials; kernels are never issued them automatically), normalizes
    and installs them, and proves authentication succeeds. Idempotent and
    cheap to call repeatedly — src/models/kaggle_persist.py calls this
    before every push/restore rather than assuming an earlier call in the
    session already ran.
    """
    _require_kaggle()

    from kaggle_secrets import UserSecretsClient

    secrets = UserSecretsClient()
    try:
        raw = {
            "username": secrets.get_secret("KAGGLE_USERNAME"),
            "key": secrets.get_secret("KAGGLE_KEY"),
        }
    except Exception as exc:
        raise RuntimeError(
            "Could not read KAGGLE_USERNAME / KAGGLE_KEY from this "
            "notebook's Secrets. Add them via Add-ons -> Secrets in the "
            "Kaggle notebook editor (values from your Kaggle Account -> API "
            "-> Create New Token kaggle.json), then attach them to this "
            "notebook and re-run."
        ) from exc

    if not raw["username"] or not raw["key"]:
        raise RuntimeError(
            "This notebook's KAGGLE_USERNAME / KAGGLE_KEY Secrets are empty. "
            "Set both via Add-ons -> Secrets, attach them to this notebook, "
            "and re-run."
        )

    creds = _normalize(raw)
    _install_and_verify(creds)
