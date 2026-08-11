"""Cross-session checkpoint persistence for Kaggle, via the Kaggle API.

**The problem this solves.** Real training is roughly 4 GPU-hours per
model across several Kaggle sessions, but /kaggle/working — where
config.CHECKPOINT_DIR resolves on Kaggle — does NOT survive a session
restart or the ~9-12 hour session time limit (see
src/models/checkpoint.py's docstring). Without this module, a session
ending mid-training would destroy everything trained so far.

**The resume story, accurately.** This module pushes each model's
checkpoint (weights + resume-state JSON + phase-complete markers) to a
private Kaggle Dataset named config.KAGGLE_PERSIST_DATASET_SLUG
("plantguard-artifacts"), owned by whichever account authenticates — after
phase 1 completes, after phase 2 completes, and once more with
history/manifest JSONs included once the whole run finishes (so a session
dying at any of those points loses at most the epoch in progress within the
current phase, same guarantee CHECKPOINT_DIR's own per-epoch save already
gives within a session — src/train.py calls push_checkpoint() at each of
those points). On startup, restore_checkpoint() downloads that model's
persisted files back into config.CHECKPOINT_DIR *before* training starts,
so a brand-new session's checkpoint directory looks exactly like the
previous session left it — src/models/checkpoint.py's existing state.json /
phaseN_complete.json resume logic then picks it up completely unchanged, no
different from resuming an interrupted-but-not-restarted session. A fresh
session therefore loses at most the training done since the last phase
boundary, never an entire session's work, and never requires a manual
"Save Version and re-attach" step the way relying on /kaggle/working alone
would.

**Storage layout.** Kaggle Dataset versions are full-snapshot replacements,
not diffs — creating a new version from a folder makes that folder's
contents the *entire* new version, so naively pushing only the model
currently being trained would silently delete every other model's already-
persisted checkpoint. To avoid that, every push first downloads the
dataset's current full content, then only adds/overwrites the file for the
model being pushed, then re-uploads everything. Files live in one flat
namespace inside the dataset (no subdirectories, to sidestep the Kaggle
API's own directory-upload mode entirely) named "<model_name>__<filename>",
e.g. "MobileNetV2__phase2.weights.h5", "MobileNetV2__history_MobileNetV2.json".

**Credentials.** Read the same way as everywhere else in this project —
src/data/kaggle_auth.py's install_credentials_kaggle() (Kaggle-notebook
Secrets, normalized, written to the same config locations, verified with a
real API call) — and never printed or logged here.

**Failure handling.** A push failure raises immediately (CLAUDE.md rule 1)
rather than being logged and swallowed: a silent persistence failure is
only discovered after the session that would have been protected by it is
already gone, which is worse than no persistence at all.

**The download-after-upload race.** Kaggle ingests a new dataset version
asynchronously: `dataset_create_version`/`dataset_create_new` returning
"ok" only means the upload was accepted, not that the version has finished
processing server-side. A `dataset_download_files` call moments later (the
very next push_checkpoint() call's merge-download, or restore_checkpoint()
racing a push from another session) can therefore 404 even though the
prior upload fully succeeded and that content is safe.
src/models/kaggle_retry.py's call_with_retry() retries such 404s (and
5xx/network errors) with bounded exponential backoff before giving up
(split into its own module to stay under CLAUDE.md rule 12's ~300-line
guideline). If it still fails after retries, the resulting
error is deliberately worded differently from an upload failure: a failed
*download* means nothing new was uploaded this call — anything already
persisted by a previous successful push is untouched — whereas a failed
*upload* (`_check_create_result`/the `dataset_create_version` except block
below) means this call's content genuinely never made it to Kaggle. Those
are very different situations for whoever is watching the session and must
never be reported with the same message.

**What this module cannot test outside Kaggle.** kaggle_secrets and a live
authenticated Kaggle Dataset are only available on an actual Kaggle
notebook instance — every function here is unit-tested against a mocked
KaggleApi (tests/test_kaggle_persist.py), not a real account.

**Progress reporting** (which models have a persisted checkpoint, how far
they got, and their best validation macro-F1 so far) is
src/models/kaggle_persist_report.py, not here — split out to stay under
CLAUDE.md rule 12's ~300-line guideline; it reuses this module's
authenticated-API and staging helpers rather than duplicating them.
"""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config  # noqa: E402
from src.data import kaggle_auth  # noqa: E402
from src.models import kaggle_retry  # noqa: E402

_SEPARATOR = "__"


def _authenticated_api():
    """Installs/verifies Kaggle credentials the same way
    src/data/kaggle_auth.py does everywhere else, then returns a ready-to-use
    KaggleApi. Cheap and safe to call repeatedly — install_credentials_kaggle()
    re-proves auth each time rather than assuming an earlier call succeeded.
    """
    kaggle_auth.install_credentials_kaggle()
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def _dataset_ref(api) -> str:
    username = api.config_values.get(api.CONFIG_NAME_USER)
    if not username:
        raise RuntimeError(
            "Authenticated Kaggle API has no username in its config — cannot "
            "determine the persisted dataset's owner/ref."
        )
    return f"{username}/{config.KAGGLE_PERSIST_DATASET_SLUG}"


def _dataset_exists(api, dataset_ref: str) -> bool:
    try:
        existing = api.dataset_list(mine=True, search=config.KAGGLE_PERSIST_DATASET_SLUG) or []
    except Exception as exc:
        raise RuntimeError(
            f"Could not list Kaggle datasets to check whether '{dataset_ref}' "
            f"already exists: {exc}."
        ) from exc
    return any(str(dataset.ref) == dataset_ref for dataset in existing)


def _fresh_staging_dir() -> Path:
    staging = config.KAGGLE_PERSIST_STAGING_DIR
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    return staging


def _prefixed(model_name: str, filename: str) -> str:
    return f"{model_name}{_SEPARATOR}{filename}"


def _check_create_result(result, *, model_name: str, dataset_ref: str) -> None:
    """Kaggle's dataset_create_new()/dataset_create_version() don't raise on
    an API-level failure — they return a response whose .status/.error must
    be checked explicitly (this is the same check the kaggle CLI's own
    reference implementation does). CLAUDE.md rule 1: a push that silently
    "succeeds" while actually failing is exactly the late failure this
    module exists to prevent.
    """
    if result is None:
        raise RuntimeError(
            f"Kaggle dataset push for '{model_name}' returned no response "
            f"(dataset: {dataset_ref}). This session's progress is NOT "
            "safely persisted."
        )
    status = (result.status or "").lower()
    if status != "ok":
        raise RuntimeError(
            f"Kaggle dataset push for '{model_name}' failed (dataset: "
            f"{dataset_ref}): {result.error}. This session's progress is "
            "NOT safely persisted — do not let the session end without "
            "resolving this, or trained weights will be lost."
        )


def push_checkpoint(model_name: str, checkpoint_dir: Path, *, include_artifacts: bool = False) -> None:
    """Pushes every file currently in `checkpoint_dir` (whatever phase(s)
    have completed so far — EarlyStopping's restore_best_weights already
    means the saved phaseN.weights.h5 is that phase's best, not just its
    last epoch) to the private KAGGLE_PERSIST_DATASET_SLUG dataset, as a new
    version if it already exists or a fresh private dataset if this is the
    first push ever. If `include_artifacts`, also pushes
    artifacts/history_<model_name>.json and
    artifacts/train_manifest_<model_name>.json (only once run_training()
    has actually written them). Preserves every other model's already-
    persisted files (see module docstring on why a naive push would
    silently delete them). Raises loudly on any failure — never swallows
    one (CLAUDE.md rule 1).
    """
    if not checkpoint_dir.is_dir() or not any(checkpoint_dir.iterdir()):
        raise RuntimeError(
            f"Refusing to push an empty/missing checkpoint directory "
            f"({checkpoint_dir}) for '{model_name}' — nothing to persist."
        )

    api = _authenticated_api()
    dataset_ref = _dataset_ref(api)
    staging = _fresh_staging_dir()

    already_exists = _dataset_exists(api, dataset_ref)
    if already_exists:
        try:
            kaggle_retry.call_with_retry(
                lambda: api.dataset_download_files(dataset_ref, path=str(staging), unzip=True, quiet=True),
                description=f"Downloading existing '{dataset_ref}' content to merge before pushing {model_name}",
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not download existing Kaggle dataset '{dataset_ref}' content to merge "
                f"before pushing {model_name}'s checkpoint, even after retrying: {exc}. "
                "IMPORTANT: this call uploaded NOTHING — whatever was already durably "
                f"persisted on Kaggle from a previous successful push (for {model_name} or any "
                "other model) is untouched and safe. Only this call's own new content "
                f"({checkpoint_dir}"
                f"{', plus history/manifest artifacts' if include_artifacts else ''}) was not "
                "uploaded this round — retry the push to persist it."
            ) from exc

    prefix = model_name + _SEPARATOR
    for stale in staging.glob(prefix + "*"):
        if stale.is_file():
            stale.unlink()

    for f in checkpoint_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, staging / _prefixed(model_name, f.name))

    if include_artifacts:
        for filename in (f"history_{model_name}.json", f"train_manifest_{model_name}.json"):
            src = config.ARTIFACTS_DIR / filename
            if src.is_file():
                shutil.copy2(src, staging / _prefixed(model_name, filename))

    metadata = {
        "title": config.KAGGLE_PERSIST_DATASET_SLUG,
        "id": dataset_ref,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (staging / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    try:
        if already_exists:
            result = api.dataset_create_version(
                str(staging), version_notes=f"Update {model_name} checkpoint", quiet=False
            )
        else:
            result = api.dataset_create_new(str(staging), public=False, quiet=False)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to push {model_name}'s checkpoint to Kaggle dataset "
            f"'{dataset_ref}': {exc}. This session's progress is NOT safely "
            "persisted — do not let the session end without resolving this, "
            "or trained weights will be lost."
        ) from exc
    _check_create_result(result, model_name=model_name, dataset_ref=dataset_ref)

    shutil.rmtree(staging, ignore_errors=True)
    print(f"[kaggle_persist] Pushed {model_name}'s checkpoint to {dataset_ref}.")


def restore_checkpoint(model_name: str, checkpoint_dir: Path) -> bool:
    """If `checkpoint_dir` is empty or missing (nothing in-session to
    resume from — a brand-new session) and a persisted checkpoint for
    `model_name` exists in KAGGLE_PERSIST_DATASET_SLUG, downloads it into
    `checkpoint_dir` so src/models/checkpoint.py's existing resume logic
    picks it up unchanged. Never overwrites in-session data (a checkpoint
    directory that already has files is left completely alone — that data
    is at least as recent as anything persisted). Returns whether anything
    was restored.
    """
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        return False

    api = _authenticated_api()
    dataset_ref = _dataset_ref(api)

    if not _dataset_exists(api, dataset_ref):
        print(f"[kaggle_persist] No persisted '{config.KAGGLE_PERSIST_DATASET_SLUG}' dataset yet — starting {model_name} fresh.")
        return False

    staging = _fresh_staging_dir()
    try:
        kaggle_retry.call_with_retry(
            lambda: api.dataset_download_files(dataset_ref, path=str(staging), unzip=True, quiet=True),
            description=f"Downloading persisted '{dataset_ref}' content to restore {model_name}",
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not download persisted Kaggle dataset '{dataset_ref}' content to restore "
            f"{model_name}'s checkpoint, even after retrying: {exc}. Nothing was restored — "
            f"{model_name} will start from scratch this session unless this is resolved."
        ) from exc

    prefix = model_name + _SEPARATOR
    restored_files = list(staging.glob(prefix + "*"))
    if not restored_files:
        print(
            f"[kaggle_persist] '{config.KAGGLE_PERSIST_DATASET_SLUG}' exists but has no "
            f"persisted checkpoint for {model_name} yet — starting fresh."
        )
        shutil.rmtree(staging, ignore_errors=True)
        return False

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for f in restored_files:
        if f.is_file() and f.name.endswith((".weights.h5", ".json")):
            original_name = f.name[len(prefix):]
            if original_name.startswith(("history_", "train_manifest_")):
                continue  # artifacts restore to artifacts/, not the checkpoint dir — see restore_artifacts() below
            shutil.copy2(f, checkpoint_dir / original_name)

    shutil.rmtree(staging, ignore_errors=True)
    print(f"[kaggle_persist] Restored {model_name}'s persisted checkpoint from {dataset_ref} into {checkpoint_dir}.")
    return True


def restore_artifacts(model_name: str) -> bool:
    """Companion to restore_checkpoint(): if this model's
    history_<model_name>.json / train_manifest_<model_name>.json aren't
    already present locally and a persisted copy exists in
    KAGGLE_PERSIST_DATASET_SLUG (only ever pushed once a full run actually
    completed — see push_checkpoint(..., include_artifacts=True)),
    downloads them into config.ARTIFACTS_DIR. Handles the case where a
    model finished training in an earlier session and this session re-runs
    it (e.g. by mistake) — without this, src/train.py would otherwise
    overwrite the real, already-persisted history with a null one (CLAUDE.md
    rule 5: a report must never show a fabricated or silently-lost number).
    Never overwrites a locally present file. Returns whether anything was
    restored.
    """
    history_path = config.ARTIFACTS_DIR / f"history_{model_name}.json"
    manifest_path = config.ARTIFACTS_DIR / f"train_manifest_{model_name}.json"
    if history_path.is_file() or manifest_path.is_file():
        return False

    api = _authenticated_api()
    dataset_ref = _dataset_ref(api)
    if not _dataset_exists(api, dataset_ref):
        return False

    staging = _fresh_staging_dir()
    prefix = model_name + _SEPARATOR
    restored_any = False
    for filename, dest in (
        (f"history_{model_name}.json", history_path),
        (f"train_manifest_{model_name}.json", manifest_path),
    ):
        remote_name = prefix + filename
        try:
            api.dataset_download_file(dataset_ref, remote_name, path=str(staging), quiet=True, force=True)
        except Exception:
            continue  # not pushed yet -- this model hasn't fully completed in any prior session
        downloaded = staging / remote_name
        if downloaded.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(downloaded, dest)
            restored_any = True

    shutil.rmtree(staging, ignore_errors=True)
    if restored_any:
        print(f"[kaggle_persist] Restored {model_name}'s persisted history/manifest from {dataset_ref}.")
    return restored_any


