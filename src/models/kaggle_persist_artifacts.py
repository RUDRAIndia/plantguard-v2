"""Cross-session persistence for artifacts/ on Kaggle, via the Kaggle API.

**The problem this solves.** /kaggle/working (where config.ARTIFACTS_DIR
resolves on Kaggle) does not survive a session restart or the ~9-12 hour
session time limit — see src/models/kaggle_persist.py's module docstring
for the full explanation of that constraint. That module already solves
this for training checkpoints; this module solves it for everything else
under artifacts/ that a fresh session would otherwise have to pay for
again: results.json (CLAUDE.md rule 5 — the one file every report/figure
number is read from) and the dedupe/split/inventory/mapping outputs
(src/data/dedupe.py, split_report.py, inventory.py, mapping_report.py),
which together take roughly 20 minutes to regenerate.

**Storage layout.** Reuses config.KAGGLE_PERSIST_DATASET_SLUG — the exact
same private dataset src/models/kaggle_persist.py pushes checkpoints to —
rather than a second dataset, so there is only ever one thing to
authenticate against and one merge-before-push story to reason about. Files
live in the same flat namespace (no subdirectories — see that module's
docstring for why), under an "artifacts__" prefix that can never collide
with a "<model_name>__" checkpoint prefix (no CANDIDATE_MODELS entry is
named "artifacts"). config.KAGGLE_PERSIST_ARTIFACT_FILENAMES' fixed files
become "artifacts__<filename>"; config.EVAL_FIGURES_DIR's contents (which
do have one level of real subdirectory, figures/gradcam/) are flattened by
replacing "/" with the same "__" separator and reversed the same way on
restore — safe because none of the actual filenames
src/evaluate/metrics.py / calibration.py / gradcam.py write contain a
double underscore.

**When this runs.** push_data_artifacts() is called once, at the very end
of src/evaluate/runner.py's run(), after results.json is written — not
incrementally after dedupe/split/inventory/mapping individually. These
derived artifacts are only durably worth persisting once a full run (data
prep through evaluation) has actually succeeded, the same reasoning
push_checkpoint(..., include_artifacts=True) uses for history/manifest
JSONs. restore_data_artifacts() is called from
src/data/prepare_artifacts.py's ensure_data_artifacts(), before deciding
whether dedupe/split/inventory/mapping need to run at all.

**Failure handling and the merge-before-push story** are identical to
src/models/kaggle_persist.py — see that module's docstring for the full
reasoning (pull-then-merge-then-push so pushing artifacts/ can never
silently delete an already-persisted checkpoint or vice versa; a failed
merge-download is worded distinctly from a failed upload; kaggle_retry.py
retries transient 404s/5xx). Reuses that module's authenticated-API,
dataset-ref, dataset-exists, and staging-dir helpers rather than
duplicating them (same pattern src/models/kaggle_persist_report.py
follows) — split into its own module to stay under CLAUDE.md rule 12's
~300-line guideline for kaggle_persist.py itself.
"""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config  # noqa: E402
from src.models import kaggle_persist, kaggle_retry  # noqa: E402

_SEPARATOR = kaggle_persist._SEPARATOR
_PREFIX = "artifacts" + _SEPARATOR
_FIGURES_PREFIX = "figures" + _SEPARATOR


def _figure_files() -> list[Path]:
    if not config.EVAL_FIGURES_DIR.is_dir():
        return []
    return sorted(p for p in config.EVAL_FIGURES_DIR.rglob("*") if p.is_file())


def _missing_expected_paths() -> list[str]:
    """Names exactly which expected file(s)/dir push_data_artifacts() can't
    find locally, so a push refusal (see below) says precisely what's
    missing rather than just failing.
    """
    missing = [
        name
        for name in config.KAGGLE_PERSIST_ARTIFACT_FILENAMES
        if not (config.ARTIFACTS_DIR / name).is_file()
    ]
    if not _figure_files():
        missing.append(str(config.EVAL_FIGURES_DIR))
    return missing


def _is_locally_complete() -> bool:
    return not _missing_expected_paths()


def push_data_artifacts() -> None:
    """Pushes config.KAGGLE_PERSIST_ARTIFACT_FILENAMES plus every file under
    config.EVAL_FIGURES_DIR to the private KAGGLE_PERSIST_DATASET_SLUG
    dataset (the same one push_checkpoint() uses), as a new version if it
    already exists or a fresh private dataset otherwise. Preserves every
    already-persisted checkpoint and any previously-pushed artifacts/ file
    (see module docstring on why a naive push would silently delete them).
    Raises loudly, naming exactly what's missing, if this run's artifacts/
    is incomplete — this only ever runs after src/evaluate/runner.py's
    run() has already written everything, so a missing file means something
    upstream is broken, not a partial push to paper over (CLAUDE.md rule 1).
    """
    missing = _missing_expected_paths()
    if missing:
        raise RuntimeError(
            f"Refusing to push artifacts/ — missing expected file(s)/directory: {missing}. "
            "push_data_artifacts() is only meant to run after src/evaluate/runner.py's "
            "run() has finished writing everything; investigate why one of those outputs "
            "is missing rather than pushing an incomplete bundle."
        )

    api = kaggle_persist._authenticated_api()
    dataset_ref = kaggle_persist._dataset_ref(api)
    staging = kaggle_persist._fresh_staging_dir()

    already_exists = kaggle_persist._dataset_exists(api, dataset_ref)
    if already_exists:
        try:
            kaggle_retry.call_with_retry(
                lambda: api.dataset_download_files(dataset_ref, path=str(staging), unzip=True, quiet=True),
                description=f"Downloading existing '{dataset_ref}' content to merge before pushing artifacts/",
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not download existing Kaggle dataset '{dataset_ref}' content to merge "
                f"before pushing artifacts/, even after retrying: {exc}. IMPORTANT: this call "
                "uploaded NOTHING — whatever was already durably persisted on Kaggle from a "
                "previous successful push (any model's checkpoint, or a prior artifacts/ push) "
                "is untouched and safe. Only this call's own new artifacts/ content was not "
                "uploaded this round — retry the push to persist it."
            ) from exc

    for stale in staging.glob(_PREFIX + "*"):
        if stale.is_file():
            stale.unlink()

    for filename in config.KAGGLE_PERSIST_ARTIFACT_FILENAMES:
        shutil.copy2(config.ARTIFACTS_DIR / filename, staging / (_PREFIX + filename))

    for figure_path in _figure_files():
        relative_path = figure_path.relative_to(config.EVAL_FIGURES_DIR).as_posix()
        remote_name = _PREFIX + _FIGURES_PREFIX + relative_path.replace("/", _SEPARATOR)
        shutil.copy2(figure_path, staging / remote_name)

    metadata = {
        "title": config.KAGGLE_PERSIST_DATASET_SLUG,
        "id": dataset_ref,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (staging / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    try:
        if already_exists:
            result = api.dataset_create_version(str(staging), version_notes="Update artifacts/", quiet=False)
        else:
            result = api.dataset_create_new(str(staging), public=False, quiet=False)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to push artifacts/ to Kaggle dataset '{dataset_ref}': {exc}. This "
            "session's results.json / dedupe / split / inventory / mapping outputs are NOT "
            "safely persisted — do not let the session end without resolving this, or the "
            "next session will pay the ~20-minute regeneration cost again."
        ) from exc
    kaggle_persist._check_create_result(result, model_name="artifacts/", dataset_ref=dataset_ref)

    shutil.rmtree(staging, ignore_errors=True)
    print(f"[kaggle_persist_artifacts] Pushed artifacts/ to {dataset_ref}.")


def restore_data_artifacts() -> bool:
    """If artifacts/ is missing any of config.KAGGLE_PERSIST_ARTIFACT_FILENAMES
    or has an empty/missing EVAL_FIGURES_DIR, and a persisted artifacts/
    bundle exists in KAGGLE_PERSIST_DATASET_SLUG, downloads whichever pieces
    are locally absent. Never overwrites a locally-present file (a file that
    already exists is at least as recent as anything persisted — same
    invariant as restore_checkpoint()/restore_artifacts()). Returns whether
    anything was restored.
    """
    if _is_locally_complete():
        return False

    api = kaggle_persist._authenticated_api()
    dataset_ref = kaggle_persist._dataset_ref(api)

    if not kaggle_persist._dataset_exists(api, dataset_ref):
        print(f"[kaggle_persist_artifacts] No persisted '{config.KAGGLE_PERSIST_DATASET_SLUG}' dataset yet — nothing to restore.")
        return False

    staging = kaggle_persist._fresh_staging_dir()
    try:
        kaggle_retry.call_with_retry(
            lambda: api.dataset_download_files(dataset_ref, path=str(staging), unzip=True, quiet=True),
            description="Downloading persisted artifacts/ content to restore",
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not download persisted Kaggle dataset '{dataset_ref}' content to restore "
            f"artifacts/, even after retrying: {exc}. Nothing was restored — dedupe/split/"
            "inventory/mapping will be regenerated from scratch this session unless this is "
            "resolved."
        ) from exc

    restored_files = [f for f in staging.glob(_PREFIX + "*") if f.is_file()]
    if not restored_files:
        print(
            f"[kaggle_persist_artifacts] '{config.KAGGLE_PERSIST_DATASET_SLUG}' exists but has "
            "no persisted artifacts/ bundle yet — regenerating from scratch this session."
        )
        shutil.rmtree(staging, ignore_errors=True)
        return False

    restored_any = False
    for f in restored_files:
        remainder = f.name[len(_PREFIX):]
        if remainder.startswith(_FIGURES_PREFIX):
            relative_path = remainder[len(_FIGURES_PREFIX):].replace(_SEPARATOR, "/")
            dest = config.EVAL_FIGURES_DIR / relative_path
        elif remainder in config.KAGGLE_PERSIST_ARTIFACT_FILENAMES:
            dest = config.ARTIFACTS_DIR / remainder
        else:
            # Unrecognized entry (e.g. pushed by a newer/older version of
            # this module) -- not this restore's business to guess at.
            continue

        if dest.exists():
            continue  # never overwrite in-session/local data

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)
        restored_any = True

    shutil.rmtree(staging, ignore_errors=True)
    if restored_any:
        print(f"[kaggle_persist_artifacts] Restored artifacts/ from {dataset_ref}.")
    return restored_any
