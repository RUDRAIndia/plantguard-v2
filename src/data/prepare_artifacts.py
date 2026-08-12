"""Ensures the dedupe/split/inventory/mapping artifacts under artifacts/
exist and are valid before training or evaluation runs, restoring them from
the Kaggle-persisted artifacts/ bundle (src/models/kaggle_persist_artifacts.py)
first when possible, and regenerating from scratch only when nothing usable
was restored or what's on disk is stale.

**Never weakens the split provenance check.** Validity of a
(restored-or-already-local) artifacts/splits.json is decided by calling
src/data/split.py's compute_split_inputs() and hash_split_inputs() — the
identical two functions src/data/pipeline.py's load_splits() already calls
at training time — never a re-derived or looser copy of that check. A
restored splits.json whose hash doesn't match current inputs is treated
exactly like a stale local one: full regeneration, the same
dedupe.main() -> split_report.main() sequence colab/01_data_setup.ipynb's
Cells 5 & 7 already run unconditionally every session.

results.json is deliberately excluded from the "is artifacts/ already
valid" check here — it's produced by src/evaluate/runner.py at the very
end of a run and has nothing to do with whether dedupe/split/inventory/
mapping need to be (re)computed.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config  # noqa: E402
from src.data import dedupe, inventory, mapping_report, split, split_report  # noqa: E402
from src.models import kaggle_persist_artifacts  # noqa: E402

_DATA_PREP_FILENAMES = tuple(
    name for name in config.KAGGLE_PERSIST_ARTIFACT_FILENAMES if name != config.RESULTS_JSON_PATH.name
)


def _local_data_prep_artifacts_valid() -> bool:
    """True iff every dedupe/split/inventory/mapping output file is present
    under artifacts/ AND artifacts/splits.json's recorded split_input_hash
    matches a fresh recomputation of the current split inputs.
    """
    if any(not (config.ARTIFACTS_DIR / name).is_file() for name in _DATA_PREP_FILENAMES):
        return False

    manifest = json.loads((config.ARTIFACTS_DIR / "splits.json").read_text(encoding="utf-8"))["manifest"]
    recorded_hash = manifest.get("split_input_hash")
    if recorded_hash is None:
        return False

    return recorded_hash == split.hash_split_inputs(split.compute_split_inputs())


def ensure_data_artifacts() -> None:
    """Restores artifacts/ from the Kaggle-persisted bundle when on Kaggle,
    then regenerates dedupe/split/inventory/mapping from scratch unless
    what's now on disk is both complete and valid for the current split
    inputs.
    """
    if config.IS_KAGGLE:
        kaggle_persist_artifacts.restore_data_artifacts()

    if _local_data_prep_artifacts_valid():
        print(
            "[prepare_artifacts] artifacts/ already has complete, valid dedupe/split/"
            "inventory/mapping outputs for the current split inputs -- skipping regeneration."
        )
        return

    print(
        "[prepare_artifacts] artifacts/ is missing dedupe/split/inventory/mapping outputs, "
        "or they're stale for the current split inputs -- regenerating from scratch "
        "(dedupe.py -> split_report.py -> inventory.py -> mapping_report.py, ~20 minutes)."
    )
    dedupe.main()
    split_report.main()
    inventory.main()
    mapping_report.main()


def main() -> None:
    ensure_data_artifacts()


if __name__ == "__main__":
    main()
