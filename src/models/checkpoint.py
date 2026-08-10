"""Per-epoch checkpointing for src/train.py's two-phase loop: saves weights
and a small resume-state JSON after every epoch, so a Colab disconnection
never costs more than the epoch in progress (CLAUDE.md: checkpoint every
epoch to Drive; config.CHECKPOINT_DIR resolves to Drive on Colab).
"""

import json
import sys
from pathlib import Path

from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class EpochCheckpointCallback(keras.callbacks.Callback):
    """Must be the LAST callback in fit()'s callbacks list (Keras runs
    callbacks in list order): its on_train_end fires after EarlyStopping's
    own, which is when restore_best_weights may have swapped in an earlier
    epoch's weights — re-saving there captures that final state rather than
    freezing the on-disk checkpoint at the last epoch trained.
    """

    def __init__(self, phase: int, checkpoint_dir: Path):
        super().__init__()
        self.phase = phase
        self.checkpoint_dir = checkpoint_dir
        self._last_epoch = None

    def weights_path(self) -> Path:
        return self.checkpoint_dir / f"phase{self.phase}.weights.h5"

    def _save(self, epoch: int) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        weights_path = self.weights_path()
        self.model.save_weights(weights_path)
        state = {"phase": self.phase, "epoch": epoch, "weights_path": str(weights_path)}
        (self.checkpoint_dir / "state.json").write_text(
            json.dumps(state, indent=2), encoding="utf-8"
        )

    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        self._last_epoch = epoch
        self._save(epoch)

    def on_train_end(self, logs: dict = None) -> None:
        if self._last_epoch is not None:
            self._save(self._last_epoch)


def read_state(checkpoint_dir: Path) -> dict:
    """Reads checkpoint_dir/state.json, or None if this model has never
    checkpointed (first run, nothing to resume from).
    """
    state_path = checkpoint_dir / "state.json"
    if not state_path.is_file():
        return None
    return json.loads(state_path.read_text(encoding="utf-8"))
