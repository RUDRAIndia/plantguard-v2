"""Tests for src/models/build.py against every real config.CANDIDATE_MODELS
backbone (real ImageNet weights — architecture correctness doesn't depend
on the smoke-test image cap in CLAUDE.md rule 10, which is about dataset
size, not model construction).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from tensorflow import keras

from src import config
from src.models import build


def _count_trainable_params(model: keras.Model) -> int:
    return sum(int(np.prod(v.shape)) for v in model.trainable_weights)


@pytest.mark.parametrize("model_name", config.CANDIDATE_MODELS)
def test_build_model_output_shape(model_name):
    model = build.build_model(model_name)
    assert model.output_shape == (None, config.NUM_CLASSES)


@pytest.mark.parametrize("model_name", config.CANDIDATE_MODELS)
def test_freeze_and_unfreeze_change_trainable_param_count(model_name):
    model = build.build_model(model_name)
    default_count = _count_trainable_params(model)

    build.freeze_backbone(model)
    frozen_count = _count_trainable_params(model)

    build.unfreeze_top_blocks(model, model_name)
    unfrozen_top_count = _count_trainable_params(model)

    # Frozen: only the Dropout/Dense head trains. Unfrozen-top: head +
    # config.UNFREEZE_BLOCKS[model_name] backbone blocks. Fully trainable
    # (the model's default state): every backbone layer too.
    assert frozen_count < unfrozen_top_count < default_count


def test_get_backbone_fails_loudly_on_a_non_build_model():
    plain_model = keras.Sequential([keras.layers.Dense(4, input_shape=(4,))])
    with pytest.raises(RuntimeError, match="Expected exactly one nested backbone"):
        build.get_backbone(plain_model)


def test_unfreeze_top_blocks_rejects_unfreeze_count_over_available_blocks(monkeypatch):
    monkeypatch.setitem(config.UNFREEZE_BLOCKS, "MobileNetV2", 999)
    model = build.build_model("MobileNetV2")
    with pytest.raises(ValueError, match="exceeds the"):
        build.unfreeze_top_blocks(model, "MobileNetV2")
