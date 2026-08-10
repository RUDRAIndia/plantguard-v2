"""Builds the Keras 3 model architecture used for training. Architecture
choice and hyperparameters are read from src/config.py only.

Every candidate backbone is a keras.applications ImageNet-pretrained model
with its classifier head removed (include_top=False, pooling="avg") and
replaced with a Dropout + Dense(config.NUM_CLASSES, softmax) head sized for
this project's 38 classes. src/data/pipeline.py already applies each
backbone's own preprocess_input before batching (see that module's
docstring), so the backbone here is fed already-normalized input directly —
build_model never adds a second rescale.

Freezing and unfreezing operate on the backbone's own nested layer list
(each keras.applications model, built with include_top=False, is itself a
functional keras.Model that shows up as exactly one sub-layer of the outer
classification model), not the outer model's 3-layer list (Input, backbone,
head), so per-layer freeze state actually reaches the backbone's internals.
"""

import re
import sys
from pathlib import Path

from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config  # noqa: E402

# keras.applications constructors, one per config.CANDIDATE_MODELS entry.
# Kept as live callables here (unlike config.PREPROCESSING_ENTRYPOINTS'
# string-based lookup) since this module already requires TensorFlow to do
# anything useful, so there's no importability benefit to indirection.
_BACKBONE_CONSTRUCTORS = {
    "MobileNetV2": keras.applications.MobileNetV2,
    "MobileNetV3Small": keras.applications.MobileNetV3Small,
    "MobileNetV3Large": keras.applications.MobileNetV3Large,
    "EfficientNetB0": keras.applications.EfficientNetB0,
    "EfficientNetV2B0": keras.applications.EfficientNetV2B0,
}
assert set(_BACKBONE_CONSTRUCTORS) == set(config.CANDIDATE_MODELS), (
    "_BACKBONE_CONSTRUCTORS must have exactly one entry per "
    "config.CANDIDATE_MODELS entry."
)


def _resolve_constructor(model_name: str):
    if model_name not in _BACKBONE_CONSTRUCTORS:
        raise ValueError(
            f"Unknown model '{model_name}'. Known models: "
            f"{sorted(_BACKBONE_CONSTRUCTORS)} (config.CANDIDATE_MODELS)."
        )
    return _BACKBONE_CONSTRUCTORS[model_name]


def build_model(model_name: str) -> keras.Model:
    """Builds `model_name` (must be one of config.CANDIDATE_MODELS): an
    ImageNet-pretrained backbone with a fresh Dropout(config.DROPOUT_RATE) +
    Dense(config.NUM_CLASSES, softmax) classifier head. The backbone is
    trainable by default (Keras' normal state for a freshly built model) —
    callers doing phase-1 transfer learning must call freeze_backbone()
    themselves before compiling.
    """
    constructor = _resolve_constructor(model_name)
    backbone = constructor(
        include_top=False,
        weights="imagenet",
        input_shape=config.IMAGE_SHAPE,
        pooling="avg",
    )

    inputs = keras.Input(shape=config.IMAGE_SHAPE, name="image")
    x = backbone(inputs)
    x = keras.layers.Dropout(config.DROPOUT_RATE, name="head_dropout")(x)
    outputs = keras.layers.Dense(config.NUM_CLASSES, activation="softmax", name="predictions")(x)

    return keras.Model(inputs, outputs, name=f"{model_name}_plantguard")


def get_backbone(model: keras.Model) -> keras.Model:
    """Returns the single nested keras.Model sub-layer build_model() wires
    up as the backbone. Fails loudly (CLAUDE.md rule 1) rather than guessing
    if `model` wasn't built by build_model() above (0 matches) or has been
    restructured to hold more than one nested sub-model (>1 matches,
    ambiguous). Public: src/models/finetune.py also needs to locate the
    backbone (to split its variables from the head's for phase 2's two
    optimizers), not just this module's own freeze/unfreeze functions.
    """
    candidates = [layer for layer in model.layers if isinstance(layer, keras.Model)]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one nested backbone sub-model in '{model.name}', "
            f"found {len(candidates)}. This function only works on models built "
            "by src/models/build.py's build_model()."
        )
    return candidates[0]


def freeze_backbone(model: keras.Model) -> None:
    """Phase 1: freezes every backbone layer (including BatchNorm running
    statistics), so only the Dropout/Dense head trains. Must be called
    before model.compile() — Keras only picks up a trainable-state change
    at the next compile.
    """
    get_backbone(model).trainable = False


def unfreeze_top_blocks(model: keras.Model, model_name: str) -> None:
    """Phase 2: unfreezes the last config.UNFREEZE_BLOCKS[model_name] blocks
    of `model_name`'s backbone (plus any trailing head-conv layers after the
    last numbered block, e.g. EfficientNet's top_conv/top_bn — those belong
    to the same "top of the backbone" fine-tuning region), leaving every
    earlier layer frozen. Block boundaries are identified via
    config.BACKBONE_BLOCK_NAME_PATTERNS[model_name] — never a hardcoded
    layer-index cut, which would silently point at the wrong depth if a
    keras.applications architecture ever changes its layer count.

    Raises if config.UNFREEZE_BLOCKS[model_name] exceeds the number of
    distinct blocks actually found (CLAUDE.md rule 1: fail loudly rather
    than silently unfreezing "as many as there are").
    """
    backbone = get_backbone(model)
    backbone.trainable = True  # unlock the container; per-layer flags below decide the rest

    pattern = re.compile(config.BACKBONE_BLOCK_NAME_PATTERNS[model_name])
    layer_blocks = []
    for layer in backbone.layers:
        match = pattern.match(layer.name)
        layer_blocks.append(int(match.group(1)) if match else None)

    found_blocks = sorted({b for b in layer_blocks if b is not None})
    if not found_blocks:
        raise RuntimeError(
            f"No layer in '{model_name}' backbone ('{backbone.name}') matched "
            f"block pattern '{pattern.pattern}' — check "
            "config.BACKBONE_BLOCK_NAME_PATTERNS against the backbone's actual "
            "layer names."
        )

    n = config.UNFREEZE_BLOCKS[model_name]
    if n > len(found_blocks):
        raise ValueError(
            f"config.UNFREEZE_BLOCKS['{model_name}'] = {n} exceeds the "
            f"{len(found_blocks)} distinct block(s) found ({found_blocks})."
        )
    unfreeze_from = found_blocks[-n]

    first_unfrozen_index = next(
        i for i, b in enumerate(layer_blocks) if b is not None and b >= unfreeze_from
    )
    for i, layer in enumerate(backbone.layers):
        layer.trainable = i >= first_unfrozen_index
