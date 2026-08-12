"""Shared fixtures for the test suite.

Every test runs against a small synthetic PlantVillage-shaped image tree
(well under CLAUDE.md rule 10's 200-image local-smoke-test cap) — the real
54k-image dataset only ever exists on Colab (rule 10), so tests build their
own tiny fixture on disk and monkeypatch src.config to point at it, rather
than depending on real downloaded data.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from PIL import Image

from src import config
from src.data import dedupe, split, split_report
from src.models import build

# Fake classes, each with several distinct "physical leaves" (some
# represented by more than one near-duplicate photo) — enough groups per
# class (>= 8) that the >=3-groups-per-class feasibility check comfortably
# passes and per-class proportions are meaningful.
FAKE_CLASSES = ("FakeSpeciesA___disease_one", "FakeSpeciesB___healthy", "FakeSpeciesC___disease_two")
IMAGE_SIZE = (32, 32)


def _make_base_image(seed: int) -> Image.Image:
    # A distinct solid-ish color per seed, so phash treats different seeds
    # as different leaves but near-identical crops of the same seed as the
    # same leaf.
    color = ((seed * 37) % 256, (seed * 91) % 256, (seed * 53) % 256)
    img = Image.new("RGB", IMAGE_SIZE, color=color)
    for x in range(IMAGE_SIZE[0]):
        for y in range(IMAGE_SIZE[1]):
            if (x + y + seed) % 7 == 0:
                img.putpixel((x, y), (255, 255, 255))
    return img


@pytest.fixture(scope="module")
def synthetic_dataset(tmp_path_factory, monkeypatch_module_scoped):
    color_dir = tmp_path_factory.mktemp("plantvillage_color")
    artifacts_dir = tmp_path_factory.mktemp("artifacts")

    leaf_seed = 0
    total_images = 0
    for class_name in FAKE_CLASSES:
        class_dir = color_dir / class_name
        class_dir.mkdir(parents=True)
        # 30 distinct physical leaves per class: leaves 0-2 get 3 near-
        # duplicate photos each (a "cluster"), the rest are singletons. With
        # 30 groups per class, a handful of size-3 clusters are a small
        # enough fraction of the class total that achieved proportions can
        # land close to target — a class with only a few, lumpy groups
        # (as in an earlier, smaller version of this fixture) inherently
        # cannot; that's a property of the data, not a bug in split.py.
        for leaf_index in range(30):
            leaf_seed += 1
            base = _make_base_image(leaf_seed)
            num_photos = 3 if leaf_index < 3 else 1
            for photo_index in range(num_photos):
                img = base.copy()
                if photo_index > 0:
                    # A trivial near-duplicate perturbation (a couple of
                    # pixels changed) — small enough to stay within
                    # config.DEDUPE_HAMMING_THRESHOLD of the base photo.
                    img.putpixel((0, 0), (photo_index, photo_index, photo_index))
                img.save(class_dir / f"leaf{leaf_index}_photo{photo_index}.jpg")
                total_images += 1

    # One deliberate cross-class collision: the same image content saved
    # into two different classes.
    collision_img = _make_base_image(9999)
    collision_img.save(color_dir / FAKE_CLASSES[0] / "collision.jpg")
    collision_img.save(color_dir / FAKE_CLASSES[1] / "collision.jpg")
    total_images += 2

    monkeypatch_module_scoped.setattr(config, "PLANTVILLAGE_COLOR_DIR", color_dir)
    monkeypatch_module_scoped.setattr(config, "PLANTVILLAGE_CLASS_NAMES", FAKE_CLASSES)
    monkeypatch_module_scoped.setattr(config, "NUM_CLASSES", len(FAKE_CLASSES))
    monkeypatch_module_scoped.setattr(config, "ARTIFACTS_DIR", artifacts_dir)

    # split_report.py's manifest now includes split.compute_split_inputs(),
    # which reads PlantVillage's recorded provenance — DATASET_PROVENANCE_PATH
    # is precomputed at config.py's import time from the real ARTIFACTS_DIR,
    # so it needs its own monkeypatch rather than following the ARTIFACTS_DIR
    # patch above.
    provenance_path = artifacts_dir / "dataset_provenance.json"
    provenance_path.write_text(
        json.dumps(
            {
                "plantvillage": {
                    "source": "synthetic-test-fixture",
                    "observed_image_count": total_images,
                    "observed_class_count": len(FAKE_CLASSES),
                    "sha256_of_sorted_relative_paths": "synthetic-fixture-hash",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch_module_scoped.setattr(config, "DATASET_PROVENANCE_PATH", provenance_path)

    dedupe.main()
    split_report.main()

    image_to_group = split.load_image_groups()
    splits = split.build_split(image_to_group)

    return {
        "color_dir": color_dir,
        "artifacts_dir": artifacts_dir,
        "image_to_group": image_to_group,
        "splits": splits,
        "total_images": total_images,
    }


@pytest.fixture(scope="module")
def monkeypatch_module_scoped():
    """pytest's built-in `monkeypatch` fixture is function-scoped; this is
    the standard workaround for needing monkeypatching in a module-scoped
    fixture (building the synthetic dataset + running the real pipeline
    once per module, not once per test, keeps the suite fast).
    """
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()


EXPORT_MODEL_NAME = "MobileNetV2"


@pytest.fixture(scope="module")
def export_env(synthetic_dataset, tmp_path_factory, monkeypatch_module_scoped):
    """Shared by tests/test_export_to_tflite.py and
    tests/test_export_verify_tflite.py: a real (untrained) MobileNetV2
    checkpoint saved into a faked, module-scoped CHECKPOINT_DIR (exercising
    the real src.evaluate.inference.load_trained_model() path, not a
    shortcut around it).

    Deliberately does NOT touch TFLITE_OUTPUT_DIR/ANDROID_MODEL_METADATA_PATH/
    ANDROID_MODEL_TFLITE_PATH/RESULTS_JSON_PATH — those are mutated by
    src.export.to_tflite.export() itself, so sharing them across tests in
    one module-scoped fixture would make every test after the first see
    whatever the previous export() call already wrote (deployed assets,
    an "export" section already in results.json). Tests that call export()
    for real must set those four up fresh per-test (see
    tests/test_export_to_tflite.py's `fresh_export_targets` fixture) —
    CHECKPOINT_DIR is safe to share since export() only ever reads from it.
    """
    checkpoint_dir = tmp_path_factory.mktemp("checkpoints")
    monkeypatch_module_scoped.setattr(config, "CHECKPOINT_DIR", checkpoint_dir)

    model = build.build_model(EXPORT_MODEL_NAME)
    model_checkpoint_dir = checkpoint_dir / EXPORT_MODEL_NAME
    model_checkpoint_dir.mkdir(parents=True)
    weights_path = model_checkpoint_dir / "phase1.weights.h5"
    model.save_weights(weights_path)

    return {
        "model_name": EXPORT_MODEL_NAME,
        "model": model,
        "checkpoint_dir": checkpoint_dir,
        "weights_path": weights_path,
        "param_count": int(model.count_params()),
        "checkpoint_file_size_bytes": weights_path.stat().st_size,
        "checkpoint_mtime": weights_path.stat().st_mtime,
    }
