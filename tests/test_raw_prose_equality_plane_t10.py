from __future__ import annotations

import inspect

from experiments import raw_prose_equality_plane_t10 as experiment


def test_program_coordinate_accounting() -> None:
    assert experiment.MEMORY_COORDS == 32
    assert experiment.PROGRAM_DIMS == 55
    assert experiment.PROGRAM_CHANNELS == 64
    assert experiment.PRODUCT_START + experiment.VALUE_BITS == experiment.PROGRAM_DIMS


def test_writer_cannot_accept_latent_world() -> None:
    assert tuple(inspect.signature(experiment.compile_equality_plane).parameters) == (
        "model",
        "compiled",
        "layout",
    )


def test_bipolar_codes_are_unique() -> None:
    value_codes = {
        tuple(experiment.bipolar_code(index, experiment.VALUE_BITS).tolist())
        for index in range(experiment.VALUES)
    }
    relation_codes = {
        tuple(experiment.bipolar_code(index, experiment.RELATION_BITS).tolist())
        for index in range(experiment.RELATIONS)
    }
    assert len(value_codes) == experiment.VALUES
    assert len(relation_codes) == experiment.RELATIONS


def test_natural_data_contract_matches_model_import_chain() -> None:
    assert experiment.TRAIN_FILE.parent.resolve() == experiment.natural_data.OUTPUT_DIR.resolve()
    assert experiment.TRAIN_FILE.name == "train.uint16.bin"
    assert experiment.VALIDATION_FILE.name == "validation.uint16.bin"
