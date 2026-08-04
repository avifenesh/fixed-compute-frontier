from __future__ import annotations

import inspect

from experiments import raw_prose_equality_plane_t11 as experiment


def test_writer_cannot_accept_latent_world() -> None:
    assert tuple(inspect.signature(experiment.compile_equality_plane).parameters) == (
        "model",
        "compiled",
        "layout",
    )


def test_triangular_contract_is_minimal_and_complete() -> None:
    assert experiment.BLOCK1_PROGRAM_CHANNELS == 64
    assert experiment.BLOCK2_PROGRAM_CHANNELS == 8
    assert len(experiment.BLOCK0_VALUE_COORDS) == 38
    assert experiment.PROTECTED_OUTPUT_DIMS == (42, 43, 44, 45, 46, 51, 52, 53, 54)
    assert experiment.OUTPUT_SCALE == 2.0


def test_t11_reuses_frozen_t10_data_and_query_contract() -> None:
    assert experiment.TRAIN_FILE == experiment.base.TRAIN_FILE
    assert experiment.VALIDATION_FILE == experiment.base.VALIDATION_FILE
    assert experiment.direct_dataset is experiment.base.direct_dataset
    assert experiment.equality_dataset is experiment.base.equality_dataset
