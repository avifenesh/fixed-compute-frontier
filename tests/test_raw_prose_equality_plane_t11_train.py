from __future__ import annotations

from experiments import raw_prose_equality_plane_t10_train as t10_train
from experiments import raw_prose_equality_plane_t11 as t11
from experiments import raw_prose_equality_plane_t11_train as train


def test_training_harness_is_bound_to_t11_without_protocol_drift() -> None:
    assert t10_train.t10 is t11
    assert train.CHECKPOINTS_1X == (250, 500, 1_000)
    assert train.STEPS_2X == 2_000
    assert train.MODEL_SEED == 6_401
    assert train.MUON_LEARNING_RATE == 0.005
    assert train.REPRESENTATION == t11.OUTPUT
