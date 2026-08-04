from __future__ import annotations

import inspect

import torch

from experiments import hotpot_semantic_address_t12 as experiment


def test_shared_base_trainer_cannot_accept_qa_or_labels() -> None:
    assert tuple(inspect.signature(experiment.train_shared_base).parameters) == (
        "natural",
        "documents",
        "validation",
        "device",
    )


def test_frozen_training_ledger() -> None:
    assert experiment.STEPS == 20_000
    assert experiment.BATCH_SIZE == 16
    assert experiment.CONTEXT == 128
    assert experiment.HOT_INTERVAL == 20
    assert experiment.STEPS * experiment.BATCH_SIZE * experiment.CONTEXT == 40_960_000
    assert (experiment.STEPS // experiment.HOT_INTERVAL) * experiment.BATCH_SIZE * experiment.CONTEXT == 2_048_000


def test_deterministic_codes_are_stable_and_bipolar() -> None:
    first = experiment.deterministic_token_code(123)
    second = experiment.deterministic_token_code(123)
    other = experiment.deterministic_token_code(124)
    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    assert len(first) == experiment.ADDRESS_DIMS
    assert set(first.tolist()) == {-1.0, 1.0}


def test_schedule_has_frozen_endpoints() -> None:
    assert experiment.schedule_fraction(1) == 1 / experiment.WARMUP_STEPS
    assert experiment.schedule_fraction(experiment.WARMUP_STEPS) == 1.0
    assert abs(experiment.schedule_fraction(experiment.STEPS) - 0.1) < 1e-12
