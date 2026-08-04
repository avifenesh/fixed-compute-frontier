from __future__ import annotations

import torch
import pytest

from experiments import heldout_record_oracle_t23a as t23


def test_nearest_q16_uses_only_frozen_levels() -> None:
    values = torch.linspace(-4.0, 4.0, 10_001).reshape(-1, 1)
    projected = t23.nearest_q16(values)
    assert len(torch.unique(projected)) == 16
    assert torch.equal(projected, t23.nearest_q16(projected))
    assert torch.equal(
        projected,
        t23.nearest_q16(projected.to(torch.bfloat16).float()),
    )


def test_gain_retention_separates_projection_and_cross_reader_cases() -> None:
    assert t23.gain_retention(10.0, 5.0, 5.5) == 0.9
    assert t23.cross_reader_gain_retention(10.0, 5.0, 8.0, 4.4) == pytest.approx(0.9)


def test_row_adam_updates_only_selected_rows() -> None:
    parameters = torch.zeros(3, 2, requires_grad=True)
    loss = (parameters[1] - 1.0).square().sum()
    loss.backward()
    first = torch.zeros_like(parameters)
    second = torch.zeros_like(parameters)
    counts = torch.zeros(3, dtype=torch.long)
    with torch.no_grad():
        t23.row_adam_step(
            parameters,
            first,
            second,
            counts,
            torch.tensor([1]),
        )
    assert counts.tolist() == [0, 1, 0]
    assert torch.equal(parameters[0], torch.zeros(2))
    assert torch.equal(parameters[2], torch.zeros(2))
    assert bool((parameters[1] > 0).all())

    row_one_after_first_step = parameters[1].clone()
    loss = (parameters[2] - 1.0).square().sum()
    loss.backward()
    with torch.no_grad():
        t23.row_adam_step(
            parameters,
            first,
            second,
            counts,
            torch.tensor([2]),
        )
    assert torch.equal(parameters[1], row_one_after_first_step)
    assert counts.tolist() == [0, 1, 1]


def test_frozen_compile_schedule_is_balanced() -> None:
    documents = 207
    batch = t23.t22.BATCH
    updates = t23.COMPILE_ROUNDS * ((documents + batch - 1) // batch)
    presentations = t23.COMPILE_ROUNDS * documents
    views = presentations * t23.COMPILE_VIEWS_PER_PRESENTATION
    assert updates == 520
    assert presentations == 8_280
    assert views == 16_560


def test_preregistration_hash_is_sealed() -> None:
    assert t23.sha256_file(t23.PREREGISTRATION) == t23.PREREGISTRATION_SHA256
    assert (
        t23.sha256_file(t23.INTEGRITY_AMENDMENT)
        == t23.INTEGRITY_AMENDMENT_SHA256
    )
    assert t23.sha256_file(t23.T22_SOURCE) == t23.T22_SOURCE_SHA256
    assert t23.sha256_file(t23.T22_RESULT) == t23.T22_RESULT_SHA256


def test_random_control_hash_bytes_are_sealed() -> None:
    assert t23.hash_bytes("doc-7", 0).hex() == (
        "af1f0e87eb63a8a59af573f632d33225ba7ea5cfb43dff2ce9f45caae1005cce"
    )
