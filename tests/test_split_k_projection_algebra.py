import torch

from experiments.split_k_projection_algebra import (
    alternating_partition,
    hessian_rank_witness,
    information_rank_witness,
    projection_ledger,
    run_self_test,
    split_int8_projection,
    split_swiglu,
)


def test_int8_sum_and_contrast_are_exact() -> None:
    generator = torch.Generator().manual_seed(19)
    inputs = torch.randint(-127, 128, (5, 32), dtype=torch.int8, generator=generator)
    weights = torch.randint(-127, 128, (9, 32), dtype=torch.int8, generator=generator)
    partition = alternating_partition(32)

    _, _, summed, contrast = split_int8_projection(inputs, weights, partition)

    assert torch.equal(summed, inputs.to(torch.int32) @ weights.to(torch.int32).T)
    assert torch.equal(
        contrast,
        (inputs.to(torch.int32) * partition) @ weights.to(torch.int32).T,
    )


def test_zero_coefficient_is_exact_swiglu_endpoint() -> None:
    generator = torch.Generator().manual_seed(23)
    inputs = torch.randn(3, 8, dtype=torch.float64, generator=generator)
    gate = torch.randn(6, 8, dtype=torch.float64, generator=generator)
    up = torch.randn(6, 8, dtype=torch.float64, generator=generator)
    coefficient = torch.zeros((), dtype=torch.float64, requires_grad=True)

    baseline, candidate = split_swiglu(
        inputs,
        gate,
        up,
        alternating_partition(8).to(torch.float64),
        coefficient,
    )

    assert torch.equal(candidate, baseline)
    candidate.sum().backward()
    assert coefficient.grad is not None
    assert coefficient.grad.abs().item() > 0


def test_sum_contrast_can_recover_twice_the_linear_rank() -> None:
    witness = information_rank_witness()
    assert witness == {"ordinary_rank": 2, "sum_contrast_rank": 4, "input_width": 4}


def test_minimum_correction_increases_quadratic_rank() -> None:
    witness = hessian_rank_witness()
    assert witness["ordinary_quadratic_rank"] == 2
    assert witness["sum_contrast_quadratic_rank"] == 4


def test_native_instruction_and_payload_ledgers_match() -> None:
    ledger = projection_ledger(14336, 4096, 256, tile_n=128)
    assert ledger.dense_wgmma_instructions == ledger.split_wgmma_instructions
    assert ledger.weight_payload_bytes == 14336 * 4096
    assert ledger.input_payload_bytes == 256 * 4096
    assert ledger.split_accumulator_registers_per_thread == 2 * ledger.baseline_accumulator_registers_per_thread


def test_self_test_passes() -> None:
    result = run_self_test()
    assert result["exact_sum"]
    assert result["exact_contrast"]
    assert result["exact_zero_endpoint"]

