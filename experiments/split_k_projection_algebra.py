"""Algebra and resource ledger for contrast-preserving split-K projections."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class ProjectionLedger:
    output_width: int
    input_width: int
    rows: int
    tile_n: int
    dense_wgmma_instructions: int
    split_wgmma_instructions: int
    weight_payload_bytes: int
    input_payload_bytes: int
    baseline_accumulator_registers_per_thread: int
    split_accumulator_registers_per_thread: int
    split_epilogue_integer_add_sub_per_output: int


def alternating_partition(width: int, *, device: torch.device | None = None) -> torch.Tensor:
    if width <= 0 or width % 2:
        raise ValueError("width must be a positive even integer")
    partition = torch.ones(width, dtype=torch.int32, device=device)
    partition[1::2] = -1
    return partition


def split_int8_projection(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    partition: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return first half, second half, exact sum, and partition contrast."""

    if inputs.dtype != torch.int8 or weights.dtype != torch.int8:
        raise TypeError("inputs and weights must be int8")
    if inputs.shape[-1] != weights.shape[-1] or partition.shape != (weights.shape[-1],):
        raise ValueError("incompatible input, weight, or partition shape")
    if not torch.all((partition == 1) | (partition == -1)):
        raise ValueError("partition entries must be +/-1")

    positive = partition == 1
    negative = ~positive
    first = inputs[..., positive].to(torch.int32) @ weights[:, positive].to(torch.int32).T
    second = inputs[..., negative].to(torch.int32) @ weights[:, negative].to(torch.int32).T
    return first, second, first + second, first - second


def split_swiglu(
    inputs: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    partition: torch.Tensor,
    coefficient: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Baseline SwiGLU plus the minimum non-absorbable split correction."""

    positive = partition > 0
    negative = ~positive

    gate_first = inputs[..., positive] @ gate_weight[:, positive].T
    gate_second = inputs[..., negative] @ gate_weight[:, negative].T
    up_first = inputs[..., positive] @ up_weight[:, positive].T
    up_second = inputs[..., negative] @ up_weight[:, negative].T

    gate_sum = gate_first + gate_second
    up_sum = up_first + up_second
    gate_contrast = gate_first - gate_second
    up_contrast = up_first - up_second

    baseline = functional.silu(gate_sum) * up_sum
    candidate = torch.addcmul(
        baseline,
        gate_contrast,
        up_contrast,
        value=float(coefficient.detach()),
    )
    if coefficient.requires_grad:
        candidate = baseline + coefficient * gate_contrast * up_contrast
    return baseline, candidate


def projection_ledger(
    output_width: int,
    input_width: int,
    rows: int,
    *,
    tile_n: int = 128,
) -> ProjectionLedger:
    if input_width % 32:
        raise ValueError("input_width must be divisible by the S8 WGMMA K=32 extent")
    if tile_n % 8 or not 8 <= tile_n <= 256:
        raise ValueError("tile_n must be an H100 WGMMA-supported multiple of eight")

    instructions = (
        math.ceil(output_width / 64)
        * math.ceil(rows / tile_n)
        * (input_width // 32)
    )
    accumulator_elements = 64 * tile_n
    accumulator_registers = math.ceil(accumulator_elements / 128)
    return ProjectionLedger(
        output_width=output_width,
        input_width=input_width,
        rows=rows,
        tile_n=tile_n,
        dense_wgmma_instructions=instructions,
        split_wgmma_instructions=instructions,
        weight_payload_bytes=output_width * input_width,
        input_payload_bytes=rows * input_width,
        baseline_accumulator_registers_per_thread=accumulator_registers,
        split_accumulator_registers_per_thread=2 * accumulator_registers,
        split_epilogue_integer_add_sub_per_output=2,
    )


def information_rank_witness() -> dict[str, int]:
    weights = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    partition = alternating_partition(4).to(torch.float64)
    ordinary = weights
    lifted = torch.cat((weights, weights * partition), dim=0)
    return {
        "ordinary_rank": int(torch.linalg.matrix_rank(ordinary).item()),
        "sum_contrast_rank": int(torch.linalg.matrix_rank(lifted).item()),
        "input_width": weights.shape[1],
    }


def hessian_rank_witness() -> dict[str, int]:
    dtype = torch.float64
    partition = alternating_partition(4).to(dtype)
    gate = torch.tensor([1.0, 2.0, 3.0, 5.0], dtype=dtype)
    up = torch.tensor([7.0, 11.0, 13.0, 17.0], dtype=dtype)

    def ordinary(inputs: torch.Tensor) -> torch.Tensor:
        return functional.silu(gate @ inputs) * (up @ inputs)

    def lifted(inputs: torch.Tensor) -> torch.Tensor:
        gate_sum = gate @ inputs
        up_sum = up @ inputs
        gate_contrast = (partition * gate) @ inputs
        up_contrast = (partition * up) @ inputs
        return functional.silu(gate_sum) * up_sum + gate_contrast * up_contrast

    origin = torch.zeros(4, dtype=dtype)
    ordinary_hessian = torch.autograd.functional.hessian(ordinary, origin)
    lifted_hessian = torch.autograd.functional.hessian(lifted, origin)
    return {
        "ordinary_quadratic_rank": int(torch.linalg.matrix_rank(ordinary_hessian).item()),
        "sum_contrast_quadratic_rank": int(torch.linalg.matrix_rank(lifted_hessian).item()),
        "input_width": origin.numel(),
    }


def run_self_test() -> dict[str, object]:
    generator = torch.Generator().manual_seed(20260728)
    inputs = torch.randint(-127, 128, (7, 64), dtype=torch.int8, generator=generator)
    weights = torch.randint(-127, 128, (32, 64), dtype=torch.int8, generator=generator)
    partition = alternating_partition(64)
    first, second, summed, contrast = split_int8_projection(inputs, weights, partition)
    direct = inputs.to(torch.int32) @ weights.to(torch.int32).T
    signed_direct = (inputs.to(torch.int32) * partition) @ weights.to(torch.int32).T

    gate = torch.randn(12, 16, dtype=torch.float64, generator=generator)
    up = torch.randn(12, 16, dtype=torch.float64, generator=generator)
    activations = torch.randn(5, 16, dtype=torch.float64, generator=generator)
    zero = torch.zeros((), dtype=torch.float64, requires_grad=True)
    baseline, candidate = split_swiglu(
        activations,
        gate,
        up,
        alternating_partition(16).to(torch.float64),
        zero,
    )

    result = {
        "exact_sum": bool(torch.equal(summed, direct)),
        "exact_contrast": bool(torch.equal(contrast, signed_direct)),
        "exact_zero_endpoint": bool(torch.equal(candidate, baseline)),
        "first_second_reconstruction": bool(torch.equal(first + second, direct)),
        "information_rank": information_rank_witness(),
        "hessian_rank": hessian_rank_witness(),
        "ledger": asdict(projection_ledger(14336, 4096, 256)),
    }
    if not all(
        result[key]
        for key in (
            "exact_sum",
            "exact_contrast",
            "exact_zero_endpoint",
            "first_second_reconstruction",
        )
    ):
        raise AssertionError(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    arguments = parser.parse_args()
    result = run_self_test()
    rendered = json.dumps(result, indent=2) + "\n"
    if arguments.output:
        with open(arguments.output, "w", encoding="utf-8") as handle:
            handle.write(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()

