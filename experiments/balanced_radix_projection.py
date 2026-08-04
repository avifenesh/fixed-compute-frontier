#!/usr/bin/env python3
"""Exact two-for-one integer projection algebra.

This module deliberately stops short of a performance claim.  It proves the
integer identity, states the per-tile decoding requirement, and publishes the
FFN physical ledger that a kernel and a learning experiment must satisfy.

For one K-element tile, let ``a`` and two independent weight streams be
ternary.  Packing the weights as ``p = w0 + B*w1`` gives

    dot(a, p) = dot(a, w0) + B*dot(a, w1).

With K=32, every low digit is in [-32, 32].  B=65 therefore gives a unique
balanced-radix representation while keeping p in signed INT8 [-66, 66].
The packed INT32 accumulator must be decoded after each K=32 tile.  Accumulating
multiple tiles before decoding invalidates the digit bound.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


TILE_K = 32
RADIX = 2 * TILE_K + 1
BINARY_RADIX = 2 * TILE_K
SIGNED_INT8_MIN = -128
SIGNED_INT8_MAX = 127


def _require_ternary(name: str, value: torch.Tensor) -> None:
    if value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError(f"{name} must be an integer tensor")
    if not bool(torch.all((value >= -1) & (value <= 1))):
        raise ValueError(f"{name} must contain only -1, 0, or 1")


def _require_binary(name: str, value: torch.Tensor) -> None:
    _require_ternary(name, value)
    if not bool(torch.all(value.abs() == 1)):
        raise ValueError(f"{name} must contain only -1 or 1")


def pack_two_ternary_weights(
    weight_low: torch.Tensor,
    weight_high: torch.Tensor,
    radix: int = RADIX,
) -> torch.Tensor:
    """Pack two independent ternary weight matrices into one signed INT8 matrix."""

    if weight_low.shape != weight_high.shape:
        raise ValueError("weight streams must have identical shapes")
    _require_ternary("weight_low", weight_low)
    _require_ternary("weight_high", weight_high)
    packed = weight_low.to(torch.int16) + radix * weight_high.to(torch.int16)
    if int(packed.min()) < SIGNED_INT8_MIN or int(packed.max()) > SIGNED_INT8_MAX:
        raise OverflowError("packed weights do not fit signed INT8")
    return packed.to(torch.int8)


def decode_balanced_tile(
    packed_accumulator: torch.Tensor,
    radix: int = RADIX,
    low_digit_bound: int = TILE_K,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode one balanced-radix INT32 tile accumulator exactly.

    Python/PyTorch floor division makes the signed formula compact:

        high = floor((packed + bound) / radix)
        low  = packed - radix * high

    It is exact whenever ``radix == 2*bound + 1`` and the true low digit lies
    in ``[-bound, bound]``.
    """

    if radix != 2 * low_digit_bound + 1:
        raise ValueError("radix must equal 2*low_digit_bound + 1")
    packed = packed_accumulator.to(torch.int64)
    high = torch.div(packed + low_digit_bound, radix, rounding_mode="floor")
    low = packed - radix * high
    if not bool(torch.all((low >= -low_digit_bound) & (low <= low_digit_bound))):
        raise AssertionError("decoded low digit escaped its promised range")
    return low.to(torch.int32), high.to(torch.int32)


def pack_two_binary_weights(
    weight_low: torch.Tensor,
    weight_high: torch.Tensor,
) -> torch.Tensor:
    """Pack binary streams with radix 64, using their dot-product parity invariant."""

    if weight_low.shape != weight_high.shape:
        raise ValueError("weight streams must have identical shapes")
    _require_binary("weight_low", weight_low)
    _require_binary("weight_high", weight_high)
    return (
        weight_low.to(torch.int16) + BINARY_RADIX * weight_high.to(torch.int16)
    ).to(torch.int8)


def decode_binary_tile(packed_accumulator: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode radix-64 binary-weight dots using signed ties-to-even rounding.

    Both logical dots have the parity of the count of nonzero activations.
    Therefore, at the only touching points (low digit +/-32), the true high
    digit is even.  Nearest-even decoding is exact.  The explicit integer
    implementation maps to abs, shift, mask, comparisons, and adds; it needs no
    integer division.
    """

    packed = packed_accumulator.to(torch.int64)
    magnitude = packed.abs()
    quotient = torch.bitwise_right_shift(magnitude, 6)
    remainder = torch.bitwise_and(magnitude, 63)
    round_up = (remainder > 32) | ((remainder == 32) & (quotient % 2 == 1))
    rounded_magnitude = quotient + round_up.to(quotient.dtype)
    high = torch.where(packed < 0, -rounded_magnitude, rounded_magnitude)
    low = packed - BINARY_RADIX * high
    if not bool(torch.all((low >= -TILE_K) & (low <= TILE_K))):
        raise AssertionError("decoded binary low digit escaped K=32 range")
    return low.to(torch.int32), high.to(torch.int32)


def packed_two_projection(
    activation: torch.Tensor,
    weight_low: torch.Tensor,
    weight_high: torch.Tensor,
    tile_k: int = TILE_K,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference packed projection with mandatory decode at every K tile.

    Shapes use the PyTorch linear convention: activation ``[..., K]`` and
    weights ``[N, K]``.  The implementation is an algebra oracle, not a fast
    kernel.
    """

    _require_ternary("activation", activation)
    if activation.shape[-1] != weight_low.shape[-1]:
        raise ValueError("activation and weight K dimensions differ")
    if activation.shape[-1] % tile_k:
        raise ValueError("K must be divisible by tile_k")
    if tile_k != TILE_K:
        raise ValueError("the signed-INT8 construction is frozen to K=32 tiles")
    packed_weight = pack_two_ternary_weights(weight_low, weight_high)
    flat = activation.reshape(-1, activation.shape[-1]).to(torch.int32)
    low_total = torch.zeros(
        flat.shape[0], weight_low.shape[0], dtype=torch.int32, device=flat.device
    )
    high_total = torch.zeros_like(low_total)
    for start in range(0, flat.shape[-1], tile_k):
        stop = start + tile_k
        partial = flat[:, start:stop] @ packed_weight[:, start:stop].to(torch.int32).T
        low, high = decode_balanced_tile(partial)
        low_total += low
        high_total += high
    output_shape = (*activation.shape[:-1], weight_low.shape[0])
    return low_total.reshape(output_shape), high_total.reshape(output_shape)


def packed_two_binary_projection(
    activation: torch.Tensor,
    weight_low: torch.Tensor,
    weight_high: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Radix-64 reference oracle for ternary activations and binary weights."""

    _require_ternary("activation", activation)
    if activation.shape[-1] != weight_low.shape[-1]:
        raise ValueError("activation and weight K dimensions differ")
    if activation.shape[-1] % TILE_K:
        raise ValueError("K must be divisible by 32")
    packed_weight = pack_two_binary_weights(weight_low, weight_high)
    flat = activation.reshape(-1, activation.shape[-1]).to(torch.int32)
    low_total = torch.zeros(
        flat.shape[0], weight_low.shape[0], dtype=torch.int32, device=flat.device
    )
    high_total = torch.zeros_like(low_total)
    for start in range(0, flat.shape[-1], TILE_K):
        stop = start + TILE_K
        partial = flat[:, start:stop] @ packed_weight[:, start:stop].to(torch.int32).T
        low, high = decode_binary_tile(partial)
        low_total += low
        high_total += high
    output_shape = (*activation.shape[:-1], weight_low.shape[0])
    return low_total.reshape(output_shape), high_total.reshape(output_shape)


@dataclass(frozen=True)
class FfnPhysicalLedger:
    hidden_size: int
    baseline_width: int
    candidate_width: int
    baseline_resident_weight_bytes: int
    candidate_resident_weight_bytes: int
    baseline_int8_scalar_products: int
    candidate_int8_scalar_products: int
    baseline_logical_projection_outputs: int
    candidate_logical_projection_outputs: int
    baseline_independent_ternary_weight_digits: int
    candidate_independent_ternary_weight_digits: int


def matched_ffn_ledger(hidden_size: int, baseline_width: int) -> FfnPhysicalLedger:
    """Ledger for ordinary SwiGLU M versus packed SwiGLU 1.5M.

    Serving weights are resident signed INT8.  Baseline issues gate, up, and
    down projections at width M.  Candidate issues one packed gate/up
    projection and one down projection at width 1.5M.  The down projection is
    counted as one ordinary INT8 matrix in both systems.
    """

    if baseline_width % 2:
        raise ValueError("baseline_width must be even")
    candidate_width = 3 * baseline_width // 2
    baseline_products = 3 * hidden_size * baseline_width
    candidate_products = 2 * hidden_size * candidate_width
    baseline_bytes = baseline_products
    candidate_bytes = candidate_products
    return FfnPhysicalLedger(
        hidden_size=hidden_size,
        baseline_width=baseline_width,
        candidate_width=candidate_width,
        baseline_resident_weight_bytes=baseline_bytes,
        candidate_resident_weight_bytes=candidate_bytes,
        baseline_int8_scalar_products=baseline_products,
        candidate_int8_scalar_products=candidate_products,
        baseline_logical_projection_outputs=3 * baseline_width,
        candidate_logical_projection_outputs=3 * candidate_width,
        baseline_independent_ternary_weight_digits=baseline_products,
        candidate_independent_ternary_weight_digits=(
            3 * hidden_size * candidate_width
        ),
    )


def main() -> None:
    print(asdict(matched_ffn_ledger(hidden_size=384, baseline_width=1024)))


if __name__ == "__main__":
    main()
