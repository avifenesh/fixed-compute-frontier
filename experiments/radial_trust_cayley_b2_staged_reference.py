#!/usr/bin/env python3
"""Clear PyTorch staged-BF16 reference for the frozen B2 operator.

The graph makes every frozen BF16 boundary and reduction dependency explicit.
CPU execution validates topology, shapes, routing, and staged semantics, but it
does not claim bit identity with SM90 libdevice or instruction selection.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Sequence

import numpy as np
import torch


FROZEN_WIDTH = 4096
FROZEN_NODES = 7166
FROZEN_LAYERS = 12
BASES = 3
MATCHINGS = 3
NEUMANN_ORDER = 4
ROUTE_MULTIPLIER = 1_103_515_247
ROUTE_DEPTH_STRIDE = 12_345
RESIDUAL_SCALE_F32 = struct.unpack("<f", struct.pack("<I", 0x3E8E00D5))[0]

TOPOLOGY_SEEDS = (21, 124, 233, 93, 160, 191, 18, 31, 38)
AFFINE_MULTIPLIERS = (61, 359, 169, 373, 131, 115, 113, 19, 29)


@dataclass(frozen=True)
class TorchFormulaTopology:
    partner: torch.Tensor
    pair: torch.Tensor
    sign: torch.Tensor


@dataclass(frozen=True)
class TorchLayerArtifacts:
    basis: torch.Tensor
    payload: torch.Tensor
    threshold: torch.Tensor


@dataclass(frozen=True)
class StagedTraceStep:
    depth: int
    node: int
    bit: int
    coordinate: int
    payload_edge: int
    trust_scale: float


@dataclass(frozen=True)
class StagedReferenceResult:
    output: torch.Tensor
    final_hidden: torch.Tensor
    traces: tuple[tuple[StagedTraceStep, ...], ...]


def torch_bf16_from_raw_bits(values: np.ndarray, *, device: torch.device) -> torch.Tensor:
    bits = np.asarray(values)
    if bits.dtype != np.dtype("<u2") or not bits.flags.c_contiguous:
        raise ValueError("BF16 storage must be C-order little-endian uint16")
    tensor = torch.from_numpy(bits).view(torch.bfloat16)
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("BF16 storage contains NaN or infinity")
    return tensor.clone().to(device=device)


def _permuted_position(position: int, seed: int, multiplier: int, width: int) -> int:
    left_quadratic = 6 * (2 * seed + 1)
    left_offset = 97 * seed + 17
    right_quadratic = 6 * (2 * ((53 * seed + 7) % 257) + 1)
    right_offset = 193 * seed + 29
    first = (position + left_quadratic * position * position) % width
    second = (multiplier * first + left_offset) % width
    return (second + right_quadratic * second * second + right_offset) % width


def build_formula_topology(width: int, *, device: torch.device) -> TorchFormulaTopology:
    if width <= 0 or width > FROZEN_WIDTH or width % 2:
        raise ValueError("width must be positive, even, and no greater than 4096")
    partner = torch.empty((BASES, MATCHINGS, width), dtype=torch.int64)
    pair = torch.empty((BASES, MATCHINGS, width), dtype=torch.int64)
    sign = torch.empty((BASES, MATCHINGS, width), dtype=torch.bool)
    expected = list(range(width))
    for flat, (seed, multiplier) in enumerate(
        zip(TOPOLOGY_SEEDS, AFFINE_MULTIPLIERS, strict=True)
    ):
        if math.gcd(multiplier, width) != 1:
            raise ValueError("affine multiplier is not coprime to width")
        permutation = [
            _permuted_position(position, seed, multiplier, width)
            for position in range(width)
        ]
        if sorted(permutation) != expected:
            raise RuntimeError("topology formula is not a permutation")
        basis, matching = divmod(flat, MATCHINGS)
        for pair_index in range(width // 2):
            positive = permutation[2 * pair_index]
            negative = permutation[2 * pair_index + 1]
            partner[basis, matching, positive] = negative
            pair[basis, matching, positive] = pair_index
            sign[basis, matching, positive] = False
            partner[basis, matching, negative] = positive
            pair[basis, matching, negative] = pair_index
            sign[basis, matching, negative] = True
    return TorchFormulaTopology(
        partner=partner.to(device=device),
        pair=pair.to(device=device),
        sign=sign.to(device=device),
    )


def _validate_artifacts(
    artifacts: TorchLayerArtifacts,
    *,
    width: int,
    nodes: int,
    device: torch.device,
) -> None:
    expected = (
        (artifacts.basis, (BASES, MATCHINGS, width // 2), "basis"),
        (artifacts.payload, (nodes, 2, 3, width), "payload"),
        (artifacts.threshold, (nodes,), "threshold"),
    )
    for tensor, shape, name in expected:
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != torch.bfloat16
            or tensor.device != device
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"{name} must have shape {shape}, BF16 dtype, matching device, "
                "and contiguous storage"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} contains NaN or infinity")


def route_coordinates(
    nodes: torch.Tensor, depth: int, layer: int, width: int
) -> torch.Tensor:
    if depth < 0 or not 0 <= layer < FROZEN_LAYERS:
        raise ValueError("invalid route coordinate input")
    if width <= 0 or width & (width - 1):
        raise ValueError("route width must be a positive power of two")
    route_seed = 815 + 104_729 * layer + 7_919
    return (
        nodes.to(torch.int64) * ROUTE_MULTIPLIER
        + route_seed
        + depth * ROUTE_DEPTH_STRIDE
    ).bitwise_and(0xFFFFFFFF).bitwise_and(width - 1)


def sparse_apply_staged(
    values: torch.Tensor,
    artifacts: TorchLayerArtifacts,
    topology: TorchFormulaTopology,
    basis_index: int,
    *,
    transpose_sign: bool,
) -> torch.Tensor:
    """One sparse application with explicit FP32 operations and BF16 store."""
    if values.ndim != 2 or values.dtype != torch.bfloat16:
        raise ValueError("values must be rank-two BF16")
    basis = basis_index % BASES
    result: torch.Tensor | None = None
    for matching in range(MATCHINGS):
        partner = topology.partner[basis, matching]
        pair = topology.pair[basis, matching]
        stored_sign = topology.sign[basis, matching]
        sign = torch.logical_not(stored_sign) if transpose_sign else stored_sign
        gathered = values.index_select(1, partner).float()
        coefficient = artifacts.basis[basis, matching].index_select(0, pair).float()
        coefficient = torch.where(sign, -coefficient, coefficient)
        if result is None:
            result = torch.mul(gathered, coefficient)
        else:
            result = torch.addcmul(result, gathered, coefficient)
    assert result is not None
    return result.to(torch.bfloat16)


def cayley_staged(
    values: torch.Tensor,
    artifacts: TorchLayerArtifacts,
    topology: TorchFormulaTopology,
    basis_index: int,
    *,
    transpose: bool,
) -> torch.Tensor:
    source = values
    term = values
    accumulator = values.float()
    for _ in range(NEUMANN_ORDER):
        term = sparse_apply_staged(
            term,
            artifacts,
            topology,
            basis_index,
            transpose_sign=not transpose,
        )
        accumulator = torch.add(accumulator, term.float())
    result = torch.addcmul(
        -source.float(), accumulator, torch.ones_like(accumulator), value=2.0
    )
    return result.to(torch.bfloat16)


def fixed_pairwise_sum(squares: torch.Tensor) -> torch.Tensor:
    if squares.ndim != 2 or squares.dtype != torch.float32:
        raise ValueError("squares must have shape [batch,width] in FP32")
    width = squares.shape[1]
    if width <= 0 or width & (width - 1):
        raise ValueError("reduction width must be a positive power of two")
    work = squares.clone()
    stride = width // 2
    while stride:
        work[:, :stride] = torch.add(work[:, :stride], work[:, stride : 2 * stride])
        stride //= 2
    return work[:, 0]


def run_staged_reference(
    hidden: torch.Tensor,
    artifacts: TorchLayerArtifacts,
    *,
    layer: int,
    width: int,
    nodes: int,
    forced_bits: torch.Tensor | None = None,
    residual_scale: float = RESIDUAL_SCALE_F32,
) -> StagedReferenceResult:
    if (
        hidden.ndim != 2
        or tuple(hidden.shape)[1] != width
        or hidden.shape[0] == 0
        or hidden.dtype != torch.bfloat16
        or not hidden.is_contiguous()
    ):
        raise ValueError("hidden must be nonempty contiguous BF16 [batch,width]")
    if not bool(torch.isfinite(hidden).all()):
        raise ValueError("hidden contains NaN or infinity")
    if not 0 <= layer < FROZEN_LAYERS or nodes <= 0:
        raise ValueError("invalid layer or tree")
    if not math.isfinite(residual_scale) or residual_scale <= 0.0:
        raise ValueError("invalid residual scale")
    if forced_bits is not None:
        if (
            forced_bits.ndim != 2
            or forced_bits.shape[0] != hidden.shape[0]
            or forced_bits.shape[1] == 0
            or forced_bits.dtype not in (torch.bool, torch.uint8, torch.int8)
            or forced_bits.device != hidden.device
        ):
            raise ValueError("forced_bits has the wrong shape, dtype, or device")
        if not bool(torch.logical_or(forced_bits == 0, forced_bits == 1).all()):
            raise ValueError("forced_bits must contain only zero or one")
    _validate_artifacts(artifacts, width=width, nodes=nodes, device=hidden.device)
    topology = build_formula_topology(width, device=hidden.device)

    initial = hidden.clone()
    state = hidden.clone()
    current_nodes = torch.zeros(
        hidden.shape[0], dtype=torch.int64, device=hidden.device
    )
    traces: list[list[StagedTraceStep]] = [[] for _ in range(hidden.shape[0])]
    max_depth = forced_bits.shape[1] if forced_bits is not None else 64

    for depth in range(max_depth):
        active = torch.nonzero(current_nodes < nodes, as_tuple=False).flatten()
        if active.numel() == 0:
            break
        active_hidden = state.index_select(0, active)
        basis = depth % BASES
        projected = cayley_staged(
            active_hidden, artifacts, topology, basis, transpose=False
        )
        selected_nodes = current_nodes.index_select(0, active)
        coordinates = route_coordinates(selected_nodes, depth, layer, width)
        if forced_bits is None:
            projected_coordinate = projected.gather(1, coordinates[:, None]).squeeze(1)
            threshold = artifacts.threshold.index_select(0, selected_nodes)
            route_value = torch.sub(projected_coordinate.float(), threshold.float())
            if not bool(torch.isfinite(route_value).all()):
                raise FloatingPointError("router became non-finite")
            selected_bits = torch.ge(route_value, 0.0).to(torch.int64)
        else:
            selected_bits = forced_bits[active, depth].to(torch.int64)
        selected = artifacts.payload[selected_nodes, selected_bits]
        gate, up, down = selected[:, 0], selected[:, 1], selected[:, 2]

        projected_fp32 = projected.float()
        gate_projection = torch.mul(gate.float(), projected_fp32)
        negative = torch.neg(gate_projection)
        exponential = torch.exp(negative)
        denominator = torch.add(exponential, 1.0)
        reciprocal = torch.div(torch.ones_like(denominator), denominator)
        silu = torch.mul(gate_projection, reciprocal)
        up_projection = torch.mul(up.float(), projected_fp32)
        activation = torch.mul(silu, up_projection)
        local_delta = torch.mul(down.float(), activation).to(torch.bfloat16)
        raw_delta = cayley_staged(
            local_delta, artifacts, topology, basis, transpose=True
        )

        raw_fp32 = raw_delta.float()
        sum_squares = fixed_pairwise_sum(torch.mul(raw_fp32, raw_fp32))
        mean = torch.mul(sum_squares, float(1.0 / width))
        radial_denominator = torch.sqrt(torch.add(mean, 1.0))
        trust_scale = torch.div(torch.ones_like(radial_denominator), radial_denominator)
        if not bool(torch.isfinite(trust_scale).all()):
            raise FloatingPointError("trust scale became non-finite")
        trusted_delta = torch.mul(raw_fp32, trust_scale[:, None]).to(torch.bfloat16)
        updated = torch.addcmul(
            active_hidden.float(),
            trusted_delta.float(),
            torch.ones_like(trusted_delta, dtype=torch.float32),
            value=float(residual_scale),
        ).to(torch.bfloat16)
        if not bool(torch.isfinite(updated).all()):
            raise FloatingPointError("staged reference became non-finite")
        state.index_copy_(0, active, updated)

        for local, token in enumerate(active.tolist()):
            node = int(selected_nodes[local].item())
            bit = int(selected_bits[local].item())
            traces[token].append(
                StagedTraceStep(
                    depth=depth,
                    node=node,
                    bit=bit,
                    coordinate=int(coordinates[local].item()),
                    payload_edge=2 * node + bit,
                    trust_scale=float(trust_scale[local].item()),
                )
            )
        current_nodes.index_copy_(
            0, active, 2 * selected_nodes + 1 + selected_bits
        )

    if bool((current_nodes < nodes).any()):
        raise ValueError("route ended before every token reached a leaf")
    output = torch.sub(state.float(), initial.float()).to(torch.bfloat16)
    return StagedReferenceResult(
        output=output,
        final_hidden=state,
        traces=tuple(tuple(trace) for trace in traces),
    )


def run_frozen_staged_reference(
    hidden: torch.Tensor,
    artifacts: TorchLayerArtifacts,
    *,
    layer: int,
    forced_bits: torch.Tensor | None = None,
) -> StagedReferenceResult:
    return run_staged_reference(
        hidden,
        artifacts,
        layer=layer,
        width=FROZEN_WIDTH,
        nodes=FROZEN_NODES,
        forced_bits=forced_bits,
        residual_scale=RESIDUAL_SCALE_F32,
    )
