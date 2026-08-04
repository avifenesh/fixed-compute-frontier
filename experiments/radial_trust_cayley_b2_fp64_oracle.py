#!/usr/bin/env python3
"""Independent CPU FP64 forced-route oracle for the frozen B2 operator.

This is a mathematical diagnostic, not the staged-BF16 serving reference.  It
reconstructs topology from the formulas rather than decoding packed
descriptors, converts served BF16 arrays exactly to FP64, and then keeps every
operator intermediate in FP64.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


FROZEN_WIDTH = 4096
FROZEN_NODES = 7166
FROZEN_LAYERS = 12
BASES = 3
MATCHINGS = 3
NEUMANN_ORDER = 4
ROUTE_MULTIPLIER = 1_103_515_247
ROUTE_DEPTH_STRIDE = 12_345
RESIDUAL_SCALE = 1.0 / math.sqrt(13.0)

TOPOLOGY_SEEDS = (21, 124, 233, 93, 160, 191, 18, 31, 38)
AFFINE_MULTIPLIERS = (61, 359, 169, 373, 131, 115, 113, 19, 29)


@dataclass(frozen=True)
class FormulaTopology:
    partner: np.ndarray
    pair: np.ndarray
    sign: np.ndarray


@dataclass(frozen=True)
class OracleLayerArtifacts:
    basis_bf16: np.ndarray
    payload_bf16: np.ndarray
    threshold_bf16: np.ndarray


@dataclass(frozen=True)
class ForcedTraceStep:
    depth: int
    node: int
    bit: int
    coordinate: int
    raw_delta_rms: float
    trusted_delta_rms: float


@dataclass(frozen=True)
class ForcedOracleResult:
    output: np.ndarray
    final_hidden: np.ndarray
    traces: tuple[tuple[ForcedTraceStep, ...], ...]


def _require_bf16_bits(values: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    if (
        values.shape != shape
        or values.dtype != np.dtype("<u2")
        or not values.flags.c_contiguous
    ):
        raise ValueError(
            f"{name} must have shape {shape}, dtype little-endian uint16, "
            "and C-order storage"
        )


def fp64_from_bf16_bits(values: np.ndarray) -> np.ndarray:
    """Decode raw BF16 words exactly and reject non-finite values."""
    bits = np.asarray(values)
    if bits.dtype != np.dtype("<u2"):
        raise ValueError("BF16 storage must be little-endian uint16")
    fp32 = (bits.astype("<u4") << np.uint32(16)).view("<f4")
    if not np.isfinite(fp32).all():
        raise ValueError("BF16 storage contains NaN or infinity")
    return fp32.astype(np.float64)


def _permuted_position(position: int, seed: int, multiplier: int, width: int) -> int:
    left_quadratic = 6 * (2 * seed + 1)
    left_offset = 97 * seed + 17
    right_quadratic = 6 * (2 * ((53 * seed + 7) % 257) + 1)
    right_offset = 193 * seed + 29
    first = (position + left_quadratic * position * position) % width
    second = (multiplier * first + left_offset) % width
    return (second + right_quadratic * second * second + right_offset) % width


def build_formula_topology(width: int) -> FormulaTopology:
    """Build signed matchings directly from the nine frozen formulas."""
    if width <= 0 or width > FROZEN_WIDTH or width % 2:
        raise ValueError("width must be positive, even, and no greater than 4096")
    partner = np.empty((BASES, MATCHINGS, width), dtype=np.int64)
    pair = np.empty((BASES, MATCHINGS, width), dtype=np.int64)
    sign = np.empty((BASES, MATCHINGS, width), dtype=np.int8)
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
            sign[basis, matching, positive] = 0
            partner[basis, matching, negative] = positive
            pair[basis, matching, negative] = pair_index
            sign[basis, matching, negative] = 1
    return FormulaTopology(partner=partner, pair=pair, sign=sign)


def route_coordinate(node: int, depth: int, layer: int, width: int) -> int:
    if node < 0 or depth < 0 or not 0 <= layer < FROZEN_LAYERS:
        raise ValueError("invalid route coordinate input")
    if width <= 0 or width & (width - 1):
        raise ValueError("route width must be a positive power of two")
    route_seed = 815 + 104_729 * layer + 7_919
    wrapped = (
        node * ROUTE_MULTIPLIER + route_seed + depth * ROUTE_DEPTH_STRIDE
    ) & 0xFFFFFFFF
    return wrapped & (width - 1)


def apply_sparse_skew_fp64(
    values: np.ndarray,
    basis_weights: np.ndarray,
    topology: FormulaTopology,
    basis_index: int,
) -> np.ndarray:
    """Apply one formula-generated sparse skew operator in FP64."""
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 2:
        raise ValueError("values must have shape [batch,width]")
    width = source.shape[1]
    if basis_weights.shape != (BASES, MATCHINGS, width // 2):
        raise ValueError("basis weights have the wrong shape")
    if topology.partner.shape != (BASES, MATCHINGS, width):
        raise ValueError("topology has the wrong shape")
    basis = basis_index % BASES
    result = np.zeros_like(source)
    for matching in range(MATCHINGS):
        partners = topology.partner[basis, matching]
        pairs = topology.pair[basis, matching]
        signs = topology.sign[basis, matching]
        signed_weights = basis_weights[basis, matching, pairs].copy()
        signed_weights[signs != 0] *= -1.0
        result += source[:, partners] * signed_weights[None, :]
    if not np.isfinite(result).all():
        raise FloatingPointError("sparse skew application became non-finite")
    return result


def cayley_neumann_fp64(
    values: np.ndarray,
    basis_weights: np.ndarray,
    topology: FormulaTopology,
    basis_index: int,
    *,
    transpose: bool = False,
) -> np.ndarray:
    """Apply the frozen order-four polynomial C or its transpose in FP64."""
    source = np.asarray(values, dtype=np.float64)
    term = source.copy()
    accumulator = source.copy()
    recurrence_sign = 1.0 if transpose else -1.0
    for _ in range(NEUMANN_ORDER):
        term = recurrence_sign * apply_sparse_skew_fp64(
            term, basis_weights, topology, basis_index
        )
        accumulator += term
    result = 2.0 * accumulator - source
    if not np.isfinite(result).all():
        raise FloatingPointError("Cayley polynomial became non-finite")
    return result


def _validated_artifacts(
    artifacts: OracleLayerArtifacts, width: int, nodes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _require_bf16_bits(
        artifacts.basis_bf16,
        (BASES, MATCHINGS, width // 2),
        "basis_bf16",
    )
    _require_bf16_bits(
        artifacts.payload_bf16,
        (nodes, 2, 3, width),
        "payload_bf16",
    )
    _require_bf16_bits(artifacts.threshold_bf16, (nodes,), "threshold_bf16")
    return (
        fp64_from_bf16_bits(artifacts.basis_bf16),
        fp64_from_bf16_bits(artifacts.payload_bf16),
        fp64_from_bf16_bits(artifacts.threshold_bf16),
    )


def _stable_rms_fp64(row: np.ndarray) -> float:
    """Return RMS without overflowing on finite FP64 coordinates."""
    scale = 0.0
    scaled_sum = 1.0
    for raw_value in row:
        value = abs(float(raw_value))
        if not math.isfinite(value):
            raise FloatingPointError("RMS input is non-finite")
        if value == 0.0:
            continue
        if scale < value:
            ratio = scale / value
            scaled_sum = 1.0 + scaled_sum * ratio * ratio
            scale = value
        else:
            ratio = value / scale
            scaled_sum += ratio * ratio
    if scale == 0.0:
        return 0.0
    rms = scale * math.sqrt(scaled_sum / row.size)
    if not math.isfinite(rms):
        raise FloatingPointError("RMS calculation became non-finite")
    return rms


def run_forced_oracle(
    hidden: np.ndarray,
    artifacts: OracleLayerArtifacts,
    forced_bits: np.ndarray,
    *,
    layer: int,
    width: int,
    nodes: int,
    residual_scale: float = RESIDUAL_SCALE,
) -> ForcedOracleResult:
    """Run one layer on forced paths with FP64 arithmetic throughout."""
    source = np.asarray(hidden, dtype=np.float64)
    bits = np.asarray(forced_bits)
    if source.ndim != 2 or source.shape[1] != width or source.shape[0] == 0:
        raise ValueError("hidden must have nonempty shape [batch,width]")
    if not np.isfinite(source).all():
        raise ValueError("hidden must be finite")
    if not 0 <= layer < FROZEN_LAYERS:
        raise ValueError("layer must be in the frozen range")
    if bits.ndim != 2 or bits.shape[0] != source.shape[0] or bits.shape[1] == 0:
        raise ValueError("forced_bits must have shape [batch,max_depth]")
    if bits.dtype.kind not in "biu" or not np.isin(bits, (0, 1)).all():
        raise ValueError("forced bits must be integers containing only zero or one")
    if nodes <= 0 or not np.isfinite(residual_scale) or residual_scale <= 0.0:
        raise ValueError("invalid tree or residual scale")
    basis_weights, payload, _threshold = _validated_artifacts(
        artifacts, width, nodes
    )
    topology = build_formula_topology(width)

    initial = source.copy()
    state = source.copy()
    current_nodes = np.zeros(source.shape[0], dtype=np.int64)
    traces: list[list[ForcedTraceStep]] = [[] for _ in range(source.shape[0])]

    for depth in range(bits.shape[1]):
        active = np.flatnonzero(current_nodes < nodes)
        if active.size == 0:
            break
        active_hidden = state[active]
        basis = depth % BASES
        projected = cayley_neumann_fp64(
            active_hidden, basis_weights, topology, basis
        )
        selected_bits = bits[active, depth].astype(np.int64, copy=False)
        selected_nodes = current_nodes[active].copy()
        selected = payload[selected_nodes, selected_bits]
        gate, up, down = selected[:, 0], selected[:, 1], selected[:, 2]
        gate_projection = gate * projected
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            silu = gate_projection / (1.0 + np.exp(-gate_projection))
            local_delta = down * silu * (up * projected)
        raw_delta = cayley_neumann_fp64(
            local_delta,
            basis_weights,
            topology,
            basis,
            transpose=True,
        )
        raw_rms = np.array(
            [_stable_rms_fp64(row) for row in raw_delta], dtype=np.float64
        )
        trust_denominator = np.array(
            [math.hypot(1.0, float(value)) for value in raw_rms],
            dtype=np.float64,
        )
        if not np.isfinite(trust_denominator).all():
            raise FloatingPointError("radial denominator became non-finite")
        trusted_delta = raw_delta / trust_denominator[:, None]
        if not np.isfinite(trusted_delta).all():
            raise FloatingPointError("trusted delta became non-finite")
        trusted_rms = np.array(
            [_stable_rms_fp64(row) for row in trusted_delta], dtype=np.float64
        )
        updated = active_hidden + residual_scale * trusted_delta
        if not np.isfinite(updated).all():
            raise FloatingPointError("forced oracle became non-finite")
        state[active] = updated
        for local_index, token in enumerate(active):
            traces[int(token)].append(
                ForcedTraceStep(
                    depth=depth,
                    node=int(selected_nodes[local_index]),
                    bit=int(selected_bits[local_index]),
                    coordinate=route_coordinate(
                        int(selected_nodes[local_index]), depth, layer, width
                    ),
                    raw_delta_rms=float(raw_rms[local_index]),
                    trusted_delta_rms=float(trusted_rms[local_index]),
                )
            )
        current_nodes[active] = (
            2 * selected_nodes + 1 + selected_bits
        )

    if np.any(current_nodes < nodes):
        raise ValueError("forced bits ended before every token reached a leaf")
    return ForcedOracleResult(
        output=state - initial,
        final_hidden=state,
        traces=tuple(tuple(trace) for trace in traces),
    )


def run_frozen_forced_oracle(
    hidden_bf16: np.ndarray,
    artifacts: OracleLayerArtifacts,
    forced_bits: np.ndarray,
    *,
    layer: int,
) -> ForcedOracleResult:
    """Strict width-4096, 7,166-node B2 wrapper for small CPU batches."""
    hidden_bits = np.asarray(hidden_bf16)
    if hidden_bits.ndim != 2 or hidden_bits.shape[1] != FROZEN_WIDTH:
        raise ValueError("hidden_bf16 must have shape [batch,4096]")
    _require_bf16_bits(
        hidden_bits,
        (hidden_bits.shape[0], FROZEN_WIDTH),
        "hidden_bf16",
    )
    return run_forced_oracle(
        fp64_from_bf16_bits(hidden_bits),
        artifacts,
        forced_bits,
        layer=layer,
        width=FROZEN_WIDTH,
        nodes=FROZEN_NODES,
        residual_scale=RESIDUAL_SCALE,
    )
