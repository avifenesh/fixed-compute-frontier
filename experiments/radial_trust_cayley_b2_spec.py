#!/usr/bin/env python3
"""CPU-verifiable specification for the frozen B2 physical experiment.

This module contains no CUDA entrypoint and cannot run the B2 gate.  It makes
the paper's integer topology, byte ledger, artifact packing, traces, and fixed
randomized schedules executable before kernel implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np


D = 4096
M = 14336
N = 7166
E = 2 * N
K = 3
Q = 3
P = 4
LAYERS = 12
PRIMARY_BATCHES = (1, 2, 4, 8)
EVALUATION_ROWS = 8192
TUNING_ROWS = 2048
EVALUATION_TRACE_SEEDS = {"gaussian": 9_000_815, "rademacher": 9_100_815}
TUNING_TRACE_SEEDS = {"gaussian": 8_000_815, "rademacher": 8_100_815}

TOPOLOGY_SEEDS = (21, 124, 233, 93, 160, 191, 18, 31, 38)
AFFINE_MULTIPLIERS = (61, 359, 169, 373, 131, 115, 113, 19, 29)

ROUTE_MULTIPLIER = 1_103_515_247
ROUTE_DEPTH_STRIDE = 12_345
RESIDUAL_SCALE_F32_BITS = 0x3E8E00D5
MEAN_MULTIPLIER_F32_BITS = 0x39800000

ARMS = ("candidate", "dense", "moe", "monarch")
FAMILIES = ("gaussian", "rademacher")
PRIMARY_CELLS = tuple(
    (family, batch) for family in FAMILIES for batch in PRIMARY_BATCHES
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def round_up_256(value: int) -> int:
    if value < 0:
        raise ValueError("byte count must be nonnegative")
    return (value + 255) & ~255


@dataclass(frozen=True)
class ResidentLedger:
    payload_scalars: int
    basis_scalars: int
    threshold_scalars: int
    candidate_layer_scalars: int
    dense_layer_scalars: int
    candidate_layer_allocated_bytes: int
    dense_layer_allocated_bytes: int
    topology_bytes: int
    candidate_panel_bytes: int
    dense_panel_bytes: int

    @property
    def candidate_minus_dense_panel_bytes(self) -> int:
        return self.candidate_panel_bytes - self.dense_panel_bytes


def resident_ledger() -> ResidentLedger:
    payload = N * 2 * 3 * D
    basis = K * Q * (D // 2)
    thresholds = N
    candidate = payload + basis + thresholds
    dense = 3 * D * M
    candidate_allocated = (
        round_up_256(payload * 2)
        + round_up_256(basis * 2)
        + round_up_256(thresholds * 2)
    )
    dense_allocated = round_up_256(dense * 2)
    topology_bytes = K * Q * D * 4
    return ResidentLedger(
        payload_scalars=payload,
        basis_scalars=basis,
        threshold_scalars=thresholds,
        candidate_layer_scalars=candidate,
        dense_layer_scalars=dense,
        candidate_layer_allocated_bytes=candidate_allocated,
        dense_layer_allocated_bytes=dense_allocated,
        topology_bytes=topology_bytes,
        candidate_panel_bytes=LAYERS * candidate_allocated + topology_bytes,
        dense_panel_bytes=LAYERS * dense_allocated,
    )


def _matching_permutation(
    positions: np.ndarray, seed: int, multiplier: int, width: int
) -> np.ndarray:
    if positions.dtype != np.int64:
        positions = positions.astype(np.int64, copy=False)
    left_quadratic = 6 * (2 * seed + 1)
    left_offset = 97 * seed + 17
    right_quadratic = 6 * (2 * ((53 * seed + 7) % 257) + 1)
    right_offset = 193 * seed + 29
    permuted = (positions + left_quadratic * positions * positions) % width
    permuted = (multiplier * permuted + left_offset) % width
    return (
        permuted + right_quadratic * permuted * permuted + right_offset
    ) % width


def build_packed_descriptors(width: int = D) -> np.ndarray:
    """Generate `[basis,matching,output]` uint32 descriptors from formulas."""
    if width <= 0 or width > 4096 or width % 2:
        raise ValueError("descriptor width must be positive, even, and <=4096")
    if width // 2 > 2048:
        raise ValueError("pair index does not fit 11 bits")
    positions = np.arange(width, dtype=np.int64)
    descriptors = np.empty((K, Q, width), dtype="<u4")
    for flat_index, (seed, multiplier) in enumerate(
        zip(TOPOLOGY_SEEDS, AFFINE_MULTIPLIERS, strict=True)
    ):
        if math.gcd(multiplier, width) != 1:
            raise ValueError("affine multiplier is not coprime to width")
        permutation = _matching_permutation(
            positions, seed, multiplier, width
        )
        if not np.array_equal(np.sort(permutation), positions):
            raise RuntimeError("topology formula is not a permutation")
        basis, matching = divmod(flat_index, Q)
        row = descriptors[basis, matching]
        for pair_index in range(width // 2):
            positive = int(permutation[2 * pair_index])
            negative = int(permutation[2 * pair_index + 1])
            row[positive] = np.uint32(negative | (pair_index << 12))
            row[negative] = np.uint32(
                positive | (pair_index << 12) | (1 << 23)
            )
    return descriptors


def decode_descriptor(descriptor: int) -> tuple[int, int, int]:
    value = int(descriptor)
    if value < 0 or value > 0xFFFFFFFF:
        raise ValueError("descriptor must be uint32")
    if value >> 24:
        raise ValueError("reserved descriptor bits are nonzero")
    partner = value & 0xFFF
    pair_index = (value >> 12) & 0x7FF
    sign = (value >> 23) & 1
    return partner, pair_index, sign


def verify_packed_descriptors(descriptors: np.ndarray) -> None:
    if descriptors.shape != (K, Q, D) or descriptors.dtype != np.dtype("<u4"):
        raise ValueError("descriptor array has the wrong shape or dtype")
    for basis in range(K):
        for matching in range(Q):
            pair_counts = np.zeros(D // 2, dtype=np.int64)
            for output in range(D):
                partner, pair, sign = decode_descriptor(
                    int(descriptors[basis, matching, output])
                )
                if partner == output or not 0 <= partner < D:
                    raise ValueError("descriptor is not a fixed-point-free match")
                reverse_partner, reverse_pair, reverse_sign = decode_descriptor(
                    int(descriptors[basis, matching, partner])
                )
                if (reverse_partner, reverse_pair, reverse_sign) != (
                    output,
                    pair,
                    1 - sign,
                ):
                    raise ValueError("descriptor matching is not a signed involution")
                pair_counts[pair] += 1
            if not np.array_equal(pair_counts, np.full(D // 2, 2)):
                raise ValueError("each pair index must occur exactly twice")


def route_coordinate(node: int, depth: int, layer: int) -> int:
    if not 0 <= layer < LAYERS or node < 0 or depth < 0:
        raise ValueError("invalid route coordinate input")
    route_seed = 815 + 104_729 * layer + 7_919
    wrapped = (
        node * ROUTE_MULTIPLIER
        + route_seed
        + depth * ROUTE_DEPTH_STRIDE
    ) & 0xFFFFFFFF
    return wrapped & (D - 1)


def path_from_bits(bits: Iterable[int]) -> tuple[tuple[int, ...], int]:
    node = 0
    nodes: list[int] = []
    for raw_bit in bits:
        if node >= N:
            break
        bit = int(raw_bit)
        if bit not in (0, 1):
            raise ValueError("route bits must be zero or one")
        nodes.append(node)
        node = 2 * node + 1 + bit
    if node < N:
        raise ValueError("route bits ended before reaching a leaf")
    return tuple(nodes), node


def leaf_depth_counts() -> dict[int, int]:
    counts: dict[int, int] = {}
    stack = [(0, 0)]
    while stack:
        node, depth = stack.pop()
        for bit in (0, 1):
            child = 2 * node + 1 + bit
            child_depth = depth + 1
            if child >= N:
                counts[child_depth] = counts.get(child_depth, 0) + 1
            else:
                stack.append((child, child_depth))
    return counts


def bf16_rne_from_fp64(values: np.ndarray | Sequence[float]) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    if not np.isfinite(source).all():
        raise ValueError("BF16 source must be finite")
    fp32 = source.astype("<f4")
    if not np.isfinite(fp32).all():
        raise ValueError("FP64-to-FP32 conversion became non-finite")
    bits = fp32.view("<u4")
    bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    rounded = bits + bias
    return (rounded >> np.uint32(16)).astype("<u2")


def fp32_from_bf16_bits(values: np.ndarray | Sequence[int]) -> np.ndarray:
    bits = np.asarray(values, dtype="<u2")
    return (bits.astype("<u4") << np.uint32(16)).view("<f4")


def normalize_rows_fp64(values: np.ndarray) -> np.ndarray:
    """Normalize in the paper's scalar coordinate order."""
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] <= 0:
        raise ValueError("values must be a nonempty rank-two array")
    if not np.isfinite(source).all():
        raise ValueError("trace source must be finite")
    output = np.empty_like(source)
    multiplier = np.float64(1.0 / source.shape[1])
    for row in range(source.shape[0]):
        total = np.float64(0.0)
        for coordinate in range(source.shape[1]):
            value = np.float64(source[row, coordinate])
            product = np.float64(value * value)
            total = np.float64(total + product)
        rms = np.sqrt(np.float64(total * multiplier))
        if not np.isfinite(rms) or rms == 0.0:
            raise ValueError("trace row has invalid RMS")
        for coordinate in range(source.shape[1]):
            output[row, coordinate] = np.float64(
                np.float64(source[row, coordinate]) / rms
            )
    return output


def generate_trace(
    family: str,
    *,
    rows: int,
    width: int = D,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if family not in FAMILIES or rows <= 0 or width <= 0:
        raise ValueError("invalid trace request")
    rng = np.random.Generator(np.random.PCG64(seed))
    if family == "gaussian":
        raw = rng.standard_normal(size=(rows, width), dtype=np.float64)
    else:
        bits = rng.integers(
            0,
            2,
            size=(rows, width),
            dtype=np.uint8,
            endpoint=False,
        )
        raw = bits.astype(np.float64)
        raw *= np.float64(2.0)
        raw -= np.float64(1.0)
    normalized = normalize_rows_fp64(raw)
    return normalized, bf16_rne_from_fp64(normalized)


def generate_frozen_trace(
    family: str,
    *,
    tuning: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    seeds = TUNING_TRACE_SEEDS if tuning else EVALUATION_TRACE_SEEDS
    if family not in seeds:
        raise ValueError("invalid trace family")
    frozen_rows = TUNING_ROWS if tuning else EVALUATION_ROWS
    return generate_trace(
        family,
        rows=frozen_rows,
        width=D,
        seed=seeds[family],
    )


def evaluation_row(
    process: int,
    replay: int,
    batch: int,
    lane: int,
    layer: int,
) -> int:
    if not (0 <= process < 10 and batch > 0 and 0 <= lane < batch):
        raise ValueError("invalid evaluation row input")
    if not 0 <= layer < LAYERS:
        raise ValueError("invalid layer")
    return (((1000 * process + replay) * batch + lane + 683 * layer) & 8191)


def tuning_row(replay: int, batch: int, lane: int, layer: int) -> int:
    if batch <= 0 or not 0 <= lane < batch or not 0 <= layer < LAYERS:
        raise ValueError("invalid tuning row input")
    return (replay * batch + lane + 683 * layer) % TUNING_ROWS


def _permuted(items: Sequence[object], seed: int) -> tuple[object, ...]:
    order = np.random.Generator(np.random.PCG64(seed)).permutation(len(items))
    return tuple(items[int(index)] for index in order)


def timing_chunk_order(process: int, round_index: int) -> tuple[tuple[str, int, str], ...]:
    if not 0 <= process < 10 or not 0 <= round_index < 20:
        raise ValueError("invalid timing block")
    canonical = tuple(
        (family, batch, arm)
        for family, batch in PRIMARY_CELLS
        for arm in ARMS
    )
    return _permuted(
        canonical, 9_900_815 + 1000 * process + round_index
    )  # type: ignore[return-value]


def counter_range_order(process: int) -> tuple[tuple[str, int, str], ...]:
    if not 0 <= process < 10:
        raise ValueError("invalid counter process")
    canonical = tuple(
        (family, batch, arm)
        for family, batch in PRIMARY_CELLS
        for arm in ("candidate", "dense")
    )
    return _permuted(canonical, 9_975_815 + process)  # type: ignore[return-value]


def energy_loop_order(process: int) -> tuple[tuple[str, int, str], ...]:
    if not 0 <= process < 10:
        raise ValueError("invalid energy process")
    canonical = tuple(
        (family, batch, arm)
        for family, batch in PRIMARY_CELLS
        for arm in ("candidate", "dense")
    )
    return _permuted(canonical, 9_950_815 + process)  # type: ignore[return-value]


def memory_pair_order(process: int) -> tuple[str, str]:
    if not 0 <= process < 5:
        raise ValueError("invalid memory process")
    return _permuted(
        ("candidate", "dense"), 9_990_815 + process
    )  # type: ignore[return-value]


@dataclass(frozen=True)
class OperationLedger:
    path_length: int
    sparse_applications: int
    multiplications_before_silu_expansion: int
    additions_before_final_output: int
    silu_evaluations: int
    square_roots: int
    scalar_divisions: int
    final_output_subtractions: int
    descriptor_requested_bytes: int
    coefficient_requested_bytes: int


def operation_ledger(path_length: int) -> OperationLedger:
    if path_length not in (12, 13):
        raise ValueError("B2 paths have length 12 or 13")
    sparse = 8 * path_length
    return OperationLedger(
        path_length=path_length,
        sparse_applications=sparse,
        multiplications_before_silu_expansion=(33 * D + 1) * path_length,
        additions_before_final_output=(28 * D + 1) * path_length,
        silu_evaluations=D * path_length,
        square_roots=path_length,
        scalar_divisions=path_length,
        final_output_subtractions=D,
        descriptor_requested_bytes=sparse * Q * D * 4,
        coefficient_requested_bytes=sparse * Q * D * 2,
    )


def iter_candidate_layer_draws(
    layer: int,
    *,
    width: int = D,
    nodes: int = N,
) -> Iterator[tuple[str, np.ndarray]]:
    """Yield basis, packed payload, and threshold served arrays.

    The default payload draws are intentionally large.  Callers should consume
    and persist one yielded array before requesting the next.
    """
    if not 0 <= layer < LAYERS or width <= 0 or width % 2 or nodes <= 0:
        raise ValueError("invalid candidate layer request")
    seed = 815 + 104_729 * layer
    rng = np.random.Generator(np.random.PCG64(seed))
    raw_basis = np.empty((K, Q, width // 2), dtype=np.float64)
    for basis in range(K):
        for matching in range(Q):
            raw_basis[basis, matching] = rng.normal(
                0.0, 1.0, size=(width // 2,)
            )
    bounded = np.tanh(raw_basis)
    bounded *= np.float64(1.0 / 12.0)
    yield "basis", bf16_rne_from_fp64(bounded)
    del raw_basis, bounded

    packed_payload = np.empty((nodes, 2, 3, width), dtype="<u2")
    for plane, (mean, standard_deviation) in enumerate(
        ((1.0, 0.02), (1.0, 0.02), (0.0, 0.15))
    ):
        payload = rng.normal(
            mean, standard_deviation, size=(nodes, 2, width)
        )
        packed_payload[:, :, plane, :] = bf16_rne_from_fp64(payload)
        del payload
    yield "payload", packed_payload
    yield "threshold", np.zeros(nodes, dtype="<u2")


def tensor_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()
