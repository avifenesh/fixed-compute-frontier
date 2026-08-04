#!/usr/bin/env python3
"""Algebra and exact-budget gate for block-triangular FFN microdepth.

The candidate replaces a width-M SwiGLU

    down(silu(gate(x)) * up(x))

with two wider dense projections and a tiny causal circuit between them:

    a = U x
    c_i = silu(a_i + sum_{j<i} H_ij c_j)
    y = V c.

The hidden coordinates are partitioned into groups capped at eight.  Thus the
extra depth is scalar/block-local; it does not add another D-wide matrix
multiplication.  H=0 recovers an ordinary wider SiLU MLP exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def packed_edges(group_size: int) -> int:
    return group_size * (group_size - 1) // 2


def triangular_coefficients(
    base: np.ndarray,
    packed_coupling: np.ndarray,
    activation=silu,
) -> np.ndarray:
    """Evaluate [..., groups, r] coefficients with coupling [groups, r(r-1)/2]."""
    group_size = base.shape[-1]
    groups = base.shape[-2]
    if packed_coupling.shape != (groups, packed_edges(group_size)):
        raise ValueError("invalid packed coupling shape")
    coefficients = []
    for i in range(group_size):
        value = base[..., i]
        if i:
            start = i * (i - 1) // 2
            previous = np.stack(coefficients, axis=-1)
            value = value + np.sum(
                previous * packed_coupling[:, start : start + i], axis=-1
            )
        coefficients.append(activation(value))
    return np.stack(coefficients, axis=-1)


def direct_group_program(
    x: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    activation=silu,
) -> np.ndarray:
    """Run each rank-one group from x and add the group updates."""
    batch, dimension = x.shape
    groups, group_size, vector_dimension = u.shape
    if dimension != vector_dimension or u.shape != v.shape:
        raise ValueError("invalid vector shapes")
    updates = np.zeros((batch, dimension), dtype=np.float64)
    for group in range(groups):
        state = x.copy()
        for i in range(group_size):
            coefficient = activation(np.sum(state * u[group, i], axis=-1))
            state = state + coefficient[:, None] * v[group, i]
        updates += state - x
    return x + updates


def compiled_group_program(
    x: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    activation=silu,
) -> np.ndarray:
    base = np.einsum("bd,gid->bgi", x, u)
    full_coupling = np.einsum("gid,gjd->gij", u, v)
    packed = np.concatenate(
        [full_coupling[:, i, :i] for i in range(1, u.shape[1])], axis=-1
    )
    coefficients = triangular_coefficients(base, packed, activation)
    return x + np.einsum("bgi,gid->bd", coefficients, v)


def tent_iterate_four(x: np.ndarray) -> np.ndarray:
    """Eight triangular ReLUs implementing four iterates of the tent map."""
    coefficients: list[np.ndarray] = []
    coefficients.append(np.maximum(2.0 * x, 0.0))
    coefficients.append(np.maximum(2.0 * x - 1.0, 0.0))
    for stage in range(1, 4):
        previous = coefficients[2 * stage - 2] - 2.0 * coefficients[2 * stage - 1]
        coefficients.append(np.maximum(2.0 * previous, 0.0))
        coefficients.append(np.maximum(2.0 * previous - 1.0, 0.0))
    return coefficients[-2] - 2.0 * coefficients[-1]


def count_linear_segments_on_unit_interval() -> int:
    # Midpoint slopes avoid the exact k/16 breakpoints.
    grid = np.arange(33, dtype=np.float64) / 32.0
    values = tent_iterate_four(grid)
    slopes = np.diff(values) * 32.0
    return int(1 + np.count_nonzero(np.abs(np.diff(slopes)) > 1e-10))


def matched_width(
    dimension: int,
    swiglu_width: int,
    group_size: int,
) -> tuple[int, int]:
    """Maximal group-aligned width; leftover scalars become learned biases."""
    budget = 3 * dimension * swiglu_width
    per_feature = 2 * dimension + (group_size - 1) / 2
    width = int(budget // per_feature)
    width -= width % group_size
    coupling = width * (group_size - 1) // 2
    bias_count = budget - (2 * dimension * width + coupling)
    if not 0 <= bias_count <= width:
        raise ValueError("budget cannot be closed with partial learned biases")
    return width, bias_count


def partial_bias_indices(width: int, group_size: int, bias_count: int) -> np.ndarray:
    """Odd threshold sites in every group, then position zero in early groups."""
    groups = width // group_size
    indices = [
        group * group_size + offset
        for group in range(groups)
        for offset in (1, 3, 5, 7)
    ]
    extras = bias_count - len(indices)
    if extras < 0 or extras > groups:
        raise ValueError("bias count is incompatible with the frozen group mask")
    indices.extend(group * group_size for group in range(extras))
    return np.asarray(indices, dtype=np.int64)


def resource_ledger(
    dimension: int = 384,
    swiglu_width: int = 1024,
    group_size: int = 8,
) -> dict[str, int | float]:
    width, bias_count = matched_width(dimension, swiglu_width, group_size)
    groups = width // group_size
    coupling = groups * packed_edges(group_size)
    baseline_dense = 3 * dimension * swiglu_width
    candidate_dense = 2 * dimension * width
    candidate_parameters = candidate_dense + coupling + bias_count
    return {
        "dimension": dimension,
        "baseline_swiglu_width": swiglu_width,
        "group_size": group_size,
        "candidate_width": width,
        "candidate_width_ratio": width / swiglu_width,
        "groups": groups,
        "baseline_dense_weight_parameters": baseline_dense,
        "candidate_dense_weight_parameters": candidate_dense,
        "candidate_packed_coupling_parameters": coupling,
        "candidate_partial_bias_parameters": bias_count,
        "candidate_total_parameters": candidate_parameters,
        "parameter_difference": candidate_parameters - baseline_dense,
        "baseline_dense_macs": baseline_dense,
        "candidate_dense_macs": candidate_dense,
        "candidate_triangular_macs": coupling,
        "candidate_total_macs_before_bias_and_activation": candidate_dense + coupling,
        "mac_difference_before_pointwise": candidate_dense + coupling - baseline_dense,
        "candidate_extra_macs_fraction": coupling / candidate_dense,
        "baseline_hidden_scalars_before_product": 2 * swiglu_width,
        "candidate_hidden_scalars": width,
    }


def run_gate(seed: int = 271, trials: int = 128) -> dict:
    rng = np.random.default_rng(seed)
    maximum_compilation_error = 0.0
    maximum_zero_coupling_error = 0.0
    for _ in range(trials):
        batch, groups, group_size, dimension = 7, 3, 8, 19
        x = rng.normal(size=(batch, dimension))
        u = rng.normal(size=(groups, group_size, dimension)) / math.sqrt(dimension)
        v = rng.normal(size=(groups, group_size, dimension)) / math.sqrt(dimension)
        direct = direct_group_program(x, u, v)
        compiled = compiled_group_program(x, u, v)
        maximum_compilation_error = max(
            maximum_compilation_error, float(np.max(np.abs(direct - compiled)))
        )
        base = rng.normal(size=(batch, groups, group_size))
        zero = np.zeros((groups, packed_edges(group_size)))
        maximum_zero_coupling_error = max(
            maximum_zero_coupling_error,
            float(np.max(np.abs(triangular_coefficients(base, zero) - silu(base)))),
        )

    small = resource_ledger()
    target = resource_ledger(4096, 11008, 8)
    small_bias_indices = partial_bias_indices(
        int(small["candidate_width"]),
        8,
        int(small["candidate_partial_bias_parameters"]),
    )
    observed_segments = count_linear_segments_on_unit_interval()
    shallow_width_eight_limit = 9
    gates = {
        "compiled_matches_direct_rank_one_program_below_1e_12": (
            maximum_compilation_error < 1e-12
        ),
        "zero_coupling_exactly_recovers_wide_silu": maximum_zero_coupling_error == 0.0,
        "eight_unit_microdepth_exceeds_shallow_relu_region_limit": (
            observed_segments > shallow_width_eight_limit
        ),
        "small_lm_parameter_budget_exact": small["parameter_difference"] == 0,
        "target_parameter_budget_exact": target["parameter_difference"] == 0,
        "small_lm_width_gain_at_least_1p49x": small["candidate_width_ratio"] >= 1.49,
        "target_scalar_recurrence_below_0p05_percent_dense_macs": (
            target["candidate_extra_macs_fraction"] < 0.0005
        ),
        "group_size_respects_measured_h100_cap": small["group_size"] <= 8,
        "bias_mask_is_unique_and_exact": (
            len(small_bias_indices) == small["candidate_partial_bias_parameters"]
            and len(np.unique(small_bias_indices)) == len(small_bias_indices)
        ),
    }
    return {
        "candidate": "block-triangular-microdepth-ffn",
        "gate": "G0-algebra-capacity-and-exact-budget",
        "seed": seed,
        "trials": trials,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "metrics": {
            "maximum_compilation_error": maximum_compilation_error,
            "maximum_zero_coupling_error": maximum_zero_coupling_error,
            "tent_iterate_linear_segments_on_unit_interval": observed_segments,
            "shallow_width_eight_relu_maximum_segments_on_a_line": (
                shallow_width_eight_limit
            ),
        },
        "small_lm_ledger": small,
        "target_shape_ledger": target,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "claim_boundary": {
            "proved": (
                "strict finite-width depth separation for the ReLU limiting variant, exact "
                "inclusion of a wider plain SiLU MLP at H=0, and exact parameter matching "
                "without a third dense matmul"
            ),
            "not_proved": "better language loss, optimization, or full-block serving latency",
            "changed_currency": (
                "a capped serial scalar dependency inside each eight-feature group plus "
                "a structured-composition prior"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=271)
    parser.add_argument("--trials", type=int, default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_gate(args.seed, args.trials)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    print(payload)
    raise SystemExit(0 if result["all_gates_pass"] else 1)


if __name__ == "__main__":
    main()
