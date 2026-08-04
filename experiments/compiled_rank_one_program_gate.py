#!/usr/bin/env python3
"""Exact algebra and resource gate for compiled ordered rank-one programs.

The candidate executes

    x_i = x_{i-1} + c_i v_i
    c_i = scale_i * phi(u_i^T x_{i-1} + bias_i)

without materializing every intermediate D-vector.  The compiled evaluator
uses a_i = u_i^T x_0 + bias_i and G_ij = u_i^T v_j:

    c_i = scale_i * phi(a_i + sum_{j<i} G_ij c_j)
    x_L = x_0 + sum_i c_i v_i.

The equality is exact over real arithmetic and follows by induction.  This
file tests the equality numerically, checks that ordering buys a function that
an additive singleton-expert mixture cannot express, and emits an honest byte
and operation ledger for the proposed serving shape.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np


Array = np.ndarray


def silu(x: Array) -> Array:
    return x / (1.0 + np.exp(-x))


ACTIVATIONS: dict[str, Callable[[Array], Array]] = {
    "tanh": np.tanh,
    "silu": silu,
    "relu": lambda x: np.maximum(x, 0.0),
}


def direct_program(
    x: Array,
    u: Array,
    v: Array,
    bias: Array,
    scale: Array,
    activation: Callable[[Array], Array],
) -> Array:
    y = np.array(x, dtype=np.float64, copy=True)
    for i in range(u.shape[0]):
        coefficient = scale[i] * activation(np.asarray(y @ u[i] + bias[i]))
        y = y + coefficient * v[i]
    return y


def compile_program(u: Array, v: Array) -> Array:
    """Return G_ij = u_i^T v_j; only its strict lower triangle is used."""
    return u @ v.T


def compiled_program(
    x: Array,
    u: Array,
    v: Array,
    bias: Array,
    scale: Array,
    activation: Callable[[Array], Array],
    gram: Array | None = None,
) -> tuple[Array, Array]:
    gram = compile_program(u, v) if gram is None else gram
    base = x @ u.T + bias
    coefficients = np.empty(u.shape[0], dtype=np.float64)
    for i in range(u.shape[0]):
        preactivation = base[i] + gram[i, :i] @ coefficients[:i]
        coefficients[i] = scale[i] * activation(np.asarray(preactivation))
    return x + coefficients @ v, coefficients


def additive_mixture(
    x: Array,
    u: Array,
    v: Array,
    bias: Array,
    scale: Array,
    activation: Callable[[Array], Array],
) -> Array:
    return x + (scale * activation(x @ u.T + bias)) @ v


def packed_bits(count: int, bits_per_item: int) -> int:
    return (count * bits_per_item + 7) // 8


def resource_ledger(
    *,
    dimension: int,
    bank_size: int,
    program_count: int,
    program_length: int,
    scalar_bytes: int = 2,
) -> dict[str, int | float]:
    id_bits = math.ceil(math.log2(bank_size))
    strict_triangle = program_length * (program_length - 1) // 2

    # Packed is the information-oriented record.  Physical uses uint16 IDs at
    # this frozen bank size and BF16/FP16 cached cross terms.
    packed_record_bytes = packed_bits(program_length, id_bits) + (
        strict_triangle * scalar_bytes
    )
    physical_id_bytes = 2 if bank_size <= 2**16 else 4
    physical_record_bytes = (
        program_length * physical_id_bytes + strict_triangle * scalar_bytes
    )

    expert_vector_bytes = bank_size * 2 * dimension * scalar_bytes
    expert_metadata_bytes = bank_size * 2 * scalar_bytes  # bias and scale
    bank_bytes = expert_vector_bytes + expert_metadata_bytes
    compiled_program_bytes = program_count * physical_record_bytes
    candidate_bytes = bank_bytes + compiled_program_bytes

    singleton_peer_bytes = program_count * (2 * dimension + 2) * scalar_bytes
    independent_depth_l_bytes = (
        program_count * program_length * (2 * dimension + 2) * scalar_bytes
    )

    active_expert_bytes = program_length * (2 * dimension + 2) * scalar_bytes
    vector_macs = 2 * program_length * dimension
    scalar_recurrence_macs = strict_triangle

    return {
        "dimension": dimension,
        "bank_size": bank_size,
        "program_count": program_count,
        "program_length": program_length,
        "id_bits": id_bits,
        "strict_triangle_scalars": strict_triangle,
        "packed_program_record_bytes": packed_record_bytes,
        "physical_program_record_bytes": physical_record_bytes,
        "shared_expert_vector_bytes": expert_vector_bytes,
        "shared_expert_metadata_bytes": expert_metadata_bytes,
        "shared_expert_bank_bytes": bank_bytes,
        "compiled_program_table_bytes": compiled_program_bytes,
        "candidate_total_bytes_excluding_router": candidate_bytes,
        "peer_singleton_pool_bytes_for_program_count": singleton_peer_bytes,
        "independent_depth_l_pool_bytes": independent_depth_l_bytes,
        "compression_vs_singleton_peer": singleton_peer_bytes / candidate_bytes,
        "compression_vs_independent_depth_l": independent_depth_l_bytes
        / candidate_bytes,
        "active_expert_bytes_per_token": active_expert_bytes,
        "vector_macs_per_token": vector_macs,
        "scalar_recurrence_macs_per_token": scalar_recurrence_macs,
        "scalar_recurrence_overhead_fraction": scalar_recurrence_macs
        / vector_macs,
    }


def run_gate(seed: int, trials: int, dimension: int, length: int) -> dict:
    rng = np.random.default_rng(seed)
    equality_errors: dict[str, list[float]] = {name: [] for name in ACTIVATIONS}

    for activation_name, activation in ACTIVATIONS.items():
        for _ in range(trials):
            x = rng.normal(size=dimension)
            # The scale keeps long random programs numerically well behaved
            # without making the equality easier.
            u = rng.normal(size=(length, dimension)) / math.sqrt(dimension)
            v = rng.normal(size=(length, dimension)) / math.sqrt(dimension)
            bias = rng.normal(scale=0.2, size=length)
            scale = rng.uniform(0.2, 0.8, size=length)
            direct = direct_program(x, u, v, bias, scale, activation)
            compiled, _ = compiled_program(x, u, v, bias, scale, activation)
            denominator = max(float(np.linalg.norm(direct)), 1e-12)
            equality_errors[activation_name].append(
                float(np.linalg.norm(direct - compiled) / denominator)
            )

    # A fixed two-step witness: additive execution is permutation invariant,
    # while sequential execution is generically order sensitive.
    witness_x = np.array([0.75, -0.25], dtype=np.float64)
    witness_u = np.array([[1.0, 0.5], [-0.25, 1.25]], dtype=np.float64)
    witness_v = np.array([[0.4, 0.8], [1.1, -0.3]], dtype=np.float64)
    witness_bias = np.array([0.1, -0.2], dtype=np.float64)
    witness_scale = np.array([0.9, 0.7], dtype=np.float64)
    reverse = np.array([1, 0])

    additive_forward = additive_mixture(
        witness_x,
        witness_u,
        witness_v,
        witness_bias,
        witness_scale,
        np.tanh,
    )
    additive_reverse = additive_mixture(
        witness_x,
        witness_u[reverse],
        witness_v[reverse],
        witness_bias[reverse],
        witness_scale[reverse],
        np.tanh,
    )
    program_forward = direct_program(
        witness_x,
        witness_u,
        witness_v,
        witness_bias,
        witness_scale,
        np.tanh,
    )
    program_reverse = direct_program(
        witness_x,
        witness_u[reverse],
        witness_v[reverse],
        witness_bias[reverse],
        witness_scale[reverse],
        np.tanh,
    )

    max_equality_error = max(max(values) for values in equality_errors.values())
    additive_order_distance = float(np.linalg.norm(additive_forward - additive_reverse))
    program_order_distance = float(np.linalg.norm(program_forward - program_reverse))

    gates = {
        "compiled_matches_direct_below_1e-12": max_equality_error < 1e-12,
        "additive_is_order_invariant_below_1e-12": additive_order_distance < 1e-12,
        "program_is_order_sensitive_above_1e-3": program_order_distance > 1e-3,
    }

    source = Path(__file__).read_bytes()
    return {
        "candidate": "compiled-ordered-rank-one-program",
        "seed": seed,
        "trials_per_activation": trials,
        "test_dimension": dimension,
        "test_program_length": length,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "algebra": {
            "relative_errors": equality_errors,
            "max_relative_error": max_equality_error,
            "additive_order_distance": additive_order_distance,
            "program_order_distance": program_order_distance,
        },
        "resource_ledger": resource_ledger(
            dimension=4096,
            bank_size=16384,
            program_count=1_000_000,
            program_length=8,
        ),
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "claim_boundary": {
            "buys": "ordered nonlinear depth and compact reuse of a shared rank-one expert bank",
            "does_not_buy": "arbitrary independent program functions or free routing information",
            "different_currency": "a short scalar dependency chain, compiled cross-term bytes, and harder training",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--length", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = run_gate(args.seed, args.trials, args.dimension, args.length)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload + "\n")
    print(payload)
    raise SystemExit(0 if result["all_gates_pass"] else 1)


if __name__ == "__main__":
    main()
