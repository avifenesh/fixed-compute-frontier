#!/usr/bin/env python3
"""Exact resource ledger and separation witness for bit-budgeted conditional matrices."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def matrix_weights(dimension: int, hidden: int) -> int:
    return 3 * dimension * hidden


def candidate_storage_bytes(
    dimension: int,
    hidden: int,
    experts: int,
    base_bits: int,
    delta_bits: int,
    scale_bits: int,
    router_bits: int,
) -> int:
    weights = matrix_weights(dimension, hidden)
    payload_bits = weights * (base_bits + experts * delta_bits)
    scale_values = (experts + 1) * (2 * hidden + dimension)
    router_values = dimension * experts + experts
    total_bits = payload_bits + scale_values * scale_bits + router_values * router_bits
    return math.ceil(total_bits / 8)


def largest_equal_storage_hidden(args: argparse.Namespace) -> int:
    baseline_bytes = matrix_weights(args.dimension, args.hidden) * args.baseline_bits // 8
    hidden = args.hidden - (args.hidden % args.alignment)
    while hidden > 0:
        if candidate_storage_bytes(
            args.dimension,
            hidden,
            args.experts,
            args.base_bits,
            args.delta_bits,
            args.scale_bits,
            args.router_bits,
        ) <= baseline_bytes:
            return hidden
        hidden -= args.alignment
    raise RuntimeError("no positive equal-storage hidden width")


def piecewise_witness() -> dict[str, object]:
    xs = [[-2.0, 0.0], [0.0, -2.0], [0.0, 2.0], [2.0, 0.0]]
    expected_routes = [0, 1, 2, 3]
    router_weights = [[-1.0, 0.0], [0.0, -1.0], [0.0, 1.0], [1.0, 0.0]]
    router_biases = [0.0, 0.0, 0.0, 0.0]
    base_int8_codes = [0, 0]
    base_scale = 1.0
    delta_two_bit_alphabet = [-1, 0, 1]
    delta_codes = [[-1, 0], [0, -1], [0, 1], [1, 0]]
    delta_scale = 1.0
    decoded_weights = [
        [base_scale * base + delta_scale * delta for base, delta in zip(base_int8_codes, route_codes, strict=True)]
        for route_codes in delta_codes
    ]

    def dot(left: list[float], right: list[float]) -> float:
        return sum(a * b for a, b in zip(left, right, strict=True))

    def routed_by_affine_logits(x: list[float]) -> int:
        logits = [dot(weight, x) + bias for weight, bias in zip(router_weights, router_biases, strict=True)]
        return max(range(len(logits)), key=logits.__getitem__)

    actual_routes = [routed_by_affine_logits(x) for x in xs]
    ys = [dot(decoded_weights[route], x) for x, route in zip(xs, expected_routes, strict=True)]
    coordinate_energy = [sum(x[index] ** 2 for x in xs) for index in range(2)]
    dense_weights = [
        sum(x[index] * y for x, y in zip(xs, ys, strict=True)) / coordinate_energy[index]
        for index in range(2)
    ]
    dense_predictions = [dot(dense_weights, x) for x in xs]
    dense_mse = sum((prediction - target) ** 2 for prediction, target in zip(dense_predictions, ys, strict=True)) / len(xs)
    routed_predictions = [
        dot(decoded_weights[route], x) for x, route in zip(xs, actual_routes, strict=True)
    ]
    routed_mse = sum((prediction - target) ** 2 for prediction, target in zip(routed_predictions, ys, strict=True)) / len(xs)
    return {
        "inputs": xs,
        "expected_routes": expected_routes,
        "actual_affine_router_routes": actual_routes,
        "router_weights": router_weights,
        "router_biases": router_biases,
        "base_int8_codes": base_int8_codes,
        "base_scale": base_scale,
        "delta_two_bit_alphabet": delta_two_bit_alphabet,
        "unused_two_bit_pattern_count": 1,
        "delta_codes": delta_codes,
        "delta_scale": delta_scale,
        "decoded_route_weights": decoded_weights,
        "targets": ys,
        "best_single_bias_free_linear": {"weights": dense_weights, "mse": dense_mse},
        "routed": {
            "mse": routed_mse,
            "multiplications_per_example_after_routing": 2,
            "routes_match": actual_routes == expected_routes,
        },
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.base_bits + args.experts * args.delta_bits != args.baseline_bits:
        raise ValueError("frozen payload bit allocation must exactly equal baseline bits")
    if (args.baseline_bits, args.base_bits, args.delta_bits, args.experts) != (16, 8, 2, 4):
        raise ValueError("the v3 ternary witness is frozen specifically to 16=8+4x2")
    hidden = largest_equal_storage_hidden(args)
    baseline_weights = matrix_weights(args.dimension, args.hidden)
    candidate_weights = matrix_weights(args.dimension, hidden)
    baseline_bytes = baseline_weights * args.baseline_bits // 8
    candidate_bytes = candidate_storage_bytes(
        args.dimension,
        hidden,
        args.experts,
        args.base_bits,
        args.delta_bits,
        args.scale_bits,
        args.router_bits,
    )
    router_macs = args.dimension * args.experts
    candidate_macs = candidate_weights + router_macs
    allocation_expert_cap = (args.baseline_bits - args.base_bits) // args.delta_bits
    witness = piecewise_witness()
    gates = {
        "exact_resident_storage_not_above_bf16": candidate_bytes <= baseline_bytes,
        "active_matrix_plus_router_macs_not_above_baseline": candidate_macs <= baseline_weights,
        "frozen_8_plus_k_times_2_allocation_cap_is_four": allocation_expert_cap == args.experts == 4,
        "affine_router_realizes_four_frozen_regions": witness["routed"]["routes_match"],
        "conditional_witness_exact": witness["routed"]["mse"] == 0.0,
        "single_bias_free_linear_witness_inexact":
            witness["best_single_bias_free_linear"]["mse"] > 1e-6,
        "exact_physical_payload_is_16_bits":
            args.base_bits + args.experts * args.delta_bits == args.baseline_bits,
        "witness_uses_deployed_ternary_delta_alphabet":
            witness["delta_two_bit_alphabet"] == [-1, 0, 1],
    }
    preregistration = Path("results/bit_budgeted_conditional_matrix_stage0_v3_preregistration.md")
    return {
        "candidate": "Bit-Budgeted Conditional Matrix",
        "scope": "stage-0 exact arithmetic and artificial function-class witness only",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "preregistration_sha256": hashlib.sha256(preregistration.read_bytes()).hexdigest(),
        "args": {**vars(args), "output": str(args.output)},
        "ledger": {
            "baseline_hidden": args.hidden,
            "equal_storage_aligned_hidden": hidden,
            "hidden_reduction": args.hidden - hidden,
            "baseline_matrix_weights": baseline_weights,
            "candidate_matrix_weights": candidate_weights,
            "logical_base_plus_expert_weight_values": candidate_weights * (args.experts + 1),
            "baseline_bytes": baseline_bytes,
            "candidate_bytes_including_scales_and_router": candidate_bytes,
            "unused_bytes": baseline_bytes - candidate_bytes,
            "baseline_matrix_macs_per_token": baseline_weights,
            "candidate_matrix_plus_router_macs_per_token": candidate_macs,
            "router_bias_additions_per_token": args.experts,
            "mac_reduction_fraction": (baseline_weights - candidate_macs) / baseline_weights,
            "payload_bits_per_coordinate": args.base_bits + args.experts * args.delta_bits,
            "frozen_allocation_expert_cap": allocation_expert_cap,
            "semantic_delta_combinations_per_coordinate": 3**args.experts,
            "physical_delta_bit_patterns_per_coordinate": 2 ** (args.delta_bits * args.experts),
        },
        "conditional_witness": witness,
        "net_gain_condition": (
            "In a local quadratic loss model, routed specialization helps only when the "
            "between-route Hessian-weighted variance of optimal matrices exceeds the "
            "Hessian-weighted base-plus-delta quantization error and routing error."
        ),
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=14336)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--baseline-bits", type=int, default=16)
    parser.add_argument("--base-bits", type=int, default=8)
    parser.add_argument("--delta-bits", type=int, default=2)
    parser.add_argument("--scale-bits", type=int, default=16)
    parser.add_argument("--router-bits", type=int, default=16)
    parser.add_argument("--alignment", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/bit-budgeted-conditional-matrix-stage0-v3.json"),
    )
    args = parser.parse_args()
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gates": payload["gates"], "all_gates_pass": payload["all_gates_pass"]}, indent=2))
    raise SystemExit(0 if payload["all_gates_pass"] else 1)


if __name__ == "__main__":
    main()
