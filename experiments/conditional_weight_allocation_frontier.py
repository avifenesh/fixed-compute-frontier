#!/usr/bin/env python3
"""Exact byte/MAC ledgers and a transparent rate-distortion allocation screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def matrix_coordinates(dimension: int, hidden: int) -> int:
    return 3 * dimension * hidden


def scale_values_per_matrix_set(dimension: int, hidden: int) -> int:
    return 2 * hidden + dimension


def layout_bytes(
    *,
    dimension: int,
    hidden: int,
    routes: int,
    payload_bits_per_coordinate: int,
    scale_sets: int,
    scale_bits: int,
    router_bits: int,
) -> int:
    payload_bits = matrix_coordinates(dimension, hidden) * payload_bits_per_coordinate
    scale_bits_total = scale_values_per_matrix_set(dimension, hidden) * scale_sets * scale_bits
    router_values = dimension * routes + routes
    return math.ceil((payload_bits + scale_bits_total + router_values * router_bits) / 8)


def largest_feasible_hidden(
    args: argparse.Namespace,
    *,
    routes: int,
    payload_bits_per_coordinate: int,
    scale_sets: int,
) -> int:
    baseline_coordinates = matrix_coordinates(args.dimension, args.hidden)
    baseline_bytes = baseline_coordinates * args.baseline_bits // 8
    hidden = args.hidden - args.hidden % args.alignment
    while hidden > 0:
        candidate_bytes = layout_bytes(
            dimension=args.dimension,
            hidden=hidden,
            routes=routes,
            payload_bits_per_coordinate=payload_bits_per_coordinate,
            scale_sets=scale_sets,
            scale_bits=args.scale_bits,
            router_bits=args.router_bits,
        )
        candidate_macs = matrix_coordinates(args.dimension, hidden) + args.dimension * routes
        if candidate_bytes <= baseline_bytes and candidate_macs <= baseline_coordinates:
            return hidden
        hidden -= args.alignment
    raise RuntimeError("no feasible aligned width")


def layout_record(
    args: argparse.Namespace,
    *,
    name: str,
    routes: int,
    payload_bits_per_coordinate: int,
    scale_sets: int,
    active_payload_bits_per_coordinate: int,
) -> dict[str, int | float | str]:
    hidden = largest_feasible_hidden(
        args,
        routes=routes,
        payload_bits_per_coordinate=payload_bits_per_coordinate,
        scale_sets=scale_sets,
    )
    baseline_coordinates = matrix_coordinates(args.dimension, args.hidden)
    baseline_bytes = baseline_coordinates * args.baseline_bits // 8
    candidate_coordinates = matrix_coordinates(args.dimension, hidden)
    candidate_bytes = layout_bytes(
        dimension=args.dimension,
        hidden=hidden,
        routes=routes,
        payload_bits_per_coordinate=payload_bits_per_coordinate,
        scale_sets=scale_sets,
        scale_bits=args.scale_bits,
        router_bits=args.router_bits,
    )
    candidate_macs = candidate_coordinates + args.dimension * routes
    return {
        "name": name,
        "routes": routes,
        "aligned_hidden": hidden,
        "hidden_reduction": args.hidden - hidden,
        "resident_bytes": candidate_bytes,
        "storage_slack_bytes": baseline_bytes - candidate_bytes,
        "matrix_plus_router_macs": candidate_macs,
        "mac_slack": baseline_coordinates - candidate_macs,
        "payload_bits_per_coordinate": payload_bits_per_coordinate,
        "active_payload_bits_per_coordinate": active_payload_bits_per_coordinate,
        "active_payload_fraction_of_bf16": active_payload_bits_per_coordinate / args.baseline_bits,
        "scale_sets": scale_sets,
    }


def high_rate_crossover(base_bits: int, delta_bits: int, independent_bits: int = 4) -> dict[str, float | int | None]:
    """Optimistic scalar high-rate model; constants and codebook effects are omitted."""
    q_base = 2.0 ** (-2 * base_bits)
    q_delta = 2.0 ** (-2 * delta_bits)
    q_independent = 2.0 ** (-2 * independent_bits)
    denominator = q_delta - q_independent
    ratio = None if denominator <= 0 else max(0.0, (q_independent - q_base) / denominator)
    correlation = None if ratio is None else 1.0 / (1.0 + ratio)
    return {
        "base_bits": base_bits,
        "delta_bits": delta_bits,
        "independent_bits": independent_bits,
        "base_distortion_factor": q_base,
        "delta_distortion_factor": q_delta,
        "independent_distortion_factor": q_independent,
        "maximum_delta_to_base_variance_ratio_for_structured_win": ratio,
        "minimum_pairwise_route_correlation_for_structured_win": correlation,
    }


def allocation_neutral_witness() -> dict[str, object]:
    corners = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
    base = [1, 2]

    def dot(left: list[int], right: list[int]) -> int:
        return sum(a * b for a, b in zip(left, right, strict=True))

    route_weights = [
        [base_coordinate + delta_coordinate for base_coordinate, delta_coordinate in zip(base, corner, strict=True)]
        for corner in corners
    ]
    actual_routes = [
        max(range(4), key=lambda route: dot(corners[route], x))
        for x in corners
    ]
    targets = [dot(route_weights[route], x) for route, x in enumerate(corners)]
    dense_predictions = [dot(base, x) for x in corners]
    dense_mse = sum((prediction - target) ** 2 for prediction, target in zip(dense_predictions, targets, strict=True)) / 4
    routed_predictions = [dot(route_weights[route], x) for route, x in zip(actual_routes, corners, strict=True)]
    routed_mse = sum((prediction - target) ** 2 for prediction, target in zip(routed_predictions, targets, strict=True)) / 4
    return {
        "inputs_and_router_rows": corners,
        "base": base,
        "route_deltas": corners,
        "route_weights": route_weights,
        "actual_routes": actual_routes,
        "targets": targets,
        "best_single_bias_free_linear": {"weights": base, "mse": dense_mse},
        "routed_mse": routed_mse,
        "exactly_representable_by": [
            "shared_12_plus_4x1",
            "shared_8_plus_4x2_ternary",
            "shared_4_plus_4x3",
            "independent_k4_w4",
        ],
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    layouts = [
        layout_record(
            args,
            name="shared_base_12_plus_4x1",
            routes=4,
            payload_bits_per_coordinate=16,
            scale_sets=5,
            active_payload_bits_per_coordinate=13,
        ),
        layout_record(
            args,
            name="shared_base_8_plus_4x2_ternary",
            routes=4,
            payload_bits_per_coordinate=16,
            scale_sets=5,
            active_payload_bits_per_coordinate=10,
        ),
        layout_record(
            args,
            name="shared_base_4_plus_4x3",
            routes=4,
            payload_bits_per_coordinate=16,
            scale_sets=5,
            active_payload_bits_per_coordinate=7,
        ),
        layout_record(
            args,
            name="independent_k4_w4",
            routes=4,
            payload_bits_per_coordinate=16,
            scale_sets=4,
            active_payload_bits_per_coordinate=4,
        ),
        layout_record(
            args,
            name="independent_k8_w2",
            routes=8,
            payload_bits_per_coordinate=16,
            scale_sets=8,
            active_payload_bits_per_coordinate=2,
        ),
        layout_record(
            args,
            name="independent_k2_w8",
            routes=2,
            payload_bits_per_coordinate=16,
            scale_sets=2,
            active_payload_bits_per_coordinate=8,
        ),
    ]
    crossovers = [high_rate_crossover(base, delta) for base, delta in ((12, 1), (8, 2), (4, 3))]
    witness = allocation_neutral_witness()
    gates = {
        "all_layouts_fit_bf16_bytes": all(record["storage_slack_bytes"] >= 0 for record in layouts),
        "all_layouts_fit_bf16_matrix_macs_including_router": all(record["mac_slack"] >= 0 for record in layouts),
        "independent_k4_w4_reads_less_active_payload_than_8_plus_4x2":
            next(record for record in layouts if record["name"] == "independent_k4_w4")["active_payload_bits_per_coordinate"]
            < next(record for record in layouts if record["name"] == "shared_base_8_plus_4x2_ternary")["active_payload_bits_per_coordinate"],
        "eight_plus_four_times_two_requires_route_correlation_above_0p93_in_high_rate_model":
            crossovers[1]["minimum_pairwise_route_correlation_for_structured_win"] > 0.93,
        "allocation_neutral_witness_routes_and_reconstructs_exactly":
            witness["actual_routes"] == [0, 1, 2, 3]
            and witness["routed_mse"] == 0
            and witness["best_single_bias_free_linear"]["mse"] == 4,
    }
    preregistration = Path("results/conditional-weight-allocation-frontier-preregistration.md")
    return {
        "scope": "single-GPU exact ledger plus optimistic scalar high-rate screen; no LLM-quality claim",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "preregistration_sha256": hashlib.sha256(preregistration.read_bytes()).hexdigest(),
        "args": {**vars(args), "output": str(args.output)},
        "layouts": layouts,
        "high_rate_model": {
            "assumption": "distortion is proportional to variance times 2^(-2 bits); codebook constants, ternary unused code, cross terms, routing, and nonlinear loss are omitted",
            "crossovers_against_independent_k4_w4": crossovers,
        },
        "allocation_neutral_witness": witness,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=14336)
    parser.add_argument("--baseline-bits", type=int, default=16)
    parser.add_argument("--scale-bits", type=int, default=16)
    parser.add_argument("--router-bits", type=int, default=16)
    parser.add_argument("--alignment", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("results/conditional-weight-allocation-frontier.json"))
    args = parser.parse_args()
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gates": payload["gates"], "all_gates_pass": payload["all_gates_pass"]}, indent=2))
    raise SystemExit(0 if payload["all_gates_pass"] else 1)


if __name__ == "__main__":
    main()
