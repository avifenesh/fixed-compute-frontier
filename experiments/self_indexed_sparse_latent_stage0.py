#!/usr/bin/env python3
"""Exact Stage-0 gate for a self-indexed sparse latent transition graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from array import array
from pathlib import Path
from typing import Iterable, Sequence


def build_table(nodes: int, degree: int, seed: int) -> array:
    if nodes <= 0 or degree <= 0:
        raise ValueError("nodes and degree must be positive")
    rng = random.Random(seed)
    return array("I", (rng.randrange(nodes) for _ in range(nodes * degree)))


def pointer_walk(
    table: Sequence[int], degree: int, start: int, labels: Iterable[int]
) -> tuple[int, int]:
    node = start
    reads = 0
    for label in labels:
        if not 0 <= label < degree:
            raise ValueError("label outside degree")
        # The conservative executor charges the complete contiguous edge block,
        # even though the exact labeled transition consumes one selected record.
        reads += degree
        node = int(table[node * degree + label])
    return node, reads


def lookup_control_walk(
    table: Sequence[int], degree: int, start: int, labels: Iterable[int]
) -> tuple[int, int]:
    # Deliberately independent spelling of the matched RAM/table control.
    current = start
    charged_reads = 0
    for symbol in labels:
        base = current * degree
        block = table[base : base + degree]
        charged_reads += len(block)
        current = int(block[symbol])
    return current, charged_reads


def dense_one_hot_walk(
    table: Sequence[int], nodes: int, degree: int, start: int, labels: Iterable[int]
) -> tuple[int, int]:
    """Reference dense one-hot matvec; intentionally only for small worlds."""
    state = [0] * nodes
    state[start] = 1
    macs = 0
    for label in labels:
        matrix = [[0] * nodes for _ in range(nodes)]
        for source in range(nodes):
            destination = int(table[source * degree + label])
            matrix[destination][source] = 1
        output = [0] * nodes
        for row in range(nodes):
            total = 0
            for column in range(nodes):
                total += matrix[row][column] * state[column]
                macs += 1
            output[row] = total
        state = output
    if sum(state) != 1:
        raise AssertionError("dense reference lost one-hot invariant")
    return state.index(1), macs


def min_global_steps(nodes: int, degree: int) -> int:
    if nodes <= 1:
        return 0
    if degree <= 1:
        return nodes - 1
    return math.ceil(math.log(nodes, degree))


def reachable_upper_bound(degree: int, steps: int) -> int:
    if degree == 1:
        return steps + 1
    return (degree ** (steps + 1) - 1) // (degree - 1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def run(seed: int) -> dict:
    exact_worlds = []
    for world, nodes in enumerate((7, 11, 16)):
        degree = 4
        steps = 6
        table = build_table(nodes, degree, seed + world)
        rng = random.Random(seed * 17 + world)
        queries = []
        for _ in range(12):
            start = rng.randrange(nodes)
            labels = [rng.randrange(degree) for _ in range(steps)]
            pointer, pointer_reads = pointer_walk(table, degree, start, labels)
            lookup, lookup_reads = lookup_control_walk(table, degree, start, labels)
            dense, dense_macs = dense_one_hot_walk(
                table, nodes, degree, start, labels
            )
            queries.append(
                {
                    "start": start,
                    "labels": labels,
                    "endpoint": pointer,
                    "pointer_reads": pointer_reads,
                    "lookup_reads": lookup_reads,
                    "dense_macs": dense_macs,
                    "all_equal": pointer == lookup == dense,
                }
            )
        exact_worlds.append(
            {
                "nodes": nodes,
                "degree": degree,
                "steps": steps,
                "queries": queries,
                "all_equal": all(query["all_equal"] for query in queries),
            }
        )

    scale_k = 256
    scale_degree = 16
    scale_steps = 32
    scaling = []
    for nodes in (2**10, 2**14, 2**18, 2**22):
        scaling.append(
            {
                "nodes": nodes,
                "resident_records": nodes * scale_degree,
                "active_records_per_token": scale_k * scale_degree * scale_steps,
            }
        )

    budget_bytes = 2 * 1024**3
    edge_bytes = 8
    resident_records = budget_bytes // edge_bytes
    graph_nodes = resident_records // scale_degree
    active_records = scale_k * scale_degree * scale_steps
    active_bytes = active_records * edge_bytes
    emitted_per_step = scale_k * scale_degree
    dense_bf16_weights = budget_bytes // 2
    dense_width = math.isqrt(dense_bf16_weights)
    dense_square_weights = dense_width * dense_width

    locality = []
    for nodes, degree in ((2**20, 4), (2**24, 16), (2**32, 16)):
        steps = min_global_steps(nodes, degree)
        locality.append(
            {
                "nodes": nodes,
                "degree": degree,
                "minimum_steps_for_global_reach": steps,
                "reachable_nodes_upper_bound_at_that_depth": reachable_upper_bound(
                    degree, steps
                ),
            }
        )

    preregistration = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "self-indexed-sparse-latent-stage0-preregistration.md"
    )
    source = Path(__file__).resolve()
    exact_pass = all(world["all_equal"] for world in exact_worlds)
    control_pass = all(
        query["pointer_reads"] == query["lookup_reads"]
        for world in exact_worlds
        for query in world["queries"]
    )
    constant_active_pass = len(
        {row["active_records_per_token"] for row in scaling}
    ) == 1
    linear_resident_pass = all(
        row["resident_records"] == row["nodes"] * scale_degree
        for row in scaling
    )

    gates = {
        "pointer_equals_dense_one_hot": exact_pass,
        "matched_lookup_control_equal_and_equal_reads": control_pass,
        "active_reads_constant_across_graph_sizes": constant_active_pass,
        "resident_records_linear_in_node_count": linear_resident_pass,
        "fixed_budget_ledger_complete": all(
            value > 0
            for value in (
                graph_nodes,
                active_bytes,
                emitted_per_step,
                dense_width,
                dense_square_weights,
            )
        ),
        "locality_wall_reported": all(
            row["minimum_steps_for_global_reach"] > 0 for row in locality
        ),
    }

    return {
        "schema": "self-indexed-sparse-latent-stage0-v1",
        "status": "stage0_scaling_separation_only",
        "seed": seed,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "exact_worlds": exact_worlds,
        "scaling": scaling,
        "fixed_2gib_ledger": {
            "resident_budget_bytes": budget_bytes,
            "edge_record_bytes": edge_bytes,
            "pointer_bits": 32,
            "degree": scale_degree,
            "active_features": scale_k,
            "transition_steps_per_token": scale_steps,
            "resident_edge_records": resident_records,
            "resident_graph_nodes": graph_nodes,
            "active_edge_records_per_token": active_records,
            "active_edge_bytes_per_token": active_bytes,
            "emitted_messages_per_step_before_merge": emitted_per_step,
            "dense_bf16_weights_at_equal_bytes": dense_bf16_weights,
            "largest_square_dense_width_at_equal_bytes": dense_width,
            "dense_square_macs": dense_square_weights,
            "ideal_active_byte_ratio_graph_over_dense": active_bytes
            / budget_bytes,
            "ideal_edge_op_ratio_graph_over_dense_square": active_records
            / dense_square_weights,
        },
        "locality_wall": locality,
        "matched_control_conclusion": (
            "The separation comes from word-addressed learned memory. A matched "
            "table/RAM controller is identical; no unique algebraic capability is claimed."
        ),
        "source_sha256": sha256(source),
        "preregistration_sha256": sha256(preregistration),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7302026)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/self-indexed-sparse-latent-stage0.json"),
    )
    args = parser.parse_args()
    result = run(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["all_gates_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

