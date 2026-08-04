#!/usr/bin/env python3
"""T24 Stage 0: exact XOR-coded functional records over a raw-only quotient."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import raw_self_query_functional_record_stage0 as source


PREREGISTRATION = ROOT / "results/xor-functional-record-t24-stage0-preregistration.md"
PREREGISTRATION_SHA256 = (
    "be07f9de6c7ebbafaa9f0d832ce79bc5af07c57b3ac0630de2231ceaa695060f"
)
OUTPUT = ROOT / "results/xor-functional-record-t24-stage0.json"
SCHEMA = "xor-functional-record-t24-stage0-v1"

DOCUMENTS = 2_405
LOGICAL_SLOTS = 109
LOW_CELLS = LOGICAL_SLOTS
HIGH_CELLS = LOGICAL_SLOTS
RESERVED_CELLS = 2
PHYSICAL_CELLS = LOW_CELLS + HIGH_CELLS + RESERVED_CELLS
CELL_BITS = 4
PAYLOAD_BITS = 7
HASHES = 3
HASH_FAMILY = 1
SHUFFLE_SEED = 24_001
RANDOM_SEED = 24_003
HASH_DOMAIN = b"xor-functional-t24-v1"

EXPECTED_SKELETONS = 3_392
EXPECTED_EDGES = 14_146
EXPECTED_COVERED_DOCUMENTS = 2_248
EXPECTED_MAX_TARGET_CARDINALITY = 65
EXPECTED_MAX_DOCUMENT_EDGES = 101


def sha256_file(path: Path) -> str:
    return source.sha256_file(path)


def canonical_skeleton(skeleton: tuple[tuple[int, str], ...]) -> bytes:
    return json.dumps(
        skeleton, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def hash_positions(
    skeleton: tuple[tuple[int, str], ...],
    family: int = HASH_FAMILY,
    slots: int = LOGICAL_SLOTS,
) -> tuple[int, ...]:
    """Return three distinct, deterministic logical slots."""
    encoded = canonical_skeleton(skeleton)
    positions: list[int] = []
    counter = 0
    while len(positions) < HASHES:
        digest = hashlib.sha256(
            HASH_DOMAIN
            + b"|"
            + str(family).encode()
            + b"|"
            + encoded
            + b"|"
            + str(counter).encode()
        ).digest()
        for offset in range(0, len(digest), 2):
            position = int.from_bytes(digest[offset : offset + 2], "big") % slots
            if position not in positions:
                positions.append(position)
                if len(positions) == HASHES:
                    break
        counter += 1
    return tuple(positions)


def row_mask(skeleton: tuple[tuple[int, str], ...]) -> int:
    mask = 0
    for position in hash_positions(skeleton):
        mask |= 1 << position
    return mask


def gf2_rank(rows: list[int]) -> int:
    basis: dict[int, int] = {}
    for original in rows:
        value = original
        while value:
            pivot = value.bit_length() - 1
            if pivot in basis:
                value ^= basis[pivot]
            else:
                basis[pivot] = value
                break
    return len(basis)


def solve_full_row_rank(
    rows: list[int], payloads: list[int], columns: int = LOGICAL_SLOTS
) -> list[int]:
    """Solve A Z = Y over GF(2), carrying all payload bits in each RHS int."""
    if len(rows) != len(payloads):
        raise ValueError("row/payload length mismatch")
    if any(value < 0 or value >= 1 << PAYLOAD_BITS for value in payloads):
        raise ValueError("payload is outside seven bits")
    masks = list(rows)
    right = list(payloads)
    pivot_columns: list[int] = []
    rank = 0
    for column in range(columns):
        pivot = next(
            (index for index in range(rank, len(masks)) if (masks[index] >> column) & 1),
            None,
        )
        if pivot is None:
            continue
        masks[rank], masks[pivot] = masks[pivot], masks[rank]
        right[rank], right[pivot] = right[pivot], right[rank]
        for index in range(rank + 1, len(masks)):
            if (masks[index] >> column) & 1:
                masks[index] ^= masks[rank]
                right[index] ^= right[rank]
        pivot_columns.append(column)
        rank += 1
        if rank == len(masks):
            break
    if rank != len(masks):
        raise ValueError(f"matrix lacks full row rank: {rank} != {len(masks)}")

    solution = [0] * columns
    for row in range(rank - 1, -1, -1):
        pivot = pivot_columns[row]
        value = right[row]
        remaining = masks[row] & ~(1 << pivot)
        while remaining:
            bit = remaining & -remaining
            column = bit.bit_length() - 1
            value ^= solution[column]
            remaining ^= bit
        solution[pivot] = value
    return solution


def encode_record(
    skeletons: list[tuple[tuple[int, str], ...]], payloads: list[int]
) -> tuple[list[int], int]:
    rows = [row_mask(skeleton) for skeleton in skeletons]
    rank = gf2_rank(rows)
    if rank != len(rows):
        raise ValueError(f"rank-deficient document: {rank}/{len(rows)}")
    solution = solve_full_row_rank(rows, payloads)
    low = [value & 0xF for value in solution]
    high = [(value >> 4) & 0x7 for value in solution]
    record = low + high + [0] * RESERVED_CELLS
    if len(record) != PHYSICAL_CELLS:
        raise RuntimeError("wrong physical record width")
    return record, rank


def read_record(
    record: list[int], skeleton: tuple[tuple[int, str], ...]
) -> int:
    positions = hash_positions(skeleton)
    low = 0
    high = 0
    for position in positions:
        low ^= record[position]
        high ^= record[LOW_CELLS + position]
    return low | (high << 4)


def load_quotient() -> tuple[
    list[dict[str, str]],
    list[source.Probe],
    dict[tuple[tuple[int, str], ...], set[tuple[int, str]]],
]:
    documents = [json.loads(line) for line in source.CORPUS.read_text().splitlines()]
    probes, _ = source.build_probes(documents)
    collisions, _, _ = source.single_valued_collision_groups(probes)
    return documents, probes, collisions


def build_edges(
    probes: list[source.Probe],
    collisions: dict[tuple[tuple[int, str], ...], set[tuple[int, str]]],
) -> tuple[
    dict[int, list[tuple[tuple[tuple[int, str], ...], int, str]]],
    dict[tuple[int, tuple[tuple[int, str], ...], str], str],
    dict[tuple[tuple[int, str], ...], tuple[str, ...]],
]:
    split_by_edge = {
        (probe.document_index, probe.skeleton, probe.target): probe.split
        for probe in probes
    }
    targets_by_skeleton = {
        skeleton: tuple(sorted({target for _, target in values}))
        for skeleton, values in collisions.items()
    }
    by_document: dict[
        int, list[tuple[tuple[tuple[int, str], ...], int, str]]
    ] = collections.defaultdict(list)
    for skeleton, values in collisions.items():
        vocabulary = targets_by_skeleton[skeleton]
        ranks = {target: rank for rank, target in enumerate(vocabulary)}
        for document, target in values:
            split = split_by_edge[(document, skeleton, target)]
            by_document[document].append((skeleton, ranks[target], split))
    for values in by_document.values():
        values.sort(key=lambda item: canonical_skeleton(item[0]))
    return by_document, split_by_edge, targets_by_skeleton


def evaluate_records(
    records: list[list[int]],
    by_document: dict[int, list[tuple[tuple[tuple[int, str], ...], int, str]]],
) -> dict[str, object]:
    counts = collections.Counter()
    correct = collections.Counter()
    for document, edges in by_document.items():
        record = records[document]
        for skeleton, target, split in edges:
            prediction = read_record(record, skeleton)
            counts["all"] += 1
            counts[split] += 1
            if prediction == target:
                correct["all"] += 1
                correct[split] += 1
    return {
        split: {
            "examples": counts[split],
            "correct": correct[split],
            "accuracy": correct[split] / counts[split] if counts[split] else 0.0,
        }
        for split in ("all", "train", "evaluation")
    }


def bf16_level_check() -> dict[str, object]:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("authoritative Stage 0 requires torch for BF16 check") from error
    indices = torch.arange(16, dtype=torch.float32)
    levels = 2.0 * indices - 15.0
    roundtrip = levels.to(torch.bfloat16).float()
    decoded = torch.round((roundtrip + 15.0) / 2.0).to(torch.int64)
    return {
        "levels": [float(value) for value in levels.tolist()],
        "roundtrip_values_exact": bool(torch.equal(levels, roundtrip)),
        "decoded_indices_exact": bool(
            torch.equal(decoded, torch.arange(16, dtype=torch.int64))
        ),
    }


def run() -> dict[str, object]:
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise RuntimeError("preregistration hash mismatch")
    if sha256_file(source.CORPUS) != (
        "a02d2bdc714bae0c6e42d8527333de01506120845095aec936a301634fb7d00a"
    ):
        raise RuntimeError("raw corpus hash mismatch")

    documents, probes, collisions = load_quotient()
    by_document, _, targets_by_skeleton = build_edges(probes, collisions)
    records = [[0] * PHYSICAL_CELLS for _ in range(len(documents))]
    ranks: list[int] = []
    document_edge_counts: list[int] = []
    semantic_bits_by_document = [0] * len(documents)
    for document in range(len(documents)):
        edges = by_document.get(document, [])
        document_edge_counts.append(len(edges))
        semantic_bits_by_document[document] = sum(
            max(1, math.ceil(math.log2(len(targets_by_skeleton[skeleton]))))
            for skeleton, _, _ in edges
        )
        if not edges:
            continue
        skeletons = [skeleton for skeleton, _, _ in edges]
        payloads = [payload for _, payload, _ in edges]
        records[document], rank = encode_record(skeletons, payloads)
        ranks.append(rank)

    correct = evaluate_records(records, by_document)

    permutation = list(range(len(records)))
    random.Random(SHUFFLE_SEED).shuffle(permutation)
    shuffled_records = [records[index] for index in permutation]
    shuffled = evaluate_records(shuffled_records, by_document)
    zero = evaluate_records(
        [[0] * PHYSICAL_CELLS for _ in records], by_document
    )
    generator = random.Random(RANDOM_SEED)
    random_records = [
        [generator.randrange(16) for _ in range(LOW_CELLS)]
        + [generator.randrange(8) for _ in range(HIGH_CELLS)]
        + [0] * RESERVED_CELLS
        for _ in records
    ]
    random_control = evaluate_records(random_records, by_document)
    bf16 = bf16_level_check()

    target_cardinalities = [len(values) for values in targets_by_skeleton.values()]
    quotient_edges = sum(len(values) for values in collisions.values())
    covered_documents = len(by_document)
    maximum_edges = max(document_edge_counts)
    quotient_contract = (
        len(collisions) == EXPECTED_SKELETONS
        and quotient_edges == EXPECTED_EDGES
        and covered_documents == EXPECTED_COVERED_DOCUMENTS
        and max(target_cardinalities) == EXPECTED_MAX_TARGET_CARDINALITY
        and maximum_edges == EXPECTED_MAX_DOCUMENT_EDGES
    )
    full_rank = len(ranks) == covered_documents and all(
        rank == document_edge_counts[document]
        for document, rank in zip(
            (index for index, count in enumerate(document_edge_counts) if count),
            ranks,
            strict=True,
        )
    )
    cells_valid = all(
        len(record) == PHYSICAL_CELLS
        and all(0 <= value <= 15 for value in record[:LOW_CELLS])
        and all(0 <= value <= 7 for value in record[LOW_CELLS : LOW_CELLS + HIGH_CELLS])
        and record[-RESERVED_CELLS:] == [0] * RESERVED_CELLS
        for record in records
    )
    evaluation_accuracy = float(correct["evaluation"]["accuracy"])
    controls_below = all(
        evaluation_accuracy - float(control["evaluation"]["accuracy"]) >= 0.20
        for control in (shuffled, zero, random_control)
    )
    gates = {
        "raw_quotient_exact": quotient_contract,
        "single_global_hash_family_full_row_rank": full_rank,
        "all_registered_reads_bit_exact": correct["all"]["correct"] == EXPECTED_EDGES,
        "all_evaluation_reads_bit_exact": correct["evaluation"]["accuracy"] == 1.0,
        "physical_cells_and_ranges_exact": cells_valid,
        "bf16_levels_and_indices_exact": bf16["roundtrip_values_exact"]
        and bf16["decoded_indices_exact"],
        "all_controls_at_least_20_points_below": controls_below,
        "fixed_record_budget_exact": PHYSICAL_CELLS == 220
        and PHYSICAL_CELLS * CELL_BITS == 880,
    }
    passed = all(gates.values())
    return {
        "schema": SCHEMA,
        "status": "stage0_only_not_capability_evidence",
        "corpus": {
            "path": str(source.CORPUS.relative_to(ROOT)),
            "sha256": sha256_file(source.CORPUS),
            "documents": len(documents),
        },
        "quotient": {
            "skeletons": len(collisions),
            "edges": quotient_edges,
            "covered_documents": covered_documents,
            "maximum_target_cardinality": max(target_cardinalities),
            "maximum_edges_per_document": maximum_edges,
            "mean_edges_per_covered_document": quotient_edges / covered_documents,
        },
        "record": {
            "hash_family": HASH_FAMILY,
            "hashes_per_query": HASHES,
            "logical_slots": LOGICAL_SLOTS,
            "low_cells": LOW_CELLS,
            "high_cells": HIGH_CELLS,
            "reserved_cells": RESERVED_CELLS,
            "physical_cells_per_document": PHYSICAL_CELLS,
            "physical_bits_per_document": PHYSICAL_CELLS * CELL_BITS,
            "logical_solution_bits_per_document": LOGICAL_SLOTS * PAYLOAD_BITS,
            "maximum_rank": max(ranks),
            "minimum_rank_over_covered_documents": min(ranks),
            "per_document_hash_retries": 0,
        },
        "information": {
            "payload_bits": PAYLOAD_BITS,
            "semantic_bits_total": sum(semantic_bits_by_document),
            "semantic_bits_maximum_document": max(semantic_bits_by_document),
            "semantic_bits_mean_covered_document": sum(semantic_bits_by_document)
            / covered_documents,
            "unused_physical_bits_are_not_counted_as_compression_gain": True,
        },
        "conditions": {
            "correct": correct,
            "shuffled": shuffled,
            "zero": zero,
            "random": random_control,
        },
        "bf16": bf16,
        "gates": gates,
        "stage0_pass": passed,
        "decision": (
            "admit_one_learned_language_interface_experiment"
            if passed
            else "close_exact_t24_xor_record"
        ),
        "claim_boundary": (
            "Exact raw-derived storage/read identity only; no natural query parsing, "
            "knowledge gain, physical Transformer export, or smarter-model claim."
        ),
        "provenance": {
            "preregistration_sha256": sha256_file(PREREGISTRATION),
            "source_sha256": sha256_file(Path(__file__).resolve()),
            "source_stage0_sha256": sha256_file(Path(source.__file__).resolve()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
