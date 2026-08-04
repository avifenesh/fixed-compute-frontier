"""Exact Stage-0 gate for a quotient-group memory head.

The candidate stores equivalence classes plus a relative element of a finite
group.  A weighted disjoint-set forest is checked against an independent graph
solver on non-commutative S3 relations.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from collections import deque
from pathlib import Path
from typing import Literal


Permutation = tuple[int, ...]
LinkStatus = Literal["merged", "consistent", "contradiction"]
ROOT_DIR = Path(__file__).resolve().parents[1]
IDENTITY_S3: Permutation = (0, 1, 2)


def compose(left: Permutation, right: Permutation) -> Permutation:
    """Return left after right."""

    return tuple(left[right[index]] for index in range(len(left)))


def inverse(value: Permutation) -> Permutation:
    result = [0] * len(value)
    for source, destination in enumerate(value):
        result[destination] = source
    return tuple(result)


def s3_elements() -> tuple[Permutation, ...]:
    return tuple(itertools.permutations(range(3)))


class QuotientGroupMemory:
    """Weighted union-find with the invariant z_x = potential[x] * z_parent."""

    def __init__(self, size: int, identity: Permutation) -> None:
        self.identity = identity
        self.parent = list(range(size))
        self.rank = [0] * size
        self.potential = [identity] * size

    def find(self, item: int) -> tuple[int, Permutation]:
        parent = self.parent[item]
        if parent == item:
            return item, self.identity
        root, parent_to_root = self.find(parent)
        item_to_root = compose(self.potential[item], parent_to_root)
        self.parent[item] = root
        self.potential[item] = item_to_root
        return root, item_to_root

    def relation(self, source: int, destination: int) -> Permutation | None:
        source_root, source_to_root = self.find(source)
        destination_root, destination_to_root = self.find(destination)
        if source_root != destination_root:
            return None
        return compose(destination_to_root, inverse(source_to_root))

    def link(
        self, source: int, destination: int, relation: Permutation
    ) -> LinkStatus:
        """Assert z_destination = relation * z_source."""

        source_root, source_to_root = self.find(source)
        destination_root, destination_to_root = self.find(destination)
        if source_root == destination_root:
            return (
                "consistent"
                if compose(destination_to_root, inverse(source_to_root)) == relation
                else "contradiction"
            )

        if self.rank[source_root] >= self.rank[destination_root]:
            self.parent[destination_root] = source_root
            self.potential[destination_root] = compose(
                compose(inverse(destination_to_root), relation), source_to_root
            )
            if self.rank[source_root] == self.rank[destination_root]:
                self.rank[source_root] += 1
        else:
            self.parent[source_root] = destination_root
            self.potential[source_root] = compose(
                compose(inverse(source_to_root), inverse(relation)),
                destination_to_root,
            )
        return "merged"


class GraphReference:
    """Independent breadth-first relation solver over the asserted edge graph."""

    def __init__(self, size: int, identity: Permutation) -> None:
        self.identity = identity
        self.edges: list[list[tuple[int, Permutation]]] = [
            [] for _ in range(size)
        ]

    def relation(self, source: int, destination: int) -> Permutation | None:
        queue = deque([(source, self.identity)])
        seen = {source}
        while queue:
            item, source_to_item = queue.popleft()
            if item == destination:
                return source_to_item
            for neighbour, item_to_neighbour in self.edges[item]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(
                        (neighbour, compose(item_to_neighbour, source_to_item))
                    )
        return None

    def link(
        self, source: int, destination: int, relation: Permutation
    ) -> LinkStatus:
        existing = self.relation(source, destination)
        if existing is not None:
            return "consistent" if existing == relation else "contradiction"
        self.edges[source].append((destination, relation))
        self.edges[destination].append((source, inverse(relation)))
        return "merged"


def labeled_partition_state_count(size: int, group_order: int) -> int:
    """Count partitions with one relative group label per non-root entity."""

    if size < 1 or group_order < 1:
        raise ValueError("size and group_order must be positive")
    stirling = [0] * (size + 1)
    stirling[0] = 1
    for elements in range(1, size + 1):
        next_row = [0] * (size + 1)
        for blocks in range(1, elements + 1):
            next_row[blocks] = stirling[blocks - 1] + blocks * stirling[blocks]
        stirling = next_row
    return sum(
        stirling[blocks] * group_order ** (size - blocks)
        for blocks in range(1, size + 1)
    )


def logical_state_ledger(size: int, group_order: int) -> dict[str, int | float | str]:
    if size < 2:
        raise ValueError("the distinguishing-bit ledger requires at least two entities")
    distinguishable_states = labeled_partition_state_count(size, group_order)
    minimum_bits = (distinguishable_states - 1).bit_length()
    parent_bits = size * math.ceil(math.log2(size))
    maximum_rank = math.floor(math.log2(size))
    rank_bits = size * math.ceil(math.log2(maximum_rank + 1))
    potential_bits = size * math.ceil(math.log2(group_order))
    represented_bits = parent_bits + rank_bits + potential_bits
    return {
        "logical_state_scope": "predeclared_dense_entity_slots_only",
        "entities": size,
        "group_order": group_order,
        "minimum_distinguishing_bits": minimum_bits,
        "parent_bits": parent_bits,
        "rank_bits": rank_bits,
        "potential_bits": potential_bits,
        "simple_packed_representation_bits": represented_bits,
        "simple_bit_packed_array_payload_bytes": math.ceil(represented_bits / 8),
        "representation_to_lower_bound_ratio": represented_bits / minimum_bits,
        "hypothetical_packed_uint32_uint8_uint8_array_payload_bytes": 6 * size,
    }


def crosscheck_trace(
    *, seed: int, size: int = 64, operations: int = 4_096
) -> dict[str, int]:
    rng = random.Random(seed)
    group = s3_elements()
    tested = QuotientGroupMemory(size, IDENTITY_S3)
    reference = GraphReference(size, IDENTITY_S3)
    counts = {"merged": 0, "consistent": 0, "contradiction": 0}
    relation_checks = 0

    for _ in range(operations):
        source = rng.randrange(size)
        destination = rng.randrange(size)
        existing = reference.relation(source, destination)
        if existing is None:
            proposed = rng.choice(group)
        elif rng.random() < 0.7:
            proposed = existing
        else:
            proposed = rng.choice(tuple(value for value in group if value != existing))

        expected_status = reference.link(source, destination, proposed)
        actual_status = tested.link(source, destination, proposed)
        if actual_status != expected_status:
            raise AssertionError("link status disagrees with graph reference")
        counts[actual_status] += 1

        for _ in range(4):
            left = rng.randrange(size)
            right = rng.randrange(size)
            if tested.relation(left, right) != reference.relation(left, right):
                raise AssertionError("relative group query disagrees with graph reference")
            relation_checks += 1

    return {
        "seed": seed,
        "operations": operations,
        "relation_checks": relation_checks,
        **counts,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_stage0_gate() -> dict[str, object]:
    traces = [crosscheck_trace(seed=seed) for seed in (17, 29, 43)]
    return {
        "schema_version": 2,
        "status": "algebra_cpu_pass_architecture_claim_closed",
        "candidate_number": None,
        "gpu_required": False,
        "method": "quotient_group_memory",
        "group": "non_commutative_S3",
        "group_operation_model": {
            "scope": "fixed_word_sized_S3",
            "potential_bits_per_entity": 3,
            "logical_compose": "constant_size_S3_permutation_composition",
            "logical_inverse": "constant_size_S3_permutation_inversion",
            "excluded_from_packed_array_payload": (
                "Python_object_overhead_allocator_and_executable_code"
            ),
            "variable_group_caveat": (
                "charge_the_group_representation_and_nonconstant_operation_cost"
            ),
        },
        "resource_model": {
            "machine": "online_word_RAM_or_cell_probe",
            "word_bits": "Theta(log2(entities))_and_large_enough_for_one_group_element",
            "charged_per_traversed_edge": (
                "parent_probe_plus_fixed_word_sized_group_operations"
            ),
            "request_state_theorem_scope": "predeclared_dense_entity_slots",
        },
        "operator": {
            "link": "assert z_destination = relation * z_source",
            "rel": "return relation or disconnected",
            "contradiction": "reject inconsistent relation without semantic mutation",
        },
        "state_ledger": logical_state_ledger(64, len(s3_elements())),
        "traces": traces,
        "total_operations": sum(trace["operations"] for trace in traces),
        "total_relation_checks": sum(trace["relation_checks"] for trace in traces),
        "source_hashes": {
            "executable_sha256": _sha256_file(
                ROOT_DIR / "experiments" / "quotient_group_memory.py"
            ),
            "focused_test_sha256": _sha256_file(
                ROOT_DIR / "tests" / "test_quotient_group_memory.py"
            ),
        },
        "decision": "retain_as_classical_typed_ram_primitive_not_architecture_candidate",
        "direct_collision": {
            "work": "ED-Batch_Algorithm_5",
            "mechanism": (
                "extended_union_find_with_parent_relative_noncommutative_"
                "permutations_and_compatibility_rejection"
            ),
            "scope_note": (
                "neural_systems_memory_planning_not_an_LLM_request_memory_head"
            ),
        },
        "excluded_accounting": [
            "language_or_entity_name_to_dense_slot_mapping",
            "active_slot_free_list_and_generation_metadata",
            "capacity_overflow_eviction_and_deletion_policy",
            "operation_stream_storage_and_evidence_or_provenance",
            "Python_object_allocator_and_interpreter_overhead",
            "packed_kernel_code_and_fixed_group_operation_tables",
        ],
        "not_proven": [
            "novel_data_structure_or_new_model_algebra",
            "learned_language_to_entity_and_relation_interface",
            "gpu_latency_or_energy_advantage",
            "general_language_model_quality_noninferiority",
            "separation_from_a_matched_recurrent_machine_with_random_access_memory",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(run_stage0_gate(), indent=2, sort_keys=True))
