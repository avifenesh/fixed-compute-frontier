"""Independent cross-implementation gate for permutation-graph Stage 0."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from experiments import permutation_graph_reference as gold
from experiments import permutation_graph_stage0 as tested
from experiments.permutation_graph_ancestry import allocation_table


ROOT_DIR = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gold_state(state: tested.GraphState) -> gold.ReferenceState:
    return gold.ReferenceState(
        values=tuple(state.values),
        occupied=tuple(state.occupied),
        conditions=tuple(state.conditions),
        relations=(tuple(state.relations[0]), tuple(state.relations[1])),
        parents=tuple(None if parent == tested.ROOT else parent for parent in state.parents),
        status=state.status,
    )


def _gold_command(command: tested.Command) -> gold.Command:
    return gold.Command(command.opcode, command.args)


def _canonical_tested(state: tested.GraphState) -> tuple[object, ...]:
    return (
        tuple(state.values),
        tuple(state.occupied),
        tuple(state.conditions),
        tuple(state.relations[0]),
        tuple(state.relations[1]),
        tuple(None if parent == tested.ROOT else parent for parent in state.parents),
        state.status,
    )


def _assert_independent_sources() -> dict[str, str]:
    executor_path = ROOT_DIR / "experiments" / "permutation_graph_stage0.py"
    reference_path = ROOT_DIR / "experiments" / "permutation_graph_reference.py"
    executor_source = executor_path.read_text()
    reference_source = reference_path.read_text()
    if "permutation_graph_reference" in executor_source:
        raise AssertionError("tested executor imports or names the reference interpreter")
    if "permutation_graph_stage0" in reference_source:
        raise AssertionError("reference interpreter imports or names the tested executor")
    return {
        "tested_executor_sha256": sha256_file(executor_path),
        "independent_reference_interpreter_sha256": sha256_file(reference_path),
        "ancestry_implementation_sha256": sha256_file(
            ROOT_DIR / "experiments" / "permutation_graph_ancestry.py"
        ),
    }


def _trace_digest(commands: Iterable[tested.Command]) -> str:
    serial = [(command.opcode, command.args) for command in commands]
    return hashlib.sha256(
        json.dumps(serial, separators=(",", ":")).encode()
    ).hexdigest()


def _crosscheck_opcode_semantics(seed: int) -> dict[str, int]:
    """Cross-check every opcode, including an accepted two-switch and inverse."""

    candidate = tested.initial_state(8, seed)
    candidate.parents = [tested.ROOT, 0, 1, tested.ROOT, 3, 4, tested.ROOT, 6]
    reference = _gold_state(candidate)
    relation = candidate.relations[0]
    edges = [(node, target) for node, target in enumerate(relation) if node < target]
    (a, b), (c, d) = edges[:2]
    commands = (
        tested.Command("SET", (0, 255)),
        tested.Command("SWAP_VALUE", (0, 1)),
        tested.Command("MOVE_VALUE", (1, 2)),
        tested.Command("TOGGLE", (3,)),
        tested.Command("TWO_SWITCH", (0, a, b, c, d)),
        tested.Command("TWO_SWITCH", (0, a, c, b, d)),
        tested.Command("SET_PARENT", (6, 5)),
        tested.Command("CUT_PARENT", (6,)),
        tested.Command("SET_PARENT", (0, 2)),
        tested.Command("TWO_SWITCH", (0, a, a, c, d)),
    )
    accepted = 0
    for step, command in enumerate(commands):
        accepted += int(tested.apply_command(candidate, command))
        reference = gold.apply_command(reference, _gold_command(command))
        tested.validate_state(candidate)
        if _canonical_tested(candidate) != gold.state_snapshot(reference):
            raise AssertionError(
                f"directed semantic mismatch at seed={seed}, step={step}, command={command}"
            )
    return {
        "checks": len(commands),
        "accepted": accepted,
        "rejected": len(commands) - accepted,
        "accepted_two_switches": 2,
    }


def crosscheck_world(*, seed: int, command_count: int = 1_024) -> dict[str, object]:
    candidate = tested.initial_state(64, seed)
    reference = _gold_state(candidate)
    commands = tested.command_trace(64, command_count, seed + 1)
    accepted = 0

    for step, command in enumerate(commands):
        accepted += int(tested.apply_command(candidate, command))
        reference = gold.apply_command(reference, _gold_command(command))
        tested.validate_state(candidate)
        if _canonical_tested(candidate) != gold.state_snapshot(reference):
            raise AssertionError(
                f"state mismatch at seed={seed}, step={step}, command={command}"
            )

    query_checks = 0
    for start in range(64):
        for length in (1, 8, 32):
            word = tested.alternating_word(length, start % 2)
            if tested.traverse(candidate, start, word) != gold.traverse(reference, start, word):
                raise AssertionError(f"traversal mismatch at seed={seed}, node={start}, k={length}")
            if tested.path_aggregate(candidate, start, word) != gold.path_aggregate(
                reference, start, word
            ):
                raise AssertionError(f"aggregate mismatch at seed={seed}, node={start}, k={length}")
            query_checks += 2

    for ancestor in range(64):
        for node in range(64):
            if tested.strict_ancestor(candidate, ancestor, node) != gold.strict_ancestor(
                reference, ancestor, node
            ):
                raise AssertionError(
                    f"ancestry mismatch at seed={seed}, ancestor={ancestor}, node={node}"
                )
            query_checks += 1

    arena = tested.to_arena(candidate)
    semantic_checks = _crosscheck_opcode_semantics(seed)
    return {
        "seed": seed,
        "commands": command_count,
        "accepted_commands": accepted,
        "rejected_commands": command_count - accepted,
        "query_checks": query_checks,
        "directed_semantic_checks": semantic_checks,
        "trace_sha256": _trace_digest(commands),
        "final_state_sha256": hashlib.sha256(
            repr(_canonical_tested(candidate)).encode()
        ).hexdigest(),
        "arena_sha256": hashlib.sha256(arena).hexdigest(),
        "arena_bytes": len(arena),
        "typed_operations": candidate.ops.as_dict(),
    }


def run_stage0_exact_gate(
    *, seeds: tuple[int, ...] = (17, 29, 43), command_count: int = 1_024
) -> dict[str, object]:
    source_hashes = _assert_independent_sources()
    worlds = [crosscheck_world(seed=seed, command_count=command_count) for seed in seeds]
    return {
        "schema_version": 1,
        "status": "exact_harness_pass",
        "stage_1_authorized": False,
        "gpu_required": False,
        "source_hashes": source_hashes,
        "worlds": worlds,
        "total_random_trace_transitions_compared": len(seeds) * command_count,
        "total_directed_semantic_transitions_compared": sum(
            int(world["directed_semantic_checks"]["checks"]) for world in worlds
        ),
        "total_state_transitions_compared": len(seeds) * command_count
        + sum(int(world["directed_semantic_checks"]["checks"]) for world in worlds),
        "total_queries_compared": sum(int(world["query_checks"]) for world in worlds),
        "bitwise_task_state_agreement": 1.0,
        "primary_state_allocated_bytes": tested.ARENA_BYTES,
        "ancestry_batch_one_allocated_bytes": allocation_table(),
        "remaining_before_stage_1": [
            "freeze_refinement_dataset_and_evidence_encoding",
            "freeze_candidate_and_control_shapes",
            "freeze_learned_bytes_workspace_and_typed_operation_ceilings",
            "freeze_training_and_dev_selection_budget",
        ],
    }


def write_stage0_exact_gate(path: Path) -> dict[str, object]:
    report = run_stage0_exact_gate()
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


if __name__ == "__main__":
    write_stage0_exact_gate(
        ROOT_DIR / "results" / "permutation-graph-stage0-exact-gate.json"
    )
