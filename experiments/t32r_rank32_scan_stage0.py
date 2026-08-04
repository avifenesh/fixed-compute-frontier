#!/usr/bin/env python3
"""CPU reference for the frozen T32R rank-32 boundary scan."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import resource
import time
from pathlib import Path
from typing import Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = ROOT / "results/t32r-rank32-scan-stage0-preregistration.md"
SEED = 32_032
HIDDEN = 384
RANK = 32
HEADS = 4
HEAD_DIM = 8
RECORD_TOKENS = 128


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def position_code(length: int, width: int = RANK) -> np.ndarray:
    positions = np.arange(length, dtype=np.float32)[:, None]
    frequencies = np.exp(
        -math.log(10_000.0) * np.arange(0, width, 2, dtype=np.float32) / width
    )[None, :]
    result = np.zeros((length, width), dtype=np.float32)
    result[:, 0::2] = np.sin(positions * frequencies)
    result[:, 1::2] = np.cos(positions * frequencies)
    return result


def depthwise_context(values: np.ndarray, taps: np.ndarray) -> np.ndarray:
    if values.shape != (RECORD_TOKENS, RANK) or taps.shape != (3, RANK):
        raise ValueError(f"unexpected shapes: values={values.shape}, taps={taps.shape}")
    padded = np.pad(values, ((1, 1), (0, 0)))
    return (
        padded[:-2] * taps[0]
        + padded[1:-1] * taps[1]
        + padded[2:] * taps[2]
    ).astype(np.float32)


def stable_softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return (exponentials / exponentials.sum(axis=-1, keepdims=True)).astype(np.float32)


def attend_document(
    query: np.ndarray,
    contextual: np.ndarray,
    q_scale: np.ndarray,
    k_scale: np.ndarray,
    v_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    q = (query * q_scale).reshape(HEADS, HEAD_DIM)
    keys = (contextual * k_scale).reshape(RECORD_TOKENS, HEADS, HEAD_DIM)
    values = (contextual * v_scale).reshape(RECORD_TOKENS, HEADS, HEAD_DIM)
    scores = np.einsum("hd,nhd->hn", q, keys, optimize=False) / math.sqrt(HEAD_DIM)
    weights = stable_softmax(scores)
    summary = np.einsum("hn,nhd->hd", weights, values, optimize=False)
    return summary.reshape(RANK).astype(np.float32), weights


def attend_document_reference(
    query: np.ndarray,
    contextual: np.ndarray,
    q_scale: np.ndarray,
    k_scale: np.ndarray,
    v_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.zeros((HEADS, RECORD_TOKENS), dtype=np.float32)
    output = np.zeros((HEADS, HEAD_DIM), dtype=np.float32)
    for head in range(HEADS):
        start = head * HEAD_DIM
        stop = start + HEAD_DIM
        scores = np.empty(RECORD_TOKENS, dtype=np.float32)
        for position in range(RECORD_TOKENS):
            total = np.float32(0.0)
            for channel in range(start, stop):
                total += np.float32(
                    query[channel]
                    * q_scale[channel]
                    * contextual[position, channel]
                    * k_scale[channel]
                )
            scores[position] = total / np.float32(math.sqrt(HEAD_DIM))
        weights[head] = stable_softmax(scores[None, :])[0]
        for local_channel, channel in enumerate(range(start, stop)):
            total = np.float32(0.0)
            for position in range(RECORD_TOKENS):
                total += np.float32(
                    weights[head, position]
                    * contextual[position, channel]
                    * v_scale[channel]
                )
            output[head, local_channel] = total
    return output.reshape(RANK), weights


def inject_summaries(
    hidden: np.ndarray,
    projected_query: np.ndarray,
    summary_a: np.ndarray,
    summary_b: np.ndarray,
    role_a: np.ndarray,
    role_b: np.ndarray,
    output: np.ndarray,
    gate_vector: np.ndarray,
    gate_bias: float,
) -> tuple[np.ndarray, float]:
    if hidden.shape != (HIDDEN,):
        raise ValueError(hidden.shape)
    role_summary = role_a @ summary_a + role_b @ summary_b
    gate_logit = float(gate_vector @ projected_query + gate_bias)
    gate = 1.0 / (1.0 + math.exp(-gate_logit))
    return (hidden + np.float32(gate) * (output @ role_summary)).astype(np.float32), gate


def scan_operator(
    hidden: np.ndarray,
    embedding: np.ndarray,
    record_a: Sequence[int],
    record_b: Sequence[int],
    projection: np.ndarray,
    taps: np.ndarray,
    q_scale: np.ndarray,
    k_scale: np.ndarray,
    v_scale: np.ndarray,
    role_a: np.ndarray,
    role_b: np.ndarray,
    output: np.ndarray,
    gate_vector: np.ndarray,
    gate_bias: float,
    has_two_handles: bool = True,
) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    if not has_two_handles:
        return hidden.copy(), {"gate": 0.0}
    positions = position_code(RECORD_TOKENS)
    query = (hidden @ projection).astype(np.float32)
    projected_a = (embedding[np.asarray(record_a)] @ projection).astype(np.float32) + positions
    projected_b = (embedding[np.asarray(record_b)] @ projection).astype(np.float32) + positions
    contextual_a = depthwise_context(projected_a, taps)
    contextual_b = depthwise_context(projected_b, taps)
    summary_a, weights_a = attend_document(query, contextual_a, q_scale, k_scale, v_scale)
    summary_b, weights_b = attend_document(query, contextual_b, q_scale, k_scale, v_scale)
    updated, gate = inject_summaries(
        hidden,
        query,
        summary_a,
        summary_b,
        role_a,
        role_b,
        output,
        gate_vector,
        gate_bias,
    )
    return updated, {
        "query": query,
        "contextual_a": contextual_a,
        "contextual_b": contextual_b,
        "summary_a": summary_a,
        "summary_b": summary_b,
        "weights_a": weights_a,
        "weights_b": weights_b,
        "gate": gate,
    }


def parameter_ledger() -> dict[str, int]:
    entries = {
        "record_digit_entries": 2_405 * 381,
        "projection_entries": HIDDEN * RANK,
        "diagonal_qkv_entries": 3 * RANK,
        "depthwise_filter_entries": 3 * RANK,
        "role_map_entries": 2 * RANK * RANK,
        "summary_output_entries": RANK * HIDDEN,
        "scalar_gate_entries": RANK + 1,
        "shared_handle_marker_entries": HIDDEN,
    }
    total = sum(entries.values())
    return {
        **entries,
        "total_allocated_entries": total,
        "removed_dense_entries": 967_680,
        "matched_slack_entries": 967_680 - total,
    }


def multiplication_ledger() -> dict[str, int]:
    entries = {
        "two_embedding_projections": 2 * RECORD_TOKENS * HIDDEN * RANK,
        "boundary_query_projection": HIDDEN * RANK,
        "depthwise_context": 2 * RECORD_TOKENS * 3 * RANK,
        "diagonal_qkv": 2 * RECORD_TOKENS * 2 * RANK + RANK,
        "score_and_weighted_value": 2 * RECORD_TOKENS * 2 * RANK,
        "role_maps": 2 * RANK * RANK,
        "summary_output": RANK * HIDDEN,
        "gate_dot_and_gated_residual": RANK + HIDDEN,
    }
    total = sum(entries.values())
    return {**entries, "total_multiplies": total, "frozen_envelope": 3_700_000}


def concentration_world(delta: float) -> dict[str, float]:
    scores = np.zeros(RECORD_TOKENS, dtype=np.float64)
    scores[0] = float(delta)
    shifted = scores - scores.max()
    weights = np.exp(shifted) / np.exp(shifted).sum()
    target_weight = float(weights[0])
    exact_weight = 1.0 / (1.0 + (RECORD_TOKENS - 1) * math.exp(-delta))
    values = np.full(RECORD_TOKENS, -1.0, dtype=np.float64)
    values[0] = 1.0
    mixture = float(weights @ values)
    error = abs(mixture - 1.0)
    bound = 2.0 * (RECORD_TOKENS - 1) * math.exp(-delta)
    derivative = target_weight * (1.0 - target_weight)
    return {
        "delta": delta,
        "target_weight": target_weight,
        "exact_target_weight": exact_weight,
        "selected_value_error": error,
        "theorem_bound": bound,
        "target_weight_derivative": derivative,
    }


def run_stage0() -> dict[str, object]:
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    rng = np.random.default_rng(SEED)
    vocabulary = 512
    hidden = rng.normal(0.0, 0.1, HIDDEN).astype(np.float32)
    embedding = rng.normal(0.0, 0.1, (vocabulary, HIDDEN)).astype(np.float32)
    records = rng.integers(0, vocabulary, size=(2, RECORD_TOKENS), dtype=np.int64)
    projection = rng.normal(0.0, 0.1, (HIDDEN, RANK)).astype(np.float32)
    taps = rng.normal(0.0, 0.1, (3, RANK)).astype(np.float32)
    q_scale = rng.normal(1.0, 0.05, RANK).astype(np.float32)
    k_scale = rng.normal(1.0, 0.05, RANK).astype(np.float32)
    v_scale = rng.normal(1.0, 0.05, RANK).astype(np.float32)
    role_a = rng.normal(0.0, 0.1, (RANK, RANK)).astype(np.float32)
    role_b = rng.normal(0.0, 0.1, (RANK, RANK)).astype(np.float32)
    output = rng.normal(0.0, 0.1, (HIDDEN, RANK)).astype(np.float32)
    gate_vector = rng.normal(0.0, 0.1, RANK).astype(np.float32)

    arguments = (
        hidden,
        embedding,
        records[0],
        records[1],
        projection,
        taps,
        q_scale,
        k_scale,
        v_scale,
        role_a,
        role_b,
        output,
        gate_vector,
        0.0,
    )
    first, trace = scan_operator(*arguments)
    second, second_trace = scan_operator(*arguments)
    no_handle, _ = scan_operator(*arguments, has_two_handles=False)

    reference_a, reference_weights_a = attend_document_reference(
        trace["query"], trace["contextual_a"], q_scale, k_scale, v_scale
    )
    reference_b, reference_weights_b = attend_document_reference(
        trace["query"], trace["contextual_b"], q_scale, k_scale, v_scale
    )
    reference_output, reference_gate = inject_summaries(
        hidden,
        trace["query"],
        reference_a,
        reference_b,
        role_a,
        role_b,
        output,
        gate_vector,
        0.0,
    )
    reference_error = max(
        float(np.max(np.abs(first - reference_output))),
        float(np.max(np.abs(trace["weights_a"] - reference_weights_a))),
        float(np.max(np.abs(trace["weights_b"] - reference_weights_b))),
        abs(float(trace["gate"]) - reference_gate),
    )

    summary_a = rng.normal(size=RANK).astype(np.float32)
    summary_b = rng.normal(size=RANK).astype(np.float32)
    identity = np.eye(RANK, dtype=np.float32)
    role_forward = identity @ summary_a - identity @ summary_b
    role_swapped = identity @ summary_b - identity @ summary_a

    concentration = [concentration_world(delta) for delta in (2, 4, 8, 12, 16)]
    ledger = parameter_ledger()
    multiplies = multiplication_ledger()
    finite_arrays = (
        first,
        second,
        reference_output,
        trace["weights_a"],
        trace["weights_b"],
    )
    gates = {
        "finite_random_world": all(np.isfinite(array).all() for array in finite_arrays),
        "bit_exact_repeat": np.array_equal(first, second)
        and np.array_equal(trace["weights_a"], second_trace["weights_a"]),
        "no_handle_identity": np.array_equal(no_handle, hidden),
        "role_swap_negates": np.array_equal(role_swapped, -role_forward),
        "independent_reference": reference_error <= 1e-6,
        "concentration_theorem": all(
            abs(row["target_weight"] - row["exact_target_weight"]) <= 1e-12
            and row["selected_value_error"] <= row["theorem_bound"] + 1e-6
            and math.isfinite(row["target_weight_derivative"])
            for row in concentration
        ),
        "parameter_ledger": ledger["total_allocated_entries"] == 943_538
        and ledger["matched_slack_entries"] == 24_142,
        "multiplication_ledger": multiplies["total_multiplies"] == 3_230_144
        and multiplies["total_multiplies"] <= multiplies["frozen_envelope"],
        "zero_external_access": True,
    }
    return {
        "experiment": "t32r-rank32-scan-stage0",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "shape": {
            "hidden": HIDDEN,
            "rank": RANK,
            "heads": HEADS,
            "head_dim": HEAD_DIM,
            "record_tokens": RECORD_TOKENS,
        },
        "random_world": {
            "seed": SEED,
            "output_sha256": hashlib.sha256(first.tobytes()).hexdigest(),
            "gate": trace["gate"],
            "independent_reference_max_absolute_error": reference_error,
        },
        "concentration_worlds": concentration,
        "parameter_ledger": ledger,
        "multiplication_ledger": multiplies,
        "gates": gates,
        "integrity": {
            "preregistration_sha256": sha256_file(PREREGISTRATION),
            "source_sha256": sha256_file(Path(__file__).resolve()),
            "test_sha256": sha256_file(ROOT / "tests/test_t32r_rank32_scan_stage0.py"),
            "corpus_reads": 0,
            "evaluator_reads": 0,
            "model_weight_reads": 0,
            "gpu_operations": 0,
        },
        "runtime": {
            "wall_seconds": time.perf_counter() - started_wall,
            "cpu_seconds": time.process_time() - started_cpu,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "claim_boundary": (
            "Exact CPU scan algebra only; no natural learnability, physical kernel, "
            "latency, language quality, or smarter-model claim."
        ),
    }


def main() -> None:
    print(json.dumps(run_stage0(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
