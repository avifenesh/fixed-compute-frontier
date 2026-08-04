from __future__ import annotations

import hashlib
import itertools
import json
import math
import platform
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch


Pair = tuple[float, float]
TOKEN_CODEBOOK: tuple[Pair, ...] = (
    (0.25, -0.50),
    (0.50, 0.25),
    (0.75, 0.50),
    (0.9375, -0.25),
)
A_LEVELS: tuple[float, ...] = tuple(index / 16 for index in range(15)) + (1.0,)
B_LEVELS: tuple[float, ...] = (
    -2.0,
    -1.5,
    -1.25,
    -1.0,
    -0.75,
    -0.5,
    -0.25,
    -0.125,
    0.0,
    0.125,
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
)
INCOMING_STATES: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)
TOLERANCE = 1e-12
VECTOR_WIDTH = 110
DOCUMENTS = 2_405
PLANE_CELLS = 220
ENTRY_CAP = 743_734


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_pair(pair: Pair, state: float) -> float:
    return pair[0] * state + pair[1]


def compose(after: Pair, before: Pair) -> Pair:
    """Return the pair for applying before and then after."""
    a_after, b_after = after
    a_before, b_before = before
    return a_after * a_before, a_after * b_before + b_after


def sequential_state(sequence: Sequence[Pair], state: float) -> float:
    for pair in sequence:
        state = apply_pair(pair, state)
    return state


def fold_left(sequence: Sequence[Pair]) -> Pair:
    result: Pair = (1.0, 0.0)
    for pair in sequence:
        result = compose(pair, result)
    return result


def fold_balanced(sequence: Sequence[Pair]) -> Pair:
    if not sequence:
        return 1.0, 0.0
    if len(sequence) == 1:
        return sequence[0]
    middle = len(sequence) // 2
    left = fold_balanced(sequence[:middle])
    right = fold_balanced(sequence[middle:])
    return compose(right, left)


def fold_black_box(sequence: Sequence[Pair]) -> Pair:
    b_value = sequential_state(sequence, 0.0)
    a_value = sequential_state(sequence, 1.0) - b_value
    return a_value, b_value


def all_sequences(max_length: int) -> tuple[tuple[Pair, ...], ...]:
    sequences: list[tuple[Pair, ...]] = []
    for length in range(max_length + 1):
        sequences.extend(itertools.product(TOKEN_CODEBOOK, repeat=length))
    return tuple(sequences)


def nearest_level(value: float, levels: Sequence[float]) -> tuple[int, float, bool]:
    clipped = value < levels[0] or value > levels[-1]
    distances = [abs(value - level) for level in levels]
    index = min(range(len(levels)), key=lambda candidate: (distances[candidate], candidate))
    return index, levels[index], clipped


def quantize_pair(pair: Pair) -> tuple[tuple[int, int], Pair, tuple[bool, bool]]:
    a_index, a_value, a_clipped = nearest_level(pair[0], A_LEVELS)
    b_index, b_value, b_clipped = nearest_level(pair[1], B_LEVELS)
    return (a_index, b_index), (a_value, b_value), (a_clipped, b_clipped)


def max_pair_error(left: Pair, right: Pair) -> float:
    return max(abs(left[0] - right[0]), abs(left[1] - right[1]))


def check_exact_forms() -> dict[str, Any]:
    sequences = all_sequences(6)
    maximum_pair_error = 0.0
    maximum_state_error = 0.0
    mismatches = 0
    for sequence in sequences:
        left = fold_left(sequence)
        balanced = fold_balanced(sequence)
        black_box = fold_black_box(sequence)
        maximum_pair_error = max(
            maximum_pair_error,
            max_pair_error(left, balanced),
            max_pair_error(left, black_box),
        )
        for state in INCOMING_STATES:
            exact = sequential_state(sequence, state)
            errors = (
                abs(exact - apply_pair(left, state)),
                abs(exact - apply_pair(balanced, state)),
                abs(exact - apply_pair(black_box, state)),
            )
            maximum_state_error = max(maximum_state_error, *errors)
            mismatches += sum(error > TOLERANCE for error in errors)
    return {
        "sequences": len(sequences),
        "state_evaluations": len(sequences) * len(INCOMING_STATES),
        "max_pair_error": maximum_pair_error,
        "max_state_error": maximum_state_error,
        "mismatches": mismatches,
    }


def check_associativity() -> dict[str, Any]:
    pair_domain = set(TOKEN_CODEBOOK)
    pair_domain.update(
        compose(right, left) for left in TOKEN_CODEBOOK for right in TOKEN_CODEBOOK
    )
    ordered = tuple(sorted(pair_domain))
    maximum_error = 0.0
    mismatches = 0
    for first, second, third in itertools.product(ordered, repeat=3):
        left = compose(third, compose(second, first))
        right = compose(compose(third, second), first)
        error = max_pair_error(left, right)
        maximum_error = max(maximum_error, error)
        mismatches += int(error > TOLERANCE)
    identity_errors = []
    for pair in ordered:
        identity_errors.extend(
            (max_pair_error(compose((1.0, 0.0), pair), pair),
             max_pair_error(compose(pair, (1.0, 0.0)), pair))
        )
    return {
        "pair_domain": len(ordered),
        "ordered_triples": len(ordered) ** 3,
        "max_associativity_error": maximum_error,
        "max_identity_error": max(identity_errors, default=0.0),
        "mismatches": mismatches + sum(error > TOLERANCE for error in identity_errors),
    }


def check_title_substitution() -> dict[str, Any]:
    short_sequences = all_sequences(2)
    maximum_error = 0.0
    mismatches = 0
    cases = 0
    for prefix, document, suffix, title_pair, state in itertools.product(
        short_sequences,
        short_sequences,
        short_sequences,
        TOKEN_CODEBOOK,
        INCOMING_STATES,
    ):
        expanded = sequential_state(prefix + (title_pair,) + document + suffix, state)
        compiled_title = compose(fold_left(document), title_pair)
        compiled = sequential_state(prefix + (compiled_title,) + suffix, state)
        error = abs(expanded - compiled)
        maximum_error = max(maximum_error, error)
        mismatches += int(error > TOLERANCE)
        cases += 1

    one_sequences = all_sequences(1)
    two_title_cases = 0
    for prefix, first_doc, middle, second_doc, suffix, first_title, second_title, state in itertools.product(
        one_sequences,
        one_sequences,
        one_sequences,
        one_sequences,
        one_sequences,
        TOKEN_CODEBOOK,
        TOKEN_CODEBOOK,
        INCOMING_STATES,
    ):
        expanded_sequence = (
            prefix
            + (first_title,)
            + first_doc
            + middle
            + (second_title,)
            + second_doc
            + suffix
        )
        compiled_sequence = (
            prefix
            + (compose(fold_left(first_doc), first_title),)
            + middle
            + (compose(fold_left(second_doc), second_title),)
            + suffix
        )
        error = abs(
            sequential_state(expanded_sequence, state)
            - sequential_state(compiled_sequence, state)
        )
        maximum_error = max(maximum_error, error)
        mismatches += int(error > TOLERANCE)
        two_title_cases += 1

    unequal_pairs = 0
    insensitive_pairs = 0
    for first_doc, second_doc in itertools.combinations(short_sequences, 2):
        first_summary = fold_left(first_doc)
        second_summary = fold_left(second_doc)
        if max_pair_error(first_summary, second_summary) <= TOLERANCE:
            continue
        unequal_pairs += 1
        sensitive = any(
            abs(apply_pair(first_summary, state) - apply_pair(second_summary, state))
            > TOLERANCE
            for state in INCOMING_STATES
        )
        insensitive_pairs += int(not sensitive)
    return {
        "single_title_cases": cases,
        "two_title_cases": two_title_cases,
        "max_error": maximum_error,
        "mismatches": mismatches,
        "unequal_document_summary_pairs": unequal_pairs,
        "insensitive_wrong_document_pairs": insensitive_pairs,
    }


def check_codec() -> dict[str, Any]:
    decoded_a = torch.tensor(A_LEVELS, dtype=torch.float32, device="cpu").to(torch.bfloat16).to(torch.float64)
    decoded_b = torch.tensor(B_LEVELS, dtype=torch.float32, device="cpu").to(torch.bfloat16).to(torch.float64)
    a_roundtrip_errors = [abs(float(value) - expected) for value, expected in zip(decoded_a, A_LEVELS)]
    b_roundtrip_errors = [abs(float(value) - expected) for value, expected in zip(decoded_b, B_LEVELS)]
    index_failures = 0
    for index, value in enumerate(decoded_a.tolist()):
        recovered, _, _ = nearest_level(float(value), A_LEVELS)
        index_failures += int(recovered != index)
    for index, value in enumerate(decoded_b.tolist()):
        recovered, _, _ = nearest_level(float(value), B_LEVELS)
        index_failures += int(recovered != index)

    documents = all_sequences(6)
    suffixes = all_sequences(3)
    clipping = 0
    bound_violations = 0
    cases = 0
    maximum_error = 0.0
    maximum_bound = 0.0
    worst_witness: dict[str, Any] = {}
    for document in documents:
        exact_pair = fold_left(document)
        indices, quantized_pair, clipped = quantize_pair(exact_pair)
        clipping += sum(clipped)
        epsilon_a = abs(quantized_pair[0] - exact_pair[0])
        epsilon_b = abs(quantized_pair[1] - exact_pair[1])
        for suffix in suffixes:
            alpha_suffix = math.prod(abs(pair[0]) for pair in suffix)
            for state in INCOMING_STATES:
                exact_final = sequential_state(suffix, apply_pair(exact_pair, state))
                quantized_final = sequential_state(suffix, apply_pair(quantized_pair, state))
                error = abs(exact_final - quantized_final)
                bound = alpha_suffix * (epsilon_a * abs(state) + epsilon_b)
                bound_violations += int(error > bound + TOLERANCE)
                cases += 1
                maximum_bound = max(maximum_bound, bound)
                if error > maximum_error:
                    maximum_error = error
                    worst_witness = {
                        "document_length": len(document),
                        "suffix_length": len(suffix),
                        "incoming_state": state,
                        "exact_pair": exact_pair,
                        "quantized_indices": indices,
                        "quantized_pair": quantized_pair,
                        "error": error,
                        "bound": bound,
                    }
    return {
        "a_levels": len(A_LEVELS),
        "b_levels": len(B_LEVELS),
        "max_bf16_a_error": max(a_roundtrip_errors),
        "max_bf16_b_error": max(b_roundtrip_errors),
        "bf16_index_failures": index_failures,
        "quantized_cases": cases,
        "clipped_coordinates": clipping,
        "bound_violations": bound_violations,
        "max_final_state_error": maximum_error,
        "max_bound": maximum_bound,
        "worst_witness": worst_witness,
    }


def compose_vector(
    after: tuple[np.ndarray, np.ndarray],
    before: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    return after[0] * before[0], after[0] * before[1] + after[1]


def fold_vector_left(
    sequence: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    result = (np.ones(VECTOR_WIDTH, dtype=np.float64), np.zeros(VECTOR_WIDTH, dtype=np.float64))
    for pair in sequence:
        result = compose_vector(pair, result)
    return result


def fold_vector_balanced(
    sequence: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    if not sequence:
        return np.ones(VECTOR_WIDTH, dtype=np.float64), np.zeros(VECTOR_WIDTH, dtype=np.float64)
    if len(sequence) == 1:
        return sequence[0]
    middle = len(sequence) // 2
    return compose_vector(
        fold_vector_balanced(sequence[middle:]),
        fold_vector_balanced(sequence[:middle]),
    )


def check_vector_and_dense_boundary() -> dict[str, Any]:
    vector_sequence: list[tuple[np.ndarray, np.ndarray]] = []
    for step in range(6):
        pairs = [TOKEN_CODEBOOK[(coordinate + step) % len(TOKEN_CODEBOOK)] for coordinate in range(VECTOR_WIDTH)]
        vector_sequence.append(
            (
                np.asarray([pair[0] for pair in pairs], dtype=np.float64),
                np.asarray([pair[1] for pair in pairs], dtype=np.float64),
            )
        )
    left = fold_vector_left(vector_sequence)
    balanced = fold_vector_balanced(vector_sequence)
    state = np.linspace(-2.0, 2.0, VECTOR_WIDTH, dtype=np.float64)
    sequential = state.copy()
    for a_value, b_value in vector_sequence:
        sequential = a_value * sequential + b_value
    folded = left[0] * state + left[1]
    vector_error = max(
        float(np.max(np.abs(left[0] - balanced[0]))),
        float(np.max(np.abs(left[1] - balanced[1]))),
        float(np.max(np.abs(sequential - folded))),
    )

    prefix = vector_sequence[:2]
    title = vector_sequence[2]
    document = vector_sequence[3:5]
    suffix = vector_sequence[5:]
    expanded = fold_vector_left(prefix + [title] + document + suffix)
    compiled_title = compose_vector(fold_vector_left(document), title)
    compiled = fold_vector_left(prefix + [compiled_title] + suffix)
    title_error = max(
        float(np.max(np.abs(expanded[0] - compiled[0]))),
        float(np.max(np.abs(expanded[1] - compiled[1]))),
    )

    dense_first = np.asarray([[1.0, 1.0], [0.0, 1.0]])
    dense_second = np.asarray([[1.0, 0.0], [1.0, 1.0]])
    commutator = dense_second @ dense_first - dense_first @ dense_second
    return {
        "width": VECTOR_WIDTH,
        "max_vector_error": vector_error,
        "max_title_substitution_error": title_error,
        "dense_noncommuting_witness_max": float(np.max(np.abs(commutator))),
    }


def check_ledger() -> dict[str, Any]:
    title_codes = 132_800
    address_gates = 76_960
    thresholds = DOCUMENTS
    up_constants = DOCUMENTS
    payload = DOCUMENTS * PLANE_CELLS
    total = title_codes + address_gates + thresholds + up_constants + payload
    return {
        "memory_state_width": VECTOR_WIDTH,
        "stored_pair_cells": 2 * VECTOR_WIDTH,
        "logical_bits_per_document": PLANE_CELLS * 4,
        "documents": DOCUMENTS,
        "payload_entries": payload,
        "title_code_entries": title_codes,
        "address_gate_entries": address_gates,
        "threshold_entries": thresholds,
        "up_constant_entries": up_constants,
        "total_entries": total,
        "cap_entries": ENTRY_CAP,
        "spare_entries": ENTRY_CAP - total,
        "recurrent_multiplies_per_token_candidate": VECTOR_WIDTH,
        "recurrent_additions_per_token_candidate": VECTOR_WIDTH,
        "recurrent_multiplies_per_token_control": VECTOR_WIDTH,
        "recurrent_additions_per_token_control": VECTOR_WIDTH,
    }


def run_stage0() -> dict[str, Any]:
    start = time.perf_counter()
    exact = check_exact_forms()
    associativity = check_associativity()
    title = check_title_substitution()
    codec = check_codec()
    vector = check_vector_and_dense_boundary()
    ledger = check_ledger()
    gates = {
        "exact_forms": exact["mismatches"] == 0
        and exact["max_pair_error"] <= TOLERANCE
        and exact["max_state_error"] <= TOLERANCE,
        "associativity_identity": associativity["mismatches"] == 0
        and associativity["max_associativity_error"] <= TOLERANCE
        and associativity["max_identity_error"] <= TOLERANCE,
        "title_substitution": title["mismatches"] == 0
        and title["max_error"] <= TOLERANCE,
        "wrong_document_sensitivity": title["insensitive_wrong_document_pairs"] == 0,
        "bf16_codec": codec["max_bf16_a_error"] == 0.0
        and codec["max_bf16_b_error"] == 0.0
        and codec["bf16_index_failures"] == 0,
        "no_clipping": codec["clipped_coordinates"] == 0,
        "quantization_bound": codec["bound_violations"] == 0,
        "vector_identity": vector["max_vector_error"] <= TOLERANCE
        and vector["max_title_substitution_error"] <= TOLERANCE,
        "dense_boundary": vector["dense_noncommuting_witness_max"] > 0.0,
        "ledger": ledger["stored_pair_cells"] == PLANE_CELLS
        and ledger["logical_bits_per_document"] == 880
        and ledger["total_entries"] == 743_670
        and ledger["spare_entries"] == 64
        and ledger["recurrent_multiplies_per_token_candidate"]
        == ledger["recurrent_multiplies_per_token_control"]
        and ledger["recurrent_additions_per_token_candidate"]
        == ledger["recurrent_additions_per_token_control"],
    }
    root = Path(__file__).resolve().parents[1]
    prereg_path = root / "results/title-triggered-affine-prefix-t29-stage0-preregistration.md"
    test_path = root / "tests/test_title_triggered_affine_prefix_t29_stage0.py"
    return {
        "experiment": "title-triggered-affine-prefix-t29-stage0",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "exact_forms": exact,
        "associativity": associativity,
        "title_substitution": title,
        "codec": codec,
        "vector_and_boundary": vector,
        "ledger": ledger,
        "gates": gates,
        "runtime": {
            "device": "cpu",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "wall_seconds": time.perf_counter() - start,
        },
        "integrity": {
            "preregistration_sha256": sha256_file(prereg_path),
            "source_sha256": sha256_file(Path(__file__).resolve()),
            "test_sha256": sha256_file(test_path),
        },
        "claim_boundary": (
            "Exact affine algebra, BF16 codec, error-bound, and static ledger only; "
            "no natural-information, learnability, physical-kernel, or production claim."
        ),
    }


def main() -> None:
    print(json.dumps(run_stage0(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
