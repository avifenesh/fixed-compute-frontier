from __future__ import annotations

import hashlib
import itertools
import json
import math
import platform
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


Group = tuple[int, int]
Operator = tuple[Group, np.ndarray, np.ndarray]

WIDTH = 109
DOCUMENTS = 2_405
PLANE_CELLS = 220
ENTRY_CAP = 743_734
TOLERANCE = 1e-12

IDENTITY_GROUP: Group = (1, 0)
SCALAR_CODEBOOK: tuple[tuple[float, float], ...] = (
    (0.25, -0.50),
    (0.50, 0.25),
    (0.75, 0.50),
    (0.9375, -0.25),
)
TOKEN_GROUPS: tuple[Group, ...] = (
    (1, 0),
    (1, 1),
    (-1, 0),
    (1, 17),
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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_group(group: Group) -> Group:
    sign, rotation = group
    if sign not in (-1, 1):
        raise ValueError(f"invalid dihedral sign: {sign}")
    return sign, rotation % WIDTH


def compose_group(after: Group, before: Group) -> Group:
    after_sign, after_rotation = normalize_group(after)
    before_sign, before_rotation = normalize_group(before)
    return (
        after_sign * before_sign,
        (after_rotation + after_sign * before_rotation) % WIDTH,
    )


def inverse_group(group: Group) -> Group:
    sign, rotation = normalize_group(group)
    return sign, (-sign * rotation) % WIDTH


def encode_group(group: Group) -> int:
    sign, rotation = normalize_group(group)
    return rotation if sign == 1 else WIDTH + rotation


def decode_group(code: int) -> Group:
    if not 0 <= int(code) < 2 * WIDTH:
        raise ValueError(f"invalid D_{WIDTH} code: {code}")
    return (1, int(code)) if code < WIDTH else (-1, int(code) - WIDTH)


def group_nibbles(group: Group) -> tuple[int, int]:
    code = encode_group(group)
    return code % 16, code // 16


def group_from_nibbles(low: int, high: int) -> Group:
    if not 0 <= int(low) < 16 or not 0 <= int(high) < 16:
        raise ValueError("group nibbles must be in [0,15]")
    return decode_group(int(low) + 16 * int(high))


def destination_indices(group: Group) -> np.ndarray:
    sign, rotation = normalize_group(group)
    source = np.arange(WIDTH, dtype=np.int64)
    return (sign * source + rotation) % WIDTH


def permute(group: Group, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.shape != (WIDTH,):
        raise ValueError(f"expected width-{WIDTH} vector, got {values.shape}")
    output = np.empty_like(values)
    output[destination_indices(group)] = values
    return output


def permutation_matrix(group: Group) -> np.ndarray:
    matrix = np.zeros((WIDTH, WIDTH), dtype=np.float64)
    source = np.arange(WIDTH)
    matrix[destination_indices(group), source] = 1.0
    return matrix


def identity_operator() -> Operator:
    return (
        IDENTITY_GROUP,
        np.ones(WIDTH, dtype=np.float64),
        np.zeros(WIDTH, dtype=np.float64),
    )


def token_operators() -> tuple[Operator, ...]:
    operators: list[Operator] = []
    for token_index, group in enumerate(TOKEN_GROUPS):
        pairs = [
            SCALAR_CODEBOOK[(coordinate + token_index) % len(SCALAR_CODEBOOK)]
            for coordinate in range(WIDTH)
        ]
        operators.append(
            (
                group,
                np.asarray([pair[0] for pair in pairs], dtype=np.float64),
                np.asarray([pair[1] for pair in pairs], dtype=np.float64),
            )
        )
    return tuple(operators)


TOKENS = token_operators()


def compose(after: Operator, before: Operator) -> Operator:
    after_group, after_a, after_b = after
    before_group, before_a, before_b = before
    return (
        compose_group(after_group, before_group),
        after_a * permute(after_group, before_a),
        after_a * permute(after_group, before_b) + after_b,
    )


def apply_operator(operator: Operator, state: np.ndarray) -> np.ndarray:
    group, a_value, b_value = operator
    return a_value * permute(group, state) + b_value


def sequential_state(sequence: Sequence[Operator], state: np.ndarray) -> np.ndarray:
    result = np.asarray(state, dtype=np.float64).copy()
    for operator in sequence:
        result = apply_operator(operator, result)
    return result


def fold_left(sequence: Sequence[Operator]) -> Operator:
    result = identity_operator()
    for operator in sequence:
        result = compose(operator, result)
    return result


def fold_balanced(sequence: Sequence[Operator]) -> Operator:
    if not sequence:
        return identity_operator()
    if len(sequence) == 1:
        group, a_value, b_value = sequence[0]
        return group, a_value.copy(), b_value.copy()
    middle = len(sequence) // 2
    return compose(
        fold_balanced(sequence[middle:]),
        fold_balanced(sequence[:middle]),
    )


def materialized_apply(operator: Operator, state: np.ndarray) -> np.ndarray:
    group, a_value, b_value = operator
    matrix = a_value[:, None] * permutation_matrix(group)
    return matrix @ state + b_value


def all_sequences(max_length: int) -> tuple[tuple[Operator, ...], ...]:
    sequences: list[tuple[Operator, ...]] = []
    for length in range(max_length + 1):
        sequences.extend(itertools.product(TOKENS, repeat=length))
    return tuple(sequences)


def incoming_states() -> tuple[np.ndarray, ...]:
    markers = np.zeros(WIDTH, dtype=np.float64)
    markers[0] = 1.0
    markers[1] = 2.0
    return (
        np.zeros(WIDTH, dtype=np.float64),
        np.ones(WIDTH, dtype=np.float64),
        np.linspace(-2.0, 2.0, WIDTH, dtype=np.float64),
        markers,
    )


STATES = incoming_states()


def operator_error(left: Operator, right: Operator) -> float:
    if normalize_group(left[0]) != normalize_group(right[0]):
        return math.inf
    return max(
        float(np.max(np.abs(left[1] - right[1]))),
        float(np.max(np.abs(left[2] - right[2]))),
    )


def check_group() -> dict[str, Any]:
    elements = tuple(decode_group(code) for code in range(2 * WIDTH))
    code_failures = sum(decode_group(encode_group(group)) != group for group in elements)
    nibble_failures = sum(group_from_nibbles(*group_nibbles(group)) != group for group in elements)
    identity_failures = 0
    inverse_failures = 0
    action_failures = 0
    table = np.empty((len(elements), len(elements)), dtype=np.int16)
    for after_code, after in enumerate(elements):
        for before_code, before in enumerate(elements):
            product = compose_group(after, before)
            table[after_code, before_code] = encode_group(product)
            identity_failures += int(
                compose_group(IDENTITY_GROUP, before) != before
                or compose_group(before, IDENTITY_GROUP) != before
            ) if after_code == 0 else 0
            expected_destinations = destination_indices(after)[destination_indices(before)]
            action_failures += int(
                not np.array_equal(destination_indices(product), expected_destinations)
            )
        inverse = inverse_group(after)
        inverse_failures += int(
            compose_group(after, inverse) != IDENTITY_GROUP
            or compose_group(inverse, after) != IDENTITY_GROUP
        )
    associativity_failures = 0
    for third_code in range(len(elements)):
        left = table[third_code, table]
        right = table[table[third_code, :, None], np.arange(len(elements))[None, :]]
        associativity_failures += int(np.count_nonzero(left != right))
    return {
        "elements": len(elements),
        "ordered_pairs": len(elements) ** 2,
        "ordered_triples": len(elements) ** 3,
        "code_failures": code_failures,
        "nibble_failures": nibble_failures,
        "identity_failures": identity_failures,
        "inverse_failures": inverse_failures,
        "action_failures": action_failures,
        "associativity_failures": associativity_failures,
    }


def check_exact_forms() -> dict[str, Any]:
    sequences = all_sequences(5)
    max_fold_error = 0.0
    max_state_error = 0.0
    mismatches = 0
    for sequence in sequences:
        left = fold_left(sequence)
        balanced = fold_balanced(sequence)
        fold_error = operator_error(left, balanced)
        max_fold_error = max(max_fold_error, fold_error)
        mismatches += int(fold_error > TOLERANCE)
        for state in STATES:
            sequential = sequential_state(sequence, state)
            folded = apply_operator(left, state)
            materialized = materialized_apply(left, state)
            error = max(
                float(np.max(np.abs(sequential - folded))),
                float(np.max(np.abs(sequential - materialized))),
            )
            max_state_error = max(max_state_error, error)
            mismatches += int(error > TOLERANCE)
    return {
        "sequences": len(sequences),
        "state_evaluations": len(sequences) * len(STATES),
        "max_fold_error": max_fold_error,
        "max_state_error": max_state_error,
        "mismatches": mismatches,
    }


def reconstruct_black_box(operator: Operator) -> Operator:
    zero = np.zeros(WIDTH, dtype=np.float64)
    b_value = apply_operator(operator, zero)
    matrix = np.empty((WIDTH, WIDTH), dtype=np.float64)
    for source in range(WIDTH):
        basis = np.zeros(WIDTH, dtype=np.float64)
        basis[source] = 1.0
        matrix[:, source] = apply_operator(operator, basis) - b_value
    destinations = np.argmax(np.abs(matrix), axis=0)
    rotation = int(destinations[0])
    step = int((destinations[1] - rotation) % WIDTH)
    if step == 1:
        sign = 1
    elif step == WIDTH - 1:
        sign = -1
    else:
        raise RuntimeError(f"black-box matrix is not dihedral: step={step}")
    group = (sign, rotation)
    expected = destination_indices(group)
    if not np.array_equal(destinations, expected):
        raise RuntimeError("black-box destinations do not form one D_109 action")
    a_value = np.empty(WIDTH, dtype=np.float64)
    a_value[expected] = matrix[expected, np.arange(WIDTH)]
    reconstructed_matrix = a_value[:, None] * permutation_matrix(group)
    if not np.array_equal(matrix, reconstructed_matrix):
        raise RuntimeError("black-box matrix has non-monomial entries")
    return group, a_value, b_value


def check_black_box() -> dict[str, Any]:
    sequences = all_sequences(3)
    maximum_error = 0.0
    mismatches = 0
    for sequence in sequences:
        exact = fold_left(sequence)
        reconstructed = reconstruct_black_box(exact)
        error = operator_error(exact, reconstructed)
        maximum_error = max(maximum_error, error)
        mismatches += int(error > TOLERANCE)
    return {
        "sequences": len(sequences),
        "basis_probes": len(sequences) * WIDTH,
        "max_error": maximum_error,
        "mismatches": mismatches,
    }


def homogeneous_operator(group: Group) -> Operator:
    return (
        group,
        np.ones(WIDTH, dtype=np.float64),
        np.zeros(WIDTH, dtype=np.float64),
    )


def check_separation() -> dict[str, Any]:
    rho: Group = (1, 1)
    tau: Group = (-1, 0)
    relation = compose_group(tau, compose_group(rho, tau))
    inverse_rho = inverse_group(rho)
    forward = compose(homogeneous_operator(tau), homogeneous_operator(rho))
    reverse = compose(homogeneous_operator(rho), homogeneous_operator(tau))
    marker_forward = apply_operator(forward, STATES[-1])
    marker_reverse = apply_operator(reverse, STATES[-1])
    matrices_differ = not np.array_equal(
        permutation_matrix(forward[0]), permutation_matrix(reverse[0])
    )
    marker_difference = float(np.max(np.abs(marker_forward - marker_reverse)))

    distinct = {
        permute(decode_group(code), np.arange(WIDTH, dtype=np.float64)).tobytes()
        for code in range(2 * WIDTH)
    }
    ordered_token_noncommuting = 0
    for first, second in itertools.product(TOKENS, repeat=2):
        forward_pair = compose(second, first)
        reverse_pair = compose(first, second)
        if forward_pair[0] != reverse_pair[0] or not np.array_equal(
            forward_pair[1][:, None] * permutation_matrix(forward_pair[0]),
            reverse_pair[1][:, None] * permutation_matrix(reverse_pair[0]),
        ):
            ordered_token_noncommuting += 1

    diagonal_max_error = 0.0
    for sequence in all_sequences(5):
        diagonal_sequence = tuple((IDENTITY_GROUP, item[1], item[2]) for item in sequence)
        observed = fold_left(diagonal_sequence)
        expected_a = np.ones(WIDTH, dtype=np.float64)
        expected_b = np.zeros(WIDTH, dtype=np.float64)
        for _, a_value, b_value in diagonal_sequence:
            expected_a, expected_b = (
                a_value * expected_a,
                a_value * expected_b + b_value,
            )
        diagonal_max_error = max(
            diagonal_max_error,
            float(np.max(np.abs(observed[1] - expected_a))),
            float(np.max(np.abs(observed[2] - expected_b))),
        )
    return {
        "tau_rho_tau_equals_rho_inverse": relation == inverse_rho,
        "tau_rho_differs_from_rho_tau": forward[0] != reverse[0],
        "materialized_matrices_differ": matrices_differ,
        "two_marker_max_difference": marker_difference,
        "distinct_group_actions": len(distinct),
        "ordered_noncommuting_token_pairs": ordered_token_noncommuting,
        "diagonal_reduction_max_error": diagonal_max_error,
    }


def check_title_substitution() -> dict[str, Any]:
    short = all_sequences(2)
    maximum_error = 0.0
    mismatches = 0
    single_cases = 0
    for prefix, document, suffix, title, state in itertools.product(
        short, short, short, TOKENS, STATES
    ):
        expanded = sequential_state(prefix + (title,) + document + suffix, state)
        compiled_title = compose(fold_left(document), title)
        compiled = sequential_state(prefix + (compiled_title,) + suffix, state)
        error = float(np.max(np.abs(expanded - compiled)))
        maximum_error = max(maximum_error, error)
        mismatches += int(error > TOLERANCE)
        single_cases += 1

    one = all_sequences(1)
    two_cases = 0
    for prefix, first_doc, middle, second_doc, suffix, first_title, second_title, state in itertools.product(
        one, one, one, one, one, TOKENS, TOKENS, STATES
    ):
        expanded = (
            prefix
            + (first_title,)
            + first_doc
            + middle
            + (second_title,)
            + second_doc
            + suffix
        )
        compiled = (
            prefix
            + (compose(fold_left(first_doc), first_title),)
            + middle
            + (compose(fold_left(second_doc), second_title),)
            + suffix
        )
        error = float(
            np.max(
                np.abs(
                    sequential_state(expanded, state)
                    - sequential_state(compiled, state)
                )
            )
        )
        maximum_error = max(maximum_error, error)
        mismatches += int(error > TOLERANCE)
        two_cases += 1

    unequal_pairs = 0
    insensitive_pairs = 0
    for first_doc, second_doc in itertools.combinations(short, 2):
        first = fold_left(first_doc)
        second = fold_left(second_doc)
        if operator_error(first, second) <= TOLERANCE:
            continue
        unequal_pairs += 1
        sensitive = False
        for state, suffix in itertools.product(STATES, one):
            first_final = sequential_state(suffix, apply_operator(first, state))
            second_final = sequential_state(suffix, apply_operator(second, state))
            if float(np.max(np.abs(first_final - second_final))) > TOLERANCE:
                sensitive = True
                break
        insensitive_pairs += int(not sensitive)
    return {
        "single_title_cases": single_cases,
        "two_title_cases": two_cases,
        "max_error": maximum_error,
        "mismatches": mismatches,
        "unequal_document_operator_pairs": unequal_pairs,
        "insensitive_wrong_document_pairs": insensitive_pairs,
    }


def lazy_step(
    frame: Group, physical: np.ndarray, operator: Operator
) -> tuple[Group, np.ndarray]:
    new_frame = compose_group(operator[0], frame)
    inverse = inverse_group(new_frame)
    physical_a = permute(inverse, operator[1])
    physical_b = permute(inverse, operator[2])
    return new_frame, physical_a * physical + physical_b


def lazy_sequence(
    sequence: Sequence[Operator], state: np.ndarray
) -> tuple[Group, np.ndarray, np.ndarray]:
    frame = IDENTITY_GROUP
    physical = np.asarray(state, dtype=np.float64).copy()
    for operator in sequence:
        frame, physical = lazy_step(frame, physical, operator)
    return frame, physical, permute(frame, physical)


def check_lazy_frame() -> dict[str, Any]:
    sequences = all_sequences(5)
    maximum_error = 0.0
    frame_failures = 0
    cases = 0
    vector_reads = 0
    vector_writes = 0
    multiplies = 0
    additions = 0
    physical_permutation_copies = 0
    for sequence, state in itertools.product(sequences, STATES):
        frame, _, logical = lazy_sequence(sequence, state)
        expected_operator = fold_left(sequence)
        expected = sequential_state(sequence, state)
        error = float(np.max(np.abs(expected - logical)))
        maximum_error = max(maximum_error, error)
        frame_failures += int(frame != expected_operator[0])
        cases += 1
        operations = len(sequence) * WIDTH
        vector_reads += operations
        vector_writes += operations
        multiplies += operations
        additions += operations
    return {
        "cases": cases,
        "max_error": maximum_error,
        "frame_failures": frame_failures,
        "physical_vector_reads": vector_reads,
        "physical_vector_writes": vector_writes,
        "multiplies": multiplies,
        "additions": additions,
        "physical_permutation_copies": physical_permutation_copies,
    }


def nearest_levels(
    values: np.ndarray, levels: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    level_values = np.asarray(levels, dtype=np.float64)
    distances = np.abs(values[:, None] - level_values[None, :])
    indices = np.argmin(distances, axis=1)
    quantized = level_values[indices]
    clipped = (values < level_values[0]) | (values > level_values[-1])
    return indices, quantized, clipped


def quantize_operator(operator: Operator) -> tuple[Operator, int]:
    a_indices, a_values, a_clipped = nearest_levels(operator[1], A_LEVELS)
    b_indices, b_values, b_clipped = nearest_levels(operator[2], B_LEVELS)
    del a_indices, b_indices
    return (operator[0], a_values, b_values), int(a_clipped.sum() + b_clipped.sum())


def suffix_scale(sequence: Sequence[Operator]) -> float:
    return math.prod(float(np.max(np.abs(operator[1]))) for operator in sequence)


def check_codec() -> dict[str, Any]:
    decoded_a = torch.tensor(A_LEVELS, dtype=torch.float32, device="cpu").to(torch.bfloat16).to(torch.float64)
    decoded_b = torch.tensor(B_LEVELS, dtype=torch.float32, device="cpu").to(torch.bfloat16).to(torch.float64)
    bf16_a_error = max(abs(float(value) - expected) for value, expected in zip(decoded_a, A_LEVELS))
    bf16_b_error = max(abs(float(value) - expected) for value, expected in zip(decoded_b, B_LEVELS))
    group_roundtrip_failures = 0
    for code in range(2 * WIDTH):
        group = decode_group(code)
        low, high = group_nibbles(group)
        group_roundtrip_failures += int(group_from_nibbles(low, high) != group)

    documents = all_sequences(5)
    suffixes = all_sequences(2)
    clipping = 0
    bound_violations = 0
    cases = 0
    maximum_error = 0.0
    maximum_bound = 0.0
    worst_witness: dict[str, Any] = {}
    quantized_documents: list[tuple[Operator, Operator]] = []
    for document in documents:
        exact_operator = fold_left(document)
        quantized_operator, clipped = quantize_operator(exact_operator)
        clipping += clipped
        quantized_documents.append((exact_operator, quantized_operator))
        epsilon_a = float(np.max(np.abs(quantized_operator[1] - exact_operator[1])))
        epsilon_b = float(np.max(np.abs(quantized_operator[2] - exact_operator[2])))
        for suffix, (state_index, state) in itertools.product(
            suffixes, enumerate(STATES)
        ):
            exact_final = sequential_state(suffix, apply_operator(exact_operator, state))
            quantized_final = sequential_state(
                suffix, apply_operator(quantized_operator, state)
            )
            error = float(np.max(np.abs(exact_final - quantized_final)))
            bound = suffix_scale(suffix) * (
                epsilon_a * float(np.max(np.abs(state))) + epsilon_b
            )
            bound_violations += int(error > bound + TOLERANCE)
            cases += 1
            maximum_bound = max(maximum_bound, bound)
            if error > maximum_error:
                maximum_error = error
                worst_witness = {
                    "document_length": len(document),
                    "suffix_length": len(suffix),
                    "incoming_state_index": state_index,
                    "error": error,
                    "bound": bound,
                    "group_code": encode_group(exact_operator[0]),
                }

    short_documents = all_sequences(2)
    short_pairs = []
    for document in short_documents:
        exact = fold_left(document)
        quantized, _ = quantize_operator(exact)
        short_pairs.append((exact, quantized))
    two_trigger_cases = 0
    two_trigger_violations = 0
    maximum_two_trigger_error = 0.0
    maximum_two_trigger_bound = 0.0
    for (exact_first, quantized_first), (exact_second, quantized_second), suffix, state in itertools.product(
        short_pairs, short_pairs, suffixes, STATES
    ):
        exact_middle = apply_operator(exact_first, state)
        quantized_middle = apply_operator(quantized_first, state)
        exact_second_state = apply_operator(exact_second, exact_middle)
        quantized_second_state = apply_operator(quantized_second, quantized_middle)
        exact_final = sequential_state(suffix, exact_second_state)
        quantized_final = sequential_state(suffix, quantized_second_state)
        error = float(np.max(np.abs(exact_final - quantized_final)))

        epsilon_a_first = float(np.max(np.abs(quantized_first[1] - exact_first[1])))
        epsilon_b_first = float(np.max(np.abs(quantized_first[2] - exact_first[2])))
        first_bound = epsilon_a_first * float(np.max(np.abs(state))) + epsilon_b_first
        epsilon_a_second = float(np.max(np.abs(quantized_second[1] - exact_second[1])))
        epsilon_b_second = float(np.max(np.abs(quantized_second[2] - exact_second[2])))
        second_bound = (
            float(np.max(np.abs(quantized_second[1]))) * first_bound
            + epsilon_a_second * float(np.max(np.abs(exact_middle)))
            + epsilon_b_second
        )
        bound = suffix_scale(suffix) * second_bound
        two_trigger_violations += int(error > bound + TOLERANCE)
        two_trigger_cases += 1
        maximum_two_trigger_error = max(maximum_two_trigger_error, error)
        maximum_two_trigger_bound = max(maximum_two_trigger_bound, bound)

    return {
        "a_levels": len(A_LEVELS),
        "b_levels": len(B_LEVELS),
        "group_codes": 2 * WIDTH,
        "max_bf16_a_error": bf16_a_error,
        "max_bf16_b_error": bf16_b_error,
        "group_roundtrip_failures": group_roundtrip_failures,
        "clipped_coordinates": clipping,
        "one_trigger_cases": cases,
        "one_trigger_bound_violations": bound_violations,
        "max_one_trigger_error": maximum_error,
        "max_one_trigger_bound": maximum_bound,
        "worst_one_trigger_witness": worst_witness,
        "two_trigger_cases": two_trigger_cases,
        "two_trigger_bound_violations": two_trigger_violations,
        "max_two_trigger_error": maximum_two_trigger_error,
        "max_two_trigger_bound": maximum_two_trigger_bound,
    }


def check_ledger() -> dict[str, Any]:
    state_value_bytes = WIDTH * 2
    frame_bytes = 2
    record_cells = 2 * WIDTH + 2
    title_codes = 132_800
    address_gates = 76_960
    thresholds = DOCUMENTS
    up_constants = DOCUMENTS
    payload = DOCUMENTS * record_cells
    total = title_codes + address_gates + thresholds + up_constants + payload
    return {
        "state_width": WIDTH,
        "state_value_bytes": state_value_bytes,
        "frame_bytes": frame_bytes,
        "total_state_bytes": state_value_bytes + frame_bytes,
        "record_cells": record_cells,
        "logical_bits_per_document": record_cells * 4,
        "documents": DOCUMENTS,
        "payload_entries": payload,
        "title_code_entries": title_codes,
        "address_gate_entries": address_gates,
        "threshold_entries": thresholds,
        "up_constant_entries": up_constants,
        "total_entries": total,
        "cap_entries": ENTRY_CAP,
        "spare_entries": ENTRY_CAP - total,
        "multiplies_per_token": WIDTH,
        "additions_per_token": WIDTH,
        "state_reads_per_token": WIDTH,
        "state_writes_per_token": WIDTH,
    }


def run_stage0() -> dict[str, Any]:
    started = time.perf_counter()
    group = check_group()
    exact = check_exact_forms()
    black_box = check_black_box()
    separation = check_separation()
    title = check_title_substitution()
    lazy = check_lazy_frame()
    codec = check_codec()
    ledger = check_ledger()
    gates = {
        "group": all(
            group[key] == 0
            for key in (
                "code_failures",
                "nibble_failures",
                "identity_failures",
                "inverse_failures",
                "action_failures",
                "associativity_failures",
            )
        ),
        "exact_forms": exact["mismatches"] == 0
        and exact["max_fold_error"] <= TOLERANCE
        and exact["max_state_error"] <= TOLERANCE,
        "black_box": black_box["mismatches"] == 0
        and black_box["max_error"] <= TOLERANCE,
        "separation": separation["tau_rho_tau_equals_rho_inverse"]
        and separation["tau_rho_differs_from_rho_tau"]
        and separation["materialized_matrices_differ"]
        and separation["two_marker_max_difference"] > 0.0
        and separation["distinct_group_actions"] == 2 * WIDTH
        and separation["ordered_noncommuting_token_pairs"] > 0
        and separation["diagonal_reduction_max_error"] <= TOLERANCE,
        "title_substitution": title["mismatches"] == 0
        and title["max_error"] <= TOLERANCE
        and title["insensitive_wrong_document_pairs"] == 0,
        "lazy_frame": lazy["frame_failures"] == 0
        and lazy["max_error"] <= TOLERANCE
        and lazy["physical_permutation_copies"] == 0,
        "codec": codec["max_bf16_a_error"] == 0.0
        and codec["max_bf16_b_error"] == 0.0
        and codec["group_roundtrip_failures"] == 0
        and codec["clipped_coordinates"] == 0
        and codec["one_trigger_bound_violations"] == 0
        and codec["two_trigger_bound_violations"] == 0,
        "ledger": ledger["total_state_bytes"] == 220
        and ledger["record_cells"] == PLANE_CELLS
        and ledger["logical_bits_per_document"] == 880
        and ledger["payload_entries"] == 529_100
        and ledger["total_entries"] == 743_670
        and ledger["spare_entries"] == 64
        and ledger["multiplies_per_token"] == WIDTH
        and ledger["additions_per_token"] == WIDTH
        and ledger["state_reads_per_token"] == WIDTH
        and ledger["state_writes_per_token"] == WIDTH,
    }
    root = Path(__file__).resolve().parents[1]
    preregistration = root / "results/dihedral-monomial-prefix-t30-stage0-preregistration.md"
    paper = root / "results/dihedral-monomial-prefix-t30-paper.md"
    tests = root / "tests/test_dihedral_monomial_prefix_t30_stage0.py"
    return {
        "experiment": "dihedral-monomial-prefix-t30-stage0",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "group": group,
        "exact_forms": exact,
        "black_box_reconstruction": black_box,
        "separation": separation,
        "title_substitution": title,
        "lazy_frame": lazy,
        "codec": codec,
        "ledger": ledger,
        "gates": gates,
        "runtime": {
            "device": "cpu",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "wall_seconds": time.perf_counter() - started,
        },
        "integrity": {
            "paper_sha256": sha256_file(paper),
            "preregistration_sha256": sha256_file(preregistration),
            "source_sha256": sha256_file(Path(__file__).resolve()),
            "test_sha256": sha256_file(tests),
        },
        "claim_boundary": (
            "Exact D_109, monomial-affine, lazy-frame, codec, error-bound, and "
            "ledger claims only; no routing-learnability, natural-language, GPU, "
            "novelty, or production claim."
        ),
    }


def main() -> None:
    print(json.dumps(run_stage0(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
