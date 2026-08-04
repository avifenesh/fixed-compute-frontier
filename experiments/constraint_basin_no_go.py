"""Executable Stage-0 no-go gate for constraint-basin completion.

The gate checks four claims without training:

1. prototype log-sum-exp settling is a softmax attention read;
2. sparse parity-energy descent is factor-to-variable message passing;
3. its first binary step is a threshold bit-flipping decoder; and
4. a direct Hamming syndrome decoder already reaches the MAP ceiling on the
   proposed bounded-corruption workload.
"""

from __future__ import annotations

import itertools
import hashlib
import json
import math
import random
from collections.abc import Callable, Sequence
from pathlib import Path


Vector = tuple[float, ...]
Factor = tuple[int, ...]

HAMMING_7_4_CHECKS: tuple[Factor, ...] = (
    (0, 2, 4, 6),
    (1, 2, 5, 6),
    (3, 4, 5, 6),
)
ROOT_DIR = Path(__file__).resolve().parents[1]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _clip(value: float) -> float:
    return max(-1.0, min(1.0, value))


def attention_read(
    patterns: Sequence[Sequence[float]], query: Sequence[float], beta: float
) -> Vector:
    """Single-query softmax attention with patterns as both keys and values."""

    logits = [beta * _dot(pattern, query) for pattern in patterns]
    maximum = max(logits)
    weights = [math.exp(logit - maximum) for logit in logits]
    normalizer = sum(weights)
    return tuple(
        sum(weight * pattern[axis] for weight, pattern in zip(weights, patterns, strict=True))
        / normalizer
        for axis in range(len(query))
    )


def prototype_energy(
    patterns: Sequence[Sequence[float]], state: Sequence[float], beta: float
) -> float:
    """E(x) = ||x||^2/2 - logsumexp(beta * p_i.x)/beta."""

    logits = [beta * _dot(pattern, state) for pattern in patterns]
    maximum = max(logits)
    log_partition = maximum + math.log(
        sum(math.exp(logit - maximum) for logit in logits)
    )
    return 0.5 * _dot(state, state) - log_partition / beta


def numerical_unit_energy_step(
    patterns: Sequence[Sequence[float]],
    state: Sequence[float],
    beta: float,
    *,
    epsilon: float = 1e-5,
) -> Vector:
    """Take x - grad E using an independent central-difference gradient."""

    gradient: list[float] = []
    for axis in range(len(state)):
        left = list(state)
        right = list(state)
        left[axis] -= epsilon
        right[axis] += epsilon
        gradient.append(
            (prototype_energy(patterns, right, beta) - prototype_energy(patterns, left, beta))
            / (2.0 * epsilon)
        )
    return tuple(value - derivative for value, derivative in zip(state, gradient, strict=True))


def parity_energy(
    state: Sequence[float],
    cue: Sequence[float],
    factors: Sequence[Factor],
    penalty: float,
) -> float:
    fidelity = 0.5 * sum((value - source) ** 2 for value, source in zip(state, cue, strict=True))
    violations = 0.5 * penalty * sum(
        1.0 - math.prod(state[index] for index in factor) for factor in factors
    )
    return fidelity + violations


def numerical_parity_step(
    state: Sequence[float],
    cue: Sequence[float],
    factors: Sequence[Factor],
    *,
    penalty: float,
    step_size: float,
    epsilon: float = 1e-5,
) -> Vector:
    """Take a parity-energy step using an independent numerical gradient."""

    result: list[float] = []
    for axis, value in enumerate(state):
        left = list(state)
        right = list(state)
        left[axis] -= epsilon
        right[axis] += epsilon
        derivative = (
            parity_energy(right, cue, factors, penalty)
            - parity_energy(left, cue, factors, penalty)
        ) / (2.0 * epsilon)
        result.append(_clip(value - step_size * derivative))
    return tuple(result)


def basin_step(
    state: Sequence[float],
    cue: Sequence[float],
    factors: Sequence[Factor],
    *,
    penalty: float,
    step_size: float,
) -> Vector:
    """Direct coordinate gradient of the proposed sparse-factor energy."""

    result: list[float] = []
    for index, (value, source) in enumerate(zip(state, cue, strict=True)):
        derivative = value - source
        for factor in factors:
            if index not in factor:
                continue
            leave_one_out = 1.0
            for neighbour in factor:
                if neighbour != index:
                    leave_one_out *= state[neighbour]
            derivative -= 0.5 * penalty * leave_one_out
        result.append(_clip(value - step_size * derivative))
    return tuple(result)


def message_passing_step(
    state: Sequence[float],
    cue: Sequence[float],
    factors: Sequence[Factor],
    *,
    penalty: float,
    step_size: float,
) -> Vector:
    """The same update written independently as hypergraph message passing."""

    messages = [0.0] * len(state)
    for factor in factors:
        for destination in factor:
            message = 1.0
            for source in factor:
                if source != destination:
                    message *= state[source]
            messages[destination] += message

    gain = 0.5 * step_size * penalty
    return tuple(
        _clip(
            (1.0 - step_size) * value
            + step_size * source
            + gain * messages[index]
        )
        for index, (value, source) in enumerate(zip(state, cue, strict=True))
    )


def recurrent_trajectory(
    initial: Sequence[float], update: Callable[[Vector], Vector], steps: int
) -> tuple[Vector, ...]:
    """Generic tied recurrent executor; the basin update is one admissible phi."""

    state = tuple(initial)
    trajectory = [state]
    for _ in range(steps):
        state = update(state)
        trajectory.append(state)
    return tuple(trajectory)


def hamming_7_4_codeword(message: Sequence[int]) -> tuple[int, ...]:
    if len(message) != 4 or any(bit not in (0, 1) for bit in message):
        raise ValueError("message must contain four binary values")
    d1, d2, d3, d4 = message
    return (
        d1 ^ d2 ^ d4,
        d1 ^ d3 ^ d4,
        d1,
        d2 ^ d3 ^ d4,
        d2,
        d3,
        d4,
    )


def hamming_syndrome(word: Sequence[int]) -> int:
    syndrome = 0
    for position, bit in enumerate(word, start=1):
        if bit:
            syndrome ^= position
    return syndrome


def syndrome_decode(word: Sequence[int]) -> tuple[int, ...]:
    decoded = list(word)
    syndrome = hamming_syndrome(decoded)
    if syndrome:
        decoded[syndrome - 1] ^= 1
    return tuple(decoded)


def nearest_codeword_decode(
    word: Sequence[int], codebook: Sequence[Sequence[int]]
) -> tuple[int, ...]:
    """Exact MAP for equiprobable codewords and a binary symmetric channel."""

    return min(
        (tuple(codeword) for codeword in codebook),
        key=lambda codeword: sum(a != b for a, b in zip(word, codeword, strict=True)),
    )


def _prototype_attention_gate() -> dict[str, float | int]:
    rng = random.Random(20260726)
    cases = 12
    maximum_error = 0.0
    for _ in range(cases):
        patterns = tuple(
            tuple(rng.uniform(-0.8, 0.8) for _ in range(5)) for _ in range(7)
        )
        state = tuple(rng.uniform(-0.7, 0.7) for _ in range(5))
        beta = rng.uniform(0.6, 2.4)
        attention = attention_read(patterns, state, beta)
        energy_step = numerical_unit_energy_step(patterns, state, beta)
        maximum_error = max(
            maximum_error,
            max(abs(a - b) for a, b in zip(attention, energy_step, strict=True)),
        )
    tolerance = 2e-8
    if maximum_error > tolerance:
        raise AssertionError(f"prototype/attention mismatch: {maximum_error}")
    return {
        "cases": cases,
        "coordinates_compared": cases * 5,
        "finite_difference_epsilon": 1e-5,
        "max_abs_error": maximum_error,
        "tolerance": tolerance,
    }


def _message_passing_gate() -> dict[str, float | int | bool]:
    rng = random.Random(1701)
    trajectories = 16
    steps = 8
    comparisons = 0
    message_maximum_error = 0.0
    numerical_maximum_error = 0.0
    recurrent_embedding_bitwise_equal = True
    for _ in range(trajectories):
        initial = tuple(rng.uniform(-0.95, 0.95) for _ in range(7))
        cue = tuple(rng.uniform(-0.95, 0.95) for _ in range(7))
        penalty = rng.uniform(0.2, 1.8)
        step_size = rng.uniform(0.05, 0.45)

        def direct_update(state: Vector) -> Vector:
            return basin_step(
                state,
                cue,
                HAMMING_7_4_CHECKS,
                penalty=penalty,
                step_size=step_size,
            )

        manual_state = initial
        direct_states = [manual_state]
        for _ in range(steps):
            manual_state = direct_update(manual_state)
            direct_states.append(manual_state)
        direct = tuple(direct_states)

        embedded = recurrent_trajectory(initial, direct_update, steps)
        generic = recurrent_trajectory(
            initial,
            lambda state: message_passing_step(
                state,
                cue,
                HAMMING_7_4_CHECKS,
                penalty=penalty,
                step_size=step_size,
            ),
            steps,
        )
        recurrent_embedding_bitwise_equal &= direct == embedded
        for step_index, (direct_state, message_state) in enumerate(
            zip(direct, generic, strict=True)
        ):
            for direct_value, message_value in zip(direct_state, message_state, strict=True):
                comparisons += 1
                message_maximum_error = max(
                    message_maximum_error, abs(direct_value - message_value)
                )
            if step_index < steps:
                numerical_next = numerical_parity_step(
                    direct_state,
                    cue,
                    HAMMING_7_4_CHECKS,
                    penalty=penalty,
                    step_size=step_size,
                )
                numerical_maximum_error = max(
                    numerical_maximum_error,
                    max(
                        abs(a - b)
                        for a, b in zip(
                            direct[step_index + 1], numerical_next, strict=True
                        )
                    ),
                )

    message_tolerance = 2e-15
    numerical_tolerance = 2e-9
    if (
        message_maximum_error > message_tolerance
        or numerical_maximum_error > numerical_tolerance
        or not recurrent_embedding_bitwise_equal
    ):
        raise AssertionError("basin update is not reproduced by the recurrent control")
    return {
        "trajectories": trajectories,
        "steps_per_trajectory": steps,
        "coordinates_compared": comparisons,
        "message_passing_max_abs_error": message_maximum_error,
        "message_passing_tolerance": message_tolerance,
        "message_passing_within_tolerance": True,
        "finite_difference_epsilon": 1e-5,
        "finite_difference_max_abs_error": numerical_maximum_error,
        "finite_difference_tolerance": numerical_tolerance,
        "basin_embedded_in_generic_executor_bitwise_equal": (
            recurrent_embedding_bitwise_equal
        ),
    }


def _binary_bit_flip_gate() -> dict[str, int]:
    checks = 0
    for bits in itertools.product((-1.0, 1.0), repeat=7):
        for coupling in (0.37, 0.83, 1.71):
            step_size = 0.5
            penalty = 2.0 * coupling / step_size
            updated = basin_step(
                bits,
                bits,
                HAMMING_7_4_CHECKS,
                penalty=penalty,
                step_size=step_size,
            )
            for index in range(7):
                incident = [factor for factor in HAMMING_7_4_CHECKS if index in factor]
                unsatisfied = sum(
                    math.prod(bits[node] for node in factor) < 0.0 for factor in incident
                )
                threshold_flip = unsatisfied > (len(incident) + 1.0 / coupling) / 2.0
                observed_flip = updated[index] * bits[index] < 0.0
                if observed_flip != threshold_flip:
                    raise AssertionError("binary basin step is not threshold bit-flipping")
                checks += 1
    return {"binary_states": 128, "couplings": 3, "coordinate_checks": checks}


def _hamming_ceiling_gate() -> dict[str, float | int]:
    codebook = tuple(
        hamming_7_4_codeword(message)
        for message in itertools.product((0, 1), repeat=4)
    )
    if len(set(codebook)) != 16 or any(hamming_syndrome(word) for word in codebook):
        raise AssertionError("invalid Hamming(7,4) codebook")

    episodes = 0
    syndrome_correct = 0
    map_correct = 0
    decoder_agreement = 0
    for source in codebook:
        for flipped_position in range(-1, 7):
            observed = list(source)
            if flipped_position >= 0:
                observed[flipped_position] ^= 1
            observed_word = tuple(observed)
            syndrome_result = syndrome_decode(observed_word)
            map_result = nearest_codeword_decode(observed_word, codebook)
            episodes += 1
            syndrome_correct += syndrome_result == source
            map_correct += map_result == source
            decoder_agreement += syndrome_result == map_result

    if syndrome_correct != episodes or map_correct != episodes:
        raise AssertionError("direct decoder failed below the exact-recovery ceiling")
    return {
        "codewords": len(codebook),
        "clean_plus_single_flip_episodes": episodes,
        "syndrome_accuracy": syndrome_correct / episodes,
        "exact_map_accuracy": map_correct / episodes,
        "decoder_agreement": decoder_agreement / episodes,
    }


def run_no_go_gate() -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "closed_before_training",
        "candidate_number": None,
        "gpu_required": False,
        "closed_claim": "distinct_best_achievable_capability_algebra",
        "resource_dominance_claimed": False,
        "frozen_workload": {
            "code": "Hamming(7,4)",
            "parity_check_edges": 12,
            "logical_parity_matrix_bits": 21,
            "logical_parity_matrix_packed_bytes": 3,
            "corruptions": "clean_or_exactly_one_flip",
            "episodes": 128,
        },
        "execution": {
            "command": "python experiments/constraint_basin_no_go.py",
            "arithmetic": "Python_binary64_float_and_integer_standard_library",
            "random_seeds": [20260726, 1701],
        },
        "source_hashes": {
            "executable_sha256": _sha256_file(
                ROOT_DIR / "experiments" / "constraint_basin_no_go.py"
            ),
            "focused_test_sha256": _sha256_file(
                ROOT_DIR / "tests" / "test_constraint_basin_no_go.py"
            ),
        },
        "prototype_energy_is_attention": _prototype_attention_gate(),
        "parity_basin_is_message_passing": _message_passing_gate(),
        "binary_step_is_threshold_bit_flipping": _binary_bit_flip_gate(),
        "direct_control_reaches_map_ceiling": _hamming_ceiling_gate(),
        "decision": "no_distinct_capability_algebra_admitted",
        "surviving_scope": (
            "possible finite-optimizer or decoder-state tradeoff only; "
            "not a fixed-serving capability frontier"
        ),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    print(json.dumps(run_no_go_gate(), indent=2, sort_keys=True))
