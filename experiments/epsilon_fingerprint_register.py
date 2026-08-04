"""Stage-0 algebra gate for a randomized finite-field fingerprint register.

The register compares two explicitly delimited streams with one-sided error.
It uses a fresh random evaluation point per request and a fixed Horner update,
so no learned per-token routing decision is part of the theorem.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Iterable


ROOT_DIR = Path(__file__).resolve().parents[1]
FIELD_PRIME_64 = 18_446_744_073_709_551_557
LEFT_PHASE = 0
RIGHT_PHASE = 1
FINISHED_PHASE = 2


def is_prime_64(value: int) -> bool:
    """Deterministic Miller-Rabin primality test for unsigned 64-bit inputs."""

    if value < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for prime in small_primes:
        if value % prime == 0:
            return value == prime

    odd_part = value - 1
    power_of_two = 0
    while odd_part % 2 == 0:
        power_of_two += 1
        odd_part //= 2

    for base in (2, 325, 9_375, 28_178, 450_775, 9_780_504, 1_795_265_022):
        if base % value == 0:
            continue
        witness = pow(base, odd_part, value)
        if witness in (1, value - 1):
            continue
        for _ in range(power_of_two - 1):
            witness = witness * witness % value
            if witness == value - 1:
                break
        else:
            return False
    return True


def fingerprint(values: Iterable[int], evaluation_point: int, prime: int) -> int:
    """Evaluate the token polynomial with a fixed Horner recurrence."""

    if not 0 <= evaluation_point < prime:
        raise ValueError("evaluation_point must be a field element")
    result = 0
    for value in values:
        if not 0 <= value < prime - 1:
            raise ValueError("token value must fit after the injective +1 encoding")
        result = (result * evaluation_point + value + 1) % prime
    return result


class FingerprintRegister:
    """Bounded two-segment streaming register; it never stores the token stream."""

    __slots__ = (
        "evaluation_point",
        "prime",
        "maximum_length",
        "left_hash",
        "right_hash",
        "left_length",
        "right_length",
        "phase",
    )

    def __init__(
        self, *, evaluation_point: int, prime: int, maximum_length: int
    ) -> None:
        if not is_prime_64(prime):
            raise ValueError("prime must be prime")
        if not 0 <= evaluation_point < prime:
            raise ValueError("evaluation_point must be a field element")
        if maximum_length < 0:
            raise ValueError("maximum_length must be nonnegative")
        self.evaluation_point = evaluation_point
        self.prime = prime
        self.maximum_length = maximum_length
        self.left_hash = 0
        self.right_hash = 0
        self.left_length = 0
        self.right_length = 0
        self.phase = LEFT_PHASE

    def snapshot(self) -> tuple[int, int, int, int, int, int]:
        return (
            self.evaluation_point,
            self.left_hash,
            self.right_hash,
            self.left_length,
            self.right_length,
            self.phase,
        )

    def push(self, token: int) -> None:
        if self.phase == FINISHED_PHASE:
            raise ValueError("cannot push after finish")
        if not 0 <= token < self.prime - 1:
            raise ValueError("token must fit after the injective +1 encoding")

        if self.phase == LEFT_PHASE:
            if self.left_length >= self.maximum_length:
                raise OverflowError("left segment exceeds maximum_length")
            self.left_hash = (
                self.left_hash * self.evaluation_point + token + 1
            ) % self.prime
            self.left_length += 1
            return

        if self.right_length >= self.maximum_length:
            raise OverflowError("right segment exceeds maximum_length")
        self.right_hash = (
            self.right_hash * self.evaluation_point + token + 1
        ) % self.prime
        self.right_length += 1

    def separator(self) -> None:
        if self.phase != LEFT_PHASE:
            raise ValueError("separator is valid exactly once after the left segment")
        self.phase = RIGHT_PHASE

    def finish(self) -> bool:
        if self.phase != RIGHT_PHASE:
            raise ValueError("finish requires exactly one separator")
        self.phase = FINISHED_PHASE
        return (
            self.left_length == self.right_length
            and self.left_hash == self.right_hash
        )


def equal_with_fingerprint(
    left: tuple[int, ...], right: tuple[int, ...], evaluation_point: int, prime: int
) -> bool:
    return len(left) == len(right) and fingerprint(
        left, evaluation_point, prime
    ) == fingerprint(right, evaluation_point, prime)


def streaming_equal_with_fingerprint(
    left: Iterable[int],
    right: Iterable[int],
    *,
    evaluation_point: int,
    prime: int,
    maximum_length: int,
) -> bool:
    register = FingerprintRegister(
        evaluation_point=evaluation_point,
        prime=prime,
        maximum_length=maximum_length,
    )
    for token in left:
        register.push(token)
    register.separator()
    for token in right:
        register.push(token)
    return register.finish()


def bounded_streaming_gate(*, maximum_length: int = 1_000_000) -> dict[str, object]:
    register = FingerprintRegister(
        evaluation_point=1_000_003,
        prime=FIELD_PRIME_64,
        maximum_length=maximum_length,
    )
    for _ in range(maximum_length):
        register.push(173)
    left_boundary_snapshot = register.snapshot()
    try:
        register.push(173)
    except OverflowError:
        overflow_rejected_without_mutation = register.snapshot() == left_boundary_snapshot
    else:
        raise AssertionError("maximum_length overflow was not rejected")

    register.separator()
    for _ in range(maximum_length):
        register.push(173)
    accepted_at_boundary = register.finish()
    if not accepted_at_boundary:
        raise AssertionError("equal maximum-length segments must be accepted")

    empty_accepted = streaming_equal_with_fingerprint(
        (),
        (),
        evaluation_point=7,
        prime=257,
        maximum_length=0,
    )
    unequal_length_rejected = not streaming_equal_with_fingerprint(
        (1,),
        (),
        evaluation_point=7,
        prime=257,
        maximum_length=1,
    )
    return {
        "maximum_length_per_segment": maximum_length,
        "accepted_equal_segments_at_boundary": accepted_at_boundary,
        "overflow_rejected_without_mutation": overflow_rejected_without_mutation,
        "empty_segments_accepted": empty_accepted,
        "unequal_length_rejected": unequal_length_rejected,
        "stored_fields": list(FingerprintRegister.__slots__),
    }


def collision_points(
    left: tuple[int, ...], right: tuple[int, ...], prime: int
) -> tuple[int, ...]:
    if len(left) != len(right) or left == right:
        raise ValueError("root counting requires unequal, equal-length streams")
    return tuple(
        point
        for point in range(prime)
        if equal_with_fingerprint(left, right, point, prime)
    )


def exhaustive_root_bound_gate(
    *, alphabet_size: int = 2, maximum_length: int = 6, prime: int = 257
) -> dict[str, object]:
    """Exhaustively verify the polynomial root bound on a small field."""

    if not is_prime_64(prime):
        raise ValueError("the modulus must be prime")
    maximum_roots = -1
    maximum_example: dict[str, object] | None = None
    unequal_pairs = 0
    field_evaluations = 0

    for length in range(1, maximum_length + 1):
        streams = tuple(itertools.product(range(alphabet_size), repeat=length))
        for left in streams:
            for right in streams:
                if left == right:
                    continue
                roots = collision_points(left, right, prime)
                unequal_pairs += 1
                field_evaluations += prime
                if len(roots) > length - 1:
                    raise AssertionError("a nonzero degree-(n-1) polynomial has too many roots")
                if len(roots) > maximum_roots:
                    maximum_roots = len(roots)
                    maximum_example = {
                        "left": list(left),
                        "right": list(right),
                        "roots": list(roots),
                    }

    return {
        "alphabet_size": alphabet_size,
        "maximum_length": maximum_length,
        "prime": prime,
        "unequal_pairs": unequal_pairs,
        "field_evaluations": field_evaluations,
        "maximum_collision_roots": maximum_roots,
        "maximum_example": maximum_example,
    }


def state_ledger(
    *, maximum_length: int, alphabet_size: int, prime: int
) -> dict[str, int | float | str]:
    if maximum_length < 2 or alphabet_size < 2 or prime <= alphabet_size:
        raise ValueError("invalid ledger parameters")
    if not is_prime_64(prime):
        raise ValueError("the modulus must be prime")

    field_bits = (prime - 1).bit_length()
    length_bits = math.ceil(math.log2(maximum_length + 1))
    fingerprint_state_bits = 3 * field_bits + 2 * length_bits + 2
    deterministic_exact_bits = math.ceil(
        maximum_length * math.log2(alphabet_size)
    )
    collision_bound = (maximum_length - 1) / prime
    return {
        "maximum_length_per_segment": maximum_length,
        "alphabet_size": alphabet_size,
        "field_prime": prime,
        "field_bits": field_bits,
        "request_random_evaluation_point_bits": field_bits,
        "two_fingerprint_bits": 2 * field_bits,
        "two_length_counter_bits": 2 * length_bits,
        "phase_bits": 2,
        "logical_register_bits": fingerprint_state_bits,
        "logical_register_bytes_ceiling": math.ceil(fingerprint_state_bits / 8),
        "deterministic_exact_one_pass_bits_lower_bound": deterministic_exact_bits,
        "deterministic_exact_one_pass_bytes_lower_bound": math.ceil(
            deterministic_exact_bits / 8
        ),
        "exact_to_randomized_state_ratio": (
            deterministic_exact_bits / fingerprint_state_bits
        ),
        "worst_case_false_positive_probability_bound": collision_bound,
        "negative_log2_error_bound": -math.log2(collision_bound),
        "error_model": "fresh_hidden_random_point_independent_of_nonadaptive_input",
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_stage0_gate() -> dict[str, object]:
    exhaustive = exhaustive_root_bound_gate()
    streaming = bounded_streaming_gate()
    ledger = state_ledger(
        maximum_length=1_000_000,
        alphabet_size=256,
        prime=FIELD_PRIME_64,
    )
    return {
        "schema_version": 2,
        "status": "algebra_and_bounded_streaming_reference_pass",
        "candidate_number": None,
        "gpu_required": False,
        "method": "fixed_update_randomized_finite_field_fingerprint",
        "claim_scope": "explicitly_delimited_equal_length_byte_string_equality",
        "operator": {
            "update": "hash = (hash * fresh_request_point + token + 1) mod prime",
            "read": "equal_lengths and equal_hashes",
            "learned_per_token_router": False,
        },
        "theorem": {
            "false_negative_probability": 0,
            "false_positive_probability": "at_most_(length-1)/prime",
            "deterministic_exact_state_lower_bound": (
                "ceil(length*log2(alphabet_size))_bits"
            ),
            "field_choice": "prime > max(alphabet_size, (length-1)/epsilon)",
        },
        "state_ledger": ledger,
        "exhaustive_small_field_gate": exhaustive,
        "bounded_streaming_gate": streaming,
        "resource_model": {
            "logical_machine": "word_RAM_with_word_bits_at_least_ceil_log2_prime",
            "logical_update": "one_modular_Horner_step_plus_counter_and_phase_update",
            "physical_caveat": (
                "64x64_product_uses_a_128_bit_intermediate_and_modular_reduction_"
                "may_require_multiple_CPU_or_GPU_instructions"
            ),
        },
        "matched_control": (
            "a_randomized_recurrent_digital_cell_executes_the_identical_update_"
            "with_identical_state_arithmetic_randomness_and_error"
        ),
        "direct_classical_collision": {
            "work": "Karp_Rabin_1987_randomized_string_fingerprints",
            "novelty_scope": "no_new_streaming_algorithm_claim",
        },
        "excluded_accounting": [
            "random_number_generation_latency_and_entropy_source",
            "token_to_field_encoding",
            "packed_versus_allocated_state_bytes",
            "physical_wide_multiply_and_modular_reduction_kernel",
        ],
        "required_assumptions": [
            "prime_field_arithmetic_is_exact",
            "evaluation_point_is_fresh_hidden_and_independent_of_input",
            "input_is_nonadaptive_to_internal_collision_outcomes",
            "segment_boundaries_are_explicit_and_correct",
        ],
        "not_proven": [
            "new_model_architecture_or_novel_randomized_streaming_algorithm",
            "general_language_model_capability_gain",
            "quality_noninferiority_after_replacing_neural_state_or_compute",
            "gpu_latency_energy_or_kernel_efficiency",
            "arbitrary_substring_or_semantic_entity_comparison",
        ],
        "decision": "retain_as_classical_epsilon_trade_not_architecture_candidate",
        "source_hashes": {
            "executable_sha256": _sha256_file(
                ROOT_DIR / "experiments" / "epsilon_fingerprint_register.py"
            ),
            "focused_test_sha256": _sha256_file(
                ROOT_DIR / "tests" / "test_epsilon_fingerprint_register.py"
            ),
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_stage0_gate(), indent=2, sort_keys=True))
