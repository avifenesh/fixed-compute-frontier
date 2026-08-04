from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from pathlib import Path
from typing import Iterable

import numpy as np


WIDTH = 109
IDENTITY = (1, 0)
RHO = (1, 1)
TAU = (-1, 0)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group_elements(width: int = WIDTH) -> tuple[tuple[int, int], ...]:
    return tuple((sign, rotation) for sign in (1, -1) for rotation in range(width))


def compose(
    after: tuple[int, int], before: tuple[int, int], width: int = WIDTH
) -> tuple[int, int]:
    return (
        after[0] * before[0],
        (after[1] + after[0] * before[1]) % width,
    )


def canonical_word(group: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    sign, rotation = group
    return ((TAU,) if sign == -1 else ()) + (RHO,) * rotation


def fold_word(
    word: Iterable[tuple[int, int]], width: int = WIDTH
) -> tuple[int, int]:
    result = IDENTITY
    for token in word:
        result = compose(token, result, width)
    return result


def source_indices(group: tuple[int, int], width: int = WIDTH) -> np.ndarray:
    sign, rotation = group
    destination = np.arange(width)
    return (sign * (destination - rotation)) % width


def permutation_matrix(group: tuple[int, int], width: int = WIDTH) -> np.ndarray:
    matrix = np.zeros((width, width), dtype=np.float64)
    matrix[np.arange(width), source_indices(group, width)] = 1.0
    return matrix


def apply_group(states: np.ndarray, group: tuple[int, int]) -> np.ndarray:
    return states[:, source_indices(group, states.shape[1])]


def fixed_points(group: tuple[int, int], width: int = WIDTH) -> int:
    source = source_indices(group, width)
    return int(np.count_nonzero(source == np.arange(width)))


def centered_basis(width: int = WIDTH) -> np.ndarray:
    basis = math.sqrt(width) * np.eye(width, dtype=np.float64)
    return np.concatenate((basis, -basis), axis=0)


def optimal_diagonal(group: tuple[int, int], width: int = WIDTH) -> np.ndarray:
    matrix = permutation_matrix(group, width)
    return np.diag(matrix).copy()


def normalized_mse(target: np.ndarray, observed: np.ndarray) -> float:
    return float(np.mean(np.square(target - observed)))


def run() -> dict[str, object]:
    started = time.perf_counter()
    groups = group_elements()
    inputs = centered_basis()
    mean_error = float(np.max(np.abs(inputs.mean(axis=0))))
    covariance = inputs.T @ inputs / len(inputs)
    covariance_error = float(np.max(np.abs(covariance - np.eye(WIDTH))))

    word_failures = 0
    action_codes: set[bytes] = set()
    formula_failures = 0
    routed_failures = 0
    maximum_formula_error = 0.0
    maximum_routed_error = 0.0
    relaxed_errors: list[float] = []
    fixed_point_histogram: dict[int, int] = {}

    for group in groups:
        word_failures += fold_word(canonical_word(group)) != group
        matrix = permutation_matrix(group)
        action_codes.add(matrix.tobytes())
        fixed = fixed_points(group)
        fixed_point_histogram[fixed] = fixed_point_histogram.get(fixed, 0) + 1

        target = apply_group(inputs, group)
        diagonal = optimal_diagonal(group)
        relaxed = inputs * diagonal[None, :]
        measured = normalized_mse(target, relaxed)
        expected = 1.0 - fixed / WIDTH
        error = abs(measured - expected)
        maximum_formula_error = max(maximum_formula_error, error)
        formula_failures += error > 1e-12
        relaxed_errors.append(measured)

        routed = apply_group(inputs, fold_word(canonical_word(group)))
        routed_error = float(np.max(np.abs(target - routed)))
        maximum_routed_error = max(maximum_routed_error, routed_error)
        routed_failures += routed_error != 0.0

    rf = permutation_matrix(fold_word((RHO, TAU)))
    fr = permutation_matrix(fold_word((TAU, RHO)))
    noncommutative_difference = float(np.max(np.abs(rf - fr)))
    observed_average = float(np.mean(relaxed_errors))
    expected_average = 108.0 / 109.0

    gates = {
        "all_canonical_words_exact": word_failures == 0,
        "all_actions_distinct": len(action_codes) == 2 * WIDTH,
        "generators_noncommute": noncommutative_difference > 0.0,
        "centered_basis_exact": mean_error <= 1e-12 and covariance_error <= 1e-12,
        "per_group_formula_exact": formula_failures == 0
        and maximum_formula_error <= 1e-12,
        "uniform_bound_exact": abs(observed_average - expected_average) <= 1e-12,
        "constructive_routed_witness_exact": routed_failures == 0
        and maximum_routed_error == 0.0,
        "state_ledger_220_bytes": WIDTH * 2 + 2 == 220,
    }

    root = Path(__file__).resolve().parents[1]
    preregistration = (
        root
        / "results/dihedral-monomial-prefix-t30-separation-preflight-preregistration.md"
    )
    tests = root / "tests/test_dihedral_monomial_prefix_t30_separation_preflight.py"
    return {
        "experiment": "dihedral-monomial-prefix-t30-separation-preflight",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "domain": {
            "width": WIDTH,
            "group_elements": len(groups),
            "centered_basis_inputs_per_group": len(inputs),
            "state_evaluations": len(groups) * len(inputs),
        },
        "measurements": {
            "fixed_point_histogram": fixed_point_histogram,
            "distinct_actions": len(action_codes),
            "word_failures": word_failures,
            "noncommutative_max_difference": noncommutative_difference,
            "centered_basis_mean_error": mean_error,
            "centered_basis_covariance_error": covariance_error,
            "formula_failures": formula_failures,
            "maximum_formula_error": maximum_formula_error,
            "observed_relaxed_diagonal_average_nmse": observed_average,
            "expected_relaxed_diagonal_average_nmse": expected_average,
            "routed_failures": routed_failures,
            "maximum_routed_error": maximum_routed_error,
            "state_bytes": WIDTH * 2 + 2,
        },
        "gates": gates,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "device": "cpu",
            "wall_seconds": time.perf_counter() - started,
        },
        "integrity": {
            "preregistration_sha256": sha256_file(preregistration),
            "source_sha256": sha256_file(Path(__file__).resolve()),
            "test_sha256": sha256_file(tests),
        },
        "claim_boundary": (
            "Exact arbitrary-state operator separation only; no learned, raw-prose, "
            "natural-semantic, novelty, GPU-cost, or production claim."
        ),
    }


def main() -> None:
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
