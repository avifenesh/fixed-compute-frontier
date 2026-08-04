#!/usr/bin/env python3
"""Fatal algebra and finite-precision gates for spectral successor targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


PREREGISTRATION = Path(
    "results/spectral-successor-stage0-preregistration.md"
)
TEST = Path("tests/test_spectral_successor_stage0.py")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixed_codes(vocabulary: int, code_width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.choice((-1.0, 1.0), size=(vocabulary, code_width))
    return values / math.sqrt(code_width)


def spectral_points(count: int, radius: float = 0.97) -> np.ndarray:
    # Half-bin offset avoids the real axes while retaining uniform coverage.
    frequency = 2.0 * np.pi * (np.arange(count) + 0.5) / count
    return radius * np.exp(1j * frequency)


def recurrence_targets(
    tokens: np.ndarray, codes: np.ndarray, points: np.ndarray, dtype: np.dtype
) -> np.ndarray:
    token_codes = codes[tokens].astype(dtype, copy=False)
    complex_dtype = np.complex128 if dtype == np.float64 else np.complex64
    points = points.astype(complex_dtype, copy=False)
    output = np.zeros(
        (tokens.shape[0], points.shape[0], codes.shape[1]), dtype=complex_dtype
    )
    running = np.zeros((points.shape[0], codes.shape[1]), dtype=complex_dtype)
    for position in range(tokens.shape[0] - 2, -1, -1):
        running = token_codes[position + 1][None] + points[:, None] * running
        output[position] = running
    return output


def direct_targets(
    tokens: np.ndarray, codes: np.ndarray, points: np.ndarray
) -> np.ndarray:
    output = np.zeros(
        (tokens.shape[0], points.shape[0], codes.shape[1]),
        dtype=np.complex128,
    )
    for position in range(tokens.shape[0] - 1):
        future = codes[tokens[position + 1 :]]
        powers = points[:, None] ** np.arange(future.shape[0])[None]
        output[position] = np.einsum("mh,hr->mr", powers, future)
    return output


def real_flatten(values: np.ndarray) -> np.ndarray:
    return np.concatenate((values.real, values.imag), axis=-1).reshape(
        values.shape[0], -1
    )


def permutation_separation(
    *,
    pairs: int,
    length: int,
    vocabulary: int,
    code_width: int,
    frequencies: int,
    seed: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    codes = fixed_codes(vocabulary, code_width, seed + 1)
    points = spectral_points(frequencies)
    relative_distances = []
    bow_collisions = 0
    spectral_collisions = 0
    for _ in range(pairs):
        original = rng.integers(0, vocabulary, size=length, dtype=np.int64)
        permuted = original[rng.permutation(length)]
        while np.array_equal(original, permuted):
            permuted = original[rng.permutation(length)]
        bow_original = codes[original].sum(axis=0)
        bow_permuted = codes[permuted].sum(axis=0)
        bow_collisions += int(np.array_equal(bow_original, bow_permuted))
        original_target = recurrence_targets(
            np.concatenate(([0], original)), codes, points, np.float32
        )[0]
        permuted_target = recurrence_targets(
            np.concatenate(([0], permuted)), codes, points, np.float32
        )[0]
        left = real_flatten(original_target[None])[0]
        right = real_flatten(permuted_target[None])[0]
        distance = float(np.linalg.norm(left.astype(np.float64) - right))
        scale = max(
            float(np.linalg.norm(left.astype(np.float64))),
            float(np.linalg.norm(right.astype(np.float64))),
            1e-12,
        )
        relative = distance / scale
        relative_distances.append(relative)
        spectral_collisions += int(np.array_equal(left, right))
    values = np.asarray(relative_distances)
    return {
        "pairs": pairs,
        "bow_collisions": bow_collisions,
        "spectral_float32_collisions": spectral_collisions,
        "minimum_relative_distance": float(values.min()),
        "p01_relative_distance": float(np.quantile(values, 0.01)),
        "median_relative_distance": float(np.median(values)),
    }


def vandermonde_recovery(
    horizon: int, code_width: int, seed: int
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    radius = 0.95
    points = radius * np.exp(2j * np.pi * np.arange(horizon) / horizon)
    future = rng.normal(size=(horizon, code_width))
    vandermonde = points[:, None] ** np.arange(horizon)[None]
    summaries = vandermonde @ future
    recovered = np.linalg.solve(vandermonde, summaries)
    return {
        "maximum_reconstruction_error": float(
            np.max(np.abs(recovered - future))
        ),
        "condition_number": float(np.linalg.cond(vandermonde)),
    }


def resource_ledger(
    hidden: int = 384,
    target_width: int = 64,
    model_parameters: int = 37_758_336,
    vocabulary: int = 49_152,
    context: int = 512,
) -> dict[str, float | int]:
    head = hidden * target_width
    return {
        "hidden": hidden,
        "target_width": target_width,
        "training_only_head_parameters": head,
        "head_fraction_of_model": head / model_parameters,
        "head_fraction_of_unembedding": head / (hidden * vocabulary),
        "target_recurrence_scalar_ops_per_token_upper_bound": 2 * target_width,
        "bf16_target_bits_per_prefix": 16 * target_width,
        "arbitrary_suffix_symbol_bits_lower_bound": context
        * math.log2(vocabulary),
        "extra_exported_parameters": 0,
        "extra_inference_operations": 0,
    }


def run(seed: int, pairs: int) -> dict:
    rng = np.random.default_rng(seed)
    vocabulary = 97
    codes = fixed_codes(vocabulary, 4, seed + 10)
    points = spectral_points(8)
    tokens = rng.integers(0, vocabulary, size=73, dtype=np.int64)
    direct = direct_targets(tokens, codes, points)
    recurrent64 = recurrence_targets(tokens, codes, points, np.float64)
    recurrent32 = recurrence_targets(tokens, codes, points, np.float32)
    recurrence = {
        "float64_maximum_error": float(np.max(np.abs(recurrent64 - direct))),
        "float32_maximum_error": float(np.max(np.abs(recurrent32 - direct))),
    }
    separation = permutation_separation(
        pairs=pairs,
        length=64,
        vocabulary=96,
        code_width=4,
        frequencies=8,
        seed=seed + 20,
    )
    recovery = vandermonde_recovery(8, 4, seed + 30)
    ledger = resource_ledger()
    gates = {
        "reverse_recurrence_matches_direct_float64": recurrence[
            "float64_maximum_error"
        ]
        <= 1e-12,
        "reverse_recurrence_stable_float32": recurrence[
            "float32_maximum_error"
        ]
        <= 1e-4,
        "bow_collides_on_every_same_multiset_pair": separation[
            "bow_collisions"
        ]
        == pairs,
        "spectral_code_separates_every_pair_float32": separation[
            "spectral_float32_collisions"
        ]
        == 0,
        "spectral_p01_margin_above_one_percent": separation[
            "p01_relative_distance"
        ]
        >= 0.01,
        "vandermonde_recovers_eight_horizons": recovery[
            "maximum_reconstruction_error"
        ]
        <= 1e-10
        and recovery["condition_number"] <= 2.0,
        "compact_target_does_not_evade_suffix_bit_bound": ledger[
            "bf16_target_bits_per_prefix"
        ]
        < ledger["arbitrary_suffix_symbol_bits_lower_bound"],
        "export_graph_unchanged": ledger["extra_exported_parameters"] == 0
        and ledger["extra_inference_operations"] == 0,
    }
    return {
        "schema": "spectral-successor-stage0-v1",
        "seed": seed,
        "pairs": pairs,
        "source_sha256": sha256_file(Path(__file__)),
        "test_sha256": sha256_file(TEST),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "measurements": {
            "recurrence": recurrence,
            "permutation_separation": separation,
            "vandermonde_recovery": recovery,
        },
        "resource_ledger": ledger,
        "gates": gates,
        "pass": all(gates.values()),
        "claim_boundary": {
            "proved_if_pass": (
                "cheap recurrence correctness, order separation, finite-precision "
                "margin, finite-horizon recoverability, and export accounting"
            ),
            "not_proved": (
                "language learning gain, compute-equivalent gain, novelty beyond "
                "future-summary prediction, or breakthrough"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=260730)
    parser.add_argument("--pairs", type=int, default=4096)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/spectral-successor-stage0.json"),
    )
    args = parser.parse_args()
    result = run(args.seed, args.pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
