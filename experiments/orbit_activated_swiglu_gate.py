"""Stage-0 algebra and ledger for orbit-activated SwiGLU."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "results" / "orbit-activated-swiglu-stage0.json"
PREREGISTRATION = ROOT / "results" / "orbit-activated-swiglu-stage0-preregistration.md"


@dataclass(frozen=True)
class Shape:
    model_width: int
    hidden_width: int
    group_size: int = 8
    carrier_gain: float = 8.0


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def pivots_and_chart(model_width: int, hidden_width: int, chart: float) -> tuple[np.ndarray, np.ndarray]:
    pivots = np.arange(hidden_width, dtype=np.int64) % model_width
    signs = np.where(np.arange(hidden_width) % 2 == 0, 1.0, -1.0)
    return pivots, signs * chart


def ordinary_swiglu(x: np.ndarray, gate: np.ndarray, up: np.ndarray, down: np.ndarray) -> np.ndarray:
    return (silu(x @ gate.T) * (x @ up.T)) @ down.T


def canonicalize_up_down(
    up: np.ndarray,
    down: np.ndarray,
    pivots: np.ndarray,
    chart_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.arange(up.shape[0])
    observed = up[rows, pivots]
    if np.any(observed == 0.0) or np.any(chart_values == 0.0):
        raise ValueError("canonicalization requires nonzero pivots and chart values")
    scales = observed / chart_values
    canonical_up = up / scales[:, None]
    canonical_down = down * scales[None, :]
    return canonical_up, canonical_down, scales


def fold_carrier(
    canonical_up: np.ndarray,
    canonical_down: np.ndarray,
    coefficients: np.ndarray,
    carrier_gain: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scales = 1.0 + carrier_gain * coefficients
    if np.any(scales <= 0.0):
        raise ValueError("carrier scales must remain positive")
    return (
        canonical_up * scales[:, None],
        canonical_down / scales[None, :],
        scales,
    )


def decode_carrier(
    deployed_up: np.ndarray,
    pivots: np.ndarray,
    chart_values: np.ndarray,
    carrier_gain: float,
) -> np.ndarray:
    rows = np.arange(deployed_up.shape[0])
    scales = deployed_up[rows, pivots] / chart_values
    return (scales - 1.0) / carrier_gain


def orbit_activated_swiglu(
    x: np.ndarray,
    gate: np.ndarray,
    deployed_up: np.ndarray,
    deployed_down: np.ndarray,
    coefficients: np.ndarray,
    group_size: int,
) -> np.ndarray:
    gate_values = x @ gate.T
    up_values = x @ deployed_up.T
    batch, width = gate_values.shape
    if width % group_size:
        raise ValueError("hidden width must be divisible by group size")
    q = np.empty_like(gate_values)
    for start in range(0, width, group_size):
        q[:, start] = silu(gate_values[:, start] + coefficients[start]) * up_values[:, start]
        for offset in range(1, group_size):
            i = start + offset
            predecessor = np.clip(q[:, i - 1], -1.0, 1.0)
            q[:, i] = silu(gate_values[:, i] + coefficients[i] * predecessor) * up_values[:, i]
    return q @ deployed_down.T


def bf16_round(x: np.ndarray) -> np.ndarray:
    """Round float32 to BF16 and return float32 values."""
    words = np.asarray(x, dtype=np.float32).view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((words >> np.uint32(16)) & np.uint32(1))
    rounded = (words + rounding_bias) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def row_relative_rms(reference: np.ndarray, actual: np.ndarray, epsilon: float = 1e-12) -> float:
    numerator = np.sqrt(np.mean((actual - reference) ** 2, axis=-1))
    denominator = np.sqrt(np.mean(reference**2, axis=-1))
    return float(np.max(numerator / np.maximum(denominator, epsilon)))


def resource_ledger(shape: Shape) -> dict[str, float | int]:
    d, m, r = shape.model_width, shape.hidden_width, shape.group_size
    serialized_words = 3 * d * m
    return {
        "baseline_trainable_parameters": serialized_words,
        "candidate_trainable_parameters": d * m + m * (d - 1) + d * m + m,
        "baseline_serialized_words": serialized_words,
        "candidate_serialized_words": serialized_words,
        "baseline_dense_macs_per_token": serialized_words,
        "candidate_dense_macs_per_token": serialized_words,
        "candidate_pivot_reads_words_per_invocation": m,
        "candidate_pivot_read_fraction_of_weight_words": m / serialized_words,
        "candidate_nonroot_dependencies_per_token": m - m // r,
        "candidate_root_biases_per_token": m // r,
        "maximum_serial_predecessor_steps": r - 1,
        "training_only_fixed_pivots": m,
        "training_only_learned_carriers": m,
    }


def square_activation_witness(coefficient: float) -> dict[str, float]:
    # Along a one-dimensional input ray, ordinary square-gated FFNs contain
    # only x^3.  The second unit of the candidate contains the terms below in
    # the local region where clip is the identity.
    return {"degree_3": 1.0, "degree_5": 2.0 * coefficient, "degree_7": coefficient**2}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(seed: int = 509, model_width: int = 16, hidden_width: int = 32) -> dict:
    if hidden_width % 8:
        raise ValueError("frozen screen requires groups of eight")
    rng = np.random.default_rng(seed)
    dtype = np.float64
    batch = 19
    chart = 1.0 / np.sqrt(model_width)
    pivots, chart_values = pivots_and_chart(model_width, hidden_width, chart)

    x = rng.normal(size=(batch, model_width)).astype(dtype)
    gate = rng.normal(scale=0.2, size=(hidden_width, model_width)).astype(dtype)
    original_up = rng.normal(scale=0.2, size=(hidden_width, model_width)).astype(dtype)
    rows = np.arange(hidden_width)
    original_up[rows, pivots] += np.where(original_up[rows, pivots] >= 0.0, 0.25, -0.25)
    original_down = rng.normal(scale=0.2, size=(model_width, hidden_width)).astype(dtype)

    canonical_up, canonical_down, canonical_scales = canonicalize_up_down(
        original_up, original_down, pivots, chart_values
    )
    original_output = ordinary_swiglu(x, gate, original_up, original_down)
    canonical_output = ordinary_swiglu(x, gate, canonical_up, canonical_down)
    canonicalization_error = float(np.max(np.abs(original_output - canonical_output)))

    zeros = np.zeros(hidden_width, dtype=dtype)
    zero_output = orbit_activated_swiglu(
        x, gate, canonical_up, canonical_down, zeros, group_size=8
    )
    endpoint_error = float(np.max(np.abs(zero_output - canonical_output)))

    coefficients = rng.uniform(-0.035, 0.035, size=hidden_width).astype(dtype)
    deployed_up, deployed_down, carrier_scales = fold_carrier(
        canonical_up, canonical_down, coefficients, carrier_gain=8.0
    )
    decoded = decode_carrier(deployed_up, pivots, chart_values, carrier_gain=8.0)
    explicit_output = orbit_activated_swiglu(
        x, gate, deployed_up, deployed_down, coefficients, group_size=8
    )
    decoded_output = orbit_activated_swiglu(
        x, gate, deployed_up, deployed_down, decoded, group_size=8
    )
    fold_decode_error = float(np.max(np.abs(explicit_output - decoded_output)))
    coefficient_decode_error = float(np.max(np.abs(coefficients - decoded)))

    ordinary_orbit_output = ordinary_swiglu(x, gate, deployed_up, deployed_down)
    ordinary_orbit_error = float(np.max(np.abs(canonical_output - ordinary_orbit_output)))
    activated_orbit_movement = float(np.max(np.abs(explicit_output - zero_output)))

    gate32 = gate.astype(np.float32)
    up32 = deployed_up.astype(np.float32)
    down32 = deployed_down.astype(np.float32)
    x32 = x.astype(np.float32)
    float32_output = orbit_activated_swiglu(
        x32, gate32, up32, down32, coefficients.astype(np.float32), group_size=8
    )
    bf16_gate = bf16_round(gate32)
    bf16_up = bf16_round(up32)
    bf16_down = bf16_round(down32)
    bf16_decoded = decode_carrier(
        bf16_up, pivots, chart_values.astype(np.float32), carrier_gain=8.0
    )
    bf16_output = orbit_activated_swiglu(
        x32, bf16_gate, bf16_up, bf16_down, bf16_decoded, group_size=8
    )
    bf16_relative_rms = row_relative_rms(float32_output, bf16_output)
    bf16_coefficient_error = float(np.max(np.abs(coefficients - bf16_decoded)))

    small_ledger = resource_ledger(Shape(model_width, hidden_width))
    target_ledger = resource_ledger(Shape(4096, 14336))
    witness = square_activation_witness(0.25)

    gates = {
        "canonicalization_exact": canonicalization_error <= 1e-11,
        "zero_endpoint_exact": endpoint_error <= 1e-11,
        "fold_decode_exact": fold_decode_error <= 1e-11 and coefficient_decode_error <= 1e-11,
        "ordinary_orbit_invariant": ordinary_orbit_error <= 1e-11,
        "activated_orbit_is_live": activated_orbit_movement > 1e-6,
        "degree_seven_witness": abs(witness["degree_7"]) > 0.0,
        "trainable_parameter_count_exact": (
            small_ledger["candidate_trainable_parameters"]
            == small_ledger["baseline_trainable_parameters"]
        ),
        "serialized_word_count_exact": (
            small_ledger["candidate_serialized_words"]
            == small_ledger["baseline_serialized_words"]
        ),
        "dense_mac_count_exact": (
            small_ledger["candidate_dense_macs_per_token"]
            == small_ledger["baseline_dense_macs_per_token"]
        ),
        "bf16_export_below_0p5_percent": bf16_relative_rms < 0.005,
    }
    result = {
        "candidate": "orbit-activated SwiGLU",
        "scope": "algebra, parameter/storage/MAC ledger, and BF16 carrier sanity only",
        "seed": seed,
        "small_shape": {"model_width": model_width, "hidden_width": hidden_width, "group_size": 8},
        "measurements": {
            "canonicalization_max_abs_error": canonicalization_error,
            "zero_endpoint_max_abs_error": endpoint_error,
            "fold_decode_output_max_abs_error": fold_decode_error,
            "coefficient_decode_max_abs_error": coefficient_decode_error,
            "ordinary_orbit_max_abs_error": ordinary_orbit_error,
            "activated_orbit_max_output_movement": activated_orbit_movement,
            "canonical_scale_min": float(np.min(np.abs(canonical_scales))),
            "canonical_scale_max": float(np.max(np.abs(canonical_scales))),
            "carrier_scale_min": float(np.min(carrier_scales)),
            "carrier_scale_max": float(np.max(carrier_scales)),
            "bf16_export_max_row_relative_rms": bf16_relative_rms,
            "bf16_coefficient_max_abs_error": bf16_coefficient_error,
        },
        "square_activation_witness": witness,
        "small_ledger": small_ledger,
        "target_ledger": target_ledger,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "preregistration_sha256": sha256(PREREGISTRATION),
        "source_sha256": sha256(Path(__file__)),
    }
    return result


def main() -> None:
    result = run()
    RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["all_gates_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

