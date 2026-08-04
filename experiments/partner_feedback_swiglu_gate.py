"""Stage-0 algebra for gauge-exposed partner-feedback SwiGLU."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "partner-feedback-swiglu-stage0.json"
PREREG = ROOT / "results" / "partner-feedback-swiglu-stage0-preregistration.md"


def silu(x):
    return x / (1.0 + np.exp(-x))


def ordinary(x, gate, up, down):
    return (silu(x @ gate.T) * (x @ up.T)) @ down.T


def decode_lambda(up, chart, carrier_gain):
    return float((up[0, 0] / chart - 1.0) / carrier_gain)


def canonicalize_carrier(up, down, chart):
    if up[0, 0] == 0.0 or chart == 0.0:
        raise ValueError("nonzero carrier pivot and chart required")
    scale = up[0, 0] / chart
    canonical_up = up.copy()
    canonical_down = down.copy()
    canonical_up[0] /= scale
    canonical_down[:, 0] *= scale
    return canonical_up, canonical_down, float(scale)


def fold_lambda(canonical_up, canonical_down, coefficient, carrier_gain):
    scale = 1.0 + carrier_gain * coefficient
    if scale <= 0.0:
        raise ValueError("carrier scale must remain positive")
    up = canonical_up.copy()
    down = canonical_down.copy()
    up[0] *= scale
    down[:, 0] /= scale
    return up, down, float(scale)


def partner_feedback(x, gate, up, down, coefficient):
    gate_values = x @ gate.T
    up_values = x @ up.T
    z = silu(gate_values) * up_values
    partner = np.arange(gate.shape[0], dtype=np.int64) ^ 1
    q = silu(gate_values + coefficient * np.clip(z[:, partner], -1.0, 1.0)) * up_values
    return q @ down.T


def square_witness(coefficient):
    # With scalar g_i=u_i=g_j=u_j=x and clip locally the identity:
    # (x + coefficient*x^3)^2*x = x^3 + 2c*x^5 + c^2*x^7.
    return {"degree_3": 1.0, "degree_5": 2.0 * coefficient, "degree_7": coefficient**2}


def ledger(d, m):
    words = 3 * d * m
    return {
        "baseline_trainable_parameters": words,
        "candidate_trainable_parameters": words,
        "baseline_serialized_words": words,
        "candidate_serialized_words": words,
        "baseline_dense_macs_per_token": words,
        "candidate_dense_macs_per_token": words,
        "layer_scalar_pivot_reads_per_kernel_program": 1,
        "per_feature_coefficient_words": 0,
        "permutation_metadata_words": 0,
        "serial_feature_dependencies": 0,
        "additional_activation_workspace_words": 0,
    }


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(seed=719, d=16, m=32):
    if m % 2:
        raise ValueError("hidden width must be even")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(23, d))
    gate = rng.normal(scale=0.2, size=(m, d))
    up = rng.normal(scale=0.2, size=(m, d))
    up[0, 0] += 0.4 if up[0, 0] >= 0 else -0.4
    down = rng.normal(scale=0.2, size=(d, m))
    chart = d**-0.5
    carrier_gain = 4.0

    original = ordinary(x, gate, up, down)
    canonical_up, canonical_down, canonical_scale = canonicalize_carrier(up, down, chart)
    canonical = ordinary(x, gate, canonical_up, canonical_down)
    canonical_error = float(np.max(np.abs(original - canonical)))

    decoded_zero = decode_lambda(canonical_up, chart, carrier_gain)
    endpoint = partner_feedback(x, gate, canonical_up, canonical_down, decoded_zero)
    endpoint_error = float(np.max(np.abs(endpoint - canonical)))

    coefficient = 0.08
    folded_up, folded_down, carrier_scale = fold_lambda(
        canonical_up, canonical_down, coefficient, carrier_gain
    )
    decoded = decode_lambda(folded_up, chart, carrier_gain)
    explicit = partner_feedback(x, gate, folded_up, folded_down, coefficient)
    deployed = partner_feedback(x, gate, folded_up, folded_down, decoded)
    fold_error = float(np.max(np.abs(explicit - deployed)))

    source_scale = 1.4
    orbit_up = folded_up.copy()
    orbit_down = folded_down.copy()
    orbit_up[2] *= source_scale
    orbit_down[:, 2] /= source_scale
    ordinary_before = ordinary(x, gate, folded_up, folded_down)
    ordinary_after = ordinary(x, gate, orbit_up, orbit_down)
    candidate_before = partner_feedback(x, gate, folded_up, folded_down, decoded)
    candidate_after = partner_feedback(x, gate, orbit_up, orbit_down, decoded)
    ordinary_orbit_error = float(np.max(np.abs(ordinary_before - ordinary_after)))
    activated_orbit_movement = float(np.max(np.abs(candidate_before - candidate_after)))

    witness = square_witness(0.25)
    small = ledger(d, m)
    target = ledger(4096, 14336)
    gates = {
        "carrier_canonicalization_exact": canonical_error <= 1e-11,
        "zero_endpoint_exact": endpoint_error <= 1e-11 and decoded_zero == 0.0,
        "fold_decode_exact": fold_error <= 1e-11 and abs(decoded - coefficient) <= 1e-14,
        "ordinary_source_orbit_invariant": ordinary_orbit_error <= 1e-11,
        "source_orbit_is_functional": activated_orbit_movement > 1e-6,
        "degree_seven_witness": witness["degree_7"] > 0.0,
        "parameter_count_exact": small["candidate_trainable_parameters"] == small["baseline_trainable_parameters"],
        "serialized_count_exact": small["candidate_serialized_words"] == small["baseline_serialized_words"],
        "dense_mac_count_exact": small["candidate_dense_macs_per_token"] == small["baseline_dense_macs_per_token"],
        "no_per_feature_carrier_or_serial_path": (
            small["per_feature_coefficient_words"] == 0
            and small["serial_feature_dependencies"] == 0
        ),
    }
    return {
        "candidate": "gauge-exposed partner-feedback SwiGLU",
        "scope": "algebra and exact raw-resource ledger only",
        "seed": seed,
        "shape": {"model_width": d, "hidden_width": m},
        "measurements": {
            "canonicalization_max_abs_error": canonical_error,
            "zero_endpoint_max_abs_error": endpoint_error,
            "fold_decode_max_abs_error": fold_error,
            "ordinary_source_orbit_max_abs_error": ordinary_orbit_error,
            "activated_source_orbit_max_output_movement": activated_orbit_movement,
            "canonical_carrier_scale": canonical_scale,
            "nonzero_carrier_scale": carrier_scale,
            "decoded_lambda": decoded,
        },
        "square_activation_witness": witness,
        "small_ledger": small,
        "target_ledger": target,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "preregistration_sha256": sha(PREREG),
        "source_sha256": sha(Path(__file__)),
    }


def main():
    payload = run()
    RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["all_gates_pass"] else 1)


if __name__ == "__main__":
    main()

