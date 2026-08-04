#!/usr/bin/env python3
"""Exact algebra and resource ledger for gauge-coded activation bits.

The construction uses a scale/sign gauge of a SwiGLU up/down channel.  A
discrete activation selector is paid for by one removed gauge coordinate per
group, then serialized in the sign of already-present row quantization scales.
The signed-scale representation is a conditional serving assumption, not a
claim about ordinary dense BF16 checkpoints.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def even_gate(x: np.ndarray) -> np.ndarray:
    """The alternative gate: even, quadratic at the origin, linear at infinity."""

    return x * np.tanh(x)


def representative_pivots(
    model_width: int,
    hidden_width: int,
    group_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if hidden_width % group_size:
        raise ValueError("hidden width must be divisible by group size")
    rows = np.arange(0, hidden_width, group_size, dtype=np.int64)
    pivots = np.arange(rows.size, dtype=np.int64) % model_width
    return rows, pivots


def canonicalize_representatives(
    up: np.ndarray,
    down: np.ndarray,
    rows: np.ndarray,
    pivots: np.ndarray,
    chart: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fix one up-row coordinate per group and fold its scale into down."""

    observed = up[rows, pivots]
    if chart <= 0.0 or np.any(observed == 0.0):
        raise ValueError("canonicalization needs a positive chart and nonzero pivots")
    scales = observed / chart
    canonical_up = up.copy()
    canonical_down = down.copy()
    canonical_up[rows] /= scales[:, None]
    canonical_down[:, rows] *= scales[None, :]
    return canonical_up, canonical_down, scales


def ordinary_swiglu(
    x: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
) -> np.ndarray:
    return (silu(x @ gate.T) * (x @ up.T)) @ down.T


def gaugebit_swiglu(
    x: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
    group_bits: np.ndarray,
    group_size: int,
) -> np.ndarray:
    if up.shape[0] != group_bits.size * group_size:
        raise ValueError("one selector bit is required per complete group")
    gate_values = x @ gate.T
    channel_bits = np.repeat(group_bits.astype(bool), group_size)
    activation = np.where(channel_bits[None, :], even_gate(gate_values), silu(gate_values))
    return (activation * (x @ up.T)) @ down.T


def export_sign_coded(
    up: np.ndarray,
    down: np.ndarray,
    group_bits: np.ndarray,
    group_size: int,
    positive_row_scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode bits in signed row scales while preserving the up/down product."""

    if np.any(positive_row_scales <= 0.0):
        raise ValueError("input quantization scales must be positive")
    channel_bits = np.repeat(group_bits.astype(bool), group_size)
    if channel_bits.size != up.shape[0] or positive_row_scales.size != up.shape[0]:
        raise ValueError("shape mismatch in sign-coded export")
    signs = np.where(channel_bits, -1.0, 1.0)
    return (
        up * signs[:, None],
        down * signs[None, :],
        positive_row_scales * signs,
    )


def decode_group_bits(signed_row_scales: np.ndarray, group_size: int) -> np.ndarray:
    if signed_row_scales.size % group_size:
        raise ValueError("row scales do not form complete groups")
    signs = np.signbit(signed_row_scales).reshape(-1, group_size)
    if np.any(signs != signs[:, :1]):
        raise ValueError("a serving group contains mixed selector signs")
    return signs[:, 0]


@dataclass(frozen=True)
class Shape:
    model_width: int
    hidden_width: int
    group_size: int = 32


def resource_ledger(shape: Shape) -> dict[str, int | str]:
    d, m, r = shape.model_width, shape.hidden_width, shape.group_size
    if m % r:
        raise ValueError("hidden width must be divisible by group size")
    groups = m // r
    baseline = 3 * d * m
    candidate = d * m + (d * m - groups) + d * m + groups
    return {
        "baseline_learned_scalars": baseline,
        "candidate_learned_scalars": candidate,
        "baseline_serialized_weight_values": baseline,
        "candidate_serialized_weight_values": baseline,
        "baseline_dense_macs_per_token": baseline,
        "candidate_dense_macs_per_token": baseline,
        "selector_groups": groups,
        "extra_serialized_selector_bits_if_signed_scales_exist": 0,
        "activation_evaluations_per_token": m,
        "serving_condition": "existing per-row scales must preserve a sign bit",
    }


def run(seed: int = 2718) -> dict:
    rng = np.random.default_rng(seed)
    shape = Shape(model_width=16, hidden_width=64, group_size=8)
    x = rng.normal(size=(23, shape.model_width))
    gate = rng.normal(scale=0.2, size=(shape.hidden_width, shape.model_width))
    up = rng.normal(scale=0.2, size=(shape.hidden_width, shape.model_width))
    down = rng.normal(scale=0.2, size=(shape.model_width, shape.hidden_width))
    rows, pivots = representative_pivots(
        shape.model_width, shape.hidden_width, shape.group_size
    )
    up[rows, pivots] += np.where(up[rows, pivots] >= 0.0, 0.3, -0.3)
    chart = 1.0 / np.sqrt(shape.model_width)
    canonical_up, canonical_down, _ = canonicalize_representatives(
        up, down, rows, pivots, chart
    )
    baseline = ordinary_swiglu(x, gate, up, down)
    canonical = ordinary_swiglu(x, gate, canonical_up, canonical_down)
    zeros = np.zeros(rows.size, dtype=bool)
    endpoint = gaugebit_swiglu(
        x, gate, canonical_up, canonical_down, zeros, shape.group_size
    )
    bits = rng.integers(0, 2, size=rows.size).astype(bool)
    explicit = gaugebit_swiglu(
        x, gate, canonical_up, canonical_down, bits, shape.group_size
    )
    positive_scales = rng.uniform(0.01, 0.2, size=shape.hidden_width)
    deployed_up, deployed_down, signed_scales = export_sign_coded(
        canonical_up, canonical_down, bits, shape.group_size, positive_scales
    )
    decoded = decode_group_bits(signed_scales, shape.group_size)
    deployed = gaugebit_swiglu(
        x, gate, deployed_up, deployed_down, decoded, shape.group_size
    )
    identity_x = np.linspace(-8.0, 8.0, 10001)
    identity_rhs = silu(2.0 * identity_x) - silu(identity_x) + silu(-identity_x)
    ledger = resource_ledger(shape)
    errors = {
        "canonical_endpoint_max_abs": float(np.max(np.abs(baseline - canonical))),
        "zero_bit_endpoint_max_abs": float(np.max(np.abs(canonical - endpoint))),
        "sign_coded_export_max_abs": float(np.max(np.abs(explicit - deployed))),
        "activation_identity_max_abs": float(np.max(np.abs(even_gate(identity_x) - identity_rhs))),
    }
    gates = {
        "canonical_endpoint_exact": errors["canonical_endpoint_max_abs"] <= 1e-11,
        "zero_bit_endpoint_exact": errors["zero_bit_endpoint_max_abs"] <= 1e-11,
        "sign_coded_export_exact": errors["sign_coded_export_max_abs"] <= 1e-11,
        "activation_identity_exact": errors["activation_identity_max_abs"] <= 1e-12,
        "learned_count_exact": ledger["baseline_learned_scalars"] == ledger["candidate_learned_scalars"],
        "serialized_count_exact_under_signed_scale_condition": (
            ledger["baseline_serialized_weight_values"]
            == ledger["candidate_serialized_weight_values"]
            and ledger["extra_serialized_selector_bits_if_signed_scales_exist"] == 0
        ),
        "dense_mac_count_exact": ledger["baseline_dense_macs_per_token"] == ledger["candidate_dense_macs_per_token"],
    }
    return {
        "candidate": "GaugeBit SwiGLU",
        "shape": asdict(shape),
        "errors": errors,
        "ledger": ledger,
        "gates": gates,
        "pass": all(gates.values()),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2, sort_keys=True))
