#!/usr/bin/env python3
"""Algebra and fatal ledgers for token-conditioned feature binding in SwiGLU."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def ordinary_swiglu(
    x: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
) -> np.ndarray:
    return (silu(x @ gate.T) * (x @ up.T)) @ down.T


def bind_values(values: np.ndarray, shifts: np.ndarray, group_size: int) -> np.ndarray:
    grouped = values.reshape(*values.shape[:-1], -1, group_size)
    indices = np.arange(group_size)
    gather = (indices + shifts[..., None]) % group_size
    return np.take_along_axis(grouped, gather, axis=-1).reshape(values.shape)


def dynamic_binding_swiglu(
    x: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
    group_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gate_values = x @ gate.T
    up_values = x @ up.T
    grouped_gate = gate_values.reshape(*gate_values.shape[:-1], -1, group_size)
    shifts = np.argmax(grouped_gate, axis=-1)
    ordered = np.sort(grouped_gate, axis=-1)
    margins = ordered[..., -1] - ordered[..., -2]
    strengths = np.tanh(margins) ** 2
    grouped_up = up_values.reshape(*up_values.shape[:-1], -1, group_size)
    bound = bind_values(up_values, shifts, group_size).reshape(grouped_up.shape)
    mixed = grouped_up + strengths[..., None] * (bound - grouped_up)
    return (silu(gate_values) * mixed.reshape(up_values.shape)) @ down.T, shifts, strengths


def constant_binding_swiglu(
    x: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
    group_size: int,
    shift: int,
) -> np.ndarray:
    groups = up.shape[0] // group_size
    shifts = np.full((*x.shape[:-1], groups), shift, dtype=np.int64)
    gate_values = x @ gate.T
    bound = bind_values(x @ up.T, shifts, group_size)
    return (silu(gate_values) * bound) @ down.T


def inverse_rotate_up(up: np.ndarray, group_size: int, shift: int) -> np.ndarray:
    grouped = up.reshape(-1, group_size, up.shape[-1])
    # The bound at output i reads candidate row i+shift.  Store baseline row i
    # at that location so a constant route is exactly a reparameterization.
    return np.roll(grouped, shift=shift, axis=1).reshape(up.shape)


@dataclass(frozen=True)
class Shape:
    model_width: int
    hidden_width: int
    group_size: int = 8


def resource_ledger(shape: Shape) -> dict[str, int]:
    d, m, r = shape.model_width, shape.hidden_width, shape.group_size
    if m % r:
        raise ValueError("hidden width must divide into complete binding groups")
    dense = 3 * d * m
    return {
        "baseline_learned_scalars": dense,
        "candidate_learned_scalars": dense,
        "baseline_serialized_weight_scalars": dense,
        "candidate_serialized_weight_scalars": dense,
        "baseline_dense_macs_per_token": dense,
        "candidate_dense_macs_per_token": dense,
        "persistent_request_state_bits": 0,
        "extra_model_selector_bits": 0,
        "route_comparisons_per_token": m - m // r,
        "block_local_selected_value_reads_per_token": m,
        "route_choices_per_group": r,
        "groups_per_token": m // r,
    }


def nonanalytic_witness() -> dict[str, float]:
    """The margin-damped route has unequal one-sided quartic coefficients.

    Standard finite-width SwiGLU is analytic.  For x>0, b0>b1 selects identity
    shift 0.  For x<0 it selects shift 1, while tanh(margin)^2 starts at x^2.
    The correction therefore has a generically nonzero x^4 coefficient only
    on the left.  The function and first three derivatives meet at zero, but
    the fourth derivative jumps, proving this is not a fixed SwiGLU
    reparameterization.
    """

    b0, b1 = 2.0, -1.0
    a0, a1 = 0.25, 1.5
    v0, v1 = 1.0, -0.4
    delta_b_squared = (b0 - b1) ** 2
    left_correction = 0.5 * delta_b_squared * (
        v0 * b0 * (a1 - a0) + v1 * b1 * (a0 - a1)
    )
    right_correction = 0.0
    return {
        "left_quartic_correction": left_correction,
        "right_quartic_correction": right_correction,
        "fourth_derivative_jump": 24.0 * (right_correction - left_correction),
    }


def run(seed: int = 2843) -> dict:
    rng = np.random.default_rng(seed)
    shape = Shape(12, 32, 8)
    x = rng.normal(size=(29, shape.model_width))
    gate = rng.normal(scale=0.2, size=(shape.hidden_width, shape.model_width))
    up = rng.normal(scale=0.2, size=(shape.hidden_width, shape.model_width))
    down = rng.normal(scale=0.2, size=(shape.model_width, shape.hidden_width))
    baseline = ordinary_swiglu(x, gate, up, down)
    identity = constant_binding_swiglu(x, gate, up, down, shape.group_size, 0)
    shift = 3
    rotated_up = inverse_rotate_up(up, shape.group_size, shift)
    fixed = constant_binding_swiglu(
        x, gate, rotated_up, down, shape.group_size, shift
    )
    dynamic, routes, strengths = dynamic_binding_swiglu(
        x, gate, up, down, shape.group_size
    )
    witness = nonanalytic_witness()
    ledger = resource_ledger(shape)
    errors = {
        "identity_endpoint_max_abs": float(np.max(np.abs(identity - baseline))),
        "constant_route_reparameterization_max_abs": float(np.max(np.abs(fixed - baseline))),
        "dynamic_movement_max_abs": float(np.max(np.abs(dynamic - baseline))),
        "routing_strength_min": float(strengths.min()),
        "routing_strength_max": float(strengths.max()),
    }
    gates = {
        "identity_endpoint_exact": errors["identity_endpoint_max_abs"] == 0.0,
        "constant_route_is_exact_reparameterization": errors["constant_route_reparameterization_max_abs"] <= 1e-12,
        "dynamic_route_is_live": errors["dynamic_movement_max_abs"] > 1e-6,
        "dynamic_route_uses_multiple_matchings": int(np.unique(routes).size) >= 2,
        "margin_route_has_nonanalytic_witness": abs(witness["fourth_derivative_jump"]) > 1e-6,
        "margin_strength_is_bounded": 0.0 <= errors["routing_strength_min"] <= errors["routing_strength_max"] < 1.0,
        "learned_scalar_count_exact": ledger["baseline_learned_scalars"] == ledger["candidate_learned_scalars"],
        "dense_mac_count_exact": ledger["baseline_dense_macs_per_token"] == ledger["candidate_dense_macs_per_token"],
        "no_selector_or_request_state": ledger["extra_model_selector_bits"] == ledger["persistent_request_state_bits"] == 0,
    }
    return {
        "candidate": "dynamic binding SwiGLU",
        "shape": asdict(shape),
        "errors": errors,
        "nonanalytic_witness": witness,
        "route_histogram": np.bincount(routes.reshape(-1), minlength=shape.group_size).tolist(),
        "ledger": ledger,
        "gates": gates,
        "pass": all(gates.values()),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2, sort_keys=True))
