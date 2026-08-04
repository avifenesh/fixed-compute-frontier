#!/usr/bin/env python3
"""Exact export algebra for a training-only RMS-gain scaffold."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def chart(z: np.ndarray, gamma: np.ndarray, mode: str) -> np.ndarray:
    if mode == "null":
        return z
    if mode == "scale":
        return z + gamma * z
    if mode == "bias":
        return z + gamma
    if mode == "hinge":
        return z + gamma * np.abs(z)
    if mode == "centered":
        return z + gamma * (np.abs(z) - np.sqrt(2.0 / np.pi))
    raise ValueError(mode)


def run() -> dict:
    rng = np.random.default_rng(20260728)
    z = rng.normal(size=(17, 12))
    gain = rng.uniform(0.5, 1.5, size=12)
    gamma = rng.uniform(-0.2, 0.2, size=12)
    consumers = [rng.normal(size=(width, 12)) for width in (12, 36, 48)]
    errors = {}
    for mode in ("null", "scale", "bias", "hinge", "centered"):
        transformed = chart(z, gamma, mode)
        mode_errors = []
        for weight in consumers:
            training = (gain * transformed) @ weight.T
            exported_weight = weight * gain[None, :]
            served = transformed @ exported_weight.T
            mode_errors.append(float(np.max(np.abs(training - served))))
        errors[mode] = mode_errors
    return {
        "schema": "train-time-gauge-scaffold-algebra-v1",
        "multiple_consumer_export_max_errors": errors,
        "training_norm_parameters": 24,
        "served_norm_parameters": 12,
        "baseline_norm_parameters": 12,
        "matrix_shapes_unchanged": True,
    }


def main() -> None:
    payload = run()
    output = Path("results/train-time-gauge-scaffold-algebra.json")
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

