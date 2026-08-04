#!/usr/bin/env python3
"""Frozen T84 power/sensitivity calculation source.

Source-audit only until an execution receipt exists.  These calculations are
false-negative sensitivity descriptions, not a substitute for the 106 frozen
confirmatory statements and not evidence for the candidate.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/shared-score-heterogeneous-attention-t84-power-canonical.json"
TCRIT = 4.1014945453569363208
DF = 15
SAMPLES = 1_000_000
DISTANCES = (0.75, 1.00, 1.25, 1.50, 2.00)


def one_sided_power(distance: float, seed: int) -> float:
    rng = np.random.Generator(np.random.PCG64(seed))
    numerator = rng.standard_normal(SAMPLES) + distance * math.sqrt(16.0)
    denominator = np.sqrt(rng.chisquare(DF, SAMPLES) / DF)
    return float(np.mean(numerator / denominator >= TCRIT))


def joint_macro_power(correlation: float, seed: int) -> float:
    rng = np.random.Generator(np.random.PCG64(seed))
    shared = rng.standard_normal((SAMPLES, 1))
    independent = rng.standard_normal((SAMPLES, 12))
    numerator = (
        math.sqrt(correlation) * shared
        + math.sqrt(1.0 - correlation) * independent
        + 1.25 * math.sqrt(16.0)
    )
    denominator = np.sqrt(rng.chisquare(DF, (SAMPLES, 12)) / DF)
    return float(np.mean(np.all(numerator / denominator >= TCRIT, axis=1)))


def calculate() -> dict[str, object]:
    return {
        "family_size": 106,
        "tcrit": TCRIT,
        "df": DF,
        "draws": SAMPLES,
        "single": {
            f"{distance:.2f}": one_sided_power(distance, 840150 + index)
            for index, distance in enumerate(DISTANCES)
        },
        "joint": {
            "0.0": joint_macro_power(0.0, 840160),
            "0.5": joint_macro_power(0.5, 840165),
            "0.9": joint_macro_power(0.9, 840169),
        },
    }


def canonicalize(payload: dict[str, object]) -> dict[str, object]:
    single = payload["single"]
    joint = payload["joint"]
    if not isinstance(single, dict) or not isinstance(joint, dict):
        raise TypeError("unexpected power payload")
    return {
        **payload,
        "single": {key: round(float(value), 3) for key, value in single.items()},
        "joint": {key: round(float(value), 3) for key, value in joint.items()},
    }


def main() -> None:
    calculated = canonicalize(calculate())
    frozen = json.loads(OUTPUT.read_text(encoding="utf-8"))
    if calculated != frozen:
        raise RuntimeError("frozen canonical power calculation does not replay")
    print(json.dumps(calculated, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
