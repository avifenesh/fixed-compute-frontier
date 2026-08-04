#!/usr/bin/env python3
"""Stage-0 falsification gates for the radial-trust Cayley program tree."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from experiments.cayley_program_tree import CayleyProgramTreeMLP, exact_ledger
from experiments.radial_trust_cayley_program_tree import (
    RadialTrustCayleyProgramTreeMLP,
)
from experiments.reflex_swiglu_lm_screen import sha256_file, write_payload


PREREGISTRATION = Path(
    "results/radial-trust-cayley-program-tree-stage0-preregistration.md"
)
MODULE = Path("experiments/radial_trust_cayley_program_tree.py")
TEST = Path("tests/test_radial_trust_cayley_program_tree.py")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rms(values: torch.Tensor) -> torch.Tensor:
    return values.double().square().mean(dim=-1).sqrt()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=815)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/radial-trust-cayley-program-tree-stage0.json"),
    )
    args = parser.parse_args()
    seed_everything(args.seed)

    width = 384
    depth = 9
    trust_rms = 1.0
    base = CayleyProgramTreeMLP(width=width, depth=depth, seed=args.seed)
    candidate = RadialTrustCayleyProgramTreeMLP(
        width=width, depth=depth, trust_rms=trust_rms, seed=args.seed
    )
    parameter_gate = candidate.parameter_count() == base.parameter_count()
    state_dict_gate = set(candidate.state_dict()) == set(base.state_dict())

    random_delta = torch.randn(64, width) * 1000.0
    trusted = candidate.radial_trust_region(random_delta)
    random_trust_max = float(rms(trusted).max())
    trust_bound_gate = random_trust_max < trust_rms

    small = RadialTrustCayleyProgramTreeMLP(
        width=8, depth=3, trust_rms=trust_rms, seed=args.seed
    ).double()
    point = torch.randn(8, dtype=torch.float64) * 7.0
    jacobian = torch.autograd.functional.jacobian(
        lambda value: small.radial_trust_region(value[None])[0], point
    )
    singular_values = torch.linalg.svdvals(jacobian)
    jacobian_rank = int(torch.linalg.matrix_rank(jacobian))
    full_rank_gate = jacobian_rank == 8 and float(singular_values.min()) > 0.0

    candidate.eval()
    candidate.record_diagnostics = True
    gaussian = torch.randn(256, width)
    gaussian = gaussian / rms(gaussian).float()[:, None]
    adversarial = torch.zeros(256, width)
    adversarial[:, 0] = math.sqrt(width)
    inputs = {"gaussian": gaussian, "one_hot_rms_one": adversarial}
    trajectories = {}
    trajectory_gates = []
    theoretical_bound = 1.0 + math.sqrt(depth) * trust_rms
    for name, values in inputs.items():
        output = candidate(values)
        rows = list(candidate.last_trust_diagnostics)
        maximum_hidden_rms = max(row["hidden_rms"] for row in rows)
        maximum_trusted_rms = max(row["trusted_delta_rms"] for row in rows)
        finite = bool(torch.isfinite(output).all())
        bounded = maximum_hidden_rms <= theoretical_bound + 1e-5
        trusted_bounded = maximum_trusted_rms < trust_rms + 1e-6
        trajectory_gates.append(finite and bounded and trusted_bounded)
        trajectories[name] = {
            "output_rms_max": float(rms(output).max()),
            "output_abs_max": float(output.abs().max()),
            "maximum_hidden_rms": maximum_hidden_rms,
            "maximum_trusted_delta_rms": maximum_trusted_rms,
            "theoretical_hidden_rms_bound": theoretical_bound,
            "finite": finite,
            "rows": rows,
        }

    ledger = exact_ledger()
    gates = {
        "same_parameter_count_as_v1": parameter_gate,
        "same_state_dict_keys_as_v1": state_dict_gate,
        "trust_region_strictly_bounds_rms": trust_bound_gate,
        "trust_region_jacobian_full_rank": full_rank_gate,
        "gaussian_and_adversarial_trajectories_finite_and_bounded": all(
            trajectory_gates
        ),
        "same_ideal_active_ratio_below_8p3_percent": ledger[
            "ideal_active_ratio"
        ]
        < 0.083,
    }
    payload = {
        "schema": "radial-trust-cayley-program-tree-stage0-v1",
        "source_sha256": sha256_file(Path(__file__)),
        "module_sha256": sha256_file(MODULE),
        "test_sha256": sha256_file(TEST),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "seed": args.seed,
        "ledger": ledger,
        "measurements": {
            "v1_parameters": base.parameter_count(),
            "candidate_parameters": candidate.parameter_count(),
            "random_trust_output_rms_max": random_trust_max,
            "trust_jacobian_rank": jacobian_rank,
            "trust_jacobian_singular_min": float(singular_values.min()),
            "trust_jacobian_singular_max": float(singular_values.max()),
            "trajectories": trajectories,
        },
        "gates": gates,
        "pass": all(gates.values()),
    }
    write_payload(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
