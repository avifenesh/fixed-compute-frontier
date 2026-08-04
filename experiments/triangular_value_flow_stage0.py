#!/usr/bin/env python3
"""Algebra and synthetic-capacity screen for triangular value flow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def flow(values: torch.Tensor, output_weight: torch.Tensor, tau: float) -> torch.Tensor:
    coefficient = torch.tril(output_weight, diagonal=-1) / tau
    feature = values * values.abs()
    return values + feature @ coefficient.T


def jacobian_rank(candidate: bool, seed: int = 311) -> int:
    generator = torch.Generator().manual_seed(seed)
    width = 3
    inputs = torch.randn(10, width, dtype=torch.float64, generator=generator)
    value = torch.randn(width, width, dtype=torch.float64, generator=generator)
    output = torch.triu(
        torch.randn(width, width, dtype=torch.float64, generator=generator)
    )
    output.diagonal().add_(1.0)
    parameters = torch.cat((value.reshape(-1), output.reshape(-1))).requires_grad_()

    def evaluate(flattened: torch.Tensor) -> torch.Tensor:
        current_value = flattened[:width * width].reshape(width, width)
        current_output = flattened[width * width:].reshape(width, width)
        hidden = inputs @ current_value.T
        if candidate:
            hidden = flow(hidden, current_output, 0.125)
        return (hidden @ current_output.T).reshape(-1)

    jacobian = torch.autograd.functional.jacobian(evaluate, parameters)
    singular_values = torch.linalg.svdvals(jacobian)
    tolerance = singular_values.max() * 1e-9
    return int((singular_values > tolerance).sum())


def teacher_screen(
    steps: int,
    seed: int,
    device: torch.device,
) -> dict[str, float | int | bool]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    width = 8
    tau = 0.125
    train_count = 4096
    test_count = 2048
    generator = torch.Generator(device=device).manual_seed(seed)
    train = 0.6 * torch.randn(
        train_count, width, device=device, generator=generator
    )
    test = 0.6 * torch.randn(
        test_count, width, device=device, generator=generator
    )
    teacher = torch.triu(
        0.25 * torch.randn(width, width, device=device, generator=generator)
    )
    teacher.diagonal().add_(0.8)
    teacher = teacher + torch.tril(
        0.025 * torch.randn(width, width, device=device, generator=generator),
        diagonal=-1,
    )

    def target(inputs: torch.Tensor) -> torch.Tensor:
        return flow(inputs, teacher, tau) @ teacher.T

    train_target = target(train)
    test_target = target(test)
    # Best unconstrained linear control, solved in closed form.
    linear = torch.linalg.lstsq(train, train_target).solution
    linear_test = test @ linear
    control_mse = float((linear_test - test_target).square().mean())

    candidate = torch.nn.Parameter(torch.triu(teacher.detach().clone()))
    optimizer = torch.optim.AdamW([candidate], lr=3e-3, weight_decay=0.0)
    for _ in range(steps):
        prediction = flow(train, candidate, tau) @ candidate.T
        loss = (prediction - train_target).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        candidate_test = flow(test, candidate, tau) @ candidate.T
        candidate_mse = float((candidate_test - test_target).square().mean())
        coefficient_abs_max = float(
            (torch.tril(candidate, diagonal=-1) / tau).abs().max()
        )
    return {
        "width": width,
        "learned_scalars_each": width * width,
        "control_test_mse": control_mse,
        "candidate_test_mse": candidate_mse,
        "control_over_candidate_mse": control_mse / candidate_mse,
        "candidate_coefficient_abs_max": coefficient_abs_max,
        "candidate_mse_at_most_1e_6": candidate_mse <= 1e-6,
        "candidate_at_least_100x_better": control_mse / candidate_mse >= 100.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=313)
    parser.add_argument(
        "--output", type=Path, default=Path("results/triangular-value-flow-stage0.json")
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ordinary_rank = jacobian_rank(False)
    candidate_rank = jacobian_rank(True)
    teacher = teacher_screen(args.steps, args.seed, device)
    gates = {
        "ordinary_rank_is_9": ordinary_rank == 9,
        "candidate_rank_is_12": candidate_rank == 12,
        "rank_gain_matches_strict_lower_dimension": candidate_rank - ordinary_rank == 3,
        "equal_teacher_parameter_counts": teacher["learned_scalars_each"] == 64,
        "candidate_fits_teacher": teacher["candidate_mse_at_most_1e_6"],
        "candidate_beats_linear_by_100x": teacher["candidate_at_least_100x_better"],
    }
    payload = {
        "schema": "triangular-value-flow-stage0-development-v1",
        "device": str(device),
        "jacobian": {
            "ordinary_rank": ordinary_rank,
            "candidate_rank": candidate_rank,
            "expected_gain": 3,
        },
        "teacher": teacher,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
