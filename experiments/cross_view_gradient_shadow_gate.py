#!/usr/bin/env python3
"""Causal shadow gate for norm-preserved cross-view gradient filtering.

Two independent microbatch gradients expose which matrix rows/columns agree
across samples.  Only past pairs build the filter.  The filter is scored on the
next pair without changing the model update; the model itself follows ordinary
AdamW on the mean gradient.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers

from experiments.folded_metric_lm_screen import (
    build_plain_model,
    causal_loss,
    configure_matmul_precision,
    make_optimizer,
    set_schedule,
    targeted_weights,
)
from experiments.reflex_swiglu_lm_screen import (
    TokenFile,
    lr_multiplier,
    validate_data_ledger,
    write_payload,
)


SOURCE = Path(__file__)
PREREGISTRATION = Path(
    "results/cross-view-gradient-shadow-gate-preregistration.md"
)
FAMILIES = ("q", "k", "v", "o", "gate", "up", "down")
AUTO_FILTERS = ("auto_inv_sqrt", "auto_inv", "auto_sqrt", "auto_direct")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def axis_energy(matrix: torch.Tensor, rows: bool) -> torch.Tensor:
    return matrix.square().mean(dim=1 if rows else 0)


def axis_cross(first: torch.Tensor, second: torch.Tensor, rows: bool) -> torch.Tensor:
    return (first * second).mean(dim=1 if rows else 0)


def apply_axis_weights(
    direction: torch.Tensor, weights: torch.Tensor, rows: bool
) -> torch.Tensor:
    shaped = weights[:, None] if rows else weights[None, :]
    filtered = direction * shaped
    source_norm = torch.linalg.vector_norm(direction.float())
    filtered_norm = torch.linalg.vector_norm(filtered.float())
    if not torch.isfinite(filtered_norm) or float(filtered_norm) <= 1e-30:
        return direction.clone()
    return filtered * (source_norm / filtered_norm).to(filtered.dtype)


def cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    a = first.float().reshape(-1)
    b = second.float().reshape(-1)
    denominator = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denominator) <= 1e-30:
        return 0.0
    return float(torch.dot(a, b) / denominator)


def symmetric_prediction_nmse(
    first: torch.Tensor,
    second: torch.Tensor,
    weights: torch.Tensor,
    rows: bool,
) -> float:
    shaped = weights[:, None] if rows else weights[None, :]
    numerator = 0.5 * (
        (shaped * first.float() - second.float()).square().mean()
        + (shaped * second.float() - first.float()).square().mean()
    )
    denominator = 0.5 * (
        first.float().square().mean() + second.float().square().mean()
    )
    return float(numerator / denominator.clamp_min(1e-30))


class DiagonalCrossViewState:
    def __init__(
        self,
        shape: tuple[int, int],
        seed: int,
        device: torch.device,
        beta: float = 0.95,
    ) -> None:
        output, input_ = shape
        self.rows = output <= input_
        dimension = output if self.rows else input_
        self.energy = torch.zeros(dimension, device=device, dtype=torch.float32)
        self.cross = torch.zeros_like(self.energy)
        self.shuffled_cross = torch.zeros_like(self.energy)
        self.residual_second = torch.zeros_like(self.energy)
        self.shuffled_residual_second = torch.zeros_like(self.energy)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        self.permutation = torch.randperm(dimension, generator=generator).to(device)
        self.beta = beta
        self.updates = 0

    @property
    def scalars(self) -> int:
        return self.energy.numel() * 5

    def weights(self) -> dict[str, torch.Tensor]:
        epsilon = max(float(self.energy.mean()) * 1e-12, 1e-30)
        energy = self.energy.clamp_min(epsilon)
        scalar = (self.cross.sum() / energy.sum()).clamp(0.0, 1.0)
        shuffled_scalar = (self.shuffled_cross.sum() / energy.sum()).clamp(0.0, 1.0)
        raw_cross = self.cross / energy
        raw_shuffled = self.shuffled_cross / energy
        effective_samples = ((1.0 + self.beta) / (1.0 - self.beta)) * (
            (1.0 - self.beta**max(self.updates, 1))
            / (1.0 + self.beta**max(self.updates, 1))
        )

        def shrink(
            raw: torch.Tensor,
            center: torch.Tensor,
            residual_second: torch.Tensor,
            cross_moment: torch.Tensor,
        ) -> torch.Tensor:
            difference = cross_moment - center * self.energy
            variance = (residual_second - difference.square()).clamp_min(0.0)
            standard_error_squared = variance / max(effective_samples, 1.0)
            shrinkage = difference.square() / (
                difference.square() + standard_error_squared + epsilon**2
            )
            if self.updates < 40:
                shrinkage.zero_()
            return (center + shrinkage * (raw - center)).clamp(0.0, 1.0)

        cross = shrink(raw_cross, scalar, self.residual_second, self.cross)
        shuffled = shrink(
            raw_shuffled,
            shuffled_scalar,
            self.shuffled_residual_second,
            self.shuffled_cross,
        )
        return {
            "identity": torch.ones_like(energy),
            "scalar": torch.full_like(energy, float(scalar)),
            "cross": cross,
            "shuffled_cross": shuffled,
            "auto_inv_sqrt": torch.rsqrt(energy),
            "auto_inv": energy.reciprocal(),
            "auto_sqrt": energy.sqrt(),
            "auto_direct": energy,
        }

    @torch.no_grad()
    def update(self, first: torch.Tensor, second: torch.Tensor, beta: float) -> None:
        if beta != self.beta:
            raise ValueError(f"state beta {self.beta} != update beta {beta}")
        first = first.float()
        second = second.float()
        energy = 0.5 * (
            axis_energy(first, self.rows) + axis_energy(second, self.rows)
        )
        cross = axis_cross(first, second, self.rows)
        if self.rows:
            shuffled = 0.5 * (
                axis_cross(first, second[self.permutation], True)
                + axis_cross(second, first[self.permutation], True)
            )
        else:
            shuffled = 0.5 * (
                axis_cross(first, second[:, self.permutation], False)
                + axis_cross(second, first[:, self.permutation], False)
            )
        epsilon = max(float(self.energy.mean()) * 1e-12, 1e-30)
        scalar = (
            self.cross.sum() / self.energy.clamp_min(epsilon).sum()
            if self.updates
            else torch.tensor(0.0, device=first.device)
        ).clamp(0.0, 1.0)
        shuffled_scalar = (
            self.shuffled_cross.sum() / self.energy.clamp_min(epsilon).sum()
            if self.updates
            else torch.tensor(0.0, device=first.device)
        ).clamp(0.0, 1.0)
        residual = cross - scalar * energy
        shuffled_residual = shuffled - shuffled_scalar * energy
        self.energy.mul_(beta).add_(energy, alpha=1.0 - beta)
        self.cross.mul_(beta).add_(cross, alpha=1.0 - beta)
        self.shuffled_cross.mul_(beta).add_(shuffled, alpha=1.0 - beta)
        self.residual_second.mul_(beta).addcmul_(
            residual, residual, value=1.0 - beta
        )
        self.shuffled_residual_second.mul_(beta).addcmul_(
            shuffled_residual, shuffled_residual, value=1.0 - beta
        )
        self.updates += 1

    def macro_weights(self, half_weights: torch.Tensor) -> torch.Tensor:
        """Map half-gradient reliability to reliability of their mean."""
        return 2.0 * half_weights / (1.0 + half_weights)


def hypothetical_adam_direction(
    gradient: torch.Tensor,
    optimizer_state: dict[str, Any],
    step: int,
    beta1: float = 0.9,
    beta2: float = 0.95,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    gradient = gradient.float()
    previous_first = optimizer_state.get("exp_avg")
    previous_second = optimizer_state.get("exp_avg_sq")
    if previous_first is None:
        previous_first = torch.zeros_like(gradient)
        previous_second = torch.zeros_like(gradient)
    first = beta1 * previous_first.float() + (1.0 - beta1) * gradient
    second = beta2 * previous_second.float() + (1.0 - beta2) * gradient.square()
    first = first / (1.0 - beta1**step)
    second = second / (1.0 - beta2**step)
    return first / (second.sqrt() + epsilon)


def paired_interval(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    standard_error = float(array.std(ddof=1) / math.sqrt(len(array)))
    return {
        "mean": mean,
        "standard_error": standard_error,
        "lower_95": mean - 1.96 * standard_error,
        "upper_95": mean + 1.96 * standard_error,
    }


def moving_block_interval(
    values: list[float], block_size: int, seed: int = 20_260_802, trials: int = 5_000
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    blocks_needed = math.ceil(len(array) / block_size)
    draws = np.empty(trials, dtype=np.float64)
    offsets = np.arange(block_size)
    for trial in range(trials):
        starts = generator.integers(0, len(array), size=blocks_needed)
        indices = (starts[:, None] + offsets[None, :]).reshape(-1)[: len(array)]
        draws[trial] = array[indices % len(array)].mean()
    return {
        "mean": float(array.mean()),
        "lower_95": float(np.quantile(draws, 0.025)),
        "upper_95": float(np.quantile(draws, 0.975)),
        "block_size": block_size,
        "bootstrap_trials": trials,
    }


def summarize(step_records: list[dict[str, Any]], block_size: int) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, float]]] = defaultdict(list)
    for step in step_records:
        for family, scores in step["families"].items():
            by_family[family].append(scores)

    families: dict[str, Any] = {}
    for family in FAMILIES:
        records = by_family[family]
        means = {
            name: float(np.mean([record[name] for record in records]))
            for name in records[0]
        }
        means["update_angle"] = float(
            np.median([record["update_angle"] for record in records])
        )
        means["max_norm_error"] = max(
            record["max_norm_error"] for record in records
        )
        best_auto_name = max(
            AUTO_FILTERS, key=lambda name: means[f"cos_{name}"]
        )
        mse_reduction = (
            means["nmse_scalar"] - means["nmse_cross"]
        ) / max(means["nmse_scalar"], 1e-30)
        families[family] = {
            "mean_cosines": means,
            "best_auto": best_auto_name,
            "relative_mse_reduction_vs_scalar": mse_reduction,
            "cross_minus_identity": means["cos_cross"] - means["cos_identity"],
            "cross_minus_best_auto": means["cos_cross"]
            - means[f"cos_{best_auto_name}"],
            "cross_minus_shuffled": means["cos_cross"]
            - means["cos_shuffled_cross"],
        }

    cross_minus_best, cross_minus_identity, mse_difference = [], [], []
    for step in step_records:
        per_family_best = []
        per_family_identity = []
        for family in FAMILIES:
            scores = step["families"][family]
            best_name = families[family]["best_auto"]
            per_family_best.append(
                scores["cos_cross"] - scores[f"cos_{best_name}"]
            )
            per_family_identity.append(
                scores["cos_cross"] - scores["cos_identity"]
            )
        cross_minus_best.append(float(np.mean(per_family_best)))
        cross_minus_identity.append(float(np.mean(per_family_identity)))
        mse_difference.append(
            float(
                np.mean(
                    [
                        step["families"][family]["nmse_cross"]
                        - step["families"][family]["nmse_scalar"]
                        for family in FAMILIES
                    ]
                )
            )
        )

    best_interval = moving_block_interval(cross_minus_best, block_size)
    identity_interval = moving_block_interval(
        cross_minus_identity, block_size, seed=20_260_803
    )
    mse_interval = moving_block_interval(
        mse_difference, block_size, seed=20_260_804
    )
    wins_identity = sum(
        value["cross_minus_identity"] > 0.0 for value in families.values()
    )
    wins_best_auto = sum(
        value["cross_minus_best_auto"] > 0.0 for value in families.values()
    )
    wins_shuffled = sum(
        value["cross_minus_shuffled"] > 0.0 for value in families.values()
    )
    mse_family_wins = sum(
        value["relative_mse_reduction_vs_scalar"] > 0.0
        for value in families.values()
    )
    worst_family_mse_reduction = min(
        value["relative_mse_reduction_vs_scalar"] for value in families.values()
    )
    mean_identity = float(
        np.mean([value["mean_cosines"]["cos_identity"] for value in families.values()])
    )
    mean_cross = float(
        np.mean([value["mean_cosines"]["cos_cross"] for value in families.values()])
    )
    mean_scalar_mse = float(
        np.mean([value["mean_cosines"]["nmse_scalar"] for value in families.values()])
    )
    mean_cross_mse = float(
        np.mean([value["mean_cosines"]["nmse_cross"] for value in families.values()])
    )
    relative_mse_reduction = (mean_scalar_mse - mean_cross_mse) / max(
        mean_scalar_mse, 1e-30
    )
    angular_error_reduction = (mean_cross - mean_identity) / max(
        1.0 - mean_identity, 1e-12
    )
    geometry_families = sum(
        value["mean_cosines"]["update_angle"] >= 1e-4
        for value in families.values()
    )
    max_norm_error = max(
        value["mean_cosines"]["max_norm_error"] for value in families.values()
    )
    max_clipped_ratio_fraction = max(
        value["mean_cosines"]["raw_ratio_clipped_fraction"]
        for value in families.values()
    )
    max_fallback_fraction = max(
        value["mean_cosines"]["fallback_fraction"]
        for value in families.values()
    )
    scores_finite = all(
        math.isfinite(float(score))
        for step in step_records
        for family in step["families"].values()
        for score in family.values()
    )
    gates = {
        "heldout_mse_reduced_by_0p5_percent": relative_mse_reduction >= 0.005,
        "mse_improves_in_5_of_7_families": mse_family_wins >= 5,
        "no_family_mse_worse_by_over_0p5_percent": worst_family_mse_reduction >= -0.005,
        "blocked_mse_interval_favors_cross": mse_interval["upper_95"] < 0.0,
        "cross_beats_identity_in_5_of_7_families": wins_identity >= 5,
        "cross_beats_best_auto_in_5_of_7_families": wins_best_auto >= 5,
        "cross_beats_shuffled_in_6_of_7_families": wins_shuffled >= 6,
        "blocked_interval_beats_identity": identity_interval["lower_95"] > 0.0,
        "blocked_interval_beats_best_auto": best_interval["lower_95"] > 0.0,
        "absolute_symmetric_cosine_gain_0p002": mean_cross - mean_identity >= 0.002,
        "nontrivial_geometry_in_5_of_7_families": geometry_families >= 5,
        "maximum_norm_error_at_most_1e_6": max_norm_error <= 1e-6,
        "raw_ratio_clipping_at_most_25_percent": max_clipped_ratio_fraction <= 0.25,
        "fallback_rate_at_most_1_percent": max_fallback_fraction <= 0.01,
        "all_scores_finite": scores_finite,
    }
    return {
        "families": families,
        "wins": {
            "identity": wins_identity,
            "best_auto": wins_best_auto,
            "shuffled": wins_shuffled,
            "mse": mse_family_wins,
            "nontrivial_geometry": geometry_families,
        },
        "global_mean_identity_cosine": mean_identity,
        "global_mean_cross_cosine": mean_cross,
        "relative_identity_angular_error_reduction": angular_error_reduction,
        "global_relative_mse_reduction_vs_scalar": relative_mse_reduction,
        "worst_family_relative_mse_reduction": worst_family_mse_reduction,
        "max_norm_error": max_norm_error,
        "max_raw_ratio_clipped_fraction": max_clipped_ratio_fraction,
        "max_fallback_fraction": max_fallback_fraction,
        "blocked_mse_cross_minus_scalar": mse_interval,
        "blocked_cross_minus_identity": identity_interval,
        "blocked_cross_minus_best_auto": best_interval,
        "gates": gates,
        "advance_to_lm_screen": all(gates.values()),
    }


def validate_protocol(args: argparse.Namespace) -> dict[str, Any]:
    expected = {
        "device_contains": "H100",
        "torch_version": "2.5.1+cu124",
        "cuda_version": "12.4",
        "transformers_version": "4.57.6",
        "sequence_length": 512,
        "micro_batch_size": 32,
        "steps": 256,
        "score_after": 40,
        "schedule_steps": 1525,
        "warmup_steps": 100,
        "learning_rate": 3e-4,
        "weight_decay": 0.1,
        "gradient_clip": 1.0,
        "ema_beta": 0.95,
        "block_size": 20,
        "seed": 223,
        "filter_seed": 20_260_801,
        "view_b_batch_offset": 1525,
    }
    actual = {
        "device_contains": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "transformers_version": transformers.__version__,
        **{key: getattr(args, key) for key in expected if key != "device_contains" and not key.endswith("version")},
    }
    checks = {
        key: (
            expected_value in actual[key]
            if key == "device_contains"
            else actual[key] == expected_value
        )
        for key, expected_value in expected.items()
    }
    if not all(checks.values()):
        failed = {
            key: {"expected": expected[key], "actual": actual[key]}
            for key, valid in checks.items()
            if not valid
        }
        raise ValueError(f"invalid protocol: {json.dumps(failed, sort_keys=True)}")
    return {"valid": True, "checks": checks, "expected": expected, "actual": actual}


def validate_view_separation(train_file: TokenFile, args: argparse.Namespace) -> dict[str, Any]:
    width = args.sequence_length + 1
    first_start_sequence = 0
    first_end_sequence = args.steps * args.micro_batch_size
    second_start_sequence = args.view_b_batch_offset * args.micro_batch_size
    second_end_sequence = second_start_sequence + args.steps * args.micro_batch_size
    if first_end_sequence > second_start_sequence:
        raise ValueError("gradient-view sequence ranges overlap")
    if second_end_sequence > train_file.sequence_count:
        raise ValueError("second gradient-view range exceeds training data")
    gap = train_file.tokens[
        first_end_sequence * width : second_start_sequence * width
    ]
    eos_boundaries_in_gap = int(np.count_nonzero(gap == 0))
    if eos_boundaries_in_gap < 1:
        raise ValueError("no EOS document boundary separates gradient views")
    return {
        "first_sequence_range": [first_start_sequence, first_end_sequence],
        "second_sequence_range": [second_start_sequence, second_end_sequence],
        "gap_sequences": second_start_sequence - first_end_sequence,
        "eos_boundaries_in_gap": eos_boundaries_in_gap,
        "document_disjoint_by_inserted_eos_boundary": True,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    precision = configure_matmul_precision()
    protocol = validate_protocol(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    data_ledger = validate_data_ledger(
        args.data_manifest,
        args.train_file,
        args.validation_file,
        args.sequence_length,
    )
    model = build_plain_model(device)
    targets = targeted_weights(model)
    optimizer = make_optimizer(
        model, targets, args.learning_rate, args.weight_decay
    )
    states = {
        name: DiagonalCrossViewState(
            tuple(parameter.shape),
            args.filter_seed + index,
            device,
            beta=args.ema_beta,
        )
        for index, (name, parameter) in enumerate(targets.items())
    }
    train_file = TokenFile(args.train_file, args.sequence_length)
    view_separation = validate_view_separation(train_file, args)
    parameter_list = list(model.parameters())
    target_by_id = {id(parameter): name for name, parameter in targets.items()}
    step_records: list[dict[str, Any]] = []
    losses, gradient_norms, step_seconds = [], [], []
    torch.cuda.reset_peak_memory_stats()
    for step_index in range(args.steps):
        started = time.perf_counter()
        model.train()
        set_schedule(
            optimizer,
            lr_multiplier(step_index, args.schedule_steps, args.warmup_steps),
        )
        optimizer.zero_grad(set_to_none=True)
        first_inputs, first_labels = train_file.batch(
            step_index, args.micro_batch_size, device
        )
        first_loss = causal_loss(model, first_inputs, first_labels)
        first_loss.backward()
        first_gradients = {
            id(parameter): parameter.grad.detach().clone()
            for parameter in parameter_list
            if parameter.grad is not None
        }
        optimizer.zero_grad(set_to_none=True)
        second_inputs, second_labels = train_file.batch(
            args.view_b_batch_offset + step_index,
            args.micro_batch_size,
            device,
        )
        second_loss = causal_loss(model, second_inputs, second_labels)
        second_loss.backward()
        second_target_gradients = {
            target_by_id[id(parameter)]: parameter.grad.detach().clone()
            for parameter in targets.values()
        }
        for parameter in parameter_list:
            first = first_gradients.get(id(parameter))
            if first is None:
                continue
            if parameter.grad is None:
                parameter.grad = first.mul(0.5)
            else:
                parameter.grad.add_(first).mul_(0.5)
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        )
        if step_index >= args.score_after:
            matrix_scores: dict[str, dict[str, float]] = {}
            for name, parameter in targets.items():
                first = first_gradients[id(parameter)].float()
                second = second_target_gradients[name].float()
                state = states[name]
                weights = state.weights()
                optimizer_state = optimizer.state.get(parameter, {})
                direction_first = hypothetical_adam_direction(
                    first, optimizer_state, step_index + 1
                )
                direction_second = hypothetical_adam_direction(
                    second, optimizer_state, step_index + 1
                )
                scores: dict[str, float] = {
                    "nmse_scalar": symmetric_prediction_nmse(
                        first, second, weights["scalar"], state.rows
                    ),
                    "nmse_cross": symmetric_prediction_nmse(
                        first, second, weights["cross"], state.rows
                    ),
                    "nmse_shuffled": symmetric_prediction_nmse(
                        first, second, weights["shuffled_cross"], state.rows
                    ),
                }
                for filter_name, filter_weights in weights.items():
                    filtered_first = apply_axis_weights(
                        direction_first, filter_weights, state.rows
                    )
                    filtered_second = apply_axis_weights(
                        direction_second, filter_weights, state.rows
                    )
                    scores[f"cos_{filter_name}"] = 0.5 * (
                        cosine(filtered_first, second)
                        + cosine(filtered_second, first)
                    )
                average_gradient = parameter.grad.detach().float()
                average_direction = hypothetical_adam_direction(
                    average_gradient, optimizer_state, step_index + 1
                )
                macro_weights = state.macro_weights(weights["cross"])
                shaped_macro_weights = (
                    macro_weights[:, None] if state.rows else macro_weights[None, :]
                )
                fallback = float(
                    torch.linalg.vector_norm(
                        (average_direction * shaped_macro_weights).float()
                    )
                    <= 1e-30
                )
                filtered_average = apply_axis_weights(
                    average_direction, macro_weights, state.rows
                )
                scores["update_angle"] = 1.0 - cosine(
                    filtered_average, average_direction
                )
                scores["max_norm_error"] = abs(
                    float(torch.linalg.vector_norm(filtered_average.float()))
                    - float(torch.linalg.vector_norm(average_direction.float()))
                ) / max(
                    float(torch.linalg.vector_norm(average_direction.float())),
                    1e-30,
                )
                energy = state.energy.clamp_min(
                    max(float(state.energy.mean()) * 1e-12, 1e-30)
                )
                raw_ratio = state.cross / energy
                scores["raw_ratio_clipped_fraction"] = float(
                    ((raw_ratio < 0.0) | (raw_ratio > 1.0)).float().mean()
                )
                scores["fallback_fraction"] = fallback
                matrix_scores[name] = scores
            families: dict[str, dict[str, float]] = {}
            for family in FAMILIES:
                members = [
                    scores
                    for name, scores in matrix_scores.items()
                    if name.endswith(f".{family}")
                ]
                families[family] = {}
                for key in members[0]:
                    if key == "max_norm_error":
                        reducer = max
                    elif key == "update_angle":
                        reducer = np.median
                    else:
                        reducer = np.mean
                    families[family][key] = float(
                        reducer([member[key] for member in members])
                    )
            step_records.append({"step": step_index + 1, "families": families})
        for name, parameter in targets.items():
            states[name].update(
                first_gradients[id(parameter)], second_target_gradients[name], args.ema_beta
            )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        losses.append(0.5 * (float(first_loss.detach()) + float(second_loss.detach())))
        gradient_norms.append(gradient_norm)
        step_seconds.append(time.perf_counter() - started)
        if (step_index + 1) % 40 == 0:
            print(
                json.dumps(
                    {
                        "step": step_index + 1,
                        "loss": losses[-1],
                        "gradient_norm": gradient_norm,
                        "scored_steps": len(step_records),
                    }
                ),
                flush=True,
            )
    training_finite = all(
        math.isfinite(value) for value in losses + gradient_norms
    )
    decision = summarize(step_records, args.block_size)
    decision["gates"]["all_training_finite"] = training_finite
    decision["advance_to_lm_screen"] = all(decision["gates"].values())
    payload = {
        "candidate": "norm-preserved diagonal cross-view gradient filter",
        "scope": "causal no-candidate-update shadow gate",
        "source_sha256": sha256_file(SOURCE),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "data_ledger": data_ledger,
        "device": torch.cuda.get_device_name(0),
        "precision": precision,
        "protocol": protocol,
        "view_separation": view_separation,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "optimizer_only_state_scalars": sum(state.scalars for state in states.values()),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train": {
            "mean_loss": float(np.mean(losses)),
            "final_loss": losses[-1],
            "max_gradient_norm": max(gradient_norms),
            "elapsed_seconds": sum(step_seconds),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "finite": training_finite,
        },
        "step_records": step_records,
        "decision": decision,
    }
    del optimizer, states, targets, model, first_gradients, second_target_gradients
    gc.collect()
    torch.cuda.empty_cache()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=Path("data/block-algebra-scratch/train.uint16.bin"))
    parser.add_argument("--validation-file", type=Path, default=Path("data/block-algebra-scratch/validation.uint16.bin"))
    parser.add_argument("--data-manifest", type=Path, default=Path("results/block-algebra-scratch-data-manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("results/cross-view-gradient-shadow-gate.json"))
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--score-after", type=int, default=40)
    parser.add_argument("--schedule-steps", type=int, default=1525)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--ema-beta", type=float, default=0.95)
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=223)
    parser.add_argument("--filter-seed", type=int, default=20_260_801)
    parser.add_argument("--view-b-batch-offset", type=int, default=1525)
    args = parser.parse_args()
    result = run(args)
    write_payload(args.output, result)
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
