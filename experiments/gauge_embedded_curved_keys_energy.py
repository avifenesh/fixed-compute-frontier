#!/usr/bin/env python3
"""Teacher-free squared-energy retrieval screen for gauge-funded curved keys."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = ROOT / "results" / "gauge-embedded-curved-keys-energy-preregistration.md"
TEST_SOURCE = ROOT / "tests" / "test_gauge_embedded_curved_keys_energy.py"
INTEGRITY_MANIFEST = ROOT / "results" / "gauge-embedded-curved-keys-energy-integrity-manifest.json"
OUTPUT = ROOT / "results" / "gauge-embedded-curved-keys-energy.json"
LEARNING_RESULT = ROOT / "results" / "gauge-embedded-curved-keys-learning.json"
LEARNING_SOURCE = ROOT / "experiments" / "gauge_embedded_curved_keys_learning.py"
LEARNING_PREREGISTRATION = ROOT / "results" / "gauge-embedded-curved-keys-learning-preregistration.md"
LEARNING_TEST = ROOT / "tests" / "test_gauge_embedded_curved_keys_learning.py"
LEARNING_INTEGRITY = ROOT / "results" / "gauge-embedded-curved-keys-learning-integrity-manifest.json"

ARMS = (
    "bilinear",
    "gauge_linear_control",
    "position_only_mean_square_control",
    "centered_rotation_only_geck",
    "centered_scale_only_geck",
    "full_scale_only_geck",
    "centered_geck_g2",
    "full_geck_g2",
)
DEVELOPMENT_SEED = 53
WORLD_SEEDS = (89, 127, 167, 211, 257)
HEAD_DIMENSION = 8
PAIRS = HEAD_DIMENSION // 2
CONTENT_DIMENSION = 2
QUERY_TYPES = 2
TRAIN_ITEMS = 4
TRAIN_SAMPLES = 8_192
VALIDATION_SAMPLES = 2_048
TEST_SAMPLES = 8_192
OOD_ITEMS = 8
OOD_SAMPLES = 8_192
ROPE_BASE = 10_000.0
TAU = 1.0 / math.sqrt(HEAD_DIMENSION)
RESTARTS = 4
STEPS = 4_000
BATCH_SIZE = 256
LEARNING_RATE = 3e-3
WARMUP_STEPS = 100
WEIGHT_DECAY = 1e-3
GRADIENT_CLIP = 1.0
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_CHUNK = 100


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rope_frequencies(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    pair_indices = torch.arange(PAIRS, device=device, dtype=dtype)
    return ROPE_BASE ** (-2.0 * pair_indices / HEAD_DIMENSION)


def balanced_energy_split(
    seed: int,
    samples: int,
    items: int,
    distribution: str = "gaussian",
) -> dict[str, np.ndarray]:
    balance_period = QUERY_TYPES * items * 2
    if samples % balance_period:
        raise ValueError(
            f"samples={samples} must be divisible by balance period {balance_period}"
        )
    rng = np.random.default_rng(seed)
    schedule = np.array([
        (query_type, target_slot, target_sign)
        for query_type in range(QUERY_TYPES)
        for target_slot in range(items)
        for target_sign in (-1, 1)
    ] * (samples // balance_period), dtype=np.int64)
    rng.shuffle(schedule)
    if distribution == "gaussian":
        unordered = rng.normal(size=(samples, items, CONTENT_DIMENSION))
    elif distribution == "laplace":
        unordered = rng.laplace(
            scale=1.0 / math.sqrt(2.0),
            size=(samples, items, CONTENT_DIMENSION),
        )
    else:
        raise ValueError(f"unknown distribution: {distribution}")

    keys = np.empty_like(unordered)
    targets = schedule[:, 1]
    query_types = schedule[:, 0]
    target_signs = schedule[:, 2]
    for sample in range(samples):
        query_type = int(query_types[sample])
        winner = int(np.argmax(np.abs(unordered[sample, :, query_type])))
        unordered[sample, winner, query_type] = (
            target_signs[sample]
            * abs(unordered[sample, winner, query_type])
        )
        remaining_records = [index for index in range(items) if index != winner]
        rng.shuffle(remaining_records)
        target_slot = int(targets[sample])
        remaining_slots = [index for index in range(items) if index != target_slot]
        keys[sample, target_slot] = unordered[sample, winner]
        keys[sample, remaining_slots] = unordered[sample, remaining_records]

    return {
        "keys": keys.astype(np.float32),
        "query_types": query_types,
        "targets": targets,
        "target_signs": target_signs,
    }


def split_balance_checks(split: dict[str, np.ndarray], items: int) -> dict[str, bool]:
    query_types = split["query_types"]
    targets = split["targets"]
    target_signs = split["target_signs"]
    keys = split["keys"]
    selected = keys[
        np.arange(keys.shape[0])[:, None],
        np.arange(items)[None, :],
        query_types[:, None],
    ]
    joint_counts = [
        np.sum(
            (query_types == query_type)
            & (targets == target_slot)
            & (target_signs == target_sign)
        )
        for query_type in range(QUERY_TYPES)
        for target_slot in range(items)
        for target_sign in (-1, 1)
    ]
    return {
        "query_type_exact": all(
            np.sum(query_types == query_type) == len(query_types) // QUERY_TYPES
            for query_type in range(QUERY_TYPES)
        ),
        "target_slot_exact": all(
            np.sum(targets == slot) == len(targets) // items
            for slot in range(items)
        ),
        "target_sign_exact": all(
            np.sum(target_signs == sign) == len(target_signs) // 2
            for sign in (-1, 1)
        ),
        "joint_query_type_target_slot_target_sign_exact": len(set(joint_counts)) == 1,
        "labels_follow_energy_rule": bool(np.array_equal(
            np.argmax(np.square(selected), axis=-1), targets
        )),
        "stored_target_signs_match": bool(np.array_equal(
            np.sign(selected[np.arange(len(targets)), targets]).astype(np.int64),
            target_signs,
        )),
    }


def make_splits(seed: int) -> dict[str, dict[str, np.ndarray]]:
    return {
        "train": balanced_energy_split(seed * 10 + 1, TRAIN_SAMPLES, TRAIN_ITEMS),
        "validation": balanced_energy_split(
            seed * 10 + 2, VALIDATION_SAMPLES, TRAIN_ITEMS
        ),
        "test": balanced_energy_split(seed * 10 + 3, TEST_SAMPLES, TRAIN_ITEMS),
        "ood_laplace_four_record": balanced_energy_split(
            seed * 10 + 4, OOD_SAMPLES, TRAIN_ITEMS, distribution="laplace"
        ),
        "ood_gaussian_eight_record": balanced_energy_split(
            seed * 10 + 5, OOD_SAMPLES, OOD_ITEMS, distribution="gaussian"
        ),
        "ood_laplace_eight_record": balanced_energy_split(
            seed * 10 + 6, OOD_SAMPLES, OOD_ITEMS, distribution="laplace"
        ),
    }


def make_initial_parameters(
    seed: int,
    restarts: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    query = torch.randn(
        restarts, PAIRS, 2, QUERY_TYPES, generator=generator, dtype=torch.float32
    ) / math.sqrt(CONTENT_DIMENSION)
    key = torch.randn(
        restarts, PAIRS, 2, CONTENT_DIMENSION, generator=generator, dtype=torch.float32
    ) / math.sqrt(CONTENT_DIMENSION)
    polar = torch.zeros(restarts, PAIRS, 2, dtype=torch.float32)
    return query.to(device), key.to(device), polar.to(device)


def canonicalize(
    query: torch.Tensor,
    key: torch.Tensor,
    polar: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    phi = polar[..., 0]
    log_scale = polar[..., 1]
    cosine = torch.cos(phi)
    sine = torch.sin(phi)
    rotation = torch.stack((
        torch.stack((cosine, -sine), dim=-1),
        torch.stack((sine, cosine), dim=-1),
    ), dim=-2)
    canonical_key = (
        torch.exp(-log_scale)[..., None, None]
        * torch.einsum("...ij,...jk->...ik", rotation, key)
    )
    canonical_query = (
        torch.exp(log_scale)[..., None, None]
        * torch.einsum("...ij,...jk->...ik", rotation, query)
    )
    return canonical_query, canonical_key, torch.sin(phi), torch.tanh(log_scale)


def physical_serialized_parameters(
    query: torch.Tensor,
    key: torch.Tensor,
    polar: torch.Tensor,
) -> torch.Tensor:
    phi = polar[..., 0]
    log_scale = polar[..., 1]
    radius = TAU * torch.exp(log_scale)
    pivot = torch.stack((radius * torch.sin(phi), radius * torch.cos(phi)), dim=-1)
    physical_key = torch.cat((key, pivot[..., None]), dim=-1)
    return torch.cat((query.reshape(*query.shape[:-3], -1), physical_key.reshape(*key.shape[:-3], -1)), dim=-1)


def attention_logits(
    query: torch.Tensor,
    key: torch.Tensor,
    polar: torch.Tensor,
    query_types: torch.Tensor,
    records: torch.Tensor,
    arm: str,
    *,
    quadratic_permutation: torch.Tensor | None = None,
    ablate_rotation: bool = False,
    ablate_scale: bool = False,
) -> torch.Tensor:
    """Return logits shaped [restart, batch, item]."""
    canonical_query, canonical_key, rotation_coefficient, scale_coefficient = canonicalize(
        query, key, polar
    )
    if ablate_rotation:
        rotation_coefficient = torch.zeros_like(rotation_coefficient)
    if ablate_scale:
        scale_coefficient = torch.zeros_like(scale_coefficient)
    query_one_hot = F.one_hot(query_types, QUERY_TYPES).to(records.dtype)
    query_vectors = torch.einsum(
        "rpot,bt->rbpo", canonical_query, query_one_hot
    )
    raw_keys = torch.einsum("rpoc,bic->rbipo", canonical_key, records)
    even = raw_keys[..., 0]
    odd = raw_keys[..., 1]
    square_even = even * even
    square_odd = odd * odd
    if quadratic_permutation is not None:
        gather_index = quadratic_permutation[None, :, :, None].expand(
            even.shape[0], -1, -1, PAIRS
        )
        square_even = torch.gather(square_even, dim=2, index=gather_index)
        square_odd = torch.gather(square_odd, dim=2, index=gather_index)
    mean_square = torch.sum(canonical_key * canonical_key, dim=-1)
    mean_even = mean_square[..., 0][:, None, None, :]
    mean_odd = mean_square[..., 1][:, None, None, :]
    rotation_coefficient = rotation_coefficient[:, None, None, :]
    scale_coefficient = scale_coefficient[:, None, None, :]

    if arm == "bilinear":
        transformed_even = even
        transformed_odd = odd
    elif arm == "gauge_linear_control":
        transformed_even = even + rotation_coefficient * odd
        transformed_odd = odd + scale_coefficient * even
    elif arm == "position_only_mean_square_control":
        transformed_even = even + rotation_coefficient * mean_odd
        transformed_odd = odd + scale_coefficient * mean_even
    elif arm == "centered_rotation_only_geck":
        transformed_even = even + rotation_coefficient * (square_odd - mean_odd)
        transformed_odd = odd
    elif arm == "centered_scale_only_geck":
        transformed_even = even
        transformed_odd = odd + scale_coefficient * (square_even - mean_even)
    elif arm == "full_scale_only_geck":
        transformed_even = even
        transformed_odd = odd + scale_coefficient * square_even
    elif arm == "centered_geck_g2":
        transformed_even = even + rotation_coefficient * (square_odd - mean_odd)
        transformed_odd = odd + scale_coefficient * (square_even - mean_even)
    elif arm == "full_geck_g2":
        transformed_even = even + rotation_coefficient * square_odd
        transformed_odd = odd + scale_coefficient * square_even
    else:
        raise ValueError(f"unknown arm: {arm}")

    items = records.shape[1]
    frequencies = rope_frequencies(records.device, records.dtype)
    relative_positions = (
        torch.arange(items, device=records.device, dtype=records.dtype) - items
    )
    angles = relative_positions[:, None] * frequencies[None, :]
    cosine = torch.cos(angles)[None, None, :, :]
    sine = torch.sin(angles)[None, None, :, :]
    rotated_even = cosine * transformed_even - sine * transformed_odd
    rotated_odd = sine * transformed_even + cosine * transformed_odd
    return torch.sum(
        query_vectors[..., 0][:, :, None, :] * rotated_even
        + query_vectors[..., 1][:, :, None, :] * rotated_odd,
        dim=-1,
    )


def paired_initial_parameters(
    seed: int,
    device: torch.device,
) -> tuple[torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter]:
    query, key, polar = make_initial_parameters(seed, RESTARTS, device)
    arm_count = len(ARMS)
    return (
        torch.nn.Parameter(query[None].repeat(arm_count, 1, 1, 1, 1)),
        torch.nn.Parameter(key[None].repeat(arm_count, 1, 1, 1, 1)),
        torch.nn.Parameter(polar[None].repeat(arm_count, 1, 1, 1)),
    )


def per_run_clip_gradients(
    parameters: tuple[torch.nn.Parameter, ...], max_norm: float
) -> None:
    squared_norm = None
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        reduce_dimensions = tuple(range(2, gradient.ndim))
        contribution = torch.sum(gradient * gradient, dim=reduce_dimensions)
        squared_norm = contribution if squared_norm is None else squared_norm + contribution
    if squared_norm is None:
        return
    scale = torch.clamp(max_norm / (torch.sqrt(squared_norm) + 1e-12), max=1.0)
    for parameter in parameters:
        if parameter.grad is None:
            continue
        expansion = scale[(...,) + (None,) * (parameter.ndim - 2)]
        parameter.grad.mul_(expansion)


def learning_rate_multiplier(step: int, total_steps: int) -> float:
    if step < WARMUP_STEPS:
        return (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(total_steps - WARMUP_STEPS - 1, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def tensors_from_split(
    split: dict[str, np.ndarray], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(split["keys"]).to(device),
        torch.from_numpy(split["query_types"]).to(device),
        torch.from_numpy(split["targets"]).to(device),
    )


@torch.no_grad()
def evaluate_runs(
    query: torch.Tensor,
    key: torch.Tensor,
    polar: torch.Tensor,
    split: dict[str, np.ndarray],
    arm: str,
    device: torch.device,
    **forward_options: object,
) -> dict[str, torch.Tensor]:
    records, query_types, targets = tensors_from_split(split, device)
    logits = attention_logits(
        query, key, polar, query_types, records, arm, **forward_options
    )
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.repeat(logits.shape[0]),
        reduction="none",
    ).reshape(logits.shape[0], -1).mean(dim=-1)
    predictions = torch.argmax(logits, dim=-1)
    accuracy = torch.mean((predictions == targets[None]).float(), dim=-1)
    per_type_accuracy = torch.stack([
        torch.mean(
            (predictions[:, query_types == query_type]
             == targets[None, query_types == query_type]).float(),
            dim=-1,
        )
        for query_type in range(QUERY_TYPES)
    ], dim=-1)
    return {
        "nll": losses,
        "accuracy": accuracy,
        "per_type_accuracy": per_type_accuracy,
        "predictions": predictions,
        "logits": logits,
    }


@torch.no_grad()
def evaluate_chosen(
    chosen: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    split: dict[str, np.ndarray],
    arm: str,
    device: torch.device,
    **forward_options: object,
) -> dict[str, torch.Tensor]:
    return evaluate_runs(
        chosen[0][None], chosen[1][None], chosen[2][None],
        split, arm, device, **forward_options,
    )


def split_with(
    split: dict[str, np.ndarray],
    *,
    keys: np.ndarray | None = None,
    targets: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    changed = dict(split)
    if keys is not None:
        changed["keys"] = keys
    if targets is not None:
        changed["targets"] = targets
    return changed


def independent_sign_matrix(seed: int, shape: tuple[int, ...]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice((-1.0, 1.0), size=shape).astype(np.float32)


def quadratic_derangements(seed: int, samples: int, items: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([
        np.roll(np.arange(items), int(shift))
        for shift in rng.integers(1, items, size=samples)
    ]).astype(np.int64)


def selected_metrics(evaluation: dict[str, torch.Tensor]) -> dict[str, object]:
    return {
        "nll": float(evaluation["nll"][0].item()),
        "accuracy": float(evaluation["accuracy"][0].item()),
        "accuracy_by_query_type": evaluation["per_type_accuracy"][0].cpu().tolist(),
    }


def paired_bootstrap_primary_advantage(
    evaluations: dict[str, dict[str, torch.Tensor]],
    targets: np.ndarray,
    seed: int,
) -> dict[str, float | int]:
    primary_arm = "full_scale_only_geck"
    controls = (
        "bilinear",
        "gauge_linear_control",
        "position_only_mean_square_control",
    )
    primary_logits = evaluations[primary_arm]["logits"][0].cpu()
    control_logits = torch.stack([
        evaluations[arm]["logits"][0].cpu() for arm in controls
    ])
    target_tensor = torch.from_numpy(targets)
    primary_correct = (
        torch.argmax(primary_logits, dim=-1) == target_tensor
    ).numpy().astype(np.float64)
    control_correct = (
        torch.argmax(control_logits, dim=-1) == target_tensor[None]
    ).numpy().astype(np.float64)
    primary_loss = F.cross_entropy(
        primary_logits, target_tensor, reduction="none"
    ).numpy().astype(np.float64)
    control_loss = torch.stack([
        F.cross_entropy(logits, target_tensor, reduction="none")
        for logits in control_logits
    ]).numpy().astype(np.float64)
    rng = np.random.default_rng(seed)
    accuracy_advantages: list[np.ndarray] = []
    nll_advantages: list[np.ndarray] = []
    samples = len(targets)
    for start in range(0, BOOTSTRAP_REPLICATES, BOOTSTRAP_CHUNK):
        chunk = min(BOOTSTRAP_CHUNK, BOOTSTRAP_REPLICATES - start)
        indices = rng.integers(0, samples, size=(chunk, samples))
        primary_accuracy = np.mean(primary_correct[indices], axis=-1)
        control_accuracy = np.stack([
            np.mean(correct[indices], axis=-1) for correct in control_correct
        ], axis=-1)
        primary_nll = np.mean(primary_loss[indices], axis=-1)
        control_nll = np.stack([
            np.mean(loss[indices], axis=-1) for loss in control_loss
        ], axis=-1)
        accuracy_advantages.append(
            primary_accuracy - np.max(control_accuracy, axis=-1)
        )
        nll_advantages.append(
            np.min(control_nll, axis=-1) - primary_nll
        )
    accuracy_advantage = np.concatenate(accuracy_advantages)
    nll_advantage = np.concatenate(nll_advantages)
    return {
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": seed,
        "accuracy_advantage_lower_95_percent": float(np.quantile(
            accuracy_advantage, 0.025
        )),
        "accuracy_advantage_median": float(np.median(accuracy_advantage)),
        "nll_advantage_lower_95_percent": float(np.quantile(
            nll_advantage, 0.025
        )),
        "nll_advantage_median": float(np.median(nll_advantage)),
    }


@torch.no_grad()
def counterfactual_evaluations(
    selected: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    splits: dict[str, dict[str, np.ndarray]],
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    test = splits["test"]
    ordinary_and_candidates: dict[str, object] = {}
    base_predictions: dict[str, np.ndarray] = {}
    base_evaluations: dict[str, dict[str, torch.Tensor]] = {}
    for arm in ARMS:
        base = evaluate_chosen(selected[arm], test, arm, device)
        independent_signs = independent_sign_matrix(
            seed * 1_000_000 + 17, test["keys"].shape
        )
        sign_flipped = evaluate_chosen(
            selected[arm],
            split_with(test, keys=test["keys"] * independent_signs),
            arm,
            device,
        )
        base_prediction = base["predictions"][0].cpu().numpy()
        sign_prediction = sign_flipped["predictions"][0].cpu().numpy()
        base_predictions[arm] = base_prediction
        base_evaluations[arm] = base
        ordinary_and_candidates[arm] = {
            "base": selected_metrics(base),
            "sign_flipped": selected_metrics(sign_flipped),
            "sign_flip_prediction_agreement": float(np.mean(
                base_prediction == sign_prediction
            )),
            "ood_laplace_four_record": selected_metrics(evaluate_chosen(
                selected[arm], splits["ood_laplace_four_record"], arm, device
            )),
            "ood_gaussian_eight_record": selected_metrics(evaluate_chosen(
                selected[arm], splits["ood_gaussian_eight_record"], arm, device
            )),
            "ood_laplace_eight_record": selected_metrics(evaluate_chosen(
                selected[arm], splits["ood_laplace_eight_record"], arm, device
            )),
        }

    primary_arm = "full_scale_only_geck"
    primary = selected[primary_arm]
    quadratic_permutation = quadratic_derangements(
        seed * 1_000_000 + 31, TEST_SAMPLES, TRAIN_ITEMS
    )
    shuffled_quadratics = evaluate_chosen(
        primary,
        test,
        primary_arm,
        device,
        quadratic_permutation=torch.from_numpy(quadratic_permutation).to(device),
    )
    curvature_ablated = evaluate_chosen(
        primary, test, primary_arm, device, ablate_scale=True
    )

    base_primary_prediction = base_predictions[primary_arm]
    permutation_correct: list[np.ndarray] = []
    permutation_agreement: list[np.ndarray] = []
    target_slot_correct: list[list[np.ndarray]] = [[] for _ in range(TRAIN_ITEMS)]
    for permutation_tuple in itertools.permutations(range(TRAIN_ITEMS)):
        permutation = np.asarray(permutation_tuple, dtype=np.int64)
        inverse = np.empty_like(permutation)
        inverse[permutation] = np.arange(TRAIN_ITEMS)
        permuted_targets = inverse[test["targets"]]
        permuted = split_with(
            test,
            keys=test["keys"][:, permutation],
            targets=permuted_targets,
        )
        evaluation = evaluate_chosen(
            primary, permuted, primary_arm, device
        )
        predicted_slot = evaluation["predictions"][0].cpu().numpy()
        predicted_record = permutation[predicted_slot]
        correct = predicted_record == test["targets"]
        permutation_correct.append(correct)
        permutation_agreement.append(predicted_record == base_primary_prediction)
        for target_slot in range(TRAIN_ITEMS):
            mask = permuted_targets == target_slot
            target_slot_correct[target_slot].append(correct[mask])
    all_permutation_correct = np.concatenate(permutation_correct)
    all_permutation_agreement = np.concatenate(permutation_agreement)
    target_slot_accuracies = [
        float(np.mean(np.concatenate(values))) for values in target_slot_correct
    ]
    primary_specific = {
        "quadratic_content_shuffled": selected_metrics(shuffled_quadratics),
        "scale_curvature_ablated": selected_metrics(curvature_ablated),
        "all_24_position_permutations": {
            "accuracy": float(np.mean(all_permutation_correct)),
            "prediction_agreement_with_original_positions": float(np.mean(
                all_permutation_agreement
            )),
            "accuracy_by_target_slot": target_slot_accuracies,
            "maximum_target_slot_accuracy_gap": float(
                max(target_slot_accuracies) - min(target_slot_accuracies)
            ),
        },
    }
    return {
        "arms": ordinary_and_candidates,
        "primary_arm": primary_arm,
        "primary_specific": primary_specific,
        "paired_bag_bootstrap": paired_bootstrap_primary_advantage(
            base_evaluations, test["targets"], seed * 1_000_000 + 43
        ),
    }


def recursive_numbers_finite(value: object) -> bool:
    if isinstance(value, dict):
        return all(recursive_numbers_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(recursive_numbers_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def make_world_gates(result: dict[str, object]) -> dict[str, bool]:
    arms = result["arms"]
    counterfactuals = result["counterfactuals"]
    primary_name = counterfactuals["primary_arm"]
    primary = arms[primary_name]
    nonquadratic_names = (
        "bilinear",
        "gauge_linear_control",
        "position_only_mean_square_control",
    )
    best_nonquadratic_accuracy = max(
        arms[name]["selected_test_accuracy"] for name in nonquadratic_names
    )
    bootstrap = counterfactuals["paired_bag_bootstrap"]
    primary_cf = counterfactuals["arms"][primary_name]
    primary_specific = counterfactuals["primary_specific"]
    permutation = primary_specific["all_24_position_permutations"]
    shuffled = primary_specific["quadratic_content_shuffled"]
    ablated = primary_specific["scale_curvature_ablated"]
    bilinear = arms["bilinear"]
    gauge_linear = arms["gauge_linear_control"]
    position_only = arms["position_only_mean_square_control"]
    primary_restart_advantages = [
        primary["test_accuracy_by_restart"][restart]
        - max(
            arms[name]["test_accuracy_by_restart"][restart]
            for name in nonquadratic_names
        )
        for restart in range(RESTARTS)
    ]
    ood_keys = (
        "ood_laplace_four_record",
        "ood_gaussian_eight_record",
        "ood_laplace_eight_record",
    )
    ood_minimum_accuracies = {
        "ood_laplace_four_record": 0.90,
        "ood_gaussian_eight_record": 0.85,
        "ood_laplace_eight_record": 0.85,
    }
    return {
        "training_schedule_completed_with_finite_losses": bool(
            result["training_all_finite"] and result["steps"] == STEPS
        ),
        "every_trained_restart_parameter_is_finite": bool(
            result["all_trained_restart_parameters_finite"]
        ),
        "every_reported_numeric_value_is_finite": recursive_numbers_finite({
            "initial_logits_maximum_absolute_error_across_arms": result["initial_logits_maximum_absolute_error_across_arms"],
            "arms": arms,
            "counterfactuals": counterfactuals,
        }),
        "all_split_structural_checks_pass": all(
            all(checks.values())
            for checks in result["split_balance_checks"].values()
        ),
        "all_arms_have_40_raw_parameters_and_160_fp32_bytes": all(
            arm["raw_trainable_parameters"] == 40
            and arm["physical_serialized_bytes_fp32"] == 160
            for arm in arms.values()
        ),
        "initial_logits_match_across_arms": (
            result["initial_logits_maximum_absolute_error_across_arms"] <= 1e-7
        ),
        "primary_accuracy_at_least_90_percent": (
            primary["selected_test_accuracy"] >= 0.90
        ),
        "primary_each_query_type_accuracy_at_least_88_percent": all(
            accuracy >= 0.88
            for accuracy in primary["selected_test_accuracy_by_query_type"]
        ),
        "primary_test_nll_at_most_0_25": primary["selected_test_nll"] <= 0.25,
        "bootstrap_accuracy_advantage_lower_bound_at_least_0_40": (
            bootstrap["accuracy_advantage_lower_95_percent"] >= 0.40
        ),
        "bootstrap_nll_advantage_lower_bound_at_least_0_50": (
            bootstrap["nll_advantage_lower_95_percent"] >= 0.50
        ),
        "every_paired_restart_accuracy_advantage_at_least_0_40": all(
            advantage >= 0.40 for advantage in primary_restart_advantages
        ),
        "gauge_linear_equivalent_to_bilinear_within_0_04_accuracy_and_0_03_nll": (
            abs(gauge_linear["selected_test_accuracy"] - bilinear["selected_test_accuracy"]) <= 0.04
            and abs(gauge_linear["selected_test_nll"] - bilinear["selected_test_nll"]) <= 0.03
        ),
        "position_only_equivalent_to_bilinear_within_0_04_accuracy_and_0_03_nll": (
            abs(position_only["selected_test_accuracy"] - bilinear["selected_test_accuracy"]) <= 0.04
            and abs(position_only["selected_test_nll"] - bilinear["selected_test_nll"]) <= 0.03
        ),
        "uncentered_scale_g1_noninferior_to_centered_scale_g1_and_g2": all(
            primary["selected_test_accuracy"] >= arms[name]["selected_test_accuracy"] - 0.02
            and primary["selected_test_nll"] <= arms[name]["selected_test_nll"] + 0.03
            for name in ("centered_scale_only_geck", "centered_geck_g2", "full_geck_g2")
        ),
        "centered_scale_g1_long_ood_nll_not_worse_than_centered_rotation_g1_by_0_10": all(
            counterfactuals["arms"]["centered_scale_only_geck"][key]["nll"]
            <= counterfactuals["arms"]["centered_rotation_only_geck"][key]["nll"] + 0.10
            for key in ("ood_gaussian_eight_record", "ood_laplace_eight_record")
        ),
        "independent_sign_flip_accuracy_and_prediction_invariance": (
            primary_cf["sign_flipped"]["accuracy"]
            >= primary_cf["base"]["accuracy"] - 0.02
            and primary_cf["sign_flip_prediction_agreement"] >= 0.95
        ),
        "all_position_permutations_remain_accurate_invariant_and_balanced": (
            permutation["accuracy"] >= 0.90
            and permutation["prediction_agreement_with_original_positions"] >= 0.95
            and permutation["maximum_target_slot_accuracy_gap"] <= 0.03
        ),
        "quadratic_derangement_drops_accuracy_by_0_50_and_to_control_level": (
            primary_cf["base"]["accuracy"] - shuffled["accuracy"] >= 0.50
            and shuffled["accuracy"] <= best_nonquadratic_accuracy + 0.05
        ),
        "scale_curvature_ablation_drops_accuracy_by_0_50": (
            primary_cf["base"]["accuracy"] - ablated["accuracy"] >= 0.50
        ),
        "all_ood_accuracy_nll_and_control_advantage_gates_pass": all(
            primary_cf[key]["accuracy"] >= ood_minimum_accuracies[key]
            and primary_cf[key]["nll"] <= 0.50
            and primary_cf[key]["accuracy"] - max(
                counterfactuals["arms"][name][key]["accuracy"]
                for name in nonquadratic_names
            ) >= 0.40
            for key in ood_keys
        ),
    }


def train_world(
    seed: int,
    device: torch.device,
    steps: int = STEPS,
) -> tuple[dict[str, object], dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    splits = make_splits(seed)
    query, key, polar = paired_initial_parameters(seed * 100, device)
    parameters = (query, key, polar)
    initial_logits = []
    records, query_types, _ = tensors_from_split(splits["validation"], device)
    with torch.no_grad():
        for arm_index, arm in enumerate(ARMS):
            initial_logits.append(attention_logits(
                query[arm_index], key[arm_index], polar[arm_index],
                query_types[:64], records[:64], arm,
            ))
    initial_logit_error = float(torch.max(torch.abs(
        torch.stack(initial_logits) - initial_logits[0][None]
    )).item())

    optimizer = torch.optim.AdamW(
        parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    train_records, train_query_types, train_targets = tensors_from_split(
        splits["train"], device
    )
    order_generator = torch.Generator(device="cpu")
    order_generator.manual_seed(seed * 100_000 + 19)
    order = torch.randperm(TRAIN_SAMPLES, generator=order_generator).to(device)
    cursor = 0
    all_finite = True
    for step in range(steps):
        if cursor + BATCH_SIZE > TRAIN_SAMPLES:
            order = torch.randperm(TRAIN_SAMPLES, generator=order_generator).to(device)
            cursor = 0
        indices = order[cursor:cursor + BATCH_SIZE]
        cursor += BATCH_SIZE
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for arm_index, arm in enumerate(ARMS):
            logits = attention_logits(
                query[arm_index], key[arm_index], polar[arm_index],
                train_query_types[indices], train_records[indices], arm,
            )
            arm_losses = F.cross_entropy(
                logits.reshape(-1, TRAIN_ITEMS),
                train_targets[indices].repeat(RESTARTS),
                reduction="none",
            ).reshape(RESTARTS, -1).mean(dim=-1)
            losses.append(arm_losses)
        loss_matrix = torch.stack(losses)
        if not torch.isfinite(loss_matrix).all():
            all_finite = False
            break
        torch.sum(loss_matrix).backward()
        per_run_clip_gradients(parameters, GRADIENT_CLIP)
        multiplier = learning_rate_multiplier(step, steps)
        for group in optimizer.param_groups:
            group["lr"] = LEARNING_RATE * multiplier
        optimizer.step()

    selected: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    arm_results: dict[str, object] = {}
    for arm_index, arm in enumerate(ARMS):
        validation = evaluate_runs(
            query[arm_index], key[arm_index], polar[arm_index],
            splits["validation"], arm, device,
        )
        selected_restart = int(torch.argmin(validation["nll"]).item())
        chosen = (
            query[arm_index, selected_restart].detach().clone(),
            key[arm_index, selected_restart].detach().clone(),
            polar[arm_index, selected_restart].detach().clone(),
        )
        selected[arm] = chosen
        test = evaluate_runs(
            query[arm_index], key[arm_index], polar[arm_index],
            splits["test"], arm, device,
        )
        serialized = physical_serialized_parameters(*chosen)
        arm_results[arm] = {
            "selected_restart": selected_restart,
            "validation_nll_by_restart": validation["nll"].cpu().tolist(),
            "test_nll_by_restart": test["nll"].cpu().tolist(),
            "test_accuracy_by_restart": test["accuracy"].cpu().tolist(),
            "selected_test_nll": float(test["nll"][selected_restart].item()),
            "selected_test_accuracy": float(test["accuracy"][selected_restart].item()),
            "selected_test_accuracy_by_query_type": test["per_type_accuracy"][selected_restart].cpu().tolist(),
            "raw_trainable_parameters": int(serialized.numel()),
            "physical_serialized_bytes_fp32": int(serialized.numel() * serialized.element_size()),
            "query_norm": float(torch.linalg.vector_norm(chosen[0]).item()),
            "key_content_norm": float(torch.linalg.vector_norm(chosen[1]).item()),
            "polar_norm": float(torch.linalg.vector_norm(chosen[2]).item()),
            "maximum_absolute_logit": float(torch.max(torch.abs(test["logits"][selected_restart])).item()),
            "maximum_absolute_rotation_coefficient": float(torch.max(torch.abs(torch.sin(chosen[2][..., 0]))).item()),
            "maximum_absolute_scale_coefficient": float(torch.max(torch.abs(torch.tanh(chosen[2][..., 1]))).item()),
            "parameters": {
                "query": chosen[0].cpu().tolist(),
                "key_content": chosen[1].cpu().tolist(),
                "polar": chosen[2].cpu().tolist(),
            },
        }

    result: dict[str, object] = {
        "seed": seed,
        "steps": steps,
        "initial_logits_maximum_absolute_error_across_arms": initial_logit_error,
        "training_all_finite": all_finite,
        "all_trained_restart_parameters_finite": bool(
            torch.isfinite(query).all()
            and torch.isfinite(key).all()
            and torch.isfinite(polar).all()
        ),
        "split_balance_checks": {
            name: split_balance_checks(split, split["keys"].shape[1])
            for name, split in splits.items()
        },
        "arms": arm_results,
        "counterfactuals": counterfactual_evaluations(
            selected, splits, seed, device
        ),
    }
    result["gates"] = make_world_gates(result)
    result["pass"] = all(result["gates"].values())
    return result, selected


def validate_integrity() -> dict[str, bool]:
    manifest = json.loads(INTEGRITY_MANIFEST.read_text())
    paths = {
        "source": Path(__file__),
        "preregistration": PREREGISTRATION,
        "test": TEST_SOURCE,
        "learning_source": LEARNING_SOURCE,
        "learning_preregistration": LEARNING_PREREGISTRATION,
        "learning_test": LEARNING_TEST,
        "learning_integrity": LEARNING_INTEGRITY,
        "learning_result": LEARNING_RESULT,
    }
    checks = {
        name: manifest.get(f"{name}_sha256") == sha256_file(path)
        for name, path in paths.items()
    }
    if not all(checks.values()):
        raise ValueError(f"invalid GECK energy-screen integrity: {checks}")
    learning = json.loads(LEARNING_RESULT.read_text())
    if not learning["decision"]["learning_gate_pass"]:
        raise ValueError("GECK learned-family gate did not pass")
    return checks


def frozen_runtime() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
    }


def validate_frozen_runtime() -> dict[str, str]:
    runtime = frozen_runtime()
    expected = {
        "python": "3.11.10",
        "numpy": "2.1.2",
        "torch": "2.5.1+cu124",
        "cuda": "12.4",
        "gpu": "NVIDIA H100 80GB HBM3",
        "cublas_workspace_config": ":4096:8",
    }
    if runtime != expected:
        raise RuntimeError(f"frozen GECK energy runtime mismatch: {runtime}")
    return runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-seed", type=int)
    parser.add_argument("--steps", type=int, default=STEPS)
    args = parser.parse_args()
    if (
        args.development_seed is not None
        and args.development_seed != DEVELOPMENT_SEED
    ):
        raise RuntimeError(
            f"only the frozen development seed {DEVELOPMENT_SEED} may bypass formal integrity"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("frozen energy screen requires CUDA")
    device = torch.device("cuda")
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    integrity = None
    runtime = None
    if args.development_seed is None:
        if args.steps != STEPS:
            raise RuntimeError(f"formal run requires exactly {STEPS} steps")
        integrity = validate_integrity()
        runtime = validate_frozen_runtime()
    seeds = (
        (args.development_seed,)
        if args.development_seed is not None
        else WORLD_SEEDS
    )
    worlds = []
    for seed in seeds:
        world, _ = train_world(seed, device, steps=args.steps)
        worlds.append(world)
        print(json.dumps({
            "seed": seed,
            "steps": args.steps,
            "arms": {
                arm: {
                    "nll": world["arms"][arm]["selected_test_nll"],
                    "accuracy": world["arms"][arm]["selected_test_accuracy"],
                }
                for arm in ARMS
            },
            "primary_counterfactuals": {
                "arm": world["counterfactuals"]["primary_arm"],
                "sign": world["counterfactuals"]["arms"][world["counterfactuals"]["primary_arm"]]["sign_flipped"],
                "sign_agreement": world["counterfactuals"]["arms"][world["counterfactuals"]["primary_arm"]]["sign_flip_prediction_agreement"],
                "ood": {
                    arm: {
                        key: value
                        for key, value in world["counterfactuals"]["arms"][arm].items()
                        if key.startswith("ood_")
                    }
                    for arm in ARMS
                },
                **world["counterfactuals"]["primary_specific"],
            },
        }, sort_keys=True), flush=True)
    if args.development_seed is not None:
        return
    decision = {
        "energy_screen_pass": all(world["pass"] for world in worlds),
        "worlds_passed": sum(world["pass"] for world in worlds),
        "worlds_total": len(worlds),
        "primary_candidate": "uncentered-scale-funded-GECK-G1",
        "next_gate": "standard-task no-harm screen plus fused H100 epilogue benchmark",
        "h100_kernel_gate_admitted": all(world["pass"] for world in worlds),
        "language_model_gate_admitted": False,
    }
    payload = {
        "schema": "gauge-embedded-curved-keys-energy-v1",
        "candidate": "uncentered-scale-funded-GECK-G1",
        "integrity_checks": integrity,
        "runtime": runtime,
        "hashes": {
            "source": sha256_file(Path(__file__)),
            "preregistration": sha256_file(PREREGISTRATION),
            "test": sha256_file(TEST_SOURCE),
            "learning_result": sha256_file(LEARNING_RESULT),
        },
        "protocol": {
            "development_seed_excluded": DEVELOPMENT_SEED,
            "formal_world_seeds": list(WORLD_SEEDS),
            "train_samples": TRAIN_SAMPLES,
            "validation_samples": VALIDATION_SAMPLES,
            "test_samples": TEST_SAMPLES,
            "train_items": TRAIN_ITEMS,
            "ood_items": OOD_ITEMS,
            "head_dimension": HEAD_DIMENSION,
            "raw_parameters_every_arm": 40,
            "serialized_fp32_bytes_every_arm": 160,
            "restarts": RESTARTS,
            "steps": STEPS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "warmup_steps": WARMUP_STEPS,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": GRADIENT_CLIP,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        },
        "worlds": worlds,
        "decision": decision,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
