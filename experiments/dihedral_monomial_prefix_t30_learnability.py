"""Withdrawn pre-run draft.

The matched-control audit found that this screen compares against an additive
identity recurrence rather than the full diagonal-affine T29 class and that
its document arm uses only one group generator.  Importing the module remains
supported for forensic unit checks, but ``main`` intentionally refuses to
train.  See the paired pre-run audit in ``results/``.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


WIDTH = 109
RELATIONS = 8
VALUES = 8
DOCUMENTS = 4_096
TRAIN_DOCUMENTS = 3_072
EVALUATION_DOCUMENTS = 1_024
EVALUATION_CASES = 16_384
DATA_SEED = 30_001
SHUFFLE_SEED = 30_019
MODEL_SEEDS = (3_109, 3_203, 3_301)

UPDATES = 2_500
DOCUMENT_BATCH = 256
GROUP_BATCH = 256
LEARNING_RATE = 3e-3
GRADIENT_CLIP = 1.0

TOKEN_R = 0
TOKEN_F = 1
TOKEN_FILLER_A = 2
TOKEN_FILLER_B = 3
TOKEN_VALUE_START = 4
VOCAB = TOKEN_VALUE_START + VALUES
DOCUMENT_LENGTH = RELATIONS * 7
ROUTE_CHOICES: tuple[tuple[int, int], ...] = (
    (1, 0),
    (1, 1),
    (1, 4),
    (-1, 0),
)
IDENTITY_GROUP = (1, 0)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_sha256(values: Sequence[Tensor]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def compose_group(
    after_sign: Tensor,
    after_rotation: Tensor,
    before_sign: Tensor,
    before_rotation: Tensor,
) -> tuple[Tensor, Tensor]:
    return (
        after_sign * before_sign,
        (after_rotation + after_sign * before_rotation) % WIDTH,
    )


def group_source_indices(sign: int, rotation: int, device: torch.device) -> Tensor:
    destination = torch.arange(WIDTH, device=device)
    return (sign * (destination - rotation)) % WIDTH


def fixed_permute(state: Tensor, group: tuple[int, int]) -> Tensor:
    return state.index_select(1, group_source_indices(*group, state.device))


class ScreenModel(nn.Module):
    def __init__(self, seed: int, candidate: bool) -> None:
        super().__init__()
        self.candidate = candidate
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.route_logits = nn.Parameter(0.02 * torch.randn(2, 4, generator=generator))
        self.value_offsets = nn.Parameter(
            0.02 * torch.randn(VALUES, WIDTH, generator=generator)
        )
        self.relation_embedding = nn.Parameter(
            0.02 * torch.randn(RELATIONS, 16, generator=generator)
        )
        self.decoder_first = nn.Linear(WIDTH + 16, 256)
        self.decoder_second = nn.Linear(256, VALUES)
        with torch.no_grad():
            self.decoder_first.weight.copy_(
                0.02 * torch.randn(self.decoder_first.weight.shape, generator=generator)
            )
            self.decoder_first.bias.zero_()
            self.decoder_second.weight.copy_(
                0.02 * torch.randn(self.decoder_second.weight.shape, generator=generator)
            )
            self.decoder_second.bias.zero_()

    def decode(self, state: Tensor, relation: Tensor) -> Tensor:
        relation_values = self.relation_embedding.index_select(0, relation)
        hidden = torch.cat((state, relation_values), dim=-1)
        return self.decoder_second(F.silu(self.decoder_first(hidden)))

    def hard_route_indices(self) -> Tensor:
        return self.route_logits.argmax(dim=-1)

    def routed_permutation(self, state: Tensor, tokens: Tensor) -> Tensor:
        if not self.candidate:
            return state
        output = state
        for token, route_row in ((TOKEN_R, 0), (TOKEN_F, 1)):
            mask = tokens == token
            probabilities = self.route_logits[route_row].softmax(dim=-1)
            candidates = torch.stack(
                [fixed_permute(state, group) for group in ROUTE_CHOICES], dim=1
            )
            soft = (candidates * probabilities[None, :, None]).sum(dim=1)
            hard_weights = F.one_hot(
                probabilities.argmax(), num_classes=len(ROUTE_CHOICES)
            ).to(probabilities.dtype)
            hard = (candidates * hard_weights[None, :, None]).sum(dim=1)
            straight_through = soft + (hard - soft).detach()
            output = torch.where(mask[:, None], straight_through, output)
        return output

    def token_offset(self, tokens: Tensor) -> Tensor:
        output = torch.zeros(
            len(tokens), WIDTH, device=tokens.device, dtype=self.value_offsets.dtype
        )
        mask = (tokens >= TOKEN_VALUE_START) & (tokens < TOKEN_VALUE_START + VALUES)
        indices = (tokens - TOKEN_VALUE_START).clamp(0, VALUES - 1)
        selected = self.value_offsets.index_select(0, indices)
        return torch.where(mask[:, None], selected, output)

    def step(self, state: Tensor, tokens: Tensor) -> Tensor:
        permuted = self.routed_permutation(state, tokens)
        return permuted * 1.0 + self.token_offset(tokens)

    def scan(self, tokens: Tensor, initial: Tensor | None = None) -> Tensor:
        if initial is None:
            state = torch.zeros(
                len(tokens), WIDTH, device=tokens.device, dtype=self.value_offsets.dtype
            )
        else:
            state = initial
        for position in range(tokens.shape[1]):
            state = self.step(state, tokens[:, position])
        return state


def all_document_permutations() -> np.ndarray:
    values = np.asarray(list(itertools.permutations(range(VALUES))), dtype=np.int64)
    generator = np.random.default_rng(DATA_SEED)
    generator.shuffle(values)
    return values[:DOCUMENTS]


def build_document_tokens(permutations: np.ndarray) -> np.ndarray:
    rows: list[list[int]] = []
    for values in permutations:
        row: list[int] = []
        for value in values.tolist():
            row.extend(
                (
                    TOKEN_FILLER_A,
                    TOKEN_VALUE_START + int(value),
                    TOKEN_R,
                    TOKEN_R,
                    TOKEN_R,
                    TOKEN_R,
                    TOKEN_FILLER_B,
                )
            )
        rows.append(row)
    result = np.asarray(rows, dtype=np.int64)
    if result.shape != (len(permutations), DOCUMENT_LENGTH):
        raise RuntimeError(f"document token shape mismatch: {result.shape}")
    return result


@dataclass(frozen=True)
class World:
    permutations: np.ndarray
    document_tokens: np.ndarray
    train_indices: np.ndarray
    evaluation_indices: np.ndarray


def build_world() -> World:
    permutations = all_document_permutations()
    document_tokens = build_document_tokens(permutations)
    return World(
        permutations=permutations,
        document_tokens=document_tokens,
        train_indices=np.arange(TRAIN_DOCUMENTS, dtype=np.int64),
        evaluation_indices=np.arange(TRAIN_DOCUMENTS, DOCUMENTS, dtype=np.int64),
    )


def document_multiset_failures(tokens: np.ndarray) -> int:
    reference = np.bincount(tokens[0], minlength=VOCAB)
    return sum(
        not np.array_equal(np.bincount(row, minlength=VOCAB), reference)
        for row in tokens
    )


def true_group_for_words(tokens: np.ndarray, lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sign = np.ones(len(tokens), dtype=np.int64)
    rotation = np.zeros(len(tokens), dtype=np.int64)
    for position in range(tokens.shape[1]):
        active = position < lengths
        token = tokens[:, position]
        after_sign = np.where(token == TOKEN_F, -1, 1)
        after_rotation = np.where(token == TOKEN_R, 1, 0)
        new_sign = after_sign * sign
        new_rotation = (after_rotation + after_sign * rotation) % WIDTH
        sign = np.where(active, new_sign, sign)
        rotation = np.where(active, new_rotation, rotation)
    return sign, rotation


def marker_targets(sign: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    targets = np.zeros((len(sign), WIDTH), dtype=np.float32)
    rows = np.arange(len(sign))
    targets[rows, rotation % WIDTH] = 1.0
    targets[rows, (sign + rotation) % WIDTH] = 2.0
    return targets


def group_training_batch(seed: int, step: int) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(DATA_SEED * 1_000_003 + seed * 10_007 + step)
    lengths = generator.integers(1, 17, size=GROUP_BATCH, dtype=np.int64)
    tokens = generator.integers(0, 2, size=(GROUP_BATCH, 16), dtype=np.int64)
    tokens[0] = TOKEN_R
    lengths[0] = 1
    tokens[1] = TOKEN_F
    lengths[1] = 1
    mask = np.arange(16)[None, :] >= lengths[:, None]
    tokens[mask] = TOKEN_FILLER_A
    sign, rotation = true_group_for_words(tokens, lengths)
    return tokens, marker_targets(sign, rotation)


def document_training_batch(
    world: World, seed: int, step: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(DATA_SEED * 1_000_033 + seed * 10_009 + step)
    first = generator.integers(0, TRAIN_DOCUMENTS, size=DOCUMENT_BATCH, dtype=np.int64)
    second = generator.integers(0, TRAIN_DOCUMENTS, size=DOCUMENT_BATCH, dtype=np.int64)
    relation = generator.integers(0, RELATIONS, size=DOCUMENT_BATCH, dtype=np.int64)
    target = (
        world.permutations[first, relation] - world.permutations[second, relation]
    ) % VALUES
    return first, second, relation, target


def initial_marker(batch: int, device: torch.device) -> Tensor:
    state = torch.zeros(batch, WIDTH, device=device)
    state[:, 0] = 1.0
    state[:, 1] = 2.0
    return state


def model_parameter_count(model: ScreenModel) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def train_arm(
    model: ScreenModel,
    world: World,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    maximum_loss = 0.0
    maximum_gradient = 0.0
    failed_steps = 0
    evaluation_gradient_presentations = 0
    started = time.perf_counter()
    train_tokens = torch.from_numpy(world.document_tokens[:TRAIN_DOCUMENTS]).to(device)
    for step in range(1, UPDATES + 1):
        optimizer.zero_grad()
        group_tokens_np, group_targets_np = group_training_batch(seed, step)
        first_np, second_np, relation_np, target_np = document_training_batch(
            world, seed, step
        )
        if int(first_np.max()) >= TRAIN_DOCUMENTS or int(second_np.max()) >= TRAIN_DOCUMENTS:
            evaluation_gradient_presentations += 1
        group_tokens = torch.from_numpy(group_tokens_np).to(device)
        group_targets = torch.from_numpy(group_targets_np).to(device)
        group_state = model.scan(group_tokens, initial_marker(GROUP_BATCH, device))
        group_loss = (group_state - group_targets).square().sum(dim=-1).mean()

        first = torch.from_numpy(first_np).to(device)
        second = torch.from_numpy(second_np).to(device)
        relation = torch.from_numpy(relation_np).to(device)
        target = torch.from_numpy(target_np).to(device)
        first_tokens = train_tokens.index_select(0, first)
        second_tokens = train_tokens.index_select(0, second)
        if model.candidate:
            state = model.scan(first_tokens)
            state = model.scan(second_tokens, state)
        else:
            # FP64 summation makes the exact additive alias independent of value order.
            common = model.value_offsets.double().sum(dim=0).float()
            state = 2.0 * common[None, :].expand(DOCUMENT_BATCH, -1)
        logits = model.decode(state, relation)
        document_loss = F.cross_entropy(logits, target)
        loss = group_loss + document_loss
        if not torch.isfinite(loss):
            failed_steps += 1
            raise RuntimeError(f"nonfinite loss for seed {seed} step {step}")
        loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP))
        if not math.isfinite(gradient):
            failed_steps += 1
            raise RuntimeError(f"nonfinite gradient for seed {seed} step {step}")
        optimizer.step()
        maximum_loss = max(maximum_loss, float(loss.item()))
        maximum_gradient = max(maximum_gradient, gradient)
        if step % 500 == 0:
            print(
                json.dumps(
                    {
                        "phase": "train",
                        "arm": "candidate" if model.candidate else "diagonal",
                        "seed": seed,
                        "step": step,
                        "loss": float(loss.item()),
                        "group_loss": float(group_loss.item()),
                        "document_loss": float(document_loss.item()),
                        "routes": model.hard_route_indices().tolist(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return {
        "updates": UPDATES,
        "document_presentations": UPDATES * DOCUMENT_BATCH * 2,
        "group_presentations": UPDATES * GROUP_BATCH,
        "evaluation_gradient_presentations": evaluation_gradient_presentations,
        "maximum_loss": maximum_loss,
        "maximum_gradient_norm": maximum_gradient,
        "failed_or_retried_steps": failed_steps,
        "elapsed_seconds": time.perf_counter() - started,
        "terminal_routes": model.hard_route_indices().tolist(),
    }


def canonical_group_words(lengthened: bool) -> tuple[np.ndarray, np.ndarray]:
    words: list[list[int]] = []
    groups: list[tuple[int, int]] = []
    neutral = [TOKEN_F, TOKEN_F] + [TOKEN_R] * WIDTH if lengthened else []
    for sign in (1, -1):
        for rotation in range(WIDTH):
            canonical = ([TOKEN_F] if sign == -1 else []) + [TOKEN_R] * rotation
            words.append(neutral + canonical)
            groups.append((sign, rotation))
    maximum = max(map(len, words))
    padded = np.full((len(words), maximum), TOKEN_FILLER_A, dtype=np.int64)
    lengths = np.asarray([len(word) for word in words], dtype=np.int64)
    for row, word in enumerate(words):
        padded[row, : len(word)] = word
    target = marker_targets(
        np.asarray([group[0] for group in groups]),
        np.asarray([group[1] for group in groups]),
    )
    return padded, target


@torch.no_grad()
def evaluate_group(model: ScreenModel, device: torch.device, lengthened: bool) -> dict[str, Any]:
    tokens_np, targets_np = canonical_group_words(lengthened)
    tokens = torch.from_numpy(tokens_np).to(device)
    targets = torch.from_numpy(targets_np).to(device)
    observed = model.scan(tokens, initial_marker(len(tokens), device))
    errors = (observed - targets).abs().amax(dim=-1)
    return {
        "examples": len(tokens),
        "maximum_state_error": float(errors.max().item()),
        "exact_accuracy": float((errors <= 1e-5).float().mean().item()),
        "word_length_min": int((tokens_np != TOKEN_FILLER_A).sum(axis=1).min()),
        "word_length_max": int((tokens_np != TOKEN_FILLER_A).sum(axis=1).max()),
    }


def hard_token_groups(model: ScreenModel) -> dict[int, tuple[int, int]]:
    routes = model.hard_route_indices().tolist()
    return {
        TOKEN_R: ROUTE_CHOICES[routes[0]] if model.candidate else IDENTITY_GROUP,
        TOKEN_F: ROUTE_CHOICES[routes[1]] if model.candidate else IDENTITY_GROUP,
    }


def numpy_permute(group: tuple[int, int], values: np.ndarray) -> np.ndarray:
    sign, rotation = group
    source = (sign * (np.arange(WIDTH) - rotation)) % WIDTH
    return values[..., source]


def numpy_compose_group(after: tuple[int, int], before: tuple[int, int]) -> tuple[int, int]:
    return (
        after[0] * before[0],
        (after[1] + after[0] * before[1]) % WIDTH,
    )


@torch.no_grad()
def compile_documents(model: ScreenModel, tokens: np.ndarray) -> tuple[np.ndarray, Tensor]:
    if not model.candidate:
        common = model.value_offsets.double().sum(dim=0).float()
        return (
            np.tile(np.asarray(IDENTITY_GROUP, dtype=np.int64), (len(tokens), 1)),
            common[None, :].expand(len(tokens), -1).clone(),
        )
    groups = hard_token_groups(model)
    batch = len(tokens)
    signs = np.ones(batch, dtype=np.int64)
    rotations = np.zeros(batch, dtype=np.int64)
    offsets = torch.zeros(batch, WIDTH, device=model.value_offsets.device)
    for position in range(tokens.shape[1]):
        column = tokens[:, position]
        for token, group in groups.items():
            mask_np = column == token
            if not bool(mask_np.any()):
                continue
            mask = torch.from_numpy(mask_np).to(offsets.device)
            selected = fixed_permute(offsets, group)
            offsets = torch.where(mask[:, None], selected, offsets)
            old_signs = signs.copy()
            old_rotations = rotations.copy()
            signs[mask_np] = group[0] * old_signs[mask_np]
            rotations[mask_np] = (
                group[1] + group[0] * old_rotations[mask_np]
            ) % WIDTH
        value_mask_np = (column >= TOKEN_VALUE_START) & (
            column < TOKEN_VALUE_START + VALUES
        )
        if bool(value_mask_np.any()):
            value_mask = torch.from_numpy(value_mask_np).to(offsets.device)
            values = torch.from_numpy(column[value_mask_np] - TOKEN_VALUE_START).to(
                offsets.device
            )
            addition = torch.zeros_like(offsets)
            addition[value_mask] = model.value_offsets.index_select(0, values)
            offsets = offsets + addition
    encoded_groups = np.stack((signs, rotations), axis=1)
    return encoded_groups, offsets


def apply_compiled(groups: np.ndarray, offsets: Tensor, state: Tensor) -> Tensor:
    output = torch.empty_like(state)
    unique = np.unique(groups, axis=0)
    for sign, rotation in unique.tolist():
        mask_np = (groups[:, 0] == sign) & (groups[:, 1] == rotation)
        mask = torch.from_numpy(mask_np).to(state.device)
        output[mask] = fixed_permute(state[mask], (int(sign), int(rotation))) + offsets[mask]
    return output


def evaluation_cases(world: World, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(DATA_SEED * 1_000_037 + seed)
    first_local = generator.integers(0, EVALUATION_DOCUMENTS, size=EVALUATION_CASES, dtype=np.int64)
    second_local = generator.integers(0, EVALUATION_DOCUMENTS, size=EVALUATION_CASES, dtype=np.int64)
    relation = generator.integers(0, RELATIONS, size=EVALUATION_CASES, dtype=np.int64)
    first = first_local + TRAIN_DOCUMENTS
    second = second_local + TRAIN_DOCUMENTS
    target = (
        world.permutations[first, relation] - world.permutations[second, relation]
    ) % VALUES
    return first, second, relation, target


@torch.no_grad()
def accuracy_from_logits(logits: Tensor, targets: Tensor) -> float:
    return float((logits.argmax(dim=-1) == targets).float().mean().item())


@torch.no_grad()
def evaluate_documents(
    candidate: ScreenModel,
    diagonal: ScreenModel,
    world: World,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    eval_tokens_np = world.document_tokens[TRAIN_DOCUMENTS:]
    candidate_groups, candidate_offsets = compile_documents(candidate, eval_tokens_np)
    diagonal_groups, diagonal_offsets = compile_documents(diagonal, eval_tokens_np)
    first, second, relation_np, target_np = evaluation_cases(world, seed)
    first_local = first - TRAIN_DOCUMENTS
    second_local = second - TRAIN_DOCUMENTS
    relation = torch.from_numpy(relation_np).to(device)
    target = torch.from_numpy(target_np).to(device)
    shuffle_offset = SHUFFLE_SEED % (EVALUATION_DOCUMENTS - 1) + 1

    totals = {
        "candidate_compiled": 0,
        "candidate_online": 0,
        "candidate_zero": 0,
        "candidate_shuffle": 0,
        "candidate_state_cache": 0,
        "diagonal": 0,
    }
    maximum_state_difference = 0.0
    maximum_logit_difference = 0.0
    prediction_mismatches = 0
    for start in range(0, EVALUATION_CASES, 256):
        stop = min(start + 256, EVALUATION_CASES)
        first_batch = first_local[start:stop]
        second_batch = second_local[start:stop]
        rows = stop - start
        zero = torch.zeros(rows, WIDTH, device=device)
        first_groups = candidate_groups[first_batch]
        second_groups = candidate_groups[second_batch]
        first_offsets = candidate_offsets[first_batch]
        second_offsets = candidate_offsets[second_batch]
        compiled = apply_compiled(first_groups, first_offsets, zero)
        compiled = apply_compiled(second_groups, second_offsets, compiled)

        first_tokens = torch.from_numpy(eval_tokens_np[first_batch]).to(device)
        second_tokens = torch.from_numpy(eval_tokens_np[second_batch]).to(device)
        online = candidate.scan(first_tokens)
        online = candidate.scan(second_tokens, online)

        relation_batch = relation[start:stop]
        target_batch = target[start:stop]
        compiled_logits = candidate.decode(compiled, relation_batch)
        online_logits = candidate.decode(online, relation_batch)
        totals["candidate_compiled"] += int(
            (compiled_logits.argmax(dim=-1) == target_batch).sum().item()
        )
        totals["candidate_online"] += int(
            (online_logits.argmax(dim=-1) == target_batch).sum().item()
        )
        maximum_state_difference = max(
            maximum_state_difference, float((compiled - online).abs().max().item())
        )
        maximum_logit_difference = max(
            maximum_logit_difference,
            float((compiled_logits - online_logits).abs().max().item()),
        )
        prediction_mismatches += int(
            (compiled_logits.argmax(dim=-1) != online_logits.argmax(dim=-1)).sum().item()
        )

        zero_logits = candidate.decode(zero, relation_batch)
        totals["candidate_zero"] += int(
            (zero_logits.argmax(dim=-1) == target_batch).sum().item()
        )

        shuffled_first = (first_batch + shuffle_offset) % EVALUATION_DOCUMENTS
        shuffled_second = (second_batch + shuffle_offset) % EVALUATION_DOCUMENTS
        shuffled = apply_compiled(
            candidate_groups[shuffled_first], candidate_offsets[shuffled_first], zero
        )
        shuffled = apply_compiled(
            candidate_groups[shuffled_second], candidate_offsets[shuffled_second], shuffled
        )
        shuffled_logits = candidate.decode(shuffled, relation_batch)
        totals["candidate_shuffle"] += int(
            (shuffled_logits.argmax(dim=-1) == target_batch).sum().item()
        )

        state_cache = first_offsets + second_offsets
        state_cache_logits = candidate.decode(state_cache, relation_batch)
        totals["candidate_state_cache"] += int(
            (state_cache_logits.argmax(dim=-1) == target_batch).sum().item()
        )

        diagonal_state = diagonal_offsets[first_batch] + diagonal_offsets[second_batch]
        diagonal_logits = diagonal.decode(diagonal_state, relation_batch)
        totals["diagonal"] += int(
            (diagonal_logits.argmax(dim=-1) == target_batch).sum().item()
        )

    accuracies = {name: value / EVALUATION_CASES for name, value in totals.items()}
    return {
        "examples": EVALUATION_CASES,
        "accuracies": accuracies,
        "maximum_compiled_online_state_difference": maximum_state_difference,
        "maximum_compiled_online_logit_difference": maximum_logit_difference,
        "compiled_online_prediction_mismatches": prediction_mismatches,
        "shuffle_offset": shuffle_offset,
        "shuffle_fixed_points": int(
            sum(
                index == (index + shuffle_offset) % EVALUATION_DOCUMENTS
                for index in range(EVALUATION_DOCUMENTS)
            )
        ),
        "candidate_compiled_group_codes": int(
            len({(int(sign), int(rotation)) for sign, rotation in candidate_groups.tolist()})
        ),
        "diagonal_compiled_group_codes": int(
            len({(int(sign), int(rotation)) for sign, rotation in diagonal_groups.tolist()})
        ),
    }


def state_schema(model: ScreenModel) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple((name, tuple(parameter.shape)) for name, parameter in model.named_parameters())


def run_seed(world: World, seed: int, device: torch.device) -> dict[str, Any]:
    candidate = ScreenModel(seed, True).to(device)
    diagonal = ScreenModel(seed, False).to(device)
    initial_candidate_hash = tensor_sha256(tuple(candidate.parameters()))
    initial_diagonal_hash = tensor_sha256(tuple(diagonal.parameters()))
    if initial_candidate_hash != initial_diagonal_hash:
        raise RuntimeError("candidate and diagonal initial tensors differ")

    candidate_training = train_arm(candidate, world, seed, device)
    diagonal_training = train_arm(diagonal, world, seed, device)
    candidate.eval()
    diagonal.eval()
    group = {
        "candidate_canonical": evaluate_group(candidate, device, False),
        "candidate_lengthened": evaluate_group(candidate, device, True),
        "diagonal_canonical": evaluate_group(diagonal, device, False),
        "diagonal_lengthened": evaluate_group(diagonal, device, True),
    }
    documents = evaluate_documents(candidate, diagonal, world, seed, device)
    candidate_document = documents["accuracies"]["candidate_compiled"]
    diagonal_document = documents["accuracies"]["diagonal"]
    control_names = ("candidate_zero", "candidate_shuffle", "candidate_state_cache")
    gates = {
        "candidate_group_canonical_at_least_95pct": group["candidate_canonical"]["exact_accuracy"] >= 0.95,
        "candidate_group_lengthened_at_least_95pct": group["candidate_lengthened"]["exact_accuracy"] >= 0.95,
        "candidate_group_gain_over_diagonal_at_least_70pp": min(
            group["candidate_canonical"]["exact_accuracy"] - group["diagonal_canonical"]["exact_accuracy"],
            group["candidate_lengthened"]["exact_accuracy"] - group["diagonal_lengthened"]["exact_accuracy"],
        ) >= 0.70,
        "compiled_online_exact": documents["maximum_compiled_online_state_difference"] <= 1e-5
        and documents["maximum_compiled_online_logit_difference"] <= 1e-5
        and documents["compiled_online_prediction_mismatches"] == 0,
        "candidate_document_at_least_90pct": candidate_document >= 0.90,
        "candidate_gain_over_diagonal_at_least_30pp": candidate_document - diagonal_document >= 0.30,
        "candidate_gain_over_all_controls_at_least_30pp": all(
            candidate_document - documents["accuracies"][name] >= 0.30
            for name in control_names
        ),
        "finite_stable_routes": bool(
            torch.isfinite(candidate.route_logits).all()
            and torch.isfinite(diagonal.route_logits).all()
            and len(candidate.hard_route_indices()) == 2
        ),
        "heldout_isolation_and_finite_training": candidate_training["evaluation_gradient_presentations"] == 0
        and diagonal_training["evaluation_gradient_presentations"] == 0
        and candidate_training["failed_or_retried_steps"] == 0
        and diagonal_training["failed_or_retried_steps"] == 0
        and math.isfinite(candidate_training["maximum_loss"])
        and math.isfinite(diagonal_training["maximum_loss"])
        and math.isfinite(candidate_training["maximum_gradient_norm"])
        and math.isfinite(diagonal_training["maximum_gradient_norm"]),
        "matched_resources": model_parameter_count(candidate) == model_parameter_count(diagonal)
        and state_schema(candidate) == state_schema(diagonal)
        and WIDTH * 2 + 2 == 220
        and 2 * WIDTH + 2 == 220,
    }
    return {
        "seed": seed,
        "initial_tensor_sha256": initial_candidate_hash,
        "candidate_training": candidate_training,
        "diagonal_training": diagonal_training,
        "group": group,
        "documents": documents,
        "effects": {
            "candidate_gain_over_diagonal_pp": 100.0 * (candidate_document - diagonal_document),
            "candidate_gain_over_zero_pp": 100.0 * (
                candidate_document - documents["accuracies"]["candidate_zero"]
            ),
            "candidate_gain_over_shuffle_pp": 100.0 * (
                candidate_document - documents["accuracies"]["candidate_shuffle"]
            ),
            "candidate_gain_over_state_cache_pp": 100.0 * (
                candidate_document - documents["accuracies"]["candidate_state_cache"]
            ),
        },
        "resources": {
            "candidate_parameters": model_parameter_count(candidate),
            "diagonal_parameters": model_parameter_count(diagonal),
            "state_bytes": WIDTH * 2 + 2,
            "record_cells": 2 * WIDTH + 2,
            "multiplies_per_token": WIDTH,
            "additions_per_token": WIDTH,
        },
        "gates": gates,
        "passed": all(gates.values()),
    }


def run(device: torch.device) -> dict[str, Any]:
    started = time.perf_counter()
    world = build_world()
    multiset_failures = document_multiset_failures(world.document_tokens)
    overlap = np.intersect1d(world.train_indices, world.evaluation_indices)
    seed_results = [run_seed(world, seed, device) for seed in MODEL_SEEDS]
    aggregate_gates = {
        "document_count_and_split": len(world.permutations) == DOCUMENTS
        and len(world.train_indices) == TRAIN_DOCUMENTS
        and len(world.evaluation_indices) == EVALUATION_DOCUMENTS
        and len(overlap) == 0,
        "identical_document_multisets": multiset_failures == 0,
        "three_seeds": len(seed_results) == 3,
        "every_seed_passed": all(result["passed"] for result in seed_results),
    }
    root = Path(__file__).resolve().parents[1]
    preregistration = root / "results/dihedral-monomial-prefix-t30-learnability-preregistration.md"
    stage0 = root / "results/dihedral-monomial-prefix-t30-stage0.json"
    tests = root / "tests/test_dihedral_monomial_prefix_t30_learnability.py"
    return {
        "experiment": "dihedral-monomial-prefix-t30-learnability",
        "status": "PASS" if all(aggregate_gates.values()) else "FAIL",
        "world": {
            "documents": len(world.permutations),
            "training_documents": len(world.train_indices),
            "evaluation_documents": len(world.evaluation_indices),
            "split_overlap": len(overlap),
            "document_length": DOCUMENT_LENGTH,
            "document_multiset_failures": multiset_failures,
            "evaluation_cases": EVALUATION_CASES,
        },
        "configuration": {
            "model_seeds": MODEL_SEEDS,
            "updates": UPDATES,
            "document_batch": DOCUMENT_BATCH,
            "group_batch": GROUP_BATCH,
            "learning_rate": LEARNING_RATE,
            "gradient_clip": GRADIENT_CLIP,
            "route_choices": ROUTE_CHOICES,
            "precision": "float32",
        },
        "seeds": seed_results,
        "gates": aggregate_gates,
        "runtime": {
            "device": str(device),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "wall_seconds": time.perf_counter() - started,
        },
        "integrity": {
            "preregistration_sha256": sha256_file(preregistration),
            "stage0_sha256": sha256_file(stage0),
            "source_sha256": sha256_file(Path(__file__).resolve()),
            "test_sha256": sha256_file(tests),
        },
        "claim_boundary": (
            "Synthetic D_109 route learning and permutation-binding only; no raw-prose, "
            "natural-semantic, quantized-model, physical-kernel, novelty, or production claim."
        ),
    }


def main() -> None:
    raise RuntimeError(
        "withdrawn before execution: the document arm does not test the claimed "
        "noncommutative separation against a full diagonal-affine control"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(DATA_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(DATA_SEED)
    print(json.dumps(run(device), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
