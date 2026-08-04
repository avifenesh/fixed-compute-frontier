#!/usr/bin/env python3
"""T6: infer support and algebraic family, compile, and abstain on decoys."""

from __future__ import annotations

import argparse
import functools
import hashlib
import itertools
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import shared_algebraic_interpreter_t3 as t3


PREREGISTRATION = ROOT / "results/heterogeneous-family-discovery-t6-preregistration.md"
TEST_SOURCE = ROOT / "tests/test_heterogeneous_family_discovery_t6.py"
PREDECESSOR = ROOT / "results/shared-algebraic-interpreter-t3.json"
OUTPUT = ROOT / "results/heterogeneous-family-discovery-t6.json"

VOCAB = t3.VOCAB
HIDDEN = t3.HIDDEN
LAYERS = t3.LAYERS
HEAD_DIM = t3.HEAD_DIM
FFN_WIDTH = t3.FFN_WIDTH
CONTEXT = t3.CONTEXT
INPUT_DIM = 16
SUPPORT_SIZE = 8
FAMILY_NAMES = ("parity", "majority", "exact_half", "mod3")
TASKS_PER_FAMILY = 64
STRUCTURED_TASKS = len(FAMILY_NAMES) * TASKS_PER_FAMILY
RANDOM_TASKS = 64
TASKS = STRUCTURED_TASKS + RANDOM_TASKS
PREFIX_PER_TASK = 96
PROGRAM_DIMS = 28
PROGRAM_HEADS = 2
PROGRAM_CHANNELS = 128
NATURAL_BATCH = 16
ALGORITHM_BATCH = 64
ALGORITHM_INTERVAL = 20
CHECKPOINTS_1X = (250, 500, 1000)
STEPS_2X = 2000
MODEL_SEEDS = (2039, 2281)
PEAK_LEARNING_RATE = t3.PEAK_LEARNING_RATE
RMS_EPSILON = t3.RMS_EPSILON

SUPPORT_START = 0
FAMILY_START = 16
SUM = 20
COPY_SUM = 21
CONST = 22
TASK = 23
BIT_TYPE = 24
PARITY_ROUTE = 25
COPY_ROUTE = 26
OUT = 27

TokenStream = t3.TokenStream
TokenLayout = t3.TokenLayout
SharedInterpreterLM = t3.SharedInterpreterLM
FrozenEntries = t3.FrozenEntries


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_token_layout(train: TokenStream, validation: TokenStream) -> tuple[TokenLayout, int]:
    counts = np.zeros(VOCAB, dtype=np.int64)
    for stream in (train, validation):
        counts += np.bincount(np.asarray(stream.tokens, dtype=np.int64), minlength=VOCAB)
    required = TASKS + 2 * INPUT_DIM + 4
    unused = np.flatnonzero(counts == 0)
    if len(unused) < required:
        raise RuntimeError(f"need {required} unused tokens, found {len(unused)}")
    selected = tuple(int(value) for value in unused[:required])
    cursor = 0
    task = selected[cursor : cursor + TASKS]
    cursor += TASKS
    zero = selected[cursor : cursor + INPUT_DIM]
    cursor += INPUT_DIM
    one = selected[cursor : cursor + INPUT_DIM]
    cursor += INPUT_DIM
    return TokenLayout(task, zero, one, *selected[cursor : cursor + 4]), len(unused)


@functools.lru_cache(maxsize=1)
def candidate_supports() -> Tensor:
    rows = torch.zeros(math.comb(INPUT_DIM, SUPPORT_SIZE), INPUT_DIM, dtype=torch.uint8)
    for index, selected in enumerate(itertools.combinations(range(INPUT_DIM), SUPPORT_SIZE)):
        rows[index, list(selected)] = 1
    return rows


def family_labels(counts: Tensor, family: int) -> Tensor:
    if family == 0:
        return counts.remainder(2).bool()
    if family == 1:
        return counts >= 5
    if family == 2:
        return counts == 4
    if family == 3:
        return counts.remainder(3) == 0
    raise ValueError(f"unknown family {family}")


def matching_programs(bits: Tensor, labels: Tensor) -> list[tuple[int, int]]:
    candidates = candidate_supports()
    counts = bits.to(torch.int16) @ candidates.T.to(torch.int16)
    matches: list[tuple[int, int]] = []
    for family in range(len(FAMILY_NAMES)):
        predicted = family_labels(counts, family)
        support_matches = (predicted == labels.bool()[:, None]).all(0).nonzero().flatten()
        matches.extend((int(index), family) for index in support_matches.tolist())
    return matches


@dataclass(frozen=True)
class DiscoveryWorld:
    supports: Tensor
    families: Tensor
    prefix_bits: Tensor
    prefix_labels: Tensor
    matches: tuple[tuple[tuple[int, int], ...], ...]


@functools.lru_cache(maxsize=1)
def make_world(seed: int = 131_071) -> DiscoveryWorld:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    candidates = candidate_supports()
    supports = torch.zeros(TASKS, INPUT_DIM, dtype=torch.uint8)
    families = torch.full((TASKS,), -1, dtype=torch.long)
    prefix_bits = torch.zeros(TASKS, PREFIX_PER_TASK, INPUT_DIM, dtype=torch.uint8)
    prefix_labels = torch.zeros(TASKS, PREFIX_PER_TASK, dtype=torch.long)
    all_matches: list[tuple[tuple[int, int], ...]] = []

    for task in range(TASKS):
        family = task // TASKS_PER_FAMILY if task < STRUCTURED_TASKS else -1
        families[task] = family
        if family >= 0:
            support_index = int(torch.randint(0, len(candidates), (1,), generator=generator))
            support = candidates[support_index]
            supports[task] = support
        for _attempt in range(100):
            bits = torch.randint(
                0, 2, (PREFIX_PER_TASK, INPUT_DIM), generator=generator, dtype=torch.uint8
            )
            if family >= 0:
                counts = (bits.to(torch.int16) * support.to(torch.int16)).sum(1)
                labels = family_labels(counts, family).long()
            else:
                labels = torch.randint(0, 2, (PREFIX_PER_TASK,), generator=generator)
            matches = matching_programs(bits, labels)
            if (family >= 0 and matches == [(support_index, family)]) or (
                family < 0 and not matches
            ):
                prefix_bits[task] = bits
                prefix_labels[task] = labels
                all_matches.append(tuple(matches))
                break
        else:
            raise RuntimeError(f"could not create identifiable task {task}")
    return DiscoveryWorld(supports, families, prefix_bits, prefix_labels, tuple(all_matches))


def discover(world: DiscoveryWorld) -> tuple[Tensor, Tensor, Tensor]:
    recovered_supports = torch.zeros_like(world.supports)
    recovered_families = torch.full_like(world.families, -1)
    accepted = torch.zeros(TASKS, dtype=torch.bool)
    candidates = candidate_supports()
    for task in range(TASKS):
        matches = matching_programs(world.prefix_bits[task], world.prefix_labels[task])
        if len(matches) == 1:
            support_index, family = matches[0]
            recovered_supports[task] = candidates[support_index]
            recovered_families[task] = family
            accepted[task] = True
    return recovered_supports, recovered_families, accepted


def encode_algorithm(
    bits: Tensor,
    tasks: Tensor,
    parity_route: Tensor,
    layout: TokenLayout,
    device: torch.device,
) -> Tensor:
    zero = torch.tensor(layout.zero, dtype=torch.long)
    one = torch.tensor(layout.one, dtype=torch.long)
    bit_tokens = torch.where(bits.bool(), one[None, :], zero[None, :])
    task_tokens = torch.tensor(layout.rule, dtype=torch.long)[tasks]
    queries = torch.where(
        parity_route,
        torch.full((len(tasks),), layout.parity_query, dtype=torch.long),
        torch.full((len(tasks),), layout.copy_query, dtype=torch.long),
    )
    return torch.cat((task_tokens[:, None], bit_tokens, queries[:, None]), dim=1).to(device)


def targets_for(world: DiscoveryWorld, bits: Tensor, tasks: Tensor, seed: int) -> Tensor:
    structured = tasks < STRUCTURED_TASKS
    targets = torch.empty(len(tasks), dtype=torch.long)
    if structured.any():
        selected_tasks = tasks[structured]
        counts = (
            bits[structured].to(torch.int16)
            * world.supports[selected_tasks].to(torch.int16)
        ).sum(1)
        structured_targets = torch.empty(len(selected_tasks), dtype=torch.long)
        for family in range(len(FAMILY_NAMES)):
            mask = world.families[selected_tasks] == family
            structured_targets[mask] = family_labels(counts[mask], family).long()
        targets[structured] = structured_targets
    if (~structured).any():
        generator = torch.Generator(device="cpu").manual_seed(seed + 9_999_991)
        targets[~structured] = torch.randint(0, 2, (int((~structured).sum()),), generator=generator)
    return targets


def make_algorithm_batch(
    world: DiscoveryWorld,
    layout: TokenLayout,
    count: int,
    seed: int,
    device: torch.device,
    *,
    parity_only: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tasks = torch.randint(0, TASKS, (count,), generator=generator)
    bits = torch.randint(0, 2, (count, INPUT_DIM), generator=generator, dtype=torch.uint8)
    route = (
        torch.ones(count, dtype=torch.bool)
        if parity_only
        else torch.randint(0, 2, (count,), generator=generator).bool()
    )
    algorithm_targets = targets_for(world, bits, tasks, seed)
    targets = torch.where(route, algorithm_targets, bits[:, 0].long())
    return (
        encode_algorithm(bits, tasks, route, layout, device),
        targets.to(device),
        route.to(device),
        tasks.to(device),
    )


def rms_scale(norm_squared: float) -> float:
    return math.sqrt(HIDDEN / (norm_squared + RMS_EPSILON * HIDDEN))


@torch.no_grad()
def compile_interpreter(
    model: SharedInterpreterLM,
    recovered_supports: Tensor,
    recovered_families: Tensor,
    accepted: Tensor,
    layout: TokenLayout,
) -> tuple[FrozenEntries, dict[str, object]]:
    named = dict(model.named_parameters())
    masks = {name: torch.zeros_like(parameter, dtype=torch.bool) for name, parameter in named.items()}

    def fix(name: str, index: object, value: float = 0.0) -> None:
        named[name][index] = value
        masks[name][index] = True

    fix("token.weight", (slice(None), slice(0, PROGRAM_DIMS)))
    for token_id in layout.all_special:
        fix("token.weight", (token_id, slice(None)))
    for task, token_id in enumerate(layout.rule):
        if not accepted[task]:
            continue
        signs = recovered_supports[task].float().mul(2.0).sub(1.0)
        named["token.weight"][token_id, SUPPORT_START : SUPPORT_START + INPUT_DIM] = signs.to(
            model.token.weight.device
        )
        named["token.weight"][token_id, FAMILY_START + recovered_families[task]] = 1.0
        named["token.weight"][token_id, TASK] = 1.0
    for position, token_id in enumerate(layout.zero):
        named["token.weight"][token_id, SUPPORT_START + position] = 1.0
        named["token.weight"][token_id, BIT_TYPE] = 1.0
        named["token.weight"][token_id, CONST] = 1.0
    for position, token_id in enumerate(layout.one):
        named["token.weight"][token_id, SUPPORT_START + position] = 1.0
        named["token.weight"][token_id, BIT_TYPE] = 1.0
        named["token.weight"][token_id, CONST] = -1.0
    named["token.weight"][layout.parity_query, CONST] = 1.0
    named["token.weight"][layout.parity_query, PARITY_ROUTE] = 1.0
    named["token.weight"][layout.copy_query, CONST] = 1.0
    named["token.weight"][layout.copy_query, COPY_ROUTE] = 1.0
    named["token.weight"][layout.result_zero, OUT] = 30.0
    named["token.weight"][layout.result_one, OUT] = -30.0

    program_rows = slice(0, PROGRAM_HEADS * HEAD_DIM)
    language_rows = slice(PROGRAM_HEADS * HEAD_DIM, HIDDEN)
    for layer in range(LAYERS):
        prefix = f"blocks.{layer}."
        for norm in ("attention_norm.weight", "ffn_norm.weight"):
            fix(prefix + norm, slice(0, PROGRAM_DIMS), 1.0)
        for projection in ("attention.q.weight", "attention.k.weight", "attention.v.weight"):
            fix(prefix + projection, (program_rows, slice(None)))
            fix(prefix + projection, (language_rows, slice(0, PROGRAM_DIMS)))
        fix(prefix + "attention.o.weight", (slice(0, PROGRAM_DIMS), slice(None)))
        fix(prefix + "attention.o.weight", (slice(PROGRAM_DIMS, None), program_rows))
        for projection in ("gate.weight", "up.weight"):
            fix(prefix + projection, (slice(0, PROGRAM_CHANNELS), slice(None)))
            fix(prefix + projection, (slice(PROGRAM_CHANNELS, None), slice(0, PROGRAM_DIMS)))
        fix(prefix + "down.weight", (slice(0, PROGRAM_DIMS), slice(None)))
        fix(prefix + "down.weight", (slice(PROGRAM_DIMS, None), slice(0, PROGRAM_CHANNELS)))
    fix("final_norm.weight", slice(0, PROGRAM_DIMS), 1.0)

    # Block 0 copies support+family to parity queries, selects bits for copy
    # queries, and keeps bit-position symbols at constant amplitude.
    block0 = model.blocks[0]
    alpha_task = rms_scale(INPUT_DIM + 3.0)
    alpha_bit = rms_scale(3.0)
    alpha_query = rms_scale(2.0)
    q_a = math.sqrt(math.sqrt(HEAD_DIM) * 30.0) / alpha_bit
    q_type = math.sqrt(math.sqrt(HEAD_DIM) * 30.0) / alpha_bit
    q_task = math.sqrt(math.sqrt(HEAD_DIM) * 60.0 / (alpha_query * alpha_task))
    q_copy = math.sqrt(math.sqrt(HEAD_DIM) * 60.0 / (alpha_query * alpha_bit))
    for component in range(INPUT_DIM):
        block0.attention.q.weight[component, SUPPORT_START + component] = q_a
        block0.attention.k.weight[component, SUPPORT_START + component] = q_a
    block0.attention.q.weight[16, BIT_TYPE] = q_type
    block0.attention.k.weight[16, BIT_TYPE] = q_type
    block0.attention.q.weight[17, PARITY_ROUTE] = q_task
    block0.attention.k.weight[17, TASK] = q_task
    block0.attention.q.weight[18, COPY_ROUTE] = q_copy
    block0.attention.k.weight[18, BIT_TYPE] = q_copy
    for component in range(INPUT_DIM + len(FAMILY_NAMES)):
        source = SUPPORT_START + component
        block0.attention.v.weight[component, source] = 1.0
        block0.attention.o.weight[source, component] = 1.0 / alpha_bit

    task_amplitude = 1.0 + alpha_task / alpha_bit
    bit_amplitude = 2.0
    query_amplitude = alpha_task / alpha_bit
    alpha_task_1 = rms_scale((INPUT_DIM + 1) * task_amplitude**2 + 1.0)
    alpha_bit_1 = rms_scale(bit_amplitude**2 + 2.0)
    alpha_query_1 = rms_scale((INPUT_DIM + 1) * query_amplitude**2 + 2.0)

    # Block 1 reduces selected bit signs and preserves a copy path.
    block1 = model.blocks[1]
    q_select = math.sqrt(
        math.sqrt(HEAD_DIM)
        * 30.0
        / (alpha_query_1 * alpha_bit_1 * query_amplitude * bit_amplitude)
    )
    for component in range(INPUT_DIM):
        block1.attention.q.weight[component, SUPPORT_START + component] = q_select
        block1.attention.k.weight[component, SUPPORT_START + component] = q_select
    raw_task = (
        q_select**2
        * alpha_query_1
        * alpha_task_1
        * query_amplitude
        * task_amplitude
        * INPUT_DIM
        / math.sqrt(HEAD_DIM)
    )
    raw_self = (
        q_select**2
        * alpha_query_1**2
        * query_amplitude**2
        * INPUT_DIM
        / math.sqrt(HEAD_DIM)
    )
    block1.attention.q.weight[63, PARITY_ROUTE] = 1.0
    block1.attention.k.weight[63, TASK] = (
        -math.sqrt(HEAD_DIM) * (raw_task + 120.0) / (alpha_query_1 * alpha_task_1)
    )
    block1.attention.k.weight[63, PARITY_ROUTE] = (
        -math.sqrt(HEAD_DIM) * (raw_self + 120.0) / alpha_query_1**2
    )
    block1.attention.v.weight[0, CONST] = 1.0
    block1.attention.o.weight[SUM, 0] = SUPPORT_SIZE / alpha_bit_1

    copy_head = HEAD_DIM
    strong = 20.0
    block1.attention.q.weight[copy_head, COPY_ROUTE] = strong
    block1.attention.k.weight[copy_head, SUPPORT_START] = strong
    block1.attention.q.weight[copy_head + 1, COPY_ROUTE] = strong
    block1.attention.k.weight[copy_head + 1, BIT_TYPE] = strong
    block1.attention.v.weight[copy_head, CONST] = 1.0
    block1.attention.o.weight[COPY_SUM, copy_head] = 1.0 / alpha_bit_1

    # Block 2 holds four fixed count tables, selected by the discovered family.
    block2 = model.blocks[2]
    beta = 16.0
    channel = 0
    truth_tables: dict[str, list[int]] = {}
    for family, name in enumerate(FAMILY_NAMES):
        table: list[int] = []
        for count in range(SUPPORT_SIZE + 1):
            label = bool(family_labels(torch.tensor([count]), family).item())
            sign = -1.0 if label else 1.0
            table.append(int(label))
            integer_sum = SUPPORT_SIZE - 2 * count
            for threshold, coefficient in (
                (integer_sum - 1, 1.0),
                (integer_sum, -2.0),
                (integer_sum + 1, 1.0),
            ):
                block2.gate.weight[channel, SUM] = beta
                block2.gate.weight[channel, CONST] = -beta * threshold
                block2.up.weight[channel, FAMILY_START + family] = 1.0 / beta
                block2.down.weight[OUT, channel] = sign * coefficient
                channel += 1
        truth_tables[name] = table
    block2.gate.weight[channel, COPY_SUM] = 1.0
    block2.gate.weight[channel + 1, COPY_SUM] = -1.0
    block2.up.weight[channel, COPY_ROUTE] = 1.0
    block2.up.weight[channel + 1, COPY_ROUTE] = 1.0
    block2.down.weight[OUT, channel] = 1.0
    block2.down.weight[OUT, channel + 1] = -1.0
    used_channels = channel + 2

    values = {name: parameter.detach().clone() for name, parameter in named.items()}
    frozen = FrozenEntries(masks, values, sum(int(mask.sum().item()) for mask in masks.values()))
    frozen.enforce(model)
    fixed_nonzero = sum(
        int(((values[name] != 0) & mask).sum().item()) for name, mask in masks.items()
    )
    return frozen, {
        "structured_tasks": STRUCTURED_TASKS,
        "random_tasks": RANDOM_TASKS,
        "accepted_tasks": int(accepted.sum().item()),
        "accepted_random_tasks": int(accepted[STRUCTURED_TASKS:].sum().item()),
        "exact_supports": int(
            (recovered_supports[:STRUCTURED_TASKS] == make_world().supports[:STRUCTURED_TASKS])
            .all(1)
            .sum()
            .item()
        ),
        "exact_families": int(
            (recovered_families[:STRUCTURED_TASKS] == make_world().families[:STRUCTURED_TASKS])
            .sum()
            .item()
        ),
        "description_entries": STRUCTURED_TASKS * (INPUT_DIM + len(FAMILY_NAMES)),
        "program_hidden_coordinates": PROGRAM_DIMS,
        "program_attention_heads": PROGRAM_HEADS,
        "program_ffn_channels": PROGRAM_CHANNELS,
        "used_decoder_channels": used_channels,
        "shared_width_independent_of_tasks": True,
        "frozen_parameter_entries": frozen.count,
        "fixed_nonzero_entries": fixed_nonzero,
        "truth_tables": truth_tables,
    }


@torch.no_grad()
def evaluate_algorithm(
    model: SharedInterpreterLM,
    world: DiscoveryWorld,
    layout: TokenLayout,
    device: torch.device,
    seed: int,
    examples_per_task: int = 128,
) -> dict[str, object]:
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tasks = torch.arange(TASKS).repeat_interleave(examples_per_task)
    bits = torch.randint(0, 2, (len(tasks), INPUT_DIM), generator=generator, dtype=torch.uint8)
    route = torch.ones(len(tasks), dtype=torch.bool)
    targets = targets_for(world, bits, tasks, seed)
    tokens = encode_algorithm(bits, tasks, route, layout, device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model.binary_logits(tokens, layout).float()
    correct = (logits.argmax(-1).cpu() == targets).view(TASKS, examples_per_task).float().mean(1)
    per_family: dict[str, object] = {}
    for family, name in enumerate(FAMILY_NAMES):
        values = correct[
            family * TASKS_PER_FAMILY : (family + 1) * TASKS_PER_FAMILY
        ]
        per_family[name] = {
            "mean": float(values.mean().item()),
            "min": float(values.min().item()),
            "max": float(values.max().item()),
        }
    random_values = correct[STRUCTURED_TASKS:]

    copy_tokens, _, _, _ = make_algorithm_batch(
        world, layout, 4096, seed + 1, device, parity_only=False
    )
    copy_tokens[:, -1] = layout.copy_query
    copy_targets = (copy_tokens[:, 1].cpu() != layout.zero[0]).long().to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        copy_logits = model.binary_logits(copy_tokens, layout).float()
    return {
        "per_family": per_family,
        "structured_mean": float(correct[:STRUCTURED_TASKS].mean().item()),
        "structured_min": float(correct[:STRUCTURED_TASKS].min().item()),
        "random_mean": float(random_values.mean().item()),
        "random_min": float(random_values.min().item()),
        "random_max": float(random_values.max().item()),
        "per_task": correct.tolist(),
        "protected_copy": float((copy_logits.argmax(-1) == copy_targets).float().mean().item()),
    }


def scheduled_learning_rate(step: int) -> float:
    return t3.scheduled_learning_rate(step)


def train_arm(
    arm: str,
    model_seed: int,
    world: DiscoveryWorld,
    recovered_supports: Tensor,
    recovered_families: Tensor,
    accepted: Tensor,
    layout: TokenLayout,
    natural: TokenStream,
    validation: TokenStream,
    device: torch.device,
    steps: int,
) -> dict[str, object]:
    model = t3.build_model(model_seed, device)
    initial_hash = t3.state_sha256(model)
    frozen: FrozenEntries | None = None
    compiler: dict[str, object] | None = None
    if arm == "compiler_1x":
        frozen, compiler = compile_interpreter(
            model, recovered_supports, recovered_families, accepted, layout
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=PEAK_LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.0
    )
    flat_bits = world.prefix_bits.reshape(-1, INPUT_DIM)
    flat_labels = world.prefix_labels.reshape(-1)
    flat_tasks = torch.arange(TASKS).repeat_interleave(PREFIX_PER_TASK)
    prefix_losses: list[float] = []
    if arm != "compiler_1x":
        permutation = torch.randperm(
            len(flat_bits), generator=torch.Generator().manual_seed(model_seed + 50_000)
        )
        for prefix_step, start in enumerate(range(0, len(permutation), ALGORITHM_BATCH), 1):
            indices = permutation[start : start + ALGORITHM_BATCH]
            tokens = encode_algorithm(
                flat_bits[indices],
                flat_tasks[indices],
                torch.ones(len(indices), dtype=torch.bool),
                layout,
                device,
            )
            targets = flat_labels[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model.binary_logits(tokens, layout)
            loss = F.cross_entropy(logits.float(), targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite {arm} prefix loss at update {prefix_step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for group in optimizer.param_groups:
                group["lr"] = scheduled_learning_rate(min(prefix_step, STEPS_2X))
            optimizer.step()
            prefix_losses.append(float(loss.item()))

    checkpoints: dict[str, object] = {}
    maximum_gradient_norm = 0.0
    maximum_loss = max(prefix_losses, default=0.0)
    natural_tokens = 0
    algorithm_tokens = len(flat_bits) * (INPUT_DIM + 2)
    started = time.perf_counter()
    checkpoint_set = set(CHECKPOINTS_1X) | ({STEPS_2X} if steps == STEPS_2X else set())
    for step in range(1, steps + 1):
        model.train()
        for group in optimizer.param_groups:
            group["lr"] = scheduled_learning_rate(step)
        optimizer.zero_grad(set_to_none=True)
        if step % ALGORITHM_INTERVAL == 0:
            tokens, targets, _, _ = make_algorithm_batch(
                world, layout, ALGORITHM_BATCH, model_seed * 1_000_003 + step, device
            )
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model.binary_logits(tokens, layout)
            loss = F.cross_entropy(logits.float(), targets)
            algorithm_tokens += ALGORITHM_BATCH * (INPUT_DIM + 2)
        else:
            inputs, targets = natural.batch(
                model_seed * 1_000_003 + step, NATURAL_BATCH, CONTEXT, device
            )
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(inputs)
            loss = F.cross_entropy(logits.float().reshape(-1, VOCAB), targets.reshape(-1))
            natural_tokens += NATURAL_BATCH * CONTEXT
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite {arm} loss at step {step}")
        loss.backward()
        if frozen is not None:
            frozen.mask_gradients(model)
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        if not math.isfinite(gradient_norm):
            raise RuntimeError(f"nonfinite {arm} gradient at step {step}")
        optimizer.step()
        if frozen is not None:
            frozen.enforce(model)
        maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)
        maximum_loss = max(maximum_loss, float(loss.item()))
        if step in checkpoint_set:
            checkpoints[str(step)] = {
                "natural": t3.evaluate_natural(
                    model, validation, device, model_seed + 700_000, batches=32
                ),
                "algorithm": evaluate_algorithm(
                    model, world, layout, device, model_seed + 800_000 + step
                ),
                "natural_tokens": natural_tokens,
                "algorithm_tokens": algorithm_tokens,
            }
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {
        "arm": arm,
        "steps": steps,
        "initial_hash": initial_hash,
        "compiler": compiler,
        "prefix_updates": 0 if compiler else len(prefix_losses),
        "prefix_loss_first": prefix_losses[0] if prefix_losses else None,
        "prefix_loss_last": prefix_losses[-1] if prefix_losses else None,
        "checkpoints": checkpoints,
        "maximum_gradient_norm": maximum_gradient_norm,
        "maximum_loss": maximum_loss,
        "elapsed_seconds": time.perf_counter() - started,
        "terminal_hash": t3.state_sha256(model),
    }


def run(device: torch.device, quick: bool = False) -> dict[str, object]:
    natural = TokenStream(t3.TRAIN_FILE)
    validation = TokenStream(t3.VALIDATION_FILE)
    layout, unused_count = choose_token_layout(natural, validation)
    world = make_world()
    recovered_supports, recovered_families, accepted = discover(world)
    exact_supports = bool(
        torch.equal(
            recovered_supports[:STRUCTURED_TASKS], world.supports[:STRUCTURED_TASKS]
        )
    )
    exact_families = bool(
        torch.equal(
            recovered_families[:STRUCTURED_TASKS], world.families[:STRUCTURED_TASKS]
        )
    )
    zero_false_accepts = not bool(accepted[STRUCTURED_TASKS:].any())

    worlds: dict[str, object] = {}
    if quick:
        model = t3.build_model(MODEL_SEEDS[0], device)
        initial_hash = t3.state_sha256(model)
        frozen, compiler = compile_interpreter(
            model, recovered_supports, recovered_families, accepted, layout
        )
        frozen.enforce(model)
        worlds[str(MODEL_SEEDS[0])] = {
            "initial_hash": initial_hash,
            "compiler": compiler,
            "algorithm": evaluate_algorithm(
                model, world, layout, device, MODEL_SEEDS[0] + 900_000, examples_per_task=64
            ),
        }
    else:
        for seed in MODEL_SEEDS:
            baseline = train_arm(
                "baseline_1x",
                seed,
                world,
                recovered_supports,
                recovered_families,
                accepted,
                layout,
                natural,
                validation,
                device,
                CHECKPOINTS_1X[-1],
            )
            candidate = train_arm(
                "compiler_1x",
                seed,
                world,
                recovered_supports,
                recovered_families,
                accepted,
                layout,
                natural,
                validation,
                device,
                CHECKPOINTS_1X[-1],
            )
            doubled = train_arm(
                "baseline_2x",
                seed,
                world,
                recovered_supports,
                recovered_families,
                accepted,
                layout,
                natural,
                validation,
                device,
                STEPS_2X,
            )
            worlds[str(seed)] = {
                "arms": {
                    "baseline_1x": baseline,
                    "compiler_1x": candidate,
                    "baseline_2x": doubled,
                },
                "identical_initial_hashes": len(
                    {baseline["initial_hash"], candidate["initial_hash"], doubled["initial_hash"]}
                )
                == 1,
            }

    gates: dict[str, bool] = {}
    if quick:
        result = next(iter(worlds.values()))
        algorithm = result["algorithm"]
        gates = {
            "exact_supports": exact_supports,
            "exact_families": exact_families,
            "zero_random_accepts": zero_false_accepts,
            "all_family_means_ge_99": all(
                values["mean"] >= 0.99 for values in algorithm["per_family"].values()
            ),
            "all_family_mins_ge_99": all(
                values["min"] >= 0.99 for values in algorithm["per_family"].values()
            ),
            "random_between_40_60": 0.40 <= algorithm["random_mean"] <= 0.60,
            "copy_ge_99": algorithm["protected_copy"] >= 0.99,
        }
    else:
        for seed, result in worlds.items():
            arms = result["arms"]
            baseline = arms["baseline_1x"]
            candidate = arms["compiler_1x"]
            doubled = arms["baseline_2x"]
            gates[f"{seed}_same_initialization"] = result["identical_initial_hashes"]
            gates[f"{seed}_exact_supports"] = exact_supports
            gates[f"{seed}_exact_families"] = exact_families
            gates[f"{seed}_zero_random_accepts"] = zero_false_accepts
            for checkpoint in CHECKPOINTS_1X:
                key = str(checkpoint)
                algorithm = candidate["checkpoints"][key]["algorithm"]
                gates[f"{seed}_{key}_all_family_means_ge_99"] = all(
                    values["mean"] >= 0.99 for values in algorithm["per_family"].values()
                )
                gates[f"{seed}_{key}_all_family_mins_ge_99"] = all(
                    values["min"] >= 0.99 for values in algorithm["per_family"].values()
                )
                gates[f"{seed}_{key}_random_between_40_60"] = (
                    0.40 <= algorithm["random_mean"] <= 0.60
                )
                gates[f"{seed}_{key}_copy_ge_99"] = algorithm["protected_copy"] >= 0.99
                candidate_nll = candidate["checkpoints"][key]["natural"]["nll"]
                baseline_nll = baseline["checkpoints"][key]["natural"]["nll"]
                gates[f"{seed}_{key}_natural_within_0p5pct"] = candidate_nll <= 1.005 * baseline_nll
            doubled_algorithm = doubled["checkpoints"][str(STEPS_2X)]["algorithm"]
            gates[f"{seed}_baseline2x_parity_lt_80"] = (
                doubled_algorithm["per_family"]["parity"]["mean"] < 0.80
            )
            gates[f"{seed}_baseline2x_mod3_lt_80"] = (
                doubled_algorithm["per_family"]["mod3"]["mean"] < 0.80
            )
            gates[f"{seed}_baseline2x_copy_ge_95"] = doubled_algorithm["protected_copy"] >= 0.95
            gates[f"{seed}_candidate_terminal_improves"] = (
                candidate["checkpoints"]["1000"]["natural"]["nll"]
                <= candidate["checkpoints"]["500"]["natural"]["nll"]
            )
            gates[f"{seed}_finite_training"] = all(
                math.isfinite(arm["maximum_gradient_norm"])
                and math.isfinite(arm["maximum_loss"])
                and arm["maximum_loss"] < 100.0
                for arm in (baseline, candidate, doubled)
            )

    return {
        "schema": "heterogeneous-family-discovery-t6-v1",
        "status": "quick" if quick else "complete",
        "device": str(device),
        "torch_version": torch.__version__,
        "configuration": {
            "vocab": VOCAB,
            "hidden": HIDDEN,
            "layers": LAYERS,
            "heads": t3.HEADS,
            "ffn_width": FFN_WIDTH,
            "parameter_count": t3.parameter_count(),
            "tasks": TASKS,
            "structured_tasks": STRUCTURED_TASKS,
            "random_tasks": RANDOM_TASKS,
            "families": FAMILY_NAMES,
            "input_dim": INPUT_DIM,
            "support_size": SUPPORT_SIZE,
            "prefix_per_task": PREFIX_PER_TASK,
            "candidate_supports": len(candidate_supports()),
            "unused_tokens_available": unused_count,
            "selected_special_tokens": layout.all_special,
            "model_seeds": MODEL_SEEDS,
        },
        "discovery": {
            "exact_supports": exact_supports,
            "exact_families": exact_families,
            "accepted_structured": int(accepted[:STRUCTURED_TASKS].sum().item()),
            "accepted_random": int(accepted[STRUCTURED_TASKS:].sum().item()),
        },
        "integrity": {
            "source_sha256": sha256_file(Path(__file__)),
            "test_sha256": sha256_file(TEST_SOURCE),
            "preregistration_sha256": sha256_file(PREREGISTRATION),
            "predecessor_sha256": sha256_file(PREDECESSOR),
            "train_sha256": sha256_file(t3.TRAIN_FILE),
            "validation_sha256": sha256_file(t3.VALIDATION_FILE),
        },
        "worlds": worlds,
        "gates": gates,
        "all_gates_pass": bool(gates) and all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    result = run(torch.device(arguments.device), quick=arguments.quick)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(arguments.output), "all_gates_pass": result["all_gates_pass"]}, indent=2))


if __name__ == "__main__":
    main()
