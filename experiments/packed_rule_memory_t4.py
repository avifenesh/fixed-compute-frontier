#!/usr/bin/env python3
"""T4: active nibble-packed rule memory inside a fixed causal LM."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import heterogeneous_algebraic_compilation_t1 as t1
from experiments import shared_algebraic_interpreter_t3 as t3


PREREGISTRATION = ROOT / "results/packed-rule-memory-t4-preregistration.md"
TEST_SOURCE = ROOT / "tests/test_packed_rule_memory_t4.py"
PREDECESSOR = ROOT / "results/shared-algebraic-interpreter-t3.json"
OUTPUT = ROOT / "results/packed-rule-memory-t4.json"

VOCAB = t3.VOCAB
HIDDEN = t3.HIDDEN
LAYERS = t3.LAYERS
HEADS = t3.HEADS
HEAD_DIM = t3.HEAD_DIM
FFN_WIDTH = t3.FFN_WIDTH
CONTEXT = t3.CONTEXT
INPUT_DIM = 32
RULES = 768
SUPPORT_SIZE = 16
PREFIX_PER_RULE = 64
PROGRAM_DIMS = 49
PROGRAM_HEADS = 2
DECODE_CHANNELS = 160
EXECUTE_CHANNELS = 64
NATURAL_BATCH = 16
ALGORITHM_BATCH = 64
ALGORITHM_INTERVAL = 20
CHECKPOINTS_1X = (250, 500, 1000)
STEPS_2X = 2000
MODEL_SEEDS = (1601, 1877)
PEAK_LEARNING_RATE = t3.PEAK_LEARNING_RATE
WARMUP_STEPS = t3.WARMUP_STEPS
SCHEDULE_STEPS = t3.SCHEDULE_STEPS
RMS_EPSILON = t3.RMS_EPSILON

NIBBLE_START = 0
NIBBLES = 8
REFERENCE = 8
SUPPORT_START = 9
SUM = 41
COPY_SUM = 42
CONST = 43
TASK = 44
BIT_TYPE = 45
PARITY = 46
COPY = 47
OUT = 48

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
    required = RULES + 2 * INPUT_DIM + 4
    unused = np.flatnonzero(counts == 0)
    if len(unused) < required:
        raise RuntimeError(f"need {required} unused token IDs, found {len(unused)}")
    selected = tuple(int(value) for value in unused[:required])
    cursor = 0
    rule = selected[cursor : cursor + RULES]
    cursor += RULES
    zero = selected[cursor : cursor + INPUT_DIM]
    cursor += INPUT_DIM
    one = selected[cursor : cursor + INPUT_DIM]
    cursor += INPUT_DIM
    return TokenLayout(rule, zero, one, *selected[cursor : cursor + 4]), len(unused)


def make_supports(seed: int = 103_001) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    supports = torch.zeros(RULES, INPUT_DIM, dtype=torch.uint8)
    seen: set[tuple[int, ...]] = set()
    for rule in range(RULES):
        while True:
            selected = torch.randperm(INPUT_DIM, generator=generator)[:SUPPORT_SIZE]
            row = torch.zeros(INPUT_DIM, dtype=torch.uint8)
            row[selected] = 1
            signature = tuple(row.tolist())
            if signature not in seen:
                seen.add(signature)
                supports[rule] = row
                break
    return supports


def pack_supports(supports: Tensor) -> Tensor:
    weights = torch.tensor((1, 2, 4, 8), dtype=torch.int16)
    return (supports.view(RULES, NIBBLES, 4).to(torch.int16) * weights).sum(-1).to(torch.uint8)


def unpack_supports(packed: Tensor) -> Tensor:
    bits = [packed.bitwise_right_shift(bit).bitwise_and(1) for bit in range(4)]
    return torch.stack(bits, dim=-1).reshape(RULES, INPUT_DIM).to(torch.uint8)


def make_prefix(supports: Tensor, seed: int = 113_003) -> tuple[Tensor, Tensor, Tensor]:
    all_bits: list[Tensor] = []
    all_targets: list[Tensor] = []
    all_rules: list[Tensor] = []
    for rule in range(RULES):
        generator = torch.Generator(device="cpu").manual_seed(seed + rule)
        while True:
            bits = torch.randint(
                0, 2, (PREFIX_PER_RULE, INPUT_DIM), generator=generator, dtype=torch.uint8
            )
            if int(torch.linalg.matrix_rank(bits.float()).item()) == INPUT_DIM:
                break
        targets = (bits.to(torch.int16) @ supports[rule].to(torch.int16)).remainder(2).long()
        all_bits.append(bits)
        all_targets.append(targets)
        all_rules.append(torch.full((PREFIX_PER_RULE,), rule, dtype=torch.long))
    return torch.cat(all_bits), torch.cat(all_targets), torch.cat(all_rules)


def solve_prefix(bits: Tensor, targets: Tensor, rules: Tensor) -> tuple[Tensor, int]:
    recovered = torch.zeros(RULES, INPUT_DIM, dtype=torch.uint8)
    operations = 0
    for rule in range(RULES):
        mask = rules == rule
        solution, rank, xor_ops = t1.solve_gf2(bits[mask], targets[mask].to(torch.uint8))
        if solution is None or rank != INPUT_DIM:
            raise RuntimeError(f"rule {rule} solve failed at rank {rank}")
        recovered[rule] = solution
        operations += xor_ops
    return recovered, operations


def encode_algorithm(
    bits: Tensor,
    rules: Tensor,
    parity_route: Tensor,
    layout: TokenLayout,
    device: torch.device,
) -> Tensor:
    count = bits.shape[0]
    zero = torch.tensor(layout.zero, dtype=torch.long)
    one = torch.tensor(layout.one, dtype=torch.long)
    bit_tokens = torch.where(bits.bool(), one[None, :], zero[None, :])
    rule_tokens = torch.tensor(layout.rule, dtype=torch.long)[rules]
    queries = torch.where(
        parity_route,
        torch.full((count,), layout.parity_query, dtype=torch.long),
        torch.full((count,), layout.copy_query, dtype=torch.long),
    )
    return torch.cat((rule_tokens[:, None], bit_tokens, queries[:, None]), dim=1).to(device)


def make_algorithm_batch(
    supports: Tensor,
    layout: TokenLayout,
    count: int,
    seed: int,
    device: torch.device,
    *,
    parity_only: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rules = torch.randint(0, RULES, (count,), generator=generator)
    bits = torch.randint(0, 2, (count, INPUT_DIM), generator=generator, dtype=torch.uint8)
    parity_route = (
        torch.ones(count, dtype=torch.bool)
        if parity_only
        else torch.randint(0, 2, (count,), generator=generator).bool()
    )
    parity = (bits.to(torch.int16) * supports[rules].to(torch.int16)).sum(1).remainder(2).long()
    targets = torch.where(parity_route, parity, bits[:, 0].long())
    return (
        encode_algorithm(bits, rules, parity_route, layout, device),
        targets.to(device),
        parity_route.to(device),
        rules.to(device),
    )


def rms_scale(norm_squared: float) -> float:
    return math.sqrt(HIDDEN / (norm_squared + RMS_EPSILON * HIDDEN))


def bit_sign(value: int, bit: int) -> float:
    return 1.0 if ((value >> bit) & 1) else -1.0


@torch.no_grad()
def compile_packed_interpreter(
    model: SharedInterpreterLM,
    recovered: Tensor,
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

    packed = pack_supports(recovered).to(model.token.weight.device)
    for rule, token_id in enumerate(layout.rule):
        named["token.weight"][token_id, NIBBLE_START : NIBBLE_START + NIBBLES] = packed[
            rule
        ].float()
        named["token.weight"][token_id, REFERENCE] = 1.0
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
    named["token.weight"][layout.parity_query, PARITY] = 1.0
    named["token.weight"][layout.copy_query, CONST] = 1.0
    named["token.weight"][layout.copy_query, COPY] = 1.0
    named["token.weight"][layout.result_zero, OUT] = 30.0
    named["token.weight"][layout.result_one, OUT] = -30.0

    # Temporally isolate only the four blocks that execute the packed program.
    program_rows = slice(0, PROGRAM_HEADS * HEAD_DIM)
    language_rows = slice(PROGRAM_HEADS * HEAD_DIM, HIDDEN)
    for layer in range(4):
        prefix = f"blocks.{layer}."
        channels = DECODE_CHANNELS if layer == 0 else EXECUTE_CHANNELS
        for norm in ("attention_norm.weight", "ffn_norm.weight"):
            fix(prefix + norm, slice(0, PROGRAM_DIMS), 1.0)
        for projection in ("attention.q.weight", "attention.k.weight", "attention.v.weight"):
            fix(prefix + projection, (program_rows, slice(None)))
            fix(prefix + projection, (language_rows, slice(0, PROGRAM_DIMS)))
        fix(prefix + "attention.o.weight", (slice(0, PROGRAM_DIMS), slice(None)))
        fix(prefix + "attention.o.weight", (slice(PROGRAM_DIMS, None), program_rows))
        for projection in ("gate.weight", "up.weight"):
            fix(prefix + projection, (slice(0, channels), slice(None)))
            fix(prefix + projection, (slice(channels, None), slice(0, PROGRAM_DIMS)))
        fix(prefix + "down.weight", (slice(0, PROGRAM_DIMS), slice(None)))
        fix(prefix + "down.weight", (slice(PROGRAM_DIMS, None), slice(0, channels)))

    # Once OUT exists, later blocks may reuse every other coordinate.
    for layer in range(4, LAYERS):
        fix(f"blocks.{layer}.attention.o.weight", (OUT, slice(None)))
        fix(f"blocks.{layer}.down.weight", (OUT, slice(None)))
    fix("final_norm.weight", OUT, 1.0)

    # Block 0 FFN: unpack every nibble.  Four output-bit functions share the
    # same 14 hinges, so the entire decoder needs only 130 active channels.
    block0 = model.blocks[0]
    beta = 16.0
    channel = 0
    # Shared constant -1 for all decoded support bits.
    block0.gate.weight[channel, REFERENCE] = 1.0
    block0.gate.weight[channel + 1, REFERENCE] = -1.0
    block0.up.weight[channel, TASK] = 1.0
    block0.up.weight[channel + 1, TASK] = 1.0
    for output in range(INPUT_DIM):
        block0.down.weight[SUPPORT_START + output, channel] = -1.0
        block0.down.weight[SUPPORT_START + output, channel + 1] = 1.0
    channel += 2
    for nibble in range(NIBBLES):
        # One exact z*task pair is shared by all four decoded bits.
        block0.gate.weight[channel, NIBBLE_START + nibble] = 1.0
        block0.gate.weight[channel + 1, NIBBLE_START + nibble] = -1.0
        block0.up.weight[channel, TASK] = 1.0
        block0.up.weight[channel + 1, TASK] = 1.0
        for bit in range(4):
            slope0 = bit_sign(1, bit) - bit_sign(0, bit)
            output = SUPPORT_START + 4 * nibble + bit
            block0.down.weight[output, channel] = slope0
            block0.down.weight[output, channel + 1] = -slope0
        channel += 2
        for knot in range(1, 15):
            block0.gate.weight[channel, NIBBLE_START + nibble] = beta
            block0.gate.weight[channel, REFERENCE] = -beta * knot
            block0.up.weight[channel, TASK] = 1.0 / beta
            for bit in range(4):
                previous_slope = bit_sign(knot, bit) - bit_sign(knot - 1, bit)
                next_slope = bit_sign(knot + 1, bit) - bit_sign(knot, bit)
                output = SUPPORT_START + 4 * nibble + bit
                block0.down.weight[output, channel] = next_slope - previous_slope
            channel += 1
    decoder_channels = channel
    if decoder_channels > DECODE_CHANNELS:
        raise RuntimeError("packed decoder exceeded reserved channels")

    # Compute the exact per-rule normalized support amplitude produced by the
    # shared decoder; it is positive and may vary with the packed digits.
    packed_float = packed.float().cpu()
    input_norms = packed_float.pow(2).sum(1) + 2.0
    decoded_amplitude = HIDDEN / (input_norms + RMS_EPSILON * HIDDEN)
    task_norms_1 = input_norms + INPUT_DIM * decoded_amplitude.pow(2)
    alpha_task_1 = torch.sqrt(HIDDEN / (task_norms_1 + RMS_EPSILON * HIDDEN))

    # Block 1: bit tokens self-attend; parity queries copy the decoded support.
    block1 = model.blocks[1]
    alpha_bit_1 = rms_scale(3.0)
    alpha_query_1 = rms_scale(2.0)
    gap = 30.0
    base = 30.0
    query_gap = 60.0
    q_a = math.sqrt(math.sqrt(HEAD_DIM) * gap) / alpha_bit_1
    q_type = math.sqrt(math.sqrt(HEAD_DIM) * base) / alpha_bit_1
    minimum_task_scale = float(alpha_task_1.min().item())
    q_query = math.sqrt(
        math.sqrt(HEAD_DIM) * query_gap / (alpha_query_1 * minimum_task_scale)
    )
    for component in range(INPUT_DIM):
        row = component
        block1.attention.q.weight[row, SUPPORT_START + component] = q_a
        block1.attention.k.weight[row, SUPPORT_START + component] = q_a
        block1.attention.v.weight[row, SUPPORT_START + component] = 1.0
        block1.attention.o.weight[SUPPORT_START + component, row] = 1.0 / alpha_bit_1
    block1.attention.q.weight[32, BIT_TYPE] = q_type
    block1.attention.k.weight[32, BIT_TYPE] = q_type
    block1.attention.q.weight[33, PARITY] = q_query
    block1.attention.k.weight[33, TASK] = q_query

    bit_amplitude = 2.0
    query_amplitude = alpha_task_1 * decoded_amplitude / alpha_bit_1
    task_amplitude = decoded_amplitude * (1.0 + alpha_task_1 / alpha_bit_1)

    # Block 2: support-conditioned selection, calibrated at the smallest rule
    # amplitude and protected against task/self keys at the largest amplitude.
    block2 = model.blocks[2]
    alpha_bit_2 = rms_scale(bit_amplitude**2 + 2.0)
    query_norms_2 = INPUT_DIM * query_amplitude.pow(2) + 2.0
    alpha_query_2 = torch.sqrt(HIDDEN / (query_norms_2 + RMS_EPSILON * HIDDEN))
    task_norms_2 = (
        input_norms + INPUT_DIM * task_amplitude.pow(2)
    )
    alpha_task_2 = torch.sqrt(HIDDEN / (task_norms_2 + RMS_EPSILON * HIDDEN))
    normalized_query_support = alpha_query_2 * query_amplitude
    minimum_query_support = float(normalized_query_support.min().item())
    selection_gap = 30.0
    q_select = math.sqrt(
        math.sqrt(HEAD_DIM)
        * selection_gap
        / (minimum_query_support * alpha_bit_2 * bit_amplitude)
    )
    for component in range(INPUT_DIM):
        block2.attention.q.weight[component, SUPPORT_START + component] = q_select
        block2.attention.k.weight[component, SUPPORT_START + component] = q_select

    raw_task = (
        q_select**2
        * alpha_query_2
        * alpha_task_2
        * query_amplitude
        * task_amplitude
        * INPUT_DIM
        / math.sqrt(HEAD_DIM)
    )
    raw_self = (
        q_select**2
        * alpha_query_2.pow(2)
        * query_amplitude.pow(2)
        * INPUT_DIM
        / math.sqrt(HEAD_DIM)
    )
    required_task_penalty = (
        (raw_task + 4 * selection_gap)
        * math.sqrt(HEAD_DIM)
        / (alpha_query_2 * alpha_task_2)
    ).max()
    required_self_penalty = (
        (raw_self + 4 * selection_gap)
        * math.sqrt(HEAD_DIM)
        / alpha_query_2.pow(2)
    ).max()
    block2.attention.q.weight[63, PARITY] = 1.0
    block2.attention.k.weight[63, TASK] = -float(required_task_penalty.item())
    block2.attention.k.weight[63, PARITY] = -float(required_self_penalty.item())
    block2.attention.v.weight[0, CONST] = 1.0
    block2.attention.o.weight[SUM, 0] = SUPPORT_SIZE / alpha_bit_2

    # Protected copy in head 1 selects the first position token.
    copy_head = HEAD_DIM
    strong = 20.0
    block2.attention.q.weight[copy_head, COPY] = strong
    block2.attention.k.weight[copy_head, SUPPORT_START] = strong
    block2.attention.q.weight[copy_head + 1, COPY] = strong
    block2.attention.k.weight[copy_head + 1, BIT_TYPE] = strong
    block2.attention.v.weight[copy_head, CONST] = 1.0
    block2.attention.o.weight[COPY_SUM, copy_head] = 1.0 / alpha_bit_2

    # Block 3 shared count decoder and exact protected-copy product.
    block3 = model.blocks[3]
    beta = 16.0
    channel = 0
    for integer_sum in range(-SUPPORT_SIZE, SUPPORT_SIZE + 1, 2):
        negatives = (SUPPORT_SIZE - integer_sum) // 2
        parity_sign = -1.0 if negatives % 2 else 1.0
        for threshold, coefficient in (
            (integer_sum - 1, 1.0),
            (integer_sum, -2.0),
            (integer_sum + 1, 1.0),
        ):
            block3.gate.weight[channel, SUM] = beta
            block3.gate.weight[channel, CONST] = -beta * threshold
            block3.up.weight[channel, PARITY] = 1.0 / beta
            block3.down.weight[OUT, channel] = parity_sign * coefficient
            channel += 1
    block3.gate.weight[channel, COPY_SUM] = 1.0
    block3.gate.weight[channel + 1, COPY_SUM] = -1.0
    block3.up.weight[channel, COPY] = 1.0
    block3.up.weight[channel + 1, COPY] = 1.0
    block3.down.weight[OUT, channel] = 1.0
    block3.down.weight[OUT, channel + 1] = -1.0
    execute_decoder_channels = channel + 2

    values = {name: parameter.detach().clone() for name, parameter in named.items()}
    frozen = FrozenEntries(masks, values, sum(int(mask.sum().item()) for mask in masks.values()))
    frozen.enforce(model)
    fixed_nonzero = sum(
        int(((values[name] != 0) & mask).sum().item()) for name, mask in masks.items()
    )
    return frozen, {
        "rules": RULES,
        "description_entries": RULES * NIBBLES,
        "bits_per_description_entry": 4,
        "decoded_support_bits": RULES * INPUT_DIM,
        "density_vs_t3_rule_table": 12.0,
        "program_hidden_coordinates": PROGRAM_DIMS,
        "decoder_channels": decoder_channels,
        "execute_decoder_channels": execute_decoder_channels,
        "shared_width_independent_of_rules": True,
        "frozen_parameter_entries": frozen.count,
        "fixed_nonzero_entries": fixed_nonzero,
        "decoded_amplitude_min": float(decoded_amplitude.min().item()),
        "decoded_amplitude_max": float(decoded_amplitude.max().item()),
        "minimum_normalized_query_support": minimum_query_support,
    }


@torch.no_grad()
def evaluate_algorithm(
    model: SharedInterpreterLM,
    supports: Tensor,
    layout: TokenLayout,
    device: torch.device,
    seed: int,
    examples_per_rule: int = 64,
) -> dict[str, object]:
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rules = torch.arange(RULES).repeat_interleave(examples_per_rule)
    bits = torch.randint(0, 2, (len(rules), INPUT_DIM), generator=generator, dtype=torch.uint8)
    route = torch.ones(len(rules), dtype=torch.bool)
    targets = (bits.to(torch.int16) * supports[rules].to(torch.int16)).sum(1).remainder(2).long()
    tokens = encode_algorithm(bits, rules, route, layout, device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model.binary_logits(tokens, layout).float()
    correct = logits.argmax(-1).cpu() == targets
    per_rule = correct.view(RULES, examples_per_rule).float().mean(1)

    copy_tokens, _, _, _ = make_algorithm_batch(
        supports, layout, 4096, seed + 1, device, parity_only=False
    )
    copy_tokens[:, -1] = layout.copy_query
    copy_targets = (copy_tokens[:, 1].cpu() != layout.zero[0]).long().to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        copy_logits = model.binary_logits(copy_tokens, layout).float()
    return {
        "parity_mean": float(per_rule.mean().item()),
        "parity_min": float(per_rule.min().item()),
        "parity_max": float(per_rule.max().item()),
        "per_rule": per_rule.tolist(),
        "protected_copy": float((copy_logits.argmax(-1) == copy_targets).float().mean().item()),
    }


def scheduled_learning_rate(step: int) -> float:
    return t3.scheduled_learning_rate(step)


def train_arm(
    arm: str,
    model_seed: int,
    supports: Tensor,
    prefix_bits: Tensor,
    prefix_targets: Tensor,
    prefix_rules: Tensor,
    recovered: Tensor,
    xor_ops: int,
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
        frozen, compiler = compile_packed_interpreter(model, recovered, layout)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=PEAK_LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.0
    )
    prefix_losses: list[float] = []
    if arm != "compiler_1x":
        permutation = torch.randperm(
            len(prefix_bits), generator=torch.Generator().manual_seed(model_seed + 50_000)
        )
        for prefix_step, start in enumerate(range(0, len(permutation), ALGORITHM_BATCH), 1):
            indices = permutation[start : start + ALGORITHM_BATCH]
            tokens = encode_algorithm(
                prefix_bits[indices],
                prefix_rules[indices],
                torch.ones(len(indices), dtype=torch.bool),
                layout,
                device,
            )
            targets = prefix_targets[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model.binary_logits(tokens, layout)
            loss = F.cross_entropy(logits.float(), targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite {arm} prefix loss at update {prefix_step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for group in optimizer.param_groups:
                group["lr"] = scheduled_learning_rate(min(prefix_step, SCHEDULE_STEPS))
            optimizer.step()
            prefix_losses.append(float(loss.item()))

    checkpoints: dict[str, object] = {}
    maximum_gradient_norm = 0.0
    maximum_loss = max(prefix_losses, default=0.0)
    natural_tokens = 0
    algorithm_tokens = len(prefix_bits) * (INPUT_DIM + 2)
    started = time.perf_counter()
    checkpoint_set = set(CHECKPOINTS_1X) | ({STEPS_2X} if steps == STEPS_2X else set())
    for step in range(1, steps + 1):
        model.train()
        for group in optimizer.param_groups:
            group["lr"] = scheduled_learning_rate(step)
        optimizer.zero_grad(set_to_none=True)
        if step % ALGORITHM_INTERVAL == 0:
            tokens, targets, _, _ = make_algorithm_batch(
                supports, layout, ALGORITHM_BATCH, model_seed * 1_000_003 + step, device
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
                    model, supports, layout, device, model_seed + 800_000 + step
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
        "support_exact": bool(torch.equal(recovered, supports)) if compiler else None,
        "xor_bit_ops": xor_ops if compiler else None,
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
    supports = make_supports()
    prefix_bits, prefix_targets, prefix_rules = make_prefix(supports)
    recovered, xor_ops = solve_prefix(prefix_bits, prefix_targets, prefix_rules)
    if not torch.equal(recovered, supports):
        raise RuntimeError("sealed prefix did not recover every support")
    if not torch.equal(unpack_supports(pack_supports(recovered)), recovered):
        raise RuntimeError("nibble pack round-trip failed")

    worlds: dict[str, object] = {}
    if quick:
        model = t3.build_model(MODEL_SEEDS[0], device)
        initial_hash = t3.state_sha256(model)
        frozen, compiler = compile_packed_interpreter(model, recovered, layout)
        frozen.enforce(model)
        worlds[str(MODEL_SEEDS[0])] = {
            "initial_hash": initial_hash,
            "compiler": compiler,
            "support_exact": True,
            "algorithm": evaluate_algorithm(
                model, supports, layout, device, MODEL_SEEDS[0] + 900_000, examples_per_rule=16
            ),
        }
    else:
        for seed in MODEL_SEEDS:
            baseline = train_arm(
                "baseline_1x",
                seed,
                supports,
                prefix_bits,
                prefix_targets,
                prefix_rules,
                recovered,
                xor_ops,
                layout,
                natural,
                validation,
                device,
                CHECKPOINTS_1X[-1],
            )
            candidate = train_arm(
                "compiler_1x",
                seed,
                supports,
                prefix_bits,
                prefix_targets,
                prefix_rules,
                recovered,
                xor_ops,
                layout,
                natural,
                validation,
                device,
                CHECKPOINTS_1X[-1],
            )
            doubled = train_arm(
                "baseline_2x",
                seed,
                supports,
                prefix_bits,
                prefix_targets,
                prefix_rules,
                recovered,
                xor_ops,
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
    if not quick:
        for seed, result in worlds.items():
            arms = result["arms"]
            baseline = arms["baseline_1x"]
            candidate = arms["compiler_1x"]
            doubled = arms["baseline_2x"]
            gates[f"{seed}_same_initialization"] = result["identical_initial_hashes"]
            gates[f"{seed}_all_supports_exact"] = candidate["support_exact"] is True
            gates[f"{seed}_description_6144"] = candidate["compiler"]["description_entries"] == 6144
            gates[f"{seed}_shared_width_constant"] = candidate["compiler"][
                "shared_width_independent_of_rules"
            ] is True
            for checkpoint in CHECKPOINTS_1X:
                key = str(checkpoint)
                algorithm = candidate["checkpoints"][key]["algorithm"]
                gates[f"{seed}_{key}_candidate_every_rule_ge_99"] = algorithm["parity_min"] >= 0.99
                gates[f"{seed}_{key}_candidate_copy_ge_99"] = algorithm["protected_copy"] >= 0.99
                candidate_nll = candidate["checkpoints"][key]["natural"]["nll"]
                baseline_nll = baseline["checkpoints"][key]["natural"]["nll"]
                gates[f"{seed}_{key}_natural_within_0p5pct"] = candidate_nll <= 1.005 * baseline_nll
            doubled_algorithm = doubled["checkpoints"][str(STEPS_2X)]["algorithm"]
            gates[f"{seed}_baseline2x_mean_parity_lt_80"] = doubled_algorithm["parity_mean"] < 0.80
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
        "schema": "packed-rule-memory-t4-v1",
        "status": "quick" if quick else "complete",
        "device": str(device),
        "torch_version": torch.__version__,
        "configuration": {
            "vocab": VOCAB,
            "hidden": HIDDEN,
            "layers": LAYERS,
            "heads": HEADS,
            "ffn_width": FFN_WIDTH,
            "parameter_count": t3.parameter_count(),
            "rules": RULES,
            "support_size": SUPPORT_SIZE,
            "prefix_per_rule": PREFIX_PER_RULE,
            "unused_tokens_available": unused_count,
            "selected_special_tokens": layout.all_special,
            "model_seeds": MODEL_SEEDS,
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
