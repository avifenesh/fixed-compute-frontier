#!/usr/bin/env python3
"""T3: 256 discovered parity rules sharing one in-place LM interpreter."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import heterogeneous_algebraic_compilation_t1 as t1


PREREGISTRATION = ROOT / "results/shared-algebraic-interpreter-t3-preregistration.md"
TEST_SOURCE = ROOT / "tests/test_shared_algebraic_interpreter_t3.py"
PREDECESSOR = ROOT / "results/heterogeneous-algebraic-compilation-t2-lm.json"
TRAIN_FILE = ROOT / "data/block-algebra-scratch-v4576/train.uint16.bin"
VALIDATION_FILE = ROOT / "data/block-algebra-scratch-v4576/validation.uint16.bin"
OUTPUT = ROOT / "results/shared-algebraic-interpreter-t3.json"

VOCAB = 49_152
HIDDEN = 384
LAYERS = 10
HEADS = 6
HEAD_DIM = HIDDEN // HEADS
FFN_WIDTH = 1_024
CONTEXT = 128
INPUT_DIM = 32
RULES = 256
SUPPORT_SIZE = 16
PREFIX_PER_RULE = 64
PROGRAM_DIMS = 40
PROGRAM_HEADS = 2
PROGRAM_CHANNELS = 64
NATURAL_BATCH = 16
ALGORITHM_BATCH = 64
ALGORITHM_INTERVAL = 20
CHECKPOINTS_1X = (250, 500, 1000)
STEPS_2X = 2000
MODEL_SEEDS = (1181, 1429)
PEAK_LEARNING_RATE = 3e-4
WARMUP_STEPS = 50
SCHEDULE_STEPS = STEPS_2X
RMS_EPSILON = 1e-5

# Program hidden coordinates.
A_START = 0
SUM = 32
COPY_SUM = 33
CONST = 34
TASK = 35
BIT_TYPE = 36
PARITY = 37
COPY = 38
OUT = 39


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class TokenStream:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.tokens = np.memmap(path, mode="r", dtype="<u2")

    def batch(
        self, seed: int, batch_size: int, length: int, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        generator = np.random.default_rng(seed)
        starts = generator.integers(0, len(self.tokens) - length - 1, size=batch_size)
        rows = np.stack(
            [np.asarray(self.tokens[start : start + length + 1], dtype=np.int64) for start in starts]
        )
        values = torch.from_numpy(rows).to(device)
        return values[:, :-1], values[:, 1:]


@dataclass(frozen=True)
class TokenLayout:
    rule: tuple[int, ...]
    zero: tuple[int, ...]
    one: tuple[int, ...]
    parity_query: int
    copy_query: int
    result_zero: int
    result_one: int

    @property
    def all_special(self) -> tuple[int, ...]:
        return (
            self.rule
            + self.zero
            + self.one
            + (
                self.parity_query,
                self.copy_query,
                self.result_zero,
                self.result_one,
            )
        )


def choose_token_layout(train: TokenStream, validation: TokenStream) -> TokenLayout:
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
    return TokenLayout(rule, zero, one, *selected[cursor : cursor + 4])


class RMSNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(HIDDEN))

    def forward(self, hidden: Tensor) -> Tensor:
        scale = hidden.float().pow(2).mean(dim=-1, keepdim=True).add(RMS_EPSILON).rsqrt()
        return hidden * scale.to(hidden.dtype) * self.weight


def apply_rope_language_heads(query: Tensor, key: Tensor) -> tuple[Tensor, Tensor]:
    """Apply RoPE to the four language heads; program heads stay unrotated."""
    length = query.shape[-2]
    positions = torch.arange(length, device=query.device, dtype=torch.float32)
    inv = 1.0 / (10_000 ** (torch.arange(0, HEAD_DIM, 2, device=query.device).float() / HEAD_DIM))
    angles = torch.outer(positions, inv)
    cos = angles.cos().to(query.dtype)[None, None, :, :]
    sin = angles.sin().to(query.dtype)[None, None, :, :]

    def rotate(values: Tensor) -> Tensor:
        fixed = values[:, :PROGRAM_HEADS]
        moving = values[:, PROGRAM_HEADS:]
        even = moving[..., 0::2]
        odd = moving[..., 1::2]
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
        return torch.cat((fixed, rotated.flatten(-2)), dim=1)

    return rotate(query), rotate(key)


class CausalAttention(nn.Module):
    def __init__(self, depth_scale: float) -> None:
        super().__init__()
        self.q = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.k = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.v = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.o = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.depth_scale = depth_scale

    def reset_parameters(self) -> None:
        for parameter in (self.q.weight, self.k.weight, self.v.weight):
            nn.init.normal_(parameter, mean=0.0, std=0.02)
        nn.init.normal_(self.o.weight, mean=0.0, std=0.02 * self.depth_scale)

    def forward(self, hidden: Tensor) -> Tensor:
        batch, length, _ = hidden.shape
        query = self.q(hidden).view(batch, length, HEADS, HEAD_DIM).transpose(1, 2)
        key = self.k(hidden).view(batch, length, HEADS, HEAD_DIM).transpose(1, 2)
        value = self.v(hidden).view(batch, length, HEADS, HEAD_DIM).transpose(1, 2)
        query, key = apply_rope_language_heads(query, key)
        scores = query @ key.transpose(-1, -2) / math.sqrt(HEAD_DIM)
        causal = torch.ones(length, length, dtype=torch.bool, device=hidden.device).triu(1)
        probabilities = scores.masked_fill(causal, float("-inf")).softmax(dim=-1)
        attended = probabilities @ value
        attended = attended.transpose(1, 2).contiguous().view(batch, length, HIDDEN)
        return self.o(attended)


class Block(nn.Module):
    def __init__(self, depth_scale: float) -> None:
        super().__init__()
        self.attention_norm = RMSNorm()
        self.attention = CausalAttention(depth_scale)
        self.ffn_norm = RMSNorm()
        self.gate = nn.Linear(HIDDEN, FFN_WIDTH, bias=False)
        self.up = nn.Linear(HIDDEN, FFN_WIDTH, bias=False)
        self.down = nn.Linear(FFN_WIDTH, HIDDEN, bias=False)
        self.depth_scale = depth_scale

    def reset_parameters(self) -> None:
        self.attention_norm.weight.data.fill_(1.0)
        self.ffn_norm.weight.data.fill_(1.0)
        self.attention.reset_parameters()
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.up.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02 * self.depth_scale)

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden))
        normalized = self.ffn_norm(hidden)
        return hidden + self.down(F.silu(self.gate(normalized)) * self.up(normalized))


class SharedInterpreterLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token = nn.Embedding(VOCAB, HIDDEN)
        depth_scale = 1.0 / math.sqrt(2 * LAYERS)
        self.blocks = nn.ModuleList(Block(depth_scale) for _ in range(LAYERS))
        self.final_norm = RMSNorm()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token.weight, mean=0.0, std=0.02)
        for block in self.blocks:
            block.reset_parameters()
        self.final_norm.weight.data.fill_(1.0)

    def hidden(self, tokens: Tensor) -> Tensor:
        hidden = self.token(tokens)
        for block in self.blocks:
            hidden = block(hidden)
        return self.final_norm(hidden)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.hidden(tokens) @ self.token.weight.T

    def binary_logits(self, tokens: Tensor, layout: TokenLayout) -> Tensor:
        final = self.hidden(tokens)[:, -1]
        rows = self.token.weight[[layout.result_zero, layout.result_one]]
        return final @ rows.T


def build_model(seed: int, device: torch.device) -> SharedInterpreterLM:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    return SharedInterpreterLM().to(device)


def make_supports(seed: int = 73_001) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    supports = torch.zeros(RULES, INPUT_DIM, dtype=torch.uint8)
    for rule in range(RULES):
        selected = torch.randperm(INPUT_DIM, generator=generator)[:SUPPORT_SIZE]
        supports[rule, selected] = 1
    if len({tuple(row.tolist()) for row in supports}) != RULES:
        raise RuntimeError("rule generator produced a duplicate support")
    return supports


def make_prefix(supports: Tensor, seed: int = 83_003) -> tuple[Tensor, Tensor, Tensor]:
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
    if parity_only:
        parity_route = torch.ones(count, dtype=torch.bool)
    else:
        parity_route = torch.randint(0, 2, (count,), generator=generator).bool()
    parity = (bits.to(torch.int16) * supports[rules].to(torch.int16)).sum(1).remainder(2).long()
    copied = bits[:, 0].long()
    targets = torch.where(parity_route, parity, copied)
    tokens = encode_algorithm(bits, rules, parity_route, layout, device)
    return tokens, targets.to(device), parity_route.to(device), rules.to(device)


@dataclass
class FrozenEntries:
    masks: dict[str, Tensor]
    values: dict[str, Tensor]
    count: int

    @torch.no_grad()
    def enforce(self, model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            mask = self.masks[name]
            parameter[mask] = self.values[name][mask]

    @torch.no_grad()
    def mask_gradients(self, model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                parameter.grad[self.masks[name]] = 0.0


def rms_scale(norm_squared: float) -> float:
    return math.sqrt(HIDDEN / (norm_squared + RMS_EPSILON * HIDDEN))


@torch.no_grad()
def compile_interpreter(
    model: SharedInterpreterLM,
    recovered: Tensor,
    layout: TokenLayout,
) -> tuple[FrozenEntries, dict[str, object]]:
    named = dict(model.named_parameters())
    masks = {name: torch.zeros_like(parameter, dtype=torch.bool) for name, parameter in named.items()}

    def fix(name: str, index: object, value: float = 0.0) -> None:
        named[name][index] = value
        masks[name][index] = True

    # Reserve an in-place hidden partition.  The 49,152-row vocabulary and all
    # dense matrix shapes remain unchanged.
    fix("token.weight", (slice(None), slice(0, PROGRAM_DIMS)))
    for token_id in layout.all_special:
        fix("token.weight", (token_id, slice(None)))

    support_signs = recovered.float().mul(2.0).sub(1.0).to(model.token.weight.device)
    for rule, token_id in enumerate(layout.rule):
        named["token.weight"][token_id, A_START : A_START + INPUT_DIM] = support_signs[rule]
    for position, token_id in enumerate(layout.zero):
        named["token.weight"][token_id, A_START + position] = 1.0
        named["token.weight"][token_id, BIT_TYPE] = 1.0
        named["token.weight"][token_id, CONST] = 1.0
    for position, token_id in enumerate(layout.one):
        named["token.weight"][token_id, A_START + position] = 1.0
        named["token.weight"][token_id, BIT_TYPE] = 1.0
        named["token.weight"][token_id, CONST] = -1.0
    named["token.weight"][layout.parity_query, CONST] = 1.0
    named["token.weight"][layout.parity_query, PARITY] = 1.0
    named["token.weight"][layout.copy_query, CONST] = 1.0
    named["token.weight"][layout.copy_query, COPY] = 1.0
    named["token.weight"][layout.result_zero, OUT] = 30.0
    named["token.weight"][layout.result_one, OUT] = -30.0
    for token_id in layout.rule:
        named["token.weight"][token_id, TASK] = 1.0

    # Keep RMS scales fixed only on the program coordinates.
    for layer in range(LAYERS):
        for norm in ("attention_norm.weight", "ffn_norm.weight"):
            fix(f"blocks.{layer}.{norm}", slice(0, PROGRAM_DIMS), 1.0)
    fix("final_norm.weight", slice(0, PROGRAM_DIMS), 1.0)

    program_rows = slice(0, PROGRAM_HEADS * HEAD_DIM)
    language_rows = slice(PROGRAM_HEADS * HEAD_DIM, HIDDEN)
    for layer in range(LAYERS):
        prefix = f"blocks.{layer}."
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

    # Block 0: bit tokens self-attend while the parity query copies the rule
    # sign vector.  All amplitudes below include the exact RMSNorm scale.
    block0 = model.blocks[0]
    alpha_task = rms_scale(INPUT_DIM + 1.0)
    alpha_bit = rms_scale(3.0)
    alpha_query = rms_scale(2.0)
    gap = 30.0
    base = 30.0
    query_gap = 60.0
    q_a = math.sqrt(math.sqrt(HEAD_DIM) * gap) / alpha_bit
    k_a = q_a
    q_type = math.sqrt(math.sqrt(HEAD_DIM) * base) / alpha_bit
    k_type = q_type
    q_query = math.sqrt(math.sqrt(HEAD_DIM) * query_gap / (alpha_query * alpha_task))
    k_task = q_query
    for component in range(INPUT_DIM):
        block0.attention.q.weight[component, A_START + component] = q_a
        block0.attention.k.weight[component, A_START + component] = k_a
        block0.attention.v.weight[component, A_START + component] = 1.0
        block0.attention.o.weight[A_START + component, component] = 1.0 / alpha_bit
    block0.attention.q.weight[32, BIT_TYPE] = q_type
    block0.attention.k.weight[32, BIT_TYPE] = k_type
    block0.attention.q.weight[33, PARITY] = q_query
    block0.attention.k.weight[33, TASK] = k_task

    # Amplitudes after block 0 (attention is parallel, so query sees the
    # original task value).  These calibrate block 1 exactly.
    task_amplitude = 1.0 + alpha_task / alpha_bit
    bit_amplitude = 2.0
    query_amplitude = alpha_task / alpha_bit
    alpha_task_1 = rms_scale(INPUT_DIM * task_amplitude**2 + 1.0)
    alpha_bit_1 = rms_scale(bit_amplitude**2 + 2.0)
    alpha_query_1 = rms_scale(INPUT_DIM * query_amplitude**2 + 2.0)

    # Block 1 head 0: support-conditioned selection.  Fixed support size makes
    # softmax reduction equal the arithmetic mean of the 16 selected signs.
    block1 = model.blocks[1]
    selection_gap = 30.0
    q_select = math.sqrt(
        math.sqrt(HEAD_DIM)
        * selection_gap
        / (alpha_query_1 * alpha_bit_1 * query_amplitude * bit_amplitude)
    )
    k_select = q_select
    for component in range(INPUT_DIM):
        block1.attention.q.weight[component, A_START + component] = q_select
        block1.attention.k.weight[component, A_START + component] = k_select
    raw_task = (
        q_select
        * k_select
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
    block1.attention.q.weight[63, PARITY] = 1.0
    block1.attention.k.weight[63, TASK] = (
        -math.sqrt(HEAD_DIM) * (raw_task + 4 * selection_gap) / (alpha_query_1 * alpha_task_1)
    )
    block1.attention.k.weight[63, PARITY] = (
        -math.sqrt(HEAD_DIM) * (raw_self + 4 * selection_gap) / alpha_query_1**2
    )
    block1.attention.v.weight[0, CONST] = 1.0
    block1.attention.o.weight[SUM, 0] = SUPPORT_SIZE / alpha_bit_1

    # Block 1 head 1: protected copy selects position zero.  A shared bit-type
    # base excludes the task token and leaves a 30-logit self-position margin.
    copy_head = HEAD_DIM
    strong = 20.0
    block1.attention.q.weight[copy_head, COPY] = strong
    block1.attention.k.weight[copy_head, A_START] = strong
    block1.attention.q.weight[copy_head + 1, COPY] = strong
    block1.attention.k.weight[copy_head + 1, BIT_TYPE] = strong
    block1.attention.v.weight[copy_head, CONST] = 1.0
    block1.attention.o.weight[COPY_SUM, copy_head] = 1.0 / alpha_bit_1

    # Block 2: a shared triangular ReLU limit on the 17 possible integer sums.
    # SiLU(beta*x)/beta is exponentially close to ReLU(x) at the half-grid
    # thresholds, while all terms share the same positive RMS scale.
    block2 = model.blocks[2]
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
            block2.gate.weight[channel, SUM] = beta
            block2.gate.weight[channel, CONST] = -beta * threshold
            block2.up.weight[channel, PARITY] = 1.0 / beta
            block2.down.weight[OUT, channel] = parity_sign * coefficient
            channel += 1
    # Exact sign-preserving protected copy via silu(z)-silu(-z)=z.
    block2.gate.weight[channel, COPY_SUM] = 1.0
    block2.gate.weight[channel + 1, COPY_SUM] = -1.0
    block2.up.weight[channel, COPY] = 1.0
    block2.up.weight[channel + 1, COPY] = 1.0
    block2.down.weight[OUT, channel] = 1.0
    block2.down.weight[OUT, channel + 1] = -1.0
    used_channels = channel + 2

    values = {name: parameter.detach().clone() for name, parameter in named.items()}
    frozen = FrozenEntries(masks, values, sum(int(mask.sum().item()) for mask in masks.values()))
    frozen.enforce(model)
    return frozen, {
        "rules": RULES,
        "support_size": SUPPORT_SIZE,
        "description_entries": int(recovered.numel()),
        "program_hidden_coordinates": PROGRAM_DIMS,
        "program_attention_heads": PROGRAM_HEADS,
        "program_ffn_channels": PROGRAM_CHANNELS,
        "used_decoder_channels": used_channels,
        "frozen_parameter_entries": frozen.count,
        "shared_width_independent_of_rules": True,
        "alpha_task": alpha_task,
        "alpha_bit": alpha_bit,
        "task_amplitude": task_amplitude,
        "query_amplitude": query_amplitude,
    }


def parameter_count() -> int:
    return sum(parameter.numel() for parameter in SharedInterpreterLM().parameters())


@torch.no_grad()
def evaluate_algorithm(
    model: SharedInterpreterLM,
    supports: Tensor,
    layout: TokenLayout,
    device: torch.device,
    seed: int,
    examples_per_rule: int = 128,
) -> dict[str, object]:
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rules = torch.arange(RULES).repeat_interleave(examples_per_rule)
    bits = torch.randint(
        0, 2, (len(rules), INPUT_DIM), generator=generator, dtype=torch.uint8
    )
    parity_route = torch.ones(len(rules), dtype=torch.bool)
    parity = (bits.to(torch.int16) * supports[rules].to(torch.int16)).sum(1).remainder(2).long()
    tokens = encode_algorithm(bits, rules, parity_route, layout, device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model.binary_logits(tokens, layout).float()
    correct = logits.argmax(-1).cpu() == parity
    per_rule = correct.view(RULES, examples_per_rule).float().mean(1)

    copy_tokens, copy_targets, _, _ = make_algorithm_batch(
        supports, layout, 4096, seed + 1, device, parity_only=False
    )
    copy_tokens[:, -1] = layout.copy_query
    # Reconstruct copy targets directly from position-zero token identity.
    first = copy_tokens[:, 1].cpu()
    zero_id = layout.zero[0]
    copy_targets = (first != zero_id).long().to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        copy_logits = model.binary_logits(copy_tokens, layout).float()
    copy_accuracy = (copy_logits.argmax(-1) == copy_targets).float().mean()
    return {
        "parity_mean": float(per_rule.mean().item()),
        "parity_min": float(per_rule.min().item()),
        "parity_max": float(per_rule.max().item()),
        "per_rule": per_rule.tolist(),
        "protected_copy": float(copy_accuracy.item()),
    }


@torch.no_grad()
def evaluate_natural(
    model: SharedInterpreterLM,
    stream: TokenStream,
    device: torch.device,
    seed: int,
    batches: int = 32,
) -> dict[str, object]:
    model.eval()
    losses: list[float] = []
    for index in range(batches):
        inputs, targets = stream.batch(seed + index, 8, CONTEXT, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(inputs)
        loss = F.cross_entropy(logits.float().reshape(-1, VOCAB), targets.reshape(-1))
        losses.append(float(loss.item()))
    return {"nll": float(np.mean(losses)), "per_batch_nll": losses, "tokens": batches * 8 * CONTEXT}


def scheduled_learning_rate(step: int) -> float:
    if step <= WARMUP_STEPS:
        return PEAK_LEARNING_RATE * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / (SCHEDULE_STEPS - WARMUP_STEPS)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return PEAK_LEARNING_RATE * (0.1 + 0.9 * cosine)


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
    model = build_model(model_seed, device)
    initial_hash = state_sha256(model)
    frozen: FrozenEntries | None = None
    compiler: dict[str, object] | None = None
    if arm == "compiler_1x":
        frozen, compiler = compile_interpreter(model, recovered, layout)

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
            bits = prefix_bits[indices]
            rules = prefix_rules[indices]
            route = torch.ones(len(indices), dtype=torch.bool)
            tokens = encode_algorithm(bits, rules, route, layout, device)
            targets = prefix_targets[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model.binary_logits(tokens, layout)
            loss = F.cross_entropy(logits.float(), targets)
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
                supports,
                layout,
                ALGORITHM_BATCH,
                model_seed * 1_000_003 + step,
                device,
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
        optimizer.step()
        if frozen is not None:
            frozen.enforce(model)
        maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)
        maximum_loss = max(maximum_loss, float(loss.item()))
        if step in checkpoint_set:
            checkpoints[str(step)] = {
                "natural": evaluate_natural(
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
        "terminal_hash": state_sha256(model),
    }


def run(device: torch.device, quick: bool = False) -> dict[str, object]:
    natural = TokenStream(TRAIN_FILE)
    validation = TokenStream(VALIDATION_FILE)
    layout = choose_token_layout(natural, validation)
    supports = make_supports()
    prefix_bits, prefix_targets, prefix_rules = make_prefix(supports)
    recovered, xor_ops = solve_prefix(prefix_bits, prefix_targets, prefix_rules)
    if not torch.equal(recovered, supports):
        raise RuntimeError("sealed prefix did not recover every support")

    worlds: dict[str, object] = {}
    if quick:
        model = build_model(MODEL_SEEDS[0], device)
        initial_hash = state_sha256(model)
        frozen, compiler = compile_interpreter(model, recovered, layout)
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
            gates[f"{seed}_description_8192"] = candidate["compiler"]["description_entries"] == 8192
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
            gates[f"{seed}_finite_gradient_envelope"] = all(
                arm["maximum_gradient_norm"] < 100.0 and arm["maximum_loss"] < 100.0
                for arm in (baseline, candidate, doubled)
            )

    return {
        "schema": "shared-algebraic-interpreter-t3-v1",
        "status": "quick" if quick else "complete",
        "device": str(device),
        "torch_version": torch.__version__,
        "configuration": {
            "vocab": VOCAB,
            "hidden": HIDDEN,
            "layers": LAYERS,
            "heads": HEADS,
            "ffn_width": FFN_WIDTH,
            "parameter_count": parameter_count(),
            "rules": RULES,
            "support_size": SUPPORT_SIZE,
            "prefix_per_rule": PREFIX_PER_RULE,
            "unused_tokens_available": len(
                set(range(VOCAB))
                - set(np.unique(np.concatenate((np.asarray(natural.tokens), np.asarray(validation.tokens)))))
            ),
            "selected_special_tokens": layout.all_special,
            "model_seeds": MODEL_SEEDS,
        },
        "integrity": {
            "source_sha256": sha256_file(Path(__file__)),
            "test_sha256": sha256_file(TEST_SOURCE),
            "preregistration_sha256": sha256_file(PREREGISTRATION),
            "predecessor_sha256": sha256_file(PREDECESSOR),
            "train_sha256": sha256_file(TRAIN_FILE),
            "validation_sha256": sha256_file(VALIDATION_FILE),
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
