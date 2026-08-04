#!/usr/bin/env python3
"""T7: adjacent-scale falsification of heterogeneous program compilation.

The pilot calibrates a hybrid Muon/AdamW optimizer on natural text only.  The
sealed run then compares compilation against 1x and 2x frontier controls while
keeping the deployed 110.8M-parameter causal LM identical across arms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import heterogeneous_family_discovery_t6 as t6


PREREGISTRATION = ROOT / "results/heterogeneous-family-discovery-t7-preregistration.md"
TEST_SOURCE = ROOT / "tests/test_heterogeneous_family_discovery_t7_scale.py"
PREDECESSOR = ROOT / "results/heterogeneous-family-discovery-t6.json"
PILOT_PROTOCOL = ROOT / "results/heterogeneous-family-discovery-t7-muon-pilot-protocol.md"
PILOT_OUTPUT = ROOT / "results/heterogeneous-family-discovery-t7-muon-pilot.json"
OUTPUT = ROOT / "results/heterogeneous-family-discovery-t7.json"

VOCAB = 49_152
HIDDEN = 640
LAYERS = 16
HEADS = 10
HEAD_DIM = HIDDEN // HEADS
FFN_WIDTH = 1_728
CONTEXT = 128
PROGRAM_HEADS = t6.PROGRAM_HEADS
PROGRAM_DIMS = t6.PROGRAM_DIMS
PROGRAM_CHANNELS = t6.PROGRAM_CHANNELS
RMS_EPSILON = 1e-5

NATURAL_BATCH = 16
ALGORITHM_BATCH = 64
ALGORITHM_INTERVAL = 20
CHECKPOINTS_1X = (250, 500, 1_000)
STEPS_2X = 2_000
WARMUP_STEPS = 50
SCHEDULE_STEPS = STEPS_2X
MODEL_SEEDS = (3_149, 3_499, 3_853)
WORLD_SEED = 2_000_033
EVALUATION_SEED = 3_000_017

ADAMW_LEARNING_RATE = 3e-4
MUON_PILOT_LEARNING_RATES = (0.005, 0.01, 0.02, 0.04)
PILOT_STEPS = 250
PILOT_CHECKPOINTS = (50, 100, 250)
PILOT_SEED = 2_719

TokenStream = t6.TokenStream
TokenLayout = t6.TokenLayout
FrozenEntries = t6.FrozenEntries
DiscoveryWorld = t6.DiscoveryWorld


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


class RMSNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(HIDDEN))

    def forward(self, hidden: Tensor) -> Tensor:
        scale = hidden.float().pow(2).mean(dim=-1, keepdim=True).add(RMS_EPSILON).rsqrt()
        return hidden * scale.to(hidden.dtype) * self.weight


def apply_rope_language_heads(query: Tensor, key: Tensor) -> tuple[Tensor, Tensor]:
    """Leave two compiler heads unrotated and rotate eight language heads."""
    length = query.shape[-2]
    positions = torch.arange(length, device=query.device, dtype=torch.float32)
    inv = 1.0 / (
        10_000
        ** (torch.arange(0, HEAD_DIM, 2, device=query.device).float() / HEAD_DIM)
    )
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


class ScaleLM(nn.Module):
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


def parameter_count() -> int:
    embedding = VOCAB * HIDDEN
    attention = 4 * HIDDEN * HIDDEN
    ffn = 3 * HIDDEN * FFN_WIDTH
    norms = 2 * HIDDEN
    return embedding + LAYERS * (attention + ffn + norms) + HIDDEN


def build_model(seed: int, device: torch.device) -> ScaleLM:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = ScaleLM().to(device)
    actual = sum(parameter.numel() for parameter in model.parameters())
    if actual != parameter_count():
        raise RuntimeError(f"parameter count mismatch: formula={parameter_count()} actual={actual}")
    return model


def choose_token_layout(
    train: TokenStream, validation: TokenStream
) -> tuple[TokenLayout, int]:
    return t6.choose_token_layout(train, validation)


def rms_scale(norm_squared: float) -> float:
    return math.sqrt(HIDDEN / (norm_squared + RMS_EPSILON * HIDDEN))


@torch.no_grad()
def compile_interpreter(
    model: ScaleLM,
    recovered_supports: Tensor,
    recovered_families: Tensor,
    accepted: Tensor,
    layout: TokenLayout,
    reference_world: DiscoveryWorld,
) -> tuple[FrozenEntries, dict[str, object]]:
    """Write T6's exact shared interpreter into the wider ordinary LM."""
    named = dict(model.named_parameters())
    masks = {
        name: torch.zeros_like(parameter, dtype=torch.bool)
        for name, parameter in named.items()
    }

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
        named["token.weight"][token_id, t6.SUPPORT_START : t6.SUPPORT_START + t6.INPUT_DIM] = (
            signs.to(model.token.weight.device)
        )
        named["token.weight"][token_id, t6.FAMILY_START + recovered_families[task]] = 1.0
        named["token.weight"][token_id, t6.TASK] = 1.0
    for position, token_id in enumerate(layout.zero):
        named["token.weight"][token_id, t6.SUPPORT_START + position] = 1.0
        named["token.weight"][token_id, t6.BIT_TYPE] = 1.0
        named["token.weight"][token_id, t6.CONST] = 1.0
    for position, token_id in enumerate(layout.one):
        named["token.weight"][token_id, t6.SUPPORT_START + position] = 1.0
        named["token.weight"][token_id, t6.BIT_TYPE] = 1.0
        named["token.weight"][token_id, t6.CONST] = -1.0
    named["token.weight"][layout.parity_query, t6.CONST] = 1.0
    named["token.weight"][layout.parity_query, t6.PARITY_ROUTE] = 1.0
    named["token.weight"][layout.copy_query, t6.CONST] = 1.0
    named["token.weight"][layout.copy_query, t6.COPY_ROUTE] = 1.0
    named["token.weight"][layout.result_zero, t6.OUT] = 30.0
    named["token.weight"][layout.result_one, t6.OUT] = -30.0

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

    block0 = model.blocks[0]
    alpha_task = rms_scale(t6.INPUT_DIM + 3.0)
    alpha_bit = rms_scale(3.0)
    alpha_query = rms_scale(2.0)
    q_a = math.sqrt(math.sqrt(HEAD_DIM) * 30.0) / alpha_bit
    q_type = math.sqrt(math.sqrt(HEAD_DIM) * 30.0) / alpha_bit
    q_task = math.sqrt(math.sqrt(HEAD_DIM) * 60.0 / (alpha_query * alpha_task))
    q_copy = math.sqrt(math.sqrt(HEAD_DIM) * 60.0 / (alpha_query * alpha_bit))
    for component in range(t6.INPUT_DIM):
        block0.attention.q.weight[component, t6.SUPPORT_START + component] = q_a
        block0.attention.k.weight[component, t6.SUPPORT_START + component] = q_a
    block0.attention.q.weight[16, t6.BIT_TYPE] = q_type
    block0.attention.k.weight[16, t6.BIT_TYPE] = q_type
    block0.attention.q.weight[17, t6.PARITY_ROUTE] = q_task
    block0.attention.k.weight[17, t6.TASK] = q_task
    block0.attention.q.weight[18, t6.COPY_ROUTE] = q_copy
    block0.attention.k.weight[18, t6.BIT_TYPE] = q_copy
    for component in range(t6.INPUT_DIM + len(t6.FAMILY_NAMES)):
        source = t6.SUPPORT_START + component
        block0.attention.v.weight[component, source] = 1.0
        block0.attention.o.weight[source, component] = 1.0 / alpha_bit

    task_amplitude = 1.0 + alpha_task / alpha_bit
    bit_amplitude = 2.0
    query_amplitude = alpha_task / alpha_bit
    alpha_task_1 = rms_scale((t6.INPUT_DIM + 1) * task_amplitude**2 + 1.0)
    alpha_bit_1 = rms_scale(bit_amplitude**2 + 2.0)
    alpha_query_1 = rms_scale((t6.INPUT_DIM + 1) * query_amplitude**2 + 2.0)

    block1 = model.blocks[1]
    q_select = math.sqrt(
        math.sqrt(HEAD_DIM)
        * 30.0
        / (alpha_query_1 * alpha_bit_1 * query_amplitude * bit_amplitude)
    )
    for component in range(t6.INPUT_DIM):
        block1.attention.q.weight[component, t6.SUPPORT_START + component] = q_select
        block1.attention.k.weight[component, t6.SUPPORT_START + component] = q_select
    raw_task = (
        q_select**2
        * alpha_query_1
        * alpha_task_1
        * query_amplitude
        * task_amplitude
        * t6.INPUT_DIM
        / math.sqrt(HEAD_DIM)
    )
    raw_self = (
        q_select**2
        * alpha_query_1**2
        * query_amplitude**2
        * t6.INPUT_DIM
        / math.sqrt(HEAD_DIM)
    )
    block1.attention.q.weight[63, t6.PARITY_ROUTE] = 1.0
    block1.attention.k.weight[63, t6.TASK] = (
        -math.sqrt(HEAD_DIM) * (raw_task + 120.0) / (alpha_query_1 * alpha_task_1)
    )
    block1.attention.k.weight[63, t6.PARITY_ROUTE] = (
        -math.sqrt(HEAD_DIM) * (raw_self + 120.0) / alpha_query_1**2
    )
    block1.attention.v.weight[0, t6.CONST] = 1.0
    block1.attention.o.weight[t6.SUM, 0] = t6.SUPPORT_SIZE / alpha_bit_1

    copy_head = HEAD_DIM
    strong = 20.0
    block1.attention.q.weight[copy_head, t6.COPY_ROUTE] = strong
    block1.attention.k.weight[copy_head, t6.SUPPORT_START] = strong
    block1.attention.q.weight[copy_head + 1, t6.COPY_ROUTE] = strong
    block1.attention.k.weight[copy_head + 1, t6.BIT_TYPE] = strong
    block1.attention.v.weight[copy_head, t6.CONST] = 1.0
    block1.attention.o.weight[t6.COPY_SUM, copy_head] = 1.0 / alpha_bit_1

    block2 = model.blocks[2]
    beta = 16.0
    channel = 0
    truth_tables: dict[str, list[int]] = {}
    for family, name in enumerate(t6.FAMILY_NAMES):
        table: list[int] = []
        for count in range(t6.SUPPORT_SIZE + 1):
            label = bool(t6.family_labels(torch.tensor([count]), family).item())
            sign = -1.0 if label else 1.0
            table.append(int(label))
            integer_sum = t6.SUPPORT_SIZE - 2 * count
            for threshold, coefficient in (
                (integer_sum - 1, 1.0),
                (integer_sum, -2.0),
                (integer_sum + 1, 1.0),
            ):
                block2.gate.weight[channel, t6.SUM] = beta
                block2.gate.weight[channel, t6.CONST] = -beta * threshold
                block2.up.weight[channel, t6.FAMILY_START + family] = 1.0 / beta
                block2.down.weight[t6.OUT, channel] = sign * coefficient
                channel += 1
        truth_tables[name] = table
    block2.gate.weight[channel, t6.COPY_SUM] = 1.0
    block2.gate.weight[channel + 1, t6.COPY_SUM] = -1.0
    block2.up.weight[channel, t6.COPY_ROUTE] = 1.0
    block2.up.weight[channel + 1, t6.COPY_ROUTE] = 1.0
    block2.down.weight[t6.OUT, channel] = 1.0
    block2.down.weight[t6.OUT, channel + 1] = -1.0
    used_channels = channel + 2

    values = {name: parameter.detach().clone() for name, parameter in named.items()}
    frozen = FrozenEntries(
        masks,
        values,
        sum(int(mask.sum().item()) for mask in masks.values()),
    )
    frozen.enforce(model)
    fixed_nonzero = sum(
        int(((values[name] != 0) & mask).sum().item())
        for name, mask in masks.items()
    )
    return frozen, {
        "accepted_tasks": int(accepted.sum().item()),
        "accepted_random_tasks": int(accepted[t6.STRUCTURED_TASKS :].sum().item()),
        "exact_supports": int(
            (
                recovered_supports[: t6.STRUCTURED_TASKS]
                == reference_world.supports[: t6.STRUCTURED_TASKS]
            )
            .all(1)
            .sum()
            .item()
        ),
        "exact_families": int(
            (
                recovered_families[: t6.STRUCTURED_TASKS]
                == reference_world.families[: t6.STRUCTURED_TASKS]
            )
            .sum()
            .item()
        ),
        "description_entries": t6.STRUCTURED_TASKS
        * (t6.INPUT_DIM + len(t6.FAMILY_NAMES)),
        "program_hidden_coordinates": PROGRAM_DIMS,
        "program_attention_heads": PROGRAM_HEADS,
        "used_decoder_channels": used_channels,
        "frozen_parameter_entries": frozen.count,
        "fixed_nonzero_entries": fixed_nonzero,
        "truth_tables": truth_tables,
    }


def schedule_fraction(step: int) -> float:
    if step <= WARMUP_STEPS:
        return step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / (SCHEDULE_STEPS - WARMUP_STEPS)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return 0.1 + 0.9 * cosine


@dataclass
class OptimizerBundle:
    kind: str
    matrix: torch.optim.Optimizer | None
    auxiliary: torch.optim.Optimizer
    matrix_peak_lr: float | None
    auxiliary_peak_lr: float
    matrix_parameter_count: int
    auxiliary_parameter_count: int

    def zero_grad(self) -> None:
        if self.matrix is not None:
            self.matrix.zero_grad(set_to_none=True)
        self.auxiliary.zero_grad(set_to_none=True)

    def set_step(self, step: int) -> None:
        fraction = schedule_fraction(step)
        if self.matrix is not None and self.matrix_peak_lr is not None:
            for group in self.matrix.param_groups:
                group["lr"] = self.matrix_peak_lr * fraction
        for group in self.auxiliary.param_groups:
            group["lr"] = self.auxiliary_peak_lr * fraction

    def step(self) -> None:
        if self.matrix is not None:
            self.matrix.step()
        self.auxiliary.step()

    def optimizers(self) -> Iterable[torch.optim.Optimizer]:
        if self.matrix is not None:
            yield self.matrix
        yield self.auxiliary


def build_optimizer(
    model: ScaleLM,
    kind: str,
    *,
    muon_learning_rate: float,
) -> OptimizerBundle:
    if kind == "adamw":
        parameters = list(model.parameters())
        optimizer = torch.optim.AdamW(
            parameters,
            lr=ADAMW_LEARNING_RATE,
            betas=(0.9, 0.95),
            weight_decay=0.0,
        )
        return OptimizerBundle(
            kind="adamw",
            matrix=None,
            auxiliary=optimizer,
            matrix_peak_lr=None,
            auxiliary_peak_lr=ADAMW_LEARNING_RATE,
            matrix_parameter_count=0,
            auxiliary_parameter_count=sum(parameter.numel() for parameter in parameters),
        )

    if kind != "muon":
        raise ValueError(f"unknown optimizer kind {kind}")
    matrix_parameters: list[Tensor] = []
    auxiliary_parameters: list[Tensor] = []
    for name, parameter in model.named_parameters():
        if parameter.ndim == 2 and name != "token.weight":
            matrix_parameters.append(parameter)
        else:
            auxiliary_parameters.append(parameter)
    matrix = torch.optim.Muon(
        matrix_parameters,
        lr=muon_learning_rate,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adjust_lr_fn="match_rms_adamw",
        weight_decay=0.0,
    )
    auxiliary = torch.optim.AdamW(
        auxiliary_parameters,
        lr=ADAMW_LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    return OptimizerBundle(
        kind="muon+adamw",
        matrix=matrix,
        auxiliary=auxiliary,
        matrix_peak_lr=muon_learning_rate,
        auxiliary_peak_lr=ADAMW_LEARNING_RATE,
        matrix_parameter_count=sum(parameter.numel() for parameter in matrix_parameters),
        auxiliary_parameter_count=sum(parameter.numel() for parameter in auxiliary_parameters),
    )


def optimizer_state_bytes(bundle: OptimizerBundle) -> int:
    total = 0
    for optimizer in bundle.optimizers():
        for state in optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    total += value.numel() * value.element_size()
    return total


@torch.no_grad()
def evaluate_natural(
    model: ScaleLM,
    stream: TokenStream,
    device: torch.device,
    seed: int,
    *,
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
    return {
        "nll": float(np.mean(losses)),
        "per_batch_nll": losses,
        "tokens": batches * 8 * CONTEXT,
    }


def read_gpu_power_watts() -> float | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2,
        )
        return float(output.splitlines()[0].strip())
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def model_forward_flops(batch: int, length: int, *, binary_head: bool) -> int:
    projection = 2 * batch * length * LAYERS * (
        4 * HIDDEN * HIDDEN + 3 * HIDDEN * FFN_WIDTH
    )
    attention = 4 * batch * LAYERS * length * length * HIDDEN
    output_width = 2 if binary_head else VOCAB
    output = 2 * batch * length * HIDDEN * output_width
    return projection + attention + output


def training_flops(batch: int, length: int, *, binary_head: bool) -> int:
    # Reverse mode for dense matmuls is two additional matmuls of equal size.
    return 3 * model_forward_flops(batch, length, binary_head=binary_head)


def muon_step_flops(model: nn.Module, ns_steps: int = 5) -> int:
    """Count the dominant Newton--Schulz matmuls in PyTorch Muon.

    For the transposed shape r <= c, each iteration evaluates X X^T, A A,
    and (b A + c A^2) X: 4 r^2 c + 2 r^3 multiply-add FLOPs.
    """
    total = 0
    for name, parameter in model.named_parameters():
        if parameter.ndim != 2 or name == "token.weight":
            continue
        rows, columns = parameter.shape
        r, c = min(rows, columns), max(rows, columns)
        total += ns_steps * (4 * r * r * c + 2 * r * r * r)
    return total


def run_muon_pilot(device: torch.device) -> dict[str, object]:
    natural = TokenStream(t6.t3.TRAIN_FILE)
    validation = TokenStream(t6.t3.VALIDATION_FILE)
    arms: dict[str, object] = {}
    learning_rates: tuple[tuple[str, float], ...] = (("adamw", ADAMW_LEARNING_RATE),) + tuple(
        ("muon", value) for value in MUON_PILOT_LEARNING_RATES
    )
    for kind, learning_rate in learning_rates:
        label = kind if kind == "adamw" else f"muon_{learning_rate:g}"
        model = build_model(PILOT_SEED, device)
        initial_hash = state_sha256(model)
        optimizer = build_optimizer(
            model,
            kind,
            muon_learning_rate=learning_rate,
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()
        started = time.perf_counter()
        checkpoints: dict[str, object] = {}
        maximum_loss = 0.0
        maximum_gradient_norm = 0.0
        power_samples: list[float] = []
        for step in range(1, PILOT_STEPS + 1):
            model.train()
            optimizer.set_step(step)
            optimizer.zero_grad()
            inputs, targets = natural.batch(
                PILOT_SEED * 1_000_003 + step,
                NATURAL_BATCH,
                CONTEXT,
                device,
            )
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(inputs)
            loss = F.cross_entropy(logits.float().reshape(-1, VOCAB), targets.reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite {label} pilot loss at step {step}")
            loss.backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
            if not math.isfinite(gradient_norm):
                raise RuntimeError(f"nonfinite {label} pilot gradient at step {step}")
            optimizer.step()
            maximum_loss = max(maximum_loss, float(loss.item()))
            maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)
            if step % 10 == 0:
                sample = read_gpu_power_watts() if device.type == "cuda" else None
                if sample is not None:
                    power_samples.append(sample)
            if step in PILOT_CHECKPOINTS:
                checkpoints[str(step)] = evaluate_natural(
                    model,
                    validation,
                    device,
                    EVALUATION_SEED,
                    batches=16,
                )
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        arms[label] = {
            "kind": optimizer.kind,
            "learning_rate": learning_rate,
            "initial_hash": initial_hash,
            "checkpoints": checkpoints,
            "maximum_loss": maximum_loss,
            "maximum_gradient_norm": maximum_gradient_norm,
            "elapsed_seconds": elapsed,
            "gpu_seconds": elapsed if device.type == "cuda" else 0.0,
            "mean_sampled_power_watts": float(np.mean(power_samples)) if power_samples else None,
            "estimated_joules": float(np.mean(power_samples) * elapsed) if power_samples else None,
            "peak_hbm_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
            "peak_hbm_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
            ),
            "optimizer_state_bytes": optimizer_state_bytes(optimizer),
            "matrix_parameter_count": optimizer.matrix_parameter_count,
            "auxiliary_parameter_count": optimizer.auxiliary_parameter_count,
            "mathematical_training_flops": PILOT_STEPS
            * training_flops(NATURAL_BATCH, CONTEXT, binary_head=False),
            "muon_optimizer_flops": (
                PILOT_STEPS * muon_step_flops(model) if kind == "muon" else 0
            ),
            "terminal_hash": state_sha256(model),
        }
        del optimizer, model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    terminal_nll = {
        label: result["checkpoints"][str(PILOT_STEPS)]["nll"]
        for label, result in arms.items()
    }
    stable = {
        label: math.isfinite(result["maximum_loss"])
        and math.isfinite(result["maximum_gradient_norm"])
        and result["maximum_loss"] < 100.0
        for label, result in arms.items()
    }
    eligible_muon = {
        label: nll
        for label, nll in terminal_nll.items()
        if label.startswith("muon_") and stable[label]
    }
    winner = min(eligible_muon, key=eligible_muon.get)
    selected_learning_rate = float(winner.removeprefix("muon_"))
    return {
        "schema": "heterogeneous-family-discovery-t7-muon-pilot-v1",
        "status": "complete",
        "device": str(device),
        "torch_version": torch.__version__,
        "configuration": {
            "vocab": VOCAB,
            "hidden": HIDDEN,
            "layers": LAYERS,
            "heads": HEADS,
            "ffn_width": FFN_WIDTH,
            "parameter_count": parameter_count(),
            "pilot_seed": PILOT_SEED,
            "pilot_steps": PILOT_STEPS,
            "pilot_checkpoints": PILOT_CHECKPOINTS,
            "muon_learning_rates": MUON_PILOT_LEARNING_RATES,
            "adamw_learning_rate": ADAMW_LEARNING_RATE,
            "natural_batch": NATURAL_BATCH,
            "context": CONTEXT,
        },
        "integrity": {
            "source_sha256": sha256_file(Path(__file__)),
            "pilot_protocol_sha256": sha256_file(PILOT_PROTOCOL),
            "train_sha256": sha256_file(t6.t3.TRAIN_FILE),
            "validation_sha256": sha256_file(t6.t3.VALIDATION_FILE),
        },
        "identical_initial_hashes": len(
            {result["initial_hash"] for result in arms.values()}
        )
        == 1,
        "arms": arms,
        "terminal_nll": terminal_nll,
        "stable": stable,
        "selection_rule": "lowest step-250 held-out NLL among finite Muon arms",
        "selected_arm": winner,
        "selected_muon_learning_rate": selected_learning_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if not arguments.pilot:
        raise SystemExit("the full run is enabled only after the Muon pilot is sealed")
    result = run_muon_pilot(torch.device(arguments.device))
    output = arguments.output or PILOT_OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "selected_arm": result["selected_arm"],
                "selected_muon_learning_rate": result["selected_muon_learning_rate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
