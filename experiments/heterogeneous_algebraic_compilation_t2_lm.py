#!/usr/bin/env python3
"""37M-class FineWeb coexistence screen for algebraic weight compilation."""

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
from experiments import heterogeneous_algebraic_compilation_t1b as t1b
from experiments import heterogeneous_algebraic_compilation_t1c as t1c


PREREGISTRATION = ROOT / "results/heterogeneous-algebraic-compilation-t2-lm-v3-preregistration.md"
TEST_SOURCE = ROOT / "tests/test_heterogeneous_algebraic_compilation_t2_lm.py"
PREDECESSOR = ROOT / "results/heterogeneous-algebraic-compilation-t1c.json"
TRAIN_FILE = ROOT / "data/block-algebra-scratch-v4576/train.uint16.bin"
VALIDATION_FILE = ROOT / "data/block-algebra-scratch-v4576/validation.uint16.bin"
OUTPUT = ROOT / "results/heterogeneous-algebraic-compilation-t2-lm.json"

BASE_VOCAB = 49_152
BIT_ZERO = BASE_VOCAB
BIT_ONE = BASE_VOCAB + 1
PARITY_QUERY = BASE_VOCAB + 2
COPY_QUERY = BASE_VOCAB + 3
VOCAB = BASE_VOCAB + 4
HIDDEN = 384
LAYERS = 10
HEADS = 6
HEAD_DIM = HIDDEN // HEADS
FFN_WIDTH = 1_024
CONTEXT = 128
MAX_SUPPORT = t1.INPUT_DIM
LEAF_START = 9
MAX_COMPILER_DIMS = LEAF_START + MAX_SUPPORT
NATURAL_BATCH = 16
ALGEBRA_BATCH = 64
ALGEBRA_INTERVAL = 20
CHECKPOINTS_1X = (250, 500, 1000)
STEPS_2X = 2000
MODEL_SEEDS = (731, 947)
PEAK_LEARNING_RATE = 3e-4
WARMUP_STEPS = 50
SCHEDULE_STEPS = STEPS_2X

CONST = t1c.CONST
BIT = t1c.BIT
ROUTE_SIGN = t1c.ROUTE_SIGN
PARITY_ROUTE = t1c.PARITY_ROUTE
PARITY_KEY = t1c.PARITY_KEY
COPY_KEY = t1c.COPY_KEY
PARITY_AVERAGE = t1c.PARITY_AVERAGE
COPY_VALUE = t1c.COPY_VALUE
PARITY_FEATURE = t1c.PARITY_FEATURE


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
        scores = query @ key.transpose(-1, -2) / math.sqrt(HEAD_DIM)
        causal = torch.ones(length, length, dtype=torch.bool, device=hidden.device).triu(1)
        probabilities = scores.masked_fill(causal, float("-inf")).softmax(dim=-1)
        attended = probabilities @ value
        attended = attended.transpose(1, 2).contiguous().view(batch, length, HIDDEN)
        return self.o(attended)


class Block(nn.Module):
    def __init__(self, depth_scale: float) -> None:
        super().__init__()
        self.attention = CausalAttention(depth_scale)
        self.gate = nn.Linear(HIDDEN, FFN_WIDTH, bias=False)
        self.up = nn.Linear(HIDDEN, FFN_WIDTH, bias=False)
        self.down = nn.Linear(FFN_WIDTH, HIDDEN, bias=False)
        self.depth_scale = depth_scale

    def reset_parameters(self) -> None:
        self.attention.reset_parameters()
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.up.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02 * self.depth_scale)

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = hidden + self.attention(hidden)
        return hidden + self.down(F.silu(self.gate(hidden)) * self.up(hidden))


class CoexistenceLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token = nn.Embedding(VOCAB, HIDDEN)
        self.position = nn.Embedding(CONTEXT, HIDDEN)
        depth_scale = 1.0 / math.sqrt(2 * LAYERS)
        self.blocks = nn.ModuleList(Block(depth_scale) for _ in range(LAYERS))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position.weight, mean=0.0, std=0.02)
        for block in self.blocks:
            block.reset_parameters()

    def hidden(self, tokens: Tensor) -> Tensor:
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.token(tokens) + self.position(positions)[None, :, :]
        for block in self.blocks:
            hidden = block(hidden)
        return hidden

    def forward(self, tokens: Tensor) -> Tensor:
        return self.hidden(tokens) @ self.token.weight.T

    def binary_logits(self, tokens: Tensor) -> Tensor:
        final = self.hidden(tokens)[:, -1]
        return final @ self.token.weight[[BIT_ZERO, BIT_ONE]].T


def build_model(seed: int, device: torch.device) -> CoexistenceLM:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    return CoexistenceLM().to(device)


def make_algebra_batch(
    world: t1b.MixedWorld,
    count: int,
    seed: int,
    device: torch.device,
    *,
    parity_only: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    latent = torch.randint(0, 2, (count, t1.INPUT_DIM), generator=generator, dtype=torch.uint8)
    observed = (
        latent.to(torch.int16) @ t1b.mixing_tensor(world).to(torch.int16)
    ).remainder(2).to(torch.uint8)
    if parity_only:
        parity_route = torch.ones(count, dtype=torch.bool)
    else:
        parity_route = torch.randint(0, 2, (count,), generator=generator).bool()
    parity = latent[:, list(world.latent_support)].sum(dim=1).remainder(2)
    copied = observed[:, 0]
    target = torch.where(parity_route, parity, copied).long()
    bit_tokens = torch.where(
        observed.bool(),
        torch.full_like(observed, BIT_ONE, dtype=torch.long),
        torch.full_like(observed, BIT_ZERO, dtype=torch.long),
    )
    query = torch.where(
        parity_route,
        torch.full((count,), PARITY_QUERY, dtype=torch.long),
        torch.full((count,), COPY_QUERY, dtype=torch.long),
    )
    tokens = torch.cat((bit_tokens, query[:, None]), dim=1)
    return tokens.to(device), target.to(device), parity_route.to(device)


def solve_prefix(world: t1b.MixedWorld, device: torch.device) -> tuple[tuple[int, ...], int, int]:
    tokens, target, _ = make_algebra_batch(
        world,
        t1.PREFIX_EXAMPLES,
        world.seed + 660_000,
        device,
        parity_only=True,
    )
    matrix = (tokens[:, : t1.INPUT_DIM] == BIT_ONE).to(torch.uint8).cpu()
    solution, rank, xor_bit_ops = t1.solve_gf2(matrix, target.to(torch.uint8).cpu())
    if solution is None:
        raise RuntimeError(f"LM prefix solve failed at rank {rank}")
    support = tuple(torch.nonzero(solution, as_tuple=False).flatten().tolist())
    return support, rank, xor_bit_ops


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


@torch.no_grad()
def compile_partition(model: CoexistenceLM, support: tuple[int, ...]) -> tuple[FrozenEntries, dict[str, object]]:
    named = dict(model.named_parameters())
    masks = {name: torch.zeros_like(parameter, dtype=torch.bool) for name, parameter in named.items()}
    compiler_dims = LEAF_START + len(support)

    def fix(name: str, index: object, value: float = 0.0) -> None:
        named[name][index] = value
        masks[name][index] = True

    fix("token.weight", (slice(0, BASE_VOCAB), slice(0, compiler_dims)))
    fix("token.weight", (slice(BASE_VOCAB, VOCAB), slice(None)))
    named["token.weight"][BIT_ZERO, BIT] = 1.0
    named["token.weight"][BIT_ONE, BIT] = -1.0
    output_scale = 30.0
    named["token.weight"][BIT_ZERO, PARITY_FEATURE] = output_scale
    named["token.weight"][BIT_ONE, PARITY_FEATURE] = -output_scale
    named["token.weight"][PARITY_QUERY, CONST] = 1.0
    named["token.weight"][PARITY_QUERY, ROUTE_SIGN] = 1.0
    named["token.weight"][PARITY_QUERY, PARITY_ROUTE] = 1.0
    named["token.weight"][COPY_QUERY, CONST] = 1.0
    named["token.weight"][COPY_QUERY, ROUTE_SIGN] = -1.0

    fix("position.weight", (slice(None), slice(0, compiler_dims)))
    leaf_coordinates = tuple(range(LEAF_START, LEAF_START + len(support)))
    for position, coordinate in zip(support, leaf_coordinates):
        named["position.weight"][position, coordinate] = 1.0
    named["position.weight"][: t1.INPUT_DIM, PARITY_KEY] = 1.0
    named["position.weight"][0, COPY_KEY] = 1.0
    named["position.weight"][t1.INPUT_DIM, PARITY_KEY] = -1.0
    named["position.weight"][t1.INPUT_DIM, COPY_KEY] = -1.0

    # Isolate the compiler hidden coordinates in every layer.  Unmasked
    # language-language entries retain their paired random initialization.
    for layer_index in range(LAYERS):
        layer_prefix = f"blocks.{layer_index}."
        for projection in ("attention.q.weight", "attention.k.weight", "attention.v.weight"):
            fix(layer_prefix + projection, (slice(None), slice(0, compiler_dims)))
        fix(layer_prefix + "attention.o.weight", (slice(0, compiler_dims), slice(None)))
        for projection in ("gate.weight", "up.weight"):
            fix(layer_prefix + projection, (slice(None), slice(0, compiler_dims)))
        fix(layer_prefix + "down.weight", (slice(0, compiler_dims), slice(None)))

    # Block 0 routes each selected bit into its own position-indicator
    # coordinate: residual indicator + bit*indicator = 1 + bit.
    routing = model.blocks[0]
    routing_channels = 2 * len(support)
    for projection in ("gate.weight", "up.weight"):
        fix(f"blocks.0.{projection}", (slice(0, routing_channels), slice(None)))
    fix(
        "blocks.0.down.weight",
        (slice(compiler_dims, None), slice(0, routing_channels)),
    )
    for index, coordinate in enumerate(leaf_coordinates):
        channel = 2 * index
        routing.gate.weight[channel, BIT] = 1.0
        routing.gate.weight[channel + 1, BIT] = -1.0
        routing.up.weight[channel, coordinate] = 1.0
        routing.up.weight[channel + 1, coordinate] = 1.0
        routing.down.weight[coordinate, channel] = 1.0
        routing.down.weight[coordinate, channel + 1] = -1.0

    # Block 1 gathers all leaf coordinates in one causal head and copies the
    # first bit in a second head.  The query constant makes copy-route leaves
    # equal one, so the following product tree evaluates to zero there.
    gather = model.blocks[1]
    parity_qk_component = HEAD_DIM - 1
    copy_component = HEAD_DIM
    value_components = tuple(range(len(support)))
    for projection, rows in (
        ("attention.q.weight", (parity_qk_component, copy_component)),
        ("attention.k.weight", (parity_qk_component, copy_component)),
        ("attention.v.weight", value_components + (copy_component,)),
    ):
        for row in rows:
            fix(f"blocks.1.{projection}", (row, slice(None)))
    gap = 30.0
    gather.attention.q.weight[parity_qk_component, ROUTE_SIGN] = gap * math.sqrt(HEAD_DIM)
    gather.attention.k.weight[parity_qk_component, PARITY_KEY] = 1.0
    gather.attention.q.weight[copy_component, ROUTE_SIGN] = -gap * math.sqrt(HEAD_DIM)
    gather.attention.k.weight[copy_component, COPY_KEY] = 1.0
    gather.attention.v.weight[copy_component, BIT] = 1.0
    for component, coordinate in zip(value_components, leaf_coordinates):
        gather.attention.v.weight[component, coordinate] = 1.0
        gather.attention.v.weight[component, CONST] = 1.0 / t1.INPUT_DIM
        gather.attention.o.weight[coordinate, component] = float(t1.INPUT_DIM)
        fix(
            "blocks.1.attention.o.weight",
            (slice(compiler_dims, None), component),
        )
    gather.attention.o.weight[COPY_VALUE, copy_component] = 1.0
    fix(
        "blocks.1.attention.o.weight",
        (slice(compiler_dims, None), copy_component),
    )

    # Blocks 2 onward form an exact BF16-stable product tree.  Four channels
    # overwrite each left node with left*right: two for multiplication and two
    # for subtracting the old residual value.
    nodes = [(coordinate, True) for coordinate in leaf_coordinates]
    product_layers = 0
    maximum_product_channels = 0
    while len(nodes) > 1:
        layer_index = 2 + product_layers
        if layer_index >= LAYERS:
            raise RuntimeError("insufficient depth for compiled product tree")
        block = model.blocks[layer_index]
        pairs = len(nodes) // 2
        channels = 4 * pairs
        maximum_product_channels = max(maximum_product_channels, channels)
        for projection in ("gate.weight", "up.weight"):
            fix(
                f"blocks.{layer_index}.{projection}",
                (slice(0, channels), slice(None)),
            )
        fix(
            f"blocks.{layer_index}.down.weight",
            (slice(compiler_dims, None), slice(0, channels)),
        )
        next_nodes: list[tuple[int, bool]] = []
        for pair in range(pairs):
            left, left_needs_centering = nodes[2 * pair]
            right, right_needs_centering = nodes[2 * pair + 1]
            channel = 4 * pair
            block.gate.weight[channel, left] = 1.0
            block.gate.weight[channel + 1, left] = -1.0
            block.up.weight[channel, right] = 1.0
            block.up.weight[channel + 1, right] = 1.0
            if left_needs_centering:
                block.gate.weight[channel, CONST] = -1.0
                block.gate.weight[channel + 1, CONST] = 1.0
            if right_needs_centering:
                block.up.weight[channel, CONST] = -1.0
                block.up.weight[channel + 1, CONST] = -1.0
            block.down.weight[left, channel] = 1.0
            block.down.weight[left, channel + 1] = -1.0
            # -old_left = -[silu(1)-silu(-1)]*old_left.
            block.gate.weight[channel + 2, CONST] = 1.0
            block.gate.weight[channel + 3, CONST] = -1.0
            block.up.weight[channel + 2, left] = 1.0
            block.up.weight[channel + 3, left] = 1.0
            block.down.weight[left, channel + 2] = -1.0
            block.down.weight[left, channel + 3] = 1.0
            next_nodes.append((left, False))
        if len(nodes) % 2:
            next_nodes.append(nodes[-1])
        nodes = next_nodes
        product_layers += 1

    final_coordinate, final_needs_centering = nodes[0]
    if final_needs_centering:
        raise RuntimeError("uncombined centered leaf remained after product tree")
    # A final FFN copies parity or protected copy into PARITY_FEATURE.  Keeping
    # the tied output weights on this otherwise-unused coordinate avoids
    # corrupting the input leaf coordinates of the bit-token embeddings.
    output_layer_index = 2 + product_layers
    if output_layer_index >= LAYERS:
        raise RuntimeError("insufficient depth for compiled output projection")
    output_block = model.blocks[output_layer_index]
    output_channels = 4
    for projection in ("gate.weight", "up.weight"):
        fix(
            f"blocks.{output_layer_index}.{projection}",
            (slice(0, output_channels), slice(None)),
        )
    fix(
        f"blocks.{output_layer_index}.down.weight",
        (slice(compiler_dims, None), slice(0, output_channels)),
    )
    for channel, source in ((0, final_coordinate), (2, COPY_VALUE)):
        output_block.gate.weight[channel, CONST] = 1.0
        output_block.gate.weight[channel + 1, CONST] = -1.0
        output_block.up.weight[channel, source] = 1.0
        output_block.up.weight[channel + 1, source] = 1.0
        output_block.down.weight[PARITY_FEATURE, channel] = 1.0
        output_block.down.weight[PARITY_FEATURE, channel + 1] = -1.0

    values = {name: parameter.detach().clone() for name, parameter in named.items()}
    frozen = FrozenEntries(masks, values, sum(int(mask.sum().item()) for mask in masks.values()))
    frozen.enforce(model)
    return frozen, {
        "support_size": len(support),
        "reserved_hidden_coordinates": compiler_dims,
        "leaf_coordinates": leaf_coordinates,
        "product_layers": product_layers,
        "final_product_coordinate": final_coordinate,
        "output_layer": output_layer_index,
        "reserved_routing_channels": routing_channels,
        "maximum_reserved_product_channels": maximum_product_channels,
        "frozen_parameter_entries": frozen.count,
        "bf16_stable_product_tree": True,
    }


def parameter_count() -> int:
    return sum(parameter.numel() for parameter in CoexistenceLM().parameters())


@torch.no_grad()
def evaluate_algebra(
    model: CoexistenceLM,
    world: t1b.MixedWorld,
    device: torch.device,
    seed: int,
    count: int = 4_096,
) -> dict[str, float]:
    model.eval()
    tokens, target, parity_route = make_algebra_batch(world, count, seed, device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model.binary_logits(tokens).float()
    predicted = logits.argmax(dim=-1)
    correct = predicted == target
    copy_route = ~parity_route
    return {
        "accuracy": float(correct.float().mean().item()),
        "parity_accuracy": float(correct[parity_route].float().mean().item()),
        "protected_accuracy": float(correct[copy_route].float().mean().item()),
        "cross_entropy": float(F.cross_entropy(logits, target).item()),
    }


@torch.no_grad()
def evaluate_natural(
    model: CoexistenceLM,
    stream: TokenStream,
    device: torch.device,
    seed: int,
    batches: int,
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


def train_loss(
    model: CoexistenceLM,
    step: int,
    world: t1b.MixedWorld,
    natural: TokenStream,
    device: torch.device,
) -> tuple[Tensor, str, int]:
    if step % ALGEBRA_INTERVAL == 0:
        tokens, target, _ = make_algebra_batch(
            world, ALGEBRA_BATCH, world.seed * 1_000_003 + step, device
        )
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model.binary_logits(tokens)
        return F.cross_entropy(logits.float(), target), "algebra", ALGEBRA_BATCH * (t1.INPUT_DIM + 1)
    inputs, targets = natural.batch(
        world.seed * 1_000_003 + step, NATURAL_BATCH, CONTEXT, device
    )
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(inputs)
    loss = F.cross_entropy(logits.float().reshape(-1, VOCAB), targets.reshape(-1))
    return loss, "natural", NATURAL_BATCH * CONTEXT


def scheduled_learning_rate(step: int) -> float:
    if step <= WARMUP_STEPS:
        return PEAK_LEARNING_RATE * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / (SCHEDULE_STEPS - WARMUP_STEPS)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return PEAK_LEARNING_RATE * (0.1 + 0.9 * cosine)


def train_arm(
    arm: str,
    seed: int,
    world: t1b.MixedWorld,
    natural: TokenStream,
    validation: TokenStream,
    device: torch.device,
    steps: int,
) -> dict[str, object]:
    model = build_model(seed + 20_000, device)
    initial_hash = state_sha256(model)
    frozen: FrozenEntries | None = None
    compiler: dict[str, object] | None = None
    recovered_support: tuple[int, ...] | None = None
    rank: int | None = None
    xor_bit_ops: int | None = None
    if arm == "compiler_1x":
        recovered_support, rank, xor_bit_ops = solve_prefix(world, device)
        frozen, compiler = compile_partition(model, recovered_support)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=PEAK_LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.0
    )
    prefix_losses: list[float] = []
    if arm != "compiler_1x":
        for prefix_step in range(4):
            for group in optimizer.param_groups:
                group["lr"] = scheduled_learning_rate(prefix_step + 1)
            tokens, target, _ = make_algebra_batch(
                world, ALGEBRA_BATCH, world.seed + 760_000 + prefix_step, device, parity_only=True
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model.binary_logits(tokens)
            loss = F.cross_entropy(logits.float(), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            prefix_losses.append(float(loss.item()))

    checkpoints: dict[str, object] = {}
    maximum_gradient_norm = 0.0
    maximum_loss = 0.0
    natural_tokens = 0
    algebra_tokens = t1.PREFIX_EXAMPLES * (t1.INPUT_DIM + 1)
    started = time.perf_counter()
    checkpoint_set = set(CHECKPOINTS_1X) | ({STEPS_2X} if steps == STEPS_2X else set())
    for step in range(1, steps + 1):
        model.train()
        for group in optimizer.param_groups:
            group["lr"] = scheduled_learning_rate(step)
        optimizer.zero_grad(set_to_none=True)
        loss, kind, tokens = train_loss(model, step, world, natural, device)
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
        if kind == "natural":
            natural_tokens += tokens
        else:
            algebra_tokens += tokens
        if step in checkpoint_set:
            checkpoints[str(step)] = {
                "natural": evaluate_natural(
                    model, validation, device, seed + 860_000, batches=32
                ),
                "algebra": evaluate_algebra(
                    model, world, device, seed + 960_000 + step
                ),
                "natural_tokens": natural_tokens,
                "algebra_tokens": algebra_tokens,
            }
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {
        "arm": arm,
        "seed": seed,
        "steps": steps,
        "initial_hash": initial_hash,
        "recovered_support": recovered_support,
        "rank": rank,
        "xor_bit_ops": xor_bit_ops,
        "support_exact": recovered_support == world.observed_support if recovered_support else None,
        "compiler": compiler,
        "prefix_losses": prefix_losses,
        "checkpoints": checkpoints,
        "maximum_gradient_norm": maximum_gradient_norm,
        "maximum_loss": maximum_loss,
        "terminal_learning_rate": scheduled_learning_rate(steps),
        "elapsed_seconds": time.perf_counter() - started,
        "terminal_hash": state_sha256(model),
    }


def run(device: torch.device, quick: bool = False) -> dict[str, object]:
    natural = TokenStream(TRAIN_FILE)
    validation = TokenStream(VALIDATION_FILE)
    worlds: dict[str, object] = {}
    for seed in MODEL_SEEDS:
        world = t1b.make_mixed_world(seed)
        if quick:
            candidate_model = build_model(seed + 20_000, device)
            initial_hash = state_sha256(candidate_model)
            support, rank, xor_bit_ops = solve_prefix(world, device)
            frozen, compiler = compile_partition(candidate_model, support)
            frozen.enforce(candidate_model)
            worlds[str(seed)] = {
                "quick": {
                    "initial_hash": initial_hash,
                    "support_exact": support == world.observed_support,
                    "rank": rank,
                    "xor_bit_ops": xor_bit_ops,
                    "compiler": compiler,
                    "algebra": evaluate_algebra(candidate_model, world, device, seed + 1_060_000),
                }
            }
            continue
        baseline = train_arm(
            "baseline_1x", seed, world, natural, validation, device, CHECKPOINTS_1X[-1]
        )
        candidate = train_arm(
            "compiler_1x", seed, world, natural, validation, device, CHECKPOINTS_1X[-1]
        )
        doubled = train_arm(
            "baseline_2x", seed, world, natural, validation, device, STEPS_2X
        )
        worlds[str(seed)] = {
            "observed_support": world.observed_support,
            "arms": {
                "baseline_1x": baseline,
                "compiler_1x": candidate,
                "baseline_2x": doubled,
            },
            "identical_initial_hashes": len(
                {baseline["initial_hash"], candidate["initial_hash"], doubled["initial_hash"]}
            ) == 1,
        }

    gates: dict[str, bool] = {}
    if not quick:
        for seed, result in worlds.items():
            arms = result["arms"]
            baseline = arms["baseline_1x"]
            candidate = arms["compiler_1x"]
            doubled = arms["baseline_2x"]
            gates[f"{seed}_same_initialization"] = result["identical_initial_hashes"]
            gates[f"{seed}_support_exact"] = candidate["support_exact"] is True
            for checkpoint in CHECKPOINTS_1X:
                key = str(checkpoint)
                candidate_algebra = candidate["checkpoints"][key]["algebra"]
                gates[f"{seed}_{key}_candidate_parity_ge_99"] = candidate_algebra["parity_accuracy"] >= 0.99
                gates[f"{seed}_{key}_candidate_protected_ge_99"] = candidate_algebra["protected_accuracy"] >= 0.99
                candidate_nll = candidate["checkpoints"][key]["natural"]["nll"]
                baseline_nll = baseline["checkpoints"][key]["natural"]["nll"]
                gates[f"{seed}_{key}_natural_within_0p5pct"] = candidate_nll <= 1.005 * baseline_nll
            doubled_algebra = doubled["checkpoints"][str(STEPS_2X)]["algebra"]
            gates[f"{seed}_baseline2x_parity_lt_80"] = doubled_algebra["parity_accuracy"] < 0.80
            gates[f"{seed}_baseline2x_protected_ge_95"] = doubled_algebra["protected_accuracy"] >= 0.95
            candidate_nlls = [
                candidate["checkpoints"][str(checkpoint)]["natural"]["nll"]
                for checkpoint in CHECKPOINTS_1X
            ]
            gates[f"{seed}_candidate_terminal_improves"] = candidate_nlls[-1] <= candidate_nlls[-2]
            gates[f"{seed}_finite_gradient_envelope"] = all(
                arm["maximum_gradient_norm"] < 100 and arm["maximum_loss"] < 100
                for arm in arms.values()
            )

    result = {
        "schema": "heterogeneous-algebraic-compilation-t2-lm-v1",
        "status": "quick" if quick else "complete",
        "device": str(device),
        "torch_version": torch.__version__,
        "configuration": {
            "base_vocab": BASE_VOCAB,
            "vocab": VOCAB,
            "hidden": HIDDEN,
            "layers": LAYERS,
            "heads": HEADS,
            "ffn_width": FFN_WIDTH,
            "context": CONTEXT,
            "parameter_count": parameter_count(),
            "model_seeds": MODEL_SEEDS,
            "checkpoints_1x": CHECKPOINTS_1X,
            "steps_2x": STEPS_2X,
            "peak_learning_rate": PEAK_LEARNING_RATE,
            "warmup_steps": WARMUP_STEPS,
            "schedule_steps": SCHEDULE_STEPS,
        },
        "worlds": worlds,
        "gates": gates,
        "all_gates_pass": bool(gates) and all(gates.values()),
        "integrity": {
            "source_sha256": sha256_file(Path(__file__)),
            "test_sha256": sha256_file(TEST_SOURCE),
            "preregistration_sha256": sha256_file(PREREGISTRATION),
            "predecessor_sha256": sha256_file(PREDECESSOR),
            "train_sha256": sha256_file(TRAIN_FILE),
            "validation_sha256": sha256_file(VALIDATION_FILE),
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = run(torch.device(args.device), quick=args.quick)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "all_gates_pass": result["all_gates_pass"]}, indent=2))


if __name__ == "__main__":
    main()
