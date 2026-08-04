#!/usr/bin/env python3
"""Causal Transformer transfer gate for heterogeneous algebraic compilation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import heterogeneous_algebraic_compilation_t1 as t1
from experiments import heterogeneous_algebraic_compilation_t1b as t1b


PREREGISTRATION = ROOT / "results/heterogeneous-algebraic-compilation-t1c-preregistration.md"
TEST_SOURCE = ROOT / "tests/test_heterogeneous_algebraic_compilation_t1c.py"
PREDECESSOR = ROOT / "results/heterogeneous-algebraic-compilation-t1b.json"
OUTPUT = ROOT / "results/heterogeneous-algebraic-compilation-t1c.json"

VOCAB_SIZE = 4
PARITY_QUERY = 2
COPY_QUERY = 3
CONTEXT = t1.INPUT_DIM
SEQUENCE = CONTEXT + 1
HIDDEN = 64
HEADS = 4
HEAD_DIM = HIDDEN // HEADS
FFN_WIDTH = 128
CHECKPOINTS = (4, 8, 500, 2000)
ADAMW_LRS = (1e-3, 3e-3)
MUON_LRS = (1e-3, 3e-3)
RADEMACHER_MULTIPLIERS = (0.5, 1.0)

CONST = 0
BIT = 1
ROUTE_SIGN = 2
PARITY_ROUTE = 3
PARITY_KEY = 4
COPY_KEY = 5
PARITY_AVERAGE = 6
COPY_VALUE = 7
PARITY_FEATURE = 8


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


class CausalAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.k = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.v = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.o = nn.Linear(HIDDEN, HIDDEN, bias=False)

    def forward(self, hidden: Tensor) -> Tensor:
        batch, length, _ = hidden.shape
        query = self.q(hidden).view(batch, length, HEADS, HEAD_DIM).transpose(1, 2)
        key = self.k(hidden).view(batch, length, HEADS, HEAD_DIM).transpose(1, 2)
        value = self.v(hidden).view(batch, length, HEADS, HEAD_DIM).transpose(1, 2)
        scores = query @ key.transpose(-1, -2) / math.sqrt(HEAD_DIM)
        causal = torch.ones(length, length, dtype=torch.bool, device=hidden.device).triu(1)
        scores = scores.masked_fill(causal, float("-inf"))
        probabilities = scores.softmax(dim=-1)
        attended = probabilities @ value
        attended = attended.transpose(1, 2).contiguous().view(batch, length, HIDDEN)
        return self.o(attended)


class CausalSwiGLUDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token = nn.Embedding(VOCAB_SIZE, HIDDEN)
        self.position = nn.Embedding(SEQUENCE, HIDDEN)
        self.attention = CausalAttention()
        self.gate = nn.Linear(HIDDEN, FFN_WIDTH, bias=False)
        self.up = nn.Linear(HIDDEN, FFN_WIDTH, bias=False)
        self.down = nn.Linear(FFN_WIDTH, HIDDEN, bias=False)
        self.head = nn.Linear(HIDDEN, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in self.parameters():
            nn.init.normal_(parameter, mean=0.0, std=0.02)

    def forward(self, tokens: Tensor) -> Tensor:
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.token(tokens) + self.position(positions)[None, :, :]
        hidden = hidden + self.attention(hidden)
        hidden = hidden + self.down(F.silu(self.gate(hidden)) * self.up(hidden))
        return self.head(hidden[:, -1]).squeeze(-1)


def build_model(seed: int, device: torch.device) -> CausalSwiGLUDecoder:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    return CausalSwiGLUDecoder().to(device)


def make_batch(
    world: t1b.MixedWorld,
    count: int,
    seed: int,
    device: torch.device,
    *,
    parity_only: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    latent = torch.randint(0, 2, (count, CONTEXT), generator=generator, dtype=torch.uint8)
    observed = (
        latent.to(torch.int16) @ t1b.mixing_tensor(world).to(torch.int16)
    ).remainder(2).to(torch.uint8)
    if parity_only:
        parity_route = torch.ones(count, dtype=torch.bool)
    else:
        parity_route = torch.randint(0, 2, (count,), generator=generator).bool()
    parity = latent[:, list(world.latent_support)].sum(dim=1).remainder(2)
    copied = observed[:, 0]
    target = torch.where(parity_route, parity, copied)
    query = torch.where(
        parity_route,
        torch.full((count,), PARITY_QUERY, dtype=torch.uint8),
        torch.full((count,), COPY_QUERY, dtype=torch.uint8),
    )
    tokens = torch.cat((observed, query[:, None]), dim=1).long()
    target_sign = 1.0 - 2.0 * target.float()
    return tokens.to(device), target_sign.to(device), parity_route.to(device)


def solve_prefix(
    world: t1b.MixedWorld, device: torch.device
) -> tuple[tuple[int, ...], int, int]:
    tokens, target, _ = make_batch(
        world,
        t1.PREFIX_EXAMPLES,
        world.seed + 360_000,
        device,
        parity_only=True,
    )
    matrix = tokens[:, :CONTEXT].to(torch.uint8).cpu()
    rhs = (target < 0).to(torch.uint8).cpu()
    solution, rank, xor_bit_ops = t1.solve_gf2(matrix, rhs)
    if solution is None:
        raise RuntimeError(f"causal prefix failed at rank {rank}")
    support = tuple(torch.nonzero(solution, as_tuple=False).flatten().tolist())
    return support, rank, xor_bit_ops


def interpolation_coefficients(support_size: int) -> tuple[Tensor, Tensor, Tensor]:
    nodes = torch.linspace(-1.0, 1.0, support_size + 1, dtype=torch.float64)
    thresholds = (nodes[:-1] + nodes[1:]) / 2
    alpha = torch.tensor(40.0, dtype=torch.float64)
    features = torch.empty(support_size + 1, support_size + 1, dtype=torch.float64)
    features[:, 0] = F.silu(torch.ones_like(nodes))
    features[:, 1:] = F.silu(alpha * (nodes[:, None] - thresholds[None, :]))
    counts = torch.arange(support_size, -1, -1)
    targets = torch.where(counts.remainder(2) == 0, 1.0, -1.0).to(torch.float64)
    coefficients = torch.linalg.solve(features, targets)
    return thresholds, coefficients, features @ coefficients


@torch.no_grad()
def compile_causal(model: CausalSwiGLUDecoder, support: tuple[int, ...]) -> dict[str, object]:
    if not support:
        raise ValueError("empty parity support")
    for parameter in model.parameters():
        parameter.zero_()

    model.token.weight[0, BIT] = 1.0
    model.token.weight[1, BIT] = -1.0
    model.token.weight[PARITY_QUERY, CONST] = 1.0
    model.token.weight[PARITY_QUERY, ROUTE_SIGN] = 1.0
    model.token.weight[PARITY_QUERY, PARITY_ROUTE] = 1.0
    model.token.weight[COPY_QUERY, CONST] = 1.0
    model.token.weight[COPY_QUERY, ROUTE_SIGN] = -1.0

    model.position.weight[list(support), PARITY_KEY] = 1.0
    model.position.weight[0, COPY_KEY] = 1.0
    model.position.weight[CONTEXT, PARITY_KEY] = -1.0
    model.position.weight[CONTEXT, COPY_KEY] = -1.0

    attention_logit = 30.0
    model.attention.q.weight[0, ROUTE_SIGN] = attention_logit * math.sqrt(HEAD_DIM)
    model.attention.k.weight[0, PARITY_KEY] = 1.0
    model.attention.v.weight[0, BIT] = 1.0
    model.attention.o.weight[PARITY_AVERAGE, 0] = 1.0

    second_head = HEAD_DIM
    model.attention.q.weight[second_head, ROUTE_SIGN] = -attention_logit * math.sqrt(HEAD_DIM)
    model.attention.k.weight[second_head, COPY_KEY] = 1.0
    model.attention.v.weight[second_head, BIT] = 1.0
    model.attention.o.weight[COPY_VALUE, second_head] = 1.0

    thresholds, coefficients, fitted = interpolation_coefficients(len(support))
    model.gate.weight[0, CONST] = 1.0
    model.up.weight[0, PARITY_ROUTE] = 1.0
    model.down.weight[PARITY_FEATURE, 0] = coefficients[0].float()
    for index, (threshold, coefficient) in enumerate(
        zip(thresholds, coefficients[1:]), start=1
    ):
        model.gate.weight[index, PARITY_AVERAGE] = 40.0
        model.gate.weight[index, CONST] = float(-40.0 * threshold)
        model.up.weight[index, PARITY_ROUTE] = 1.0
        model.down.weight[PARITY_FEATURE, index] = coefficient.float()

    model.head.weight[0, PARITY_FEATURE] = 8.0
    model.head.weight[0, COPY_VALUE] = 8.0
    return {
        "support_size": len(support),
        "interpolation_channels": len(support) + 1,
        "interpolation_max_abs_error_float64": float(
            (fitted - torch.where(
                torch.arange(len(support), -1, -1).remainder(2) == 0,
                1.0,
                -1.0,
            )).abs().max().item()
        ),
        "attention_logit_gap": attention_logit,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    world: t1b.MixedWorld,
    device: torch.device,
    seed: int,
    count: int = 8_192,
) -> dict[str, float]:
    tokens, target, parity_route = make_batch(world, count, seed, device)
    logits = model(tokens)
    correct = (logits >= 0) == (target >= 0)
    copy_route = ~parity_route
    return {
        "accuracy": float(correct.float().mean().item()),
        "parity_accuracy": float(correct[parity_route].float().mean().item()),
        "protected_accuracy": float(correct[copy_route].float().mean().item()),
        "bce": float(F.binary_cross_entropy_with_logits(logits, (target + 1) / 2).item()),
    }


@torch.no_grad()
def rademacher_rebirth(model: CausalSwiGLUDecoder, seed: int, multiplier: float) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for parameter in model.parameters():
        fan_in = parameter.shape[1]
        values = torch.randint(0, 2, parameter.shape, generator=generator).float()
        values = values.mul_(2).sub_(1).mul_(multiplier / math.sqrt(fan_in))
        parameter.copy_(values.to(parameter.device, parameter.dtype))


def train_control(
    family: str,
    lr: float,
    world: t1b.MixedWorld,
    model_seed: int,
    device: torch.device,
    *,
    multiplier: float | None = None,
) -> dict[str, object]:
    model = build_model(model_seed, device)
    initial_hash = state_sha256(model)
    if family == "rademacher_hinge":
        assert multiplier is not None
        rademacher_rebirth(model, world.seed + int(multiplier * 1000), multiplier)
    if family == "muon":
        optimizer = torch.optim.Muon(
            model.parameters(),
            lr=lr,
            weight_decay=0.0,
            adjust_lr_fn="match_rms_adamw",
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    checkpoints: dict[str, dict[str, float]] = {}
    started = time.perf_counter()
    for step in range(1, CHECKPOINTS[-1] + 1):
        tokens, target, _ = make_batch(
            world, 64, world.seed * 1_000_003 + step, device
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens)
        if family == "rademacher_hinge":
            loss = F.relu(1.0 - target * logits).mean()
        else:
            loss = F.binary_cross_entropy_with_logits(logits, (target + 1) / 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in CHECKPOINTS:
            checkpoints[str(step)] = evaluate(
                model, world, device, world.seed + 460_000 + step
            )
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {
        "family": family,
        "lr": lr,
        "multiplier": multiplier,
        "initial_hash": initial_hash,
        "checkpoints": checkpoints,
        "elapsed_seconds": time.perf_counter() - started,
        "examples": CHECKPOINTS[-1] * 64,
    }


def best_run(runs: list[dict[str, object]]) -> dict[str, object]:
    return max(
        runs,
        key=lambda run: (
            run["checkpoints"][str(CHECKPOINTS[-1])]["accuracy"],
            run["checkpoints"][str(CHECKPOINTS[-1])]["parity_accuracy"],
        ),
    )


def candidate_run(
    world: t1b.MixedWorld, model_seed: int, device: torch.device
) -> dict[str, object]:
    model = build_model(model_seed, device)
    initial_hash = state_sha256(model)
    started = time.perf_counter()
    support, rank, xor_bit_ops = solve_prefix(world, device)
    compilation = compile_causal(model, support)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {
        "initial_hash": initial_hash,
        "rank": rank,
        "xor_bit_ops": xor_bit_ops,
        "recovered_support": support,
        "support_exact": support == world.observed_support,
        "compilation": compilation,
        "elapsed_seconds": time.perf_counter() - started,
        "evaluation": evaluate(model, world, device, world.seed + 560_000),
    }


def parameter_count() -> int:
    return sum(parameter.numel() for parameter in CausalSwiGLUDecoder().parameters())


def run(device: torch.device, quick: bool = False) -> dict[str, object]:
    worlds: dict[str, object] = {}
    for seed in t1.WORLD_SEEDS:
        world = t1b.make_mixed_world(seed)
        model_seed = seed + 9_000
        candidate = candidate_run(world, model_seed, device)
        families: dict[str, object] = {}
        if not quick:
            specifications = {
                "adamw": [(lr, None) for lr in ADAMW_LRS],
                "muon": [(lr, None) for lr in MUON_LRS],
                "rademacher_hinge": [
                    (lr, multiplier)
                    for multiplier in RADEMACHER_MULTIPLIERS
                    for lr in ADAMW_LRS
                ],
            }
            for family, configurations in specifications.items():
                runs = [
                    train_control(
                        family,
                        lr,
                        world,
                        model_seed,
                        device,
                        multiplier=multiplier,
                    )
                    for lr, multiplier in configurations
                ]
                families[family] = {"runs": runs, "best": best_run(runs)}
        hashes = [candidate["initial_hash"]]
        for family in families.values():
            hashes.extend(control["initial_hash"] for control in family["runs"])
        worlds[str(seed)] = {
            "observed_support": world.observed_support,
            "candidate": candidate,
            "controls": families,
            "identical_initial_hashes": len(set(hashes)) == 1,
        }

    gates: dict[str, bool] = {}
    if not quick:
        for seed, result in worlds.items():
            candidate = result["candidate"]
            evaluation = candidate["evaluation"]
            gates[f"{seed}_candidate_full_ge_99"] = evaluation["accuracy"] >= 0.99
            gates[f"{seed}_candidate_parity_ge_99"] = evaluation["parity_accuracy"] >= 0.99
            gates[f"{seed}_candidate_protected_ge_99"] = evaluation["protected_accuracy"] >= 0.99
            gates[f"{seed}_support_exact"] = candidate["support_exact"]
            gates[f"{seed}_same_initialization"] = result["identical_initial_hashes"]
            best = [family["best"] for family in result["controls"].values()]
            gates[f"{seed}_controls_step8_parity_lt_80"] = all(
                control["checkpoints"]["8"]["parity_accuracy"] < 0.80
                for control in best
            )
            gates[f"{seed}_controls_step2000_parity_lt_80"] = all(
                control["checkpoints"]["2000"]["parity_accuracy"] < 0.80
                for control in best
            )
            gates[f"{seed}_some_control_protected_ge_95"] = any(
                control["checkpoints"]["2000"]["protected_accuracy"] >= 0.95
                for control in best
            )

    result = {
        "schema": "heterogeneous-algebraic-compilation-t1c-v1",
        "status": "quick" if quick else "complete",
        "device": str(device),
        "torch_version": torch.__version__,
        "configuration": {
            "vocab_size": VOCAB_SIZE,
            "context": CONTEXT,
            "hidden": HIDDEN,
            "heads": HEADS,
            "ffn_width": FFN_WIDTH,
            "parameter_count": parameter_count(),
            "world_seeds": t1.WORLD_SEEDS,
            "checkpoints": CHECKPOINTS,
        },
        "ledger": {
            "candidate_dense_training_steps": 0,
            "candidate_prefix_examples": t1.PREFIX_EXAMPLES,
            "control_examples_per_run": CHECKPOINTS[-1] * 64,
            "candidate_parameter_writes_upper_bound": parameter_count(),
            "same_inference_graph": True,
        },
        "worlds": worlds,
        "gates": gates,
        "all_gates_pass": bool(gates) and all(gates.values()),
        "integrity": {
            "source_sha256": sha256_file(Path(__file__)),
            "test_sha256": sha256_file(TEST_SOURCE),
            "preregistration_sha256": sha256_file(PREREGISTRATION),
            "predecessor_sha256": sha256_file(PREDECESSOR),
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

