#!/usr/bin/env python3
"""Adversarial T1b: dense GF(2) mixing, raw recurrence, Rademacher controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import heterogeneous_algebraic_compilation_t1 as base


PREREGISTRATION = ROOT / "results/heterogeneous-algebraic-compilation-t1b-preregistration.md"
TEST_SOURCE = ROOT / "tests/test_heterogeneous_algebraic_compilation_t1b.py"
PREDECESSOR = ROOT / "results/heterogeneous-algebraic-compilation-t1.json"
OUTPUT = ROOT / "results/heterogeneous-algebraic-compilation-t1b.json"

MULTIPLIERS = (0.25, 0.5, 1.0)
LEARNING_RATES = (1e-3, 3e-3)
LOSSES = ("bce", "hinge")
MAX_STEPS = 2_000
BATCH_SIZE = 64


@dataclass(frozen=True)
class MixedWorld:
    seed: int
    mixing: tuple[tuple[int, ...], ...]
    latent_support: tuple[int, ...]
    observed_support: tuple[int, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _random_invertible_matrix(seed: int) -> Tensor:
    generator = random.Random(seed)
    matrix = torch.eye(base.INPUT_DIM, dtype=torch.uint8)
    for _ in range(12 * base.INPUT_DIM):
        left, right = generator.sample(range(base.INPUT_DIM), 2)
        if generator.random() < 0.2:
            temporary = matrix[:, left].clone()
            matrix[:, left] = matrix[:, right]
            matrix[:, right] = temporary
        else:
            matrix[:, right] ^= matrix[:, left]
    return matrix


def make_mixed_world(seed: int) -> MixedWorld:
    latent_generator = random.Random(seed + 17)
    latent_support = tuple(
        sorted(latent_generator.sample(range(base.INPUT_DIM), base.SUPPORT_SIZE))
    )
    latent_mask = torch.zeros(base.INPUT_DIM, dtype=torch.uint8)
    latent_mask[list(latent_support)] = 1
    mixing = _random_invertible_matrix(seed + 29)
    observed_mask, rank, _ = base.solve_gf2(mixing, latent_mask)
    if observed_mask is None or rank != base.INPUT_DIM:
        raise RuntimeError("constructed mixing matrix is not invertible")
    observed_support = tuple(
        torch.nonzero(observed_mask, as_tuple=False).flatten().tolist()
    )
    return MixedWorld(
        seed,
        tuple(tuple(int(value) for value in row) for row in mixing.tolist()),
        latent_support,
        observed_support,
    )


def mixing_tensor(world: MixedWorld) -> Tensor:
    return torch.tensor(world.mixing, dtype=torch.uint8)


def make_mixed_batch(
    world: MixedWorld,
    count: int,
    seed: int,
    device: torch.device,
    *,
    route_one_only: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    latent = torch.randint(
        0, 2, (count, base.INPUT_DIM), generator=generator, dtype=torch.uint8
    )
    observed = (latent.to(torch.int16) @ mixing_tensor(world).to(torch.int16)).remainder(2).to(torch.uint8)
    if route_one_only:
        route = torch.ones(count, dtype=torch.long)
    else:
        route = torch.randint(0, 2, (count,), generator=generator)
    parity = latent[:, list(world.latent_support)].sum(dim=1).remainder(2)
    protected = observed[:, 0]
    target = torch.where(route.bool(), parity, protected)
    hidden = torch.zeros(count, base.HIDDEN_DIM)
    hidden[:, : base.INPUT_DIM] = 1.0 - 2.0 * observed.float()
    hidden[:, base.INPUT_DIM] = route.float()
    target_sign = 1.0 - 2.0 * target.float()
    return hidden.to(device), target_sign.to(device), route.to(device)


def _recurrence_windows(mask: Tensor, count: int, seed: int) -> tuple[Tensor, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    state = torch.randint(0, 2, (base.INPUT_DIM,), generator=generator, dtype=torch.uint8)
    windows = torch.empty(count, base.INPUT_DIM, dtype=torch.uint8)
    targets = torch.empty(count, dtype=torch.uint8)
    support = torch.nonzero(mask, as_tuple=False).flatten()
    for index in range(count):
        windows[index] = state
        next_bit = state[support].sum().remainder(2).to(torch.uint8)
        targets[index] = next_bit
        state = torch.cat((state[1:], next_bit[None]))
    return windows, targets


def full_rank_recurrence_prefix(world: MixedWorld) -> tuple[Tensor, Tensor, int]:
    mask = torch.zeros(base.INPUT_DIM, dtype=torch.uint8)
    mask[list(world.observed_support)] = 1
    for attempt in range(512):
        seed = world.seed * 10_000 + attempt
        windows, targets = _recurrence_windows(mask, base.PREFIX_EXAMPLES, seed)
        solution, rank, _ = base.solve_gf2(windows, targets)
        if solution is not None and rank == base.INPUT_DIM:
            return windows, targets, seed
    raise RuntimeError("could not find a cyclic full-rank recurrence state")


def make_recurrence_batch(
    world: MixedWorld,
    count: int,
    seed: int,
    device: torch.device,
    *,
    route_one_only: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    mask = torch.zeros(base.INPUT_DIM, dtype=torch.uint8)
    mask[list(world.observed_support)] = 1
    windows, parity = _recurrence_windows(mask, count, seed)
    generator = torch.Generator(device="cpu").manual_seed(seed + 1_000_000)
    if route_one_only:
        route = torch.ones(count, dtype=torch.long)
    else:
        route = torch.randint(0, 2, (count,), generator=generator)
    protected = windows[:, 0]
    target = torch.where(route.bool(), parity, protected)
    hidden = torch.zeros(count, base.HIDDEN_DIM)
    hidden[:, : base.INPUT_DIM] = 1.0 - 2.0 * windows.float()
    hidden[:, base.INPUT_DIM] = route.float()
    target_sign = 1.0 - 2.0 * target.float()
    return hidden.to(device), target_sign.to(device), route.to(device)


@torch.no_grad()
def rademacher_rebirth(model: base.ResidualSwiGLUParity, seed: int, multiplier: float) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def fill(parameter: Tensor, scale: float) -> None:
        signs = torch.randint(0, 2, parameter.shape, generator=generator, dtype=torch.float32)
        signs = signs.mul_(2).sub_(1).mul_(scale)
        parameter.copy_(signs.to(parameter.device, parameter.dtype))

    for block in model.blocks:
        fill(block.gate.weight, multiplier / math.sqrt(base.HIDDEN_DIM))
        fill(block.up.weight, multiplier / math.sqrt(base.HIDDEN_DIM))
        fill(
            block.down.weight,
            multiplier / math.sqrt(base.FFN_WIDTH * base.DEPTH),
        )
    fill(model.head.weight, multiplier / math.sqrt(base.HIDDEN_DIM))


BatchMaker = Callable[[MixedWorld, int, int, torch.device], tuple[Tensor, Tensor, Tensor]]


@torch.no_grad()
def evaluate(
    model: nn.Module,
    world: MixedWorld,
    batch_maker: BatchMaker,
    device: torch.device,
    seed: int,
    count: int = 8_192,
) -> dict[str, float]:
    hidden, target, route = batch_maker(world, count, seed, device)
    logits = model(hidden)
    correct = (logits >= 0) == (target >= 0)
    parity = route == 1
    protected = ~parity
    return {
        "accuracy": float(correct.float().mean().item()),
        "parity_accuracy": float(correct[parity].float().mean().item()),
        "protected_accuracy": float(correct[protected].float().mean().item()),
        "bce": float(F.binary_cross_entropy_with_logits(logits, (target + 1) / 2).item()),
    }


def candidate(
    world: MixedWorld,
    view: str,
    batch_maker: BatchMaker,
    model_seed: int,
    device: torch.device,
) -> dict[str, object]:
    model = base.build_model(model_seed, device)
    initial_hash = base.state_sha256(model)
    if view == "recurrence":
        windows, targets, stream_seed = full_rank_recurrence_prefix(world)
        hidden = torch.zeros(base.PREFIX_EXAMPLES, base.HIDDEN_DIM, device=device)
        hidden[:, : base.INPUT_DIM] = (1.0 - 2.0 * windows.float()).to(device)
        target = (1.0 - 2.0 * targets.float()).to(device)
    else:
        stream_seed = world.seed + 60_000
        hidden, target, _ = batch_maker(
            world,
            base.PREFIX_EXAMPLES,
            stream_seed,
            device,
            route_one_only=True,
        )
    matrix, rhs = base.signs_to_gf2(hidden, target)
    started = time.perf_counter()
    solution, rank, xor_bit_ops = base.solve_gf2(matrix, rhs)
    if solution is None:
        raise RuntimeError(f"{view} solver failed at rank {rank}")
    recovered = tuple(torch.nonzero(solution, as_tuple=False).flatten().tolist())
    compilation = base.compile_support(model, recovered)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "initial_hash": initial_hash,
        "stream_seed": stream_seed,
        "rank": rank,
        "xor_bit_ops": xor_bit_ops,
        "recovered_support": recovered,
        "support_exact": recovered == world.observed_support,
        "compilation": compilation,
        "elapsed_seconds": elapsed,
        "evaluation": evaluate(
            model, world, batch_maker, device, world.seed + 160_000
        ),
    }


def train_rademacher_control(
    world: MixedWorld,
    batch_maker: BatchMaker,
    model_seed: int,
    device: torch.device,
    multiplier: float,
    lr: float,
    loss_name: str,
) -> dict[str, object]:
    model = base.build_model(model_seed, device)
    initial_hash = base.state_sha256(model)
    rademacher_rebirth(model, world.seed + int(multiplier * 1000), multiplier)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    started = time.perf_counter()
    for step in range(1, MAX_STEPS + 1):
        hidden, target, _ = batch_maker(
            world,
            BATCH_SIZE,
            world.seed * 1_000_003 + step,
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(hidden)
        if loss_name == "bce":
            loss = F.binary_cross_entropy_with_logits(logits, (target + 1) / 2)
        elif loss_name == "hinge":
            loss = F.relu(1.0 - target * logits).mean()
        else:
            raise ValueError(loss_name)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "multiplier": multiplier,
        "lr": lr,
        "loss": loss_name,
        "initial_hash": initial_hash,
        "examples": MAX_STEPS * BATCH_SIZE,
        "elapsed_seconds": elapsed,
        "evaluation": evaluate(
            model, world, batch_maker, device, world.seed + 260_000
        ),
    }


def best_for_loss(runs: list[dict[str, object]], loss_name: str) -> dict[str, object]:
    matching = [run for run in runs if run["loss"] == loss_name]
    return max(
        matching,
        key=lambda run: (
            run["evaluation"]["accuracy"],
            run["evaluation"]["parity_accuracy"],
        ),
    )


def run(device: torch.device, quick: bool = False) -> dict[str, object]:
    views: tuple[tuple[str, BatchMaker], ...] = (
        ("mixed", make_mixed_batch),
        ("recurrence", make_recurrence_batch),
    )
    worlds: dict[str, object] = {}
    multipliers = MULTIPLIERS[:1] if quick else MULTIPLIERS
    learning_rates = LEARNING_RATES[:1] if quick else LEARNING_RATES
    for seed in base.WORLD_SEEDS:
        world = make_mixed_world(seed)
        model_seed = seed + 7_000
        view_results: dict[str, object] = {}
        for view_name, batch_maker in views:
            candidate_result = candidate(
                world, view_name, batch_maker, model_seed, device
            )
            control_runs: list[dict[str, object]] = []
            if not quick:
                control_runs = [
                    train_rademacher_control(
                        world,
                        batch_maker,
                        model_seed,
                        device,
                        multiplier,
                        lr,
                        loss_name,
                    )
                    for loss_name in LOSSES
                    for multiplier in multipliers
                    for lr in learning_rates
                ]
            hashes = [candidate_result["initial_hash"]] + [
                control["initial_hash"] for control in control_runs
            ]
            view_results[view_name] = {
                "candidate": candidate_result,
                "controls": control_runs,
                "best": {
                    loss_name: best_for_loss(control_runs, loss_name)
                    for loss_name in LOSSES
                }
                if control_runs
                else {},
                "identical_initial_hashes": len(set(hashes)) == 1,
            }
        matrix = mixing_tensor(world)
        worlds[str(seed)] = {
            "latent_support": world.latent_support,
            "observed_support": world.observed_support,
            "mixing_density": float(matrix.float().mean().item()),
            "mixing_is_permutation": bool(
                (matrix.sum(dim=0) == 1).all() and (matrix.sum(dim=1) == 1).all()
            ),
            "views": view_results,
        }

    gates: dict[str, bool] = {}
    if not quick:
        for seed, world_result in worlds.items():
            gates[f"{seed}_mixing_not_permutation"] = not world_result["mixing_is_permutation"]
            for view_name, view_result in world_result["views"].items():
                candidate_result = view_result["candidate"]
                evaluation = candidate_result["evaluation"]
                prefix = f"{seed}_{view_name}"
                gates[f"{prefix}_candidate_full_ge_99"] = evaluation["accuracy"] >= 0.99
                gates[f"{prefix}_candidate_parity_ge_99"] = evaluation["parity_accuracy"] >= 0.99
                gates[f"{prefix}_candidate_protected_ge_99"] = evaluation["protected_accuracy"] >= 0.99
                gates[f"{prefix}_support_exact"] = candidate_result["support_exact"]
                gates[f"{prefix}_same_initialization"] = view_result["identical_initial_hashes"]
                best = list(view_result["best"].values())
                gates[f"{prefix}_rademacher_parity_lt_80"] = all(
                    control["evaluation"]["parity_accuracy"] < 0.80
                    for control in best
                )
                gates[f"{prefix}_rademacher_protected_ge_95"] = any(
                    control["evaluation"]["protected_accuracy"] >= 0.95
                    for control in view_result["controls"]
                )

    result = {
        "schema": "heterogeneous-algebraic-compilation-t1b-v1",
        "status": "quick" if quick else "complete",
        "device": str(device),
        "torch_version": torch.__version__,
        "configuration": {
            "world_seeds": base.WORLD_SEEDS,
            "prefix_examples": base.PREFIX_EXAMPLES,
            "multipliers": MULTIPLIERS,
            "learning_rates": LEARNING_RATES,
            "losses": LOSSES,
            "max_steps": MAX_STEPS,
            "batch_size": BATCH_SIZE,
            "parameter_count": base.parameter_count(),
        },
        "ledger": {
            "candidate_dense_training_steps": 0,
            "candidate_prefix_examples": base.PREFIX_EXAMPLES,
            "control_examples_per_run": MAX_STEPS * BATCH_SIZE,
            "rademacher_rebirth_parameter_writes": base.parameter_count(),
            "datatypes_reported_separately": True,
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
