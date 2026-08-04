#!/usr/bin/env python3
"""Fatal T1 gate for finite-field discovery compiled into fixed SwiGLU weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = ROOT / "results/heterogeneous-algebraic-compilation-t1-preregistration.md"
TEST_SOURCE = ROOT / "tests/test_heterogeneous_algebraic_compilation_t1.py"
OUTPUT = ROOT / "results/heterogeneous-algebraic-compilation-t1.json"

INPUT_DIM = 32
SUPPORT_SIZE = 24
HIDDEN_DIM = 2 * INPUT_DIM + 4
DEPTH = math.ceil(math.log2(INPUT_DIM)) + 1
FFN_WIDTH = 2 * INPUT_DIM
PREFIX_EXAMPLES = 8 * INPUT_DIM
WORLD_SEEDS = (731, 947, 1213)
CHECKPOINTS = (4, 8, 100, 500, 2000)
ADAMW_LRS = (3e-4, 1e-3, 3e-3, 1e-2)
MUON_LRS = (1e-3, 3e-3, 1e-2, 3e-2)


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


class ResidualSwiGLUBlock(nn.Module):
    def __init__(self, hidden_dim: int, ffn_width: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_dim, ffn_width, bias=False)
        self.up = nn.Linear(hidden_dim, ffn_width, bias=False)
        self.down = nn.Linear(ffn_width, hidden_dim, bias=False)

    def forward(self, hidden: Tensor) -> Tensor:
        return hidden + self.down(F.silu(self.gate(hidden)) * self.up(hidden))


class ResidualSwiGLUParity(nn.Module):
    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        depth: int = DEPTH,
        ffn_width: int = FFN_WIDTH,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.blocks = nn.ModuleList(
            ResidualSwiGLUBlock(hidden_dim, ffn_width) for _ in range(depth)
        )
        self.head = nn.Linear(hidden_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        scale = 0.2 / math.sqrt(self.hidden_dim)
        for parameter in self.parameters():
            nn.init.normal_(parameter, mean=0.0, std=scale)

    def forward(self, hidden: Tensor) -> Tensor:
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(hidden).squeeze(-1)


@dataclass(frozen=True)
class World:
    seed: int
    support: tuple[int, ...]
    permutation: tuple[int, ...]


def make_world(seed: int) -> World:
    generator = random.Random(seed)
    latent_support = set(generator.sample(range(INPUT_DIM), SUPPORT_SIZE))
    permutation = list(range(INPUT_DIM))
    generator.shuffle(permutation)
    presented_support = tuple(
        sorted(permutation.index(index) for index in latent_support)
    )
    return World(seed, presented_support, tuple(permutation))


def make_batch(
    world: World,
    count: int,
    seed: int,
    device: torch.device,
    *,
    route_one_only: bool = False,
    random_labels: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    latent_bits = torch.randint(0, 2, (count, INPUT_DIM), generator=generator)
    presented_bits = latent_bits[:, list(world.permutation)]
    signs = 1.0 - 2.0 * presented_bits.float()
    if route_one_only:
        route = torch.ones(count, dtype=torch.long)
    else:
        route = torch.randint(0, 2, (count,), generator=generator)

    support = torch.tensor(world.support, dtype=torch.long)
    parity_bit = presented_bits[:, support].sum(dim=1).remainder(2)
    protected_bit = presented_bits[:, 0]
    target_bit = torch.where(route.bool(), parity_bit, protected_bit)
    if random_labels:
        target_bit = torch.randint(0, 2, (count,), generator=generator)

    hidden = torch.zeros(count, HIDDEN_DIM)
    hidden[:, :INPUT_DIM] = signs
    hidden[:, INPUT_DIM] = route.float()
    target_sign = 1.0 - 2.0 * target_bit.float()
    return hidden.to(device), target_sign.to(device), route.to(device)


def solve_gf2(matrix: Tensor, rhs: Tensor) -> tuple[Tensor | None, int, int]:
    """Solve an overdetermined GF(2) system; return None on rank loss/inconsistency."""
    augmented = torch.cat(
        [matrix.to(torch.uint8).cpu().clone(), rhs.to(torch.uint8).cpu()[:, None]],
        dim=1,
    )
    rows, columns_plus_one = augmented.shape
    columns = columns_plus_one - 1
    pivot_row = 0
    xor_rows = 0
    pivots: list[int] = []
    for column in range(columns):
        candidates = torch.nonzero(augmented[pivot_row:, column], as_tuple=False)
        if candidates.numel() == 0:
            continue
        selected = pivot_row + int(candidates[0, 0])
        if selected != pivot_row:
            temporary = augmented[pivot_row].clone()
            augmented[pivot_row] = augmented[selected]
            augmented[selected] = temporary
        for row in range(rows):
            if row != pivot_row and int(augmented[row, column]):
                augmented[row] ^= augmented[pivot_row]
                xor_rows += 1
        pivots.append(column)
        pivot_row += 1
        if pivot_row == rows:
            break

    inconsistent = bool(
        ((augmented[:, :columns].sum(dim=1) == 0) & (augmented[:, columns] == 1))
        .any()
        .item()
    )
    if inconsistent or len(pivots) != columns:
        return None, len(pivots), xor_rows * columns_plus_one
    solution = torch.zeros(columns, dtype=torch.uint8)
    for row, column in enumerate(pivots):
        solution[column] = augmented[row, columns]
    return solution, len(pivots), xor_rows * columns_plus_one


def signs_to_gf2(hidden: Tensor, target_sign: Tensor) -> tuple[Tensor, Tensor]:
    matrix = (hidden[:, :INPUT_DIM].cpu() < 0).to(torch.uint8)
    rhs = (target_sign.cpu() < 0).to(torch.uint8)
    return matrix, rhs


def _write_product(
    block: ResidualSwiGLUBlock,
    channel: int,
    left: int,
    right: int,
    target: int,
) -> None:
    block.gate.weight[channel, left] = 1.0
    block.gate.weight[channel + 1, left] = -1.0
    block.up.weight[channel, right] = 1.0
    block.up.weight[channel + 1, right] = 1.0
    block.down.weight[target, channel] = 1.0
    block.down.weight[target, channel + 1] = -1.0


@torch.no_grad()
def compile_support(model: ResidualSwiGLUParity, support: Iterable[int]) -> dict[str, int]:
    support = list(support)
    if not support:
        raise ValueError("empty support requires a constant feature not present in this gate")
    for parameter in model.parameters():
        parameter.zero_()

    nodes = support
    next_workspace = INPUT_DIM + 1
    nonzero_writes = 0
    tree_layers = math.ceil(math.log2(len(nodes))) if len(nodes) > 1 else 0
    for layer_index in range(tree_layers):
        block = model.blocks[layer_index]
        next_nodes: list[int] = []
        channel = 0
        pair_index = 0
        while pair_index + 1 < len(nodes):
            target = next_workspace
            next_workspace += 1
            _write_product(block, channel, nodes[pair_index], nodes[pair_index + 1], target)
            nonzero_writes += 6
            next_nodes.append(target)
            channel += 2
            pair_index += 2
        if pair_index < len(nodes):
            next_nodes.append(nodes[pair_index])
        nodes = next_nodes

    parity_coordinate = nodes[0]
    route_block = model.blocks[tree_layers]
    routed_parity = next_workspace
    next_workspace += 1
    routed_protected = next_workspace
    _write_product(route_block, 0, INPUT_DIM, parity_coordinate, routed_parity)
    _write_product(route_block, 2, INPUT_DIM, 0, routed_protected)
    nonzero_writes += 12

    confidence = 8.0
    model.head.weight[0, 0] = confidence
    model.head.weight[0, routed_parity] = confidence
    model.head.weight[0, routed_protected] = -confidence
    nonzero_writes += 3
    return {
        "support_size": len(support),
        "tree_layers": tree_layers,
        "parity_coordinate": parity_coordinate,
        "routed_parity_coordinate": routed_parity,
        "routed_protected_coordinate": routed_protected,
        "nonzero_parameter_writes": nonzero_writes,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    world: World,
    device: torch.device,
    seed: int,
    count: int = 16_384,
    *,
    random_labels: bool = False,
) -> dict[str, float]:
    hidden, target, route = make_batch(
        world, count, seed, device, random_labels=random_labels
    )
    logits = model(hidden)
    correct = (logits >= 0) == (target >= 0)
    route_one = route == 1
    route_zero = ~route_one
    return {
        "accuracy": float(correct.float().mean().item()),
        "parity_accuracy": float(correct[route_one].float().mean().item()),
        "protected_accuracy": float(correct[route_zero].float().mean().item()),
        "bce": float(F.binary_cross_entropy_with_logits(logits, (target + 1) / 2).item()),
    }


def build_model(seed: int, device: torch.device) -> ResidualSwiGLUParity:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    return ResidualSwiGLUParity().to(device)


def train_control(
    optimizer_name: str,
    lr: float,
    world: World,
    model_seed: int,
    device: torch.device,
    max_steps: int,
) -> dict[str, object]:
    model = build_model(model_seed, device)
    initial_hash = state_sha256(model)
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    elif optimizer_name == "muon":
        optimizer = torch.optim.Muon(
            model.parameters(),
            lr=lr,
            weight_decay=0.0,
            adjust_lr_fn="match_rms_adamw",
        )
    else:
        raise ValueError(optimizer_name)

    checkpoints: dict[str, dict[str, float]] = {}
    started = time.perf_counter()
    for step in range(1, max_steps + 1):
        hidden, target, _ = make_batch(
            world, 64, world.seed * 1_000_003 + step, device
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(hidden)
        loss = F.binary_cross_entropy_with_logits(logits, (target + 1) / 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in CHECKPOINTS:
            checkpoints[str(step)] = evaluate(
                model, world, device, seed=world.seed + 90_000 + step
            )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "optimizer": optimizer_name,
        "lr": lr,
        "initial_hash": initial_hash,
        "checkpoints": checkpoints,
        "elapsed_seconds": elapsed,
        "examples": max_steps * 64,
    }


def best_control(runs: list[dict[str, object]]) -> dict[str, object]:
    return max(
        runs,
        key=lambda run: (
            run["checkpoints"][str(CHECKPOINTS[-1])]["accuracy"],
            run["checkpoints"][str(CHECKPOINTS[-1])]["parity_accuracy"],
        ),
    )


def candidate_run(world: World, model_seed: int, device: torch.device) -> dict[str, object]:
    model = build_model(model_seed, device)
    initial_hash = state_sha256(model)
    prefix_hidden, prefix_target, _ = make_batch(
        world,
        PREFIX_EXAMPLES,
        world.seed + 40_000,
        device,
        route_one_only=True,
    )
    matrix, rhs = signs_to_gf2(prefix_hidden, prefix_target)
    started = time.perf_counter()
    solution, rank, xor_bit_ops = solve_gf2(matrix, rhs)
    if solution is None:
        raise RuntimeError(f"consistent world failed to solve at rank {rank}")
    recovered = tuple(torch.nonzero(solution, as_tuple=False).flatten().tolist())
    compilation = compile_support(model, recovered)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    evaluation = evaluate(model, world, device, seed=world.seed + 80_000)
    return {
        "initial_hash": initial_hash,
        "rank": rank,
        "xor_bit_ops": xor_bit_ops,
        "recovered_support": recovered,
        "support_exact": recovered == world.support,
        "compilation": compilation,
        "elapsed_seconds": elapsed,
        "evaluation": evaluation,
    }


def random_label_run(world: World, model_seed: int, device: torch.device) -> dict[str, object]:
    model = build_model(model_seed, device)
    prefix_hidden, prefix_target, _ = make_batch(
        world,
        PREFIX_EXAMPLES,
        world.seed + 50_000,
        device,
        route_one_only=True,
        random_labels=True,
    )
    matrix, rhs = signs_to_gf2(prefix_hidden, prefix_target)
    solution, rank, xor_bit_ops = solve_gf2(matrix, rhs)
    abstained = solution is None
    evaluation = evaluate(
        model,
        world,
        device,
        seed=world.seed + 100_000,
        random_labels=True,
    )
    return {
        "rank": rank,
        "xor_bit_ops": xor_bit_ops,
        "abstained": abstained,
        "evaluation": evaluation,
    }


def parameter_count() -> int:
    return sum(parameter.numel() for parameter in ResidualSwiGLUParity().parameters())


def run(device: torch.device, quick: bool = False) -> dict[str, object]:
    max_steps = 100 if quick else CHECKPOINTS[-1]
    adam_lrs = ADAMW_LRS[:2] if quick else ADAMW_LRS
    muon_lrs = MUON_LRS[:2] if quick else MUON_LRS
    worlds: dict[str, object] = {}
    for seed in WORLD_SEEDS:
        world = make_world(seed)
        model_seed = seed + 7_000
        candidate = candidate_run(world, model_seed, device)
        random_label = random_label_run(world, model_seed, device)
        controls: dict[str, object] = {}
        if not quick:
            for optimizer_name, lrs in (("adamw", adam_lrs), ("muon", muon_lrs)):
                runs = [
                    train_control(
                        optimizer_name, lr, world, model_seed, device, max_steps
                    )
                    for lr in lrs
                ]
                controls[optimizer_name] = {"runs": runs, "best": best_control(runs)}
        hashes = [candidate["initial_hash"]]
        for control in controls.values():
            hashes.extend(run_result["initial_hash"] for run_result in control["runs"])
        worlds[str(seed)] = {
            "world": asdict(world),
            "candidate": candidate,
            "random_label": random_label,
            "controls": controls,
            "identical_initial_hashes": len(set(hashes)) == 1,
        }

    gates: dict[str, bool] = {}
    if not quick:
        for seed, result in worlds.items():
            candidate = result["candidate"]
            gates[f"{seed}_candidate_full_ge_99"] = candidate["evaluation"]["accuracy"] >= 0.99
            gates[f"{seed}_candidate_parity_ge_99"] = candidate["evaluation"]["parity_accuracy"] >= 0.99
            gates[f"{seed}_candidate_protected_ge_99"] = candidate["evaluation"]["protected_accuracy"] >= 0.99
            gates[f"{seed}_support_exact"] = candidate["support_exact"]
            gates[f"{seed}_same_initialization"] = result["identical_initial_hashes"]
            gates[f"{seed}_random_abstained"] = result["random_label"]["abstained"]
            random_accuracy = result["random_label"]["evaluation"]["accuracy"]
            gates[f"{seed}_random_chance"] = 0.45 <= random_accuracy <= 0.55
            best_runs = [result["controls"][name]["best"] for name in ("adamw", "muon")]
            gates[f"{seed}_controls_step8_parity_lt_80"] = all(
                run_result["checkpoints"]["8"]["parity_accuracy"] < 0.80
                for run_result in best_runs
            )
            gates[f"{seed}_controls_step2000_parity_lt_80"] = all(
                run_result["checkpoints"]["2000"]["parity_accuracy"] < 0.80
                for run_result in best_runs
            )
            gates[f"{seed}_some_control_protected_ge_95"] = any(
                run_result["checkpoints"]["2000"]["protected_accuracy"] >= 0.95
                for run_result in best_runs
            )

    result = {
        "schema": "heterogeneous-algebraic-compilation-t1-v1",
        "status": "quick" if quick else "complete",
        "device": str(device),
        "torch_version": torch.__version__,
        "configuration": {
            "input_dim": INPUT_DIM,
            "support_size": SUPPORT_SIZE,
            "hidden_dim": HIDDEN_DIM,
            "depth": DEPTH,
            "ffn_width": FFN_WIDTH,
            "prefix_examples": PREFIX_EXAMPLES,
            "world_seeds": WORLD_SEEDS,
            "checkpoints": CHECKPOINTS,
            "adamw_lrs": ADAMW_LRS,
            "muon_lrs": MUON_LRS,
            "parameter_count": parameter_count(),
        },
        "ledger": {
            "candidate_dense_training_steps": 0,
            "candidate_prefix_examples": PREFIX_EXAMPLES,
            "control_batch_size": 64,
            "control_max_examples": max_steps * 64,
            "candidate_parameter_writes_upper_bound": parameter_count(),
            "datatypes_reported_separately": True,
        },
        "worlds": worlds,
        "gates": gates,
        "all_gates_pass": bool(gates) and all(gates.values()),
        "integrity": {
            "source_sha256": sha256_file(Path(__file__)),
            "test_sha256": sha256_file(TEST_SOURCE),
            "preregistration_sha256": sha256_file(PREREGISTRATION),
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

