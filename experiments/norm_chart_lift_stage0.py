#!/usr/bin/env python3
"""Stage-0 screen for a gauge-funded nonlinear chart before dense projections."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "norm-chart-lift-stage0.json"
PREREGISTRATION = ROOT / "results" / "norm-chart-lift-stage0-preregistration.md"

FORMAL_SEEDS = (1424, 4436, 7493, 1911, 5377)
DEVELOPMENT_SEEDS = (9731,)
ARMS = ("dense", "carrier_null", "self_chart", "partner_chart")
TASKS = ("ordinary", "self_interaction", "partner_interaction")


@dataclass(frozen=True)
class Config:
    width: int = 64
    hidden: int = 64
    classes: int = 16
    bits: int = 4
    blocks: int = 2
    train_steps: int = 1200
    batch_size: int = 512
    eval_examples: int = 16384
    learning_rate: float = 2e-3
    weight_decay: float = 1e-3
    alpha_max: float = 0.75
    alpha_scale: float = 0.5
    epsilon: float = 2.0
    interaction_strength: float = 1.5


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def rms_unit(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6)


@dataclass(frozen=True)
class Teacher:
    linear: torch.Tensor
    self_terms: torch.Tensor
    partner_terms: torch.Tensor


def make_teacher(config: Config, seed: int, device: torch.device) -> Teacher:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed * 1009 + 71)

    def signs(shape: tuple[int, ...]) -> torch.Tensor:
        return (
            torch.randint(0, 2, shape, generator=generator, device=device).float()
            * 2.0
            - 1.0
        )

    return Teacher(
        linear=signs((config.width, config.bits)),
        self_terms=signs((config.width // 2, config.bits)),
        partner_terms=signs((config.width // 2, config.bits)),
    )


def make_batch(
    config: Config,
    teacher: Teacher,
    task: str,
    size: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn((size, config.width), generator=generator, device=device)
    u = rms_unit(x)
    score = (u @ teacher.linear) / math.sqrt(config.width)
    if task == "self_interaction":
        terms = u[:, 0::2].square() - u[:, 1::2].square()
        score = score + config.interaction_strength * (
            terms @ teacher.self_terms
        ) / math.sqrt(config.width // 2)
    elif task == "partner_interaction":
        terms = u[:, 0::2] * u[:, 1::2]
        score = score + config.interaction_strength * (
            terms @ teacher.partner_terms
        ) / math.sqrt(config.width // 2)
    elif task != "ordinary":
        raise ValueError(task)
    labels = torch.zeros(size, dtype=torch.long, device=device)
    for bit in range(config.bits):
        labels |= score[:, bit].gt(0).long() << bit
    return x, labels


class ChartNorm(nn.Module):
    def __init__(self, config: Config, arm: str, block_index: int):
        super().__init__()
        self.config = config
        self.arm = arm
        self.block_index = block_index
        self.weight = nn.Parameter(torch.ones(config.width))

    def alpha(self) -> torch.Tensor:
        return self.config.alpha_max * torch.tanh(
            (self.weight[0] - 1.0) / self.config.alpha_scale
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        ablate: bool = False,
        shift_partner: bool = False,
    ) -> torch.Tensor:
        unit = rms_unit(x)
        if self.arm == "dense":
            return unit * self.weight

        # The first ordinary RMSNorm scale is functionally redundant with all
        # following projection columns. It is interpreted as the chart scalar;
        # its effective normalization scale remains one.
        scale = torch.cat((torch.ones_like(self.weight[:1]), self.weight[1:]))
        unit = unit * scale
        if self.arm == "carrier_null" or ablate:
            return unit

        alpha = self.alpha()
        even = unit[:, 0::2]
        odd = unit[:, 1::2]
        if self.arm == "self_chart":
            if self.block_index % 2 == 0:
                gated = odd * (
                    1.0
                    + alpha
                    * torch.clamp(odd / self.config.epsilon, min=-1.0, max=1.0)
                )
                return torch.stack((even, gated), dim=-1).flatten(-2)
            gated = even * (
                1.0
                + alpha
                * torch.clamp(even / self.config.epsilon, min=-1.0, max=1.0)
            )
            return torch.stack((gated, odd), dim=-1).flatten(-2)
        if self.arm != "partner_chart":
            raise ValueError(self.arm)

        if self.block_index % 2 == 0:
            conditioner = even
            if shift_partner:
                conditioner = torch.roll(conditioner, shifts=1, dims=-1)
            gated = odd * (
                1.0
                + alpha
                * torch.clamp(
                    conditioner / self.config.epsilon, min=-1.0, max=1.0
                )
            )
            return torch.stack((even, gated), dim=-1).flatten(-2)
        conditioner = odd
        if shift_partner:
            conditioner = torch.roll(conditioner, shifts=1, dims=-1)
        gated = even * (
            1.0
            + alpha
            * torch.clamp(conditioner / self.config.epsilon, min=-1.0, max=1.0)
        )
        return torch.stack((gated, odd), dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, config: Config, arm: str, block_index: int):
        super().__init__()
        self.norm = ChartNorm(config, arm, block_index)
        self.gate = nn.Linear(config.width, config.hidden, bias=False)
        self.up = nn.Linear(config.width, config.hidden, bias=False)
        self.down = nn.Linear(config.hidden, config.width, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        *,
        ablate: bool = False,
        shift_partner: bool = False,
    ) -> torch.Tensor:
        z = self.norm(x, ablate=ablate, shift_partner=shift_partner)
        return x + self.down(F.silu(self.gate(z)) * self.up(z))


class Classifier(nn.Module):
    def __init__(self, config: Config, arm: str):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm = arm
        self.blocks = nn.ModuleList(
            Block(config, arm, index) for index in range(config.blocks)
        )
        self.final_norm = nn.RMSNorm(config.width)
        self.readout = nn.Linear(config.width, config.classes, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        *,
        ablate: bool = False,
        shift_partner: bool = False,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, ablate=ablate, shift_partner=shift_partner)
        return self.readout(self.final_norm(x))

    def alphas(self) -> list[float]:
        return [float(block.norm.alpha().detach().cpu()) for block in self.blocks]


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def logical_ledger(config: Config) -> dict[str, int | float]:
    input_projection_macs = config.blocks * 2 * config.width * config.hidden
    output_projection_macs = config.blocks * config.width * config.hidden
    chart_pairs = config.blocks * config.width // 2
    return {
        "dense_matrix_macs_per_example": input_projection_macs
        + output_projection_macs
        + config.width * config.classes,
        "chart_pairs_per_example": chart_pairs,
        "chart_scalar_ops_upper_bound": 5 * chart_pairs,
        "chart_to_dense_mac_ratio_upper_bound": (5 * chart_pairs)
        / (
            input_projection_macs
            + output_projection_macs
            + config.width * config.classes
        ),
        "persistent_chart_index_values": 0,
        "additional_learned_values": 0,
    }


def train_arm(
    config: Config,
    arm: str,
    task: str,
    seed: int,
    base_state: dict[str, torch.Tensor],
    teacher: Teacher,
    device: torch.device,
) -> tuple[Classifier, str, dict[str, float]]:
    model = Classifier(config, arm).to(device)
    model.load_state_dict(base_state)
    model.train()
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        (no_decay if "norm" in name else decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed * 100_003 + TASKS.index(task) * 997 + 19)
    trace = hashlib.sha256()
    maximum_loss = 0.0
    maximum_gradient = 0.0
    for step in range(config.train_steps):
        x, labels = make_batch(
            config, teacher, task, config.batch_size, generator, device
        )
        if step in (0, config.train_steps - 1):
            trace.update(x.detach().cpu().contiguous().numpy().tobytes())
            trace.update(labels.detach().cpu().contiguous().numpy().tobytes())
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), labels)
        if not torch.isfinite(loss):
            raise FloatingPointError((arm, task, seed, step))
        loss.backward()
        grad_sq = 0.0
        for parameter in model.parameters():
            if parameter.grad is not None:
                grad_sq += float(parameter.grad.detach().float().square().sum().item())
        maximum_gradient = max(maximum_gradient, math.sqrt(grad_sq))
        maximum_loss = max(maximum_loss, float(loss.detach()))
        optimizer.step()
    return model, trace.hexdigest(), {
        "maximum_loss": maximum_loss,
        "maximum_gradient_norm": maximum_gradient,
    }


@torch.inference_mode()
def evaluate(
    config: Config,
    model: Classifier,
    teacher: Teacher,
    task: str,
    seed: int,
    device: torch.device,
    *,
    ablate: bool = False,
    shift_partner: bool = False,
) -> dict[str, Any]:
    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed * 1_000_003 + TASKS.index(task) * 7919 + 401)
    correct = 0
    loss_sum = 0.0
    counts = torch.zeros(config.classes, dtype=torch.long, device=device)
    processed = 0
    while processed < config.eval_examples:
        size = min(config.batch_size, config.eval_examples - processed)
        x, labels = make_batch(config, teacher, task, size, generator, device)
        logits = model(x, ablate=ablate, shift_partner=shift_partner)
        correct += int(logits.argmax(dim=-1).eq(labels).sum().item())
        loss_sum += float(F.cross_entropy(logits, labels, reduction="sum").item())
        counts += torch.bincount(labels, minlength=config.classes)
        processed += size
    return {
        "accuracy": correct / processed,
        "loss": loss_sum / processed,
        "class_prior_max": float(counts.max().item() / processed),
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--eval-examples", type=int)
    args = parser.parse_args()
    device = torch.device(args.device)
    base_config = Config()
    config = Config(
        **{
            **asdict(base_config),
            **({"train_steps": args.steps} if args.steps is not None else {}),
            **(
                {"eval_examples": args.eval_examples}
                if args.eval_examples is not None
                else {}
            ),
        }
    )
    if args.formal and (args.steps is not None or args.eval_examples is not None):
        raise ValueError("formal protocol forbids overrides")
    if args.formal and not PREREGISTRATION.exists():
        raise FileNotFoundError(PREREGISTRATION)

    seeds = FORMAL_SEEDS if args.formal else DEVELOPMENT_SEEDS
    torch.use_deterministic_algorithms(True)
    runs: list[dict[str, Any]] = []
    initialization_hashes: dict[str, dict[str, str]] = {}
    training_traces: dict[str, dict[str, dict[str, str]]] = {}
    parameter_counts: dict[str, int] = {}
    stability: dict[str, dict[str, dict[str, dict[str, float]]]] = {}

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        template = Classifier(config, "dense").to(device)
        base_state = copy.deepcopy(template.state_dict())
        base_hash = tensor_hash(base_state)
        teacher = make_teacher(config, seed, device)
        initialization_hashes[str(seed)] = {}
        training_traces[str(seed)] = {}
        stability[str(seed)] = {}
        for task in TASKS:
            training_traces[str(seed)][task] = {}
            stability[str(seed)][task] = {}
            for arm in ARMS:
                probe = Classifier(config, arm).to(device)
                probe.load_state_dict(base_state)
                initialization_hashes[str(seed)][arm] = tensor_hash(probe.state_dict())
                parameter_counts[arm] = parameter_count(probe)
                if initialization_hashes[str(seed)][arm] != base_hash:
                    raise AssertionError("state initialization mismatch")
                diagnostic_generator = torch.Generator(device=device)
                diagnostic_generator.manual_seed(seed * 13 + 5)
                diagnostic = torch.randn(
                    (64, config.width), generator=diagnostic_generator, device=device
                )
                dense_probe = Classifier(config, "dense").to(device)
                dense_probe.load_state_dict(base_state)
                initial_error = float(
                    (probe(diagnostic) - dense_probe(diagnostic)).abs().max().item()
                )
                if initial_error != 0.0:
                    raise AssertionError((arm, initial_error))
                model, trace, health = train_arm(
                    config, arm, task, seed, base_state, teacher, device
                )
                training_traces[str(seed)][task][arm] = trace
                stability[str(seed)][task][arm] = health
                metrics = evaluate(config, model, teacher, task, seed, device)
                runs.append(
                    {
                        "seed": seed,
                        "task": task,
                        "arm": arm,
                        "initial_max_error_vs_dense": initial_error,
                        "alphas": model.alphas(),
                        **metrics,
                    }
                )
                if arm == "partner_chart" and task == "partner_interaction":
                    for ablation, kwargs in (
                        ("alpha_zero", {"ablate": True}),
                        ("shift_partner", {"shift_partner": True}),
                    ):
                        ablated = evaluate(
                            config, model, teacher, task, seed, device, **kwargs
                        )
                        runs.append(
                            {
                                "seed": seed,
                                "task": task,
                                "arm": f"partner_chart_{ablation}",
                                "base_arm": "partner_chart",
                                "alphas": model.alphas(),
                                **ablated,
                            }
                        )

    def rows(arm: str, task: str) -> list[dict[str, Any]]:
        return [run for run in runs if run["arm"] == arm and run["task"] == task]

    summary: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in ARMS:
        summary[arm] = {}
        for task in TASKS:
            selected = rows(arm, task)
            summary[arm][task] = {
                "mean_accuracy": mean([row["accuracy"] for row in selected]),
                "accuracies": [row["accuracy"] for row in selected],
                "mean_loss": mean([row["loss"] for row in selected]),
            }

    gates: dict[str, bool] | None = None
    passed: bool | None = None
    if args.formal:
        partner = rows("partner_chart", "partner_interaction")
        dense_pair = rows("dense", "partner_interaction")
        carrier_pair = rows("carrier_null", "partner_interaction")
        self_pair = rows("self_chart", "partner_interaction")
        partner_ordinary = rows("partner_chart", "ordinary")
        dense_ordinary = rows("dense", "ordinary")
        carrier_ordinary = rows("carrier_null", "ordinary")
        self_self = rows("self_chart", "self_interaction")
        partner_self = rows("partner_chart", "self_interaction")
        alpha_zero = rows("partner_chart_alpha_zero", "partner_interaction")
        shifted = rows("partner_chart_shift_partner", "partner_interaction")

        best_noncandidate_mean = max(
            mean([row["accuracy"] for row in arm_rows])
            for arm_rows in (dense_pair, carrier_pair, self_pair)
        )
        gains_each = []
        for index in range(len(seeds)):
            best = max(
                dense_pair[index]["accuracy"],
                carrier_pair[index]["accuracy"],
                self_pair[index]["accuracy"],
            )
            gains_each.append(partner[index]["accuracy"] - best)
        ordinary_margins = [
            partner_ordinary[index]["accuracy"] - dense_ordinary[index]["accuracy"]
            for index in range(len(seeds))
        ]
        carrier_margins = [
            carrier_ordinary[index]["accuracy"] - dense_ordinary[index]["accuracy"]
            for index in range(len(seeds))
        ]
        self_specificity = [
            self_self[index]["accuracy"] - partner_self[index]["accuracy"]
            for index in range(len(seeds))
        ]
        causal_drops = [
            partner[index]["accuracy"] - alpha_zero[index]["accuracy"]
            for index in range(len(seeds))
        ]
        pairing_drops = [
            partner[index]["accuracy"] - shifted[index]["accuracy"]
            for index in range(len(seeds))
        ]
        all_traces_paired = all(
            len(set(training_traces[str(seed)][task].values())) == 1
            for seed in seeds
            for task in TASKS
        )
        all_finite = all(
            math.isfinite(run["accuracy"])
            and math.isfinite(run["loss"])
            and all(math.isfinite(value) for value in run["alphas"])
            for run in runs
        )
        maximum_prior = max(run["class_prior_max"] for run in runs)
        gates = {
            "integrity": len(set(parameter_counts.values())) == 1
            and all_traces_paired
            and all_finite
            and maximum_prior <= 0.08,
            "pair_gain": mean([row["accuracy"] for row in partner])
            >= best_noncandidate_mean + 0.05
            and min(gains_each) >= 0.02,
            "ordinary_protection": mean(ordinary_margins) >= -0.01
            and min(ordinary_margins) >= -0.02,
            "carrier_valid": mean(carrier_margins) >= -0.01
            and min(carrier_margins) >= -0.02,
            "specificity": mean(self_specificity) >= 0.05
            and min(self_specificity) >= 0.02,
            "causal_use": min(causal_drops) >= 0.05,
            "pairing_use": min(pairing_drops) >= 0.05,
        }
        passed = all(gates.values())
        summary["formal_diagnostics"] = {
            "best_noncandidate_pair_mean": best_noncandidate_mean,
            "pair_gains_each": gains_each,
            "ordinary_margins": ordinary_margins,
            "carrier_margins": carrier_margins,
            "self_specificity_each": self_specificity,
            "causal_drops_each": causal_drops,
            "pairing_drops_each": pairing_drops,
            "maximum_class_prior": maximum_prior,
        }

    result = {
        "schema": "norm-chart-lift-stage0-v1",
        "formal": args.formal,
        "pass": passed,
        "gates": gates,
        "config": asdict(config),
        "seeds": list(seeds),
        "arms": list(ARMS),
        "tasks": list(TASKS),
        "ledger": logical_ledger(config),
        "parameter_counts": parameter_counts,
        "initialization_hashes": initialization_hashes,
        "training_traces": training_traces,
        "stability": stability,
        "summary": summary,
        "runs": runs,
        "source_sha256": sha256(Path(__file__)),
        "preregistration_sha256": sha256(PREREGISTRATION)
        if args.formal
        else None,
        "device": str(device),
        "torch_version": torch.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pass": passed, "gates": gates, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
