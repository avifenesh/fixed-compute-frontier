#!/usr/bin/env python3
"""CPU-only Stage-0 gate for replacing dense projections with a feature DAG."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = ROOT / "results" / "feature-dag-matmul-stage0-preregistration.md"
TARGETS = ("prefix", "hierarchical", "mixed", "haar")
VARIANTS = (
    "dense_linear",
    "dense_ffn",
    "low_rank",
    "butterfly",
    "lookup",
    "feature_dag",
)
OFFSETS = (1, 2, 4, 8, 16)
WORLD_SEEDS = (31013, 31019, 31033, 31039, 31051)


@dataclass(frozen=True)
class Config:
    width: int = 32
    train_examples: int = 4096
    validation_examples: int = 1024
    test_examples: int = 2048
    batch_size: int = 256
    steps: int = 1200
    validation_interval: int = 40
    learning_rate: float = 0.01
    mixed_hierarchy_weight: float = 0.75


CONFIG = Config()


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _predecessor(values: np.ndarray, offset: int) -> np.ndarray:
    result = np.zeros_like(values)
    result[:, offset:] = values[:, :-offset]
    return result


def dag_numpy(
    inputs: np.ndarray,
    coefficients: dict[str, np.ndarray],
) -> np.ndarray:
    values = np.asarray(inputs, dtype=np.float64)
    for stage, offset in enumerate(OFFSETS):
        previous = _predecessor(values, offset)
        inner = (
            coefficients["d"][stage] * values
            + coefficients["e"][stage] * previous
            + coefficients["f"][stage]
        )
        values = (
            coefficients["a"][stage] * values
            + coefficients["b"][stage] * previous
            + coefficients["c"][stage] * np.tanh(inner)
        )
    return values * coefficients["output_scale"] + coefficients["output_bias"]


def prefix_coefficients(width: int = CONFIG.width) -> dict[str, np.ndarray]:
    shape = (len(OFFSETS), width)
    return {
        "a": np.ones(shape),
        "b": np.ones(shape),
        "c": np.zeros(shape),
        "d": np.zeros(shape),
        "e": np.zeros(shape),
        "f": np.zeros(shape),
        "output_scale": 1.0 / np.sqrt(np.arange(1, width + 1)),
        "output_bias": np.zeros(width),
    }


def prefix_algebra_check(width: int = CONFIG.width) -> dict[str, Any]:
    rng = np.random.default_rng(9191)
    inputs = rng.normal(size=(17, width))
    expected = np.cumsum(inputs, axis=1) / np.sqrt(np.arange(1, width + 1))
    actual = dag_numpy(inputs, prefix_coefficients(width))
    return {
        "width": width,
        "matrix_rank": width,
        "dense_nonzero_terms": width * (width + 1) // 2,
        "serial_prefix_additions": width - 1,
        "parallel_scan_additions": width * int(math.log2(width))
        - (width - 1),
        "max_abs_error": float(np.max(np.abs(expected - actual))),
    }


def _hierarchy_coefficients(world_seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(_stable_seed("hierarchy", world_seed))
    shape = (len(OFFSETS), CONFIG.width)
    return {
        "a": rng.normal(0.72, 0.08, size=shape),
        "b": rng.normal(0.22, 0.10, size=shape),
        "c": rng.normal(0.00, 0.42, size=shape),
        "d": rng.normal(0.00, 0.55, size=shape),
        "e": rng.normal(0.00, 0.55, size=shape),
        "f": rng.normal(0.00, 0.08, size=shape),
        "output_scale": rng.uniform(0.75, 1.25, size=CONFIG.width),
        "output_bias": rng.normal(0.0, 0.05, size=CONFIG.width),
    }


def _haar_matrix(world_seed: int) -> np.ndarray:
    rng = np.random.default_rng(_stable_seed("haar", world_seed))
    raw = rng.normal(size=(CONFIG.width, CONFIG.width))
    orthogonal, triangular = np.linalg.qr(raw)
    signs = np.sign(np.diag(triangular))
    signs[signs == 0.0] = 1.0
    return orthogonal * signs


def _standardize(
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    if np.any(scale < 1e-6):
        raise RuntimeError("teacher produced a degenerate target coordinate")
    return (
        (train - mean) / scale,
        (validation - mean) / scale,
        (test - mean) / scale,
        mean,
        scale,
    )


def generate_target_data(world_seed: int, target: str) -> dict[str, np.ndarray]:
    if target not in TARGETS:
        raise ValueError(f"unknown target: {target}")
    sizes = (
        CONFIG.train_examples,
        CONFIG.validation_examples,
        CONFIG.test_examples,
    )
    inputs = []
    for split, size in zip(("train", "validation", "test"), sizes, strict=True):
        rng = np.random.default_rng(_stable_seed("inputs", world_seed, split))
        inputs.append(rng.normal(size=(size, CONFIG.width)))

    hierarchy = _hierarchy_coefficients(world_seed)
    haar = _haar_matrix(world_seed)
    raw_hierarchy = [dag_numpy(values, hierarchy) for values in inputs]
    raw_haar = [values @ haar for values in inputs]
    raw_prefix = [
        np.cumsum(values, axis=1) / np.sqrt(np.arange(1, CONFIG.width + 1))
        for values in inputs
    ]
    standardized_hierarchy = _standardize(*raw_hierarchy)[:3]
    standardized_haar = _standardize(*raw_haar)[:3]

    if target == "prefix":
        outputs = _standardize(*raw_prefix)[:3]
    elif target == "hierarchical":
        outputs = standardized_hierarchy
    elif target == "haar":
        outputs = standardized_haar
    else:
        weight = CONFIG.mixed_hierarchy_weight
        raw_mixed = [
            weight * hierarchical + (1.0 - weight) * linear
            for hierarchical, linear in zip(
                standardized_hierarchy, standardized_haar, strict=True
            )
        ]
        outputs = _standardize(*raw_mixed)[:3]

    return {
        "train_x": inputs[0].astype(np.float32),
        "validation_x": inputs[1].astype(np.float32),
        "test_x": inputs[2].astype(np.float32),
        "train_y": outputs[0].astype(np.float32),
        "validation_y": outputs[1].astype(np.float32),
        "test_y": outputs[2].astype(np.float32),
    }


class DenseLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(CONFIG.width, CONFIG.width))
        nn.init.orthogonal_(self.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs @ self.weight.T


class DenseFFN(nn.Module):
    def __init__(self, nonlinear: bool) -> None:
        super().__init__()
        hidden = CONFIG.width // 2
        self.up = nn.Parameter(torch.empty(hidden, CONFIG.width))
        self.down = nn.Parameter(torch.empty(CONFIG.width, hidden))
        nn.init.orthogonal_(self.up)
        nn.init.orthogonal_(self.down)
        self.nonlinear = nonlinear

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs @ self.up.T
        if self.nonlinear:
            hidden = torch.tanh(hidden)
        return hidden @ self.down.T


def _butterfly_pairs(offset: int) -> tuple[list[int], list[int]]:
    left = [index for index in range(CONFIG.width) if index & offset == 0]
    return left, [index | offset for index in left]


class Butterfly(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        stages = 16
        schedule = (1, 2, 4, 8, 16, 16, 8, 4, 2, 1, 1, 2, 4, 8, 16, 16)
        left, right = zip(*(_butterfly_pairs(offset) for offset in schedule), strict=True)
        self.register_buffer("left", torch.tensor(left, dtype=torch.long))
        self.register_buffer("right", torch.tensor(right, dtype=torch.long))
        weight = torch.zeros(stages, CONFIG.width // 2, 2, 2)
        weight[:, :, 0, 0] = 1.0
        weight[:, :, 1, 1] = 1.0
        weight += 0.01 * torch.randn_like(weight)
        self.weight = nn.Parameter(weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = inputs
        for stage in range(self.weight.shape[0]):
            first = values.index_select(1, self.left[stage])
            second = values.index_select(1, self.right[stage])
            weight = self.weight[stage]
            first_out = first * weight[:, 0, 0] + second * weight[:, 0, 1]
            second_out = first * weight[:, 1, 0] + second * weight[:, 1, 1]
            output = torch.empty_like(values)
            output[:, self.left[stage]] = first_out
            output[:, self.right[stage]] = second_out
            values = output
        return values


class Lookup(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        indices = (
            (0, 7, 19),
            (3, 13, 29),
            (5, 11, 23),
            (2, 17, 31),
        )
        self.register_buffer("indices", torch.tensor(indices, dtype=torch.long))
        self.register_buffer("bit_weights", torch.tensor((1, 2, 4), dtype=torch.long))
        self.tables = nn.Parameter(torch.empty(4, 8, CONFIG.width))
        nn.init.normal_(self.tables, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(inputs)
        for table in range(self.tables.shape[0]):
            bits = inputs.index_select(1, self.indices[table]) > 0.0
            address = (bits.to(torch.long) * self.bit_weights).sum(dim=1)
            output = output + self.tables[table].index_select(0, address)
        return output


class FeatureDAG(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        shape = (len(OFFSETS), CONFIG.width)
        self.a = nn.Parameter(torch.ones(shape) + 0.01 * torch.randn(shape))
        self.b = nn.Parameter(0.05 * torch.randn(shape))
        self.c = nn.Parameter(0.10 * torch.randn(shape))
        self.d = nn.Parameter(0.25 * torch.randn(shape))
        self.e = nn.Parameter(0.25 * torch.randn(shape))
        self.f = nn.Parameter(torch.zeros(shape))
        self.output_scale = nn.Parameter(torch.ones(CONFIG.width))
        self.output_bias = nn.Parameter(torch.zeros(CONFIG.width))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = inputs
        for stage, offset in enumerate(OFFSETS):
            previous = torch.zeros_like(values)
            previous[:, offset:] = values[:, :-offset]
            inner = self.d[stage] * values + self.e[stage] * previous + self.f[stage]
            values = (
                self.a[stage] * values
                + self.b[stage] * previous
                + self.c[stage] * torch.tanh(inner)
            )
        return values * self.output_scale + self.output_bias


def build_model(variant: str, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    if variant == "dense_linear":
        model: nn.Module = DenseLinear()
    elif variant == "dense_ffn":
        model = DenseFFN(nonlinear=True)
    elif variant == "low_rank":
        model = DenseFFN(nonlinear=False)
    elif variant == "butterfly":
        model = Butterfly()
    elif variant == "lookup":
        model = Lookup()
    elif variant == "feature_dag":
        model = FeatureDAG()
    else:
        raise ValueError(f"unknown variant: {variant}")
    actual = sum(parameter.numel() for parameter in model.parameters())
    if actual != 1024:
        raise AssertionError(f"{variant} has {actual} parameters instead of 1024")
    return model


def variant_ledger(variant: str) -> dict[str, int]:
    common = {
        "trainable_scalars": 1024,
        "fp32_training_parameter_bytes": 4096,
        "bf16_serving_parameter_bytes": 2048,
    }
    ledgers = {
        "dense_linear": {
            "parameter_scalar_reads": 1024,
            "multiplications": 1024,
            "additions": 992,
            "transcendentals": 0,
            "comparisons": 0,
            "table_vector_lookups": 0,
            "runtime_fixed_buffer_bytes": 0,
        },
        "dense_ffn": {
            "parameter_scalar_reads": 1024,
            "multiplications": 1024,
            "additions": 976,
            "transcendentals": 16,
            "comparisons": 0,
            "table_vector_lookups": 0,
            "runtime_fixed_buffer_bytes": 0,
        },
        "low_rank": {
            "parameter_scalar_reads": 1024,
            "multiplications": 1024,
            "additions": 976,
            "transcendentals": 0,
            "comparisons": 0,
            "table_vector_lookups": 0,
            "runtime_fixed_buffer_bytes": 0,
        },
        "butterfly": {
            "parameter_scalar_reads": 1024,
            "multiplications": 1024,
            "additions": 512,
            "transcendentals": 0,
            "comparisons": 0,
            "table_vector_lookups": 0,
            "runtime_fixed_buffer_bytes": 4096,
        },
        "lookup": {
            "parameter_scalar_reads": 128,
            "multiplications": 0,
            "additions": 128,
            "transcendentals": 0,
            "comparisons": 12,
            "table_vector_lookups": 4,
            "runtime_fixed_buffer_bytes": 120,
        },
        "feature_dag": {
            "parameter_scalar_reads": 1024,
            "multiplications": 832,
            "additions": 672,
            "transcendentals": 160,
            "comparisons": 0,
            "table_vector_lookups": 0,
            "runtime_fixed_buffer_bytes": 0,
        },
    }
    if variant not in ledgers:
        raise ValueError(f"unknown variant: {variant}")
    return common | ledgers[variant]


@torch.no_grad()
def _nrmse(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> float:
    error = torch.mean(torch.square(model(inputs) - targets))
    scale = torch.mean(torch.square(targets - targets.mean(dim=0)))
    return float(torch.sqrt(error / scale).item())


def train_once(
    world_seed: int,
    target: str,
    variant: str,
    *,
    steps: int = CONFIG.steps,
    train_examples: int | None = None,
) -> dict[str, Any]:
    data = generate_target_data(world_seed, target)
    if train_examples is not None:
        data["train_x"] = data["train_x"][:train_examples]
        data["train_y"] = data["train_y"][:train_examples]
    tensors = {key: torch.from_numpy(value) for key, value in data.items()}
    model_seed = _stable_seed("model", world_seed, target, variant)
    model = build_model(variant, model_seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.learning_rate)
    batch_rng = torch.Generator(device="cpu")
    batch_rng.manual_seed(_stable_seed("batch", world_seed, target, variant))

    best_step = 0
    best_validation = _nrmse(
        model, tensors["validation_x"], tensors["validation_y"]
    )
    best_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    started = time.perf_counter()
    for step in range(1, steps + 1):
        indices = torch.randint(
            tensors["train_x"].shape[0],
            (min(CONFIG.batch_size, tensors["train_x"].shape[0]),),
            generator=batch_rng,
        )
        prediction = model(tensors["train_x"].index_select(0, indices))
        loss = torch.mean(
            torch.square(prediction - tensors["train_y"].index_select(0, indices))
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % CONFIG.validation_interval == 0 or step == steps:
            validation = _nrmse(
                model, tensors["validation_x"], tensors["validation_y"]
            )
            if validation < best_validation:
                best_validation = validation
                best_step = step
                best_state = {
                    name: value.detach().clone()
                    for name, value in model.state_dict().items()
                }

    model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    return {
        "world_seed": world_seed,
        "target": target,
        "variant": variant,
        "model_seed": model_seed,
        "steps": steps,
        "best_step": best_step,
        "train_nrmse": _nrmse(model, tensors["train_x"], tensors["train_y"]),
        "validation_nrmse": _nrmse(
            model, tensors["validation_x"], tensors["validation_y"]
        ),
        "test_nrmse": _nrmse(model, tensors["test_x"], tensors["test_y"]),
        "elapsed_seconds": elapsed,
    }


def screen_decision(result: dict[str, Any]) -> dict[str, Any]:
    runs = result.get("runs", [])
    keyed = {
        (run["world_seed"], run["target"], run["variant"]): run for run in runs
    }
    expected = {
        (world, target, variant)
        for world in WORLD_SEEDS
        for target in TARGETS
        for variant in VARIANTS
    }
    missing = sorted(expected - set(keyed))
    if missing:
        return {"status": "incomplete", "missing": missing}

    summaries: dict[str, dict[str, float]] = {}
    for target in TARGETS:
        summaries[target] = {}
        for variant in VARIANTS:
            values = [
                keyed[(world, target, variant)]["test_nrmse"]
                for world in WORLD_SEEDS
            ]
            summaries[target][variant] = float(np.median(values))

    prefix_pass = summaries["prefix"]["feature_dag"] <= 0.05
    hierarchy_wins = 0
    hierarchy_improvements = []
    mixed_noninferior = 0
    haar_ratios = []
    paired: list[dict[str, Any]] = []
    for world in WORLD_SEEDS:
        hierarchy_candidate = keyed[(world, "hierarchical", "feature_dag")][
            "test_nrmse"
        ]
        hierarchy_control = min(
            keyed[(world, "hierarchical", variant)]["test_nrmse"]
            for variant in VARIANTS
            if variant != "feature_dag"
        )
        hierarchy_wins += int(hierarchy_candidate < hierarchy_control)
        hierarchy_improvements.append(
            (hierarchy_control - hierarchy_candidate) / hierarchy_control
        )

        mixed_candidate = keyed[(world, "mixed", "feature_dag")]["test_nrmse"]
        mixed_control = min(
            keyed[(world, "mixed", variant)]["test_nrmse"]
            for variant in VARIANTS
            if variant != "feature_dag"
        )
        mixed_noninferior += int(mixed_candidate <= 1.05 * mixed_control)

        haar_candidate = keyed[(world, "haar", "feature_dag")]["test_nrmse"]
        haar_dense = keyed[(world, "haar", "dense_linear")]["test_nrmse"]
        haar_ratios.append(haar_candidate / haar_dense)
        paired.append(
            {
                "world_seed": world,
                "hierarchy_relative_improvement": hierarchy_improvements[-1],
                "mixed_relative_to_best_control": mixed_candidate / mixed_control,
                "haar_relative_to_dense_linear": haar_ratios[-1],
            }
        )

    median_hierarchy_improvement = float(np.median(hierarchy_improvements))
    structured_pass = (
        prefix_pass
        and hierarchy_wins >= 4
        and median_hierarchy_improvement >= 0.10
        and mixed_noninferior >= 4
    )
    haar_ratio = float(np.median(haar_ratios))
    if not structured_pass:
        status = "reject"
    elif haar_ratio <= 1.10:
        status = "pre_candidate"
    else:
        status = "mechanism_only"
    return {
        "status": status,
        "prefix_pass": prefix_pass,
        "hierarchy_world_wins": hierarchy_wins,
        "median_hierarchy_relative_improvement": median_hierarchy_improvement,
        "mixed_noninferior_worlds": mixed_noninferior,
        "median_haar_relative_to_dense_linear": haar_ratio,
        "median_test_nrmse": summaries,
        "paired_worlds": paired,
        "gpu_rental_justified": False,
    }


def self_test() -> dict[str, Any]:
    algebra = prefix_algebra_check()
    models = {
        variant: sum(
            parameter.numel()
            for parameter in build_model(variant, _stable_seed("self", variant)).parameters()
        )
        for variant in VARIANTS
    }
    sample = torch.randn(7, CONFIG.width)
    shapes = {
        variant: list(build_model(variant, 17)(sample).shape) for variant in VARIANTS
    }
    if algebra["max_abs_error"] >= 1e-6:
        raise AssertionError("prefix circuit failed its exact algebra check")
    if set(models.values()) != {1024}:
        raise AssertionError("parameter ledgers are not equal")
    return {
        "status": "pass",
        "device": "cpu",
        "torch_cuda_available": torch.cuda.is_available(),
        "prefix_algebra": algebra,
        "parameter_counts": models,
        "output_shapes": shapes,
    }


def run_screen() -> dict[str, Any]:
    torch.set_num_threads(1)
    result: dict[str, Any] = {
        "status": "running",
        "config": asdict(CONFIG),
        "world_seeds": list(WORLD_SEEDS),
        "targets": list(TARGETS),
        "variants": list(VARIANTS),
        "source_sha256": _sha256(Path(__file__)),
        "preregistration_sha256": _sha256(PREREGISTRATION),
        "self_test": self_test(),
        "ledgers": {variant: variant_ledger(variant) for variant in VARIANTS},
        "runs": [],
    }
    for world_seed in WORLD_SEEDS:
        for target in TARGETS:
            for variant in VARIANTS:
                run = train_once(world_seed, target, variant)
                result["runs"].append(run)
                print(
                    f"world={world_seed} target={target} variant={variant} "
                    f"test_nrmse={run['test_nrmse']:.6f}",
                    flush=True,
                )
    result["decision"] = screen_decision(result)
    result["status"] = "complete"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--screen", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if sum((args.self_test, args.smoke, args.screen)) != 1:
        parser.error("choose exactly one of --self-test, --smoke, or --screen")

    if args.self_test:
        result = self_test()
    elif args.smoke:
        torch.set_num_threads(1)
        result = {
            "status": "smoke",
            "run": train_once(
                WORLD_SEEDS[0], "hierarchical", "feature_dag", steps=2,
                train_examples=64,
            ),
        }
    else:
        result = run_screen()
    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()
