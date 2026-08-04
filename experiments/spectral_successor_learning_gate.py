#!/usr/bin/env python3
"""From-zero latent-plan learning gate for spectral successor supervision."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


VOCABULARY = 512
MODEL_WIDTH = 128
LAYERS = 3
HEADS = 4
FFN_WIDTH = 512
CLASSES = 64
PLAN_ALPHABET = 17
PLAN_REPEATS = 2
PLAN_LENGTH = PLAN_ALPHABET * PLAN_REPEATS
GAP = 48
SEQUENCE_LENGTH = 2 + GAP + 1 + PLAN_LENGTH
INPUT_LENGTH = SEQUENCE_LENGTH - 1
PLAN_TARGET_START = 2 + GAP
TARGET_WIDTH = 64
CHECKPOINTS = (0, 10, 20, 40, 80, 160, 320, 640, 800)
ARMS = ("ntp", "mtp8", "fsp_bow", "spectral", "spectral_scrambled")
TASKS = ("structured", "random_mapping", "order_insensitive")
PREREGISTRATION = Path(
    "results/spectral-successor-learning-gate-preregistration.md"
)
STAGE0_RESULT = Path("results/spectral-successor-stage0.json")
TEST = Path("tests/test_spectral_successor_learning_gate.py")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def structured_plans() -> torch.Tensor:
    positions = torch.arange(PLAN_ALPHABET)
    rows = []
    for class_index in range(CLASSES):
        multiplier = 1 + class_index % (PLAN_ALPHABET - 1)
        offset = class_index // (PLAN_ALPHABET - 1)
        permutation = torch.remainder(multiplier * positions + offset, PLAN_ALPHABET)
        rows.append(permutation.repeat(PLAN_REPEATS) + 128)
    return torch.stack(rows)


def random_plans(seed: int = 7171) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    rows = [
        (torch.randperm(PLAN_ALPHABET, generator=generator) + 128).repeat(
            PLAN_REPEATS
        )
        for _ in range(CLASSES)
    ]
    return torch.stack(rows)


def plan_table(task: str) -> torch.Tensor:
    if task == "structured":
        return structured_plans()
    if task == "random_mapping":
        return random_plans()
    if task == "order_insensitive":
        return (torch.arange(PLAN_ALPHABET) + 128).repeat(
            CLASSES, PLAN_REPEATS
        )
    raise ValueError(f"unknown task: {task}")


def make_batch(
    task: str,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    classes = torch.randint(0, CLASSES, (batch_size,), generator=generator)
    fillers = torch.randint(
        96, 128, (batch_size, GAP), generator=generator
    )
    plans = plan_table(task)[classes]
    sequence = torch.cat(
        (
            torch.zeros(batch_size, 1, dtype=torch.long),
            (16 + classes)[:, None],
            fillers,
            torch.full((batch_size, 1), 4, dtype=torch.long),
            plans,
        ),
        dim=1,
    )
    return sequence.to(device, non_blocking=True)


class CausalTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(VOCABULARY, MODEL_WIDTH)
        self.position_embedding = nn.Parameter(
            torch.empty(SEQUENCE_LENGTH, MODEL_WIDTH)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=MODEL_WIDTH,
            nhead=HEADS,
            dim_feedforward=FFN_WIDTH,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            bias=False,
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=LAYERS)
        self.final_norm = nn.LayerNorm(MODEL_WIDTH, elementwise_affine=True)
        self.lm_head = nn.Linear(MODEL_WIDTH, VOCABULARY, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(SEQUENCE_LENGTH, SEQUENCE_LENGTH, dtype=torch.bool),
                diagonal=1,
            ),
            persistent=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        standard_deviation = 1.0 / math.sqrt(MODEL_WIDTH)
        nn.init.normal_(
            self.token_embedding.weight, mean=0.0, std=standard_deviation
        )
        nn.init.normal_(
            self.position_embedding, mean=0.0, std=standard_deviation
        )
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.lm_head:
                nn.init.normal_(
                    module.weight, mean=0.0, std=standard_deviation
                )

    def forward(
        self, tokens: torch.Tensor, *, return_hidden: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        length = tokens.shape[1]
        hidden = self.token_embedding(tokens) + self.position_embedding[:length]
        hidden = self.layers(
            hidden, mask=self.causal_mask[:length, :length], is_causal=True
        )
        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden)
        if return_hidden:
            return logits, hidden
        return logits


class AuxiliaryTargets:
    def __init__(self, device: torch.device, seed: int = 9917) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.spectral_codes = (
            2.0
            * torch.randint(
                0, 2, (VOCABULARY, 4), generator=generator, dtype=torch.float32
            )
            - 1.0
        ).to(device)
        self.mtp_codes = (
            2.0
            * torch.randint(
                0, 2, (VOCABULARY, 8), generator=generator, dtype=torch.float32
            )
            - 1.0
        ).to(device)
        self.bow_codes = (
            2.0
            * torch.randint(
                0,
                2,
                (VOCABULARY, TARGET_WIDTH),
                generator=generator,
                dtype=torch.float32,
            )
            - 1.0
        ).to(device)
        frequency = 2.0 * math.pi * (
            torch.arange(8, dtype=torch.float32, device=device) + 0.5
        ) / 8.0
        self.gamma = 0.97
        self.cosine = torch.cos(frequency)
        self.sine = torch.sin(frequency)

    def spectral(self, sequence: torch.Tensor) -> torch.Tensor:
        batch, length = sequence.shape
        real = torch.zeros(batch, 8, 4, device=sequence.device)
        imaginary = torch.zeros_like(real)
        rows = []
        cosine = self.cosine[None, :, None]
        sine = self.sine[None, :, None]
        for position in range(length - 2, -1, -1):
            code = self.spectral_codes[sequence[:, position + 1]][:, None, :]
            next_real = code + self.gamma * (
                cosine * real - sine * imaginary
            )
            next_imaginary = self.gamma * (
                sine * real + cosine * imaginary
            )
            real, imaginary = next_real, next_imaginary
            rows.append(
                torch.cat((real, imaginary), dim=-1).flatten(1)
            )
        scale = math.sqrt(1.0 - self.gamma**2)
        return torch.stack(rows[::-1], dim=1) * scale

    def mtp8(self, sequence: torch.Tensor) -> torch.Tensor:
        batch, length = sequence.shape
        rows = []
        for offset in range(1, 9):
            indices = torch.arange(length - 1, device=sequence.device) + offset
            valid = indices < length
            clipped = indices.clamp_max(length - 1)
            values = self.mtp_codes[sequence[:, clipped]]
            values = values * valid[None, :, None]
            rows.append(values)
        return torch.cat(rows, dim=-1)

    def bow(self, sequence: torch.Tensor) -> torch.Tensor:
        codes = self.bow_codes[sequence]
        suffix = torch.flip(
            torch.cumsum(torch.flip(codes[:, 1:], dims=(1,)), dim=1),
            dims=(1,),
        )
        counts = torch.arange(
            sequence.shape[1] - 1,
            0,
            -1,
            dtype=torch.float32,
            device=sequence.device,
        )
        return suffix / counts.sqrt()[None, :, None]

    def target(self, arm: str, sequence: torch.Tensor) -> torch.Tensor:
        if arm == "mtp8":
            return self.mtp8(sequence)
        if arm == "fsp_bow":
            return self.bow(sequence)
        if arm in {"spectral", "spectral_scrambled"}:
            target = self.spectral(sequence)
            return target if arm == "spectral" else target.roll(1, dims=0)
        raise ValueError(f"arm has no auxiliary target: {arm}")


def learning_rate_multiplier(step: int, total_steps: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(total_steps - warmup - 1, 1)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def forward_macs_per_sequence() -> int:
    transformer = INPUT_LENGTH * LAYERS * (
        12 * MODEL_WIDTH**2 + 2 * INPUT_LENGTH * MODEL_WIDTH
    )
    unembedding = INPUT_LENGTH * MODEL_WIDTH * VOCABULARY
    return transformer + unembedding


def charged_work_per_step(batch_size: int, arm: str) -> int:
    base = 3 * batch_size * forward_macs_per_sequence()
    if arm == "ntp":
        return base
    auxiliary = 3 * batch_size * INPUT_LENGTH * MODEL_WIDTH * TARGET_WIDTH
    target = batch_size * INPUT_LENGTH * 2 * TARGET_WIDTH
    return base + auxiliary + target


@torch.no_grad()
def evaluate(
    model: CausalTransformer,
    task: str,
    device: torch.device,
    seed: int,
    examples: int = 256,
    greedy: bool = False,
) -> dict[str, float]:
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    sequence = make_batch(task, examples, generator, device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(sequence[:, :-1])
    targets = sequence[:, 1:]
    float_logits = logits.float()
    total_nll = float(
        F.cross_entropy(
            float_logits.reshape(-1, VOCABULARY), targets.reshape(-1)
        )
    )
    plan_logits = float_logits[:, PLAN_TARGET_START:]
    plan_targets = targets[:, PLAN_TARGET_START:]
    plan_nll = float(
        F.cross_entropy(
            plan_logits.reshape(-1, VOCABULARY), plan_targets.reshape(-1)
        )
    )
    predictions = plan_logits.argmax(dim=-1)
    first_two = float(
        (predictions[:, :2] == plan_targets[:, :2]).all(dim=-1).float().mean()
    )
    plan_accuracy = float((predictions == plan_targets).float().mean())
    result = {
        "total_nll": total_nll,
        "plan_nll": plan_nll,
        "first_two_exact": first_two,
        "plan_token_accuracy": plan_accuracy,
    }
    if greedy:
        generated = sequence[:, : PLAN_TARGET_START + 1]
        for _ in range(PLAN_LENGTH):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                next_logits = model(generated)
            next_token = next_logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)
        gold = sequence[:, -PLAN_LENGTH:]
        produced = generated[:, -PLAN_LENGTH:]
        result["greedy_exact_plan"] = float(
            (produced == gold).all(dim=-1).float().mean()
        )
        result["greedy_plan_token_accuracy"] = float(
            (produced == gold).float().mean()
        )
    return result


def run_arm(
    *,
    arm: str,
    task: str,
    training_seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    warmup: int,
    device: torch.device,
) -> dict[str, Any]:
    seed_everything(training_seed)
    model = CausalTransformer().to(device)
    base_parameters = sum(parameter.numel() for parameter in model.parameters())
    auxiliary_head = None
    if arm != "ntp":
        torch.manual_seed(training_seed + 100_000)
        auxiliary_head = nn.Linear(
            MODEL_WIDTH, TARGET_WIDTH, bias=False, device=device
        )
        nn.init.normal_(
            auxiliary_head.weight, mean=0.0, std=1.0 / math.sqrt(MODEL_WIDTH)
        )
    parameters = list(model.parameters())
    if auxiliary_head is not None:
        parameters += list(auxiliary_head.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        fused=True,
    )
    targets_factory = AuxiliaryTargets(device)
    data_generator = torch.Generator().manual_seed(training_seed + 200_000)
    evaluations = []
    work_per_step = charged_work_per_step(batch_size, arm)
    losses = []
    auxiliary_losses = []
    gradient_norms = []
    started = time.perf_counter()
    for step in range(steps + 1):
        if step in CHECKPOINTS:
            measurement = evaluate(
                model,
                task,
                device,
                seed=training_seed + 300_000,
                greedy=step == steps,
            )
            measurement.update(
                {
                    "step": step,
                    "charged_multiply_like_work": step * work_per_step,
                }
            )
            evaluations.append(measurement)
            print(
                json.dumps(
                    {
                        "task": task,
                        "seed": training_seed,
                        "arm": arm,
                        "step": step,
                        "first_two_exact": measurement["first_two_exact"],
                        "plan_nll": measurement["plan_nll"],
                    }
                ),
                flush=True,
            )
        if step == steps:
            break
        model.train()
        if auxiliary_head is not None:
            auxiliary_head.train()
        sequence = make_batch(task, batch_size, data_generator, device)
        optimizer.param_groups[0]["lr"] = learning_rate * learning_rate_multiplier(
            step, steps, warmup
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, hidden = model(sequence[:, :-1], return_hidden=True)
        language = F.cross_entropy(
            logits.float().reshape(-1, VOCABULARY), sequence[:, 1:].reshape(-1)
        )
        auxiliary = torch.zeros((), device=device)
        if auxiliary_head is not None:
            target = targets_factory.target(arm, sequence)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predicted = auxiliary_head(hidden)
            auxiliary = F.mse_loss(predicted.float(), target.float())
        objective = language + auxiliary
        objective.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(objective) or not torch.isfinite(norm):
            raise RuntimeError(f"non-finite training: {task}/{training_seed}/{arm}")
        optimizer.step()
        losses.append(float(language.detach()))
        auxiliary_losses.append(float(auxiliary.detach()))
        gradient_norms.append(float(norm.detach()))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    exported_keys = tuple(sorted(model.state_dict()))
    exported_parameters = sum(parameter.numel() for parameter in model.parameters())
    result = {
        "arm": arm,
        "task": task,
        "training_seed": training_seed,
        "base_parameters": base_parameters,
        "exported_parameters": exported_parameters,
        "training_only_parameters": (
            0 if auxiliary_head is None else sum(p.numel() for p in auxiliary_head.parameters())
        ),
        "exported_state_keys_sha256": hashlib.sha256(
            "\n".join(exported_keys).encode()
        ).hexdigest(),
        "work_per_step": work_per_step,
        "evaluations": evaluations,
        "train": {
            "mean_language_loss": float(np.mean(losses)),
            "final_language_loss": losses[-1],
            "mean_auxiliary_loss": float(np.mean(auxiliary_losses)),
            "maximum_gradient_norm": max(gradient_norms),
            "elapsed_seconds": elapsed,
            "charged_multiply_like_work": steps * work_per_step,
        },
    }
    del optimizer, auxiliary_head, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def first_crossing(result: dict[str, Any], threshold: float) -> int | None:
    for row in result["evaluations"]:
        if row["first_two_exact"] >= threshold:
            return int(row["charged_multiply_like_work"])
    return None


def decide(results: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    by_key = {
        (row["task"], row["training_seed"], row["arm"]): row for row in results
    }
    compute_gains = []
    structured_rows = []
    for seed in seeds:
        candidate = by_key[("structured", seed, "spectral")]
        candidate_crossing = first_crossing(candidate, 0.90)
        controls = [
            by_key[("structured", seed, arm)]
            for arm in ARMS
            if arm != "spectral"
        ]
        control_crossings = [first_crossing(row, 0.90) for row in controls]
        finite_controls = [value for value in control_crossings if value is not None]
        best_control = min(finite_controls) if finite_controls else None
        gain = (
            None
            if candidate_crossing is None
            else (
                float("inf")
                if best_control is None
                else best_control / candidate_crossing
            )
        )
        compute_gains.append(gain)
        structured_rows.append(
            {
                "seed": seed,
                "candidate_work_to_90_percent": candidate_crossing,
                "best_control_work_to_90_percent": best_control,
                "compute_gain": gain,
                "candidate_terminal": candidate["evaluations"][-1],
                "best_control_terminal_total_nll": min(
                    row["evaluations"][-1]["total_nll"] for row in controls
                ),
            }
        )
    no_general_regression = []
    random_memory_noninferiority = []
    for seed in seeds:
        for task in ("order_insensitive",):
            baseline = by_key[(task, seed, "ntp")]["evaluations"][-1]
            candidate = by_key[(task, seed, "spectral")]["evaluations"][-1]
            no_general_regression.append(
                candidate["total_nll"] <= 1.02 * baseline["total_nll"]
            )
        baseline = by_key[("random_mapping", seed, "ntp")]["evaluations"][-1]
        candidate = by_key[("random_mapping", seed, "spectral")]["evaluations"][-1]
        random_memory_noninferiority.append(
            candidate["first_two_exact"] + 0.02 >= baseline["first_two_exact"]
        )
    export_hashes = {row["exported_state_keys_sha256"] for row in results}
    base_parameter_counts = {row["base_parameters"] for row in results}
    exported_parameter_counts = {row["exported_parameters"] for row in results}
    gates = {
        "spectral_reaches_90_percent_both_seeds": all(
            value is not None for value in compute_gains
        ),
        "spectral_at_least_1p25x_compute_gain_both_seeds": all(
            value is not None and value >= 1.25 for value in compute_gains
        ),
        "structured_terminal_total_nll_within_one_percent": all(
            row["candidate_terminal"]["total_nll"]
            <= 1.01 * row["best_control_terminal_total_nll"]
            for row in structured_rows
        ),
        "order_insensitive_total_nll_noninferior": all(no_general_regression),
        "random_mapping_acquisition_noninferior": all(
            random_memory_noninferiority
        ),
        "identical_export_graph_and_parameter_count": len(export_hashes) == 1
        and len(base_parameter_counts) == 1
        and len(exported_parameter_counts) == 1
        and exported_parameter_counts == base_parameter_counts,
        "training_only_state_absent_from_export": all(
            row["exported_parameters"] == row["base_parameters"]
            for row in results
        ),
    }
    return {
        "threshold": 0.90,
        "structured": structured_rows,
        "gates": gates,
        "advance": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--seeds", type=str, default="731,947")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/spectral-successor-learning-gate.json"),
    )
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    if (
        args.steps != 800
        or args.batch_size != 128
        or args.learning_rate != 1e-3
        or args.warmup != 40
        or seeds != [731, 947]
    ):
        raise ValueError("arguments do not match the frozen protocol")
    stage0 = json.loads(STAGE0_RESULT.read_text())
    if not stage0.get("pass"):
        raise RuntimeError("spectral successor Stage 0 did not pass")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    run_specs = []
    for seed in seeds:
        for arm in ARMS:
            run_specs.append(("structured", seed, arm))
        for task in ("random_mapping", "order_insensitive"):
            run_specs.append((task, seed, "ntp"))
            run_specs.append((task, seed, "spectral"))
    results = []
    for task, seed, arm in run_specs:
        results.append(
            run_arm(
                arm=arm,
                task=task,
                training_seed=seed,
                steps=args.steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                warmup=args.warmup,
                device=device,
            )
        )
    payload = {
        "schema": "spectral-successor-learning-gate-v1",
        "source_sha256": sha256_file(Path(__file__)),
        "test_sha256": sha256_file(TEST),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "stage0_result_sha256": sha256_file(STAGE0_RESULT),
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
        },
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "ledger": {
            "forward_macs_per_sequence": forward_macs_per_sequence(),
            "ntp_work_per_step": charged_work_per_step(args.batch_size, "ntp"),
            "auxiliary_work_per_step": charged_work_per_step(
                args.batch_size, "spectral"
            ),
            "work_ratio": charged_work_per_step(args.batch_size, "spectral")
            / charged_work_per_step(args.batch_size, "ntp"),
            "excluded": "softmax, normalization, GELU, optimizer scalar operations, memory traffic",
        },
        "results": results,
    }
    payload["decision"] = decide(results, seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
