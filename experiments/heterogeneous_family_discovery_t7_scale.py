#!/usr/bin/env python3
"""T7 sealed adjacent-scale comparison after frozen Muon calibration."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from experiments import heterogeneous_family_discovery_t7_pilot as pilot
from experiments.heterogeneous_family_discovery_t7_pilot import *  # noqa: F403


FULL_SOURCE = Path(__file__)
PILOT_SOURCE = Path(pilot.__file__)
SELECTED_MUON_LEARNING_RATE = 0.005
PILOT_ARTIFACT_SHA256 = "5f4792b25162b1b3abdc6fb7c9948774a225e64b29e3edfbcdac144ad107c8cb"
INSTANCE_HOURLY_USD = 3.99816


def verify_pilot_selection() -> dict[str, object]:
    if sha256_file(PILOT_OUTPUT) != PILOT_ARTIFACT_SHA256:  # noqa: F405
        raise RuntimeError("Muon pilot artifact hash mismatch")
    artifact = json.loads(PILOT_OUTPUT.read_text())  # noqa: F405
    if artifact["selected_muon_learning_rate"] != SELECTED_MUON_LEARNING_RATE:
        raise RuntimeError("sealed Muon learning rate disagrees with pilot")
    return artifact


def natural_batch_with_intervals(
    stream: TokenStream,  # noqa: F405
    seed: int,
    batch_size: int,
    length: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, list[tuple[int, int]]]:
    generator = np.random.default_rng(seed)
    starts = generator.integers(0, len(stream.tokens) - length - 1, size=batch_size)
    rows = np.stack(
        [
            np.asarray(stream.tokens[start : start + length + 1], dtype=np.int64)
            for start in starts
        ]
    )
    values = torch.from_numpy(rows).to(device)
    intervals = [(int(start), int(start) + length + 1) for start in starts]
    return values[:, :-1], values[:, 1:], intervals


def interval_union_size(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    ordered = sorted(intervals)
    total = 0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start > end:
            total += end - start
            start, end = next_start, next_end
        else:
            end = max(end, next_end)
    return total + end - start


def adamw_step_flops(model: torch.nn.Module, *, parameters: str = "all") -> int:
    """Dominant elementwise estimate: moments, normalization, decay, update."""
    count = 0
    for name, parameter in model.named_parameters():
        matrix = parameter.ndim == 2 and name != "token.weight"
        if parameters == "auxiliary" and matrix:
            continue
        count += parameter.numel()
    return 10 * count


def evaluate_algorithm_scaled(
    model: ScaleLM,  # noqa: F405
    world: DiscoveryWorld,  # noqa: F405
    layout: TokenLayout,  # noqa: F405
    device: torch.device,
    seed: int,
    *,
    examples_per_task: int = 128,
) -> dict[str, object]:
    return t6.evaluate_algorithm(  # noqa: F405
        model,
        world,
        layout,
        device,
        seed,
        examples_per_task=examples_per_task,
    )


def evaluation_flops(*, natural_batches: int, examples_per_task: int) -> int:
    natural = natural_batches * model_forward_flops(  # noqa: F405
        8, CONTEXT, binary_head=False  # noqa: F405
    )
    structured = model_forward_flops(  # noqa: F405
        t6.TASKS * examples_per_task,  # noqa: F405
        t6.INPUT_DIM + 2,  # noqa: F405
        binary_head=True,
    )
    copy = model_forward_flops(  # noqa: F405
        4_096,
        t6.INPUT_DIM + 2,  # noqa: F405
        binary_head=True,
    )
    return natural + structured + copy


def discovery_operation_estimate() -> dict[str, int]:
    supports = math.comb(t6.INPUT_DIM, t6.SUPPORT_SIZE)  # noqa: F405
    matmul_multiply_adds = (
        t6.TASKS  # noqa: F405
        * t6.PREFIX_PER_TASK  # noqa: F405
        * t6.INPUT_DIM  # noqa: F405
        * supports
    )
    family_boolean_comparisons = (
        t6.TASKS  # noqa: F405
        * len(t6.FAMILY_NAMES)  # noqa: F405
        * t6.PREFIX_PER_TASK  # noqa: F405
        * supports
    )
    return {
        "candidate_supports": supports,
        "integer_matmul_multiply_adds": matmul_multiply_adds,
        "family_boolean_comparisons": family_boolean_comparisons,
    }


def train_arm(
    arm: str,
    model_seed: int,
    world: DiscoveryWorld,  # noqa: F405
    recovered_supports: Tensor,
    recovered_families: Tensor,
    accepted: Tensor,
    layout: TokenLayout,  # noqa: F405
    natural: TokenStream,  # noqa: F405
    validation: TokenStream,  # noqa: F405
    device: torch.device,
    *,
    discovery_seconds: float,
    world_generation_seconds: float,
) -> dict[str, object]:
    if arm not in {"muon_1x", "compiler_muon_1x", "muon_2x", "adamw_2x"}:
        raise ValueError(f"unknown arm {arm}")
    compiler_arm = arm == "compiler_muon_1x"
    optimizer_kind = "adamw" if arm == "adamw_2x" else "muon"
    steps = CHECKPOINTS_1X[-1] if arm.endswith("1x") else STEPS_2X  # noqa: F405

    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    total_started = time.perf_counter()
    model_started = time.perf_counter()
    model = build_model(model_seed, device)  # noqa: F405
    if device.type == "cuda":
        torch.cuda.synchronize()
    model_initialization_seconds = time.perf_counter() - model_started
    initial_hash = state_sha256(model)  # noqa: F405

    frozen: FrozenEntries | None = None  # noqa: F405
    compiler: dict[str, object] | None = None
    compilation_seconds = 0.0
    if compiler_arm:
        compilation_started = time.perf_counter()
        frozen, compiler = compile_interpreter(  # noqa: F405
            model,
            recovered_supports,
            recovered_families,
            accepted,
            layout,
            world,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        compilation_seconds = time.perf_counter() - compilation_started

    optimizer = build_optimizer(  # noqa: F405
        model,
        optimizer_kind,
        muon_learning_rate=SELECTED_MUON_LEARNING_RATE,
    )
    flat_bits = world.prefix_bits.reshape(-1, t6.INPUT_DIM)  # noqa: F405
    flat_labels = world.prefix_labels.reshape(-1)
    flat_tasks = torch.arange(t6.TASKS).repeat_interleave(t6.PREFIX_PER_TASK)  # noqa: F405

    prefix_started = time.perf_counter()
    prefix_losses: list[float] = []
    prefix_flops = 0
    power_samples: list[float] = []
    if not compiler_arm:
        permutation = torch.randperm(
            len(flat_bits),
            generator=torch.Generator().manual_seed(model_seed + 50_000),
        )
        for prefix_step, start in enumerate(
            range(0, len(permutation), ALGORITHM_BATCH), 1  # noqa: F405
        ):
            indices = permutation[start : start + ALGORITHM_BATCH]  # noqa: F405
            tokens = t6.encode_algorithm(  # noqa: F405
                flat_bits[indices],
                flat_tasks[indices],
                torch.ones(len(indices), dtype=torch.bool),
                layout,
                device,
            )
            targets = flat_labels[indices].to(device)
            optimizer.set_step(min(prefix_step, STEPS_2X))  # noqa: F405
            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model.binary_logits(tokens, layout)
            loss = F.cross_entropy(logits.float(), targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite {arm} prefix loss at {prefix_step}")
            loss.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            )
            if not math.isfinite(gradient_norm):
                raise RuntimeError(f"nonfinite {arm} prefix gradient at {prefix_step}")
            optimizer.step()
            prefix_losses.append(float(loss.item()))
            prefix_flops += training_flops(  # noqa: F405
                len(indices), t6.INPUT_DIM + 2, binary_head=True  # noqa: F405
            )
            if prefix_step % 20 == 0 and device.type == "cuda":
                sample = read_gpu_power_watts()  # noqa: F405
                if sample is not None:
                    power_samples.append(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
    prefix_seconds = time.perf_counter() - prefix_started

    checkpoints: dict[str, object] = {}
    maximum_gradient_norm = 0.0
    maximum_loss = max(prefix_losses, default=0.0)
    natural_input_tokens = 0
    algorithm_raw_input_tokens = len(flat_bits) * (t6.INPUT_DIM + 2)  # noqa: F405
    algorithm_raw_targets = len(flat_bits)
    algorithm_model_input_tokens = (
        0 if compiler_arm else len(flat_bits) * (t6.INPUT_DIM + 2)  # noqa: F405
    )
    algorithm_model_target_presentations = 0 if compiler_arm else len(flat_bits)
    natural_intervals: list[tuple[int, int]] = []
    natural_training_flops = 0
    algorithm_training_flops = prefix_flops
    evaluation_total_flops = 0
    training_started = time.perf_counter()
    checkpoint_set = set(CHECKPOINTS_1X) | ({STEPS_2X} if steps == STEPS_2X else set())  # noqa: F405
    for step in range(1, steps + 1):
        model.train()
        optimizer.set_step(step)
        optimizer.zero_grad()
        if step % ALGORITHM_INTERVAL == 0:  # noqa: F405
            tokens, targets, _, _ = t6.make_algorithm_batch(  # noqa: F405
                world,
                layout,
                ALGORITHM_BATCH,  # noqa: F405
                model_seed * 1_000_003 + step,
                device,
            )
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model.binary_logits(tokens, layout)
            loss = F.cross_entropy(logits.float(), targets)
            mixed_input_tokens = ALGORITHM_BATCH * (t6.INPUT_DIM + 2)  # noqa: F405
            algorithm_raw_input_tokens += mixed_input_tokens
            algorithm_raw_targets += ALGORITHM_BATCH  # noqa: F405
            algorithm_model_input_tokens += mixed_input_tokens
            algorithm_model_target_presentations += ALGORITHM_BATCH  # noqa: F405
            algorithm_training_flops += training_flops(  # noqa: F405
                ALGORITHM_BATCH, t6.INPUT_DIM + 2, binary_head=True  # noqa: F405
            )
        else:
            inputs, targets, intervals = natural_batch_with_intervals(
                natural,
                model_seed * 1_000_003 + step,
                NATURAL_BATCH,  # noqa: F405
                CONTEXT,  # noqa: F405
                device,
            )
            natural_intervals.extend(intervals)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(inputs)
            loss = F.cross_entropy(logits.float().reshape(-1, VOCAB), targets.reshape(-1))  # noqa: F405
            natural_input_tokens += NATURAL_BATCH * CONTEXT  # noqa: F405
            natural_training_flops += training_flops(  # noqa: F405
                NATURAL_BATCH, CONTEXT, binary_head=False  # noqa: F405
            )
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite {arm} loss at step {step}")
        loss.backward()
        if frozen is not None:
            frozen.mask_gradients(model)
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        )
        if not math.isfinite(gradient_norm):
            raise RuntimeError(f"nonfinite {arm} gradient at step {step}")
        optimizer.step()
        if frozen is not None:
            frozen.enforce(model)
        maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)
        maximum_loss = max(maximum_loss, float(loss.item()))
        if step % 20 == 0 and device.type == "cuda":
            sample = read_gpu_power_watts()  # noqa: F405
            if sample is not None:
                power_samples.append(sample)
        if step in checkpoint_set:
            evaluation_started = time.perf_counter()
            natural_result = evaluate_natural(  # noqa: F405
                model,
                validation,
                device,
                EVALUATION_SEED + model_seed,  # noqa: F405
                batches=32,
            )
            algorithm_result = evaluate_algorithm_scaled(
                model,
                world,
                layout,
                device,
                EVALUATION_SEED + model_seed + step,  # noqa: F405
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            eval_seconds = time.perf_counter() - evaluation_started
            eval_flops = evaluation_flops(natural_batches=32, examples_per_task=128)
            evaluation_total_flops += eval_flops
            checkpoints[str(step)] = {
                "natural": natural_result,
                "algorithm": algorithm_result,
                "natural_input_tokens": natural_input_tokens,
                "algorithm_raw_input_tokens": algorithm_raw_input_tokens,
                "algorithm_model_input_token_presentations": algorithm_model_input_tokens,
                "evaluation_seconds": eval_seconds,
                "evaluation_flops": eval_flops,
            }
    if device.type == "cuda":
        torch.cuda.synchronize()
    mixed_training_seconds = time.perf_counter() - training_started
    optimizer_steps = (0 if compiler_arm else math.ceil(len(flat_bits) / ALGORITHM_BATCH)) + steps  # noqa: F405
    if optimizer_kind == "muon":
        matrix_optimizer_flops = optimizer_steps * muon_step_flops(model)  # noqa: F405
        auxiliary_optimizer_flops = optimizer_steps * adamw_step_flops(
            model, parameters="auxiliary"
        )
    else:
        matrix_optimizer_flops = 0
        auxiliary_optimizer_flops = optimizer_steps * adamw_step_flops(model)
    mean_power = float(np.mean(power_samples)) if power_samples else None
    training_model_flops = natural_training_flops + algorithm_training_flops
    terminal_hash = state_sha256(model)  # noqa: F405
    total_seconds = time.perf_counter() - total_started
    power_accounted_seconds = prefix_seconds + mixed_training_seconds
    return {
        "arm": arm,
        "optimizer": optimizer.kind,
        "steps": steps,
        "initial_hash": initial_hash,
        "terminal_hash": terminal_hash,
        "compiler": compiler,
        "prefix_updates": 0 if compiler_arm else len(prefix_losses),
        "prefix_loss_first": prefix_losses[0] if prefix_losses else None,
        "prefix_loss_last": prefix_losses[-1] if prefix_losses else None,
        "checkpoints": checkpoints,
        "maximum_gradient_norm": maximum_gradient_norm,
        "maximum_loss": maximum_loss,
        "resource_ledger": {
            "model_initialization_seconds": model_initialization_seconds,
            "shared_world_generation_seconds_charged": world_generation_seconds,
            "world_discovery_seconds_charged": discovery_seconds if compiler_arm else 0.0,
            "gpu_compilation_seconds": compilation_seconds,
            "prefix_training_seconds": prefix_seconds,
            "mixed_training_and_evaluation_seconds": mixed_training_seconds,
            "total_wall_seconds": total_seconds,
            "gpu_seconds": total_seconds if device.type == "cuda" else 0.0,
            "instance_hourly_usd": INSTANCE_HOURLY_USD,
            "estimated_instance_cost_usd": total_seconds * INSTANCE_HOURLY_USD / 3_600.0,
            "mean_sampled_power_watts": mean_power,
            "power_accounted_seconds": power_accounted_seconds,
            "estimated_joules": (
                mean_power * power_accounted_seconds if mean_power is not None else None
            ),
            "peak_hbm_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
            "peak_hbm_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
            ),
            "optimizer_state_bytes": optimizer_state_bytes(optimizer),  # noqa: F405
            "natural_input_token_presentations": natural_input_tokens,
            "natural_unique_raw_positions": interval_union_size(natural_intervals),
            "algorithm_prefix_examples_consumed": len(flat_bits),
            "algorithm_raw_input_tokens_consumed": algorithm_raw_input_tokens,
            "algorithm_raw_targets_consumed": algorithm_raw_targets,
            "algorithm_model_input_token_presentations": algorithm_model_input_tokens,
            "algorithm_model_target_presentations": algorithm_model_target_presentations,
            "training_model_flops_bf16": training_model_flops,
            "muon_optimizer_flops_fp32": matrix_optimizer_flops,
            "adamw_optimizer_elementwise_flops_fp32": auxiliary_optimizer_flops,
            "evaluation_model_flops_bf16": evaluation_total_flops,
            "estimated_total_issued_flops": training_model_flops
            + matrix_optimizer_flops
            + auxiliary_optimizer_flops
            + evaluation_total_flops,
            "failed_or_retried_steps": 0,
            "discarded_candidate_samples": 0,
        },
    }


def run(device: torch.device, *, quick: bool = False) -> dict[str, object]:
    pilot_artifact = verify_pilot_selection()
    natural = TokenStream(t6.t3.TRAIN_FILE)  # noqa: F405
    validation = TokenStream(t6.t3.VALIDATION_FILE)  # noqa: F405
    layout, unused_count = choose_token_layout(natural, validation)  # noqa: F405

    world_started = time.perf_counter()
    world = t6.make_world(WORLD_SEED)  # noqa: F405
    world_generation_seconds = time.perf_counter() - world_started
    discovery_started = time.perf_counter()
    recovered_supports, recovered_families, accepted = t6.discover(world)  # noqa: F405
    discovery_seconds = time.perf_counter() - discovery_started
    exact_supports = bool(
        torch.equal(
            recovered_supports[: t6.STRUCTURED_TASKS],  # noqa: F405
            world.supports[: t6.STRUCTURED_TASKS],  # noqa: F405
        )
    )
    exact_families = bool(
        torch.equal(
            recovered_families[: t6.STRUCTURED_TASKS],  # noqa: F405
            world.families[: t6.STRUCTURED_TASKS],  # noqa: F405
        )
    )
    zero_false_accepts = not bool(accepted[t6.STRUCTURED_TASKS :].any())  # noqa: F405

    worlds: dict[str, object] = {}
    if quick:
        model = build_model(MODEL_SEEDS[0], device)  # noqa: F405
        initial_hash = state_sha256(model)  # noqa: F405
        frozen, compiler = compile_interpreter(  # noqa: F405
            model,
            recovered_supports,
            recovered_families,
            accepted,
            layout,
            world,
        )
        frozen.enforce(model)
        worlds[str(MODEL_SEEDS[0])] = {  # noqa: F405
            "initial_hash": initial_hash,
            "compiler": compiler,
            "algorithm": evaluate_algorithm_scaled(
                model,
                world,
                layout,
                device,
                EVALUATION_SEED,  # noqa: F405
                examples_per_task=64,
            ),
        }
    else:
        arm_orders = (
            ("muon_1x", "compiler_muon_1x", "muon_2x", "adamw_2x"),
            ("compiler_muon_1x", "adamw_2x", "muon_1x", "muon_2x"),
            ("muon_2x", "muon_1x", "adamw_2x", "compiler_muon_1x"),
        )
        for seed_index, seed in enumerate(MODEL_SEEDS):  # noqa: F405
            arms = {}
            for arm in arm_orders[seed_index]:
                arms[arm] = train_arm(
                    arm,
                    seed,
                    world,
                    recovered_supports,
                    recovered_families,
                    accepted,
                    layout,
                    natural,
                    validation,
                    device,
                    discovery_seconds=discovery_seconds,
                    world_generation_seconds=world_generation_seconds,
                )
            worlds[str(seed)] = {
                "arms": arms,
                "execution_order": arm_orders[seed_index],
                "identical_initial_hashes": len(
                    {result["initial_hash"] for result in arms.values()}
                )
                == 1,
            }

    gates: dict[str, bool] = {}
    if quick:
        algorithm = next(iter(worlds.values()))["algorithm"]
        gates = {
            "exact_supports": exact_supports,
            "exact_families": exact_families,
            "zero_random_accepts": zero_false_accepts,
            "all_family_means_ge_99": all(
                value["mean"] >= 0.99 for value in algorithm["per_family"].values()
            ),
            "all_family_mins_ge_99": all(
                value["min"] >= 0.99 for value in algorithm["per_family"].values()
            ),
            "random_between_40_60": 0.40 <= algorithm["random_mean"] <= 0.60,
            "copy_ge_99": algorithm["protected_copy"] >= 0.99,
        }
    else:
        for seed, result in worlds.items():
            arms = result["arms"]
            baseline = arms["muon_1x"]
            candidate = arms["compiler_muon_1x"]
            muon_2x = arms["muon_2x"]
            adamw_2x = arms["adamw_2x"]
            gates[f"{seed}_same_initialization"] = result["identical_initial_hashes"]
            gates[f"{seed}_exact_supports"] = exact_supports
            gates[f"{seed}_exact_families"] = exact_families
            gates[f"{seed}_zero_random_accepts"] = zero_false_accepts
            for checkpoint in CHECKPOINTS_1X:  # noqa: F405
                key = str(checkpoint)
                algorithm = candidate["checkpoints"][key]["algorithm"]
                gates[f"{seed}_{key}_all_family_means_ge_99"] = all(
                    value["mean"] >= 0.99
                    for value in algorithm["per_family"].values()
                )
                gates[f"{seed}_{key}_all_family_mins_ge_99"] = all(
                    value["min"] >= 0.99
                    for value in algorithm["per_family"].values()
                )
                gates[f"{seed}_{key}_random_between_40_60"] = (
                    0.40 <= algorithm["random_mean"] <= 0.60
                )
                gates[f"{seed}_{key}_copy_ge_99"] = algorithm["protected_copy"] >= 0.99
                candidate_nll = candidate["checkpoints"][key]["natural"]["nll"]
                baseline_nll = baseline["checkpoints"][key]["natural"]["nll"]
                gates[f"{seed}_{key}_natural_within_0p5pct"] = (
                    candidate_nll <= 1.005 * baseline_nll
                )
            for control_name, control in (("muon2x", muon_2x), ("adamw2x", adamw_2x)):
                algorithm = control["checkpoints"][str(STEPS_2X)]["algorithm"]  # noqa: F405
                gates[f"{seed}_{control_name}_parity_lt_80"] = (
                    algorithm["per_family"]["parity"]["mean"] < 0.80
                )
                gates[f"{seed}_{control_name}_mod3_lt_80"] = (
                    algorithm["per_family"]["mod3"]["mean"] < 0.80
                )
                gates[f"{seed}_{control_name}_copy_ge_95"] = (
                    algorithm["protected_copy"] >= 0.95
                )
            gates[f"{seed}_candidate_terminal_improves"] = (
                candidate["checkpoints"]["1000"]["natural"]["nll"]
                <= candidate["checkpoints"]["500"]["natural"]["nll"]
            )
            gates[f"{seed}_finite_training"] = all(
                math.isfinite(arm["maximum_gradient_norm"])
                and math.isfinite(arm["maximum_loss"])
                and arm["maximum_loss"] < 100.0
                and arm["resource_ledger"]["failed_or_retried_steps"] == 0
                for arm in arms.values()
            )

    return {
        "schema": "heterogeneous-family-discovery-t7-adjacent-scale-v1",
        "status": "quick" if quick else "complete",
        "device": str(device),
        "torch_version": torch.__version__,
        "configuration": {
            "vocab": VOCAB,  # noqa: F405
            "hidden": HIDDEN,  # noqa: F405
            "layers": LAYERS,  # noqa: F405
            "heads": HEADS,  # noqa: F405
            "ffn_width": FFN_WIDTH,  # noqa: F405
            "parameter_count": parameter_count(),  # noqa: F405
            "model_seeds": MODEL_SEEDS,  # noqa: F405
            "world_seed": WORLD_SEED,  # noqa: F405
            "evaluation_seed": EVALUATION_SEED,  # noqa: F405
            "selected_muon_learning_rate": SELECTED_MUON_LEARNING_RATE,
            "tasks": t6.TASKS,  # noqa: F405
            "structured_tasks": t6.STRUCTURED_TASKS,  # noqa: F405
            "random_tasks": t6.RANDOM_TASKS,  # noqa: F405
            "families": t6.FAMILY_NAMES,  # noqa: F405
            "unused_tokens_available": unused_count,
            "selected_special_tokens": layout.all_special,
        },
        "discovery": {
            "world_generation_seconds": world_generation_seconds,
            "compiler_discovery_seconds": discovery_seconds,
            "operations": discovery_operation_estimate(),
            "exact_supports": exact_supports,
            "exact_families": exact_families,
            "accepted_structured": int(accepted[: t6.STRUCTURED_TASKS].sum().item()),  # noqa: F405
            "accepted_random": int(accepted[t6.STRUCTURED_TASKS :].sum().item()),  # noqa: F405
        },
        "integrity": {
            "full_source_sha256": sha256_file(FULL_SOURCE),  # noqa: F405
            "pilot_source_sha256": sha256_file(PILOT_SOURCE),  # noqa: F405
            "test_sha256": sha256_file(TEST_SOURCE),  # noqa: F405
            "preregistration_sha256": sha256_file(PREREGISTRATION),  # noqa: F405
            "pilot_artifact_sha256": sha256_file(PILOT_OUTPUT),  # noqa: F405
            "predecessor_sha256": sha256_file(PREDECESSOR),  # noqa: F405
            "train_sha256": sha256_file(t6.t3.TRAIN_FILE),  # noqa: F405
            "validation_sha256": sha256_file(t6.t3.VALIDATION_FILE),  # noqa: F405
        },
        "pilot_selection": {
            "selected_arm": pilot_artifact["selected_arm"],
            "selected_muon_learning_rate": pilot_artifact[
                "selected_muon_learning_rate"
            ],
            "terminal_nll": pilot_artifact["terminal_nll"],
        },
        "worlds": worlds,
        "gates": gates,
        "all_gates_pass": bool(gates) and all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)  # noqa: F405
    arguments = parser.parse_args()
    result = run(torch.device(arguments.device), quick=arguments.quick)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "all_gates_pass": result["all_gates_pass"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
