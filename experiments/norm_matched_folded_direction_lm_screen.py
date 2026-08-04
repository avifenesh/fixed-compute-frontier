#!/usr/bin/env python3
"""Training-only low-rank update rotation with exact AdamW norm matching."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
import transformers

from experiments.folded_metric_lm_screen import (
    FACTOR_SEED, RANK, SCALE, FoldedMetricController, adam_tensor_step,
    build_plain_model, causal_loss, configure_matmul_precision,
    factor_learning_rate, make_optimizer, reload_check, set_schedule,
    targeted_weights,
)
from experiments.reflex_swiglu_lm_screen import (
    TokenFile, evaluate, lr_multiplier, paired_loss_interval, sha256_file,
    validate_data_ledger, write_payload,
)


ARMS = ("fixed_direction", "learned_direction")
PREDECESSOR = Path("results/folded-metric-lm-screen.json")
PREDECESSOR_SHA256 = "1994f3fe210017c01f6f363fcb5f83d4cd363d5adb2b4f790df91036ac7945dc"
PREREGISTRATION = Path("results/norm-matched-folded-direction-preregistration.md")
DIAGNOSTIC_STEPS = {1, 100, 305}


def norm_matched_rotation(base_delta: torch.Tensor, factor_delta: torch.Tensor):
    base = base_delta.float()
    factor = factor_delta.float()
    base_norm = torch.linalg.vector_norm(base)
    base_squared = torch.sum(base * base)
    if float(base_squared) == 0.0:
        return base.clone(), factor.clone(), base.new_zeros(())
    projection = torch.sum(factor * base) / base_squared
    perpendicular = factor - projection * base
    raw = base + perpendicular
    raw_norm = torch.linalg.vector_norm(raw)
    rotated = raw * (base_norm / raw_norm.clamp_min(torch.finfo(raw.dtype).tiny))
    return rotated, perpendicular, projection


class NormMatchedFoldedController(FoldedMetricController):
    """Use factor state only to rotate each ordinary AdamW matrix update."""

    def prepare(self, record_diagnostics: bool) -> None:
        if self.prepared is not None:
            raise RuntimeError("controller step already prepared")
        prepared = {}
        for name, parameter in self.targets.items():
            if parameter.grad is None:
                raise RuntimeError(f"missing gradient for {name}")
            state = self.states[name]
            gradient = parameter.grad.detach().float()
            a, b = state["a"], state["b"]
            record = {
                "old_product": a @ b,
                "grad_b": self.scale * (a.T @ gradient),
                "weight_before": parameter.detach().clone(),
            }
            if self.learn_a:
                record["grad_a"] = self.scale * (gradient @ b.T)
            prepared[name] = record
        self.prepared = prepared

    @torch.no_grad()
    def finish(self, learning_rate: float, record_diagnostics: bool):
        if self.prepared is None:
            raise RuntimeError("controller step was not prepared")
        self.step_number += 1
        records = []
        maximum_relative_norm_error = 0.0
        maximum_orthogonality_cosine = 0.0
        for name, parameter in self.targets.items():
            state = self.states[name]
            prepared = self.prepared[name]
            a, b = state["a"], state["b"]
            if self.learn_a:
                adam_tensor_step(
                    a, prepared["grad_a"], state["m_a"], state["v_a"],
                    self.step_number, learning_rate,
                )
            adam_tensor_step(
                b, prepared["grad_b"], state["m_b"], state["v_b"],
                self.step_number, learning_rate,
            )
            factor_delta = self.scale * (a @ b - prepared["old_product"])
            base_delta = parameter.detach().float() - prepared["weight_before"].float()
            rotated, perpendicular, removed_projection = norm_matched_rotation(
                base_delta, factor_delta
            )
            base_norm = float(torch.linalg.vector_norm(base_delta))
            rotated_norm = float(torch.linalg.vector_norm(rotated))
            perpendicular_norm = float(torch.linalg.vector_norm(perpendicular))
            relative_norm_error = abs(rotated_norm - base_norm) / max(base_norm, 1e-30)
            orthogonality_cosine = abs(
                float(torch.sum(base_delta * perpendicular))
            ) / max(base_norm * perpendicular_norm, 1e-30)
            maximum_relative_norm_error = max(maximum_relative_norm_error, relative_norm_error)
            maximum_orthogonality_cosine = max(maximum_orthogonality_cosine, orthogonality_cosine)
            parameter.copy_((prepared["weight_before"].float() + rotated).to(parameter.dtype))
            if record_diagnostics:
                factor_norm = float(torch.linalg.vector_norm(factor_delta))
                rotation_cosine = float(torch.sum(base_delta * rotated)) / max(
                    base_norm * rotated_norm, 1e-30
                )
                records.append({
                    "name": name,
                    "base_update_norm": base_norm,
                    "rotated_update_norm": rotated_norm,
                    "relative_norm_error": relative_norm_error,
                    "raw_factor_to_base_norm": factor_norm / max(base_norm, 1e-30),
                    "perpendicular_to_base_norm": perpendicular_norm / max(base_norm, 1e-30),
                    "perpendicular_base_abs_cosine": orthogonality_cosine,
                    "rotation_cosine": rotation_cosine,
                    "removed_projection_scalar": float(removed_projection),
                    "a_change_norm": float(torch.linalg.vector_norm(a - state["a_initial"])),
                    "b_norm": float(torch.linalg.vector_norm(b)),
                })
        self.prepared = None
        summary = {
            "maximum_relative_norm_error": maximum_relative_norm_error,
            "maximum_orthogonality_cosine": maximum_orthogonality_cosine,
        }
        if record_diagnostics:
            summary.update({
                "per_matrix": records,
                "median_relative_norm_error": float(np.median([r["relative_norm_error"] for r in records])),
                "median_perpendicular_to_base_norm": float(np.median([r["perpendicular_to_base_norm"] for r in records])),
                "median_rotation_cosine": float(np.median([r["rotation_cosine"] for r in records])),
                "median_a_change_norm": float(np.median([r["a_change_norm"] for r in records])),
            })
        return summary


def train_arm(arm, args, train_file, validation_file, device):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = build_plain_model(device)
    targets = targeted_weights(model)
    controller = NormMatchedFoldedController(
        targets, learn_a=arm == "learned_direction"
    )
    optimizer = make_optimizer(
        model, targets, args.learning_rate, args.weight_decay
    )
    evaluations = {"0": evaluate(model, validation_file, args.eval_batches, args.eval_batch_size, device)}
    optimizer.zero_grad(set_to_none=True)
    step_losses, step_seconds, gradient_norms = [], [], []
    diagnostics = {}
    max_norm_error = 0.0
    max_orthogonality_cosine = 0.0
    for step in range(args.steps):
        model.train()
        set_schedule(optimizer, lr_multiplier(step, args.steps, args.warmup_steps))
        started = time.perf_counter()
        accumulated = 0.0
        for micro in range(args.gradient_accumulation):
            inputs, labels = train_file.batch(
                step * args.gradient_accumulation + micro,
                args.micro_batch_size,
                device,
            )
            loss = causal_loss(model, inputs, labels)
            (loss / args.gradient_accumulation).backward()
            accumulated += float(loss.detach()) / args.gradient_accumulation
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip))
        if not math.isfinite(accumulated) or not math.isfinite(gradient_norm):
            raise RuntimeError(f"nonfinite {arm} step {step + 1}")
        record = step + 1 in DIAGNOSTIC_STEPS
        controller.prepare(record)
        optimizer.step()
        summary = controller.finish(factor_learning_rate(optimizer), record)
        max_norm_error = max(max_norm_error, summary["maximum_relative_norm_error"])
        max_orthogonality_cosine = max(
            max_orthogonality_cosine, summary["maximum_orthogonality_cosine"]
        )
        if record:
            diagnostics[str(step + 1)] = summary
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step_losses.append(accumulated)
        gradient_norms.append(gradient_norm)
        step_seconds.append(time.perf_counter() - started)
    evaluations[str(args.steps)] = evaluate(
        model, validation_file, args.eval_batches, args.eval_batch_size, device
    )
    served = reload_check(model, validation_file, device)
    result = {
        "arm": arm,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "optimizer_factor_scalars": controller.factor_scalars,
        "evaluations": evaluations,
        "direction_diagnostics": diagnostics,
        "maximum_relative_update_norm_error_all_steps": max_norm_error,
        "maximum_orthogonality_cosine_all_steps": max_orthogonality_cosine,
        "served_reload_check": served,
        "train": {
            "steps": args.steps,
            "prediction_tokens": args.steps * args.gradient_accumulation * args.micro_batch_size * args.sequence_length,
            "mean_loss": float(np.mean(step_losses)),
            "final_loss": step_losses[-1],
            "max_loss": max(step_losses),
            "max_preclip_gradient_norm": max(gradient_norms),
            "elapsed_seconds": sum(step_seconds),
            "tokens_per_second": args.steps * args.gradient_accumulation * args.micro_batch_size * args.sequence_length / sum(step_seconds),
            "nonfinite": False,
        },
    }
    del optimizer, controller, targets, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def decide(results, predecessor, protocol_valid):
    if set(results) != set(ARMS):
        return {"complete": False, "missing": sorted(set(ARMS) - set(results))}
    step = "305"
    baseline = predecessor["arms"]["baseline"]
    learned = results["learned_direction"]
    fixed = results["fixed_direction"]
    losses = {
        "baseline": baseline["evaluations"][step]["loss"],
        "fixed_direction": fixed["evaluations"][step]["loss"],
        "learned_direction": learned["evaluations"][step]["loss"],
        "prior_unnormalized_learned": predecessor["arms"]["learned_metric"]["evaluations"][step]["loss"],
    }
    intervals = {
        "learned_minus_baseline": paired_loss_interval(learned, baseline, step),
        "learned_minus_fixed": paired_loss_interval(learned, fixed, step),
    }
    gain_baseline = (losses["baseline"] - losses["learned_direction"]) / losses["baseline"]
    gain_fixed = (losses["fixed_direction"] - losses["learned_direction"]) / losses["fixed_direction"]
    exact_norm = all(
        result["maximum_relative_update_norm_error_all_steps"] <= 2e-5
        and result["maximum_orthogonality_cosine_all_steps"] <= 2e-5
        for result in results.values()
    )
    plain = all(
        result["served_reload_check"]["logits_bitwise_equal"]
        and result["served_reload_check"]["loss_absolute_difference"] == 0.0
        and not result["served_reload_check"]["state_dict_contains_metric_factors"]
        for result in results.values()
    )
    gates = {
        "protocol_integrity_valid": protocol_valid,
        "all_updates_norm_matched_and_orthogonal": exact_norm,
        "finite": all(not result["train"]["nonfinite"] for result in results.values()),
        "plain_factor_free_export": plain,
        "learned_beats_baseline_by_0p1_percent": gain_baseline >= 0.001 and intervals["learned_minus_baseline"]["upper_95"] < 0,
        "learned_beats_fixed_by_0p05_percent": gain_fixed >= 0.0005 and intervals["learned_minus_fixed"]["upper_95"] < 0,
    }
    return {
        "complete": True,
        "losses": losses,
        "relative_learned_gain_vs_baseline": gain_baseline,
        "relative_learned_gain_vs_fixed": gain_fixed,
        "paired_intervals": intervals,
        "gates": gates,
        "advance_to_50m": all(gates.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=Path("data/block-algebra-scratch/train.uint16.bin"))
    parser.add_argument("--validation-file", type=Path, default=Path("data/block-algebra-scratch/validation.uint16.bin"))
    parser.add_argument("--data-manifest", type=Path, default=Path("results/block-algebra-scratch-data-manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("results/norm-matched-folded-direction-lm-screen.json"))
    parser.add_argument("--steps", type=int, default=305)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=223)
    args = parser.parse_args()
    expected = (
        args.steps, args.sequence_length, args.micro_batch_size,
        args.gradient_accumulation, args.eval_batch_size, args.eval_batches,
        args.warmup_steps, args.learning_rate, args.weight_decay,
        args.gradient_clip, args.seed,
    ) == (305, 512, 32, 2, 32, 128, 100, 3e-4, 0.1, 1.0, 223)
    expected = (
        expected
        and torch.__version__ == "2.5.1+cu124"
        and torch.version.cuda == "12.4"
        and transformers.__version__ == "4.57.6"
        and "H100" in torch.cuda.get_device_name()
    )
    if sha256_file(PREDECESSOR) != PREDECESSOR_SHA256:
        raise RuntimeError("frozen predecessor changed")
    predecessor = json.loads(PREDECESSOR.read_text())
    data_ledger = validate_data_ledger(
        args.data_manifest, args.train_file, args.validation_file, args.sequence_length
    )
    protocol_valid = expected and data_ledger["valid"]
    configure_matmul_precision()
    train_file = TokenFile(args.train_file, args.sequence_length)
    validation_file = TokenFile(args.validation_file, args.sequence_length)
    device = torch.device("cuda")
    payload = {
        "schema": "norm-matched-folded-direction-v1",
        "source_sha256": sha256_file(Path(__file__)),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "predecessor_sha256": sha256_file(PREDECESSOR),
        "protocol_valid": protocol_valid,
        "data_ledger": data_ledger,
        "arms": {},
    }
    for arm in ARMS:
        print(json.dumps({"starting_arm": arm}), flush=True)
        payload["arms"][arm] = train_arm(
            arm, args, train_file, validation_file, device
        )
        payload["decision"] = decide(payload["arms"], predecessor, protocol_valid)
        write_payload(args.output, payload)
        print(json.dumps({"finished_arm": arm, "loss": payload["arms"][arm]["evaluations"]["305"]["loss"]}), flush=True)
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

