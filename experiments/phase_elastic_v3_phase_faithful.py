#!/usr/bin/env python3
"""Phase-faithful training and mixed-cache evaluation of phase-elastic FFN v2."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import transformers

from experiments.coalesced_attention_ffn_lm_screen import build_model as build_parallel_model
from experiments.phase_elastic_residual_precision_v2_lm_screen import (
    PhaseElasticV2MLP,
    serving_ledger,
)
from experiments.phase_elastic_v2_50m_replication import configure_optimizer, seed_everything
from experiments.reflex_swiglu_lm_screen import (
    TokenFile,
    evaluate,
    lr_multiplier,
    paired_loss_interval,
    sha256_file,
    validate_data_ledger,
    write_payload,
)
from experiments.triangular_microdepth_lm_screen import causal_loss


SEED = 6673
STEPS = 305
TRAIN_FIRST_FULL = (64, 128, 256, 384, 448, 480, 496)
EVAL_FIRST_FULL = (256, 384, 448, 480, 496)
PREREGISTRATION = Path("results/phase-elastic-v3-phase-faithful-preregistration.md")
HOMOGENEOUS_RESULT = Path("results/phase-elastic-v2-50m-replication.json")
HOMOGENEOUS_RESULT_SHA256 = "49498886341fdff3f098f4086c330d1cbf59fd8b0022575ee3ba29dcf0e5cb1c"


def phase_mask(
    first_full: torch.Tensor, sequence_length: int, device: torch.device
) -> torch.Tensor:
    positions = torch.arange(sequence_length, device=device).view(1, sequence_length)
    return (positions >= first_full.to(device=device).view(-1, 1)).unsqueeze(-1)


class PhaseFaithfulV2MLP(PhaseElasticV2MLP):
    """The v2 MLP with one tokenwise phase mask shared by every layer."""

    def __init__(self, original: torch.nn.Module, std: float) -> None:
        super().__init__(original, std)
        self.mode = "split"
        self.phase_mask: torch.Tensor | None = None

    def set_mode(self, mode: str) -> None:
        if mode not in {"base", "full", "split"}:
            raise ValueError(mode)
        self.mode = mode

    def set_phase_mask(self, mask: torch.Tensor | None) -> None:
        self.phase_mask = mask

    def _mask_for(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.phase_mask is None:
            raise RuntimeError("split mode requires a phase mask")
        expected = (*hidden_states.shape[:2], 1)
        if tuple(self.phase_mask.shape) != expected:
            raise RuntimeError(
                f"phase mask shape {tuple(self.phase_mask.shape)} does not match {expected}"
            )
        return self.phase_mask.to(device=hidden_states.device, dtype=hidden_states.dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base_activation = F.silu(F.linear(hidden_states, self.base_gate())) * F.linear(
            hidden_states, self.base_up()
        )
        base = F.linear(base_activation, self.base_down())
        if self.mode == "base":
            self.last_branch_active = False
            return base
        branch_activation = F.silu(F.linear(hidden_states, self.branch_gate())) * F.linear(
            hidden_states, self.branch_up()
        )
        branch = F.linear(branch_activation, self.branch_down())
        self.last_branch_active = True
        if self.mode == "full":
            return base + branch
        return base + branch * self._mask_for(hidden_states)


def build_candidate(device: torch.device, seed: int):
    seed_everything(seed)
    model, _ = build_parallel_model(device, "parallel_baseline")
    modules = []
    for layer in model.model.layers:
        replacement = PhaseFaithfulV2MLP(
            layer.mlp, model.config.initializer_range
        ).to(device)
        layer.mlp = replacement
        modules.append(replacement)
    model.config.use_cache = False
    return model, modules


def set_mode(modules: list[PhaseFaithfulV2MLP], mode: str) -> None:
    for module in modules:
        module.set_mode(mode)


def set_phase_mask(modules: list[PhaseFaithfulV2MLP], mask: torch.Tensor | None) -> None:
    for module in modules:
        module.set_phase_mask(mask)


def token_losses(model, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=inputs, use_cache=False).logits
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).view_as(targets)


def bucket_slices(first_full: int, sequence_length: int) -> dict[str, slice]:
    return {
        "all_suffix": slice(first_full, sequence_length),
        "first_token": slice(first_full, min(first_full + 1, sequence_length)),
        "offsets_2_8": slice(first_full + 1, min(first_full + 8, sequence_length)),
        "offsets_9_32": slice(first_full + 8, min(first_full + 32, sequence_length)),
        "offsets_33_plus": slice(first_full + 32, sequence_length),
    }


@torch.no_grad()
def evaluate_phase_grid(
    model,
    modules: list[PhaseFaithfulV2MLP] | None,
    validation: TokenFile,
    boundaries: tuple[int, ...],
    batches: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    result: dict[str, Any] = {}
    for first_full in boundaries:
        per_bucket: dict[str, list[float]] = {
            name: []
            for name, region in bucket_slices(first_full, validation.sequence_length).items()
            if region.start < region.stop
        }
        started = time.perf_counter()
        for batch_index in range(batches):
            inputs, targets = validation.batch(batch_index, batch_size, device)
            if modules is not None:
                mask = phase_mask(
                    torch.full((batch_size,), first_full, device=device),
                    validation.sequence_length,
                    device,
                )
                set_mode(modules, "split")
                set_phase_mask(modules, mask)
            losses = token_losses(model, inputs, targets)
            for name, region in bucket_slices(first_full, validation.sequence_length).items():
                if region.start < region.stop:
                    per_bucket[name].append(float(losses[:, region].mean()))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        result[str(first_full)] = {
            name: {
                "loss": float(np.mean(values)),
                "per_batch_loss": values,
                "prediction_tokens": batches * batch_size * (
                    bucket_slices(first_full, validation.sequence_length)[name].stop
                    - bucket_slices(first_full, validation.sequence_length)[name].start
                ),
                "elapsed_seconds": elapsed,
            }
            for name, values in per_bucket.items()
        }
    if modules is not None:
        set_phase_mask(modules, None)
    return result


def aggregate_bucket(grid: dict[str, Any], bucket: str) -> dict[str, Any]:
    available = [grid[str(boundary)][bucket] for boundary in EVAL_FIRST_FULL if bucket in grid[str(boundary)]]
    per_batch = np.asarray([entry["per_batch_loss"] for entry in available], dtype=np.float64).mean(axis=0)
    return {
        "loss": float(per_batch.mean()),
        "per_batch_loss": per_batch.tolist(),
        "prediction_tokens": int(sum(entry["prediction_tokens"] for entry in available)),
    }


def paired_noninferior(
    candidate_loss: float,
    reference_loss: float,
    interval: dict[str, float],
    relative_margin: float,
) -> bool:
    relative_regret = (candidate_loss - reference_loss) / reference_loss
    return (
        relative_regret <= relative_margin
        and interval["upper_95"] <= relative_margin * reference_loss
    )


def evaluate_endpoint(model, modules, mode, validation, args, device):
    set_phase_mask(modules, None)
    set_mode(modules, mode)
    return evaluate(model, validation, args.eval_batches, args.eval_batch_size, device)


def cache_tensors(cache) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if hasattr(cache, "layers"):
        return [(layer.keys, layer.values) for layer in cache.layers]
    return [cache[index] for index in range(len(cache))]


@torch.no_grad()
def _cached_comparison(
    model,
    modules: list[PhaseFaithfulV2MLP],
    inputs: torch.Tensor,
    device: torch.device,
    first_full: int,
    stop: int,
    use_bf16: bool,
) -> dict[str, Any]:
    prefix_and_suffix = inputs[:, :stop]
    mask = phase_mask(torch.tensor([first_full], device=device), stop, device)
    set_mode(modules, "split")
    set_phase_mask(modules, mask)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        parallel = model(input_ids=prefix_and_suffix, use_cache=True)
    parallel_logits = parallel.logits[:, first_full:stop].float()
    parallel_cache = cache_tensors(parallel.past_key_values)

    set_phase_mask(modules, None)
    set_mode(modules, "base")
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        prefill = model(input_ids=inputs[:, :first_full], use_cache=True)
    actual_cache = prefill.past_key_values
    incremental_logits = []
    set_mode(modules, "full")
    for position in range(first_full, stop):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            decoded = model(
                input_ids=inputs[:, position : position + 1],
                past_key_values=actual_cache,
                use_cache=True,
            )
        actual_cache = decoded.past_key_values
        incremental_logits.append(decoded.logits[:, 0].float())
    actual_logits = torch.stack(incremental_logits, dim=1)
    actual_cache_tensors = cache_tensors(actual_cache)
    logit_difference = float((parallel_logits - actual_logits).abs().max())
    key_difference = max(
        float((parallel_key[..., :stop, :] - actual_key[..., :stop, :]).float().abs().max())
        for (parallel_key, _), (actual_key, _) in zip(parallel_cache, actual_cache_tensors)
    )
    value_difference = max(
        float((parallel_value[..., :stop, :] - actual_value[..., :stop, :]).float().abs().max())
        for (_, parallel_value), (_, actual_value) in zip(parallel_cache, actual_cache_tensors)
    )
    return {
        "comparison_dtype": "bfloat16" if use_bf16 else "float32_tf32_disabled",
        "max_abs_logit_difference": logit_difference,
        "max_abs_key_difference": key_difference,
        "max_abs_value_difference": value_difference,
    }


@torch.no_grad()
def verify_cached_equivalence(
    model,
    modules: list[PhaseFaithfulV2MLP],
    validation: TokenFile,
    device: torch.device,
    first_full: int = 480,
    suffix_tokens: int = 8,
) -> dict[str, Any]:
    model.eval()
    inputs, _ = validation.batch(0, 1, device)
    stop = min(inputs.shape[1], first_full + suffix_tokens)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        float32 = _cached_comparison(
            model, modules, inputs, device, first_full, stop, use_bf16=False
        )
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        bfloat16 = _cached_comparison(
            model, modules, inputs, device, first_full, stop, use_bf16=True
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        set_phase_mask(modules, None)
        set_mode(modules, "split")
    return {
        "zero_based_first_full_position": first_full,
        "base_prefill_tokens": first_full,
        "incremental_full_tokens": stop - first_full,
        "float32": float32,
        "bfloat16": bfloat16,
    }


def train_dense(args, train_file, device):
    seed_everything(args.seed)
    model, _ = build_parallel_model(device, "parallel_baseline")
    model.config.use_cache = False
    optimizer = configure_optimizer(model, args, False)
    optimizer.zero_grad(set_to_none=True)
    losses, durations, norms = [], [], []
    for step in range(args.steps):
        model.train()
        scheduled = args.learning_rate * lr_multiplier(step, args.steps, args.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = scheduled
        started = time.perf_counter()
        accumulated = 0.0
        for micro in range(args.gradient_accumulation):
            inputs, targets = train_file.batch(
                step * args.gradient_accumulation + micro, args.micro_batch_size, device
            )
            loss = causal_loss(model, inputs, targets)
            (loss / args.gradient_accumulation).backward()
            accumulated += float(loss.detach()) / args.gradient_accumulation
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip))
        if not math.isfinite(accumulated) or not math.isfinite(norm):
            raise RuntimeError(f"nonfinite dense step {step + 1}")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        losses.append(accumulated)
        norms.append(norm)
        durations.append(time.perf_counter() - started)
    train = {
        "prediction_tokens_loaded": args.steps * args.gradient_accumulation * args.micro_batch_size * args.sequence_length,
        "mean_loss": float(np.mean(losses)),
        "final_loss": losses[-1],
        "max_preclip_gradient_norm": max(norms),
        "elapsed_seconds": sum(durations),
        "nonfinite": False,
    }
    return model, train


def train_candidate(args, train_file, device):
    model, modules = build_candidate(device, args.seed)
    optimizer = configure_optimizer(model, args, True)
    rng = np.random.default_rng(args.seed + 1)
    boundary_counts = {str(boundary): 0 for boundary in TRAIN_FIRST_FULL}
    optimizer.zero_grad(set_to_none=True)
    losses, durations, norms = [], [], []
    for step in range(args.steps):
        model.train()
        set_mode(modules, "split")
        scheduled = args.learning_rate * lr_multiplier(step, args.steps, args.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = scheduled
        started = time.perf_counter()
        accumulated = 0.0
        for micro in range(args.gradient_accumulation):
            inputs, targets = train_file.batch(
                step * args.gradient_accumulation + micro, args.micro_batch_size, device
            )
            sampled = rng.choice(TRAIN_FIRST_FULL, size=args.micro_batch_size, replace=True)
            for boundary in sampled:
                boundary_counts[str(int(boundary))] += 1
            mask = phase_mask(torch.from_numpy(sampled), args.sequence_length, device)
            set_phase_mask(modules, mask)
            loss = causal_loss(model, inputs, targets)
            (loss / args.gradient_accumulation).backward()
            accumulated += float(loss.detach()) / args.gradient_accumulation
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip))
        if not math.isfinite(accumulated) or not math.isfinite(norm):
            raise RuntimeError(f"nonfinite candidate step {step + 1}")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        losses.append(accumulated)
        norms.append(norm)
        durations.append(time.perf_counter() - started)
    set_phase_mask(modules, None)
    train = {
        "prediction_tokens_loaded": args.steps * args.gradient_accumulation * args.micro_batch_size * args.sequence_length,
        "mean_loss": float(np.mean(losses)),
        "final_loss": losses[-1],
        "max_preclip_gradient_norm": max(norms),
        "elapsed_seconds": sum(durations),
        "boundary_counts": boundary_counts,
        "nonfinite": False,
    }
    return model, modules, train


def decide(dense, candidate, protocol_valid: bool) -> dict[str, Any]:
    dense_suffix = aggregate_bucket(dense["phase_grid"], "all_suffix")
    candidate_suffix = aggregate_bucket(candidate["phase_grid"], "all_suffix")
    dense_first = aggregate_bucket(dense["phase_grid"], "first_token")
    candidate_first = aggregate_bucket(candidate["phase_grid"], "first_token")
    suffix_interval = paired_loss_interval(
        {"evaluations": {"terminal": candidate_suffix}},
        {"evaluations": {"terminal": dense_suffix}},
        "terminal",
    )
    first_interval = paired_loss_interval(
        {"evaluations": {"terminal": candidate_first}},
        {"evaluations": {"terminal": dense_first}},
        "terminal",
    )
    suffix_gain = (dense_suffix["loss"] - candidate_suffix["loss"]) / dense_suffix["loss"]
    first_regret = (candidate_first["loss"] - dense_first["loss"]) / dense_first["loss"]
    boundary_regrets = {
        str(boundary): (
            candidate["phase_grid"][str(boundary)]["all_suffix"]["loss"]
            - dense["phase_grid"][str(boundary)]["all_suffix"]["loss"]
        ) / dense["phase_grid"][str(boundary)]["all_suffix"]["loss"]
        for boundary in EVAL_FIRST_FULL
    }
    semantic = candidate["cached_equivalence"]
    audits = [entry for layer in candidate["quantized_weight_audits"] for entry in layer.values()]
    gates = {
        "protocol_integrity_valid": protocol_valid,
        "exact_storage_fits": bool(serving_ledger()["fits_bf16_bytes"]),
        "all_codes_scales_positive_finite_and_bounded": all(
            audit["all_finite"]
            and audit["base_code_min"] >= -127
            and audit["base_code_max"] <= 127
            and set(audit["delta_code_values"]) <= {-1, 0, 1}
            and audit["base_scale_min"] > 0
            and audit["delta_scale_min"] > 0
            for audit in audits
        ),
        "training_finite": not dense["train"]["nonfinite"] and not candidate["train"]["nonfinite"],
        "all_phase_boundaries_exercised": all(
            count > 0 for count in candidate["train"]["boundary_counts"].values()
        ),
        "float32_cached_algebra_matches": max(
            semantic["float32"]["max_abs_logit_difference"],
            semantic["float32"]["max_abs_key_difference"],
            semantic["float32"]["max_abs_value_difference"],
        ) <= 3e-5,
        "bfloat16_cached_logits_match_calibrated_tolerance": (
            semantic["bfloat16"]["max_abs_logit_difference"] <= 0.125
        ),
        "bfloat16_cached_keys_values_match_calibrated_tolerance": max(
            semantic["bfloat16"]["max_abs_key_difference"],
            semantic["bfloat16"]["max_abs_value_difference"],
        ) <= 0.1,
        "mixed_suffix_beats_dense_by_0p1_percent": suffix_gain >= 0.001 and suffix_interval["upper_95"] < 0,
        "first_token_noninferior_within_0p05_percent": paired_noninferior(
            candidate_first["loss"],
            dense_first["loss"],
            first_interval,
            0.0005,
        ),
        "no_boundary_worse_than_dense_by_0p2_percent": max(boundary_regrets.values()) <= 0.002,
    }
    return {
        "aggregate_mixed_suffix": {
            "dense_loss": dense_suffix["loss"],
            "candidate_loss": candidate_suffix["loss"],
            "relative_candidate_gain": suffix_gain,
            "paired_interval": suffix_interval,
        },
        "aggregate_first_token": {
            "dense_loss": dense_first["loss"],
            "candidate_loss": candidate_first["loss"],
            "relative_candidate_regret": first_regret,
            "paired_interval": first_interval,
        },
        "relative_boundary_regrets": boundary_regrets,
        "gates": gates,
        "advance_to_long_horizon_multiseed": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=Path("data/block-algebra-scratch/train.uint16.bin"))
    parser.add_argument("--validation-file", type=Path, default=Path("data/block-algebra-scratch/validation.uint16.bin"))
    parser.add_argument("--data-manifest", type=Path, default=Path("results/block-algebra-scratch-data-manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("results/phase-elastic-v3-phase-faithful.json"))
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    expected = (
        args.steps,
        args.sequence_length,
        args.micro_batch_size,
        args.gradient_accumulation,
        args.eval_batch_size,
        args.eval_batches,
        args.warmup_steps,
        args.learning_rate,
        args.weight_decay,
        args.gradient_clip,
        args.seed,
    ) == (305, 512, 32, 2, 32, 64, 30, 3e-4, 0.1, 1.0, SEED)
    expected = (
        expected
        and torch.__version__ == "2.5.1+cu124"
        and torch.version.cuda == "12.4"
        and transformers.__version__ == "4.57.6"
        and "H100" in torch.cuda.get_device_name()
    )
    if sha256_file(HOMOGENEOUS_RESULT) != HOMOGENEOUS_RESULT_SHA256:
        raise RuntimeError("frozen homogeneous control changed")
    predecessor = json.loads(HOMOGENEOUS_RESULT.read_text())
    data = validate_data_ledger(
        args.data_manifest, args.train_file, args.validation_file, args.sequence_length
    )
    protocol_valid = (
        expected
        and predecessor["decision"]["advance_to_physical_h100_gate"]
        and data["valid"]
    )
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    train_file = TokenFile(args.train_file, args.sequence_length)
    validation = TokenFile(args.validation_file, args.sequence_length)

    print(json.dumps({"starting_arm": "dense_bf16"}), flush=True)
    dense_model, dense_train = train_dense(args, train_file, device)
    dense_grid = evaluate_phase_grid(
        dense_model, None, validation, EVAL_FIRST_FULL, args.eval_batches, args.eval_batch_size, device
    )
    dense = {"train": dense_train, "phase_grid": dense_grid}
    del dense_model
    gc.collect()
    torch.cuda.empty_cache()

    print(json.dumps({"starting_arm": "phase_faithful_v3"}), flush=True)
    candidate_model, modules, candidate_train = train_candidate(args, train_file, device)
    candidate_grid = evaluate_phase_grid(
        candidate_model, modules, validation, EVAL_FIRST_FULL, args.eval_batches, args.eval_batch_size, device
    )
    candidate = {
        "train": candidate_train,
        "phase_grid": candidate_grid,
        "base_endpoint": evaluate_endpoint(candidate_model, modules, "base", validation, args, device),
        "full_endpoint": evaluate_endpoint(candidate_model, modules, "full", validation, args, device),
        "cached_equivalence": verify_cached_equivalence(candidate_model, modules, validation, device),
        "quantized_weight_audits": [module.audit() for module in modules],
        "training_shadow_parameters": sum(parameter.numel() for parameter in candidate_model.parameters()),
    }
    decision = decide(dense, candidate, protocol_valid)
    payload = {
        "schema": "phase-elastic-v3-phase-faithful-v1",
        "source_sha256": sha256_file(Path(__file__)),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "homogeneous_result_sha256": sha256_file(HOMOGENEOUS_RESULT),
        "protocol_valid": protocol_valid,
        "data_ledger": data,
        "serving_ledger": serving_ledger(),
        "dense": dense,
        "candidate": candidate,
        "decision": decision,
    }
    write_payload(args.output, payload)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
