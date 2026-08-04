#!/usr/bin/env python3
"""Fresh-seed 50M packed replication of phase-faithful phase-elastic FFNs."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers

from experiments.phase_elastic_packed_artifact import (
    freeze_phase_elastic_modules,
    shadow_names,
)
from experiments.phase_elastic_residual_precision_v2_lm_screen import serving_ledger
from experiments.phase_elastic_v3_phase_faithful import (
    EVAL_FIRST_FULL,
    aggregate_bucket,
    build_candidate,
    bucket_slices,
    evaluate_endpoint,
    evaluate_phase_grid,
    paired_noninferior,
    set_mode,
    set_phase_mask,
    train_candidate,
    train_dense,
    verify_cached_equivalence,
)
from experiments.phase_elastic_v3_packed_replay import grid_max_loss_difference
from experiments.reflex_swiglu_lm_screen import (
    TokenFile,
    evaluate,
    paired_loss_interval,
    sha256_file,
    validate_data_ledger,
    write_payload,
)


SEED = 9901
STEPS = 1525
PREREGISTRATION = Path("results/phase-elastic-v3-50m-phase-faithful-preregistration.md")
DISCOVERY = Path("results/phase-elastic-v3-phase-faithful.json")
DISCOVERY_SHA256 = "e59e2ef52cfe252a4686f414e3220dafc0700477f56dab04e1a505327cb7acf5"
PACKED_REPLAY = Path("results/phase-elastic-v3-packed-replay.json")
PACKED_REPLAY_SHA256 = "aed3832515ec40559e228bd061d7f9aea28663c311958838421d463b17f667f9"
CACHED_BOUNDARIES = (256, 480)
CACHED_BATCHES = 16
CACHED_BATCH_SIZE = 8
INTEGRITY = Path("results/phase-elastic-v3-50m-phase-faithful-integrity.json")


@torch.no_grad()
def evaluate_cached_suffix(
    model,
    modules,
    validation: TokenFile,
    first_full: int,
    batches: int,
    batch_size: int,
    device: torch.device,
    prefix_mode: str = "base",
    suffix_mode: str = "full",
):
    model.eval()
    regions = {
        name: region
        for name, region in bucket_slices(first_full, validation.sequence_length).items()
        if region.start < region.stop
    }
    observations = {name: [] for name in regions}
    for batch_index in range(batches):
        inputs, targets = validation.batch(batch_index, batch_size, device)
        if modules is not None:
            set_phase_mask(modules, None)
            set_mode(modules, prefix_mode)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prefill = model(input_ids=inputs[:, :first_full], use_cache=True)
        cache = prefill.past_key_values
        if modules is not None:
            set_mode(modules, suffix_mode)
        token_losses = []
        for position in range(first_full, validation.sequence_length):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                decoded = model(
                    input_ids=inputs[:, position : position + 1],
                    past_key_values=cache,
                    use_cache=True,
                )
            cache = decoded.past_key_values
            token_losses.append(
                F.cross_entropy(
                    decoded.logits[:, 0].float(),
                    targets[:, position],
                    reduction="none",
                )
            )
        losses = torch.stack(token_losses, dim=1)
        for name, region in regions.items():
            relative = slice(region.start - first_full, region.stop - first_full)
            observations[name].extend(
                float(value) for value in losses[:, relative].mean(dim=1)
            )
    if modules is not None:
        set_mode(modules, "split")
    result = {"first_full": first_full, "observation_unit": "sequence"}
    for name, values in observations.items():
        region = regions[name]
        result[name] = {
            "loss": sum(values) / len(values),
            "per_batch_loss": values,
            "prediction_tokens": batches * batch_size * (region.stop - region.start),
        }
    return result


def compare(candidate, dense):
    return paired_loss_interval(
        {"evaluations": {"terminal": candidate}},
        {"evaluations": {"terminal": dense}},
        "terminal",
    )


def paired_t_interval(candidate, reference):
    differences = np.asarray(candidate["per_batch_loss"], dtype=np.float64) - np.asarray(
        reference["per_batch_loss"], dtype=np.float64
    )
    if differences.size != 128:
        raise RuntimeError(f"cached interval expected 128 paired sequences, got {differences.size}")
    mean = float(differences.mean())
    standard_error = float(differences.std(ddof=1) / math.sqrt(differences.size))
    critical = 1.978819534
    return {
        "candidate_minus_reference_mean": mean,
        "standard_error": standard_error,
        "degrees_of_freedom": 127,
        "critical_95": critical,
        "lower_95": mean - critical * standard_error,
        "upper_95": mean + critical * standard_error,
    }


def aggregate_cached(grids, bucket):
    observations = np.asarray(
        [grids[str(boundary)][bucket]["per_batch_loss"] for boundary in CACHED_BOUNDARIES],
        dtype=np.float64,
    ).mean(axis=0)
    return {
        "loss": float(observations.mean()),
        "per_batch_loss": observations.tolist(),
        "prediction_tokens": sum(
            grids[str(boundary)][bucket]["prediction_tokens"]
            for boundary in CACHED_BOUNDARIES
        ),
    }


def relative_grid_regrets(candidate_grid, reference_grid):
    regrets = {}
    for boundary in EVAL_FIRST_FULL:
        for bucket, candidate in candidate_grid[str(boundary)].items():
            reference = reference_grid[str(boundary)][bucket]
            regrets[f"p{boundary}_{bucket}"] = (
                candidate["loss"] - reference["loss"]
            ) / reference["loss"]
    return regrets


def relative_cached_regrets(candidate_grids, reference_grids):
    regrets = {}
    for boundary in CACHED_BOUNDARIES:
        for bucket, candidate in candidate_grids[str(boundary)].items():
            if not isinstance(candidate, dict) or "loss" not in candidate:
                continue
            reference = reference_grids[str(boundary)][bucket]
            regrets[f"p{boundary}_{bucket}"] = (
                candidate["loss"] - reference["loss"]
            ) / reference["loss"]
    return regrets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=Path("data/block-algebra-scratch/train.uint16.bin"))
    parser.add_argument("--validation-file", type=Path, default=Path("data/block-algebra-scratch/validation.uint16.bin"))
    parser.add_argument("--data-manifest", type=Path, default=Path("results/block-algebra-scratch-data-manifest.json"))
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/phase-elastic-v3-50m-packed.pt"))
    parser.add_argument("--output", type=Path, default=Path("results/phase-elastic-v3-50m-phase-faithful.json"))
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    expected = (
        args.steps, args.sequence_length, args.micro_batch_size,
        args.gradient_accumulation, args.eval_batch_size, args.eval_batches,
        args.warmup_steps, args.learning_rate, args.weight_decay,
        args.gradient_clip, args.seed,
    ) == (1525, 512, 32, 2, 32, 128, 100, 3e-4, 0.1, 1.0, SEED)
    expected = (
        expected
        and torch.__version__ == "2.5.1+cu124"
        and torch.version.cuda == "12.4"
        and transformers.__version__ == "4.57.6"
        and "H100" in torch.cuda.get_device_name()
    )
    if sha256_file(DISCOVERY) != DISCOVERY_SHA256:
        raise RuntimeError("frozen discovery changed")
    if sha256_file(PACKED_REPLAY) != PACKED_REPLAY_SHA256:
        raise RuntimeError("frozen packed replay changed")
    integrity = json.loads(INTEGRITY.read_text())
    if sha256_file(Path(__file__)) != integrity["source_sha256"]:
        raise RuntimeError("frozen source changed")
    if sha256_file(PREREGISTRATION) != integrity["preregistration_sha256"]:
        raise RuntimeError("frozen preregistration changed")
    discovery = json.loads(DISCOVERY.read_text())
    packed_predecessor = json.loads(PACKED_REPLAY.read_text())
    predecessor_evidence = packed_predecessor["decision"]["gates"]
    data = validate_data_ledger(
        args.data_manifest, args.train_file, args.validation_file, args.sequence_length
    )
    protocol_valid = (
        expected
        and discovery["decision"]["advance_to_long_horizon_multiseed"]
        and predecessor_evidence["packed_reload_matches_pre_export_within_1e_7"]
        and predecessor_evidence["exact_payload_bytes_every_layer"]
        and predecessor_evidence["no_training_shadows_after_reload"]
        and data["valid"]
    )
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    train_file = TokenFile(args.train_file, args.sequence_length)
    validation = TokenFile(args.validation_file, args.sequence_length)

    print(json.dumps({"starting_arm": "dense_bf16_50m"}), flush=True)
    dense_model, dense_train = train_dense(args, train_file, device)
    dense_grid = evaluate_phase_grid(
        dense_model, None, validation, EVAL_FIRST_FULL,
        args.eval_batches, args.eval_batch_size, device,
    )
    dense_endpoint = evaluate(
        dense_model, validation, args.eval_batches, args.eval_batch_size, device
    )
    dense_cached = {
        str(boundary): evaluate_cached_suffix(
            dense_model,
            None,
            validation,
            boundary,
            CACHED_BATCHES,
            CACHED_BATCH_SIZE,
            device,
        )
        for boundary in CACHED_BOUNDARIES
    }
    dense = {
        "train": dense_train,
        "phase_grid": dense_grid,
        "endpoint": dense_endpoint,
        "cached": dense_cached,
    }
    del dense_model
    gc.collect()
    torch.cuda.empty_cache()

    print(json.dumps({"starting_arm": "phase_faithful_v3_50m"}), flush=True)
    candidate_model, modules, candidate_train = train_candidate(args, train_file, device)
    pre_export_grid = evaluate_phase_grid(
        candidate_model, modules, validation, EVAL_FIRST_FULL,
        args.eval_batches, args.eval_batch_size, device,
    )
    layer_payloads = freeze_phase_elastic_modules(modules)
    names_before_save = shadow_names(candidate_model)
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": candidate_model.state_dict()}, args.artifact)
    artifact_sha256 = sha256_file(args.artifact)
    artifact_bytes = args.artifact.stat().st_size
    del modules, candidate_model
    gc.collect()
    torch.cuda.empty_cache()

    print(json.dumps({"stage": "packed_reload_and_decision_eval"}), flush=True)
    packed_model, packed_modules = build_candidate(device, args.seed + 99)
    freeze_phase_elastic_modules(packed_modules)
    state = torch.load(args.artifact, map_location=device, weights_only=True)["model"]
    load_result = packed_model.load_state_dict(state, strict=True)
    names_after_reload = shadow_names(packed_model)
    packed_grid = evaluate_phase_grid(
        packed_model, packed_modules, validation, EVAL_FIRST_FULL,
        args.eval_batches, args.eval_batch_size, device,
    )
    packed_base = evaluate_endpoint(
        packed_model, packed_modules, "base", validation, args, device
    )
    packed_full = evaluate_endpoint(
        packed_model, packed_modules, "full", validation, args, device
    )
    set_phase_mask(packed_modules, None)
    set_mode(packed_modules, "base")
    packed_base_grid = evaluate_phase_grid(
        packed_model, None, validation, EVAL_FIRST_FULL,
        args.eval_batches, args.eval_batch_size, device,
    )
    packed_cached = {
        str(boundary): evaluate_cached_suffix(
            packed_model,
            packed_modules,
            validation,
            boundary,
            CACHED_BATCHES,
            CACHED_BATCH_SIZE,
            device,
            prefix_mode="base",
            suffix_mode="full",
        )
        for boundary in CACHED_BOUNDARIES
    }
    packed_base_cached = {
        str(boundary): evaluate_cached_suffix(
            packed_model,
            packed_modules,
            validation,
            boundary,
            CACHED_BATCHES,
            CACHED_BATCH_SIZE,
            device,
            prefix_mode="base",
            suffix_mode="base",
        )
        for boundary in CACHED_BOUNDARIES
    }
    cached_semantics = verify_cached_equivalence(
        packed_model, packed_modules, validation, device
    )
    weight_audits = [module.audit() for module in packed_modules]

    dense_suffix = aggregate_bucket(dense_grid, "all_suffix")
    packed_suffix = aggregate_bucket(packed_grid, "all_suffix")
    dense_first = aggregate_bucket(dense_grid, "first_token")
    packed_first = aggregate_bucket(packed_grid, "first_token")
    packed_base_suffix = aggregate_bucket(packed_base_grid, "all_suffix")
    suffix_interval = compare(packed_suffix, dense_suffix)
    first_interval = compare(packed_first, dense_first)
    suffix_gain = (dense_suffix["loss"] - packed_suffix["loss"]) / dense_suffix["loss"]
    branch_gain = (
        packed_base_suffix["loss"] - packed_suffix["loss"]
    ) / packed_base_suffix["loss"]
    branch_interval = compare(packed_suffix, packed_base_suffix)
    discovery_gain = discovery["decision"]["aggregate_mixed_suffix"]["relative_candidate_gain"]
    boundary_gains = {
        str(boundary): (
            dense_grid[str(boundary)]["all_suffix"]["loss"]
            - packed_grid[str(boundary)]["all_suffix"]["loss"]
        ) / dense_grid[str(boundary)]["all_suffix"]["loss"]
        for boundary in EVAL_FIRST_FULL
    }
    base_regret = (packed_base["loss"] - dense_endpoint["loss"]) / dense_endpoint["loss"]
    full_gain = (dense_endpoint["loss"] - packed_full["loss"]) / dense_endpoint["loss"]
    base_interval = compare(packed_base, dense_endpoint)
    full_interval = compare(packed_full, dense_endpoint)
    dense_cached_suffix = aggregate_cached(dense_cached, "all_suffix")
    packed_cached_suffix = aggregate_cached(packed_cached, "all_suffix")
    packed_base_cached_suffix = aggregate_cached(packed_base_cached, "all_suffix")
    dense_cached_first = aggregate_cached(dense_cached, "first_token")
    packed_cached_first = aggregate_cached(packed_cached, "first_token")
    cached_suffix_interval = paired_t_interval(
        packed_cached_suffix, dense_cached_suffix
    )
    cached_first_interval = paired_t_interval(
        packed_cached_first, dense_cached_first
    )
    cached_branch_interval = paired_t_interval(
        packed_cached_suffix, packed_base_cached_suffix
    )
    cached_suffix_gain = (
        dense_cached_suffix["loss"] - packed_cached_suffix["loss"]
    ) / dense_cached_suffix["loss"]
    cached_branch_gain = (
        packed_base_cached_suffix["loss"] - packed_cached_suffix["loss"]
    ) / packed_base_cached_suffix["loss"]
    onepass_dense_regrets = relative_grid_regrets(packed_grid, dense_grid)
    onepass_base_regrets = relative_grid_regrets(packed_grid, packed_base_grid)
    cached_dense_regrets = relative_cached_regrets(packed_cached, dense_cached)
    cached_base_regrets = relative_cached_regrets(
        packed_cached, packed_base_cached
    )
    packed_replay_difference = grid_max_loss_difference(
        packed_grid, pre_export_grid
    )
    base_audits = [
        layer[name] for layer in weight_audits
        for name in ("base_gate", "base_up", "base_down")
    ]
    branch_audits = [
        layer[name] for layer in weight_audits
        for name in ("branch_gate", "branch_up", "branch_down")
    ]
    fp32_cache = cached_semantics["float32"]
    bf16_cache = cached_semantics["bfloat16"]
    gates = {
        "protocol_integrity_valid": protocol_valid,
        "mixed_suffix_beats_dense_by_0p25_percent": suffix_gain >= 0.0025 and suffix_interval["upper_95"] < 0,
        "retains_half_discovery_gain_and_0p1_absolute": suffix_gain >= max(0.001, 0.5 * discovery_gain),
        "every_boundary_improves": min(boundary_gains.values()) > 0,
        "decode_branch_improves_onepass_suffix_by_0p1_percent": (
            branch_gain >= 0.001 and branch_interval["upper_95"] < 0
        ),
        "no_onepass_offset_bucket_regresses_vs_dense_by_0p1_percent": (
            max(onepass_dense_regrets.values()) <= 0.001
        ),
        "no_onepass_offset_bucket_regresses_vs_base_by_0p1_percent": (
            max(onepass_base_regrets.values()) <= 0.001
        ),
        "first_token_noninferior": paired_noninferior(
            packed_first["loss"], dense_first["loss"], first_interval, 0.0005
        ),
        "base_endpoint_noninferior": paired_noninferior(
            packed_base["loss"], dense_endpoint["loss"], base_interval, 0.0005
        ),
        "full_endpoint_beats_dense_by_0p25_percent": (
            full_gain >= 0.0025 and full_interval["upper_95"] < 0
        ),
        "actual_cached_suffix_beats_dense_by_0p1_percent": (
            cached_suffix_gain >= 0.001 and cached_suffix_interval["upper_95"] < 0
        ),
        "decode_branch_improves_actual_cached_suffix_by_0p1_percent": (
            cached_branch_gain >= 0.001 and cached_branch_interval["upper_95"] < 0
        ),
        "no_actual_cached_offset_bucket_regresses_vs_dense_by_0p1_percent": (
            max(cached_dense_regrets.values()) <= 0.001
        ),
        "no_actual_cached_offset_bucket_regresses_vs_base_by_0p1_percent": (
            max(cached_base_regrets.values()) <= 0.001
        ),
        "actual_cached_first_token_noninferior": paired_noninferior(
            packed_cached_first["loss"],
            dense_cached_first["loss"],
            cached_first_interval,
            0.0005,
        ),
        "packed_reload_matches_pre_export_within_1e_7": packed_replay_difference <= 1e-7,
        "strict_artifact_reload": not load_result.missing_keys and not load_result.unexpected_keys,
        "exact_payload_bytes": all(
            entry["total_payload_bytes"] == serving_ledger()["candidate_ffn_bytes_per_layer"]
            for entry in layer_payloads
        ),
        "no_training_shadows": not names_before_save and not names_after_reload,
        "base_codes_scales_valid": all(
            entry["all_finite"] and entry["base_code_min"] >= -127
            and entry["base_code_max"] <= 127
            and set(entry["delta_code_values"]) <= {-1, 0, 1}
            and entry["base_scale_min"] > 0 and entry["delta_scale_min"] > 0
            for entry in base_audits
        ),
        "branch_codes_scales_valid": all(
            entry["all_finite"] and entry["base_code_min"] >= -7
            and entry["base_code_max"] <= 7
            and entry["delta_code_values"] == [] and entry["base_scale_min"] > 0
            for entry in branch_audits
        ),
        "cached_semantics_valid": (
            max(
                fp32_cache["max_abs_logit_difference"],
                fp32_cache["max_abs_key_difference"],
                fp32_cache["max_abs_value_difference"],
            ) <= 3e-5
            and bf16_cache["max_abs_logit_difference"] <= 0.125
            and max(
                bf16_cache["max_abs_key_difference"],
                bf16_cache["max_abs_value_difference"],
            ) <= 0.1
        ),
        "training_finite": not dense_train["nonfinite"] and not candidate_train["nonfinite"],
    }
    decision = {
        "relative_mixed_suffix_gain": suffix_gain,
        "relative_decode_branch_gain_onepass": branch_gain,
        "relative_boundary_gains": boundary_gains,
        "relative_base_endpoint_regret": base_regret,
        "relative_full_endpoint_gain": full_gain,
        "relative_actual_cached_suffix_gain": cached_suffix_gain,
        "relative_decode_branch_gain_actual_cached": cached_branch_gain,
        "relative_onepass_regrets_vs_dense": onepass_dense_regrets,
        "relative_onepass_regrets_vs_base": onepass_base_regrets,
        "relative_actual_cached_regrets_vs_dense": cached_dense_regrets,
        "relative_actual_cached_regrets_vs_base": cached_base_regrets,
        "packed_replay_max_bucket_loss_difference": packed_replay_difference,
        "paired_suffix_interval": suffix_interval,
        "paired_first_token_interval": first_interval,
        "paired_decode_branch_interval_onepass": branch_interval,
        "paired_base_endpoint_interval": base_interval,
        "paired_full_endpoint_interval": full_interval,
        "paired_actual_cached_suffix_interval": cached_suffix_interval,
        "paired_actual_cached_first_token_interval": cached_first_interval,
        "paired_decode_branch_interval_actual_cached": cached_branch_interval,
        "gates": gates,
        "advance_to_physical_h100_kernel": all(gates.values()),
    }
    payload = {
        "schema": "phase-elastic-v3-50m-phase-faithful-v1",
        "source_sha256": sha256_file(Path(__file__)),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "discovery_sha256": sha256_file(DISCOVERY),
        "packed_replay_sha256": sha256_file(PACKED_REPLAY),
        "protocol_valid": protocol_valid,
        "data_ledger": data,
        "serving_ledger": serving_ledger(),
        "dense": dense,
        "candidate": {
            "train": candidate_train,
            "pre_export_phase_grid": pre_export_grid,
            "packed_phase_grid": packed_grid,
            "packed_base_phase_grid": packed_base_grid,
            "packed_base_endpoint": packed_base,
            "packed_full_endpoint": packed_full,
            "packed_cached": packed_cached,
            "packed_base_cached": packed_base_cached,
            "cached_equivalence": cached_semantics,
            "weight_audits": weight_audits,
        },
        "artifact": {
            "path": str(args.artifact),
            "sha256": artifact_sha256,
            "file_bytes_including_non_ffn_and_container": artifact_bytes,
            "per_layer_payloads": layer_payloads,
            "shadow_names_before_save": names_before_save,
            "shadow_names_after_reload": names_after_reload,
        },
        "decision": decision,
    }
    write_payload(args.output, payload)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
