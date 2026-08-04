#!/usr/bin/env python3
"""Recreate phase-elastic v3, export packed weights, reload, and re-evaluate."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
import transformers

from experiments.phase_elastic_packed_artifact import (
    freeze_phase_elastic_modules,
    shadow_names,
)
from experiments.phase_elastic_residual_precision_v2_lm_screen import serving_ledger
from experiments.phase_elastic_v3_phase_faithful import (
    EVAL_FIRST_FULL,
    SEED,
    aggregate_bucket,
    build_candidate,
    decide as phase_decide,
    evaluate_phase_grid,
    paired_noninferior,
    train_candidate,
    verify_cached_equivalence,
)
from experiments.reflex_swiglu_lm_screen import (
    TokenFile,
    paired_loss_interval,
    sha256_file,
    validate_data_ledger,
    write_payload,
)


DISCOVERY = Path("results/phase-elastic-v3-phase-faithful.json")
DISCOVERY_SHA256 = "e59e2ef52cfe252a4686f414e3220dafc0700477f56dab04e1a505327cb7acf5"
PREREGISTRATION = Path("results/phase-elastic-v3-packed-replay-preregistration.md")


def grid_max_loss_difference(left, right) -> float:
    differences = []
    for boundary in EVAL_FIRST_FULL:
        for bucket, entry in left[str(boundary)].items():
            differences.append(abs(entry["loss"] - right[str(boundary)][bucket]["loss"]))
    return max(differences)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=Path("data/block-algebra-scratch/train.uint16.bin"))
    parser.add_argument("--validation-file", type=Path, default=Path("data/block-algebra-scratch/validation.uint16.bin"))
    parser.add_argument("--data-manifest", type=Path, default=Path("results/block-algebra-scratch-data-manifest.json"))
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/phase-elastic-v3-packed-replay.pt"))
    parser.add_argument("--output", type=Path, default=Path("results/phase-elastic-v3-packed-replay.json"))
    parser.add_argument("--steps", type=int, default=305)
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
        args.steps, args.sequence_length, args.micro_batch_size,
        args.gradient_accumulation, args.eval_batch_size, args.eval_batches,
        args.warmup_steps, args.learning_rate, args.weight_decay,
        args.gradient_clip, args.seed,
    ) == (305, 512, 32, 2, 32, 64, 30, 3e-4, 0.1, 1.0, SEED)
    expected = (
        expected
        and torch.__version__ == "2.5.1+cu124"
        and torch.version.cuda == "12.4"
        and transformers.__version__ == "4.57.6"
        and "H100" in torch.cuda.get_device_name()
    )
    if sha256_file(DISCOVERY) != DISCOVERY_SHA256:
        raise RuntimeError("frozen phase-faithful discovery changed")
    discovery = json.loads(DISCOVERY.read_text())
    data = validate_data_ledger(
        args.data_manifest, args.train_file, args.validation_file, args.sequence_length
    )
    protocol_valid = expected and discovery["decision"]["advance_to_long_horizon_multiseed"] and data["valid"]
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    train_file = TokenFile(args.train_file, args.sequence_length)
    validation = TokenFile(args.validation_file, args.sequence_length)

    print(json.dumps({"stage": "deterministic_candidate_recreation"}), flush=True)
    model, modules, train = train_candidate(args, train_file, device)
    pre_export_grid = evaluate_phase_grid(
        model, modules, validation, EVAL_FIRST_FULL,
        args.eval_batches, args.eval_batch_size, device,
    )
    recreation_max_difference = grid_max_loss_difference(
        pre_export_grid, discovery["candidate"]["phase_grid"]
    )
    set_audits = freeze_phase_elastic_modules(modules)
    names_before_save = shadow_names(model)
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, args.artifact)
    artifact_sha256 = sha256_file(args.artifact)
    artifact_bytes = args.artifact.stat().st_size
    del modules, model
    gc.collect()
    torch.cuda.empty_cache()

    print(json.dumps({"stage": "independent_packed_reload"}), flush=True)
    reloaded, reloaded_modules = build_candidate(device, args.seed + 99)
    freeze_phase_elastic_modules(reloaded_modules)
    state = torch.load(args.artifact, map_location=device, weights_only=True)["model"]
    load_result = reloaded.load_state_dict(state, strict=True)
    names_after_reload = shadow_names(reloaded)
    packed_grid = evaluate_phase_grid(
        reloaded, reloaded_modules, validation, EVAL_FIRST_FULL,
        args.eval_batches, args.eval_batch_size, device,
    )
    cached = verify_cached_equivalence(reloaded, reloaded_modules, validation, device)
    audits = [module.audit() for module in reloaded_modules]

    dense_grid = discovery["dense"]["phase_grid"]
    dense_suffix = aggregate_bucket(dense_grid, "all_suffix")
    discovery_suffix = aggregate_bucket(discovery["candidate"]["phase_grid"], "all_suffix")
    packed_suffix = aggregate_bucket(packed_grid, "all_suffix")
    dense_first = aggregate_bucket(dense_grid, "first_token")
    packed_first = aggregate_bucket(packed_grid, "first_token")
    interval = paired_loss_interval(
        {"evaluations": {"terminal": packed_suffix}},
        {"evaluations": {"terminal": dense_suffix}},
        "terminal",
    )
    first_interval = paired_loss_interval(
        {"evaluations": {"terminal": packed_first}},
        {"evaluations": {"terminal": dense_first}},
        "terminal",
    )
    discovery_gain = (dense_suffix["loss"] - discovery_suffix["loss"]) / dense_suffix["loss"]
    packed_gain = (dense_suffix["loss"] - packed_suffix["loss"]) / dense_suffix["loss"]
    base_audits = [
        layer[name]
        for layer in audits
        for name in ("base_gate", "base_up", "base_down")
    ]
    branch_audits = [
        layer[name]
        for layer in audits
        for name in ("branch_gate", "branch_up", "branch_down")
    ]
    packed_replay_max_difference = grid_max_loss_difference(
        packed_grid, pre_export_grid
    )
    cache_fp32 = cached["float32"]
    cache_bf16 = cached["bfloat16"]
    gates = {
        "protocol_integrity_valid": protocol_valid,
        "deterministic_recreation_matches_within_1e_7": recreation_max_difference <= 1e-7,
        "packed_reload_matches_pre_export_within_1e_7": packed_replay_max_difference <= 1e-7,
        "strict_artifact_reload": not load_result.missing_keys and not load_result.unexpected_keys,
        "no_training_shadows_before_save": not names_before_save,
        "no_training_shadows_after_reload": not names_after_reload,
        "exact_payload_bytes_every_layer": all(
            entry["total_payload_bytes"] == serving_ledger()["candidate_ffn_bytes_per_layer"]
            for entry in set_audits
        ),
        "base_codes_scales_valid": all(
            entry["all_finite"]
            and entry["base_code_min"] >= -127
            and entry["base_code_max"] <= 127
            and set(entry["delta_code_values"]) <= {-1, 0, 1}
            and entry["base_scale_min"] > 0
            and entry["delta_scale_min"] > 0
            for entry in base_audits
        ),
        "branch_w4_codes_scales_valid": all(
            entry["all_finite"]
            and entry["base_code_min"] >= -7
            and entry["base_code_max"] <= 7
            and entry["delta_code_values"] == []
            and entry["base_scale_min"] > 0
            for entry in branch_audits
        ),
        "retains_90_percent_of_discovery_gain": packed_gain >= 0.9 * discovery_gain and interval["upper_95"] < 0,
        "first_token_noninferior": paired_noninferior(
            packed_first["loss"], dense_first["loss"], first_interval, 0.0005
        ),
        "cached_semantics_pass": (
            max(
                cache_fp32["max_abs_logit_difference"],
                cache_fp32["max_abs_key_difference"],
                cache_fp32["max_abs_value_difference"],
            ) <= 3e-5
            and cache_bf16["max_abs_logit_difference"] <= 0.125
            and max(
                cache_bf16["max_abs_key_difference"],
                cache_bf16["max_abs_value_difference"],
            ) <= 0.1
        ),
    }
    decision = {
        "recreation_max_bucket_loss_difference": recreation_max_difference,
        "packed_replay_max_bucket_loss_difference": packed_replay_max_difference,
        "discovery_relative_suffix_gain": discovery_gain,
        "packed_relative_suffix_gain": packed_gain,
        "packed_suffix_loss": packed_suffix["loss"],
        "packed_first_token_loss": packed_first["loss"],
        "paired_suffix_interval": interval,
        "paired_first_token_interval": first_interval,
        "gates": gates,
        "advance_to_50m_phase_faithful": all(gates.values()),
    }
    payload = {
        "schema": "phase-elastic-v3-packed-replay-v1",
        "source_sha256": sha256_file(Path(__file__)),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "discovery_sha256": sha256_file(DISCOVERY),
        "protocol_valid": protocol_valid,
        "data_ledger": data,
        "serving_ledger": serving_ledger(),
        "training": train,
        "artifact": {
            "path": str(args.artifact),
            "sha256": artifact_sha256,
            "file_bytes_including_non_ffn_and_container": artifact_bytes,
            "per_layer_packed_audits": set_audits,
            "shadow_names_before_save": names_before_save,
            "shadow_names_after_reload": names_after_reload,
        },
        "pre_export_phase_grid": pre_export_grid,
        "packed_phase_grid": packed_grid,
        "cached_equivalence": cached,
        "quantized_weight_audits": audits,
        "decision": decision,
    }
    write_payload(args.output, payload)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
