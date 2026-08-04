#!/usr/bin/env python3
"""Development check for BF16 loss preservation after zero-pivot canonicalization."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import random

import numpy as np
import torch

from experiments.gauge_zero_g1_lm_pilot import ARMS, build_model
from experiments.reflex_swiglu_lm_screen import (
    TokenFile,
    evaluate,
    validate_data_ledger,
    write_payload,
)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation-file",
        type=Path,
        default=Path("data/block-algebra-scratch/validation.uint16.bin"),
    )
    parser.add_argument(
        "--data-manifest",
        type=Path,
        default=Path("results/block-algebra-scratch-data-manifest.json"),
    )
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2207)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/gauge-zero-g1-initial-eval.json"),
    )
    args = parser.parse_args()
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("H100 required")
    ledger = validate_data_ledger(
        args.data_manifest,
        Path("data/block-algebra-scratch/train.uint16.bin"),
        args.validation_file,
        args.sequence_length,
    )
    validation = TokenFile(args.validation_file, args.sequence_length)
    device = torch.device("cuda")
    arms = {}
    for arm in ARMS:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        model, _ = build_model(device, arm)
        arms[arm] = {
            "evaluation": evaluate(
                model,
                validation,
                args.eval_batches,
                args.eval_batch_size,
                device,
            ),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "parameter_tensors": len(list(model.parameters())),
        }
        del model
        gc.collect()
        torch.cuda.empty_cache()

    raw = arms["raw_baseline"]["evaluation"]
    control = arms["canonical_bilinear_control"]["evaluation"]
    candidate = arms["gauge_zero_g1"]["evaluation"]
    paired_difference = np.asarray(control["per_batch_loss"]) - np.asarray(
        raw["per_batch_loss"]
    )
    paired_mean = float(paired_difference.mean())
    paired_standard_error = float(
        paired_difference.std(ddof=1) / np.sqrt(paired_difference.size)
    )
    paired_interval = {
        "control_minus_raw_mean": paired_mean,
        "standard_error": paired_standard_error,
        "lower_95": paired_mean - 1.96 * paired_standard_error,
        "upper_95": paired_mean + 1.96 * paired_standard_error,
    }
    payload = {
        "schema": "gauge-zero-g1-initial-eval-development-v1",
        "device": torch.cuda.get_device_name(0),
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda},
        "data_ledger": ledger,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "arms": arms,
        "checks": {
            "equal_parameter_counts": len({
                result["parameters"] for result in arms.values()
            }) == 1,
            "equal_parameter_tensor_counts": len({
                result["parameter_tensors"] for result in arms.values()
            }) == 1,
            "candidate_control_loss_bit_exact": (
                candidate["loss"] == control["loss"]
                and candidate["per_batch_loss"] == control["per_batch_loss"]
            ),
            "canonical_raw_aggregate_loss_delta_at_most_1e_4": (
                abs(control["loss"] - raw["loss"]) <= 1e-4
            ),
            "canonical_raw_paired_interval_within_1e_4": (
                paired_interval["lower_95"] >= -1e-4
                and paired_interval["upper_95"] <= 1e-4
            ),
        },
        "paired_control_vs_raw": paired_interval,
        "loss_deltas": {
            "control_minus_raw": control["loss"] - raw["loss"],
            "candidate_minus_control": candidate["loss"] - control["loss"],
        },
    }
    write_payload(args.output, payload)
    print(json.dumps({
        "checks": payload["checks"],
        "loss_deltas": payload["loss_deltas"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
