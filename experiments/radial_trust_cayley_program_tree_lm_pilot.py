#!/usr/bin/env python3
"""10M-token LM pilot for the radial-trust Cayley program tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import transformers

import experiments.cayley_program_tree_lm_pilot as v1
from experiments.coalesced_attention_ffn_lm_screen import (
    build_model as build_parallel_model,
)
from experiments.radial_trust_cayley_program_tree import (
    RadialTrustCayleyProgramTreeMLP,
)
from experiments.reflex_swiglu_lm_screen import TokenFile, sha256_file, write_payload


SEED = v1.SEED
STEPS = v1.STEPS
PREREGISTRATION = Path(
    "results/radial-trust-cayley-program-tree-lm-preregistration.md"
)
STAGE0_PREREGISTRATION = Path(
    "results/radial-trust-cayley-program-tree-stage0-preregistration.md"
)
STAGE0_RESULT = Path("results/radial-trust-cayley-program-tree-stage0.json")
MODULE = Path("experiments/radial_trust_cayley_program_tree.py")


def build_candidate(
    device: torch.device, seed: int = SEED
) -> tuple[nn.Module, list[RadialTrustCayleyProgramTreeMLP]]:
    v1.seed_everything(seed)
    model, _ = build_parallel_model(device, "parallel_baseline")
    modules = []
    for layer_index, layer in enumerate(model.model.layers):
        replacement = RadialTrustCayleyProgramTreeMLP(
            seed=seed + 10_000 * layer_index,
            trust_rms=1.0,
        ).to(device)
        layer.mlp = replacement
        modules.append(replacement)
    return model, modules


def protocol_valid(
    args: argparse.Namespace,
    baseline: dict[str, Any],
    train_hash: str,
    validation_hash: str,
    stage0: dict[str, Any],
) -> bool:
    return bool(
        v1.baseline_is_compatible(baseline, train_hash, validation_hash)
        and stage0.get("pass") is True
        and stage0.get("module_sha256") == sha256_file(MODULE)
        and stage0.get("preregistration_sha256")
        == sha256_file(STAGE0_PREREGISTRATION)
        and args.steps == STEPS
        and args.sequence_length == 512
        and args.micro_batch_size == 8
        and args.gradient_accumulation == 8
        and args.eval_batch_size == 32
        and args.eval_batches == 128
        and args.warmup_steps == 30
        and args.learning_rate == 3e-4
        and args.weight_decay == 0.1
        and args.gradient_clip == 1.0
        and args.router_aux_weight == 0.01
        and args.seed == SEED
        and train_hash == v1.EXPECTED_DATA_HASHES["train"]
        and validation_hash == v1.EXPECTED_DATA_HASHES["validation"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--baseline-result", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/radial-trust-cayley-program-tree-lm-pilot.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("results/radial-trust-cayley-program-tree-lm-pilot.pt"),
    )
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--router-aux-weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    train_hash = sha256_file(args.train_file)
    validation_hash = sha256_file(args.validation_file)
    baseline = json.loads(args.baseline_result.read_text())
    stage0 = json.loads(STAGE0_RESULT.read_text())
    valid = protocol_valid(
        args, baseline, train_hash, validation_hash, stage0
    )
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    train = TokenFile(args.train_file, args.sequence_length)
    validation = TokenFile(args.validation_file, args.sequence_length)
    payload: dict[str, Any] = {
        "schema": "radial-trust-cayley-program-tree-lm-pilot-v1",
        "source_sha256": sha256_file(Path(__file__)),
        "module_sha256": sha256_file(MODULE),
        "v1_harness_sha256": sha256_file(
            Path("experiments/cayley_program_tree_lm_pilot.py")
        ),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "stage0_result_sha256": sha256_file(STAGE0_RESULT),
        "baseline_result_sha256": sha256_file(args.baseline_result),
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(),
        },
        "data": {
            "train_sha256": train_hash,
            "validation_sha256": validation_hash,
        },
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "protocol_valid": valid,
        "candidate": None,
    }
    v1.build_candidate = build_candidate
    candidate, checkpoint = v1.train_candidate(
        args, train, validation, device
    )
    payload["candidate"] = candidate
    payload["decision"] = v1.decide(candidate, baseline, valid)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.checkpoint)
    payload["checkpoint_sha256"] = sha256_file(args.checkpoint)
    write_payload(args.output, payload)
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
