#!/usr/bin/env python3
"""One-step GPU smoke for the radial-trust Cayley program tree."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch
import transformers

from experiments.cayley_program_tree import exact_ledger
from experiments.cayley_program_tree_lm_pilot import router_auxiliary
from experiments.radial_trust_cayley_program_tree_lm_pilot import (
    SEED,
    build_candidate,
)
from experiments.reflex_swiglu_lm_screen import TokenFile, sha256_file, write_payload
from experiments.triangular_microdepth_lm_screen import causal_loss


PREREGISTRATION = Path(
    "results/radial-trust-cayley-program-tree-g7-smoke-preregistration.md"
)
LANGUAGE_PREREGISTRATION = Path(
    "results/radial-trust-cayley-program-tree-lm-preregistration.md"
)
STAGE0_RESULT = Path("results/radial-trust-cayley-program-tree-stage0.json")
MODULE = Path("experiments/radial_trust_cayley_program_tree.py")
PILOT = Path("experiments/radial_trust_cayley_program_tree_lm_pilot.py")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/radial-trust-cayley-program-tree-g7-smoke.json"),
    )
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--router-aux-weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    tokens = TokenFile(args.train_file, args.sequence_length)
    model, modules = build_candidate(device, args.seed)
    model.config.use_cache = False
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        fused=True,
    )
    optimizer.zero_grad(set_to_none=True)
    inputs, targets = tokens.batch(0, args.micro_batch_size, device)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    language = causal_loss(model, inputs, targets)
    auxiliary = router_auxiliary(modules)
    objective = language + args.router_aux_weight * auxiliary
    objective.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), args.gradient_clip
    )
    optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    numeric = (
        float(language.detach()),
        float(auxiliary.detach()),
        float(gradient_norm.detach()),
    )
    finite = all(math.isfinite(value) for value in numeric)
    stage0 = json.loads(STAGE0_RESULT.read_text())
    integrity = bool(
        stage0.get("pass")
        and stage0.get("module_sha256") == sha256_file(MODULE)
        and args.sequence_length == 512
        and args.micro_batch_size == 8
        and args.gradient_clip == 1.0
        and args.router_aux_weight == 0.01
        and args.seed == SEED
    )
    payload = {
        "schema": "radial-trust-cayley-program-tree-g7-smoke-v1",
        "source_sha256": sha256_file(Path(__file__)),
        "module_sha256": sha256_file(MODULE),
        "pilot_source_sha256": sha256_file(PILOT),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "language_preregistration_sha256": sha256_file(
            LANGUAGE_PREREGISTRATION
        ),
        "stage0_result_sha256": sha256_file(STAGE0_RESULT),
        "data_sha256": sha256_file(args.train_file),
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(),
            "device_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        },
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "parameters": {
            "total": sum(parameter.numel() for parameter in model.parameters()),
            "ffn": sum(module.parameter_count() for module in modules),
        },
        "ledger": exact_ledger(),
        "measurement": {
            "language_loss": numeric[0],
            "router_auxiliary": numeric[1],
            "gradient_norm": numeric[2],
            "elapsed_seconds": elapsed,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "finite": finite,
        },
        "integrity": integrity,
        "pass": finite and integrity,
    }
    write_payload(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
