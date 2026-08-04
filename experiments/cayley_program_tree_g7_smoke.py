#!/usr/bin/env python3
"""One-step GPU feasibility smoke for the full-budget Cayley program tree."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch
import transformers

from experiments.cayley_program_tree import exact_ledger
from experiments.cayley_program_tree_lm_pilot import (
    SEED,
    build_candidate,
    router_auxiliary,
)
from experiments.reflex_swiglu_lm_screen import TokenFile, sha256_file, write_payload
from experiments.triangular_microdepth_lm_screen import causal_loss


PREREGISTRATION = Path(
    "results/cayley-program-tree-g7-smoke-preregistration.md"
)
LANGUAGE_PREREGISTRATION = Path(
    "results/cayley-program-tree-lm-pilot-preregistration.md"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/cayley-program-tree-g7-smoke.json"),
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
    print(json.dumps({"status": "building_candidate"}), flush=True)
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
    print(json.dumps({"status": "starting_optimizer_step"}), flush=True)
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

    finite = all(
        math.isfinite(value)
        for value in (
            float(language.detach()),
            float(auxiliary.detach()),
            float(gradient_norm.detach()),
        )
    )
    payload = {
        "schema": "cayley-program-tree-g7-smoke-v1",
        "source_sha256": sha256_file(Path(__file__)),
        "module_sha256": sha256_file(Path("experiments/cayley_program_tree.py")),
        "pilot_source_sha256": sha256_file(
            Path("experiments/cayley_program_tree_lm_pilot.py")
        ),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "language_preregistration_sha256": sha256_file(
            LANGUAGE_PREREGISTRATION
        ),
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
            "language_loss": float(language.detach()),
            "router_auxiliary": float(auxiliary.detach()),
            "gradient_norm": float(gradient_norm.detach()),
            "elapsed_seconds": elapsed,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "finite": finite,
        },
        "pass": finite,
    }
    write_payload(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
