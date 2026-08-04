#!/usr/bin/env python3
"""Deployment-folded H100 gate for exact-budget self-product FFN."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import transformers

from experiments.self_product_ffn_lm_screen import VALID_ARMS, build_model


CELLS = ((1, 1), (1, 512), (8, 512), (32, 512))
PREREGISTRATION = Path("results/self-product-ffn-h100-preregistration.md")
LM_PREREGISTRATION = Path("results/self-product-ffn-lm-preregistration.md")
LM_SOURCE = Path("experiments/self_product_ffn_lm_screen.py")
TEST_SOURCE = Path("tests/test_self_product_ffn_h100.py")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes())
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def build_folded(device: torch.device, arm: str):
    torch.manual_seed(1907); torch.cuda.manual_seed_all(1907)
    model, modules = build_model(device, arm)
    for module in modules:
        module.fold_for_deployment()
    return model.eval(), modules


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if args.arms != ",".join(VALID_ARMS) or args.cells != "1x1,1x512,8x512,32x512":
        raise ValueError("frozen arms/cells changed")
    if (args.warmups, args.repetitions, args.seed) != (5, 30, 1907):
        raise ValueError("frozen timing protocol changed")
    torch.set_float32_matmul_precision("high"); torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    built = {arm: build_folded(device, arm) for arm in VALID_ARMS}
    models = {arm: value[0] for arm, value in built.items()}
    counts = {arm: sum(parameter.numel() for parameter in model.parameters()) for arm, model in models.items()}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"parameter mismatch: {counts}")
    randomizer = random.Random(args.seed)
    results = {}
    for batch, tokens in CELLS:
        inputs = torch.randint(0, 49152, (batch, tokens), device=device)
        for model in models.values():
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for _ in range(args.warmups):
                    model(input_ids=inputs, use_cache=False)
        torch.cuda.synchronize()
        samples = {arm: [] for arm in VALID_ARMS}
        for _ in range(args.repetitions):
            order = list(VALID_ARMS); randomizer.shuffle(order)
            for arm in order:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    started = time.perf_counter()
                    models[arm](input_ids=inputs, use_cache=False)
                    torch.cuda.synchronize()
                samples[arm].append((time.perf_counter() - started) * 1000.0)
        baseline = float(np.median(samples["parallel_swiglu"]))
        results[f"{batch}x{tokens}"] = {
            arm: {
                "median_ms": float(np.median(samples[arm])),
                "p10_ms": percentile(samples[arm], 0.1),
                "p90_ms": percentile(samples[arm], 0.9),
                "min_ms": min(samples[arm]),
                "ratio_to_parallel_swiglu": float(np.median(samples[arm])) / baseline,
                "samples_ms": samples[arm],
            }
            for arm in VALID_ARMS
        }
    cell_passes = {
        cell: values["parallel_self_product"]["ratio_to_parallel_swiglu"] <= 1.02
        for cell, values in results.items()
    }
    return {
        "schema": "self-product-ffn-folded-h100-v2",
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "transformers_version": transformers.__version__,
        "parameter_counts": counts,
        "protocol": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "hashes": {
            "source": sha256_file(Path(__file__)),
            "preregistration": sha256_file(PREREGISTRATION),
            "lm_source": sha256_file(LM_SOURCE),
            "lm_preregistration": sha256_file(LM_PREREGISTRATION),
            "test": sha256_file(TEST_SOURCE),
        },
        "results": results,
        "candidate_cell_passes": cell_passes,
        "h100_feasibility_pass": all(cell_passes.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default=",".join(VALID_ARMS))
    parser.add_argument("--cells", default="1x1,1x512,8x512,32x512")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1907)
    parser.add_argument("--output", type=Path, default=Path("results/self-product-ffn-h100.json"))
    args = parser.parse_args()
    payload = run(args)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "h100_feasibility_pass": payload["h100_feasibility_pass"],
        "candidate_cell_passes": payload["candidate_cell_passes"],
        "medians": {cell: {arm: values[arm]["median_ms"] for arm in VALID_ARMS} for cell, values in payload["results"].items()},
        "ratios": {cell: {arm: values[arm]["ratio_to_parallel_swiglu"] for arm in VALID_ARMS} for cell, values in payload["results"].items()},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
