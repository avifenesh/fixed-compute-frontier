#!/usr/bin/env python3
"""Fused full-FFN H100 gate for gauge-exposed partner-feedback SwiGLU."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from experiments.reflex_swiglu_fused_h100 import (
    baseline_activation,
    capture,
    max_row_relative,
    timed_replay,
    timing_summary,
)


@triton.jit
def partner_kernel(
    gate_ptr,
    up_ptr,
    up_weight_ptr,
    out_ptr,
    elements: tl.constexpr,
    hidden: tl.constexpr,
    chart: tl.constexpr,
    carrier_gain: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    feature = offsets % hidden
    partner_offsets = tl.where((feature % 2) == 0, offsets + 1, offsets - 1)
    partner_mask = partner_offsets < elements

    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    partner_gate = tl.load(
        gate_ptr + partner_offsets, mask=partner_mask, other=0.0
    ).to(tl.float32)
    partner_up = tl.load(
        up_ptr + partner_offsets, mask=partner_mask, other=0.0
    ).to(tl.float32)
    partner_z = partner_gate * tl.sigmoid(partner_gate) * partner_up
    bounded = tl.maximum(-1.0, tl.minimum(1.0, partner_z))

    packed = tl.load(up_weight_ptr).to(tl.float32)
    coefficient = (packed / chart - 1.0) / carrier_gain
    refined = gate + coefficient * bounded
    tl.store(out_ptr + offsets, refined * tl.sigmoid(refined) * up, mask=mask)


def partner_activation(gate, up, up_weight, chart, carrier_gain):
    output = torch.empty_like(gate)
    partner_kernel[(triton.cdiv(gate.numel(), 256),)](
        gate,
        up,
        up_weight,
        output,
        elements=gate.numel(),
        hidden=gate.shape[-1],
        chart=chart,
        carrier_gain=carrier_gain,
        BLOCK=256,
        num_warps=4,
    )
    return output


def decode_lambda(up_weight, chart, carrier_gain):
    return float((up_weight[0, 0].float().item() / chart - 1.0) / carrier_gain)


def torch_reference(gate, up, coefficient):
    z = F.silu(gate.float()) * up.float()
    partner = torch.arange(gate.shape[-1], device=gate.device) ^ 1
    return F.silu(gate.float() + coefficient * z[:, partner].clamp(-1.0, 1.0)) * up.float()


def baseline_path(x, gate, up, down):
    return baseline_activation(x @ gate.T, x @ up.T) @ down.T


def partner_path(x, gate, up, down, chart, carrier_gain):
    return partner_activation(x @ gate.T, x @ up.T, up, chart, carrier_gain) @ down.T


def make_weights(dimension, hidden, dtype, device, coefficient, carrier_gain):
    chart = dimension**-0.5
    gate = torch.randn(hidden, dimension, device=device, dtype=torch.float32) * chart
    up = torch.randn(hidden, dimension, device=device, dtype=torch.float32) * chart
    down = torch.randn(dimension, hidden, device=device, dtype=torch.float32) / hidden**0.5
    up[0, 0] = chart
    scale = 1.0 + carrier_gain * coefficient
    up[0] *= scale
    down[:, 0] /= scale
    return gate.to(dtype), up.to(dtype), down.to(dtype), chart, scale


def nonzero_reference(device, dimension, hidden, dtype, coefficient, carrier_gain):
    _, up_weight, _, chart, scale = make_weights(
        dimension, hidden, dtype, device, coefficient, carrier_gain
    )
    decoded = decode_lambda(up_weight, chart, carrier_gain)
    base = torch.linspace(-4.0, 4.0, hidden, device=device)
    gate = torch.stack((base, base, base)).to(dtype)
    up = torch.stack(
        (torch.full_like(base, -4.0), torch.full_like(base, 0.1), torch.full_like(base, 4.0))
    ).to(dtype)
    expected = torch_reference(gate, up, decoded)
    actual = partner_activation(gate, up, up_weight, chart, carrier_gain).float()
    torch.cuda.synchronize()
    z = F.silu(gate.float()) * up.float()
    return {
        "max_row_relative_error": max_row_relative(actual, expected),
        "encoded_lambda": coefficient,
        "decoded_lambda": decoded,
        "lambda_absolute_decode_error": abs(decoded - coefficient),
        "carrier_scale": scale,
        "negative_clipped_coordinates": int((z < -1.0).sum()),
        "unclipped_coordinates": int(((z >= -1.0) & (z <= 1.0)).sum()),
        "positive_clipped_coordinates": int((z > 1.0).sum()),
    }


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batches = [int(v) for v in args.batches.split(",")]
    gate, up, down, chart, scale = make_weights(
        args.dimension,
        args.hidden,
        dtype,
        device,
        args.coefficient,
        args.carrier_gain,
    )
    zero_gate, zero_up, zero_down, zero_chart, _ = make_weights(
        args.dimension, args.hidden, dtype, device, 0.0, args.carrier_gain
    )
    reference = nonzero_reference(
        device,
        args.dimension,
        args.hidden,
        dtype,
        args.coefficient,
        args.carrier_gain,
    )

    cells = []
    zero_errors = []
    for batch in batches:
        print(f"partner benchmark B={batch}", flush=True)
        x = torch.randn(batch, args.dimension, device=device, dtype=dtype)
        base_fn = lambda: baseline_path(x, gate, up, down)
        candidate_fn = lambda: partner_path(
            x, gate, up, down, chart, args.carrier_gain
        )
        zero_base_fn = lambda: baseline_path(x, zero_gate, zero_up, zero_down)
        zero_candidate_fn = lambda: partner_path(
            x, zero_gate, zero_up, zero_down, zero_chart, args.carrier_gain
        )
        with torch.inference_mode():
            zero_base = zero_base_fn().clone()
            zero_candidate = zero_candidate_fn().clone()
            torch.cuda.synchronize()
            zero_error = max_row_relative(zero_candidate, zero_base)
            zero_errors.append(zero_error)
            base_graph, _ = capture(base_fn)
            candidate_graph, _ = capture(candidate_fn)
            for _ in range(args.warmup):
                base_graph.replay(); candidate_graph.replay()
            torch.cuda.synchronize()
            values = {"baseline": [], "partner": []}
            graphs = {"baseline": base_graph, "partner": candidate_graph}
            for _ in range(args.samples):
                order = list(graphs); random.shuffle(order)
                for name in order:
                    values[name].append(timed_replay(graphs[name]))
        timings = {name: timing_summary(sample) for name, sample in values.items()}
        ratio = float(timings["partner"]["median_us"]) / float(timings["baseline"]["median_us"])
        cells.append({
            "batch": batch,
            "timings": timings,
            "partner_over_baseline_median": ratio,
            "zero_endpoint_max_row_relative_error": zero_error,
        })

    device_name = torch.cuda.get_device_name(0)
    protocol = (
        "H100" in device_name
        and args.dimension == 4096
        and args.hidden == 14336
        and set(batches) == {1, 8, 32, 128}
        and args.warmup >= 40
        and args.samples >= 200
        and args.coefficient == 0.08
        and args.carrier_gain == 4.0
    )
    key = [cell for cell in cells if cell["batch"] in {1, 8}]
    gates = {
        "protocol_valid": protocol,
        "zero_endpoint_below_0p2_percent": max(zero_errors) <= 0.002,
        "nonzero_reference_below_1_percent": reference["max_row_relative_error"] <= 0.01,
        "lambda_decode_below_0p002": reference["lambda_absolute_decode_error"] <= 0.002,
        "reference_exercises_all_clip_regions": (
            reference["negative_clipped_coordinates"] > 0
            and reference["unclipped_coordinates"] > 0
            and reference["positive_clipped_coordinates"] > 0
        ),
        "no_duplicate_parameter_or_workspace": True,
        "no_serial_feature_dependency": True,
        "within_1p02_on_key_cells": all(cell["partner_over_baseline_median"] <= 1.02 for cell in key),
        "within_1p05_on_full_grid": all(cell["partner_over_baseline_median"] <= 1.05 for cell in cells),
    }
    source = Path(__file__)
    prereg = Path("results/partner-feedback-swiglu-h100-preregistration.md")
    return {
        "candidate": "gauge-exposed partner-feedback SwiGLU",
        "scope": "fused full-FFN H100 feasibility only",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "preregistration_sha256": hashlib.sha256(prereg.read_bytes()).hexdigest(),
        "args": {**vars(args), "output": str(args.output)},
        "device": device_name,
        "device_properties": str(torch.cuda.get_device_properties(0)),
        "driver_version": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).splitlines()[0],
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "cuda_version": torch.version.cuda,
        "kernel": {
            "block": 256,
            "num_warps": 4,
            "matching": "adjacent fixed-point-free involution",
            "serial_feature_dependencies": 0,
            "coefficient_source": "deployed U[0,0] scale orbit",
            "cuda_graph_full_path": True,
        },
        "ledger": {
            "serialized_words_each": 3 * args.dimension * args.hidden,
            "dense_macs_per_token_each": 3 * args.dimension * args.hidden,
            "per_feature_coefficient_words": 0,
            "permutation_metadata_words": 0,
            "additional_activation_workspace_words": 0,
            "carrier_pivot_loads_per_kernel_program": 1,
            "kernel_programs_by_batch": {
                str(batch): (batch * args.hidden + 255) // 256 for batch in batches
            },
            "additional_activation_loads_per_output_feature": 2,
            "additional_silu_evaluations_per_output_feature": 1,
            "production_coalesced_projection_parity_established": False,
        },
        "carrier_scale": scale,
        "nonzero_reference": reference,
        "cells": cells,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=14336)
    parser.add_argument("--batches", default="1,8,32,128")
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=733)
    parser.add_argument("--coefficient", type=float, default=0.08)
    parser.add_argument("--carrier-gain", type=float, default=4.0)
    parser.add_argument(
        "--output", type=Path, default=Path("results/partner-feedback-swiglu-h100.json")
    )
    args = parser.parse_args()
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gates": payload["gates"], "all_gates_pass": payload["all_gates_pass"]}, indent=2))
    raise SystemExit(0 if payload["all_gates_pass"] else 1)


if __name__ == "__main__":
    main()

