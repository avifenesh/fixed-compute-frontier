#!/usr/bin/env python3
"""H100 end-to-end latency gate for served ShiftNorm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics

import torch
import triton
import triton.language as tl

from experiments.projection_shared_coupling_h100 import (
    HIDDEN, ROWS, WIDTHS, capture_graph, elapsed_ms, launch_baseline,
    metadata, percentile,
)


OUTPUT = Path("results/folded-gain-shiftnorm-h100.json")


@triton.jit
def rms_shift_kernel(
    x_ptr,
    shift_ptr,
    output_ptr,
    stride: tl.constexpr,
    epsilon: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + row * stride + offsets).to(tl.float32)
    inverse_rms = tl.rsqrt(tl.sum(x * x, axis=0) / stride + epsilon)
    shift = tl.load(shift_ptr + offsets).to(tl.float32)
    tl.store(output_ptr + row * stride + offsets, x * inverse_rms + shift)


def launch_shift(x, shift, output):
    return rms_shift_kernel[(x.shape[0],)](
        x, shift, output, HIDDEN, 1e-6, BLOCK=HIDDEN, num_warps=8
    )


def run(timed_iterations: int, warmup_iterations: int):
    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("frozen gate requires SM90")
    torch.manual_seed(20260728); random.seed(20260728)
    device = torch.device("cuda")
    gain = (0.75 + 0.5 * torch.rand(HIDDEN, device=device)).to(torch.bfloat16)
    shift = (0.05 * torch.randn(HIDDEN, device=device)).to(torch.bfloat16)
    records=[]; correctness=[]; resources=None
    for rows in ROWS:
        x=torch.randn((rows,HIDDEN),device=device,dtype=torch.bfloat16)
        baseline_norm=torch.empty_like(x);shift_norm=torch.empty_like(x)
        baseline_compiled=launch_baseline(x,gain,baseline_norm)
        shift_compiled=launch_shift(x,shift,shift_norm);torch.cuda.synchronize()
        values=x.float();inverse=torch.rsqrt(values.square().mean(1,keepdim=True)+1e-6)
        correctness.append({"rows":rows,
            "baseline_max_abs_error":float((baseline_norm.float()-values*inverse*gain.float()).abs().max()),
            "shift_max_abs_error":float((shift_norm.float()-(values*inverse+shift.float())).abs().max())})
        if resources is None:
            resources={"baseline":metadata(baseline_compiled),"shift":metadata(shift_compiled)}
        for width in WIDTHS:
            weight=torch.randn((HIDDEN,width),device=device,dtype=torch.bfloat16)/HIDDEN**0.5
            baseline_projection=torch.empty((rows,width),device=device,dtype=torch.bfloat16)
            shift_projection=torch.empty_like(baseline_projection)
            torch.mm(baseline_norm,weight,out=baseline_projection)
            torch.mm(shift_norm,weight,out=shift_projection);torch.cuda.synchronize()
            baseline_graph=capture_graph(lambda:(launch_baseline(x,gain,baseline_norm),torch.mm(baseline_norm,weight,out=baseline_projection)))
            shift_graph=capture_graph(lambda:(launch_shift(x,shift,shift_norm),torch.mm(shift_norm,weight,out=shift_projection)))
            for _ in range(warmup_iterations):baseline_graph.replay();shift_graph.replay()
            torch.cuda.synchronize();samples={"baseline":[],"shift":[]}
            for _ in range(timed_iterations):
                order=["baseline","shift"];random.shuffle(order)
                for arm in order:samples[arm].append(elapsed_ms(baseline_graph if arm=="baseline" else shift_graph))
            baseline=statistics.median(samples["baseline"]);candidate=statistics.median(samples["shift"])
            records.append({"rows":rows,"width":width,"baseline_median_ms":baseline,
                "shift_median_ms":candidate,"shift_over_baseline":candidate/baseline,
                "baseline_p10_ms":percentile(samples["baseline"],.1),"baseline_p90_ms":percentile(samples["baseline"],.9),
                "shift_p10_ms":percentile(samples["shift"],.1),"shift_p90_ms":percentile(samples["shift"],.9)})
    decision={"correct":all(max(item["baseline_max_abs_error"],item["shift_max_abs_error"])<=.02 for item in correctness),
        "identical_resources":resources["baseline"]==resources["shift"],
        "key_cells_at_most_1_01":all(item["shift_over_baseline"]<=1.01 for item in records if item["rows"] in (256,1024)),
        "all_cells_at_most_1_03":all(item["shift_over_baseline"]<=1.03 for item in records)}
    decision["pass"]=all(decision.values())
    return{"schema":"folded-gain-shiftnorm-h100-v1","device":torch.cuda.get_device_name(),
        "iterations":{"warmup":warmup_iterations,"timed":timed_iterations},"resources":resources,
        "correctness":correctness,"records":records,"decision":decision}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--timed-iterations",type=int,default=500);parser.add_argument("--warmup-iterations",type=int,default=50);parser.add_argument("--output",type=Path,default=OUTPUT)
    args=parser.parse_args();payload=run(args.timed_iterations,args.warmup_iterations);args.output.write_text(json.dumps(payload,indent=2)+"\n");print(json.dumps(payload,indent=2))
if __name__=="__main__":main()
