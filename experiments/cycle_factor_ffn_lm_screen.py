#!/usr/bin/env python3
"""Frozen 10M-token screen for the 2DM cycle-factor FFN."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from experiments.coalesced_attention_ffn_lm_screen import build_model as build_parallel_model
from experiments.reflex_swiglu_lm_screen import (
    TokenFile, evaluate, lr_multiplier, paired_loss_interval, sha256_file,
    validate_data_ledger, write_payload,
)
from experiments.triangular_microdepth_lm_screen import (
    BASELINE_INTERMEDIATE_SIZE as M, HIDDEN_SIZE as D, causal_loss,
)


ARMS = ("full_swiglu", "narrow_swiglu", "self_product", "cycle_factor")
NARROW_WIDTHS = (683, 683, 682) * 4
SELF_SCALE = 0.44642046792894413 / math.sqrt(2.0 / 3.0)
SEED = 815
PREREGISTRATION = Path("results/cycle-factor-ffn-preregistration.md")
STAGE0 = Path("results/cycle-factor-ffn-stage0.json")
FROZEN_HASHES = {
    Path("results/block-algebra-scratch-data-manifest.json"): "102d8518f3b3737bcffc92a79bbf6988f30515079db47f48bbcc445acb43f7b8",
    Path("experiments/cycle_factor_ffn_stage0.py"): "47ae32b88368a8aa93d1c5da3de3fde8e5b35d7c0726e2642fd82527744a0092",
    Path("tests/test_cycle_factor_ffn_stage0.py"): "d43c58d25ca70ff8bf6c933adffe6a83b38abf6ecf547a57e7fee6f9a73080f0",
    Path("results/cycle-factor-ffn-stage0.json"): "f6390f874ff1a2a3ac425ed6609e35d21fbefe801f80e36c922d0dee33fc9d3e",
    Path("results/cycle-factor-ffn-preregistration.md"): "7fc60099c6eea6e83e87703d050e0cc897ab0750f91aabad37d3e7be8ba63e08",
}


class NarrowSwiGLU(nn.Module):
    def __init__(self, width: int, std: float) -> None:
        super().__init__(); self.width = width
        self.gate_proj = nn.Linear(D, width, bias=False)
        self.up_proj = nn.Linear(D, width, bias=False)
        self.down_proj = nn.Linear(width, D, bias=False)
        for projection in (self.gate_proj, self.up_proj, self.down_proj):
            nn.init.normal_(projection.weight, mean=0.0, std=std)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        activation = F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        return self.down_proj(narrow_scale(self.width) * activation)


def narrow_scale(width: int) -> float:
    return math.sqrt(M / width)


def route_values(generated: torch.Tensor, mode: str = "full") -> torch.Tensor:
    if mode == "self":
        return generated
    grouped = generated.unflatten(-1, (M // 4, 4))
    shift = -2 if mode == "second_neighbor" else -1
    return grouped.roll(shift, dims=-1).flatten(-2)


class SharedFactorMLP(nn.Module):
    def __init__(self, arm: str, std: float) -> None:
        super().__init__(); self.arm = arm; self.ablation = "full"
        self.generator_proj = nn.Linear(D, M, bias=False)
        self.down_proj = nn.Linear(M, D, bias=False)
        for projection in (self.generator_proj, self.down_proj):
            nn.init.normal_(projection.weight, mean=0.0, std=std)

    def activation(self, hidden_states: torch.Tensor) -> torch.Tensor:
        generated = self.generator_proj(hidden_states)
        if self.arm == "self_product" or self.ablation == "self":
            value = generated
            scale = SELF_SCALE
        else:
            value = route_values(generated, self.ablation)
            scale = 1.0
        return scale * F.silu(generated) * value

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.activation(hidden_states))


def build_arm(device: torch.device, arm: str, seed: int = SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model, _ = build_parallel_model(device, "parallel_baseline")
    modules = []
    if arm == "full_swiglu":
        return model, modules
    for index, layer in enumerate(model.model.layers):
        if arm == "narrow_swiglu":
            layer.mlp = NarrowSwiGLU(NARROW_WIDTHS[index], model.config.initializer_range).to(device)
        else:
            replacement = SharedFactorMLP(arm, model.config.initializer_range).to(device)
            layer.mlp = replacement; modules.append(replacement)
    return model, modules


@torch.no_grad()
def diagnostics(model: nn.Module, modules: list[SharedFactorMLP], validation: TokenFile, device: torch.device):
    model.eval(); inputs, _ = validation.batch(0, 2, device); captured = []
    if modules:
        handles = [module.down_proj.register_forward_pre_hook(lambda _m, args: captured.append(args[0].detach())) for module in modules]
    else:
        handles = [layer.mlp.down_proj.register_forward_pre_hook(lambda _m, args: captured.append(args[0].detach())) for layer in model.model.layers]
    with torch.autocast("cuda", dtype=torch.bfloat16): model(input_ids=inputs, use_cache=False)
    for handle in handles: handle.remove()
    rms = [float(value.float().square().mean().sqrt()) for value in captured]
    finite = [float((~torch.isfinite(value)).float().mean()) for value in captured]
    return {"activation_rms_layer_median": float(np.median(rms)), "activation_rms_layer_max": max(rms),
            "nonfinite_fraction_layer_max": max(finite)}


def train_arm(arm, args, train, validation, device):
    model, modules = build_arm(device, arm, args.seed); model.config.use_cache = False
    total = sum(parameter.numel() for parameter in model.parameters())
    ffn = sum(sum(parameter.numel() for parameter in layer.mlp.parameters()) for layer in model.model.layers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8,
                                  weight_decay=args.weight_decay, fused=True)
    optimizer.param_groups[0]["base_lr"] = args.learning_rate
    evaluations = {"0": evaluate(model, validation, args.eval_batches, args.eval_batch_size, device)}
    initial_diagnostics = diagnostics(model, modules, validation, device)
    optimizer.zero_grad(set_to_none=True); losses=[]; norms=[]; durations=[]
    for step in range(args.steps):
        model.train(); optimizer.param_groups[0]["lr"] = args.learning_rate * lr_multiplier(step, args.steps, args.warmup_steps)
        started=time.perf_counter(); accumulated=0.0
        for micro in range(args.gradient_accumulation):
            inputs, targets = train.batch(step * args.gradient_accumulation + micro, args.micro_batch_size, device)
            value = causal_loss(model, inputs, targets); (value / args.gradient_accumulation).backward()
            accumulated += float(value.detach()) / args.gradient_accumulation
        norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip))
        if not math.isfinite(accumulated) or not math.isfinite(norm): raise RuntimeError(f"nonfinite {arm} step {step}")
        optimizer.step(); optimizer.zero_grad(set_to_none=True); torch.cuda.synchronize()
        losses.append(accumulated); norms.append(norm); durations.append(time.perf_counter()-started)
    evaluations[str(args.steps)] = evaluate(model, validation, args.eval_batches, args.eval_batch_size, device)
    terminal_diagnostics = diagnostics(model, modules, validation, device)
    ablations = {}
    if arm == "cycle_factor":
        for mode in ("self", "second_neighbor"):
            for module in modules: module.ablation = mode
            ablations[mode] = evaluate(model, validation, args.eval_batches, args.eval_batch_size, device)
        for module in modules: module.ablation = "full"
    result = {"arm": arm, "total_parameters": total, "ffn_parameters": ffn, "evaluations": evaluations,
              "diagnostics": {"initial": initial_diagnostics, "terminal": terminal_diagnostics},
              "terminal_ablations": ablations,
              "train": {"prediction_tokens": args.steps*args.gradient_accumulation*args.micro_batch_size*args.sequence_length,
                        "mean_loss": float(np.mean(losses)), "final_loss": losses[-1], "max_gradient_norm": max(norms),
                        "elapsed_seconds": sum(durations), "tokens_per_second": args.steps*args.gradient_accumulation*args.micro_batch_size*args.sequence_length/sum(durations),
                        "nonfinite": False}}
    del optimizer, modules, model; gc.collect(); torch.cuda.empty_cache(); return result


def decide(results, protocol_valid):
    if set(results) != set(ARMS): return {"complete": False, "missing": sorted(set(ARMS)-set(results))}
    step="305"; losses={arm:result["evaluations"][step]["loss"] for arm,result in results.items()}
    candidate=results["cycle_factor"]; comparisons={arm:paired_loss_interval(candidate, results[arm], step) for arm in ARMS[:-1]}
    baseline=losses["full_swiglu"]; margin=.0005*baseline
    beats_baseline=(baseline-losses["cycle_factor"])/baseline>=.0005 and comparisons["full_swiglu"]["upper_95"]<0
    controls=("narrow_swiglu","self_product")
    noninferior=(losses["cycle_factor"]-baseline)/baseline<=.0005 and comparisons["full_swiglu"]["upper_95"]<=margin
    beats_controls=all((losses[arm]-losses["cycle_factor"])/losses[arm]>=.0005 and comparisons[arm]["upper_95"]<0 for arm in controls)
    equal_cost={results[arm]["ffn_parameters"] for arm in ("narrow_swiglu","self_product","cycle_factor")}
    gates={"protocol_valid":protocol_valid,"candidate_total_smaller_than_full":candidate["total_parameters"]<results["full_swiglu"]["total_parameters"],
           "equal_2dm_ffn_parameters":len(equal_cost)==1,"candidate_ffn_is_two_thirds_full":3*candidate["ffn_parameters"]==2*results["full_swiglu"]["ffn_parameters"],
           "finite":all(not result["train"]["nonfinite"] and result["diagnostics"]["terminal"]["nonfinite_fraction_layer_max"]==0 for result in results.values()),
           "beats_full_path":beats_baseline,"noninferior_full":noninferior,"beats_both_equal_cost_controls":beats_controls}
    return {"complete":True,"losses":losses,"relative_cycle_improvements":{arm:(losses[arm]-losses["cycle_factor"])/losses[arm] for arm in ARMS[:-1]},
            "paired_intervals":comparisons,"terminal_ablations":{k:v["loss"] for k,v in candidate["terminal_ablations"].items()},
            "gates":gates,"advance_to_50m":gates["protocol_valid"] and gates["candidate_total_smaller_than_full"] and gates["equal_2dm_ffn_parameters"] and gates["candidate_ffn_is_two_thirds_full"] and gates["finite"] and (beats_baseline or (noninferior and beats_controls))}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--train-file",type=Path,default=Path("data/block-algebra-scratch/train.uint16.bin"))
    parser.add_argument("--validation-file",type=Path,default=Path("data/block-algebra-scratch/validation.uint16.bin")); parser.add_argument("--data-manifest",type=Path,default=Path("results/block-algebra-scratch-data-manifest.json"))
    parser.add_argument("--output",type=Path,default=Path("results/cycle-factor-ffn-lm-screen.json")); parser.add_argument("--steps",type=int,default=305)
    parser.add_argument("--sequence-length",type=int,default=512); parser.add_argument("--micro-batch-size",type=int,default=32); parser.add_argument("--gradient-accumulation",type=int,default=2)
    parser.add_argument("--eval-batch-size",type=int,default=32); parser.add_argument("--eval-batches",type=int,default=128); parser.add_argument("--warmup-steps",type=int,default=30)
    parser.add_argument("--learning-rate",type=float,default=3e-4); parser.add_argument("--weight-decay",type=float,default=.1); parser.add_argument("--gradient-clip",type=float,default=1.); parser.add_argument("--seed",type=int,default=SEED); args=parser.parse_args()
    expected=(args.steps,args.sequence_length,args.micro_batch_size,args.gradient_accumulation,args.eval_batch_size,args.eval_batches,args.warmup_steps,args.learning_rate,args.weight_decay,args.gradient_clip,args.seed)==(305,512,32,2,32,128,30,3e-4,.1,1.0,SEED)
    expected=expected and torch.__version__=="2.5.1+cu124" and torch.version.cuda=="12.4" and transformers.__version__=="4.57.6" and "H100" in torch.cuda.get_device_name()
    integrity={str(path):sha256_file(path)==digest for path,digest in FROZEN_HASHES.items()}
    stage0=json.loads(STAGE0.read_text()); ledger=validate_data_ledger(args.data_manifest,args.train_file,args.validation_file,args.sequence_length); valid=expected and all(integrity.values()) and stage0["pass"] and ledger["valid"]
    train=TokenFile(args.train_file,args.sequence_length); validation=TokenFile(args.validation_file,args.sequence_length); device=torch.device("cuda")
    torch.set_float32_matmul_precision("high"); torch.backends.cuda.matmul.allow_tf32=True
    payload={"schema":"cycle-factor-ffn-lm-v1","source_sha256":sha256_file(Path(__file__)),"preregistration_sha256":sha256_file(PREREGISTRATION),"stage0_sha256":sha256_file(STAGE0),"integrity_checks":integrity,"protocol_valid":valid,"data_ledger":ledger,"arms":{}}
    for arm in ARMS:
        print(json.dumps({"starting_arm":arm}),flush=True); payload["arms"][arm]=train_arm(arm,args,train,validation,device); payload["decision"]=decide(payload["arms"],valid); write_payload(args.output,payload)
        print(json.dumps({"finished_arm":arm,"loss":payload["arms"][arm]["evaluations"]["305"]["loss"]}),flush=True)
    print(json.dumps(payload["decision"],indent=2,sort_keys=True))


if __name__=="__main__": main()
