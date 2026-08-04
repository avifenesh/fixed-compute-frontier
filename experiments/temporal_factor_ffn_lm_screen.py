#!/usr/bin/env python3
"""Frozen temporal reuse successor to the rejected static cycle factor."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from experiments.coalesced_attention_ffn_lm_screen import build_model as build_parallel_model
from experiments.cycle_factor_ffn_lm_screen import D, M, SEED, SELF_SCALE, route_values
from experiments.reflex_swiglu_lm_screen import (
    TokenFile, evaluate, lr_multiplier, paired_loss_interval, sha256_file,
    validate_data_ledger, write_payload,
)
from experiments.triangular_microdepth_lm_screen import causal_loss


ANGLE_GROUP = 32
ANGLE_COUNT_PER_LAYER = M // ANGLE_GROUP
PREDECESSOR = Path("results/cycle-factor-ffn-lm-screen.json")
PREDECESSOR_SHA256 = "c66dfe6f5a6e3a1c4032932b9cbd2c491812af2e86a314e9cc6d46e83aba4029"
PREREGISTRATION = Path("results/temporal-factor-ffn-preregistration.md")


class TemporalFactorMLP(nn.Module):
    def __init__(self, std: float) -> None:
        super().__init__()
        self.generator_proj = nn.Linear(D, M, bias=False)
        self.down_proj = nn.Linear(M, D, bias=False)
        for projection in (self.generator_proj, self.down_proj):
            nn.init.normal_(projection.weight, mean=0.0, std=std)
        self.raw_angle = nn.Parameter(torch.zeros(ANGLE_COUNT_PER_LAYER))
        self.ablation = "full"
        self.endpoint_ste = True

    def angle(self) -> torch.Tensor:
        if self.ablation == "zero":
            return torch.zeros_like(self.raw_angle)
        return (math.pi / 2) * torch.tanh(self.raw_angle)

    def activation(self, hidden_states: torch.Tensor) -> torch.Tensor:
        generated = self.generator_proj(hidden_states)
        previous = torch.cat((generated[:, :1], generated[:, :-1]), dim=1)
        if self.ablation == "batch_misroute":
            previous = previous.roll(1, dims=0)
        routed = route_values(previous)
        angle = self.angle().repeat_interleave(ANGLE_GROUP).to(generated.dtype).view(1, 1, M)
        cosine, sine = torch.cos(angle), torch.sin(angle)
        value = cosine * generated + sine * routed
        scale = (
            SELF_SCALE
            / torch.sqrt(cosine.square() + (SELF_SCALE * sine).square())
        ).to(generated.dtype)
        activated = F.silu(generated)
        desired = scale * activated * value
        if self.endpoint_ste:
            endpoint = SELF_SCALE * activated * generated
            # Exact self-product forward at initialization, with candidate gradients.
            return endpoint.detach() + (desired - desired.detach())
        return desired

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.activation(hidden_states))


def build_candidate(device: torch.device, seed: int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model, _ = build_parallel_model(device, "parallel_baseline")
    modules = []
    for layer in model.model.layers:
        replacement = TemporalFactorMLP(model.config.initializer_range).to(device)
        layer.mlp = replacement; modules.append(replacement)
    model.config.use_cache = False
    return model, modules


def train(args, train_file, validation, device):
    model, modules = build_candidate(device, args.seed)
    angles = [module.raw_angle for module in modules]; angle_ids = {id(parameter) for parameter in angles}
    ordinary = [parameter for parameter in model.parameters() if id(parameter) not in angle_ids]
    optimizer = torch.optim.AdamW([
        {"params": ordinary, "weight_decay": args.weight_decay},
        {"params": angles, "weight_decay": 0.0},
    ], lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=True)
    for group in optimizer.param_groups: group["base_lr"] = args.learning_rate
    evaluations={"0":evaluate(model,validation,args.eval_batches,args.eval_batch_size,device)}
    optimizer.zero_grad(set_to_none=True); losses=[]; norms=[]; durations=[]
    for step in range(args.steps):
        model.train(); multiplier=lr_multiplier(step,args.steps,args.warmup_steps)
        for group in optimizer.param_groups: group["lr"] = group["base_lr"] * multiplier
        started=time.perf_counter(); accumulated=0.0
        for micro in range(args.gradient_accumulation):
            inputs,targets=train_file.batch(step*args.gradient_accumulation+micro,args.micro_batch_size,device)
            value=causal_loss(model,inputs,targets); (value/args.gradient_accumulation).backward(); accumulated+=float(value.detach())/args.gradient_accumulation
        norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),args.gradient_clip))
        if not math.isfinite(accumulated) or not math.isfinite(norm): raise RuntimeError(f"nonfinite step {step}")
        optimizer.step(); optimizer.zero_grad(set_to_none=True); torch.cuda.synchronize()
        if step == 0:
            for module in modules: module.endpoint_ste = False
        losses.append(accumulated); norms.append(norm); durations.append(time.perf_counter()-started)
    evaluations[str(args.steps)]=evaluate(model,validation,args.eval_batches,args.eval_batch_size,device)
    angles_tensor=torch.cat([module.angle().detach().float() for module in modules])
    ablations={}
    for mode in ("zero","batch_misroute"):
        for module in modules: module.ablation=mode
        ablations[mode]=evaluate(model,validation,args.eval_batches,args.eval_batch_size,device)
    for module in modules: module.ablation="full"
    result={"total_parameters":sum(p.numel() for p in model.parameters()),"ffn_dense_parameters":12*2*D*M,
            "angle_parameters":sum(p.numel() for p in angles),"decode_state_bytes_bf16":12*M*2,
            "evaluations":evaluations,"terminal_ablations":ablations,
            "angle_diagnostics":{"abs_mean":float(angles_tensor.abs().mean()),"abs_max":float(angles_tensor.abs().max()),
                                 "sin_abs_mean":float(torch.sin(angles_tensor).abs().mean()),"positive_fraction":float((angles_tensor>0).float().mean())},
            "train":{"prediction_tokens":args.steps*args.gradient_accumulation*args.micro_batch_size*args.sequence_length,
                     "mean_loss":float(np.mean(losses)),"final_loss":losses[-1],"max_gradient_norm":max(norms),
                     "elapsed_seconds":sum(durations),"tokens_per_second":args.steps*args.gradient_accumulation*args.micro_batch_size*args.sequence_length/sum(durations),"nonfinite":False}}
    del optimizer,modules,model;gc.collect();torch.cuda.empty_cache();return result


def decide(candidate, predecessor, valid):
    step="305"; controls=predecessor["arms"]
    losses={arm:controls[arm]["evaluations"][step]["loss"] for arm in ("full_swiglu","narrow_swiglu","self_product")}
    losses["temporal_factor"]=candidate["evaluations"][step]["loss"]
    wrapped={"evaluations":{step:candidate["evaluations"][step]}}
    comparisons={arm:paired_loss_interval(wrapped,controls[arm],step) for arm in losses if arm!="temporal_factor"}
    baseline=losses["full_swiglu"]; margin=.0005*baseline
    noninferior=(losses["temporal_factor"]-baseline)/baseline<=.0005 and comparisons["full_swiglu"]["upper_95"]<=margin
    beats_controls=all((losses[arm]-losses["temporal_factor"])/losses[arm]>=.0005 and comparisons[arm]["upper_95"]<0 for arm in ("narrow_swiglu","self_product"))
    full={"evaluations":{step:candidate["evaluations"][step]}}; zero={"evaluations":{step:candidate["terminal_ablations"]["zero"]}}
    zero_interval=paired_loss_interval(full,zero,step); zero_hurts=(candidate["terminal_ablations"]["zero"]["loss"]-losses["temporal_factor"])/losses["temporal_factor"]>=.0001 and zero_interval["upper_95"]<0
    initial_delta=candidate["evaluations"]["0"]["loss"]-controls["self_product"]["evaluations"]["0"]["loss"]
    gates={"protocol_integrity_valid":valid,"initial_matches_self_within_2e_5":abs(initial_delta)<=2e-5,
           "candidate_smaller_than_full":candidate["total_parameters"]<controls["full_swiglu"]["total_parameters"],
           "dense_ffn_is_two_thirds_full":3*candidate["ffn_dense_parameters"]==2*controls["full_swiglu"]["ffn_parameters"],
           "noninferior_to_full":noninferior,"beats_both_equal_2dm_controls":beats_controls,
           "learned_temporal_angle":candidate["angle_diagnostics"]["sin_abs_mean"]>=.001,"zero_angle_ablation_hurts":zero_hurts,"finite":not candidate["train"]["nonfinite"]}
    return {"losses":losses,"relative_improvements":{arm:(losses[arm]-losses["temporal_factor"])/losses[arm] for arm in losses if arm!="temporal_factor"},
            "paired_intervals":comparisons,"zero_ablation_interval":zero_interval,"initial_delta_vs_self":initial_delta,
            "gates":gates,"advance_to_50m":all(gates.values())}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--train-file",type=Path,default=Path("data/block-algebra-scratch/train.uint16.bin"));parser.add_argument("--validation-file",type=Path,default=Path("data/block-algebra-scratch/validation.uint16.bin"));parser.add_argument("--data-manifest",type=Path,default=Path("results/block-algebra-scratch-data-manifest.json"));parser.add_argument("--output",type=Path,default=Path("results/temporal-factor-ffn-lm-screen.json"));parser.add_argument("--steps",type=int,default=305);parser.add_argument("--sequence-length",type=int,default=512);parser.add_argument("--micro-batch-size",type=int,default=32);parser.add_argument("--gradient-accumulation",type=int,default=2);parser.add_argument("--eval-batch-size",type=int,default=32);parser.add_argument("--eval-batches",type=int,default=128);parser.add_argument("--warmup-steps",type=int,default=30);parser.add_argument("--learning-rate",type=float,default=3e-4);parser.add_argument("--weight-decay",type=float,default=.1);parser.add_argument("--gradient-clip",type=float,default=1.);parser.add_argument("--seed",type=int,default=SEED);args=parser.parse_args()
    expected=(args.steps,args.sequence_length,args.micro_batch_size,args.gradient_accumulation,args.eval_batch_size,args.eval_batches,args.warmup_steps,args.learning_rate,args.weight_decay,args.gradient_clip,args.seed)==(305,512,32,2,32,128,30,3e-4,.1,1.,SEED) and torch.__version__=="2.5.1+cu124" and torch.version.cuda=="12.4" and transformers.__version__=="4.57.6" and "H100" in torch.cuda.get_device_name()
    if sha256_file(PREDECESSOR)!=PREDECESSOR_SHA256: raise RuntimeError("frozen predecessor changed")
    predecessor=json.loads(PREDECESSOR.read_text());ledger=validate_data_ledger(args.data_manifest,args.train_file,args.validation_file,args.sequence_length);valid=expected and predecessor["protocol_valid"] and ledger["valid"]
    train_file=TokenFile(args.train_file,args.sequence_length);validation=TokenFile(args.validation_file,args.sequence_length);torch.set_float32_matmul_precision("high");torch.backends.cuda.matmul.allow_tf32=True
    candidate=train(args,train_file,validation,torch.device("cuda"));decision=decide(candidate,predecessor,valid)
    payload={"schema":"temporal-factor-ffn-lm-v1","source_sha256":sha256_file(Path(__file__)),"preregistration_sha256":sha256_file(PREREGISTRATION),"predecessor_sha256":sha256_file(PREDECESSOR),"protocol_valid":valid,"candidate":candidate,"decision":decision}
    write_payload(args.output,payload);print(json.dumps(decision,indent=2,sort_keys=True))


if __name__=="__main__":main()
