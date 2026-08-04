#!/usr/bin/env python3
"""50M-token replication of training-scaffolded, served ShiftNorm."""

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
from transformers import AutoModelForCausalLM

from experiments.reflex_swiglu_lm_screen import (
    MODEL, MODEL_REVISION, TokenFile, evaluate, lr_multiplier,
    paired_loss_interval, sha256_file, validate_data_ledger, write_payload,
)
from experiments.train_time_gauge_scaffold_lm_screen import (
    BOUND, ScaffoldRMSNorm, diagnostics as scaffold_diagnostics,
    make_base_state,
)


VALID_ARMS = ("raw_baseline", "direct_shift", "scaffold_scale", "scaffold_bias")
PREREGISTRATION = Path("results/folded-gain-shiftnorm-lm-replication-preregistration.md")


class DirectShiftRMSNorm(nn.Module):
    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        half = source.weight.numel() // 2
        self.theta_alpha = nn.Parameter(torch.zeros(half))
        self.theta_beta = nn.Parameter(torch.zeros(half))
        self.epsilon = source.variance_epsilon
        self.ablation_mode = "full"
        self.record_diagnostics = False
        self.last_diagnostics = None

    def gamma(self, dtype):
        theta = torch.cat((self.theta_alpha, self.theta_beta))
        return (BOUND * torch.tanh(theta / BOUND)).to(dtype)

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        values = hidden_states.float()
        z = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.epsilon)
        z = z.to(input_dtype)
        gamma = self.gamma(z.dtype)
        output = z if self.ablation_mode == "zero" else z + gamma
        if self.record_diagnostics:
            with torch.no_grad():
                self.last_diagnostics = {
                    "gamma_abs_mean": float(gamma.float().abs().mean()),
                    "gamma_abs_max": float(gamma.float().abs().max()),
                    "input_rms": float(z.float().square().mean().sqrt()),
                    "output_rms": float(output.float().square().mean().sqrt()),
                    "output_mean": float(output.float().mean()),
                }
        return output


def build_model(config, state, device, arm):
    model = AutoModelForCausalLM.from_config(config, attn_implementation="sdpa")
    model.load_state_dict(state, strict=True)
    modules = []
    if arm != "raw_baseline":
        for layer in model.model.layers:
            for attribute in ("input_layernorm", "post_attention_layernorm"):
                source = getattr(layer, attribute)
                if arm == "direct_shift":
                    replacement = DirectShiftRMSNorm(source)
                else:
                    replacement = ScaffoldRMSNorm(source, arm)
                setattr(layer, attribute, replacement)
                modules.append(replacement)
    model = model.to(device); model.config.use_cache = False
    return model, modules


@torch.no_grad()
def diagnostics(model, modules, validation_file, device):
    if not modules: return {"raw_baseline": True}
    if isinstance(modules[0], ScaffoldRMSNorm):
        return scaffold_diagnostics(model, modules, validation_file, device)
    model.eval(); inputs, _ = validation_file.batch(0, 2, device)
    for module in modules: module.record_diagnostics = True
    with torch.autocast("cuda", dtype=torch.bfloat16): model(input_ids=inputs, use_cache=False)
    records = [module.last_diagnostics for module in modules]
    for module in modules: module.record_diagnostics = False
    return {key + "_layer_median": float(np.median([record[key] for record in records])) for key in records[0]}


def causal_loss(model, inputs, targets):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=inputs, use_cache=False).logits
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))


@torch.no_grad()
def export_check(model, modules, validation_file, device):
    if not modules or not isinstance(modules[0], ScaffoldRMSNorm): return {"applicable": False}
    inputs, _ = validation_file.batch(0, 2, device)
    layer = model.model.layers[0]
    norm = modules[0]
    values = inputs[:, :1].float()
    # Use actual hidden-width activations from embedding, not token ids.
    hidden = model.model.embed_tokens(inputs[:, :1])
    raw = hidden.float(); z = raw * torch.rsqrt(raw.square().mean(-1, keepdim=True) + norm.epsilon)
    gamma = norm.gamma(z.dtype)
    charted = norm.chart(z, gamma)
    errors = {}
    for name in ("q_proj", "k_proj", "v_proj"):
        weight = getattr(layer.self_attn, name).weight.float()
        training = F.linear(norm.weight.float() * charted, weight)
        exported = weight * norm.weight.float()[None, :]
        served = F.linear(charted, exported)
        errors[name] = float((training - served).abs().max())
    return {"applicable": True, "max_abs_errors": errors, "max_abs_error": max(errors.values())}


def train_arm(arm, args, config, state, train_file, validation_file, device):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    model, modules = build_model(config, state, device, arm)
    total = sum(p.numel() for p in model.parameters())
    removed = sum(module.weight.numel() for module in modules if isinstance(module, ScaffoldRMSNorm))
    served = total - removed
    no_decay = [p for p in model.parameters() if p.ndim < 2]; ids = {id(p) for p in no_decay}
    decay = [p for p in model.parameters() if id(p) not in ids]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=True,
    )
    for group in optimizer.param_groups: group["base_lr"] = args.learning_rate
    evaluations = {"0": evaluate(model, validation_file, args.eval_batches, args.eval_batch_size, device)}
    chart_diagnostics = {"initial": diagnostics(model, modules, validation_file, device)}
    torch.cuda.reset_peak_memory_stats(); optimizer.zero_grad(set_to_none=True)
    losses=[]; durations=[]; norms=[]
    for step in range(args.steps):
        model.train(); multiplier = lr_multiplier(step, args.steps, args.warmup_steps)
        for group in optimizer.param_groups: group["lr"] = group["base_lr"] * multiplier
        started=time.perf_counter(); accumulated=0.0
        for micro in range(args.gradient_accumulation):
            index=step*args.gradient_accumulation+micro
            inputs,targets=train_file.batch(index,args.micro_batch_size,device)
            loss=causal_loss(model,inputs,targets); (loss/args.gradient_accumulation).backward()
            accumulated += float(loss.detach())/args.gradient_accumulation
        norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),args.gradient_clip))
        if not math.isfinite(accumulated) or not math.isfinite(norm): raise RuntimeError((arm,step+1))
        optimizer.step(); optimizer.zero_grad(set_to_none=True); torch.cuda.synchronize()
        losses.append(accumulated); norms.append(norm); durations.append(time.perf_counter()-started)
        if step+1 in args.eval_steps:
            evaluations[str(step+1)]=evaluate(model,validation_file,args.eval_batches,args.eval_batch_size,device)
            print(json.dumps({"arm":arm,"step":step+1,"validation_loss":evaluations[str(step+1)]["loss"]}),flush=True)
    chart_diagnostics["terminal"]=diagnostics(model,modules,validation_file,device)
    ablations={}
    if arm in {"direct_shift","scaffold_bias"}:
        for module in modules: module.ablation_mode="zero"
        ablations["zero"]=evaluate(model,validation_file,args.eval_batches,args.eval_batch_size,device)
        for module in modules: module.ablation_mode="full"
    export = export_check(model,modules,validation_file,device)
    prediction_tokens=args.steps*args.gradient_accumulation*args.micro_batch_size*args.sequence_length
    result={"arm":arm,"training_parameters":total,"served_parameters_after_export":served,
            "scaffold_parameters_removed":removed,"evaluations":evaluations,"diagnostics":chart_diagnostics,
            "terminal_ablations":ablations,"export_check":export,
            "train":{"prediction_tokens":prediction_tokens,"mean_loss":float(np.mean(losses)),"final_loss":losses[-1],
                     "max_loss":max(losses),"max_preclip_gradient_norm":max(norms),"elapsed_seconds":sum(durations),
                     "tokens_per_second":prediction_tokens/sum(durations),"peak_allocated_bytes":torch.cuda.max_memory_allocated(),"nonfinite":False}}
    del optimizer,modules,model;gc.collect();torch.cuda.empty_cache();return result


def decide(results,valid):
    if set(results)!=set(VALID_ARMS):return{"complete":False,"missing":sorted(set(VALID_ARMS)-set(results))}
    candidate=results["scaffold_bias"]; controls=("raw_baseline","direct_shift","scaffold_scale")
    losses={step:{arm:r["evaluations"][step]["loss"] for arm,r in results.items()} for step in ("305","1525")}
    best_control=min(("direct_shift","scaffold_scale"),key=lambda arm:losses["1525"][arm])
    intervals={}
    for control in ("raw_baseline",best_control):
        intervals["bias_vs_"+control]=paired_loss_interval(candidate,results[control],"1525")
    full={"evaluations":{"1525":candidate["evaluations"]["1525"]}}
    intervals["full_vs_zero"]=paired_loss_interval(full,{"evaluations":{"1525":candidate["terminal_ablations"]["zero"]}},"1525")
    initial=[r["evaluations"]["0"]["loss"] for r in results.values()]; diag=candidate["diagnostics"]["terminal"]
    full_loss=losses["1525"]["scaffold_bias"];zero_loss=candidate["terminal_ablations"]["zero"]["loss"]
    gates={"protocol_and_data_valid":valid,"initial_losses_match_2e_5":max(initial)-min(initial)<=2e-5,
           "finite_training":all(not r["train"]["nonfinite"] for r in results.values()),
           "exported_served_counts_equal":len({r["served_parameters_after_export"] for r in results.values()})==1,
           "candidate_no_worse_all_at_10m":all(losses["305"]["scaffold_bias"]<=losses["305"][arm] for arm in controls),
           "candidate_0p01_percent_better_raw_50m":(losses["1525"]["raw_baseline"]-full_loss)/losses["1525"]["raw_baseline"]>=0.0001,
           "candidate_0p01_percent_better_best_control_50m":(losses["1525"][best_control]-full_loss)/losses["1525"][best_control]>=0.0001,
           "paired_intervals_favor_candidate":all(value["upper_95"]<0 for key,value in intervals.items() if key.startswith("bias_vs_")),
           "shift_live":diag["gamma_abs_mean_layer_median"]>=0.002,"shift_bounded":diag["gamma_abs_max_layer_median"]<BOUND,
           "rms_within_5_percent":0.95<=diag["output_rms_layer_median"]/diag["input_rms_layer_median"]<=1.05,
           "zero_ablation_hurts_0p01_percent":(zero_loss-full_loss)/full_loss>=0.0001,
           "zero_interval_favors_full":intervals["full_vs_zero"]["upper_95"]<0,
           "exact_export":candidate["export_check"]["max_abs_error"]<=2e-5}
    return{"complete":True,"losses":losses,"best_control":best_control,"paired_intervals":intervals,
           "candidate_diagnostics":diag,"candidate_zero_loss":zero_loss,"export_check":candidate["export_check"],
           "gates":gates,"replicated":all(gates.values())}


def protocol(args):
    expected={"device":"NVIDIA H100 80GB HBM3","torch":"2.5.1+cu124","cuda":"12.4","transformers":"4.57.6",
              "steps":1525,"eval_steps":[305,1525],"sequence_length":512,"micro_batch_size":32,"gradient_accumulation":2,
              "eval_batch_size":32,"eval_batches":128,"warmup_steps":100,"learning_rate":3e-4,"weight_decay":0.1,"seed":813}
    actual={"device":torch.cuda.get_device_name(),"torch":torch.__version__,"cuda":torch.version.cuda,"transformers":transformers.__version__,
            **{key:getattr(args,key) for key in expected if key not in {"device","torch","cuda","transformers"}}}
    checks={key:actual[key]==value for key,value in expected.items()}
    if args.strict_protocol and not all(checks.values()):raise ValueError({k:{"expected":expected[k],"actual":actual[k]}for k,v in checks.items()if not v})
    return{"valid":all(checks.values()),"checks":checks,"expected":expected,"actual":actual}


def parse_steps(text):return[int(value)for value in text.split(",")if value]
def run(args):
    p=protocol(args);train=TokenFile(args.train_file,args.sequence_length);validation=TokenFile(args.validation_file,args.sequence_length)
    ledger=validate_data_ledger(args.data_manifest,args.train_file,args.validation_file,args.sequence_length)
    config,state,state_hash=make_base_state(args.seed);torch.set_float32_matmul_precision("high");torch.backends.cuda.matmul.allow_tf32=True
    payload={"schema":"folded-gain-shiftnorm-lm-replication-v1","source_sha256":sha256_file(Path(__file__)),
             "preregistration_sha256":sha256_file(PREREGISTRATION),"base_state_sha256":state_hash,"data_ledger":ledger,
             "protocol":p,"model":MODEL,"model_revision":MODEL_REVISION,"arms":{}}
    for arm in VALID_ARMS:
        print(json.dumps({"starting_arm":arm}),flush=True)
        payload["arms"][arm]=train_arm(arm,args,config,state,train,validation,torch.device("cuda"))
        payload["decision"]=decide(payload["arms"],p["valid"]and ledger["valid"]);write_payload(args.output,payload)
    return payload
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--train-file",type=Path,default=Path("data/block-algebra-scratch/train.uint16.bin"));parser.add_argument("--validation-file",type=Path,default=Path("data/block-algebra-scratch/validation.uint16.bin"));parser.add_argument("--data-manifest",type=Path,default=Path("results/block-algebra-scratch-data-manifest.json"));parser.add_argument("--output",type=Path,default=Path("results/folded-gain-shiftnorm-lm-replication.json"));parser.add_argument("--steps",type=int,default=1525);parser.add_argument("--eval-steps",type=parse_steps,default=parse_steps("305,1525"));parser.add_argument("--sequence-length",type=int,default=512);parser.add_argument("--micro-batch-size",type=int,default=32);parser.add_argument("--gradient-accumulation",type=int,default=2);parser.add_argument("--eval-batch-size",type=int,default=32);parser.add_argument("--eval-batches",type=int,default=128);parser.add_argument("--warmup-steps",type=int,default=100);parser.add_argument("--learning-rate",type=float,default=3e-4);parser.add_argument("--weight-decay",type=float,default=0.1);parser.add_argument("--gradient-clip",type=float,default=1.0);parser.add_argument("--seed",type=int,default=813);parser.add_argument("--strict-protocol",action=argparse.BooleanOptionalAction,default=True)
    args=parser.parse_args();payload=run(args);print(json.dumps(payload["decision"],indent=2,sort_keys=True))
if __name__=="__main__":main()
