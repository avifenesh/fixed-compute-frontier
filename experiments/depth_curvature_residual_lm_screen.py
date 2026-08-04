#!/usr/bin/env python3
"""Matched screen for reusing the previous layer's raw residual update."""

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
import transformers

from experiments.coalesced_attention_ffn_lm_screen import build_model as build_parallel_model
from experiments.cycle_factor_ffn_lm_screen import D, SEED
from experiments.reflex_swiglu_lm_screen import TokenFile, evaluate, lr_multiplier, paired_loss_interval, sha256_file, validate_data_ledger, write_payload
from experiments.triangular_microdepth_lm_screen import causal_loss


ARMS=("residual_scale","depth_curvature")
GROUP=32
PREDECESSOR=Path("results/cycle-factor-ffn-lm-screen.json")
PREDECESSOR_SHA256="c66dfe6f5a6e3a1c4032932b9cbd2c491812af2e86a314e9cc6d46e83aba4029"
PREREGISTRATION=Path("results/depth-curvature-residual-preregistration.md")


class UpdateBus:
    def __init__(self): self.previous=None
    def reset(self): self.previous=None


class ResidualWrapper(nn.Module):
    def __init__(self,source:nn.Module,bus:UpdateBus,arm:str,index:int):
        super().__init__();self.source=source;self.bus=bus;self.arm=arm;self.index=index
        self.raw_beta=nn.Parameter(torch.zeros(D//GROUP));self.ablate=False
    def beta(self,dtype):
        values=torch.zeros_like(self.raw_beta) if self.ablate else .5*torch.tanh(self.raw_beta)
        return values.repeat_interleave(GROUP).to(dtype).view(1,1,D)
    def forward(self,hidden_states:torch.Tensor,**kwargs):
        ordinary=self.source(hidden_states,**kwargs);delta=ordinary-hidden_states;previous=self.bus.previous
        if self.arm=="depth_curvature" and previous is not None:
            output=ordinary+self.beta(ordinary.dtype)*(delta-previous)
        elif self.arm=="residual_scale":
            output=ordinary+self.beta(ordinary.dtype)*delta
        else: output=ordinary
        self.bus.previous=delta
        return output


def build_arm(device,arm,seed):
    import random
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model,_=build_parallel_model(device,"parallel_baseline");bus=UpdateBus();modules=[]
    for index,source in enumerate(model.model.layers):
        wrapper=ResidualWrapper(source,bus,arm,index).to(device);model.model.layers[index]=wrapper;modules.append(wrapper)
    model.register_forward_pre_hook(lambda _module,_args:bus.reset());model.config.use_cache=False
    return model,modules


def train_arm(arm,args,train,validation,device):
    model,modules=build_arm(device,arm,args.seed);betas=[module.raw_beta for module in modules];ids={id(x)for x in betas};ordinary=[x for x in model.parameters()if id(x)not in ids]
    optimizer=torch.optim.AdamW([{"params":ordinary,"weight_decay":args.weight_decay},{"params":betas,"weight_decay":0.}],lr=args.learning_rate,betas=(.9,.95),eps=1e-8,fused=True)
    for group in optimizer.param_groups:group["base_lr"]=args.learning_rate
    evaluations={"0":evaluate(model,validation,args.eval_batches,args.eval_batch_size,device)};optimizer.zero_grad(set_to_none=True);losses=[];norms=[];durations=[]
    for step in range(args.steps):
        model.train();mult=lr_multiplier(step,args.steps,args.warmup_steps)
        for group in optimizer.param_groups:group["lr"]=group["base_lr"]*mult
        started=time.perf_counter();accumulated=0.
        for micro in range(args.gradient_accumulation):
            inputs,targets=train.batch(step*args.gradient_accumulation+micro,args.micro_batch_size,device);value=causal_loss(model,inputs,targets);(value/args.gradient_accumulation).backward();accumulated+=float(value.detach())/args.gradient_accumulation
        norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),args.gradient_clip))
        if not math.isfinite(accumulated)or not math.isfinite(norm):raise RuntimeError(f"nonfinite {arm} {step}")
        optimizer.step();optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();losses.append(accumulated);norms.append(norm);durations.append(time.perf_counter()-started)
    evaluations[str(args.steps)]=evaluate(model,validation,args.eval_batches,args.eval_batch_size,device)
    beta=torch.cat([(.5*torch.tanh(module.raw_beta)).detach().float()for module in modules]);
    for module in modules:module.ablate=True
    ablated=evaluate(model,validation,args.eval_batches,args.eval_batch_size,device)
    result={"arm":arm,"total_parameters":sum(p.numel()for p in model.parameters()),"dense_parameters":sum(p.numel()for p in model.parameters()if p.ndim>=2),"beta_parameters":sum(p.numel()for p in betas),"evaluations":evaluations,"zero_ablation":ablated,"beta_diagnostics":{"abs_mean":float(beta.abs().mean()),"abs_max":float(beta.abs().max()),"positive_fraction":float((beta>0).float().mean())},"train":{"prediction_tokens":args.steps*args.gradient_accumulation*args.micro_batch_size*args.sequence_length,"elapsed_seconds":sum(durations),"tokens_per_second":args.steps*args.gradient_accumulation*args.micro_batch_size*args.sequence_length/sum(durations),"mean_loss":float(np.mean(losses)),"final_loss":losses[-1],"max_gradient_norm":max(norms),"nonfinite":False}}
    del optimizer,modules,model;gc.collect();torch.cuda.empty_cache();return result


def decide(results,predecessor,valid):
    if set(results)!=set(ARMS):return{"complete":False,"missing":sorted(set(ARMS)-set(results))}
    step="305";baseline=predecessor["arms"]["full_swiglu"];losses={"full_swiglu":baseline["evaluations"][step]["loss"],**{arm:results[arm]["evaluations"][step]["loss"]for arm in ARMS}}
    candidate=results["depth_curvature"];comparisons={"full_swiglu":paired_loss_interval(candidate,baseline,step),"residual_scale":paired_loss_interval(candidate,results["residual_scale"],step)}
    full={"evaluations":{step:candidate["evaluations"][step]}};zero={"evaluations":{step:candidate["zero_ablation"]}};zero_interval=paired_loss_interval(full,zero,step)
    gates={"protocol_valid":valid,"initial_exact":max(abs(results[arm]["evaluations"]["0"]["loss"]-baseline["evaluations"]["0"]["loss"])for arm in ARMS)<=1e-7,"dense_parameters_unchanged":all(results[arm]["dense_parameters"]==results["residual_scale"]["dense_parameters"]for arm in ARMS),"candidate_beats_full_0p05":(losses["full_swiglu"]-losses["depth_curvature"])/losses["full_swiglu"]>=.0005 and comparisons["full_swiglu"]["upper_95"]<0,"candidate_beats_scale_0p025":(losses["residual_scale"]-losses["depth_curvature"])/losses["residual_scale"]>=.00025 and comparisons["residual_scale"]["upper_95"]<0,"beta_is_live":candidate["beta_diagnostics"]["abs_mean"]>=.001,"zero_ablation_hurts_0p01":(candidate["zero_ablation"]["loss"]-losses["depth_curvature"])/losses["depth_curvature"]>=.0001 and zero_interval["upper_95"]<0,"finite":all(not x["train"]["nonfinite"]for x in results.values())}
    return{"complete":True,"losses":losses,"relative_candidate_improvements":{"full_swiglu":(losses["full_swiglu"]-losses["depth_curvature"])/losses["full_swiglu"],"residual_scale":(losses["residual_scale"]-losses["depth_curvature"])/losses["residual_scale"]},"paired_intervals":comparisons,"zero_ablation_interval":zero_interval,"gates":gates,"advance_to_50m":all(gates.values())}


def main():
    p=argparse.ArgumentParser();p.add_argument("--train-file",type=Path,default=Path("data/block-algebra-scratch/train.uint16.bin"));p.add_argument("--validation-file",type=Path,default=Path("data/block-algebra-scratch/validation.uint16.bin"));p.add_argument("--data-manifest",type=Path,default=Path("results/block-algebra-scratch-data-manifest.json"));p.add_argument("--output",type=Path,default=Path("results/depth-curvature-residual-lm-screen.json"));p.add_argument("--steps",type=int,default=305);p.add_argument("--sequence-length",type=int,default=512);p.add_argument("--micro-batch-size",type=int,default=32);p.add_argument("--gradient-accumulation",type=int,default=2);p.add_argument("--eval-batch-size",type=int,default=32);p.add_argument("--eval-batches",type=int,default=128);p.add_argument("--warmup-steps",type=int,default=30);p.add_argument("--learning-rate",type=float,default=3e-4);p.add_argument("--weight-decay",type=float,default=.1);p.add_argument("--gradient-clip",type=float,default=1.);p.add_argument("--seed",type=int,default=SEED);args=p.parse_args()
    expected=(args.steps,args.sequence_length,args.micro_batch_size,args.gradient_accumulation,args.eval_batch_size,args.eval_batches,args.warmup_steps,args.learning_rate,args.weight_decay,args.gradient_clip,args.seed)==(305,512,32,2,32,128,30,3e-4,.1,1.,SEED)and torch.__version__=="2.5.1+cu124"and torch.version.cuda=="12.4"and transformers.__version__=="4.57.6"and"H100"in torch.cuda.get_device_name()
    if sha256_file(PREDECESSOR)!=PREDECESSOR_SHA256:raise RuntimeError("predecessor changed")
    predecessor=json.loads(PREDECESSOR.read_text());ledger=validate_data_ledger(args.data_manifest,args.train_file,args.validation_file,args.sequence_length);valid=expected and predecessor["protocol_valid"]and ledger["valid"];train=TokenFile(args.train_file,args.sequence_length);validation=TokenFile(args.validation_file,args.sequence_length);device=torch.device("cuda");torch.set_float32_matmul_precision("high");torch.backends.cuda.matmul.allow_tf32=True
    payload={"schema":"depth-curvature-residual-v1","source_sha256":sha256_file(Path(__file__)),"preregistration_sha256":sha256_file(PREREGISTRATION),"predecessor_sha256":sha256_file(PREDECESSOR),"protocol_valid":valid,"arms":{}}
    for arm in ARMS:print(json.dumps({"starting_arm":arm}),flush=True);payload["arms"][arm]=train_arm(arm,args,train,validation,device);payload["decision"]=decide(payload["arms"],predecessor,valid);write_payload(args.output,payload);print(json.dumps({"finished_arm":arm,"loss":payload["arms"][arm]["evaluations"]["305"]["loss"]}),flush=True)
    print(json.dumps(payload["decision"],indent=2,sort_keys=True))
if __name__=="__main__":main()
