#!/usr/bin/env python3
"""Matched 10M screen for training-only RMS-gain factorization."""

from __future__ import annotations

import argparse, gc, json, math, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM

from experiments.reflex_swiglu_lm_screen import TokenFile, evaluate, lr_multiplier, paired_loss_interval, sha256_file, validate_data_ledger, write_payload
from experiments.train_time_gauge_scaffold_lm_screen import make_base_state

VALID_ARMS=("raw","raw_norm_lr2","factor_add","factor_multiply")
BOUND=.25
PREREGISTRATION=Path("results/factorized-rms-gain-lm-preregistration.md")


class FactorRMSNorm(nn.Module):
    def __init__(self,source,mode):
        super().__init__();self.mode=mode;self.gain=nn.Parameter(source.weight.detach().clone());self.factor=nn.Parameter(torch.zeros_like(source.weight));self.epsilon=source.variance_epsilon
    def effective(self):
        if self.mode=="add":return self.gain+self.factor
        return self.gain*(1+BOUND*torch.tanh(self.factor/BOUND))
    def forward(self,x):
        dtype=x.dtype;values=x.float();z=values*torch.rsqrt(values.square().mean(-1,keepdim=True)+self.epsilon)
        return self.effective()*z.to(dtype)


class ExportedRMSNorm(nn.Module):
    def __init__(self,source):
        super().__init__();self.weight=nn.Parameter(source.effective().detach().clone());self.variance_epsilon=source.epsilon
    def forward(self,x):
        dtype=x.dtype;values=x.float();z=values*torch.rsqrt(values.square().mean(-1,keepdim=True)+self.variance_epsilon)
        return self.weight*z.to(dtype)


def build(config,state,device,arm):
    model=AutoModelForCausalLM.from_config(config,attn_implementation="sdpa");model.load_state_dict(state,strict=True);modules=[]
    if arm.startswith("factor_"):
        mode="add" if arm=="factor_add" else "multiply"
        for layer in model.model.layers:
            for attribute in("input_layernorm","post_attention_layernorm"):
                replacement=FactorRMSNorm(getattr(layer,attribute),mode);setattr(layer,attribute,replacement);modules.append(replacement)
    model=model.to(device);model.config.use_cache=False;return model,modules


def loss(model,inputs,targets):
    with torch.autocast("cuda",dtype=torch.bfloat16):logits=model(input_ids=inputs,use_cache=False).logits
    return F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),targets.reshape(-1))


@torch.no_grad()
def collapse(model):
    count=0
    for layer in model.model.layers:
        for attribute in("input_layernorm","post_attention_layernorm"):
            source=getattr(layer,attribute)
            if isinstance(source,FactorRMSNorm):setattr(layer,attribute,ExportedRMSNorm(source).to(source.gain.device));count+=1
    return count


def train_arm(arm,args,config,state,train,validation,device):
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    model,modules=build(config,state,device,arm);training_count=sum(p.numel()for p in model.parameters())
    norm_lr2_ids=set()
    if arm=="raw_norm_lr2":
        for layer in model.model.layers:norm_lr2_ids|={id(layer.input_layernorm.weight),id(layer.post_attention_layernorm.weight)}
    decay=[];one_d=[];norm_lr2=[]
    for parameter in model.parameters():
        if id(parameter)in norm_lr2_ids:norm_lr2.append(parameter)
        elif parameter.ndim<2:one_d.append(parameter)
        else:decay.append(parameter)
    groups=[{"params":decay,"weight_decay":args.weight_decay,"lr":args.learning_rate},{"params":one_d,"weight_decay":0.,"lr":args.learning_rate}]
    if norm_lr2:groups.append({"params":norm_lr2,"weight_decay":0.,"lr":2*args.learning_rate})
    optimizer=torch.optim.AdamW(groups,betas=(.9,.95),eps=1e-8,fused=True)
    for group in optimizer.param_groups:group["base_lr"]=group["lr"]
    evaluations={"0":evaluate(model,validation,args.eval_batches,args.eval_batch_size,device)};optimizer.zero_grad(set_to_none=True);durations=[];losses=[];norms=[]
    for step in range(args.steps):
        model.train();multiplier=lr_multiplier(step,args.steps,args.warmup_steps)
        for group in optimizer.param_groups:group["lr"]=group["base_lr"]*multiplier
        started=time.perf_counter();accumulated=0.
        for micro in range(args.gradient_accumulation):
            inputs,targets=train.batch(step*args.gradient_accumulation+micro,args.micro_batch_size,device);value=loss(model,inputs,targets);(value/args.gradient_accumulation).backward();accumulated+=float(value.detach())/args.gradient_accumulation
        norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),args.gradient_clip))
        if not math.isfinite(accumulated)or not math.isfinite(norm):raise RuntimeError((arm,step))
        optimizer.step();optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();durations.append(time.perf_counter()-started);losses.append(accumulated);norms.append(norm)
    evaluations[str(args.steps)]=evaluate(model,validation,args.eval_batches,args.eval_batch_size,device)
    effective={}
    if modules:
        values=torch.cat([module.effective().detach().float() for module in modules]);factors=torch.cat([module.factor.detach().float() for module in modules])
        effective={"gain_mean":float(values.mean()),"gain_std":float(values.std()),"factor_abs_mean":float(factors.abs().mean()),"factor_abs_max":float(factors.abs().max())}
    pre=evaluations[str(args.steps)];collapsed=collapse(model);served_count=sum(p.numel()for p in model.parameters());post=evaluate(model,validation,args.eval_batches,args.eval_batch_size,device)if collapsed else pre
    result={"arm":arm,"training_parameters":training_count,"served_parameters":served_count,"collapsed_norms":collapsed,"evaluations":evaluations,"postcollapse":post,"collapse_nll_drift":post["loss"]-pre["loss"],"factor_diagnostics":effective,
            "train":{"mean_loss":float(np.mean(losses)),"final_loss":losses[-1],"max_loss":max(losses),"max_preclip_gradient_norm":max(norms),"elapsed_seconds":sum(durations),"nonfinite":False}}
    del optimizer,modules,model;gc.collect();torch.cuda.empty_cache();return result


def decide(results,valid):
    if set(results)!=set(VALID_ARMS):return{"complete":False,"missing":sorted(set(VALID_ARMS)-set(results))}
    step="305";losses={arm:r["evaluations"][step]["loss"]for arm,r in results.items()};controls=("raw","raw_norm_lr2","factor_add");best=min(controls,key=lambda arm:losses[arm]);candidate=results["factor_multiply"]
    interval=paired_loss_interval(candidate,results[best],step);initial=[r["evaluations"]["0"]["loss"]for r in results.values()]
    gates={"protocol_data_valid":valid,"initial_match_2e_5":max(initial)-min(initial)<=2e-5,"finite":all(not r["train"]["nonfinite"]for r in results.values()),
           "candidate_0p005_percent_better_all":all((losses[c]-losses["factor_multiply"])/losses[c]>=.00005 for c in controls),"paired_interval_favors_candidate":interval["upper_95"]<0,
           "served_counts_equal":len({r["served_parameters"]for r in results.values()})==1,"collapse_drift_at_most_2e_6":abs(candidate["collapse_nll_drift"])<=2e-6}
    return{"complete":True,"losses":losses,"best_control":best,"paired_interval":interval,"candidate_diagnostics":candidate["factor_diagnostics"],"gates":gates,"advance_to_50m":all(gates.values())}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--train-file",type=Path,default=Path("data/block-algebra-scratch/train.uint16.bin"));parser.add_argument("--validation-file",type=Path,default=Path("data/block-algebra-scratch/validation.uint16.bin"));parser.add_argument("--data-manifest",type=Path,default=Path("results/block-algebra-scratch-data-manifest.json"));parser.add_argument("--output",type=Path,default=Path("results/factorized-rms-gain-lm-screen.json"));parser.add_argument("--steps",type=int,default=305);parser.add_argument("--sequence-length",type=int,default=512);parser.add_argument("--micro-batch-size",type=int,default=32);parser.add_argument("--gradient-accumulation",type=int,default=2);parser.add_argument("--eval-batch-size",type=int,default=32);parser.add_argument("--eval-batches",type=int,default=128);parser.add_argument("--warmup-steps",type=int,default=30);parser.add_argument("--learning-rate",type=float,default=3e-4);parser.add_argument("--weight-decay",type=float,default=.1);parser.add_argument("--gradient-clip",type=float,default=1.);parser.add_argument("--seed",type=int,default=814);args=parser.parse_args()
    expected=(args.steps,args.sequence_length,args.micro_batch_size,args.gradient_accumulation,args.eval_batch_size,args.eval_batches,args.warmup_steps,args.learning_rate,args.weight_decay,args.seed)==(305,512,32,2,32,128,30,3e-4,.1,814)and torch.__version__=="2.5.1+cu124"and torch.version.cuda=="12.4"and transformers.__version__=="4.57.6"and"H100"in torch.cuda.get_device_name()
    train=TokenFile(args.train_file,args.sequence_length);validation=TokenFile(args.validation_file,args.sequence_length);ledger=validate_data_ledger(args.data_manifest,args.train_file,args.validation_file,args.sequence_length);config,state,state_hash=make_base_state(args.seed);torch.set_float32_matmul_precision("high");torch.backends.cuda.matmul.allow_tf32=True
    payload={"schema":"factorized-rms-gain-lm-v1","source_sha256":sha256_file(Path(__file__)),"preregistration_sha256":sha256_file(PREREGISTRATION),"base_state_sha256":state_hash,"protocol_valid":expected,"data_ledger":ledger,"arms":{}}
    for arm in VALID_ARMS:
        print(json.dumps({"starting_arm":arm}),flush=True);payload["arms"][arm]=train_arm(arm,args,config,state,train,validation,torch.device("cuda"));payload["decision"]=decide(payload["arms"],expected and ledger["valid"]);write_payload(args.output,payload);print(json.dumps({"finished_arm":arm,"loss":payload["arms"][arm]["evaluations"]["305"]["loss"]}),flush=True)
    print(json.dumps(payload["decision"],indent=2,sort_keys=True))
if __name__=="__main__":main()
