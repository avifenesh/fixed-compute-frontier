#!/usr/bin/env python3
"""Fresh-seed 50M replication of phase-elastic residual-precision FFN v2."""

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
import transformers

from experiments.coalesced_attention_ffn_lm_screen import build_model as build_parallel_model
from experiments.phase_elastic_residual_precision_v2_lm_screen import (
    build_candidate, evaluate_mode, serving_ledger, set_mode,
)
from experiments.reflex_swiglu_lm_screen import (
    TokenFile, evaluate, lr_multiplier, paired_loss_interval, sha256_file,
    validate_data_ledger, write_payload,
)
from experiments.triangular_microdepth_lm_screen import causal_loss


SEED = 3719
STEPS = 1525
EVAL_STEPS = {305, 1525}
DISCOVERY = Path("results/phase-elastic-residual-precision-v2-lm-screen.json")
DISCOVERY_SHA256 = "c1847a55946202e58ba1c52c66c54c4e5ccfb775d04b355af0e940991cafb6ce"
PREREGISTRATION = Path("results/phase-elastic-v2-50m-replication-preregistration.md")


def configure_optimizer(model, args, quantized: bool):
    if not quantized:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95),
            eps=1e-8, weight_decay=args.weight_decay, fused=True,
        )
    else:
        scales = [parameter for name, parameter in model.named_parameters() if "log_scale" in name]
        scale_ids = {id(parameter) for parameter in scales}
        ordinary = [parameter for parameter in model.parameters() if id(parameter) not in scale_ids]
        optimizer = torch.optim.AdamW([
            {"params": ordinary, "weight_decay": args.weight_decay},
            {"params": scales, "weight_decay": 0.0},
        ], lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=True)
    return optimizer


def seed_everything(seed: int) -> None:
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


def train_dense(args, train_file, validation, device):
    seed_everything(args.seed)
    model,_=build_parallel_model(device,"parallel_baseline");model.config.use_cache=False
    optimizer=configure_optimizer(model,args,False)
    evaluations={"0":evaluate(model,validation,args.eval_batches,args.eval_batch_size,device)}
    optimizer.zero_grad(set_to_none=True);losses=[];norms=[];durations=[]
    for step in range(args.steps):
        model.train();scheduled=args.learning_rate*lr_multiplier(step,args.steps,args.warmup_steps)
        for group in optimizer.param_groups:group["lr"]=scheduled
        started=time.perf_counter();accumulated=0.0
        for micro in range(args.gradient_accumulation):
            inputs,targets=train_file.batch(step*args.gradient_accumulation+micro,args.micro_batch_size,device)
            loss=causal_loss(model,inputs,targets);(loss/args.gradient_accumulation).backward();accumulated+=float(loss.detach())/args.gradient_accumulation
        norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),args.gradient_clip))
        if not math.isfinite(accumulated) or not math.isfinite(norm):raise RuntimeError(f"nonfinite dense step {step+1}")
        optimizer.step();optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize()
        losses.append(accumulated);norms.append(norm);durations.append(time.perf_counter()-started)
        if step+1 in EVAL_STEPS:
            evaluations[str(step+1)]=evaluate(model,validation,args.eval_batches,args.eval_batch_size,device)
            print(json.dumps({"arm":"dense_bf16","step":step+1,"loss":evaluations[str(step+1)]["loss"]}),flush=True)
    result={"evaluations":evaluations,"model_parameters":sum(p.numel() for p in model.parameters()),
            "train":{"prediction_tokens":args.steps*args.gradient_accumulation*args.micro_batch_size*args.sequence_length,"mean_loss":float(np.mean(losses)),"final_loss":losses[-1],"max_loss":max(losses),"max_preclip_gradient_norm":max(norms),"elapsed_seconds":sum(durations),"tokens_per_second":args.steps*args.gradient_accumulation*args.micro_batch_size*args.sequence_length/sum(durations),"nonfinite":False}}
    del optimizer,model;gc.collect();torch.cuda.empty_cache();return result


def train_candidate(args,train_file,validation,device):
    model,modules=build_candidate(device,args.seed);optimizer=configure_optimizer(model,args,True)
    evaluations={"0":{"base":evaluate_mode(model,modules,"base",validation,args.eval_batches,args.eval_batch_size,device),"full":evaluate_mode(model,modules,"full",validation,args.eval_batches,args.eval_batch_size,device)}}
    optimizer.zero_grad(set_to_none=True);losses=[];norms=[];durations=[];active=np.zeros(len(modules),dtype=np.int64)
    for step in range(args.steps):
        model.train();set_mode(modules,"train");scheduled=args.learning_rate*lr_multiplier(step,args.steps,args.warmup_steps)
        for group in optimizer.param_groups:group["lr"]=scheduled
        started=time.perf_counter();accumulated=0.0
        for micro in range(args.gradient_accumulation):
            inputs,targets=train_file.batch(step*args.gradient_accumulation+micro,args.micro_batch_size,device)
            loss=causal_loss(model,inputs,targets);(loss/args.gradient_accumulation).backward();accumulated+=float(loss.detach())/args.gradient_accumulation
            active+=np.asarray([module.last_branch_active for module in modules],dtype=np.int64)
        norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),args.gradient_clip))
        if not math.isfinite(accumulated) or not math.isfinite(norm):raise RuntimeError(f"nonfinite candidate step {step+1}")
        optimizer.step();optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize()
        losses.append(accumulated);norms.append(norm);durations.append(time.perf_counter()-started)
        if step+1 in EVAL_STEPS:
            evaluations[str(step+1)]={"base":evaluate_mode(model,modules,"base",validation,args.eval_batches,args.eval_batch_size,device),"full":evaluate_mode(model,modules,"full",validation,args.eval_batches,args.eval_batch_size,device)}
            print(json.dumps({"arm":"phase_elastic_v2","step":step+1,"base_loss":evaluations[str(step+1)]["base"]["loss"],"full_loss":evaluations[str(step+1)]["full"]["loss"]}),flush=True)
    result={"evaluations":evaluations,"training_shadow_parameters":sum(p.numel() for p in model.parameters()),
            "branch_training_active_fraction_per_layer":(active/(args.steps*args.gradient_accumulation)).tolist(),"quantized_weight_audits":[module.audit() for module in modules],
            "train":{"prediction_tokens":args.steps*args.gradient_accumulation*args.micro_batch_size*args.sequence_length,"mean_loss":float(np.mean(losses)),"final_loss":losses[-1],"max_loss":max(losses),"max_preclip_gradient_norm":max(norms),"elapsed_seconds":sum(durations),"tokens_per_second":args.steps*args.gradient_accumulation*args.micro_batch_size*args.sequence_length/sum(durations),"nonfinite":False}}
    del optimizer,modules,model;gc.collect();torch.cuda.empty_cache();return result


def compare(candidate_eval,reference_eval,step):
    return paired_loss_interval({"evaluations":{step:candidate_eval}},{"evaluations":{step:reference_eval}},step)


def decide(dense,candidate,valid):
    terminal="1525";early="305"
    dl=dense["evaluations"][terminal]["loss"];base=candidate["evaluations"][terminal]["base"];full=candidate["evaluations"][terminal]["full"]
    intervals={"full_minus_dense":compare(full,dense["evaluations"][terminal],terminal),"base_minus_dense":compare(base,dense["evaluations"][terminal],terminal),"full_minus_base":compare(full,base,terminal)}
    full_gain=(dl-full["loss"])/dl;base_regret=(base["loss"]-dl)/dl;branch_gain=(base["loss"]-full["loss"])/base["loss"]
    early_dl=dense["evaluations"][early]["loss"];early_full=candidate["evaluations"][early]["full"]["loss"];early_gain=(early_dl-early_full)/early_dl
    audits=[entry for layer in candidate["quantized_weight_audits"] for entry in layer.values()]
    ledger=serving_ledger()
    gates={"protocol_integrity_valid":valid,"exact_storage_fits":bool(ledger["fits_bf16_bytes"]),
           "all_codes_scales_finite_and_bounded":all(a["all_finite"] and a["base_code_min"]>=-127 and a["base_code_max"]<=127 and set(a["delta_code_values"])<= {-1,0,1} for a in audits),
           "training_finite":not dense["train"]["nonfinite"] and not candidate["train"]["nonfinite"],
           "dropout_exercised_both_paths":min(candidate["branch_training_active_fraction_per_layer"])>=.45 and max(candidate["branch_training_active_fraction_per_layer"])<=.55,
           "full_beats_dense_by_0p25_percent":full_gain>=.0025 and intervals["full_minus_dense"]["upper_95"]<0,
           "base_noninferior_dense_within_0p05_percent":base_regret<=.0005 and intervals["base_minus_dense"]["upper_95"]<=.0005*dl,
           "branch_improves_base_by_0p1_percent":branch_gain>=.001 and intervals["full_minus_base"]["upper_95"]<0,
           "long_horizon_retains_half_early_edge_and_0p1_absolute":full_gain>=.001 and full_gain>=max(0.,early_gain)*.5}
    return {"terminal_losses":{"dense_bf16":dl,"candidate_base_prefill":base["loss"],"candidate_full_decode":full["loss"]},"early_losses":{"dense_bf16":early_dl,"candidate_base_prefill":candidate["evaluations"][early]["base"]["loss"],"candidate_full_decode":early_full},"relative_terminal_full_gain_vs_dense":full_gain,"relative_terminal_base_regret_vs_dense":base_regret,"relative_terminal_branch_gain_vs_base":branch_gain,"relative_early_full_gain_vs_dense":early_gain,"paired_terminal_intervals":intervals,"gates":gates,"advance_to_physical_h100_gate":all(gates.values())}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--train-file",type=Path,default=Path("data/block-algebra-scratch/train.uint16.bin"));parser.add_argument("--validation-file",type=Path,default=Path("data/block-algebra-scratch/validation.uint16.bin"));parser.add_argument("--data-manifest",type=Path,default=Path("results/block-algebra-scratch-data-manifest.json"));parser.add_argument("--output",type=Path,default=Path("results/phase-elastic-v2-50m-replication.json"));parser.add_argument("--steps",type=int,default=STEPS);parser.add_argument("--sequence-length",type=int,default=512);parser.add_argument("--micro-batch-size",type=int,default=32);parser.add_argument("--gradient-accumulation",type=int,default=2);parser.add_argument("--eval-batch-size",type=int,default=32);parser.add_argument("--eval-batches",type=int,default=128);parser.add_argument("--warmup-steps",type=int,default=100);parser.add_argument("--learning-rate",type=float,default=3e-4);parser.add_argument("--weight-decay",type=float,default=.1);parser.add_argument("--gradient-clip",type=float,default=1.);parser.add_argument("--seed",type=int,default=SEED);args=parser.parse_args()
    expected=(args.steps,args.sequence_length,args.micro_batch_size,args.gradient_accumulation,args.eval_batch_size,args.eval_batches,args.warmup_steps,args.learning_rate,args.weight_decay,args.gradient_clip,args.seed)==(1525,512,32,2,32,128,100,3e-4,.1,1.,SEED) and torch.__version__=="2.5.1+cu124" and torch.version.cuda=="12.4" and transformers.__version__=="4.57.6" and "H100" in torch.cuda.get_device_name()
    if sha256_file(DISCOVERY)!=DISCOVERY_SHA256:raise RuntimeError("frozen discovery changed")
    discovery=json.loads(DISCOVERY.read_text());data=validate_data_ledger(args.data_manifest,args.train_file,args.validation_file,args.sequence_length);valid=expected and discovery["protocol_valid"] and discovery["decision"]["advance_to_50m"] and data["valid"]
    torch.set_float32_matmul_precision("high");torch.backends.cuda.matmul.allow_tf32=True;train_file=TokenFile(args.train_file,args.sequence_length);validation=TokenFile(args.validation_file,args.sequence_length);device=torch.device("cuda")
    print(json.dumps({"starting_arm":"dense_bf16"}),flush=True);dense=train_dense(args,train_file,validation,device)
    print(json.dumps({"starting_arm":"phase_elastic_v2"}),flush=True);candidate=train_candidate(args,train_file,validation,device);decision=decide(dense,candidate,valid)
    payload={"schema":"phase-elastic-v2-50m-replication-v1","source_sha256":sha256_file(Path(__file__)),"preregistration_sha256":sha256_file(PREREGISTRATION),"discovery_sha256":sha256_file(DISCOVERY),"protocol_valid":valid,"data_ledger":data,"serving_ledger":serving_ledger(),"dense":dense,"candidate":candidate,"decision":decision}
    write_payload(args.output,payload);print(json.dumps(decision,indent=2,sort_keys=True))


if __name__=="__main__":main()

