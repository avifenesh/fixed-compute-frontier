#!/usr/bin/env python3
"""Phase-elastic v2: INT8+ternary base and a smaller decode-only W4 branch."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from experiments.bbcm_matched_learning_screen import (
    exact_symmetric_decode, exact_ternary_decode, fake_quant_symmetric,
    fake_quant_ternary, initial_symmetric_scale, initial_ternary_scale,
)
from experiments.coalesced_attention_ffn_lm_screen import build_model as build_parallel_model
from experiments.cycle_factor_ffn_lm_screen import D, M, SEED
from experiments.phase_elastic_precision_ffn_lm_screen import QuantizedWeight
from experiments.reflex_swiglu_lm_screen import (
    TokenFile, evaluate, lr_multiplier, paired_loss_interval, sha256_file,
    validate_data_ledger, write_payload,
)
from experiments.triangular_microdepth_lm_screen import causal_loss


BRANCH_WIDTH = 1472
V1_RESULT = Path("results/phase-elastic-precision-ffn-lm-screen.json")
V1_RESULT_SHA256 = "b9b3aa448fd8b395698d74593d0a8e9f4a9c1e6e5f8cac765c04f6eef5c64f2c"
CYCLE_PREDECESSOR = Path("results/cycle-factor-ffn-lm-screen.json")
CYCLE_PREDECESSOR_SHA256 = "c66dfe6f5a6e3a1c4032932b9cbd2c491812af2e86a314e9cc6d46e83aba4029"
PREREGISTRATION = Path("results/phase-elastic-residual-precision-v2-preregistration.md")


def serving_ledger() -> dict[str, int | float | bool]:
    baseline_coordinates = 3 * D * M
    branch_coordinates = 3 * D * BRANCH_WIDTH
    baseline_bytes = 2 * baseline_coordinates
    base_code_bytes = 10 * baseline_coordinates // 8
    base_scale_bytes = 4 * (2 * M + D)
    branch_code_bytes = branch_coordinates // 2
    branch_scale_bytes = 2 * (2 * BRANCH_WIDTH + D)
    candidate_bytes = base_code_bytes + base_scale_bytes + branch_code_bytes + branch_scale_bytes
    return {
        "baseline_bf16_ffn_bytes_per_layer": baseline_bytes,
        "base_int8_ternary_code_bytes_per_layer": base_code_bytes,
        "base_bf16_scale_bytes_per_layer": base_scale_bytes,
        "residual_w4_code_bytes_per_layer": branch_code_bytes,
        "residual_bf16_scale_bytes_per_layer": branch_scale_bytes,
        "candidate_ffn_bytes_per_layer": candidate_bytes,
        "storage_slack_bytes_per_layer": baseline_bytes - candidate_bytes,
        "prefill_base_matrix_macs_per_token": baseline_coordinates,
        "decode_full_matrix_macs_per_token": baseline_coordinates + branch_coordinates,
        "decode_to_baseline_mac_ratio": (baseline_coordinates + branch_coordinates) / baseline_coordinates,
        "residual_width_ratio": BRANCH_WIDTH / M,
        "fits_bf16_bytes": candidate_bytes <= baseline_bytes,
    }


class ResidualQuantizedWeight(nn.Module):
    def __init__(self, target: torch.Tensor) -> None:
        super().__init__()
        target = target.detach().float()
        base_scale = initial_symmetric_scale(target, 127)
        base_codes = torch.round(target / base_scale).clamp(-127, 127)
        residual = target - base_codes * base_scale
        delta_scale = initial_ternary_scale(residual)
        self.base_shadow = nn.Parameter(target.clone())
        self.base_log_scale = nn.Parameter(base_scale.log())
        self.delta_shadow = nn.Parameter(residual.clone())
        self.delta_log_scale = nn.Parameter(delta_scale.log())

    def forward(self) -> torch.Tensor:
        return fake_quant_symmetric(self.base_shadow, self.base_log_scale, 127) + fake_quant_ternary(
            self.delta_shadow, self.delta_log_scale
        )

    @torch.no_grad()
    def exact(self) -> torch.Tensor:
        return exact_symmetric_decode(self.base_shadow, self.base_log_scale, 127) + exact_ternary_decode(
            self.delta_shadow, self.delta_log_scale
        )

    @torch.no_grad()
    def audit(self) -> dict[str, Any]:
        base_scale = self.base_log_scale.exp().to(torch.bfloat16).to(torch.float32)
        delta_scale = self.delta_log_scale.exp().to(torch.bfloat16).to(torch.float32)
        base_codes = torch.round(self.base_shadow.detach().float() / base_scale).clamp(-127, 127)
        normalized_delta = self.delta_shadow.detach().float() / delta_scale
        delta_codes = torch.sign(normalized_delta) * (normalized_delta.abs() >= 0.5)
        return {
            "base_code_min": int(base_codes.min()),
            "base_code_max": int(base_codes.max()),
            "delta_code_values": sorted(int(value) for value in delta_codes.unique().tolist()),
            "base_scale_min": float(base_scale.min()),
            "delta_scale_min": float(delta_scale.min()),
            "all_finite": bool(
                torch.isfinite(base_codes).all() and torch.isfinite(delta_codes).all()
                and torch.isfinite(base_scale).all() and torch.isfinite(delta_scale).all()
            ),
            "output_rows": int(base_codes.shape[0]),
            "coordinates": base_codes.numel(),
        }


class PhaseElasticV2MLP(nn.Module):
    def __init__(self, original: nn.Module, std: float) -> None:
        super().__init__()
        self.base_gate = ResidualQuantizedWeight(original.gate_proj.weight)
        self.base_up = ResidualQuantizedWeight(original.up_proj.weight)
        self.base_down = ResidualQuantizedWeight(original.down_proj.weight)
        gate = torch.empty(BRANCH_WIDTH, D, device=original.gate_proj.weight.device)
        up = torch.empty_like(gate)
        down = torch.empty(D, BRANCH_WIDTH, device=gate.device)
        nn.init.normal_(gate, mean=0.0, std=std)
        nn.init.normal_(up, mean=0.0, std=std)
        nn.init.normal_(down, mean=0.0, std=std * 0.05 * math.sqrt(M / BRANCH_WIDTH))
        self.branch_gate = QuantizedWeight(gate, 7)
        self.branch_up = QuantizedWeight(up, 7)
        self.branch_down = QuantizedWeight(down, 7)
        self.mode = "train"
        self.last_branch_active = False

    def set_mode(self, mode: str) -> None:
        if mode not in {"train", "base", "full"}:
            raise ValueError(mode)
        self.mode = mode

    def _branch_is_active(self) -> bool:
        if self.mode == "base":
            return False
        if self.mode == "full":
            return True
        if not self.training:
            raise RuntimeError("evaluation requires explicit base or full mode")
        return bool(torch.rand((), device=self.base_gate.base_shadow.device) < 0.5)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base_activation = F.silu(F.linear(hidden_states, self.base_gate())) * F.linear(
            hidden_states, self.base_up()
        )
        base = F.linear(base_activation, self.base_down())
        active = self._branch_is_active()
        self.last_branch_active = active
        if not active:
            return base
        branch_activation = F.silu(F.linear(hidden_states, self.branch_gate())) * F.linear(
            hidden_states, self.branch_up()
        )
        return base + F.linear(branch_activation, self.branch_down())

    @torch.no_grad()
    def audit(self) -> dict[str, Any]:
        result = {
            name: module.audit()
            for name, module in (
                ("base_gate", self.base_gate), ("base_up", self.base_up),
                ("base_down", self.base_down),
            )
        }
        for name, module in (
            ("branch_gate", self.branch_gate), ("branch_up", self.branch_up),
            ("branch_down", self.branch_down),
        ):
            audit = module.audit()
            result[name] = {
                "base_code_min": audit["code_min"],
                "base_code_max": audit["code_max"],
                "delta_code_values": [],
                "base_scale_min": audit["scale_min"],
                "delta_scale_min": 1.0,
                "all_finite": audit["all_finite"],
                "output_rows": audit["output_rows"],
                "coordinates": audit["coordinates"],
            }
        return result


def set_mode(modules, mode):
    for module in modules:
        module.set_mode(mode)


@torch.no_grad()
def evaluate_mode(model, modules, mode, validation, batches, batch_size, device):
    set_mode(modules, mode)
    result = evaluate(model, validation, batches, batch_size, device)
    set_mode(modules, "train")
    return result


def build_candidate(device, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model, _ = build_parallel_model(device, "parallel_baseline")
    modules = []
    for layer in model.model.layers:
        replacement = PhaseElasticV2MLP(layer.mlp, model.config.initializer_range).to(device)
        layer.mlp = replacement; modules.append(replacement)
    model.config.use_cache = False
    return model, modules


def train(args, train_file, validation, device):
    model, modules = build_candidate(device, args.seed)
    scales = [parameter for name, parameter in model.named_parameters() if "log_scale" in name]
    scale_ids = {id(parameter) for parameter in scales}
    ordinary = [parameter for parameter in model.parameters() if id(parameter) not in scale_ids]
    optimizer = torch.optim.AdamW([
        {"params": ordinary, "weight_decay": args.weight_decay},
        {"params": scales, "weight_decay": 0.0},
    ], lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=True)
    evaluations = {"0": {
        "base": evaluate_mode(model, modules, "base", validation, args.eval_batches, args.eval_batch_size, device),
        "full": evaluate_mode(model, modules, "full", validation, args.eval_batches, args.eval_batch_size, device),
    }}
    optimizer.zero_grad(set_to_none=True)
    losses, durations, norms = [], [], []
    active = np.zeros(len(modules), dtype=np.int64)
    for step in range(args.steps):
        model.train(); set_mode(modules, "train")
        scheduled = args.learning_rate * lr_multiplier(step, args.steps, args.warmup_steps)
        for group in optimizer.param_groups: group["lr"] = scheduled
        started=time.perf_counter(); accumulated=0.0
        for micro in range(args.gradient_accumulation):
            inputs,targets=train_file.batch(step*args.gradient_accumulation+micro,args.micro_batch_size,device)
            loss=causal_loss(model,inputs,targets);(loss/args.gradient_accumulation).backward()
            accumulated+=float(loss.detach())/args.gradient_accumulation
            active += np.asarray([module.last_branch_active for module in modules],dtype=np.int64)
        norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),args.gradient_clip))
        if not math.isfinite(accumulated) or not math.isfinite(norm): raise RuntimeError(f"nonfinite step {step+1}")
        optimizer.step();optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize()
        losses.append(accumulated);norms.append(norm);durations.append(time.perf_counter()-started)
    evaluations[str(args.steps)]={
        "base":evaluate_mode(model,modules,"base",validation,args.eval_batches,args.eval_batch_size,device),
        "full":evaluate_mode(model,modules,"full",validation,args.eval_batches,args.eval_batch_size,device),
    }
    result={
        "training_shadow_parameters":sum(parameter.numel() for parameter in model.parameters()),
        "evaluations":evaluations,
        "branch_training_active_fraction_per_layer":(active/(args.steps*args.gradient_accumulation)).tolist(),
        "quantized_weight_audits":[module.audit() for module in modules],
        "train":{"prediction_tokens":args.steps*args.gradient_accumulation*args.micro_batch_size*args.sequence_length,
                 "mean_loss":float(np.mean(losses)),"final_loss":losses[-1],"max_loss":max(losses),
                 "max_preclip_gradient_norm":max(norms),"elapsed_seconds":sum(durations),
                 "tokens_per_second":args.steps*args.gradient_accumulation*args.micro_batch_size*args.sequence_length/sum(durations),"nonfinite":False},
    }
    del optimizer,modules,model;gc.collect();torch.cuda.empty_cache();return result


def decide(candidate, cycle, protocol_valid):
    step="305";dense=cycle["arms"]["full_swiglu"];dense_loss=dense["evaluations"][step]["loss"]
    full=candidate["evaluations"][step]["full"];base=candidate["evaluations"][step]["base"]
    fw={"evaluations":{step:full}};bw={"evaluations":{step:base}}
    intervals={"full_minus_dense":paired_loss_interval(fw,dense,step),"base_minus_dense":paired_loss_interval(bw,dense,step),"full_minus_base":paired_loss_interval(fw,bw,step)}
    full_gain=(dense_loss-full["loss"])/dense_loss;base_regret=(base["loss"]-dense_loss)/dense_loss;branch_gain=(base["loss"]-full["loss"])/base["loss"]
    audits=[entry for layer in candidate["quantized_weight_audits"] for entry in layer.values()]
    ledger=serving_ledger()
    gates={"protocol_integrity_valid":protocol_valid,"exact_storage_fits":bool(ledger["fits_bf16_bytes"]),
           "all_codes_scales_finite_and_bounded":all(a["all_finite"] and a["base_code_min"]>=-127 and a["base_code_max"]<=127 and set(a["delta_code_values"])<= {-1,0,1} for a in audits),
           "training_finite":not candidate["train"]["nonfinite"],
           "dropout_exercised_both_paths":min(candidate["branch_training_active_fraction_per_layer"])>=.4 and max(candidate["branch_training_active_fraction_per_layer"])<=.6,
           "full_beats_dense_by_0p1_percent":full_gain>=.001 and intervals["full_minus_dense"]["upper_95"]<0,
           "base_noninferior_dense_within_0p05_percent":base_regret<=.0005 and intervals["base_minus_dense"]["upper_95"]<=.0005*dense_loss,
           "decode_branch_improves_base_by_0p1_percent":branch_gain>=.001 and intervals["full_minus_base"]["upper_95"]<0}
    return {"losses":{"dense_bf16":dense_loss,"candidate_base_prefill":base["loss"],"candidate_full_decode":full["loss"]},
            "relative_full_gain_vs_dense":full_gain,"relative_base_regret_vs_dense":base_regret,"relative_decode_branch_gain_vs_base":branch_gain,
            "paired_intervals":intervals,"gates":gates,"advance_to_50m":all(gates.values())}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--train-file",type=Path,default=Path("data/block-algebra-scratch/train.uint16.bin"));parser.add_argument("--validation-file",type=Path,default=Path("data/block-algebra-scratch/validation.uint16.bin"));parser.add_argument("--data-manifest",type=Path,default=Path("results/block-algebra-scratch-data-manifest.json"));parser.add_argument("--output",type=Path,default=Path("results/phase-elastic-residual-precision-v2-lm-screen.json"));parser.add_argument("--steps",type=int,default=305);parser.add_argument("--sequence-length",type=int,default=512);parser.add_argument("--micro-batch-size",type=int,default=32);parser.add_argument("--gradient-accumulation",type=int,default=2);parser.add_argument("--eval-batch-size",type=int,default=32);parser.add_argument("--eval-batches",type=int,default=128);parser.add_argument("--warmup-steps",type=int,default=30);parser.add_argument("--learning-rate",type=float,default=3e-4);parser.add_argument("--weight-decay",type=float,default=.1);parser.add_argument("--gradient-clip",type=float,default=1.);parser.add_argument("--seed",type=int,default=SEED);args=parser.parse_args()
    expected=(args.steps,args.sequence_length,args.micro_batch_size,args.gradient_accumulation,args.eval_batch_size,args.eval_batches,args.warmup_steps,args.learning_rate,args.weight_decay,args.gradient_clip,args.seed)==(305,512,32,2,32,128,30,3e-4,.1,1.,SEED) and torch.__version__=="2.5.1+cu124" and torch.version.cuda=="12.4" and transformers.__version__=="4.57.6" and "H100" in torch.cuda.get_device_name()
    if sha256_file(V1_RESULT)!=V1_RESULT_SHA256 or sha256_file(CYCLE_PREDECESSOR)!=CYCLE_PREDECESSOR_SHA256:raise RuntimeError("frozen predecessor changed")
    v1=json.loads(V1_RESULT.read_text());cycle=json.loads(CYCLE_PREDECESSOR.read_text());ledger=validate_data_ledger(args.data_manifest,args.train_file,args.validation_file,args.sequence_length);valid=expected and v1["protocol_valid"] and cycle["protocol_valid"] and ledger["valid"]
    torch.set_float32_matmul_precision("high");torch.backends.cuda.matmul.allow_tf32=True
    candidate=train(args,TokenFile(args.train_file,args.sequence_length),TokenFile(args.validation_file,args.sequence_length),torch.device("cuda"));decision=decide(candidate,cycle,valid)
    payload={"schema":"phase-elastic-residual-precision-v2","source_sha256":sha256_file(Path(__file__)),"preregistration_sha256":sha256_file(PREREGISTRATION),"v1_result_sha256":sha256_file(V1_RESULT),"cycle_predecessor_sha256":sha256_file(CYCLE_PREDECESSOR),"protocol_valid":valid,"data_ledger":ledger,"serving_ledger":serving_ledger(),"candidate":candidate,"decision":decision}
    write_payload(args.output,payload);print(json.dumps(decision,indent=2,sort_keys=True))


if __name__=="__main__":main()

