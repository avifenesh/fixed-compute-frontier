#!/usr/bin/env python3
"""Replay and actually export the seed-813 ShiftNorm candidate."""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.folded_gain_shiftnorm_lm_replication import (
    BOUND, ScaffoldRMSNorm, build_model, causal_loss,
)
from experiments.reflex_swiglu_lm_screen import (
    TokenFile, evaluate, lr_multiplier, paired_loss_interval, sha256_file,
    validate_data_ledger,
)
from experiments.train_time_gauge_scaffold_lm_screen import make_base_state


OUTPUT = Path("results/folded-gain-shiftnorm-export-replay.json")
PREREGISTRATION = Path("results/folded-gain-shiftnorm-export-replay-preregistration.md")
EXPECTED_PREEXPORT = 5.527982976287603
STEPS = 1525
SEED = 813


class ServedShiftNorm(nn.Module):
    def __init__(self, source: ScaffoldRMSNorm) -> None:
        super().__init__()
        self.theta_alpha = nn.Parameter(source.theta_alpha.detach().clone())
        self.theta_beta = nn.Parameter(source.theta_beta.detach().clone())
        self.epsilon = source.epsilon

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        values = hidden_states.float()
        z = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.epsilon)
        theta = torch.cat((self.theta_alpha, self.theta_beta))
        shift = (BOUND * torch.tanh(theta / BOUND)).to(input_dtype)
        return z.to(input_dtype) + shift


def train_candidate(model, train_file, device):
    no_decay = [parameter for parameter in model.parameters() if parameter.ndim < 2]
    ids = {id(parameter) for parameter in no_decay}
    decay = [parameter for parameter in model.parameters() if id(parameter) not in ids]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=3e-4, betas=(0.9, 0.95), eps=1e-8, fused=True,
    )
    for group in optimizer.param_groups: group["base_lr"] = 3e-4
    optimizer.zero_grad(set_to_none=True)
    for step in range(STEPS):
        model.train(); multiplier = lr_multiplier(step, STEPS, 100)
        for group in optimizer.param_groups: group["lr"] = group["base_lr"] * multiplier
        for micro in range(2):
            inputs, targets = train_file.batch(step * 2 + micro, 32, device)
            loss = causal_loss(model, inputs, targets)
            (loss / 2).backward()
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if not math.isfinite(norm): raise RuntimeError((step, norm))
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    del optimizer


@torch.no_grad()
def export_in_place(model):
    folded = []
    for layer_index, layer in enumerate(model.model.layers):
        attention_norm = layer.input_layernorm
        if not isinstance(attention_norm, ScaffoldRMSNorm): raise TypeError(type(attention_norm))
        gain = attention_norm.weight.detach()
        for name in ("q_proj", "k_proj", "v_proj"):
            projection = getattr(layer.self_attn, name)
            projection.weight.mul_(gain[None, :])
        layer.input_layernorm = ServedShiftNorm(attention_norm).to(gain.device)
        folded.append({"layer": layer_index, "group": "attention", "consumers": 3})

        ffn_norm = layer.post_attention_layernorm
        if not isinstance(ffn_norm, ScaffoldRMSNorm): raise TypeError(type(ffn_norm))
        gain = ffn_norm.weight.detach()
        for name in ("gate_proj", "up_proj"):
            projection = getattr(layer.mlp, name)
            projection.weight.mul_(gain[None, :])
        layer.post_attention_layernorm = ServedShiftNorm(ffn_norm).to(gain.device)
        folded.append({"layer": layer_index, "group": "ffn", "consumers": 2})
    return folded


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    device=torch.device("cuda"); config,state,state_hash=make_base_state(SEED)
    # Match the per-arm reset used by the replication runner after constructing
    # the shared base state.
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    model,modules=build_model(config,state,device,"scaffold_bias")
    train_file=TokenFile(Path("data/block-algebra-scratch/train.uint16.bin"),512)
    validation_file=TokenFile(Path("data/block-algebra-scratch/validation.uint16.bin"),512)
    ledger=validate_data_ledger(Path("results/block-algebra-scratch-data-manifest.json"),train_file.path,validation_file.path,512)
    torch.set_float32_matmul_precision("high");torch.backends.cuda.matmul.allow_tf32=True
    train_candidate(model,train_file,device)
    pre=evaluate(model,validation_file,128,32,device)
    training_parameters=sum(parameter.numel() for parameter in model.parameters())
    folded=export_in_place(model)
    served_parameters=sum(parameter.numel() for parameter in model.parameters())
    post=evaluate(model,validation_file,128,32,device)
    wrapper_pre={"evaluations":{"1525":pre}};wrapper_post={"evaluations":{"1525":post}}
    interval=paired_loss_interval(wrapper_post,wrapper_pre,"1525")
    gates={"data_valid":ledger["valid"],"preexport_reproduced":abs(pre["loss"]-EXPECTED_PREEXPORT)<=2e-6,
           "served_parameter_count":served_parameters==37_758_336,
           "all_24_norms_folded":len(folded)==24,"nll_drift_at_most_2e_5":abs(post["loss"]-pre["loss"])<=2e-5,
           "paired_interval_contains_zero":interval["lower_95"]<=0<=interval["upper_95"]}
    payload={"schema":"folded-gain-shiftnorm-export-replay-v1","source_sha256":sha256_file(Path(__file__)),
             "preregistration_sha256":sha256_file(PREREGISTRATION),"base_state_sha256":state_hash,
             "training_parameters":training_parameters,"served_parameters":served_parameters,"folded":folded,
             "preexport":pre,"postexport":post,"post_minus_pre_interval":interval,"gates":gates,"pass":all(gates.values())}
    OUTPUT.write_text(json.dumps(payload,indent=2)+"\n");print(json.dumps({"pre":pre["loss"],"post":post["loss"],"interval":interval,"gates":gates,"pass":payload["pass"]},indent=2))


if __name__=="__main__":main()
