#!/usr/bin/env python3
"""Matched continued-learning screen for BBCM versus dense BF16 and K4-W4."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM

try:
    from experiments.reflex_swiglu_lm_screen import (
        MODEL,
        MODEL_REVISION,
        TokenFile,
        evaluate,
        lr_multiplier,
        paired_loss_interval,
        sha256_file,
        validate_data_ledger,
        write_payload,
    )
except ModuleNotFoundError:
    from reflex_swiglu_lm_screen import (
        MODEL,
        MODEL_REVISION,
        TokenFile,
        evaluate,
        lr_multiplier,
        paired_loss_interval,
        sha256_file,
        validate_data_ledger,
        write_payload,
    )


VALID_ARMS = ("baseline", "bbcm", "independent_k4_w4")
EXPECTED_DATA_MANIFEST_SHA256 = "955cc3d3be7992840da95881450e4483b2dee4a9567b062b17e1ebe061c817a4"


def bf16_ste(value: torch.Tensor) -> torch.Tensor:
    rounded = value.to(torch.bfloat16).to(torch.float32)
    return value + (rounded - value).detach()


def initial_symmetric_scale(weight: torch.Tensor, maximum_code: int) -> torch.Tensor:
    maximum = weight.detach().float().abs().amax(dim=-1, keepdim=True)
    raw = maximum / maximum_code
    rounded = raw.to(torch.bfloat16).to(torch.float32)
    smallest = torch.full_like(rounded, torch.finfo(torch.bfloat16).tiny)
    return torch.where(maximum > 0, torch.where(rounded > 0, rounded, smallest), torch.ones_like(rounded))


def initial_ternary_scale(weight: torch.Tensor, iterations: int = 4) -> torch.Tensor:
    source = weight.detach().float()
    maximum = source.abs().amax(dim=-1, keepdim=True)
    nonzero = maximum > 0
    scale = ((2.0 / 3.0) * maximum).to(torch.bfloat16).to(torch.float32)
    scale = torch.where(nonzero, scale.clamp_min(torch.finfo(torch.bfloat16).tiny), torch.ones_like(scale))
    for _ in range(iterations):
        assigned = source.abs() >= scale / 2.0
        count = assigned.sum(dim=-1, keepdim=True).clamp_min(1)
        update = (source.abs() * assigned).sum(dim=-1, keepdim=True) / count
        update = update.to(torch.bfloat16).to(torch.float32)
        scale = torch.where(nonzero, update.clamp_min(torch.finfo(torch.bfloat16).tiny), torch.ones_like(update))
    return scale


def fake_quant_symmetric(
    shadow: torch.Tensor,
    log_scale: torch.Tensor,
    maximum_code: int,
) -> torch.Tensor:
    scale = bf16_ste(log_scale.exp())
    normalized = shadow / scale
    hard = torch.round(normalized).clamp(-maximum_code, maximum_code)
    codes = normalized + (hard - normalized).detach()
    return codes * scale


def fake_quant_ternary(shadow: torch.Tensor, log_scale: torch.Tensor) -> torch.Tensor:
    scale = bf16_ste(log_scale.exp())
    normalized = shadow / scale
    hard = torch.sign(normalized) * (normalized.abs() >= 0.5)
    codes = normalized + (hard - normalized).detach()
    return codes * scale


def exact_symmetric_decode(shadow: torch.Tensor, log_scale: torch.Tensor, maximum_code: int) -> torch.Tensor:
    scale = log_scale.exp().to(torch.bfloat16).to(torch.float32)
    return torch.round(shadow.detach().float() / scale).clamp(-maximum_code, maximum_code) * scale


def exact_ternary_decode(shadow: torch.Tensor, log_scale: torch.Tensor) -> torch.Tensor:
    scale = log_scale.exp().to(torch.bfloat16).to(torch.float32)
    normalized = shadow.detach().float() / scale
    codes = torch.sign(normalized) * (normalized.abs() >= 0.5)
    return codes * scale


class BBCMWeightBank(nn.Module):
    def __init__(self, target: torch.Tensor, routes: int) -> None:
        super().__init__()
        target = target.detach().float()
        base_scale = initial_symmetric_scale(target, 127)
        base_codes = torch.round(target / base_scale).clamp(-127, 127)
        base_decoded = base_codes * base_scale
        residual = target - base_decoded
        delta_scale = initial_ternary_scale(residual)
        self.base_shadow = nn.Parameter(target.clone())
        self.base_log_scale = nn.Parameter(base_scale.log())
        self.delta_shadow = nn.Parameter(residual.unsqueeze(0).repeat(routes, 1, 1))
        self.delta_log_scale = nn.Parameter(delta_scale.unsqueeze(0).repeat(routes, 1, 1).log())
        self.routes = routes

    def base_weight(self) -> torch.Tensor:
        return fake_quant_symmetric(self.base_shadow, self.base_log_scale, 127)

    def weight(self, route: int, base: torch.Tensor | None = None) -> torch.Tensor:
        if base is None:
            base = self.base_weight()
        return base + fake_quant_ternary(self.delta_shadow[route], self.delta_log_scale[route])

    @torch.no_grad()
    def exact_weight(self, route: int) -> torch.Tensor:
        return exact_symmetric_decode(self.base_shadow, self.base_log_scale, 127) + exact_ternary_decode(
            self.delta_shadow[route], self.delta_log_scale[route]
        )


class IndependentW4WeightBank(nn.Module):
    def __init__(self, target: torch.Tensor, routes: int) -> None:
        super().__init__()
        target = target.detach().float()
        scale = initial_symmetric_scale(target, 7)
        self.shadow = nn.Parameter(target.unsqueeze(0).repeat(routes, 1, 1))
        self.log_scale = nn.Parameter(scale.unsqueeze(0).repeat(routes, 1, 1).log())
        self.routes = routes

    def weight(self, route: int) -> torch.Tensor:
        return fake_quant_symmetric(self.shadow[route], self.log_scale[route], 7)

    @torch.no_grad()
    def exact_weight(self, route: int) -> torch.Tensor:
        return exact_symmetric_decode(self.shadow[route], self.log_scale[route], 7)


@torch.no_grad()
def activation_aware_masks(
    model: nn.Module,
    calibration_inputs: torch.Tensor,
    target_hidden: int,
) -> tuple[list[torch.Tensor], str]:
    scores: list[torch.Tensor | None] = [None] * len(model.model.layers)
    handles = []
    for layer_index, layer in enumerate(model.model.layers):
        def capture(module: nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor, index: int = layer_index) -> None:
            hidden = inputs[0]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                activation = module.act_fn(module.gate_proj(hidden)) * module.up_proj(hidden)
            mean_magnitude = activation.detach().float().abs().mean(dim=tuple(range(activation.ndim - 1)))
            down_norm = torch.linalg.vector_norm(module.down_proj.weight.detach().float(), dim=0)
            scores[index] = mean_magnitude * down_norm

        handles.append(layer.mlp.register_forward_hook(capture))
    model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model(input_ids=calibration_inputs, use_cache=False)
    for handle in handles:
        handle.remove()
    if any(score is None for score in scores):
        raise RuntimeError("missing activation-aware pruning score")
    masks = []
    digest = hashlib.sha256()
    for score in scores:
        assert score is not None
        keep = torch.topk(score, target_hidden, largest=True, sorted=False).indices
        keep = torch.sort(keep).values
        masks.append(keep)
        digest.update(keep.detach().cpu().to(torch.int32).numpy().tobytes())
    return masks, digest.hexdigest()


class DensePrunedMLP(nn.Module):
    def __init__(self, original: nn.Module, keep: torch.Tensor) -> None:
        super().__init__()
        dimension = original.gate_proj.in_features
        hidden = keep.numel()
        self.gate_proj = nn.Linear(dimension, hidden, bias=False, dtype=original.gate_proj.weight.dtype, device=keep.device)
        self.up_proj = nn.Linear(dimension, hidden, bias=False, dtype=original.up_proj.weight.dtype, device=keep.device)
        self.down_proj = nn.Linear(hidden, dimension, bias=False, dtype=original.down_proj.weight.dtype, device=keep.device)
        self.gate_proj.weight.data.copy_(original.gate_proj.weight.detach()[keep])
        self.up_proj.weight.data.copy_(original.up_proj.weight.detach()[keep])
        self.down_proj.weight.data.copy_(original.down_proj.weight.detach()[:, keep])
        self.act_fn = original.act_fn

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class RoutedQuantizedMLP(nn.Module):
    def __init__(
        self,
        original: nn.Module,
        arm: str,
        routes: int,
        target_hidden: int,
        layer_index: int,
        warmup_steps: int,
        keep: torch.Tensor,
        routing_seed: int,
    ) -> None:
        super().__init__()
        if arm not in {"bbcm", "independent_k4_w4"}:
            raise ValueError(arm)
        if keep.numel() != target_hidden:
            raise ValueError("activation-aware mask width mismatch")
        gate = original.gate_proj.weight.detach()[keep]
        up = original.up_proj.weight.detach()[keep]
        down = original.down_proj.weight.detach()[:, keep]
        bank_type = BBCMWeightBank if arm == "bbcm" else IndependentW4WeightBank
        self.gate_bank = bank_type(gate, routes)
        self.up_bank = bank_type(up, routes)
        self.down_bank = bank_type(down, routes)
        self.act_fn = original.act_fn
        self.router_weight = nn.Parameter(torch.empty(routes, gate.shape[1], dtype=torch.float32, device=gate.device))
        self.router_bias = nn.Parameter(torch.zeros(routes, dtype=torch.float32, device=gate.device))
        nn.init.normal_(self.router_weight, mean=0.0, std=0.01)
        self.routes = routes
        self.layer_index = layer_index
        self.warmup_steps = warmup_steps
        self.routing_seed = routing_seed
        self.routing_step = warmup_steps
        self.routing_override = "learned"
        self.last_aux_loss = torch.tensor(0.0, device=gate.device)
        self.collect_routes = False
        self.register_buffer("route_counts", torch.zeros(routes, dtype=torch.long, device=gate.device), persistent=False)

    def set_step(self, step: int) -> None:
        self.routing_step = step

    def reset_route_counts(self) -> None:
        self.route_counts.zero_()

    def _route(self, flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool]:
        router_weight = bf16_ste(self.router_weight)
        router_bias = bf16_ste(self.router_bias)
        logits = F.linear(flat, router_weight, router_bias).float()
        probabilities = torch.softmax(logits, dim=-1)
        balanced_warmup = self.training and self.routing_step < self.warmup_steps
        if balanced_warmup:
            generator = torch.Generator(device=flat.device)
            generator.manual_seed(
                self.routing_seed + 1_000_003 * self.routing_step + 10_007 * self.layer_index
            )
            permutation = torch.randperm(flat.shape[0], generator=generator, device=flat.device)
            balanced = torch.arange(flat.shape[0], device=flat.device) % self.routes
            route = torch.empty_like(balanced)
            route[permutation] = balanced
        else:
            route = torch.argmax(probabilities, dim=-1)
            if self.routing_override.startswith("shuffled_"):
                shuffle_index = int(self.routing_override.removeprefix("shuffled_"))
                generator = torch.Generator(device=flat.device)
                generator.manual_seed(
                    self.routing_seed + 1_000_003 * shuffle_index + 10_007 * self.layer_index
                )
                permutation = torch.randperm(flat.shape[0], generator=generator, device=flat.device)
                route = route.index_select(0, permutation)
            elif self.routing_override.startswith("force_"):
                forced = int(self.routing_override.removeprefix("force_"))
                route = torch.full_like(route, forced)
            elif self.routing_override != "learned":
                raise ValueError(f"unknown routing override {self.routing_override}")
        hard_load = F.one_hot(route, self.routes).float().mean(dim=0)
        importance = probabilities.mean(dim=0)
        if balanced_warmup:
            self.last_aux_loss = probabilities.new_zeros(())
        else:
            self.last_aux_loss = self.routes * torch.sum(importance * hard_load.detach()) - 1.0
        if self.collect_routes:
            self.route_counts += torch.bincount(route, minlength=self.routes)
        return route, probabilities, balanced_warmup

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, original_shape[-1])
        route, probabilities, balanced_warmup = self._route(flat)
        base_gate = self.gate_bank.base_weight() if isinstance(self.gate_bank, BBCMWeightBank) else None
        base_up = self.up_bank.base_weight() if isinstance(self.up_bank, BBCMWeightBank) else None
        base_down = self.down_bank.base_weight() if isinstance(self.down_bank, BBCMWeightBank) else None
        indices_by_route = [torch.nonzero(route == expert, as_tuple=False).flatten() for expert in range(self.routes)]
        ordered_indices = []
        ordered_outputs = []
        for expert, indices in enumerate(indices_by_route):
            if indices.numel() == 0:
                continue
            selected = flat.index_select(0, indices)
            if isinstance(self.gate_bank, BBCMWeightBank):
                gate_weight = self.gate_bank.weight(expert, base_gate)
                up_weight = self.up_bank.weight(expert, base_up)
                down_weight = self.down_bank.weight(expert, base_down)
            else:
                gate_weight = self.gate_bank.weight(expert)
                up_weight = self.up_bank.weight(expert)
                down_weight = self.down_bank.weight(expert)
            activated = self.act_fn(F.linear(selected, gate_weight)) * F.linear(selected, up_weight)
            output = F.linear(activated, down_weight)
            if not balanced_warmup:
                selected_probability = probabilities.index_select(0, indices)[:, expert : expert + 1]
                output = output * (1.0 + selected_probability - selected_probability.detach())
            ordered_indices.append(indices)
            ordered_outputs.append(output)
        permutation = torch.cat(ordered_indices)
        ordered = torch.cat(ordered_outputs)
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(permutation.numel(), device=permutation.device)
        return ordered.index_select(0, inverse).reshape(original_shape)


def install_arm(
    model: nn.Module,
    arm: str,
    routes: int,
    target_hidden: int,
    warmup_steps: int,
    masks: list[torch.Tensor] | None,
    routing_seed: int,
) -> list[RoutedQuantizedMLP]:
    if arm == "baseline":
        return []
    if masks is None or len(masks) != len(model.model.layers):
        raise ValueError("quantized arms require frozen activation-aware masks")
    modules = []
    for layer_index, layer in enumerate(model.model.layers):
        replacement = RoutedQuantizedMLP(
            layer.mlp,
            arm,
            routes,
            target_hidden,
            layer_index,
            warmup_steps,
            masks[layer_index],
            routing_seed,
        )
        layer.mlp = replacement
        modules.append(replacement)
    return modules


def set_routing_step(modules: list[RoutedQuantizedMLP], step: int) -> None:
    for module in modules:
        module.set_step(step)


def training_loss(
    model: nn.Module,
    modules: list[RoutedQuantizedMLP],
    inputs: torch.Tensor,
    targets: torch.Tensor,
    load_balance_coefficient: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=inputs, use_cache=False).logits
    language_loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))
    if modules:
        auxiliary = torch.stack([module.last_aux_loss for module in modules]).mean()
    else:
        auxiliary = language_loss.new_zeros(())
    return language_loss + load_balance_coefficient * auxiliary, language_loss, auxiliary


def make_optimizer(
    model: nn.Module,
    arm: str,
    shared_lr: float,
    router_lr: float,
    bbcm_component_lr_scale: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    router = []
    scales = []
    bbcm_components = []
    shared = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "router_weight" in name or "router_bias" in name:
            router.append(parameter)
        elif "log_scale" in name:
            scales.append(parameter)
        elif arm == "bbcm" and ("base_shadow" in name or "delta_shadow" in name):
            bbcm_components.append(parameter)
        else:
            shared.append(parameter)
    groups = []
    if shared:
        groups.append({"params": shared, "lr": shared_lr, "base_lr": shared_lr, "weight_decay": weight_decay})
    if bbcm_components:
        component_lr = shared_lr * bbcm_component_lr_scale
        groups.append({"params": bbcm_components, "lr": component_lr, "base_lr": component_lr, "weight_decay": weight_decay})
    if scales:
        scale_lr = shared_lr * bbcm_component_lr_scale if arm == "bbcm" else shared_lr
        groups.append({"params": scales, "lr": scale_lr, "base_lr": scale_lr, "weight_decay": 0.0})
    if router:
        groups.append({"params": router, "lr": router_lr, "base_lr": router_lr, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8, fused=True)


def gradient_component_stats(model: nn.Module, arm: str) -> dict[str, Any]:
    grouped: dict[str, list[torch.Tensor]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "router_weight" in name or "router_bias" in name:
            category = "router"
        elif arm == "bbcm" and "base_log_scale" in name:
            category = "base_scale"
        elif arm == "bbcm" and "delta_log_scale" in name:
            category = "delta_scale"
        elif arm == "bbcm" and "base_shadow" in name:
            category = "base_shadow"
        elif arm == "bbcm" and "delta_shadow" in name:
            category = "delta_shadow"
        elif arm == "independent_k4_w4" and "log_scale" in name:
            category = "w4_scale"
        elif arm == "independent_k4_w4" and ".shadow" in name:
            category = "w4_shadow"
        else:
            category = "dense_ffn"
        if parameter.grad is not None:
            grouped.setdefault(category, []).append(parameter.grad.detach().float().reshape(-1))
        else:
            grouped.setdefault(category, [])
    result = {}
    for category, gradients in grouped.items():
        if not gradients:
            result[category] = {
                "coordinates": 0,
                "all_finite": True,
                "nonzero": False,
                "mean_abs": 0.0,
                "max_abs": 0.0,
            }
            continue
        coordinates = sum(gradient.numel() for gradient in gradients)
        absolute_sums = sum(float(gradient.abs().sum()) for gradient in gradients)
        result[category] = {
            "coordinates": coordinates,
            "all_finite": all(bool(torch.isfinite(gradient).all()) for gradient in gradients),
            "nonzero": any(bool((gradient != 0).any()) for gradient in gradients),
            "mean_abs": absolute_sums / coordinates,
            "max_abs": max(float(gradient.abs().max()) for gradient in gradients),
        }
    return result


@torch.no_grad()
def evaluate_with_route_utilization(
    model: nn.Module,
    modules: list[RoutedQuantizedMLP],
    token_file: TokenFile,
    batches: int,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not modules:
        return evaluate(model, token_file, batches, batch_size, device), {"applicable": False}
    for module in modules:
        module.reset_route_counts()
        module.collect_routes = True
    evaluation = evaluate(model, token_file, batches, batch_size, device)
    records = []
    total = torch.zeros(modules[0].routes, dtype=torch.long, device=device)
    for layer, module in enumerate(modules):
        counts = module.route_counts.clone()
        total += counts
        records.append({"layer": layer, "counts": counts.cpu().tolist()})
        module.collect_routes = False
    fractions = (total.float() / total.sum()).cpu().tolist()
    layer_entropies = []
    for record in records:
        counts = np.asarray(record["counts"], dtype=np.float64)
        probabilities = counts / counts.sum()
        positive = probabilities[probabilities > 0]
        layer_entropies.append(float(-(positive * np.log(positive)).sum() / math.log(modules[0].routes)))
    utilization = {
        "applicable": True,
        "evaluation_batches": batches,
        "evaluation_sequences": batches * batch_size,
        "total_counts": total.cpu().tolist(),
        "fractions": fractions,
        "minimum_fraction": min(fractions),
        "maximum_fraction": max(fractions),
        "median_normalized_layer_entropy": float(np.median(layer_entropies)),
        "records": records,
    }
    return evaluation, utilization


@torch.no_grad()
def routing_ablations(
    model: nn.Module,
    modules: list[RoutedQuantizedMLP],
    learned_evaluation: dict[str, Any],
    token_file: TokenFile,
    batches: int,
    batch_size: int,
    device: torch.device,
    utilization: dict[str, Any],
) -> dict[str, Any]:
    if not modules:
        return {"applicable": False}

    def set_override(value: str) -> None:
        for module in modules:
            module.routing_override = value

    shuffled_records = []
    for shuffle_index in range(3):
        set_override(f"shuffled_{shuffle_index}")
        shuffled_evaluation = evaluate(model, token_file, batches, batch_size, device)
        shuffled_records.append(
            {"shuffle_index": shuffle_index, "evaluation": shuffled_evaluation}
        )
    best_shuffled = min(shuffled_records, key=lambda record: record["evaluation"]["loss"])
    most_used = int(np.argmax(np.asarray(utilization["total_counts"])))
    set_override(f"force_{most_used}")
    forced = evaluate(model, token_file, batches, batch_size, device)
    set_override("learned")
    learned_wrapper = {"evaluations": {"terminal": learned_evaluation}}
    shuffled_wrapper = {"evaluations": {"terminal": best_shuffled["evaluation"]}}
    forced_wrapper = {"evaluations": {"terminal": forced}}
    return {
        "applicable": True,
        "shuffled_controls": shuffled_records,
        "best_shuffled_index": best_shuffled["shuffle_index"],
        "best_shuffled": best_shuffled["evaluation"],
        "forced_most_used_expert_index": most_used,
        "forced_most_used": forced,
        "learned_minus_best_shuffled_interval": paired_loss_interval(
            learned_wrapper, shuffled_wrapper, "terminal"
        ),
        "learned_minus_forced_most_used_interval": paired_loss_interval(
            learned_wrapper, forced_wrapper, "terminal"
        ),
    }


@torch.no_grad()
def route_weight_geometry(modules: list[RoutedQuantizedMLP]) -> dict[str, Any]:
    if not modules:
        return {"applicable": False}
    correlations = []
    delta_ratios = []
    records = []
    for layer_index, module in enumerate(modules):
        for projection, bank in (
            ("gate", module.gate_bank),
            ("up", module.up_bank),
            ("down", module.down_bank),
        ):
            weights = [bank.exact_weight(route).float() for route in range(module.routes)]
            mean = torch.stack(weights).mean(dim=0)
            mean_variance = float(mean.square().mean())
            route_delta_variance = float(
                torch.stack([(weight - mean).square().mean() for weight in weights]).mean()
            )
            delta_ratio = route_delta_variance / max(mean_variance, torch.finfo(torch.float32).tiny)
            pair_correlations = []
            for left in range(module.routes):
                for right in range(left + 1, module.routes):
                    a = weights[left].reshape(-1)
                    b = weights[right].reshape(-1)
                    a = a - a.mean()
                    b = b - b.mean()
                    correlation = float(
                        torch.dot(a, b)
                        / (
                            torch.linalg.vector_norm(a)
                            * torch.linalg.vector_norm(b)
                        ).clamp_min(torch.finfo(torch.float32).tiny)
                    )
                    pair_correlations.append(correlation)
                    correlations.append(correlation)
            delta_ratios.append(delta_ratio)
            records.append(
                {
                    "layer": layer_index,
                    "projection": projection,
                    "mean_pairwise_correlation": float(np.mean(pair_correlations)),
                    "minimum_pairwise_correlation": min(pair_correlations),
                    "route_delta_to_mean_variance_ratio": delta_ratio,
                }
            )
    return {
        "applicable": True,
        "mean_pairwise_correlation": float(np.mean(correlations)),
        "minimum_pairwise_correlation": min(correlations),
        "mean_route_delta_to_mean_variance_ratio": float(np.mean(delta_ratios)),
        "maximum_route_delta_to_mean_variance_ratio": max(delta_ratios),
        "records": records,
    }


@torch.no_grad()
def export_format_audit(modules: list[RoutedQuantizedMLP]) -> dict[str, Any]:
    if not modules:
        return {"applicable": False, "valid": True}
    records = []
    valid = True
    for layer_index, module in enumerate(modules):
        for projection, bank in (
            ("gate", module.gate_bank),
            ("up", module.up_bank),
            ("down", module.down_bank),
        ):
            if isinstance(bank, BBCMWeightBank):
                base_scale = bank.base_log_scale.exp().to(torch.bfloat16).to(torch.float32)
                base_codes = torch.round(bank.base_shadow.detach().float() / base_scale).clamp(-127, 127)
                delta_scale = bank.delta_log_scale.exp().to(torch.bfloat16).to(torch.float32)
                normalized = bank.delta_shadow.detach().float() / delta_scale
                delta_codes = torch.sign(normalized) * (normalized.abs() >= 0.5)
                record_valid = (
                    torch.isfinite(base_scale).all()
                    and torch.isfinite(delta_scale).all()
                    and torch.isfinite(base_codes).all()
                    and torch.isfinite(delta_codes).all()
                    and base_codes.min() >= -127
                    and base_codes.max() <= 127
                    and set(delta_codes.unique().cpu().tolist()) <= {-1.0, 0.0, 1.0}
                )
            else:
                scale = bank.log_scale.exp().to(torch.bfloat16).to(torch.float32)
                codes = torch.round(bank.shadow.detach().float() / scale).clamp(-7, 7)
                record_valid = (
                    torch.isfinite(scale).all()
                    and torch.isfinite(codes).all()
                    and codes.min() >= -7
                    and codes.max() <= 7
                )
            record_valid_bool = bool(record_valid)
            valid = valid and record_valid_bool
            records.append({"layer": layer_index, "projection": projection, "valid": record_valid_bool})
    return {
        "applicable": True,
        "valid": valid,
        "scale_storage_dtype": "bfloat16",
        "physical_code_packing_tested": False,
        "records": records,
    }


def serving_ledger(arm: str, dimension: int, original_hidden: int, target_hidden: int, routes: int) -> dict[str, Any]:
    baseline_coordinates = 3 * dimension * original_hidden
    baseline_bytes = baseline_coordinates * 2
    if arm == "baseline":
        return {
            "resident_ffn_bytes_per_layer": baseline_bytes,
            "baseline_ffn_bytes_per_layer": baseline_bytes,
            "matrix_macs_per_token": baseline_coordinates,
            "fits_bytes": True,
            "fits_macs_including_router": True,
        }
    coordinates = 3 * dimension * target_hidden
    scale_values = 2 * target_hidden + dimension
    if arm == "bbcm":
        payload_bits = coordinates * (8 + routes * 2)
        scale_sets = routes + 1
    elif arm == "independent_k4_w4":
        payload_bits = coordinates * routes * 4
        scale_sets = routes
    else:
        raise ValueError(arm)
    router_values = dimension * routes + routes
    candidate_bytes = math.ceil((payload_bits + scale_values * scale_sets * 16 + router_values * 16) / 8)
    candidate_macs = coordinates + dimension * routes
    return {
        "resident_ffn_bytes_per_layer": candidate_bytes,
        "baseline_ffn_bytes_per_layer": baseline_bytes,
        "storage_slack_bytes_per_layer": baseline_bytes - candidate_bytes,
        "selected_matrix_plus_router_macs_per_token": candidate_macs,
        "baseline_matrix_macs_per_token": baseline_coordinates,
        "mac_slack_per_token": baseline_coordinates - candidate_macs,
        "target_hidden": target_hidden,
        "fits_bytes": candidate_bytes <= baseline_bytes,
        "fits_macs_including_router": candidate_macs <= baseline_coordinates,
    }


def validate_protocol(args: argparse.Namespace, arms: tuple[str, ...]) -> dict[str, Any]:
    expected = {
        "arms": VALID_ARMS,
        "device": "NVIDIA H100 80GB HBM3",
        "torch": "2.5.1+cu124",
        "cuda": "12.4",
        "transformers": "4.57.6",
        "steps": 320,
        "eval_steps": [16, 80, 320],
        "warmup_steps": 16,
        "calibration_batch_size": 8,
        "micro_batch_size": 32,
        "gradient_accumulation": 2,
        "eval_batch_size": 32,
        "eval_batches": 128,
        "sequence_length": 512,
        "target_hidden": 1520,
        "routes": 4,
        "shared_lr": 1e-4,
        "router_lr": 5e-4,
        "bbcm_component_lr_scale": 0.5,
        "load_balance_coefficient": 0.01,
        "weight_decay": 0.1,
        "gradient_clip": 1.0,
        "seed": 211,
    }
    actual = {
        "arms": arms,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "steps": args.steps,
        "eval_steps": args.eval_steps,
        "warmup_steps": args.warmup_steps,
        "calibration_batch_size": args.calibration_batch_size,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "eval_batch_size": args.eval_batch_size,
        "eval_batches": args.eval_batches,
        "sequence_length": args.sequence_length,
        "target_hidden": args.target_hidden,
        "routes": args.routes,
        "shared_lr": args.shared_lr,
        "router_lr": args.router_lr,
        "bbcm_component_lr_scale": args.bbcm_component_lr_scale,
        "load_balance_coefficient": args.load_balance_coefficient,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "seed": args.seed,
    }
    checks = {key: actual[key] == value for key, value in expected.items()}
    if args.strict_protocol and not all(checks.values()):
        failed = {key: {"expected": expected[key], "actual": actual[key]} for key, valid in checks.items() if not valid}
        raise ValueError(f"protocol drift: {json.dumps(failed, sort_keys=True)}")
    return {"valid": all(checks.values()), "checks": checks, "expected": expected, "actual": actual}


def train_arm(
    arm: str,
    args: argparse.Namespace,
    train_file: TokenFile,
    validation_file: TokenFile,
    device: torch.device,
    frozen_mask_indices: list[list[int]],
    frozen_mask_sha256: str,
) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=MODEL_REVISION,
        dtype=torch.float32,
        attn_implementation="sdpa",
    ).to(device)
    model.config.use_cache = False
    masks = None
    mask_sha256 = None
    if arm != "baseline":
        masks = [
            torch.tensor(indices, dtype=torch.long, device=device)
            for indices in frozen_mask_indices
        ]
        mask_sha256 = frozen_mask_sha256
    modules = install_arm(
        model,
        arm,
        args.routes,
        args.target_hidden,
        args.warmup_steps,
        masks,
        args.seed,
    )
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(".mlp." in name)
    ledger = serving_ledger(arm, model.config.hidden_size, model.config.intermediate_size, args.target_hidden, args.routes)
    optimizer = make_optimizer(
        model,
        arm,
        args.shared_lr,
        args.router_lr,
        args.bbcm_component_lr_scale,
        args.weight_decay,
    )
    initial_evaluation, initial_utilization = evaluate_with_route_utilization(
        model,
        modules,
        validation_file,
        args.eval_batches,
        args.eval_batch_size,
        device,
    )
    evaluations = {"0": initial_evaluation}
    geometries = {"0": route_weight_geometry(modules)}
    utilizations = {"0": initial_utilization}
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    step_losses = []
    auxiliary_losses = []
    step_seconds = []
    gradient_norms = []
    gradient_diagnostics: dict[str, Any] = {}
    nonfinite = False
    for step in range(args.steps):
        model.train()
        set_routing_step(modules, step)
        multiplier = lr_multiplier(step, args.steps, args.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * multiplier
        started = time.perf_counter()
        accumulated_language_loss = 0.0
        accumulated_auxiliary = 0.0
        for micro in range(args.gradient_accumulation):
            batch_index = step * args.gradient_accumulation + micro
            inputs, targets = train_file.batch(batch_index, args.micro_batch_size, device)
            loss, language_loss, auxiliary = training_loss(
                model, modules, inputs, targets, args.load_balance_coefficient
            )
            (loss / args.gradient_accumulation).backward()
            accumulated_language_loss += float(language_loss.detach()) / args.gradient_accumulation
            accumulated_auxiliary += float(auxiliary.detach()) / args.gradient_accumulation
        if step + 1 in {1, args.warmup_steps + 1, args.steps}:
            gradient_diagnostics[str(step + 1)] = gradient_component_stats(model, arm)
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip))
        if not math.isfinite(gradient_norm) or not math.isfinite(accumulated_language_loss):
            nonfinite = True
            raise RuntimeError(f"nonfinite state in {arm} at step {step + 1}")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step_losses.append(accumulated_language_loss)
        auxiliary_losses.append(accumulated_auxiliary)
        step_seconds.append(time.perf_counter() - started)
        gradient_norms.append(gradient_norm)
        if step + 1 in args.eval_steps:
            set_routing_step(modules, step + 1)
            key = str(step + 1)
            evaluations[key], utilizations[key] = evaluate_with_route_utilization(
                model,
                modules,
                validation_file,
                args.eval_batches,
                args.eval_batch_size,
                device,
            )
            geometries[key] = route_weight_geometry(modules)
            print(
                json.dumps(
                    {
                        "arm": arm,
                        "step": step + 1,
                        "train_loss": accumulated_language_loss,
                        "validation_loss": evaluations[key]["loss"],
                        "route_fractions": utilizations[key].get("fractions"),
                        "mean_route_correlation": geometries[key].get("mean_pairwise_correlation"),
                    }
                ),
                flush=True,
            )
    terminal_key = str(args.steps)
    ablations = routing_ablations(
        model,
        modules,
        evaluations[terminal_key],
        validation_file,
        args.eval_batches,
        args.eval_batch_size,
        device,
        utilizations[terminal_key],
    )
    result = {
        "arm": arm,
        "serving_ledger": ledger,
        "total_training_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "activation_aware_mask_sha256": mask_sha256,
        "evaluations": evaluations,
        "route_weight_geometry": geometries,
        "route_utilization": utilizations,
        "routing_ablations": ablations,
        "export_format_audit": export_format_audit(modules),
        "gradient_diagnostics": gradient_diagnostics,
        "train": {
            "steps": args.steps,
            "prediction_tokens": args.steps * args.gradient_accumulation * args.micro_batch_size * args.sequence_length,
            "mean_language_loss": float(np.mean(step_losses)),
            "final_language_loss": step_losses[-1],
            "mean_auxiliary_loss": float(np.mean(auxiliary_losses)),
            "max_preclip_gradient_norm": max(gradient_norms),
            "nonfinite": nonfinite,
            "elapsed_seconds": sum(step_seconds),
            "tokens_per_second": args.steps * args.gradient_accumulation * args.micro_batch_size * args.sequence_length / sum(step_seconds),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
    }
    del optimizer, modules, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


@torch.no_grad()
def measure_pruned_bf16_endpoint(
    args: argparse.Namespace,
    train_file: TokenFile,
    validation_file: TokenFile,
    device: torch.device,
) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=MODEL_REVISION,
        dtype=torch.float32,
        attn_implementation="sdpa",
    ).to(device)
    model.config.use_cache = False
    calibration_inputs, _ = train_file.batch(0, args.calibration_batch_size, device)
    masks, mask_sha256 = activation_aware_masks(model, calibration_inputs, args.target_hidden)
    baseline = evaluate(model, validation_file, args.eval_batches, args.eval_batch_size, device)
    removed = []
    kept = []
    for layer_index, layer in enumerate(model.model.layers):
        keep = masks[layer_index]
        all_indices = torch.arange(model.config.intermediate_size, device=keep.device)
        removed_mask = torch.ones(model.config.intermediate_size, dtype=torch.bool, device=keep.device)
        removed_mask[keep] = False
        removed.append(all_indices[removed_mask].cpu().tolist())
        kept.append(keep.cpu().tolist())
        layer.mlp = DensePrunedMLP(layer.mlp, keep)
    pruned = evaluate(model, validation_file, args.eval_batches, args.eval_batch_size, device)
    interval = paired_loss_interval(
        {"evaluations": {"endpoint": pruned}},
        {"evaluations": {"endpoint": baseline}},
        "endpoint",
    )
    result = {
        "scope": "untrained activation-aware M1520 BF16 width-loss control",
        "calibration_train_batch_index": 0,
        "calibration_batch_size": args.calibration_batch_size,
        "mask_sha256": mask_sha256,
        "removed_neuron_indices": removed,
        "kept_neuron_indices": kept,
        "baseline": baseline,
        "pruned": pruned,
        "relative_nll_degradation": (pruned["loss"] - baseline["loss"]) / baseline["loss"],
        "paired_interval": interval,
    }
    del masks, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def decide(
    arms: dict[str, Any],
    terminal_step: int,
    pruned_endpoint: dict[str, Any],
    protocol_valid: bool,
    data_valid: bool,
) -> dict[str, Any]:
    if set(arms) != set(VALID_ARMS):
        return {"complete": False, "missing_arms": sorted(set(VALID_ARMS) - set(arms))}
    terminal = str(terminal_step)
    baseline = arms["baseline"]
    bbcm = arms["bbcm"]
    independent = arms["independent_k4_w4"]
    bbcm_vs_baseline = paired_loss_interval(bbcm, baseline, terminal)
    bbcm_vs_independent = paired_loss_interval(bbcm, independent, terminal)
    initial_gap = bbcm["evaluations"]["0"]["loss"] - baseline["evaluations"]["0"]["loss"]
    terminal_gap = bbcm["evaluations"][terminal]["loss"] - baseline["evaluations"][terminal]["loss"]
    geometry = bbcm["route_weight_geometry"][terminal]
    utilization = bbcm["route_utilization"][terminal]
    routing_ablation = bbcm["routing_ablations"]
    learned_minus_shuffled = routing_ablation["learned_minus_best_shuffled_interval"]
    learned_loss = bbcm["evaluations"][terminal]["loss"]
    shuffled_loss = routing_ablation["best_shuffled"]["loss"]
    mask_hashes = {
        bbcm["activation_aware_mask_sha256"],
        independent["activation_aware_mask_sha256"],
        pruned_endpoint["mask_sha256"],
    }
    relative_gain_vs_baseline = -terminal_gap / baseline["evaluations"][terminal]["loss"]
    step_80_gap = bbcm["evaluations"]["80"]["loss"] - baseline["evaluations"]["80"]["loss"]

    def components_live(record: dict[str, Any], categories: set[str]) -> bool:
        return all(
            category in record
            and record[category]["all_finite"]
            and record[category]["nonzero"]
            for category in categories
        )

    gradient_components_valid = (
        components_live(
            baseline["gradient_diagnostics"][terminal], {"dense_ffn"}
        )
        and components_live(
            bbcm["gradient_diagnostics"]["1"],
            {"base_shadow", "delta_shadow", "base_scale", "delta_scale"},
        )
        and components_live(
            bbcm["gradient_diagnostics"][str(16 + 1)],
            {"base_shadow", "delta_shadow", "base_scale", "delta_scale", "router"},
        )
        and components_live(
            bbcm["gradient_diagnostics"][terminal],
            {"base_shadow", "delta_shadow", "base_scale", "delta_scale", "router"},
        )
        and components_live(
            independent["gradient_diagnostics"]["1"], {"w4_shadow", "w4_scale"}
        )
        and components_live(
            independent["gradient_diagnostics"][str(16 + 1)],
            {"w4_shadow", "w4_scale", "router"},
        )
        and components_live(
            independent["gradient_diagnostics"][terminal],
            {"w4_shadow", "w4_scale", "router"},
        )
    )
    gates = {
        "frozen_protocol_and_data_valid": protocol_valid and data_valid,
        "identical_activation_aware_masks": len(mask_hashes) == 1,
        "all_serving_ledgers_fit_exact_bytes_and_macs": all(
            result["serving_ledger"]["fits_bytes"] and result["serving_ledger"]["fits_macs_including_router"]
            for result in arms.values()
        ),
        "finite_training": all(not result["train"]["nonfinite"] for result in arms.values()),
        "bounded_preclip_gradient_norms": all(
            result["train"]["max_preclip_gradient_norm"] <= 100.0 for result in arms.values()
        ),
        "all_required_gradient_components_finite_and_nonzero": gradient_components_valid,
        "exact_export_code_and_bfloat16_scale_semantics": all(
            result["export_format_audit"]["valid"] for result in arms.values()
        ),
        "bbcm_routes_do_not_collapse": utilization["minimum_fraction"] >= 0.05,
        "bbcm_median_normalized_route_entropy_at_least_0p85":
            utilization["median_normalized_layer_entropy"] >= 0.85,
        "bbcm_route_weights_diverge_from_identical_initialization": geometry["mean_pairwise_correlation"] < 0.999999,
        "bbcm_at_least_0p05_percent_better_than_continued_bf16": relative_gain_vs_baseline >= 0.0005,
        "bbcm_advantage_present_at_step_80": step_80_gap < 0.0,
        "paired_95_percent_interval_favors_bbcm_over_continued_bf16": bbcm_vs_baseline["upper_95"] < 0.0,
        "bbcm_terminal_nll_below_independent_k4_w4":
            bbcm["evaluations"][terminal]["loss"] < independent["evaluations"][terminal]["loss"],
        "paired_95_percent_interval_favors_bbcm_over_independent_k4_w4": bbcm_vs_independent["upper_95"] < 0.0,
        "bbcm_beats_independent_k4_w4_by_at_least_0p02_percent":
            (independent["evaluations"][terminal]["loss"] - bbcm["evaluations"][terminal]["loss"])
            / bbcm["evaluations"][terminal]["loss"] >= 0.0002,
        "bbcm_beats_shuffled_routing_by_at_least_0p02_percent":
            (shuffled_loss - learned_loss) / learned_loss >= 0.0002,
        "paired_95_percent_interval_favors_learned_over_shuffled_routing":
            learned_minus_shuffled["upper_95"] < 0.0,
    }
    extension_gate = (
        terminal_gap > 0.0
        and terminal_gap <= 0.0005
        and terminal_gap <= 0.5 * initial_gap
        and bbcm_vs_independent["upper_95"] < 0.0
        and utilization["minimum_fraction"] >= 0.05
    )
    return {
        "complete": True,
        "terminal_step": terminal_step,
        "initial_bbcm_minus_baseline_nll": initial_gap,
        "terminal_bbcm_minus_baseline_nll": terminal_gap,
        "relative_bbcm_gain_vs_baseline": relative_gain_vs_baseline,
        "step_80_bbcm_minus_baseline_nll": step_80_gap,
        "pruned_bf16_endpoint": pruned_endpoint,
        "bbcm_vs_baseline_interval": bbcm_vs_baseline,
        "bbcm_vs_independent_k4_w4_interval": bbcm_vs_independent,
        "gates": gates,
        "breakthrough_screen_pass": all(gates.values()),
        "authorize_one_640_step_extension": extension_gate,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("frozen screen requires H100")
    arms = tuple(item.strip() for item in args.arms.split(",") if item.strip())
    if len(arms) != len(set(arms)) or any(arm not in VALID_ARMS for arm in arms):
        raise ValueError(f"arms must be unique members of {VALID_ARMS}")
    if sha256_file(args.data_manifest) != EXPECTED_DATA_MANIFEST_SHA256:
        raise ValueError("data manifest drift")
    data_ledger = validate_data_ledger(
        args.data_manifest, args.train_file, args.validation_file, args.sequence_length
    )
    train_file = TokenFile(args.train_file, args.sequence_length)
    validation_file = TokenFile(args.validation_file, args.sequence_length)
    protocol = validate_protocol(args, arms)
    if train_file.sequence_count < args.steps * args.gradient_accumulation * args.micro_batch_size:
        raise ValueError("training file too short")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    preregistration = Path("results/bbcm-matched-learning-screen-preregistration.md")
    payload: dict[str, Any] = {
        "candidate": "BBCM exact-budget matched learning screen",
        "scope": "one-seed 320-step upcycling falsification; training resources are not serving resources",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "preregistration_sha256": hashlib.sha256(preregistration.read_bytes()).hexdigest(),
        "data_manifest_sha256": sha256_file(args.data_manifest),
        "data_ledger": data_ledger,
        "protocol": protocol,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "arms": {},
    }
    device = torch.device("cuda")
    payload["pruned_bf16_endpoint"] = measure_pruned_bf16_endpoint(
        args, train_file, validation_file, device
    )
    write_payload(args.output, payload)
    for arm in arms:
        print(json.dumps({"starting_arm": arm}), flush=True)
        payload["arms"][arm] = train_arm(
            arm,
            args,
            train_file,
            validation_file,
            device,
            payload["pruned_bf16_endpoint"]["kept_neuron_indices"],
            payload["pruned_bf16_endpoint"]["mask_sha256"],
        )
        payload["decision"] = decide(
            payload["arms"],
            args.steps,
            payload["pruned_bf16_endpoint"],
            payload["protocol"]["valid"],
            payload["data_ledger"]["valid"],
        )
        write_payload(args.output, payload)
    return payload


def parse_steps(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=Path("data/reflex-lm-screen/train.uint16.bin"))
    parser.add_argument("--validation-file", type=Path, default=Path("data/reflex-lm-screen/validation.uint16.bin"))
    parser.add_argument("--data-manifest", type=Path, default=Path("results/reflex-swiglu-lm-data-manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("results/bbcm-matched-learning-screen.json"))
    parser.add_argument("--arms", default=",".join(VALID_ARMS))
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=128)
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--eval-steps", type=parse_steps, default=parse_steps("16,80,320"))
    parser.add_argument("--warmup-steps", type=int, default=16)
    parser.add_argument("--calibration-batch-size", type=int, default=8)
    parser.add_argument("--target-hidden", type=int, default=1520)
    parser.add_argument("--routes", type=int, default=4)
    parser.add_argument("--shared-lr", type=float, default=1e-4)
    parser.add_argument("--router-lr", type=float, default=5e-4)
    parser.add_argument("--bbcm-component-lr-scale", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--load-balance-coefficient", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--strict-protocol", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps(payload.get("decision", {}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
