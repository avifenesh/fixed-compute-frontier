"""PyTorch model and runner for the frozen learned-routing experiment."""

from __future__ import annotations

import copy
import json
import math
import os
import platform
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

try:
    from experiments.gauge_curvature_learned_routing import (
        CONFIG,
        CONFIRMATION_SEEDS,
        PREREGISTRATION_PATH,
        SCREEN_SEEDS,
        VARIANTS,
        WORLD_SEEDS,
        ExperimentConfig,
        all_ledgers,
        data_invariants,
        generate_scenes,
        screen_decision,
        source_hashes,
        variant_ledger,
    )
except ModuleNotFoundError as error:
    if error.name != "experiments":
        raise
    from gauge_curvature_learned_routing import (
        CONFIG,
        CONFIRMATION_SEEDS,
        PREREGISTRATION_PATH,
        SCREEN_SEEDS,
        VARIANTS,
        WORLD_SEEDS,
        ExperimentConfig,
        all_ledgers,
        data_invariants,
        generate_scenes,
        screen_decision,
        source_hashes,
        variant_ledger,
    )


class RMSNorm(nn.Module):
    def __init__(self, width: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        source_dtype = values.dtype
        normalized = values.float()
        normalized = normalized * torch.rsqrt(
            normalized.square().mean(dim=-1, keepdim=True) + self.epsilon
        )
        return normalized.to(source_dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, width: int, hidden_width: int) -> None:
        super().__init__()
        self.gate = nn.Linear(width, hidden_width, bias=False)
        self.up = nn.Linear(width, hidden_width, bias=False)
        self.down = nn.Linear(hidden_width, width, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(values)) * self.up(values))


def _allowed_sources(width: int, kind: str) -> torch.Tensor:
    rows = []
    for output in range(width):
        forbidden = (output + 1) % width if kind == "cyclic" else output
        rows.append([source for source in range(width) if source != forbidden])
    return torch.tensor(rows, dtype=torch.long)


class LearnedAttention(nn.Module):
    def __init__(self, variant: str, config: ExperimentConfig) -> None:
        super().__init__()
        self.variant = variant
        self.config = config
        d = config.model_width
        r = config.head_width
        hq = config.query_heads
        hkv = config.kv_heads
        self.value_width = 12 if variant == "glu" else r
        self.q_proj = nn.Linear(d, hq * r, bias=False)
        self.k_proj = nn.Linear(d, hkv * r, bias=False)

        if variant in {"dense", "g2_full", "g1"}:
            self.v_proj = nn.Linear(d, hkv * r, bias=False)
        elif variant == "glu":
            self.value_a = nn.Linear(d, hkv * self.value_width, bias=False)
            self.value_g = nn.Linear(d, hkv * self.value_width, bias=False)
        else:
            self.b = nn.Parameter(torch.empty(hkv, d - r, r))

        if variant == "cyclic":
            self.alpha = nn.Parameter(torch.empty(hkv, r))
            self.h = nn.Parameter(torch.empty(hkv, r, r - 1))
            self.register_buffer(
                "allowed", _allowed_sources(r, "cyclic"), persistent=False
            )
        elif variant == "square":
            self.h = nn.Parameter(torch.empty(hkv, r, r - 1))
            self.register_buffer(
                "allowed", _allowed_sources(r, "square"), persistent=False
            )
        elif variant == "source_mlp":
            self.source_a = nn.Parameter(torch.empty(d, 5))
            self.source_c = nn.Parameter(torch.empty(5, hkv * r))
            self.source_alpha = nn.Parameter(torch.empty(hkv, r))
        elif variant == "g2_matched":
            self.gate_a = nn.Parameter(torch.empty(d, 5))
            self.gate_c = nn.Parameter(torch.empty(5, hkv * r))
            self.gate_bias = nn.Parameter(torch.empty(hkv * r))
        elif variant == "g2_full":
            self.gate_proj = nn.Linear(d, hkv * r, bias=False)
        elif variant == "g1":
            self.output_gate = nn.Linear(d, hq * r, bias=False)

        self.o_proj = nn.Linear(hq * self.value_width, d, bias=False)

    def _bda_parts(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r = self.config.head_width
        z = values[..., :r]
        rest = values[..., r:]
        rest_projection = torch.einsum("btd,hdr->bthr", rest, self.b)
        return z, rest_projection, z[:, :, None, :] + rest_projection

    def _compact_projection(self, z: torch.Tensor) -> torch.Tensor:
        selected = z[:, :, self.allowed]
        return torch.einsum("btok,hok->btho", selected, self.h)

    def values(self, hidden: torch.Tensor) -> torch.Tensor:
        bsz, length, _ = hidden.shape
        hkv = self.config.kv_heads
        if self.variant in {"dense", "g1"}:
            return self.v_proj(hidden).view(bsz, length, hkv, self.value_width)
        if self.variant == "g2_full":
            value = self.v_proj(hidden)
            gate = torch.sigmoid(self.gate_proj(hidden))
            return (value * gate).view(bsz, length, hkv, self.value_width)
        if self.variant == "glu":
            value = self.value_a(hidden) * F.silu(self.value_g(hidden))
            return value.view(bsz, length, hkv, self.value_width)

        z, rest_projection, bda = self._bda_parts(hidden)
        if self.variant == "bda":
            return bda
        if self.variant == "cyclic":
            interaction = self._compact_projection(z)
            return rest_projection + z[:, :, None, :] * (
                self.alpha[None, None, :, :] + interaction
            )
        if self.variant == "square":
            return bda + self._compact_projection(z).square()
        if self.variant == "source_mlp":
            feature = F.silu(hidden @ self.source_a) @ self.source_c
            linear = rest_projection + z[:, :, None, :] * self.source_alpha
            return linear + feature.view(bsz, length, hkv, self.value_width)
        if self.variant == "g2_matched":
            gate = (hidden @ self.gate_a) @ self.gate_c + self.gate_bias
            gate = torch.sigmoid(gate)
            return bda * gate.view(bsz, length, hkv, self.value_width)
        raise AssertionError(f"unimplemented variant: {self.variant}")

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, length, _ = hidden.shape
        hq = self.config.query_heads
        hkv = self.config.kv_heads
        r = self.config.head_width
        groups = hq // hkv
        query = self.q_proj(hidden).view(bsz, length, hq, r).transpose(1, 2)
        key = self.k_proj(hidden).view(bsz, length, hkv, r).transpose(1, 2)
        value = self.values(hidden).transpose(1, 2)
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(r)
        causal = torch.ones((length, length), dtype=torch.bool, device=hidden.device).tril()
        scores = scores.masked_fill(~causal[None, None, :, :], -torch.inf)
        attention = torch.softmax(scores.float(), dim=-1).to(query.dtype)
        output = torch.matmul(attention, value)
        if self.variant == "g1":
            gate = self.output_gate(hidden).view(bsz, length, hq, r).transpose(1, 2)
            output = output * torch.sigmoid(gate)
        output = output.transpose(1, 2).reshape(bsz, length, -1)
        return self.o_proj(output), attention


class TransformerBlock(nn.Module):
    def __init__(self, variant: str, config: ExperimentConfig) -> None:
        super().__init__()
        ledger = variant_ledger(variant)
        self.attention_norm = RMSNorm(config.model_width)
        self.attention = LearnedAttention(variant, config)
        self.ffn_norm = RMSNorm(config.model_width)
        self.ffn = SwiGLU(config.model_width, ledger["ffn_width"])

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        update, attention = self.attention(self.attention_norm(hidden))
        hidden = hidden + update
        hidden = hidden + self.ffn(self.ffn_norm(hidden))
        return hidden, attention


class AddressedBagTransformer(nn.Module):
    def __init__(self, variant: str, config: ExperimentConfig = CONFIG) -> None:
        super().__init__()
        self.variant = variant
        self.config = config
        self.blocks = nn.ModuleList(
            [TransformerBlock(variant, config) for _ in range(config.layers)]
        )
        self.final_norm = RMSNorm(config.model_width)
        self.classifier = nn.Linear(config.model_width, 1, bias=False)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = inputs
        final_attention = None
        for block in self.blocks:
            hidden, final_attention = block(hidden)
        if final_attention is None:
            raise AssertionError("model has no Transformer blocks")
        queries = self.final_norm(hidden[:, -self.config.query_tokens :, :])
        logits = self.classifier(queries).squeeze(-1)
        return logits, final_attention


def _normal(
    shape: tuple[int, ...], generator: torch.Generator, scale: float = 0.02
) -> torch.Tensor:
    return torch.randn(shape, generator=generator, dtype=torch.float32) * scale


def _layer_blueprint(seed: int, layer: int, config: ExperimentConfig) -> dict[str, Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1009 * (layer + 1))
    d = config.model_width
    r = config.head_width
    hq = config.query_heads
    hkv = config.kv_heads
    projection_scale = 0.02
    pivot_scale = projection_scale * math.sqrt(r)
    dense_value = torch.empty((hkv * r, d), dtype=torch.float32)
    b_heads = []
    pivots = []
    for head in range(hkv):
        raw = _normal((r, r), generator, 1.0)
        orthogonal, _ = torch.linalg.qr(raw)
        pivot = orthogonal * pivot_scale
        rest = _normal((d - r, r), generator, projection_scale)
        value_matrix = torch.cat((pivot, rest), dim=0)
        dense_value[head * r : (head + 1) * r] = value_matrix.T
        b_heads.append(rest @ torch.linalg.inv(pivot))
        pivots.append(pivot)
    dense_output = _normal((d, hq * r), generator, projection_scale)
    gauge_output = dense_output.clone()
    groups = hq // hkv
    for query_head in range(hq):
        kv_head = query_head // groups
        columns = slice(query_head * r, (query_head + 1) * r)
        gauge_output[:, columns] = dense_output[:, columns] @ pivots[kv_head].T
    maximum_ffn = config.base_ffn_width
    return {
        "q": _normal((hq * r, d), generator, projection_scale),
        "k": _normal((hkv * r, d), generator, projection_scale),
        "dense_value": dense_value,
        "b": torch.stack(b_heads),
        "dense_output": dense_output,
        "gauge_output": gauge_output,
        "mlp_gate": _normal((maximum_ffn, d), generator, projection_scale),
        "mlp_up": _normal((maximum_ffn, d), generator, projection_scale),
        "mlp_down": _normal((d, maximum_ffn), generator, projection_scale),
        "source_a": _normal((d, 5), generator, projection_scale),
        "matched_gate_a": _normal((d, 5), generator, projection_scale),
        "square_h": _normal((hkv, r, r - 1), generator, 1e-3),
        "glu_a": _normal((hkv * 12, d), generator, projection_scale),
        "glu_g": _normal((hkv * 12, d), generator, projection_scale),
        "glu_output": _normal((d, hq * 12), generator, projection_scale),
    }


@torch.no_grad()
def initialize_model(model: AddressedBagTransformer, seed: int) -> None:
    config = model.config
    for layer_index, block in enumerate(model.blocks):
        blueprint = _layer_blueprint(seed, layer_index, config)
        attention = block.attention
        attention.q_proj.weight.copy_(blueprint["q"])
        attention.k_proj.weight.copy_(blueprint["k"])
        if model.variant in {"dense", "g2_full", "g1"}:
            attention.v_proj.weight.copy_(blueprint["dense_value"])
            output_scale = 2.0 if model.variant in {"g2_full", "g1"} else 1.0
            attention.o_proj.weight.copy_(blueprint["dense_output"] * output_scale)
        elif model.variant == "glu":
            attention.value_a.weight.copy_(blueprint["glu_a"])
            attention.value_g.weight.copy_(blueprint["glu_g"])
            attention.o_proj.weight.copy_(blueprint["glu_output"])
        else:
            attention.b.copy_(blueprint["b"])
            output_scale = 2.0 if model.variant == "g2_matched" else 1.0
            attention.o_proj.weight.copy_(blueprint["gauge_output"] * output_scale)

        if model.variant == "cyclic":
            attention.alpha.fill_(1.0)
            attention.h.zero_()
        elif model.variant == "square":
            attention.h.copy_(blueprint["square_h"])
        elif model.variant == "source_mlp":
            attention.source_a.copy_(blueprint["source_a"])
            attention.source_c.zero_()
            attention.source_alpha.fill_(1.0)
        elif model.variant == "g2_matched":
            attention.gate_a.copy_(blueprint["matched_gate_a"])
            attention.gate_c.zero_()
            attention.gate_bias.zero_()
        elif model.variant == "g2_full":
            attention.gate_proj.weight.zero_()
        elif model.variant == "g1":
            attention.output_gate.weight.zero_()

        ffn_width = block.ffn.gate.out_features
        block.ffn.gate.weight.copy_(blueprint["mlp_gate"][:ffn_width])
        block.ffn.up.weight.copy_(blueprint["mlp_up"][:ffn_width])
        block.ffn.down.weight.copy_(blueprint["mlp_down"][:, :ffn_width])
        block.attention_norm.weight.fill_(1.0)
        block.ffn_norm.weight.fill_(1.0)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 99991)
    model.final_norm.weight.fill_(1.0)
    model.classifier.weight.copy_(
        _normal((1, config.model_width), generator, 0.02)
    )


def build_model(
    variant: str, seed: int, config: ExperimentConfig = CONFIG
) -> AddressedBagTransformer:
    torch.manual_seed(seed)
    model = AddressedBagTransformer(variant, config)
    initialize_model(model, seed)
    expected = variant_ledger(variant)["actual_trainable_parameters_expected"]
    actual = sum(parameter.numel() for parameter in model.parameters())
    if actual != expected:
        raise AssertionError(
            f"{variant} parameter ledger mismatch: expected {expected}, got {actual}"
        )
    return model


class TorchScenes:
    def __init__(self, data: dict[str, np.ndarray], device: torch.device) -> None:
        self.device = device
        self.source_content = torch.from_numpy(data["source_content"]).to(device)
        self.source_bag = torch.from_numpy(data["source_bag"]).to(device)
        self.addresses = torch.from_numpy(data["addresses"]).to(device)
        self.query_bag = torch.from_numpy(data["query_bag"]).to(device)
        self.labels = torch.from_numpy(data["labels"]).to(device)
        self.oracle_logits = torch.from_numpy(data["oracle_logits"]).to(device)
        self.size = self.labels.shape[0]

    def batch(
        self, indices: torch.Tensor, *, broken_query_address: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        config = CONFIG
        content = self.source_content.index_select(0, indices)
        source_bag = self.source_bag.index_select(0, indices)
        addresses = self.addresses.index_select(0, indices)
        query_bag = self.query_bag.index_select(0, indices)
        labels = self.labels.index_select(0, indices)
        input_query_bag = (
            torch.roll(query_bag, shifts=1, dims=1)
            if broken_query_address
            else query_bag
        )
        batch = indices.numel()
        inputs = torch.zeros(
            (batch, config.sequence_length, config.model_width),
            dtype=torch.float32,
            device=self.device,
        )
        inputs[:, : config.source_tokens, : config.content_width] = content
        scene = torch.arange(batch, device=self.device)[:, None]
        source_addresses = addresses[scene, source_bag]
        query_addresses = addresses[scene, input_query_bag]
        inputs[
            :, : config.source_tokens, config.content_width : 2 * config.content_width
        ] = config.address_scale * source_addresses
        inputs[
            :, config.source_tokens :, config.content_width : 2 * config.content_width
        ] = config.address_scale * query_addresses
        inputs[:, : config.source_tokens, 2 * config.content_width] = 1.0
        inputs[:, config.source_tokens :, 2 * config.content_width + 1] = 1.0
        return inputs, labels, source_bag, query_bag


def routing_statistics(
    attention: torch.Tensor,
    source_bag: torch.Tensor,
    query_bag: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    config = CONFIG
    query_attention = attention[
        :, :, -config.query_tokens :, : config.source_tokens
    ]
    bag_masses = []
    for bag in range(config.bags):
        mask = (source_bag == bag)[:, None, None, :]
        bag_masses.append((query_attention * mask).sum(dim=-1))
    masses = torch.stack(bag_masses, dim=-1)
    target = query_bag[:, None, :, None].expand(
        -1, config.query_heads, -1, 1
    )
    target_mass = masses.gather(dim=-1, index=target).squeeze(-1)
    argmax_accuracy = (masses.argmax(dim=-1) == query_bag[:, None, :]).float()
    return target_mass, argmax_accuracy


@torch.no_grad()
def evaluate(
    model: AddressedBagTransformer,
    data: TorchScenes,
    *,
    batch_size: int = 256,
    broken_query_address: bool = False,
) -> dict[str, float]:
    model.eval()
    nll_sum = 0.0
    correct = 0.0
    examples = 0
    target_mass_sum = 0.0
    route_correct = 0.0
    route_count = 0
    device_type = data.device.type
    for start in range(0, data.size, batch_size):
        indices = torch.arange(
            start, min(start + batch_size, data.size), device=data.device
        )
        inputs, labels, source_bag, query_bag = data.batch(
            indices, broken_query_address=broken_query_address
        )
        with torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=device_type == "cuda",
        ):
            logits, attention = model(inputs)
        logits = logits.float()
        nll_sum += float(
            F.binary_cross_entropy_with_logits(logits, labels, reduction="sum")
        )
        correct += float(((logits >= 0.0) == (labels >= 0.5)).sum())
        examples += labels.numel()
        if not broken_query_address:
            target_mass, route_accuracy = routing_statistics(
                attention.float(), source_bag, query_bag
            )
            target_mass_sum += float(target_mass.sum())
            route_correct += float(route_accuracy.sum())
            route_count += target_mass.numel()
    result = {
        "nll": nll_sum / examples,
        "accuracy": correct / examples,
    }
    if not broken_query_address:
        result.update(
            {
                "target_bag_mass": target_mass_sum / route_count,
                "bag_argmax_accuracy": route_correct / route_count,
            }
        )
    return result


def _learning_rate(step: int, config: ExperimentConfig) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    progress = (step - config.warmup_steps) / max(
        config.steps - config.warmup_steps - 1, 1
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return config.minimum_learning_rate + (
        config.learning_rate - config.minimum_learning_rate
    ) * cosine


def _parameter_group(name: str) -> str:
    if ".attention.q_proj." in name or ".attention.k_proj." in name:
        return "qk"
    if ".attention.o_proj." in name:
        return "output"
    if ".attention." in name:
        return "value_and_attention_gates"
    if ".ffn." in name:
        return "ffn"
    return "norm_and_classifier"


def _group_norms(
    model: nn.Module, reference: dict[str, torch.Tensor] | None = None
) -> dict[str, float]:
    squared: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        value = parameter.detach().float().cpu()
        if reference is not None:
            value = value - reference[name]
        group = _parameter_group(name)
        squared[group] = squared.get(group, 0.0) + float(value.square().sum())
    return {group: math.sqrt(value) for group, value in sorted(squared.items())}


def train_one(
    *,
    variant: str,
    init_seed: int,
    world_seed: int,
    train: TorchScenes,
    validation: TorchScenes,
    test: TorchScenes,
    batch_indices: np.ndarray,
    config: ExperimentConfig,
) -> dict[str, Any]:
    device = train.device
    model = build_model(variant, init_seed, config).to(device)
    expected_parameters = variant_ledger(variant)[
        "actual_trainable_parameters_expected"
    ]
    actual_parameters = sum(parameter.numel() for parameter in model.parameters())
    initial_state = {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
    }
    initial_parameter_l2_by_group = _group_norms(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
    )
    history = []
    device_type = device.type
    if device_type == "cuda":
        torch.cuda.synchronize(device)
        allocation_floor = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        allocation_floor = 0
    start_time = time.perf_counter()
    model.train()
    final_training_loss = math.nan
    for step in range(config.steps):
        learning_rate = _learning_rate(step, config)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        indices = torch.from_numpy(batch_indices[step]).to(device=device)
        inputs, labels, source_bag, query_bag = train.batch(indices)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=device_type == "cuda",
        ):
            logits, attention = model(inputs)
            bce = F.binary_cross_entropy_with_logits(logits.float(), labels)
            target_mass, _ = routing_statistics(
                attention.float(), source_bag, query_bag
            )
            route_loss = -torch.log(target_mass.clamp_min(1e-8)).mean()
            loss = bce + config.route_loss_weight * route_loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                f"non-finite loss for {variant}, world={world_seed}, step={step}"
            )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip
        )
        optimizer.step()
        final_training_loss = float(loss.detach())
        if (step + 1) % config.validation_interval == 0 or step + 1 == config.steps:
            validation_metrics = evaluate(model, validation)
            history.append(
                {
                    "step": step + 1,
                    "learning_rate": learning_rate,
                    "training_loss": final_training_loss,
                    "gradient_norm": float(gradient_norm),
                    "validation_nll": validation_metrics["nll"],
                    "validation_accuracy": validation_metrics["accuracy"],
                    "validation_target_bag_mass": validation_metrics[
                        "target_bag_mass"
                    ],
                }
            )
            model.train()
    if device_type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - start_time
    test_metrics = evaluate(model, test)
    broken_metrics = evaluate(model, test, broken_query_address=True)
    final_parameter_l2_by_group = _group_norms(model)
    parameter_update_l2_by_group = _group_norms(model, initial_state)
    if device_type == "cuda":
        peak_bytes = max(
            0, torch.cuda.max_memory_allocated(device) - allocation_floor
        )
    else:
        peak_bytes = 0
    result = {
        "variant": variant,
        "world_seed": world_seed,
        "init_seed": init_seed,
        "final_training_loss": final_training_loss,
        "validation_history": history,
        "test_nll": test_metrics["nll"],
        "test_accuracy": test_metrics["accuracy"],
        "test_target_bag_mass": test_metrics["target_bag_mass"],
        "test_bag_argmax_accuracy": test_metrics["bag_argmax_accuracy"],
        "broken_address_nll": broken_metrics["nll"],
        "broken_address_accuracy": broken_metrics["accuracy"],
        "trainable_parameters": actual_parameters,
        "parameter_ledger_matches": actual_parameters == expected_parameters,
        "initial_parameter_l2_by_group": initial_parameter_l2_by_group,
        "final_parameter_l2_by_group": final_parameter_l2_by_group,
        "parameter_update_l2_by_group": parameter_update_l2_by_group,
        "peak_incremental_allocated_bytes": peak_bytes,
        "training_seconds": training_seconds,
        "training_scenes_per_second": config.steps
        * config.batch_size
        / training_seconds,
        "training_queries_per_second": config.steps
        * config.batch_size
        * config.query_tokens
        / training_seconds,
    }
    del optimizer, model
    if device_type == "cuda":
        torch.cuda.empty_cache()
    return result


def address_only_ridge_accuracy(
    train: dict[str, np.ndarray], test: dict[str, np.ndarray]
) -> float:
    def features(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        scenes = data["query_bag"].shape[0]
        row = np.arange(scenes)[:, None]
        values = data["addresses"][row, data["query_bag"]].reshape(
            -1, CONFIG.address_width
        )
        intercept = np.ones((values.shape[0], 1), dtype=np.float32)
        labels = (2.0 * data["labels"] - 1.0).reshape(-1)
        return np.concatenate((values, intercept), axis=1).astype(np.float64), labels

    train_x, train_y = features(train)
    test_x, test_y = features(test)
    gram = train_x.T @ train_x
    gram += 1e-3 * np.eye(gram.shape[0])
    weights = np.linalg.solve(gram, train_x.T @ train_y)
    return float(np.mean((test_x @ weights >= 0.0) == (test_y >= 0.0)))


def _atomic_persist(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)


def self_test(device_name: str = "cpu") -> dict[str, Any]:
    device = torch.device(device_name)
    parameter_counts = {}
    for variant in VARIANTS:
        model = build_model(variant, SCREEN_SEEDS[0]).to(device)
        parameter_counts[variant] = sum(p.numel() for p in model.parameters())
        del model

    generator = torch.Generator(device="cpu")
    generator.manual_seed(424242)
    sample = torch.randn((3, 19, CONFIG.model_width), generator=generator).to(device)
    dense = build_model("dense", SCREEN_SEEDS[0]).to(device).eval()
    with torch.no_grad():
        dense_logits, dense_attention = dense(sample)
    equivalence = {}
    for variant in ("bda", "cyclic", "source_mlp", "g2_matched"):
        model = build_model(variant, SCREEN_SEEDS[0]).to(device).eval()
        with torch.no_grad():
            logits, attention = model(sample)
        error = max(
            float((logits - dense_logits).abs().max()),
            float((attention - dense_attention).abs().max()),
        )
        equivalence[variant] = error
        if error > 2e-5:
            raise AssertionError(f"gauge-equivalent initialization failed for {variant}: {error}")
        del model

    gradient_norms = {}
    for variant, parameter_name in (
        ("cyclic", "h"),
        ("square", "h"),
        ("source_mlp", "source_c"),
        ("g2_matched", "gate_c"),
    ):
        model = build_model(variant, SCREEN_SEEDS[0]).to(device)
        logits, _ = model(sample)
        logits.square().mean().backward()
        parameter = getattr(model.blocks[0].attention, parameter_name)
        gradient_norm = float(parameter.grad.float().norm())
        gradient_norms[variant] = gradient_norm
        if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
            raise AssertionError(f"dead initialization for {variant}")
        del model

    tiny = generate_scenes(
        scenes=32, world_seed=19001, split_seed=1900101
    )
    invariants = data_invariants(tiny)
    if invariants["paired_content_mean_max_abs"] > 1e-6:
        raise AssertionError("paired dataset lost its exact zero mean")
    if invariants["balanced_labels_max_abs_error"] != 0.0:
        raise AssertionError("scene labels are not exactly balanced")
    return {
        "status": "pass",
        "device": str(device),
        "parameter_counts": parameter_counts,
        "equivalent_initialization_max_abs_errors": equivalence,
        "initial_gradient_norms": gradient_norms,
        "tiny_data_invariants": invariants,
    }


def _environment(device: torch.device) -> dict[str, Any]:
    result = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "hostname": platform.node(),
    }
    if device.type == "cuda":
        result.update(
            {
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_capability": list(torch.cuda.get_device_capability(device)),
                "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            }
        )
    return result


def run_suite(
    *,
    output: Path,
    device: torch.device,
    variants: tuple[str, ...],
    world_seeds: tuple[int, ...],
    init_seeds: tuple[int, ...],
    config: ExperimentConfig,
    mode: str,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "gauge_curvature_learned_routing",
        "mode": mode,
        "preregistration": str(PREREGISTRATION_PATH.relative_to(PREREGISTRATION_PATH.parents[1])),
        "source_sha256": source_hashes(),
        "environment": _environment(device),
        "config": asdict(config),
        "variants": list(variants),
        "world_seeds": list(world_seeds),
        "init_seeds": list(init_seeds),
        "resource_ledgers": all_ledgers(),
        "self_test": self_test(str(device)),
        "worlds": {},
        "runs": [],
        "leakage_controls": {
            "uniform_all_bags_accuracy": 0.5,
            "uniform_all_bags_nll": math.log(2.0),
        },
    }
    address_accuracies = []
    for world_seed in world_seeds:
        train_numpy = generate_scenes(
            scenes=config.train_scenes,
            world_seed=world_seed,
            split_seed=world_seed * 100 + 1,
        )
        validation_numpy = generate_scenes(
            scenes=config.validation_scenes,
            world_seed=world_seed,
            split_seed=world_seed * 100 + 2,
        )
        test_numpy = generate_scenes(
            scenes=config.test_scenes,
            world_seed=world_seed,
            split_seed=world_seed * 100 + 3,
        )
        train_invariants = data_invariants(train_numpy)
        validation_invariants = data_invariants(validation_numpy)
        test_invariants = data_invariants(test_numpy)
        address_accuracy = address_only_ridge_accuracy(train_numpy, test_numpy)
        address_accuracies.append(address_accuracy)
        report["worlds"][str(world_seed)] = {
            "train_invariants": train_invariants,
            "validation_invariants": validation_invariants,
            "test_invariants": test_invariants,
            "address_only_ridge_accuracy": address_accuracy,
        }
        train = TorchScenes(train_numpy, device)
        validation = TorchScenes(validation_numpy, device)
        test = TorchScenes(test_numpy, device)
        del train_numpy, validation_numpy, test_numpy
        for init_seed in init_seeds:
            schedule_rng = np.random.default_rng(world_seed * 1_000_003 + init_seed)
            batch_indices = schedule_rng.integers(
                0,
                train.size,
                size=(config.steps, config.batch_size),
                dtype=np.int64,
            )
            for variant in variants:
                run = train_one(
                    variant=variant,
                    init_seed=init_seed,
                    world_seed=world_seed,
                    train=train,
                    validation=validation,
                    test=test,
                    batch_indices=batch_indices,
                    config=config,
                )
                report["runs"].append(run)
                _atomic_persist(report, output)
        del train, validation, test
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report["leakage_controls"]["address_only_accuracy"] = float(
        np.mean(address_accuracies)
    )
    cyclic_broken = [
        float(run["broken_address_accuracy"])
        for run in report["runs"]
        if run["variant"] == "cyclic"
    ]
    report["leakage_controls"]["broken_address_accuracy"] = (
        float(np.mean(cyclic_broken)) if cyclic_broken else math.nan
    )
    if (
        mode == "screen"
        and variants == VARIANTS
        and world_seeds == WORLD_SEEDS
        and init_seeds == SCREEN_SEEDS
    ):
        report["decision"] = screen_decision(report)
    else:
        report["decision"] = {"status": "diagnostic_only"}
    _atomic_persist(report, output)
    return report


def run_cli(args: Any) -> dict[str, Any]:
    if args.mode == "self-test":
        return self_test(args.device)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    variants = tuple(args.variants)
    world_seeds = tuple(args.world_seeds)
    init_seeds = tuple(args.init_seeds)
    config = CONFIG
    if args.mode == "smoke":
        world_seeds = (19001,)
        init_seeds = SCREEN_SEEDS
        config = replace(
            CONFIG,
            train_scenes=512,
            validation_scenes=256,
            test_scenes=256,
            batch_size=64,
            steps=30,
            validation_interval=15,
            warmup_steps=5,
        )
    elif args.mode == "confirm" and init_seeds == SCREEN_SEEDS:
        init_seeds = CONFIRMATION_SEEDS
    return run_suite(
        output=args.output,
        device=device,
        variants=variants,
        world_seeds=world_seeds,
        init_seeds=init_seeds,
        config=config,
        mode=args.mode,
    )
