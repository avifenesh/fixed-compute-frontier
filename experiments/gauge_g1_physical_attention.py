#!/usr/bin/env python3
"""Physical-weight, scale-gauge-funded G1 attention for norm-free RoPE."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.models.llama.modeling_llama import (
    ALL_ATTENTION_FUNCTIONS,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


VALID_ARMS = ("polar_bilinear_control", "physical_scale_g1")


class PhysicalScaleG1Attention(nn.Module):
    """Llama attention with no extra learned or deployed weight values.

    Two ordinary K weights per RoPE pair are stored during training as a polar
    chart `(phi, s)`. At deployment they materialize back into the ordinary K
    matrix. The candidate derives `c=tanh(s)` from their physical radius and
    applies `odd += c * even**2` before RoPE.
    """

    def __init__(self, source: nn.Module, arm: str) -> None:
        super().__init__()
        if arm not in VALID_ARMS:
            raise ValueError(f"invalid arm: {arm}")
        if source.q_proj.bias is not None or source.k_proj.bias is not None:
            raise ValueError("physical G1 prototype requires bias-free Q/K")
        self.arm = arm
        self.config = source.config
        self.layer_idx = source.layer_idx
        self.head_dim = source.head_dim
        if self.head_dim % 2:
            raise ValueError("RoPE head dimension must be even")
        self.pairs = self.head_dim // 2
        self.query_heads = source.config.num_attention_heads
        self.kv_heads = source.config.num_key_value_heads
        self.num_key_value_groups = self.query_heads // self.kv_heads
        self.hidden_size = source.q_proj.in_features
        self.kv_dimension = self.kv_heads * self.head_dim
        self.scaling = source.scaling
        self.attention_dropout = source.attention_dropout
        self.is_causal = True
        self.tau = 1.0 / math.sqrt(self.head_dim)

        query_weight, key_weight, pivot_columns, polar = self._gauge_initialize(
            source.q_proj.weight.detach().float(),
            source.k_proj.weight.detach().float(),
        )
        pivot_indices = []
        for kv_head in range(self.kv_heads):
            for pair in range(self.pairs):
                even_row = kv_head * self.head_dim + pair
                odd_row = even_row + self.pairs
                column = int(pivot_columns[kv_head, pair])
                pivot_indices.extend((
                    even_row * self.hidden_size + column,
                    odd_row * self.hidden_size + column,
                ))
        pivot_indices_tensor = torch.tensor(pivot_indices, dtype=torch.long)
        full_indices = torch.arange(self.kv_dimension * self.hidden_size)
        nonpivot_mask = torch.ones_like(full_indices, dtype=torch.bool)
        nonpivot_mask[pivot_indices_tensor] = False
        nonpivot_indices = full_indices[nonpivot_mask]

        self.query_weight = nn.Parameter(query_weight.to(source.q_proj.weight.dtype))
        self.key_nonpivot = nn.Parameter(
            key_weight.reshape(-1)[nonpivot_indices].to(source.k_proj.weight.dtype)
        )
        self.polar = nn.Parameter(polar.to(source.k_proj.weight.dtype))
        # The chosen atlas is real compiler metadata. The quality prototype
        # keeps it as a non-persistent runtime buffer; a serving kernel must
        # count or specialize these index bits explicitly.
        self.register_buffer("pivot_columns", pivot_columns, persistent=False)
        self.register_buffer("pivot_indices", pivot_indices_tensor, persistent=False)
        self.register_buffer("nonpivot_indices", nonpivot_indices, persistent=False)
        self.v_proj = source.v_proj
        self.o_proj = source.o_proj
        self.record_diagnostics = False
        self.last_diagnostics: dict[str, float] | None = None

    def _gauge_initialize(
        self,
        query_weight: torch.Tensor,
        key_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        query = query_weight.clone().reshape(
            self.query_heads, self.head_dim, self.hidden_size
        )
        key = key_weight.clone().reshape(
            self.kv_heads, self.head_dim, self.hidden_size
        )
        pivot_columns = torch.empty(
            self.kv_heads, self.pairs, dtype=torch.long
        )
        polar = torch.zeros(self.kv_heads, self.pairs, 2, dtype=torch.float32)
        for kv_head in range(self.kv_heads):
            for pair in range(self.pairs):
                even_row = pair
                odd_row = pair + self.pairs
                radii = torch.sqrt(
                    key[kv_head, even_row].square()
                    + key[kv_head, odd_row].square()
                )
                finite = torch.isfinite(radii) & (radii > 1e-12)
                if not bool(finite.any()):
                    raise ValueError("K pair has no finite nonzero pivot candidate")
                distance = torch.where(
                    finite,
                    torch.abs(radii - self.tau),
                    torch.full_like(radii, torch.inf),
                )
                column = int(torch.argmin(distance))
                radius = float(radii[column])
                scale = self.tau / radius
                key[kv_head, (even_row, odd_row)] *= scale
                first_query_head = kv_head * self.num_key_value_groups
                last_query_head = first_query_head + self.num_key_value_groups
                query[first_query_head:last_query_head, (even_row, odd_row)] /= scale
                even_pivot = float(key[kv_head, even_row, column])
                odd_pivot = float(key[kv_head, odd_row, column])
                polar[kv_head, pair, 0] = math.atan2(even_pivot, odd_pivot)
                polar[kv_head, pair, 1] = 0.0
                pivot_columns[kv_head, pair] = column
        return (
            query.reshape(self.query_heads * self.head_dim, self.hidden_size),
            key.reshape(self.kv_dimension, self.hidden_size),
            pivot_columns,
            polar,
        )

    def pivot_values(self) -> torch.Tensor:
        phi = self.polar[..., 0]
        log_scale = self.polar[..., 1]
        radius = self.tau * torch.exp(log_scale)
        return torch.stack(
            (radius * torch.sin(phi), radius * torch.cos(phi)), dim=-1
        )

    def curvature_coefficients(self) -> torch.Tensor:
        return torch.tanh(self.polar[..., 1])

    def materialize_key_weight(self) -> torch.Tensor:
        flat = self.key_nonpivot.new_zeros(self.kv_dimension * self.hidden_size)
        flat = flat.scatter(0, self.nonpivot_indices, self.key_nonpivot)
        flat = flat.scatter(0, self.pivot_indices, self.pivot_values().reshape(-1))
        return flat.reshape(self.kv_dimension, self.hidden_size)

    def physical_serialized_values(self) -> torch.Tensor:
        return torch.cat((
            self.query_weight.reshape(-1),
            self.materialize_key_weight().reshape(-1),
            self.v_proj.weight.reshape(-1),
            self.o_proj.weight.reshape(-1),
        ))

    def chart_parameter_count(self) -> int:
        return (
            self.query_weight.numel()
            + self.key_nonpivot.numel()
            + self.polar.numel()
            + self.v_proj.weight.numel()
            + self.o_proj.weight.numel()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = F.linear(hidden_states, self.query_weight).view(
            hidden_shape
        ).transpose(1, 2)
        key_states = F.linear(
            hidden_states, self.materialize_key_weight()
        ).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if self.arm == "physical_scale_g1":
            even = key_states[..., :self.pairs]
            odd = key_states[..., self.pairs:]
            coefficients = self.curvature_coefficients()[None, :, None, :]
            key_states = torch.cat(
                (even, odd + coefficients * even.square()), dim=-1
            )

        if self.record_diagnostics:
            with torch.no_grad():
                coefficients = self.curvature_coefficients().float()
                float_key = key_states.float()
                self.last_diagnostics = {
                    "coefficient_rms": float(coefficients.square().mean().sqrt()),
                    "coefficient_abs_max": float(coefficients.abs().max()),
                    "key_rms": float(float_key.square().mean().sqrt()),
                    "key_abs_max": float(float_key.abs().max()),
                    "key_nonfinite_fraction": float(
                        (~torch.isfinite(float_key)).float().mean()
                    ),
                }

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        if past_key_values is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )
        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]
        attention_output, attention_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=(0.0 if not self.training else self.attention_dropout),
            scaling=self.scaling,
            **kwargs,
        )
        attention_output = attention_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attention_output), attention_weights


def replace_llama_attention(model: nn.Module, arm: str) -> list[PhysicalScaleG1Attention]:
    modules = []
    for index, layer in enumerate(model.model.layers):
        replacement = PhysicalScaleG1Attention(layer.self_attn, arm).to(
            next(model.parameters()).device
        )
        layer.self_attn = replacement
        modules.append(replacement)
        if replacement.layer_idx != index:
            raise ValueError("attention layer index mismatch")
    return modules
