#!/usr/bin/env python3
"""Ordinary dense attention with one zero-gauge-funded quadratic K feature."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
    ALL_ATTENTION_FUNCTIONS,
    eager_attention_forward,
)


VALID_ARMS = ("canonical_bilinear_control", "gauge_zero_g1")


def fused_split_half_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    key_coefficient: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply optional G1 shear and RoPE in FP32, then round once.

    This is the numerical contract used by the fused serving epilogue: dense
    Q/K projections are stored in their serving dtype, all pointwise work is
    FP32, and the rotated Q/K values are cast back only at the final store.
    """
    pairs = query.shape[-1] // 2
    query_dtype = query.dtype
    key_dtype = key.dtype
    query_even = query[..., :pairs].float()
    query_odd = query[..., pairs:].float()
    key_even = key[..., :pairs].float()
    key_odd = key[..., pairs:].float()
    if key_coefficient is not None:
        key_odd = key_odd + key_coefficient.float()[None, :, None] * key_even.square()
    cos = cosine[..., :pairs].unsqueeze(1).float()
    sin = sine[..., :pairs].unsqueeze(1).float()
    return (
        torch.cat((
            cos * query_even - sin * query_odd,
            sin * query_even + cos * query_odd,
        ), dim=-1).to(query_dtype),
        torch.cat((
            cos * key_even - sin * key_odd,
            sin * key_even + cos * key_odd,
        ), dim=-1).to(key_dtype),
    )


class GaugeZeroG1Attention(nn.Module):
    """Use an existing zero-canonicalized K weight as a G1 coefficient.

    A deterministic rotation maps K column `pair_index` to `(radius, 0)` for
    each RoPE pair. The odd pivot remains an ordinary dense K weight, but the
    candidate also reads `odd_pivot / tau` as the coefficient in

        odd += coefficient * even**2.

    Thus the model keeps a conventional dense Q/K checkpoint and GEMM. The
    coefficient is coupled to a linear K weight, starts exactly at zero, and
    adds no learned value or optimizer tensor.
    """

    def __init__(self, source: nn.Module, arm: str) -> None:
        super().__init__()
        if arm not in VALID_ARMS:
            raise ValueError(f"invalid arm: {arm}")
        if source.q_proj.bias is not None or source.k_proj.bias is not None:
            raise ValueError("gauge-zero G1 requires bias-free Q/K")
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
        # A power-of-two anchor makes coefficient decoding an exact exponent
        # shift in binary floating point. It also matches 1/sqrt(64).
        self.tau = 0.125

        query_weight, key_weight = self._canonicalize(
            source.q_proj.weight.detach(), source.k_proj.weight.detach()
        )
        self.query_weight = nn.Parameter(query_weight.to(source.q_proj.weight.dtype))
        self.key_weight = nn.Parameter(key_weight.to(source.k_proj.weight.dtype))
        self.v_proj = source.v_proj
        self.o_proj = source.o_proj
        self.record_diagnostics = False
        self.last_diagnostics: dict[str, float] | None = None

    def _canonicalize(
        self,
        query_weight: torch.Tensor,
        key_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = query_weight.to(torch.float64).clone().reshape(
            self.query_heads, self.head_dim, self.hidden_size
        )
        key = key_weight.to(torch.float64).clone().reshape(
            self.kv_heads, self.head_dim, self.hidden_size
        )
        for kv_head in range(self.kv_heads):
            for pair in range(self.pairs):
                even_row = pair
                odd_row = pair + self.pairs
                column = pair
                a = key[kv_head, even_row, column]
                b = key[kv_head, odd_row, column]
                radius_squared = a.square() + b.square()
                radius = torch.sqrt(radius_squared)
                if not bool(torch.isfinite(radius)):
                    raise ValueError("K pivot is non-finite")
                if float(radius) <= 1e-12:
                    # The zero pivot already has a zero coefficient; identity
                    # preserves the ordinary score function exactly.
                    continue
                gauge = (1.0 / radius) * torch.stack((
                    torch.stack((a, b)),
                    torch.stack((-b, a)),
                ))
                key[kv_head, (even_row, odd_row)] = (
                    gauge @ key[kv_head, (even_row, odd_row)]
                )
                key[kv_head, even_row, column] = radius
                key[kv_head, odd_row, column] = 0.0
                inverse_transpose = torch.linalg.inv(gauge).T
                first_query_head = kv_head * self.num_key_value_groups
                last_query_head = first_query_head + self.num_key_value_groups
                for query_head in range(first_query_head, last_query_head):
                    query[query_head, (even_row, odd_row)] = (
                        inverse_transpose
                        @ query[query_head, (even_row, odd_row)]
                    )
        return (
            query.reshape(self.query_heads * self.head_dim, self.hidden_size).float(),
            key.reshape(self.kv_dimension, self.hidden_size).float(),
        )

    def coefficient_physical_weights(self) -> torch.Tensor:
        """View coefficient-bearing scalars directly in the dense K tensor."""
        key = self.key_weight.reshape(
            self.kv_heads, self.head_dim, self.hidden_size
        )
        return key[:, self.pairs:, :self.pairs].diagonal(dim1=-2, dim2=-1)

    def curvature_coefficients(self) -> torch.Tensor:
        return self.coefficient_physical_weights() / self.tau

    def export_dense(self) -> dict[str, torch.Tensor | float | int]:
        return {
            "query_weight": self.query_weight.detach().clone(),
            "key_weight": self.key_weight.detach().clone(),
            "value_weight": self.v_proj.weight.detach().clone(),
            "output_weight": self.o_proj.weight.detach().clone(),
            "pivot_rule": "column equals RoPE pair index",
            "tau": self.tau,
            "kv_heads": self.kv_heads,
            "pairs": self.pairs,
        }

    def dense_serialized_values(self) -> torch.Tensor:
        return torch.cat((
            self.query_weight.reshape(-1),
            self.key_weight.reshape(-1),
            self.v_proj.weight.reshape(-1),
            self.o_proj.weight.reshape(-1),
        ))

    def project_qkv_rope(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = F.linear(hidden_states, self.query_weight).view(
            hidden_shape
        ).transpose(1, 2)
        key_states = F.linear(hidden_states, self.key_weight).view(
            hidden_shape
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        coefficient = (
            self.curvature_coefficients().to(key_states.dtype)
            if self.arm == "gauge_zero_g1"
            else None
        )
        cos, sin = position_embeddings
        query_states, key_states = fused_split_half_rope(
            query_states, key_states, cos, sin, coefficient
        )
        return query_states, key_states, value_states

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
        query_states, key_states, value_states = self.project_qkv_rope(
            hidden_states, position_embeddings
        )

        if self.record_diagnostics:
            with torch.no_grad():
                coefficient = self.curvature_coefficients().float()
                float_key = key_states.float()
                self.last_diagnostics = {
                    "coefficient_rms": float(coefficient.square().mean().sqrt()),
                    "coefficient_abs_max": float(coefficient.abs().max()),
                    "key_rms": float(float_key.square().mean().sqrt()),
                    "key_abs_max": float(float_key.abs().max()),
                    "key_nonfinite_fraction": float(
                        (~torch.isfinite(float_key)).float().mean()
                    ),
                }

        cos, sin = position_embeddings
        if past_key_values is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
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


def replace_llama_attention(
    model: nn.Module, arm: str
) -> list[GaugeZeroG1Attention]:
    modules = []
    for index, layer in enumerate(model.model.layers):
        replacement = GaugeZeroG1Attention(layer.self_attn, arm).to(
            next(model.parameters()).device
        )
        layer.self_attn = replacement
        modules.append(replacement)
        if replacement.layer_idx != index:
            raise ValueError("attention layer index mismatch")
    return modules
