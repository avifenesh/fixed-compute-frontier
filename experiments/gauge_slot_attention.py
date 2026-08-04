#!/usr/bin/env python3
"""Gauge-canonical packed attention with two storage-funded key shears."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
    ALL_ATTENTION_FUNCTIONS,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


VALID_ARMS = ("canonical_bilinear_control", "gauge_slot_g2")


def decode_packed_key(
    packed_key: torch.Tensor,
    pivot_indices: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Substitute the implicit canonical K values for overloaded slots."""
    flat = packed_key.reshape(-1).clone()
    constants = flat.new_tensor((0.0, tau)).repeat(pivot_indices.numel() // 2)
    return flat.scatter(0, pivot_indices.to(flat.device), constants).view_as(packed_key)


def coefficients_from_packed(
    packed_key: torch.Tensor,
    pivot_indices: torch.Tensor,
    kv_heads: int,
    pairs: int,
) -> torch.Tensor:
    return packed_key.reshape(-1)[pivot_indices.to(packed_key.device)].reshape(
        kv_heads, pairs, 2
    )


class GaugeSlotAttention(nn.Module):
    """Llama attention using two redundant K slots as nonlinear coefficients.

    A scaled rotation sends one selected K column per RoPE pair to `(0, tau)`.
    Those two canonical values are implicit in the logical K projection. Their
    physical packed slots instead store independent coefficients `(alpha,
    beta)` for the simultaneous mutation

        even += alpha * odd**2
        odd  += beta  * even**2.

    The Python module is a quality reference. Equal resident bytes require a
    serving kernel that decodes the overloaded packed K slots in-flight.
    """

    def __init__(self, source: nn.Module, arm: str) -> None:
        super().__init__()
        if arm not in VALID_ARMS:
            raise ValueError(f"invalid arm: {arm}")
        if source.q_proj.bias is not None or source.k_proj.bias is not None:
            raise ValueError("gauge-slot prototype requires bias-free Q/K")
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

        query_weight, key_weight, pivot_columns = self._canonicalize(
            source.q_proj.weight.detach(), source.k_proj.weight.detach()
        )
        pivot_indices = []
        for kv_head in range(self.kv_heads):
            for pair in range(self.pairs):
                column = int(pivot_columns[kv_head, pair])
                even_row = kv_head * self.head_dim + pair
                odd_row = even_row + self.pairs
                pivot_indices.extend((
                    even_row * self.hidden_size + column,
                    odd_row * self.hidden_size + column,
                ))
        pivot_indices_tensor = torch.tensor(
            pivot_indices, dtype=torch.long, device=key_weight.device
        )
        self.query_weight = nn.Parameter(query_weight.to(source.q_proj.weight.dtype))
        packed_key = key_weight.reshape(-1).scatter(
            0,
            pivot_indices_tensor,
            torch.zeros(
                pivot_indices_tensor.numel(),
                dtype=key_weight.dtype,
                device=key_weight.device,
            ),
        )
        # One physical parameter tensor exactly replaces source.k_proj.weight.
        # Its overloaded pivot slots are coefficients, not linear K weights.
        self.key_packed = nn.Parameter(
            packed_key.reshape_as(key_weight).to(source.k_proj.weight.dtype)
        )
        # The atlas is compiler/checkpoint metadata, not learned state. It is
        # deliberately exported explicitly instead of hidden in state_dict.
        self.register_buffer("pivot_columns", pivot_columns, persistent=False)
        self.register_buffer("pivot_indices", pivot_indices_tensor, persistent=False)
        self.v_proj = source.v_proj
        self.o_proj = source.o_proj
        self.record_diagnostics = False
        self.last_diagnostics: dict[str, float] | None = None

    def _canonicalize(
        self,
        query_weight: torch.Tensor,
        key_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        working_dtype = torch.float64
        query = query_weight.to(working_dtype).clone().reshape(
            self.query_heads, self.head_dim, self.hidden_size
        )
        key = key_weight.to(working_dtype).clone().reshape(
            self.kv_heads, self.head_dim, self.hidden_size
        )
        pivot_columns = torch.empty(
            self.kv_heads,
            self.pairs,
            dtype=torch.long,
            device=key.device,
        )
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
                a = key[kv_head, even_row, column]
                b = key[kv_head, odd_row, column]
                radius_squared = a.square() + b.square()
                gauge = (self.tau / radius_squared) * torch.stack((
                    torch.stack((b, -a)),
                    torch.stack((a, b)),
                ))
                key_pair = key[kv_head, (even_row, odd_row)]
                key[kv_head, (even_row, odd_row)] = gauge @ key_pair
                inverse_transpose = torch.linalg.inv(gauge).T
                first_query_head = kv_head * self.num_key_value_groups
                last_query_head = first_query_head + self.num_key_value_groups
                for query_head in range(first_query_head, last_query_head):
                    query_pair = query[query_head, (even_row, odd_row)]
                    query[query_head, (even_row, odd_row)] = (
                        inverse_transpose @ query_pair
                    )
                pivot_columns[kv_head, pair] = column
        return (
            query.reshape(self.query_heads * self.head_dim, self.hidden_size).float(),
            key.reshape(self.kv_dimension, self.hidden_size).float(),
            pivot_columns,
        )

    def materialize_key_weight(self) -> torch.Tensor:
        return decode_packed_key(self.key_packed, self.pivot_indices, self.tau)

    def packed_key_weight(self) -> torch.Tensor:
        return self.key_packed

    def curvature_coefficients(self) -> torch.Tensor:
        return coefficients_from_packed(
            self.key_packed,
            self.pivot_indices,
            self.kv_heads,
            self.pairs,
        )

    def export_packed(self) -> dict[str, torch.Tensor | float | int]:
        """Return a self-contained custom-kernel artifact."""
        return {
            "query_weight": self.query_weight.detach().clone(),
            "packed_key_weight": self.packed_key_weight().detach().clone(),
            "value_weight": self.v_proj.weight.detach().clone(),
            "output_weight": self.o_proj.weight.detach().clone(),
            "pivot_columns": self.pivot_columns.detach().cpu().clone(),
            "pivot_indices": self.pivot_indices.detach().cpu().clone(),
            "tau": self.tau,
            "kv_heads": self.kv_heads,
            "pairs": self.pairs,
        }

    def packed_serialized_values(self) -> torch.Tensor:
        return torch.cat((
            self.query_weight.reshape(-1),
            self.packed_key_weight().reshape(-1),
            self.v_proj.weight.reshape(-1),
            self.o_proj.weight.reshape(-1),
        ))

    def chart_parameter_count(self) -> int:
        return (
            self.query_weight.numel()
            + self.key_packed.numel()
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

        if self.arm == "gauge_slot_g2":
            even = key_states[..., :self.pairs]
            odd = key_states[..., self.pairs:]
            coefficients = self.curvature_coefficients().to(
                key_states.dtype
            )[None, :, None]
            original_even = even
            original_odd = odd
            key_states = torch.cat((
                original_even + coefficients[..., 0] * original_odd.square(),
                original_odd + coefficients[..., 1] * original_even.square(),
            ), dim=-1)

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


def replace_llama_attention(
    model: nn.Module, arm: str
) -> list[GaugeSlotAttention]:
    modules = []
    for index, layer in enumerate(model.model.layers):
        replacement = GaugeSlotAttention(layer.self_attn, arm).to(
            next(model.parameters()).device
        )
        layer.self_attn = replacement
        modules.append(replacement)
        if replacement.layer_idx != index:
            raise ValueError("attention layer index mismatch")
    return modules
