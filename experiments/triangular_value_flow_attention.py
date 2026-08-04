#!/usr/bin/env python3
"""Gauge-canonical attention with block-triangular nonlinear value flow."""

from __future__ import annotations

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


VALID_ARMS = ("canonical_value_control", "triangular_value_flow")


def rq_decomposition(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sign-canonicalized A = R Q with R upper and Q orthogonal."""
    reversed_transpose = matrix.T.flip((0, 1))
    q_reversed, r_reversed = torch.linalg.qr(reversed_transpose)
    upper = r_reversed.T.flip((0, 1))
    orthogonal = q_reversed.T.flip((0, 1))
    signs = torch.where(
        torch.diagonal(upper) < 0,
        -torch.ones((), dtype=upper.dtype, device=upper.device),
        torch.ones((), dtype=upper.dtype, device=upper.device),
    )
    upper = upper * signs[None]
    orthogonal = signs[:, None] * orthogonal
    return upper, orthogonal


class TriangularValueFlowAttention(nn.Module):
    """Reuse gauge-zeroed O weights as a nonlinear triangular value flow."""

    def __init__(
        self,
        source: nn.Module,
        arm: str,
        block_size: int = 16,
    ) -> None:
        super().__init__()
        if arm not in VALID_ARMS:
            raise ValueError(f"invalid arm: {arm}")
        if source.v_proj.bias is not None or source.o_proj.bias is not None:
            raise ValueError("triangular value flow requires bias-free V/O")
        self.arm = arm
        self.config = source.config
        self.layer_idx = source.layer_idx
        self.head_dim = source.head_dim
        if self.head_dim % block_size:
            raise ValueError("head dimension must be divisible by block size")
        self.block_size = block_size
        self.blocks = self.head_dim // block_size
        self.query_heads = source.config.num_attention_heads
        self.kv_heads = source.config.num_key_value_heads
        self.num_key_value_groups = self.query_heads // self.kv_heads
        self.hidden_size = source.q_proj.in_features
        self.kv_dimension = self.kv_heads * self.head_dim
        self.scaling = source.scaling
        self.attention_dropout = source.attention_dropout
        self.is_causal = True
        self.tau = 0.125
        self.q_proj = source.q_proj
        self.k_proj = source.k_proj
        value_weight, output_weight = self._canonicalize(
            source.v_proj.weight.detach(), source.o_proj.weight.detach()
        )
        self.value_weight = nn.Parameter(value_weight.to(source.v_proj.weight.dtype))
        self.output_weight = nn.Parameter(output_weight.to(source.o_proj.weight.dtype))
        self.record_diagnostics = False
        self.last_diagnostics: dict[str, float] | None = None

    def _canonicalize(
        self,
        value_weight: torch.Tensor,
        output_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value = value_weight.to(torch.float64).clone().reshape(
            self.kv_heads, self.head_dim, self.hidden_size
        )
        output = output_weight.to(torch.float64).clone().reshape(
            self.hidden_size, self.query_heads, self.head_dim
        )
        for kv_head in range(self.kv_heads):
            first_query_head = kv_head * self.num_key_value_groups
            representative = output[
                :self.head_dim, first_query_head, :
            ].clone()
            _, orthogonal = rq_decomposition(representative)
            value[kv_head] = orthogonal @ value[kv_head]
            for query_head in range(
                first_query_head,
                first_query_head + self.num_key_value_groups,
            ):
                output[:, query_head] = output[:, query_head] @ orthogonal.T
            canonical = output[:self.head_dim, first_query_head]
            for row in range(self.head_dim):
                canonical[row, :row] = 0.0
        return (
            value.reshape(self.kv_dimension, self.hidden_size).float(),
            output.reshape(self.hidden_size, self.query_heads * self.head_dim).float(),
        )

    def representative_output_blocks(self) -> list[torch.Tensor]:
        output = self.output_weight.reshape(
            self.hidden_size, self.query_heads, self.head_dim
        )
        representative = output[
            :self.head_dim, ::self.num_key_value_groups, :
        ].permute(1, 0, 2)
        return [
            representative[:, start:start + self.block_size, start:start + self.block_size]
            for start in range(0, self.head_dim, self.block_size)
        ]

    def coefficient_values(self) -> torch.Tensor:
        values = [
            block[:, row, :row]
            for block in self.representative_output_blocks()
            for row in range(1, self.block_size)
        ]
        return torch.cat(values, dim=-1) / self.tau

    def apply_value_flow(self, attention_output: torch.Tensor) -> torch.Tensor:
        if self.arm != "triangular_value_flow":
            return attention_output
        grouped = attention_output.unflatten(
            -2, (self.kv_heads, self.num_key_value_groups)
        )
        pieces = []
        for start, physical in zip(
            range(0, self.head_dim, self.block_size),
            self.representative_output_blocks(),
        ):
            block = grouped[..., start:start + self.block_size]
            coefficient = torch.tril(
                physical.to(block.dtype), diagonal=-1
            ).float() / self.tau
            feature = block.float() * block.float().abs()
            delta = torch.matmul(
                coefficient[None, None, :, None], feature.unsqueeze(-1)
            ).squeeze(-1)
            pieces.append((block.float() + delta).to(block.dtype))
        return torch.cat(pieces, dim=-1).flatten(-3, -2)

    def export_dense(self) -> dict[str, torch.Tensor | float | int]:
        return {
            "query_weight": self.q_proj.weight.detach().clone(),
            "key_weight": self.k_proj.weight.detach().clone(),
            "value_weight": self.value_weight.detach().clone(),
            "output_weight": self.output_weight.detach().clone(),
            "block_size": self.block_size,
            "tau": self.tau,
            "kv_heads": self.kv_heads,
            "query_heads": self.query_heads,
            "head_dim": self.head_dim,
        }

    def dense_serialized_values(self) -> torch.Tensor:
        return torch.cat((
            self.q_proj.weight.reshape(-1),
            self.k_proj.weight.reshape(-1),
            self.value_weight.reshape(-1),
            self.output_weight.reshape(-1),
        ))

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
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = F.linear(hidden_states, self.value_weight).view(
            hidden_shape
        ).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                {"sin": sin, "cos": cos, "cache_position": cache_position},
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
        flowed = self.apply_value_flow(attention_output)
        if self.record_diagnostics:
            with torch.no_grad():
                coefficient = self.coefficient_values().float()
                delta = flowed.float() - attention_output.float()
                self.last_diagnostics = {
                    "coefficient_rms": float(coefficient.square().mean().sqrt()),
                    "coefficient_abs_max": float(coefficient.abs().max()),
                    "flow_delta_rms": float(delta.square().mean().sqrt()),
                    "flow_delta_abs_max": float(delta.abs().max()),
                    "flow_nonfinite_fraction": float(
                        (~torch.isfinite(flowed.float())).float().mean()
                    ),
                }
        attention_output = flowed.reshape(*input_shape, -1).contiguous()
        return F.linear(attention_output, self.output_weight), attention_weights


def replace_llama_attention(
    model: nn.Module,
    arm: str,
    block_size: int = 16,
) -> list[TriangularValueFlowAttention]:
    modules = []
    for index, layer in enumerate(model.model.layers):
        replacement = TriangularValueFlowAttention(
            layer.self_attn, arm, block_size
        ).to(next(model.parameters()).device)
        layer.self_attn = replacement
        modules.append(replacement)
        if replacement.layer_idx != index:
            raise ValueError("attention layer index mismatch")
    return modules
