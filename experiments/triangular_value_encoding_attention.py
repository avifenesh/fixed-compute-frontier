#!/usr/bin/env python3
"""Gauge-canonical attention with nonlinear values encoded before KV caching."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
    ALL_ATTENTION_FUNCTIONS,
    eager_attention_forward,
)

from experiments.triangular_value_flow_attention import rq_decomposition


VALID_ARMS = (
    "packed_raw_control",
    "canonical_value_control",
    "triangular_value_encoding",
)
SERVING_TOKEN_BLOCK = 64


@triton.jit
def value_encoding_forward_kernel(
    values_ptr,
    output_ptr,
    output_weight_ptr,
    TOKENS: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    KV_HEADS_VALUE: tl.constexpr,
    QUERY_GROUPS: tl.constexpr,
    HEAD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TAU_VALUE: tl.constexpr,
    TOKEN_BLOCK_SIZE: tl.constexpr,
):
    task = tl.program_id(0)
    blocks = HEAD // BLOCK_SIZE
    tasks_per_tile = KV_HEADS_VALUE * blocks
    token_tile = task // tasks_per_tile
    in_tile = task % tasks_per_tile
    kv_head = in_tile // blocks
    block_index = in_tile % blocks
    token = token_tile * TOKEN_BLOCK_SIZE + tl.arange(0, TOKEN_BLOCK_SIZE)
    token_mask = token < TOKENS
    target = tl.arange(0, BLOCK_SIZE)
    source = tl.arange(0, BLOCK_SIZE)
    block_start = block_index * BLOCK_SIZE
    value_offset = (
        token[:, None] * KV_HEADS_VALUE * HEAD
        + kv_head * HEAD
        + block_start
        + source[None, :]
    )
    value = tl.load(values_ptr + value_offset, mask=token_mask[:, None]).to(
        tl.float32
    )
    representative_head = kv_head * QUERY_GROUPS
    coefficient = (tl.load(
        output_weight_ptr
        + (block_start + target[:, None]) * OUTPUT_WIDTH
        + representative_head * HEAD
        + block_start
        + source[None, :],
        mask=source[None, :] < target[:, None],
        other=0.0,
    ).to(tl.float32) / TAU_VALUE).to(tl.bfloat16)
    feature = (value * tl.abs(value)).to(tl.bfloat16)
    delta = tl.dot(feature, tl.trans(coefficient), out_dtype=tl.float32).to(
        tl.bfloat16
    )
    encoded = value + delta.to(tl.float32)
    tl.store(output_ptr + value_offset, encoded, mask=token_mask[:, None])


@torch.no_grad()
def serving_value_encoding(
    token_major_values: torch.Tensor,
    output_weight: torch.Tensor,
    *,
    query_groups: int,
    block_size: int,
    tau: float,
) -> torch.Tensor:
    """Exact serving forward for contiguous [token, KV head, coordinate] V."""
    if (
        not token_major_values.is_cuda
        or token_major_values.dtype != torch.bfloat16
        or not token_major_values.is_contiguous()
    ):
        raise ValueError("serving value encoding requires contiguous CUDA BF16 values")
    tokens, kv_heads, head_dim = token_major_values.shape
    if head_dim % block_size:
        raise ValueError("head dimension must divide the serving block")
    output = torch.empty_like(token_major_values)
    tasks = (
        triton.cdiv(tokens, SERVING_TOKEN_BLOCK)
        * kv_heads
        * (head_dim // block_size)
    )
    value_encoding_forward_kernel[(tasks,)](
        token_major_values,
        output,
        output_weight,
        TOKENS=tokens,
        OUTPUT_WIDTH=output_weight.shape[0],
        KV_HEADS_VALUE=kv_heads,
        QUERY_GROUPS=query_groups,
        HEAD=head_dim,
        BLOCK_SIZE=block_size,
        TAU_VALUE=tau,
        TOKEN_BLOCK_SIZE=SERVING_TOKEN_BLOCK,
        num_warps=4,
    )
    return output


def apply_rotary_fp32_one_store(
    query: torch.Tensor,
    key: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Production-style RoPE: FP32 arithmetic and one output-dtype store."""
    cosine = cosine.unsqueeze(1).float()
    sine = sine.unsqueeze(1).float()
    half = query.shape[-1] // 2

    def rotate(tensor: torch.Tensor) -> torch.Tensor:
        source = tensor.float()
        rotated = torch.cat((-source[..., half:], source[..., :half]), dim=-1)
        return (source * cosine + rotated * sine).to(tensor.dtype)

    return rotate(query), rotate(key)


class TriangularValueEncodingAttention(nn.Module):
    """Reuse gauge-zeroed O weights as a pre-cache nonlinear V encoding."""

    def __init__(self, source: nn.Module, arm: str, block_size: int = 16) -> None:
        super().__init__()
        if arm not in VALID_ARMS:
            raise ValueError(f"invalid arm: {arm}")
        if source.v_proj.bias is not None or source.o_proj.bias is not None:
            raise ValueError("triangular value encoding requires bias-free V/O")
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
        if arm == "packed_raw_control":
            value_weight = source.v_proj.weight.detach().clone()
            output_weight = source.o_proj.weight.detach().clone()
        else:
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

    def _torch_value_encoding(self, values: torch.Tensor) -> torch.Tensor:
        pieces = []
        for start, physical in zip(
            range(0, self.head_dim, self.block_size),
            self.representative_output_blocks(),
        ):
            block = values[..., start:start + self.block_size]
            coefficient = (
                torch.tril(physical.to(block.dtype), diagonal=-1).float()
                / self.tau
            ).to(block.dtype)
            feature = (block.float() * block.float().abs()).to(block.dtype)
            delta = torch.matmul(
                coefficient[None, :, None], feature.unsqueeze(-1)
            ).squeeze(-1)
            pieces.append((block.float() + delta.float()).to(block.dtype))
        return torch.cat(pieces, dim=-1)

    def apply_value_encoding(self, values: torch.Tensor) -> torch.Tensor:
        if self.arm != "triangular_value_encoding":
            return values
        differentiable = self._torch_value_encoding(values)
        if values.is_cuda and values.dtype == torch.bfloat16:
            batch, _, tokens, _ = values.shape
            token_major = values.transpose(1, 2).contiguous().view(
                batch * tokens, self.kv_heads, self.head_dim
            )
            serving = serving_value_encoding(
                token_major,
                self.output_weight.to(values.dtype),
                query_groups=self.num_key_value_groups,
                block_size=self.block_size,
                tau=self.tau,
            ).view(batch, tokens, self.kv_heads, self.head_dim).transpose(1, 2)
            # Exact serving forward; algebraically equivalent PyTorch backward.
            return serving.detach() + (
                differentiable - differentiable.detach()
            )
        return differentiable

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

    def packed_qkv_weight(self) -> torch.Tensor:
        return torch.cat((
            self.q_proj.weight,
            self.k_proj.weight,
            self.value_weight,
        ), dim=0)

    def project_qkv_packed(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Use the same fused-QKV numerical contract as serving export."""
        return F.linear(hidden_states, self.packed_qkv_weight())

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
        projected = self.project_qkv_packed(hidden_states)
        query_flat, key_flat, value_flat = projected.split((
            self.query_heads * self.head_dim,
            self.kv_dimension,
            self.kv_dimension,
        ), dim=-1)
        query_states = query_flat.view(hidden_shape).transpose(1, 2)
        key_states = key_flat.view(hidden_shape).transpose(1, 2)
        value_states = value_flat.view(hidden_shape).transpose(1, 2)
        unencoded_for_diagnostics = (
            value_states.detach() if self.record_diagnostics else None
        )
        value_states = self.apply_value_encoding(value_states)
        if self.record_diagnostics:
            with torch.no_grad():
                if unencoded_for_diagnostics is None:
                    raise RuntimeError("missing packed-QKV value diagnostic")
                coefficient = self.coefficient_values().float()
                delta = value_states.float() - unencoded_for_diagnostics.float()
                self.last_diagnostics = {
                    "coefficient_rms": float(coefficient.square().mean().sqrt()),
                    "coefficient_abs_max": float(coefficient.abs().max()),
                    "value_delta_rms": float(delta.square().mean().sqrt()),
                    "value_delta_abs_max": float(delta.abs().max()),
                    "value_nonfinite_fraction": float(
                        (~torch.isfinite(value_states.float())).float().mean()
                    ),
                }
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_fp32_one_store(
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
        attention_output = attention_output.reshape(*input_shape, -1).contiguous()
        return F.linear(attention_output, self.output_weight), attention_weights


def replace_llama_attention(
    model: nn.Module,
    arm: str,
    block_size: int = 16,
) -> list[TriangularValueEncodingAttention]:
    modules = []
    for index, layer in enumerate(model.model.layers):
        replacement = TriangularValueEncodingAttention(
            layer.self_attn, arm, block_size
        ).to(next(model.parameters()).device)
        layer.self_attn = replacement
        modules.append(replacement)
        if replacement.layer_idx != index:
            raise ValueError("attention layer index mismatch")
    return modules
