#!/usr/bin/env python3
"""TVE custom autograd with Triton dV and reference-exact torch dO."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from experiments.triangular_value_encoding_autograd import (
    TOKEN_BLOCK,
    _forward_kernel,
)


@triton.jit
def _backward_values_kernel(
    values_ptr,
    gradient_ptr,
    grad_values_ptr,
    output_weight_ptr,
    TOKENS: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    KV_HEADS: tl.constexpr,
    QUERY_GROUPS: tl.constexpr,
    HEAD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TAU: tl.constexpr,
    TOKEN_BLOCK_SIZE: tl.constexpr,
):
    task = tl.program_id(0)
    blocks = HEAD // BLOCK_SIZE
    tasks_per_tile = KV_HEADS * blocks
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
        token[:, None] * KV_HEADS * HEAD
        + kv_head * HEAD
        + block_start
        + source[None, :]
    )
    value = tl.load(
        values_ptr + value_offset, mask=token_mask[:, None], other=0.0
    ).to(tl.float32)
    gradient = tl.load(
        gradient_ptr + value_offset, mask=token_mask[:, None], other=0.0
    )
    representative_head = kv_head * QUERY_GROUPS
    physical = tl.load(
        output_weight_ptr
        + (block_start + target[:, None]) * OUTPUT_WIDTH
        + representative_head * HEAD
        + block_start
        + source[None, :],
        mask=source[None, :] < target[:, None],
        other=0.0,
    ).to(tl.bfloat16).to(tl.float32)
    coefficient = (physical / TAU).to(tl.bfloat16)
    nonlinear = tl.dot(gradient, coefficient, out_dtype=tl.float32).to(tl.bfloat16)
    grad_value = gradient.to(tl.float32) + 2.0 * tl.abs(value) * nonlinear.to(tl.float32)
    tl.store(grad_values_ptr + value_offset, grad_value, mask=token_mask[:, None])


def _triton_value_gradient(
    token_major_values: torch.Tensor,
    token_major_gradient: torch.Tensor,
    output_weight: torch.Tensor,
    *,
    query_groups: int,
    block_size: int,
    tau: float,
) -> torch.Tensor:
    tokens, kv_heads, head_dim = token_major_values.shape
    token_tiles = triton.cdiv(tokens, TOKEN_BLOCK)
    tasks_per_tile = kv_heads * (head_dim // block_size)
    grad_values = torch.empty_like(token_major_values)
    _backward_values_kernel[(token_tiles * tasks_per_tile,)](
        token_major_values,
        token_major_gradient,
        grad_values,
        output_weight,
        TOKENS=tokens,
        OUTPUT_WIDTH=output_weight.shape[0],
        KV_HEADS=kv_heads,
        QUERY_GROUPS=query_groups,
        HEAD=head_dim,
        BLOCK_SIZE=block_size,
        TAU=tau,
        TOKEN_BLOCK_SIZE=TOKEN_BLOCK,
        num_warps=4,
    )
    return grad_values


def _reference_exact_weight_gradient(
    token_major_values: torch.Tensor,
    token_major_gradient: torch.Tensor,
    output_weight: torch.Tensor,
    *,
    query_groups: int,
    block_size: int,
    tau: float,
) -> torch.Tensor:
    _, kv_heads, head_dim = token_major_values.shape
    grad_weight = torch.zeros_like(output_weight)
    for start in range(0, head_dim, block_size):
        value = token_major_values[..., start:start + block_size]
        gradient = token_major_gradient[..., start:start + block_size]
        feature = (value.float() * value.float().abs()).to(value.dtype)
        grad_coefficient = torch.matmul(
            gradient.permute(1, 2, 0),
            feature.permute(1, 0, 2),
        )
        # Match the reference cast chain for every accepted tau, not only the
        # frozen power-of-two value used by the formal gate.
        grad_physical = (
            torch.tril(grad_coefficient, diagonal=-1).float() / tau
        ).to(value.dtype).float()
        for kv_head in range(kv_heads):
            row = slice(start, start + block_size)
            column_start = kv_head * query_groups * head_dim + start
            grad_weight[
                row, column_start:column_start + block_size
            ] = grad_physical[kv_head]
    return grad_weight


class _TriangularValueEncodingHybrid(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        values: torch.Tensor,
        output_weight: torch.Tensor,
        query_groups: int,
        block_size: int,
        tau: float,
    ) -> torch.Tensor:
        if not values.is_cuda or values.dtype != torch.bfloat16:
            raise ValueError("hybrid TVE requires CUDA BF16 values")
        if values.ndim != 4:
            raise ValueError("values must have shape [batch, KV head, token, coordinate]")
        if (
            not output_weight.is_cuda
            or output_weight.device != values.device
            or output_weight.dtype != torch.float32
            or not output_weight.is_contiguous()
        ):
            raise ValueError(
                "hybrid TVE requires contiguous same-device CUDA FP32 output weights"
            )
        if query_groups <= 0 or block_size < 16 or block_size & (block_size - 1):
            raise ValueError(
                "query groups must be positive and block size a power of two >= 16"
            )
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError("tau must be finite and positive")
        batch, kv_heads, tokens, head_dim = values.shape
        if min(batch, kv_heads, tokens, head_dim) <= 0:
            raise ValueError("all value dimensions must be positive")
        output_width = query_groups * kv_heads * head_dim
        if output_weight.shape != (output_width, output_width):
            raise ValueError("output weight shape does not match GQA geometry")
        if head_dim % block_size:
            raise ValueError("block size must divide head dimension")
        token_major = values.transpose(1, 2).contiguous().view(
            batch * tokens, kv_heads, head_dim
        )
        output = torch.empty_like(token_major)
        tasks = (
            triton.cdiv(batch * tokens, TOKEN_BLOCK)
            * kv_heads * (head_dim // block_size)
        )
        _forward_kernel[(tasks,)](
            token_major,
            output,
            output_weight,
            TOKENS=batch * tokens,
            OUTPUT_WIDTH=output_weight.shape[0],
            KV_HEADS=kv_heads,
            QUERY_GROUPS=query_groups,
            HEAD=head_dim,
            BLOCK_SIZE=block_size,
            TAU=tau,
            TOKEN_BLOCK_SIZE=TOKEN_BLOCK,
            num_warps=4,
        )
        ctx.save_for_backward(token_major, output_weight)
        ctx.input_shape = values.shape
        ctx.query_groups = query_groups
        ctx.block_size = block_size
        ctx.tau = tau
        return output.view(batch, tokens, kv_heads, head_dim).transpose(1, 2)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        token_major_values, output_weight = ctx.saved_tensors
        batch, kv_heads, tokens, head_dim = ctx.input_shape
        token_major_gradient = gradient.transpose(1, 2).contiguous().view(
            batch * tokens, kv_heads, head_dim
        )
        grad_values = _triton_value_gradient(
            token_major_values,
            token_major_gradient,
            output_weight,
            query_groups=ctx.query_groups,
            block_size=ctx.block_size,
            tau=ctx.tau,
        )
        grad_weight = _reference_exact_weight_gradient(
            token_major_values,
            token_major_gradient,
            output_weight,
            query_groups=ctx.query_groups,
            block_size=ctx.block_size,
            tau=ctx.tau,
        )
        values_layout = grad_values.view(
            batch, tokens, kv_heads, head_dim
        ).transpose(1, 2)
        return values_layout, grad_weight, None, None, None


def triangular_value_encoding_autograd_hybrid(
    values: torch.Tensor,
    output_weight: torch.Tensor,
    *,
    query_groups: int,
    block_size: int = 16,
    tau: float = 0.125,
) -> torch.Tensor:
    return _TriangularValueEncodingHybrid.apply(
        values, output_weight, query_groups, block_size, tau
    )
