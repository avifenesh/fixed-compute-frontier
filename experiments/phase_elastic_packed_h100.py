#!/usr/bin/env python3
"""Physical H100 kernel gate for exact-packed phase-elastic FFNs."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import subprocess
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice

from experiments.phase_elastic_packed_artifact import unpack_int4, unpack_ternary
from experiments.phase_elastic_residual_precision_v2_lm_screen import serving_ledger
from experiments.reflex_swiglu_lm_screen import sha256_file, write_payload


D = 384
BASE_WIDTH = 1024
BRANCH_WIDTH = 1472
TOTAL_WIDTH = BASE_WIDTH + BRANCH_WIDTH
K_D = tl.constexpr(384)
K_BASE_WIDTH = tl.constexpr(1024)
K_BRANCH_WIDTH = tl.constexpr(1472)
K_TOTAL_WIDTH = tl.constexpr(2496)
ROWS = (1, 8, 32, 128, 512, 2048)
DECODE_ROWS = (1, 8, 32)
PREFILL_ROWS = (128, 512, 2048)
LAYERS = 12
SEED = 42173
ARTIFACT = Path("artifacts/phase-elastic-v3-50m-packed.pt")
QUALITY_RESULT = Path("results/phase-elastic-v3-50m-phase-faithful.json")
QUALITY_RESULT_SHA256 = "065fcf71cdda587f56c1005e623e305cfdfac2246452e148f9d8d4541c16085f"
PRIOR_DENSE_GRAPH_MS = {
    1: 0.013933,
    8: 0.019141,
    32: 0.019596,
    128: 0.022028,
    512: 0.024747,
    2048: 0.039900,
}
PREREGISTRATION = Path("results/phase-elastic-packed-h100-preregistration.md")
INTEGRITY = Path("results/phase-elastic-packed-h100-integrity.json")
HELPER_FILES = (
    Path("experiments/phase_elastic_packed_artifact.py"),
    Path("experiments/phase_elastic_residual_precision_v2_lm_screen.py"),
    Path("experiments/reflex_swiglu_lm_screen.py"),
)

PACKED_COMPONENT_KEYS = {
    "bg_q8": "base_gate.base_codes", "bg_t": "base_gate.ternary_codes",
    "bg_s8": "base_gate.base_scale", "bg_st": "base_gate.delta_scale",
    "bu_q8": "base_up.base_codes", "bu_t": "base_up.ternary_codes",
    "bu_s8": "base_up.base_scale", "bu_st": "base_up.delta_scale",
    "bd_q8": "base_down.base_codes", "bd_t": "base_down.ternary_codes",
    "bd_s8": "base_down.base_scale", "bd_st": "base_down.delta_scale",
    "rg_q4": "branch_gate.codes", "rg_s": "branch_gate.scale",
    "ru_q4": "branch_up.codes", "ru_s": "branch_up.scale",
    "rd_q4": "branch_down.codes", "rd_s": "branch_down.scale",
}


UP_CONFIGS = [
    triton.Config({"BLOCK_R": 16, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_R": 16, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_R": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_R": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_R": 16, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_R": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=UP_CONFIGS, key=["rows", "INCLUDE_BRANCH"])
@triton.jit
def packed_upgate_kernel(
    x_ptr,
    bg_q8_ptr, bg_t_ptr, bg_s8_ptr, bg_st_ptr,
    bu_q8_ptr, bu_t_ptr, bu_s8_ptr, bu_st_ptr,
    rg_q4_ptr, rg_s_ptr, ru_q4_ptr, ru_s_ptr,
    out_ptr,
    rows,
    INCLUDE_BRANCH: tl.constexpr,
    BLOCK_R: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    program = tl.program_id(0)
    total_width: tl.constexpr = K_TOTAL_WIDTH if INCLUDE_BRANCH else K_BASE_WIDTH
    tiles_n = tl.cdiv(total_width, BLOCK_N)
    tile_r = program // tiles_n
    tile_n = program - tile_r * tiles_n
    offsets_r = tile_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offsets_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    base_rows = offsets_n < K_BASE_WIDTH
    branch_n = offsets_n - K_BASE_WIDTH
    gate = tl.zeros((BLOCK_R, BLOCK_N), tl.float32)
    up = tl.zeros((BLOCK_R, BLOCK_N), tl.float32)
    base_s8_gate = tl.load(bg_s8_ptr + offsets_n, mask=base_rows, other=0.0).to(tl.float32)
    base_st_gate = tl.load(bg_st_ptr + offsets_n, mask=base_rows, other=0.0).to(tl.float32)
    base_s8_up = tl.load(bu_s8_ptr + offsets_n, mask=base_rows, other=0.0).to(tl.float32)
    base_st_up = tl.load(bu_st_ptr + offsets_n, mask=base_rows, other=0.0).to(tl.float32)
    if INCLUDE_BRANCH:
        branch_rows = (offsets_n >= K_BASE_WIDTH) & (offsets_n < total_width)
        branch_s_gate = tl.load(rg_s_ptr + branch_n, mask=branch_rows, other=0.0).to(tl.float32)
        branch_s_up = tl.load(ru_s_ptr + branch_n, mask=branch_rows, other=0.0).to(tl.float32)
    for start_k in range(0, K_D, BLOCK_K):
        offsets_k = start_k + tl.arange(0, BLOCK_K)
        x = tl.load(
            x_ptr + offsets_r[:, None] * K_D + offsets_k[None, :],
            mask=(offsets_r[:, None] < rows) & (offsets_k[None, :] < K_D),
            other=0.0,
        )
        base_offsets = offsets_n[:, None] * K_D + offsets_k[None, :]
        q8_gate = tl.load(
            bg_q8_ptr + base_offsets,
            mask=base_rows[:, None] & (offsets_k[None, :] < K_D),
            other=0,
        ).to(tl.float32)
        q8_up = tl.load(
            bu_q8_ptr + base_offsets,
            mask=base_rows[:, None] & (offsets_k[None, :] < K_D),
            other=0,
        ).to(tl.float32)
        packed_k4 = tl.arange(0, BLOCK_K // 4)
        ternary_offsets = (
            offsets_n[:, None] * (K_D // 4)
            + start_k // 4
            + packed_k4[None, :]
        )
        packed_ternary_gate = tl.load(
            bg_t_ptr + ternary_offsets, mask=base_rows[:, None], other=0
        )
        packed_ternary_up = tl.load(
            bu_t_ptr + ternary_offsets, mask=base_rows[:, None], other=0
        )
        gather_k4 = tl.broadcast_to(
            (tl.arange(0, BLOCK_K) // 4)[None, :], (BLOCK_N, BLOCK_K)
        )
        shifts = (offsets_k[None, :] & 3) * 2
        ternary_gate = (
            (tl.gather(packed_ternary_gate, gather_k4, axis=1) >> shifts) & 3
        ).to(tl.int32) - 1
        ternary_up = (
            (tl.gather(packed_ternary_up, gather_k4, axis=1) >> shifts) & 3
        ).to(tl.int32) - 1
        weight_gate = q8_gate * base_s8_gate[:, None] + ternary_gate.to(tl.float32) * base_st_gate[:, None]
        weight_up = q8_up * base_s8_up[:, None] + ternary_up.to(tl.float32) * base_st_up[:, None]
        if INCLUDE_BRANCH:
            packed_k2 = tl.arange(0, BLOCK_K // 2)
            q4_offsets = (
                branch_n[:, None] * (K_D // 2)
                + start_k // 2
                + packed_k2[None, :]
            )
            packed_gate = tl.load(
                rg_q4_ptr + q4_offsets, mask=branch_rows[:, None], other=0
            )
            packed_up = tl.load(
                ru_q4_ptr + q4_offsets, mask=branch_rows[:, None], other=0
            )
            gather_k2 = tl.broadcast_to(
                (tl.arange(0, BLOCK_K) // 2)[None, :], (BLOCK_N, BLOCK_K)
            )
            q4_shifts = (offsets_k[None, :] & 1) * 4
            raw_gate = (
                tl.gather(packed_gate, gather_k2, axis=1) >> q4_shifts
            ) & 15
            raw_up = (
                tl.gather(packed_up, gather_k2, axis=1) >> q4_shifts
            ) & 15
            q4_gate = tl.where(raw_gate >= 8, raw_gate.to(tl.int32) - 16, raw_gate.to(tl.int32))
            q4_up = tl.where(raw_up >= 8, raw_up.to(tl.int32) - 16, raw_up.to(tl.int32))
            weight_gate = tl.where(
                branch_rows[:, None], q4_gate.to(tl.float32) * branch_s_gate[:, None], weight_gate
            )
            weight_up = tl.where(
                branch_rows[:, None], q4_up.to(tl.float32) * branch_s_up[:, None], weight_up
            )
        gate += tl.dot(x, tl.trans(weight_gate.to(tl.bfloat16)))
        up += tl.dot(x, tl.trans(weight_up.to(tl.bfloat16)))
    # Preserve the BF16 boundaries used by the packed quality artifact:
    # linear -> BF16, SiLU -> BF16, multiply -> BF16.
    gate_bf16 = gate.to(tl.bfloat16)
    up_bf16 = up.to(tl.bfloat16)
    sigmoid = 1.0 / (1.0 + libdevice.exp(-gate_bf16.to(tl.float32)))
    silu_bf16 = (gate_bf16.to(tl.float32) * sigmoid).to(tl.bfloat16)
    activation = (silu_bf16 * up_bf16).to(tl.bfloat16)
    tl.store(
        out_ptr + offsets_r[:, None] * total_width + offsets_n[None, :],
        activation,
        mask=(offsets_r[:, None] < rows) & (offsets_n[None, :] < total_width),
    )


@triton.autotune(configs=UP_CONFIGS, key=["rows", "INCLUDE_BRANCH"])
@triton.jit
def packed_down_kernel(
    hidden_ptr,
    bd_q8_ptr, bd_t_ptr, bd_s8_ptr, bd_st_ptr,
    rd_q4_ptr, rd_s_ptr,
    out_ptr,
    rows,
    INCLUDE_BRANCH: tl.constexpr,
    BLOCK_R: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    program = tl.program_id(0)
    tiles_n = tl.cdiv(K_D, BLOCK_N)
    tile_r = program // tiles_n
    tile_n = program - tile_r * tiles_n
    offsets_r = tile_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offsets_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    output_rows = offsets_n < K_D
    base_accumulator = tl.zeros((BLOCK_R, BLOCK_N), tl.float32)
    base_s8 = tl.load(bd_s8_ptr + offsets_n, mask=output_rows, other=0.0).to(tl.float32)
    base_st = tl.load(bd_st_ptr + offsets_n, mask=output_rows, other=0.0).to(tl.float32)
    for start_k in range(0, K_BASE_WIDTH, BLOCK_K):
        offsets_k = start_k + tl.arange(0, BLOCK_K)
        hidden = tl.load(
            hidden_ptr + offsets_r[:, None] * (K_TOTAL_WIDTH if INCLUDE_BRANCH else K_BASE_WIDTH) + offsets_k[None, :],
            mask=(offsets_r[:, None] < rows) & (offsets_k[None, :] < K_BASE_WIDTH),
            other=0.0,
        )
        base_offsets = offsets_n[:, None] * K_BASE_WIDTH + offsets_k[None, :]
        q8 = tl.load(
            bd_q8_ptr + base_offsets,
            mask=output_rows[:, None] & (offsets_k[None, :] < K_BASE_WIDTH),
            other=0,
        ).to(tl.float32)
        packed_k4 = tl.arange(0, BLOCK_K // 4)
        ternary_offsets = (
            offsets_n[:, None] * (K_BASE_WIDTH // 4)
            + start_k // 4
            + packed_k4[None, :]
        )
        packed_ternary = tl.load(
            bd_t_ptr + ternary_offsets, mask=output_rows[:, None], other=0
        )
        gather_k4 = tl.broadcast_to(
            (tl.arange(0, BLOCK_K) // 4)[None, :], (BLOCK_N, BLOCK_K)
        )
        shifts = (offsets_k[None, :] & 3) * 2
        ternary = (
            (tl.gather(packed_ternary, gather_k4, axis=1) >> shifts) & 3
        ).to(tl.int32) - 1
        weight = q8 * base_s8[:, None] + ternary.to(tl.float32) * base_st[:, None]
        base_accumulator += tl.dot(hidden, tl.trans(weight.to(tl.bfloat16)))
    if INCLUDE_BRANCH:
        branch_accumulator = tl.zeros((BLOCK_R, BLOCK_N), tl.float32)
        branch_scale = tl.load(rd_s_ptr + offsets_n, mask=output_rows, other=0.0).to(tl.float32)
        for start_k in range(0, K_BRANCH_WIDTH, BLOCK_K):
            offsets_k = start_k + tl.arange(0, BLOCK_K)
            hidden = tl.load(
                hidden_ptr + offsets_r[:, None] * K_TOTAL_WIDTH + K_BASE_WIDTH + offsets_k[None, :],
                mask=(offsets_r[:, None] < rows) & (offsets_k[None, :] < K_BRANCH_WIDTH),
                other=0.0,
            )
            packed_k2 = tl.arange(0, BLOCK_K // 2)
            packed_offsets = (
                offsets_n[:, None] * (K_BRANCH_WIDTH // 2)
                + start_k // 2
                + packed_k2[None, :]
            )
            packed_q4 = tl.load(
                rd_q4_ptr + packed_offsets, mask=output_rows[:, None], other=0
            )
            gather_k2 = tl.broadcast_to(
                (tl.arange(0, BLOCK_K) // 2)[None, :], (BLOCK_N, BLOCK_K)
            )
            shifts = (offsets_k[None, :] & 1) * 4
            raw = (tl.gather(packed_q4, gather_k2, axis=1) >> shifts) & 15
            q4 = tl.where(raw >= 8, raw.to(tl.int32) - 16, raw.to(tl.int32))
            weight = q4.to(tl.float32) * branch_scale[:, None]
            branch_accumulator += tl.dot(hidden, tl.trans(weight.to(tl.bfloat16)))
        # The quality run rounded base and branch linears independently before
        # their BF16 add.  Keeping that boundary makes its NLL result portable.
        output = (
            base_accumulator.to(tl.bfloat16)
            + branch_accumulator.to(tl.bfloat16)
        ).to(tl.bfloat16)
    else:
        output = base_accumulator.to(tl.bfloat16)
    tl.store(
        out_ptr + offsets_r[:, None] * K_D + offsets_n[None, :], output,
        mask=(offsets_r[:, None] < rows) & output_rows[None, :],
    )


@triton.autotune(configs=UP_CONFIGS, key=["rows"])
@triton.jit
def dense_upgate_kernel(
    x_ptr, gate_ptr, up_ptr, out_ptr, rows,
    BLOCK_R: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    program = tl.program_id(0)
    tiles_n = tl.cdiv(K_BASE_WIDTH, BLOCK_N)
    tile_r = program // tiles_n
    tile_n = program - tile_r * tiles_n
    offsets_r = tile_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offsets_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    gate_acc = tl.zeros((BLOCK_R, BLOCK_N), tl.float32)
    up_acc = tl.zeros((BLOCK_R, BLOCK_N), tl.float32)
    for start_k in range(0, K_D, BLOCK_K):
        offsets_k = start_k + tl.arange(0, BLOCK_K)
        x = tl.load(
            x_ptr + offsets_r[:, None] * K_D + offsets_k[None, :],
            mask=(offsets_r[:, None] < rows) & (offsets_k[None, :] < K_D), other=0.0,
        )
        weight_offsets = offsets_n[:, None] * K_D + offsets_k[None, :]
        gate = tl.load(gate_ptr + weight_offsets, mask=offsets_n[:, None] < K_BASE_WIDTH, other=0.0)
        up = tl.load(up_ptr + weight_offsets, mask=offsets_n[:, None] < K_BASE_WIDTH, other=0.0)
        gate_acc += tl.dot(x, tl.trans(gate))
        up_acc += tl.dot(x, tl.trans(up))
    gate_bf16 = gate_acc.to(tl.bfloat16)
    up_bf16 = up_acc.to(tl.bfloat16)
    sigmoid = 1.0 / (1.0 + libdevice.exp(-gate_bf16.to(tl.float32)))
    silu_bf16 = (gate_bf16.to(tl.float32) * sigmoid).to(tl.bfloat16)
    tl.store(
        out_ptr + offsets_r[:, None] * K_BASE_WIDTH + offsets_n[None, :],
        (silu_bf16 * up_bf16).to(tl.bfloat16),
        mask=(offsets_r[:, None] < rows) & (offsets_n[None, :] < K_BASE_WIDTH),
    )


@triton.autotune(configs=UP_CONFIGS, key=["rows"])
@triton.jit
def dense_down_kernel(
    hidden_ptr, weight_ptr, out_ptr, rows,
    BLOCK_R: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    program = tl.program_id(0)
    tiles_n = tl.cdiv(K_D, BLOCK_N)
    tile_r = program // tiles_n
    tile_n = program - tile_r * tiles_n
    offsets_r = tile_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offsets_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_R, BLOCK_N), tl.float32)
    for start_k in range(0, K_BASE_WIDTH, BLOCK_K):
        offsets_k = start_k + tl.arange(0, BLOCK_K)
        hidden = tl.load(
            hidden_ptr + offsets_r[:, None] * K_BASE_WIDTH + offsets_k[None, :],
            mask=(offsets_r[:, None] < rows) & (offsets_k[None, :] < K_BASE_WIDTH), other=0.0,
        )
        weight = tl.load(
            weight_ptr + offsets_n[:, None] * K_BASE_WIDTH + offsets_k[None, :],
            mask=(offsets_n[:, None] < K_D) & (offsets_k[None, :] < K_BASE_WIDTH), other=0.0,
        )
        accumulator += tl.dot(hidden, tl.trans(weight))
    tl.store(
        out_ptr + offsets_r[:, None] * K_D + offsets_n[None, :], accumulator,
        mask=(offsets_r[:, None] < rows) & (offsets_n[None, :] < K_D),
    )


def layer_buffers(state: dict[str, torch.Tensor], layer: int = 0) -> dict[str, torch.Tensor]:
    prefix = f"model.layers.{layer}.mlp."
    sources = {
        name: state[prefix + key].contiguous()
        for name, key in PACKED_COMPONENT_KEYS.items()
    }
    ordered_names = tuple(PACKED_COMPONENT_KEYS)
    byte_parts = [sources[name].view(torch.uint8).reshape(-1) for name in ordered_names]
    blob = torch.cat(byte_parts).contiguous()
    result: dict[str, torch.Tensor] = {"_blob": blob}
    offset = 0
    for name, byte_part in zip(ordered_names, byte_parts):
        byte_count = byte_part.numel()
        source = sources[name]
        result[name] = blob[offset : offset + byte_count].view(source.dtype).view(source.shape)
        offset += byte_count
    if offset != serving_ledger()["candidate_ffn_bytes_per_layer"]:
        raise RuntimeError(f"monolithic packed blob has {offset} bytes")
    return result


def packed_layout_audit(buffers: dict[str, torch.Tensor]) -> dict[str, Any]:
    blob = buffers["_blob"]
    blob_pointer = blob.data_ptr()
    segments = []
    expected_offset = 0
    for name in PACKED_COMPONENT_KEYS:
        tensor = buffers[name]
        byte_count = tensor.numel() * tensor.element_size()
        byte_offset = tensor.data_ptr() - blob_pointer
        segments.append(
            {
                "name": name,
                "offset": byte_offset,
                "bytes": byte_count,
                "pointer_128_aligned": tensor.data_ptr() % 128 == 0,
                "shared_blob_storage": (
                    tensor.untyped_storage().data_ptr()
                    == blob.untyped_storage().data_ptr()
                ),
                "expected_offset": byte_offset == expected_offset,
            }
        )
        expected_offset += byte_count
    return {
        "logical_bytes": blob.numel() * blob.element_size(),
        "storage_bytes": blob.untyped_storage().nbytes(),
        "blob_pointer_128_aligned": blob_pointer % 128 == 0,
        "all_segments_valid": all(
            segment["pointer_128_aligned"]
            and segment["shared_blob_storage"]
            and segment["expected_offset"]
            for segment in segments
        ),
        "segments": segments,
    }


def decode_weights(buffers: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = {}
    for stem, rows, columns in (
        ("bg", BASE_WIDTH, D), ("bu", BASE_WIDTH, D), ("bd", D, BASE_WIDTH),
    ):
        q8 = buffers[f"{stem}_q8"].view(rows, columns).float()
        ternary = unpack_ternary(buffers[f"{stem}_t"], rows * columns).view(rows, columns).float()
        result[stem] = (
            q8 * buffers[f"{stem}_s8"].float().view(-1, 1)
            + ternary * buffers[f"{stem}_st"].float().view(-1, 1)
        ).to(torch.bfloat16)
    for stem, rows, columns in (
        ("rg", BRANCH_WIDTH, D), ("ru", BRANCH_WIDTH, D), ("rd", D, BRANCH_WIDTH),
    ):
        q4 = unpack_int4(buffers[f"{stem}_q4"], rows * columns).view(rows, columns).float()
        result[stem] = (q4 * buffers[f"{stem}_s"].float().view(-1, 1)).to(torch.bfloat16)
    return result


def packed_launch(x, hidden, out, buffers, include_branch: bool):
    rows = x.shape[0]
    width = TOTAL_WIDTH if include_branch else BASE_WIDTH
    grid_up = lambda meta: (triton.cdiv(rows, meta["BLOCK_R"]) * triton.cdiv(width, meta["BLOCK_N"]),)
    up_kernel = packed_upgate_kernel[grid_up](
        x,
        buffers["bg_q8"], buffers["bg_t"], buffers["bg_s8"], buffers["bg_st"],
        buffers["bu_q8"], buffers["bu_t"], buffers["bu_s8"], buffers["bu_st"],
        buffers["rg_q4"], buffers["rg_s"], buffers["ru_q4"], buffers["ru_s"],
        hidden, rows=rows, INCLUDE_BRANCH=include_branch,
    )
    grid_down = lambda meta: (triton.cdiv(rows, meta["BLOCK_R"]) * triton.cdiv(D, meta["BLOCK_N"]),)
    down_kernel = packed_down_kernel[grid_down](
        hidden,
        buffers["bd_q8"], buffers["bd_t"], buffers["bd_s8"], buffers["bd_st"],
        buffers["rd_q4"], buffers["rd_s"], out,
        rows=rows, INCLUDE_BRANCH=include_branch,
    )
    return up_kernel, down_kernel


def dense_launch(x, hidden, out, weights):
    rows = x.shape[0]
    grid_up = lambda meta: (triton.cdiv(rows, meta["BLOCK_R"]) * triton.cdiv(BASE_WIDTH, meta["BLOCK_N"]),)
    up_kernel = dense_upgate_kernel[grid_up](
        x, weights["bg"], weights["bu"], hidden, rows=rows
    )
    grid_down = lambda meta: (triton.cdiv(rows, meta["BLOCK_R"]) * triton.cdiv(D, meta["BLOCK_N"]),)
    down_kernel = dense_down_kernel[grid_down](
        hidden, weights["bd"], out, rows=rows
    )
    return up_kernel, down_kernel


def packed_stream_launch(x, hidden, out, layer_buffer_list, include_branch: bool):
    for buffers in layer_buffer_list:
        packed_launch(x, hidden, out, buffers, include_branch)


def dense_stream_launch(x, hidden, out, layer_weight_list):
    for weights in layer_weight_list:
        dense_launch(x, hidden, out, weights)


def reference(x, weights, include_branch):
    base = F.linear(F.silu(F.linear(x, weights["bg"])) * F.linear(x, weights["bu"]), weights["bd"])
    if not include_branch:
        return base
    branch = F.linear(F.silu(F.linear(x, weights["rg"])) * F.linear(x, weights["ru"]), weights["rd"])
    return base + branch


def dense_cublas_reference(x, weights):
    gate_up = F.linear(x, weights["bgu"])
    gate, up = gate_up.split(BASE_WIDTH, dim=-1)
    return F.linear(F.silu(gate) * up, weights["bd"])


def dense_cublas_stream_reference(x, layer_weight_list):
    output = None
    for weights in layer_weight_list:
        output = dense_cublas_reference(x, weights)
    return output


def error_record(actual, expected):
    difference = actual.float() - expected.float()
    actual_bits = actual.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    expected_bits = expected.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    actual_ordered = torch.where(
        actual_bits >= 0x8000, 0x8000 - (actual_bits & 0x7FFF), 0x8000 + actual_bits
    )
    expected_ordered = torch.where(
        expected_bits >= 0x8000, 0x8000 - (expected_bits & 0x7FFF), 0x8000 + expected_bits
    )
    ulp_error = (actual_ordered - expected_ordered).abs()
    expected_max = expected.float().abs().max()
    return {
        "all_finite": bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()),
        "relative_l2": float(torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(expected.float()).clamp_min(1e-30)),
        "max_absolute": float(difference.abs().max()),
        "max_absolute_over_expected_max": float(difference.abs().max() / expected_max.clamp_min(1e-30)),
        "p99_bf16_ulp": float(torch.quantile(ulp_error.float(), 0.99)),
        "max_bf16_ulp": int(ulp_error.max()),
    }


def capture(function):
    function(); torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        retained_output = function()
    return graph, retained_output


def compiled_kernel_record(compiled_kernel) -> dict[str, Any]:
    if compiled_kernel is None:
        return {"error": "launcher returned no compiled kernel metadata"}
    try:
        assembly = compiled_kernel.asm
        ptx = assembly.get("ptx", "")
        if isinstance(ptx, bytes):
            ptx_bytes = ptx
            ptx_text = ptx.decode("utf-8", errors="replace")
        else:
            ptx_text = str(ptx)
            ptx_bytes = ptx_text.encode()
        metadata = compiled_kernel.metadata
        shared = (
            metadata.get("shared", 0)
            if isinstance(metadata, dict)
            else getattr(metadata, "shared", 0)
        )
        return {
            "name": getattr(compiled_kernel, "name", None),
            "registers": int(compiled_kernel.n_regs),
            "spills": int(compiled_kernel.n_spills),
            "shared_bytes": int(shared),
            "ptx_sha256": hashlib.sha256(ptx_bytes).hexdigest(),
            "ptx_has_tensor_core_mma": (
                "mma.sync" in ptx_text or "wgmma.mma" in ptx_text
            ),
            "ptx_declares_local_memory": ".local" in ptx_text,
            "assembly_keys": sorted(str(key) for key in assembly),
        }
    except (AttributeError, TypeError, ValueError) as error:
        return {"error": repr(error)}


def compiled_pair_record(compiled_pair, up_autotuner, down_autotuner):
    up_kernel, down_kernel = compiled_pair
    return {
        "up": compiled_kernel_record(up_kernel),
        "down": compiled_kernel_record(down_kernel),
        "up_best_config": repr(getattr(up_autotuner, "best_config", None)),
        "down_best_config": repr(getattr(down_autotuner, "best_config", None)),
    }


def gpu_state() -> dict[str, Any]:
    fields = (
        "clocks.current.sm,clocks.current.memory,clocks.max.sm,"
        "clocks.max.memory,power.draw,temperature.gpu,pstate"
    )
    try:
        completed = subprocess.run(
            [
                "nvidia-smi", f"--query-gpu={fields}",
                "--format=csv,noheader,nounits", "--id=0",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        values = [value.strip() for value in completed.stdout.strip().split(",")]
        if len(values) != 7:
            raise RuntimeError(completed.stdout.strip())
        return {
            "sm_clock_mhz": float(values[0]),
            "memory_clock_mhz": float(values[1]),
            "max_sm_clock_mhz": float(values[2]),
            "max_memory_clock_mhz": float(values[3]),
            "power_watts": float(values[4]),
            "temperature_c": float(values[5]),
            "pstate": values[6],
        }
    except (OSError, subprocess.CalledProcessError, RuntimeError, ValueError) as error:
        return {"error": repr(error)}


def gpu_inventory() -> dict[str, Any]:
    fields = (
        "name,uuid,driver_version,memory.total,power.limit,mig.mode.current"
    )
    try:
        completed = subprocess.run(
            [
                "nvidia-smi", f"--query-gpu={fields}",
                "--format=csv,noheader,nounits", "--id=0",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        values = [value.strip() for value in completed.stdout.strip().split(",")]
        if len(values) != 6:
            raise RuntimeError(completed.stdout.strip())
        return {
            "name": values[0],
            "uuid": values[1],
            "driver": values[2],
            "memory_mib": float(values[3]),
            "power_limit_watts": float(values[4]),
            "mig_mode": values[5],
        }
    except (OSError, subprocess.CalledProcessError, RuntimeError, ValueError) as error:
        return {"error": repr(error)}


def stable_gpu_states(record: dict[str, Any]) -> bool:
    states = tuple(record["gpu_states"].values())
    if not states or any("error" in state for state in states):
        return False
    return (
        len({state["pstate"] for state in states}) == 1
        and all(
            state["sm_clock_mhz"] >= 0.80 * state["max_sm_clock_mhz"]
            and state["memory_clock_mhz"] >= 0.95 * state["max_memory_clock_mhz"]
            for state in states
        )
    )


def interleaved_samples(
    graphs: dict[str, torch.cuda.CUDAGraph],
    *,
    seed: int,
    count: int = 1000,
    warmups: int = 200,
    l2_flush: torch.Tensor | None = None,
) -> dict[str, list[float]]:
    names = tuple(sorted(graphs))
    for _ in range(warmups):
        for name in names:
            graphs[name].replay()
    torch.cuda.synchronize()
    rng = np.random.default_rng(seed)
    values = {name: [] for name in names}
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    for _ in range(count):
        for name_index in rng.permutation(len(names)):
            name = names[int(name_index)]
            if l2_flush is not None:
                l2_flush.zero_()
            start.record()
            graphs[name].replay()
            stop.record()
            stop.synchronize()
            values[name].append(float(start.elapsed_time(stop)))
    return values


def bootstrap_ratio(candidate, baseline, seed):
    candidate = np.asarray(candidate, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    rng = np.random.default_rng(seed)
    ratios = []
    if candidate.size != baseline.size:
        raise ValueError("paired timing samples must have equal size")
    for _ in range(5000):
        indices = rng.integers(0, candidate.size, candidate.size)
        ratios.append(np.median(candidate[indices]) / np.median(baseline[indices]))
    return {
        "median_ratio": float(np.median(candidate) / np.median(baseline)),
        "lower_95": float(np.quantile(ratios, 0.025)),
        "upper_95": float(np.quantile(ratios, 0.975)),
    }


def bootstrap_service_ratio(
    candidate_prefill,
    candidate_decode,
    dense_prefill,
    dense_decode,
    *,
    decode_steps: int,
    seed: int,
):
    arrays = [
        np.asarray(values, dtype=np.float64)
        for values in (candidate_prefill, candidate_decode, dense_prefill, dense_decode)
    ]
    if arrays[0].size != arrays[2].size or arrays[1].size != arrays[3].size:
        raise ValueError("paired service cells must have matching sample counts")
    rng = np.random.default_rng(seed)

    def ratio(parts):
        cp, cd, dp, dd = (np.median(part) for part in parts)
        return (cp + decode_steps * cd) / (dp + decode_steps * dd)

    ratios = []
    for _ in range(5000):
        prefill_indices = rng.integers(0, arrays[0].size, arrays[0].size)
        decode_indices = rng.integers(0, arrays[1].size, arrays[1].size)
        resampled = [
            arrays[0][prefill_indices],
            arrays[1][decode_indices],
            arrays[2][prefill_indices],
            arrays[3][decode_indices],
        ]
        ratios.append(ratio(resampled))
    return {
        "median_ratio": float(ratio(arrays)),
        "lower_95": float(np.quantile(ratios, 0.025)),
        "upper_95": float(np.quantile(ratios, 0.975)),
    }


def run(smoke: bool) -> dict[str, Any]:
    if torch.__version__ != "2.5.1+cu124" or torch.version.cuda != "12.4":
        raise RuntimeError("frozen Torch/CUDA runtime required")
    if triton.__version__ != "3.6.0" or not str(Path(triton.__file__).resolve()).startswith("/workspace/triton36/"):
        raise RuntimeError("isolated Triton 3.6.0 required")
    device_properties = torch.cuda.get_device_properties(0)
    if (
        "H100" not in device_properties.name
        or device_properties.multi_processor_count != 132
        or device_properties.total_memory < 79_000_000_000
    ):
        raise RuntimeError(
            "full 80GB H100 SXM required: "
            f"{device_properties.name}, {device_properties.multi_processor_count} SMs, "
            f"{device_properties.total_memory} bytes"
        )
    if sha256_file(QUALITY_RESULT) != QUALITY_RESULT_SHA256:
        raise RuntimeError("frozen quality result changed")
    artifact_sha = json.loads(QUALITY_RESULT.read_text())["artifact"]["sha256"]
    if sha256_file(ARTIFACT) != artifact_sha:
        raise RuntimeError("frozen packed artifact changed")
    if not smoke:
        integrity = json.loads(INTEGRITY.read_text())
        if sha256_file(Path(__file__)) != integrity["source_sha256"]:
            raise RuntimeError("frozen source changed")
        if sha256_file(PREREGISTRATION) != integrity["preregistration_sha256"]:
            raise RuntimeError("frozen preregistration changed")
        for helper in HELPER_FILES:
            if sha256_file(helper) != integrity["helper_sha256"][str(helper)]:
                raise RuntimeError(f"frozen helper changed: {helper}")
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")
    state = torch.load(ARTIFACT, map_location=device, weights_only=True)["model"]
    layer_buffer_list = [layer_buffers(state, layer) for layer in range(LAYERS)]
    del state
    gc.collect()
    torch.cuda.empty_cache()
    candidate_isolated_allocated_bytes = torch.cuda.memory_allocated(device)
    candidate_isolated_reserved_bytes = torch.cuda.memory_reserved(device)
    layout_audits = [packed_layout_audit(buffers) for buffers in layer_buffer_list]
    buffers = layer_buffer_list[0]
    layer_weight_list = [decode_weights(layer) for layer in layer_buffer_list]
    for layer_weights in layer_weight_list:
        layer_weights["bgu"] = torch.cat(
            (layer_weights["bg"], layer_weights["bu"]), dim=0
        ).contiguous()
    weights = layer_weight_list[0]
    test_rows = (2,) if smoke else ROWS
    l2_flush = None if smoke else torch.empty(128 * 1024 * 1024, device=device, dtype=torch.uint8)
    records = {}
    for rows in test_rows:
        x = torch.randn(rows, D, device=device, dtype=torch.bfloat16)
        dense_hidden = torch.empty(rows, BASE_WIDTH, device=device, dtype=torch.bfloat16)
        dense_out = torch.empty(rows, D, device=device, dtype=torch.bfloat16)
        dense_compiled = dense_launch(x, dense_hidden, dense_out, weights)
        dense_error = error_record(dense_out, reference(x, weights, False))
        row_record = {
            "dense_error": dense_error,
            "kernel_resources": {
                "dense_triton_fused": compiled_pair_record(
                    dense_compiled, dense_upgate_kernel, dense_down_kernel
                )
            },
        }
        graphs: dict[str, torch.cuda.CUDAGraph] = {}
        retained_tensors: dict[str, Any] = {}
        for name, include_branch in (("base", False), ("full", True)):
            width = TOTAL_WIDTH if include_branch else BASE_WIDTH
            hidden = torch.empty(rows, width, device=device, dtype=torch.bfloat16)
            out = torch.empty(rows, D, device=device, dtype=torch.bfloat16)
            packed_compiled = packed_launch(
                x, hidden, out, buffers, include_branch
            )
            row_record[f"{name}_error"] = error_record(out, reference(x, weights, include_branch))
            row_record["kernel_resources"][name] = compiled_pair_record(
                packed_compiled, packed_upgate_kernel, packed_down_kernel
            )
            if not smoke:
                graph, _ = capture(
                    lambda: packed_launch(x, hidden, out, buffers, include_branch)
                )
                graphs[name] = graph
                retained_tensors[f"{name}_hidden"] = hidden
                retained_tensors[f"{name}_out"] = out
        if not smoke:
            dense_triton_graph, _ = capture(
                lambda: dense_launch(x, dense_hidden, dense_out, weights)
            )
            dense_cublas_graph, dense_cublas_output = capture(
                lambda: dense_cublas_reference(x, weights)
            )
            graphs["dense_triton_fused"] = dense_triton_graph
            graphs["dense_cublas_packed"] = dense_cublas_graph
            retained_tensors["dense_hidden"] = dense_hidden
            retained_tensors["dense_out"] = dense_out
            retained_tensors["dense_cublas_output"] = dense_cublas_output

            stream_graphs: dict[str, torch.cuda.CUDAGraph] = {}
            for name, include_branch in (("base", False), ("full", True)):
                width = TOTAL_WIDTH if include_branch else BASE_WIDTH
                stream_hidden = torch.empty(
                    rows, width, device=device, dtype=torch.bfloat16
                )
                stream_out = torch.empty(
                    rows, D, device=device, dtype=torch.bfloat16
                )
                stream_graph, _ = capture(
                    lambda: packed_stream_launch(
                        x, stream_hidden, stream_out, layer_buffer_list,
                        include_branch,
                    )
                )
                stream_graphs[name] = stream_graph
                retained_tensors[f"stream_{name}_hidden"] = stream_hidden
                retained_tensors[f"stream_{name}_out"] = stream_out
            dense_stream_hidden = torch.empty(
                rows, BASE_WIDTH, device=device, dtype=torch.bfloat16
            )
            dense_stream_out = torch.empty(
                rows, D, device=device, dtype=torch.bfloat16
            )
            dense_triton_stream_graph, _ = capture(
                lambda: dense_stream_launch(
                    x, dense_stream_hidden, dense_stream_out, layer_weight_list
                )
            )
            dense_cublas_stream_graph, dense_cublas_stream_output = capture(
                lambda: dense_cublas_stream_reference(x, layer_weight_list)
            )
            stream_graphs["dense_triton_fused"] = dense_triton_stream_graph
            stream_graphs["dense_cublas_packed"] = dense_cublas_stream_graph
            retained_tensors["dense_stream_hidden"] = dense_stream_hidden
            retained_tensors["dense_stream_out"] = dense_stream_out
            retained_tensors["dense_cublas_stream_output"] = dense_cublas_stream_output

            for _ in range(20):
                for graph_group in (graphs, stream_graphs):
                    for graph_name in sorted(graph_group):
                        graph_group[graph_name].replay()
            torch.cuda.synchronize()
            row_record["gpu_states"] = {"single_warm_before": gpu_state()}
            warm_samples = interleaved_samples(
                graphs, seed=SEED + rows, l2_flush=None
            )
            row_record["gpu_states"]["single_warm_after"] = gpu_state()
            row_record["gpu_states"]["single_cold_l2_before"] = gpu_state()
            cold_samples = interleaved_samples(
                graphs, seed=SEED + 10_000 + rows, l2_flush=l2_flush
            )
            row_record["gpu_states"]["single_cold_l2_after"] = gpu_state()
            row_record["gpu_states"]["stream_warm_before"] = gpu_state()
            stream_warm_samples = interleaved_samples(
                stream_graphs, seed=SEED + 20_000 + rows, l2_flush=None
            )
            row_record["gpu_states"]["stream_warm_after"] = gpu_state()
            row_record["gpu_states"]["stream_cold_l2_before"] = gpu_state()
            stream_cold_samples = interleaved_samples(
                stream_graphs, seed=SEED + 30_000 + rows, l2_flush=l2_flush
            )
            row_record["gpu_states"]["stream_cold_l2_after"] = gpu_state()
            dense_names = ("dense_triton_fused", "dense_cublas_packed")
            dense_warm_name = min(
                dense_names, key=lambda name: median(warm_samples[name])
            )
            dense_cold_name = min(
                dense_names, key=lambda name: median(cold_samples[name])
            )
            dense_stream_warm_name = min(
                dense_names, key=lambda name: median(stream_warm_samples[name])
            )
            dense_stream_cold_name = min(
                dense_names, key=lambda name: median(stream_cold_samples[name])
            )
            row_record["dense_control_selected"] = {
                "warm": dense_warm_name,
                "cold_l2": dense_cold_name,
                "stream_warm": dense_stream_warm_name,
                "stream_cold_l2": dense_stream_cold_name,
            }
            row_record["warm_samples_ms"] = warm_samples
            row_record["cold_l2_samples_ms"] = cold_samples
            row_record["dense_samples_ms"] = warm_samples[dense_warm_name]
            row_record["base_samples_ms"] = warm_samples["base"]
            row_record["full_samples_ms"] = warm_samples["full"]
            row_record["dense_cold_l2_samples_ms"] = cold_samples[dense_cold_name]
            row_record["base_cold_l2_samples_ms"] = cold_samples["base"]
            row_record["full_cold_l2_samples_ms"] = cold_samples["full"]
            row_record["stream_warm_samples_ms"] = stream_warm_samples
            row_record["stream_cold_l2_samples_ms"] = stream_cold_samples
            row_record["dense_stream_samples_ms"] = stream_warm_samples[dense_stream_warm_name]
            row_record["base_stream_samples_ms"] = stream_warm_samples["base"]
            row_record["full_stream_samples_ms"] = stream_warm_samples["full"]
            row_record["dense_stream_cold_l2_samples_ms"] = stream_cold_samples[dense_stream_cold_name]
            row_record["base_stream_cold_l2_samples_ms"] = stream_cold_samples["base"]
            row_record["full_stream_cold_l2_samples_ms"] = stream_cold_samples["full"]
            row_record["base_ratio"] = bootstrap_ratio(
                row_record["base_samples_ms"], row_record["dense_samples_ms"],
                SEED + rows,
            )
            row_record["full_ratio"] = bootstrap_ratio(
                row_record["full_samples_ms"], row_record["dense_samples_ms"],
                SEED + 1000 + rows,
            )
            row_record["base_cold_l2_ratio"] = bootstrap_ratio(
                row_record["base_cold_l2_samples_ms"],
                row_record["dense_cold_l2_samples_ms"],
                SEED + 20_000 + rows,
            )
            row_record["full_cold_l2_ratio"] = bootstrap_ratio(
                row_record["full_cold_l2_samples_ms"],
                row_record["dense_cold_l2_samples_ms"],
                SEED + 30_000 + rows,
            )
            row_record["base_stream_ratio"] = bootstrap_ratio(
                row_record["base_stream_samples_ms"],
                row_record["dense_stream_samples_ms"],
                SEED + 40_000 + rows,
            )
            row_record["full_stream_ratio"] = bootstrap_ratio(
                row_record["full_stream_samples_ms"],
                row_record["dense_stream_samples_ms"],
                SEED + 50_000 + rows,
            )
            row_record["base_stream_cold_l2_ratio"] = bootstrap_ratio(
                row_record["base_stream_cold_l2_samples_ms"],
                row_record["dense_stream_cold_l2_samples_ms"],
                SEED + 60_000 + rows,
            )
            row_record["full_stream_cold_l2_ratio"] = bootstrap_ratio(
                row_record["full_stream_cold_l2_samples_ms"],
                row_record["dense_stream_cold_l2_samples_ms"],
                SEED + 70_000 + rows,
            )
            dense_p99 = float(np.quantile(row_record["dense_samples_ms"], 0.99))
            row_record["base_p99_ratio"] = float(np.quantile(row_record["base_samples_ms"], 0.99) / dense_p99)
            row_record["full_p99_ratio"] = float(np.quantile(row_record["full_samples_ms"], 0.99) / dense_p99)
            dense_stream_p99 = float(
                np.quantile(row_record["dense_stream_samples_ms"], 0.99)
            )
            row_record["base_stream_p99_ratio"] = float(
                np.quantile(row_record["base_stream_samples_ms"], 0.99)
                / dense_stream_p99
            )
            row_record["full_stream_p99_ratio"] = float(
                np.quantile(row_record["full_stream_samples_ms"], 0.99)
                / dense_stream_p99
            )
            dense_stream_cold_p99 = float(
                np.quantile(row_record["dense_stream_cold_l2_samples_ms"], 0.99)
            )
            row_record["base_stream_cold_l2_p99_ratio"] = float(
                np.quantile(row_record["base_stream_cold_l2_samples_ms"], 0.99)
                / dense_stream_cold_p99
            )
            row_record["full_stream_cold_l2_p99_ratio"] = float(
                np.quantile(row_record["full_stream_cold_l2_samples_ms"], 0.99)
                / dense_stream_cold_p99
            )
        records[str(rows)] = row_record
    if smoke:
        return {"schema": "phase-elastic-packed-h100-smoke-v1", "records": records}
    ledger = serving_ledger()
    dense_layer_bytes = ledger["baseline_bf16_ffn_bytes_per_layer"]
    candidate_layer_bytes = ledger["candidate_ffn_bytes_per_layer"]
    workspace = {
        str(rows): {
            "dense_bytes": rows * BASE_WIDTH * 2,
            "candidate_base_bytes": rows * BASE_WIDTH * 2,
            "candidate_full_bytes": rows * TOTAL_WIDTH * 2,
            "dense_model_plus_workspace": LAYERS * dense_layer_bytes + rows * BASE_WIDTH * 2,
            "candidate_model_plus_full_workspace": LAYERS * candidate_layer_bytes + rows * TOTAL_WIDTH * 2,
        }
        for rows in ROWS
    }
    service_ratio = bootstrap_service_ratio(
        records["512"]["base_stream_samples_ms"],
        records["1"]["full_stream_samples_ms"],
        records["512"]["dense_stream_samples_ms"],
        records["1"]["dense_stream_samples_ms"],
        decode_steps=128,
        seed=SEED + 40_000,
    )
    service_cold_ratio = bootstrap_service_ratio(
        records["512"]["base_stream_cold_l2_samples_ms"],
        records["1"]["full_stream_cold_l2_samples_ms"],
        records["512"]["dense_stream_cold_l2_samples_ms"],
        records["1"]["dense_stream_cold_l2_samples_ms"],
        decode_steps=128,
        seed=SEED + 50_000,
    )
    service_dense = (
        median(records["512"]["dense_stream_samples_ms"])
        + 128 * median(records["1"]["dense_stream_samples_ms"])
    )
    service_candidate = (
        median(records["512"]["base_stream_samples_ms"])
        + 128 * median(records["1"]["full_stream_samples_ms"])
    )
    inventory = gpu_inventory()
    gates = {
        "all_numerics_within_1e_3_relative_l2_and_2_bf16_ulps": all(
            record[key]["all_finite"]
            and record[key]["relative_l2"] <= 0.001
            and record[key]["max_bf16_ulp"] <= 2
            for record in records.values() for key in ("base_error", "full_error")
        ),
        "persistent_bytes_do_not_exceed_dense": candidate_layer_bytes <= dense_layer_bytes,
        "twelve_monolithic_packed_blobs_have_exact_candidate_bytes": (
            len(layer_buffer_list) == LAYERS
            and all(
                audit["logical_bytes"] == candidate_layer_bytes
                and audit["storage_bytes"] == candidate_layer_bytes
                and audit["blob_pointer_128_aligned"]
                and audit["all_segments_valid"]
                for audit in layout_audits
            )
        ),
        "isolated_candidate_torch_allocation_is_exact": (
            candidate_isolated_allocated_bytes
            == LAYERS * candidate_layer_bytes
        ),
        "decode_model_plus_workspace_no_larger_at_rows_1_8_32": all(
            workspace[str(rows)]["candidate_model_plus_full_workspace"]
            <= workspace[str(rows)]["dense_model_plus_workspace"]
            for rows in DECODE_ROWS
        ),
        "engineering_base_ratio_at_most_1p10_all_rows": all(
            max(
                records[str(rows)]["base_stream_ratio"]["upper_95"],
                records[str(rows)]["base_stream_cold_l2_ratio"]["upper_95"],
            ) <= 1.10
            for rows in ROWS
        ),
        "engineering_full_ratio_caps": all(
            max(
                records[str(rows)]["full_stream_ratio"]["upper_95"],
                records[str(rows)]["full_stream_cold_l2_ratio"]["upper_95"],
            ) <= cap
            for rows, cap in zip(ROWS, (1.10, 1.10, 1.15, 1.35, 2.10, 2.56))
        ),
        "p99_ratio_no_more_than_p50_plus_0p10": all(
            records[str(rows)][f"{mode}_{condition}_p99_ratio"]
            <= records[str(rows)][f"{mode}_{condition}_ratio"]["median_ratio"] + 0.10
            for rows in ROWS
            for mode in ("base", "full")
            for condition in ("stream", "stream_cold_l2")
        ),
        "breakthrough_base_prefill_gate": (
            all(
                max(
                    records[str(rows)]["base_stream_ratio"]["median_ratio"],
                    records[str(rows)]["base_stream_cold_l2_ratio"]["median_ratio"],
                ) <= cap
                and max(
                    records[str(rows)]["base_stream_ratio"]["upper_95"],
                    records[str(rows)]["base_stream_cold_l2_ratio"]["upper_95"],
                ) <= cap + 0.02
                for rows, cap in ((128, 1.00), (512, 1.00), (2048, 1.05))
            )
        ),
        "breakthrough_full_decode_gate": (
            all(
                max(
                    records[str(rows)]["full_stream_ratio"]["median_ratio"],
                    records[str(rows)]["full_stream_cold_l2_ratio"]["median_ratio"],
                ) <= cap
                and max(
                    records[str(rows)]["full_stream_ratio"]["upper_95"],
                    records[str(rows)]["full_stream_cold_l2_ratio"]["upper_95"],
                ) <= cap + 0.02
                for rows, cap in ((1, 1.00), (8, 1.00), (32, 1.10))
            )
        ),
        "ffn_service_estimate_512_prompt_128_decode_no_slower": (
            service_ratio["median_ratio"] <= 1.00
            and service_ratio["upper_95"] <= 1.02
            and service_cold_ratio["median_ratio"] <= 1.00
            and service_cold_ratio["upper_95"] <= 1.02
        ),
        "clock_power_state_reported": all(
            "error" not in state
            for record in records.values()
            for state in record["gpu_states"].values()
        ),
        "clock_state_stable": all(stable_gpu_states(record) for record in records.values()),
        "exact_h100_sxm_inventory_and_no_mig": (
            "error" not in inventory
            and inventory["mig_mode"].lower() == "disabled"
        ),
        "dense_control_matches_prior_floor_within_5_percent": all(
            median(records[str(rows)]["dense_samples_ms"])
            <= 1.05 * PRIOR_DENSE_GRAPH_MS[rows]
            for rows in ROWS
        ),
        "candidate_kernels_use_tensor_cores_without_spills_or_local_memory": all(
            "error" not in records[str(rows)]["kernel_resources"][mode][part]
            and records[str(rows)]["kernel_resources"][mode][part]["spills"] == 0
            and records[str(rows)]["kernel_resources"][mode][part]["ptx_has_tensor_core_mma"]
            and not records[str(rows)]["kernel_resources"][mode][part]["ptx_declares_local_memory"]
            for rows in ROWS
            for mode in ("base", "full")
            for part in ("up", "down")
        ),
    }
    return {
        "schema": "phase-elastic-packed-h100-v1",
        "source_sha256": sha256_file(Path(__file__)),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "artifact_sha256": sha256_file(ARTIFACT),
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": triton.__version__,
            "gpu": torch.cuda.get_device_name(),
            "sm_count": device_properties.multi_processor_count,
            "total_memory_bytes": device_properties.total_memory,
            "inventory": inventory,
            "helper_sha256": {
                str(helper): sha256_file(helper) for helper in HELPER_FILES
            },
        },
        "scope": "FFN-only physical gate with a 12-layer weight stream; not a full-model end-to-end benchmark",
        "records": records,
        "workspace": workspace,
        "ffn_service_estimate": {
            "layers": LAYERS,
            "prompt_rows": 512,
            "decode_steps": 128,
            "decode_rows": 1,
            "dense_ms": service_dense,
            "candidate_ms": service_candidate,
            "warm_ratio": service_ratio,
            "cold_l2_ratio": service_cold_ratio,
        },
        "storage_accounting": {
            "candidate_kernel_persistent_blob_bytes": buffers["_blob"].numel(),
            "candidate_twelve_layer_persistent_blob_bytes": sum(
                layer["_blob"].numel() for layer in layer_buffer_list
            ),
            "candidate_isolated_torch_allocated_bytes": candidate_isolated_allocated_bytes,
            "candidate_isolated_torch_reserved_bytes": candidate_isolated_reserved_bytes,
            "layout_audits": layout_audits,
            "benchmark_only_expanded_oracle_and_dense_control_bytes": sum(
                tensor.numel() * tensor.element_size()
                for layer_weights in layer_weight_list
                for tensor in layer_weights.values()
            ),
            "expanded_weights_are_not_candidate_kernel_arguments": True,
            "compiler_spill_and_cuda_graph_pool_audit_pending": True,
        },
        "gates": gates,
        "prototype_gate_pass": all(gates.values()),
        "physical_breakthrough_proven": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/phase-elastic-packed-h100.json"))
    args = parser.parse_args()
    result = run(args.smoke)
    if args.smoke:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        write_payload(args.output, result)
        summary = {
            "primary_stream_ratios": {
                rows: {
                    "base_warm": result["records"][rows]["base_stream_ratio"],
                    "base_cold_l2": result["records"][rows]["base_stream_cold_l2_ratio"],
                    "full_warm": result["records"][rows]["full_stream_ratio"],
                    "full_cold_l2": result["records"][rows]["full_stream_cold_l2_ratio"],
                }
                for rows in result["records"]
            },
            "ffn_service_estimate": result["ffn_service_estimate"],
            "gates": result["gates"],
            "prototype_gate_pass": result["prototype_gate_pass"],
            "physical_breakthrough_proven": result["physical_breakthrough_proven"],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
