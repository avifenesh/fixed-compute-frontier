#!/usr/bin/env python3
"""Exact packed FFN artifact used to delete phase-elastic training shadows."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from experiments.phase_elastic_precision_ffn_lm_screen import QuantizedWeight
from experiments.phase_elastic_residual_precision_v2_lm_screen import (
    ResidualQuantizedWeight,
)


def pack_ternary(codes: torch.Tensor) -> torch.Tensor:
    flat = (codes.to(torch.int16).reshape(-1) + 1).to(torch.uint8)
    if flat.numel() % 4:
        raise ValueError("ternary coordinate count must be divisible by four")
    grouped = flat.view(-1, 4)
    return (
        grouped[:, 0]
        | (grouped[:, 1] << 2)
        | (grouped[:, 2] << 4)
        | (grouped[:, 3] << 6)
    ).contiguous()


def unpack_ternary(packed: torch.Tensor, coordinates: int) -> torch.Tensor:
    shifts = torch.tensor((0, 2, 4, 6), device=packed.device, dtype=torch.uint8)
    encoded = ((packed.view(-1, 1) >> shifts) & 0x3).reshape(-1)[:coordinates]
    if bool((encoded > 2).any()):
        raise RuntimeError("packed ternary artifact contains reserved code 3")
    return encoded.to(torch.int8) - 1


def pack_int4(codes: torch.Tensor) -> torch.Tensor:
    flat = (codes.to(torch.int16).reshape(-1) & 0xF).to(torch.uint8)
    if flat.numel() % 2:
        raise ValueError("int4 coordinate count must be divisible by two")
    grouped = flat.view(-1, 2)
    return (grouped[:, 0] | (grouped[:, 1] << 4)).contiguous()


def unpack_int4(packed: torch.Tensor, coordinates: int) -> torch.Tensor:
    low = packed & 0xF
    high = (packed >> 4) & 0xF
    encoded = torch.stack((low, high), dim=1).reshape(-1)[:coordinates].to(torch.int16)
    return torch.where(encoded >= 8, encoded - 16, encoded).to(torch.int8)


class PackedResidualWeight(nn.Module):
    def __init__(
        self,
        shape: tuple[int, int],
        base_codes: torch.Tensor,
        ternary_codes: torch.Tensor,
        base_scale: torch.Tensor,
        delta_scale: torch.Tensor,
    ) -> None:
        super().__init__()
        self.shape = shape
        self.register_buffer("base_codes", base_codes.to(torch.int8).contiguous())
        self.register_buffer("ternary_codes", ternary_codes.to(torch.uint8).contiguous())
        self.register_buffer("base_scale", base_scale.to(torch.bfloat16).reshape(-1).contiguous())
        self.register_buffer("delta_scale", delta_scale.to(torch.bfloat16).reshape(-1).contiguous())

    @classmethod
    @torch.no_grad()
    def from_training(cls, source: ResidualQuantizedWeight) -> "PackedResidualWeight":
        base_scale = source.base_log_scale.exp().to(torch.bfloat16)
        delta_scale = source.delta_log_scale.exp().to(torch.bfloat16)
        base_codes = torch.round(
            source.base_shadow.detach().float() / base_scale.float()
        ).clamp(-127, 127).to(torch.int8)
        normalized = source.delta_shadow.detach().float() / delta_scale.float()
        ternary = (
            torch.sign(normalized) * (normalized.abs() >= 0.5)
        ).to(torch.int8)
        return cls(
            tuple(base_codes.shape),
            base_codes.reshape(-1),
            pack_ternary(ternary),
            base_scale,
            delta_scale,
        )

    def forward(self) -> torch.Tensor:
        base = self.base_codes.view(self.shape).float()
        ternary = unpack_ternary(self.ternary_codes, base.numel()).view(self.shape).float()
        return (
            base * self.base_scale.float().view(-1, 1)
            + ternary * self.delta_scale.float().view(-1, 1)
        )

    def payload_bytes(self) -> int:
        return (
            self.base_codes.numel()
            + self.ternary_codes.numel()
            + 2 * self.base_scale.numel()
            + 2 * self.delta_scale.numel()
        )

    def audit(self) -> dict[str, Any]:
        ternary = unpack_ternary(self.ternary_codes, self.base_codes.numel())
        return {
            "base_code_min": int(self.base_codes.min()),
            "base_code_max": int(self.base_codes.max()),
            "delta_code_values": sorted(int(value) for value in ternary.unique().tolist()),
            "base_scale_min": float(self.base_scale.float().min()),
            "delta_scale_min": float(self.delta_scale.float().min()),
            "all_finite": bool(
                torch.isfinite(self.base_scale.float()).all()
                and torch.isfinite(self.delta_scale.float()).all()
            ),
            "output_rows": self.shape[0],
            "coordinates": self.base_codes.numel(),
        }


class PackedInt4Weight(nn.Module):
    def __init__(
        self,
        shape: tuple[int, int],
        codes: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        super().__init__()
        self.shape = shape
        self.register_buffer("codes", codes.to(torch.uint8).contiguous())
        self.register_buffer("scale", scale.to(torch.bfloat16).reshape(-1).contiguous())

    @classmethod
    @torch.no_grad()
    def from_training(cls, source: QuantizedWeight) -> "PackedInt4Weight":
        if source.maximum_code != 7:
            raise ValueError("packed branch requires signed int4 with maximum code 7")
        scale = source.log_scale.exp().to(torch.bfloat16)
        codes = torch.round(source.shadow.detach().float() / scale.float()).clamp(-7, 7).to(torch.int8)
        return cls(tuple(codes.shape), pack_int4(codes), scale)

    def forward(self) -> torch.Tensor:
        codes = unpack_int4(self.codes, self.shape[0] * self.shape[1]).view(self.shape).float()
        return codes * self.scale.float().view(-1, 1)

    def payload_bytes(self) -> int:
        return self.codes.numel() + 2 * self.scale.numel()

    def audit(self) -> dict[str, Any]:
        codes = unpack_int4(self.codes, self.shape[0] * self.shape[1])
        return {
            "code_min": int(codes.min()),
            "code_max": int(codes.max()),
            "scale_min": float(self.scale.float().min()),
            "scale_max": float(self.scale.float().max()),
            "all_finite": bool(torch.isfinite(self.scale.float()).all()),
            "output_rows": self.shape[0],
            "coordinates": codes.numel(),
        }


@torch.no_grad()
def freeze_phase_elastic_modules(modules) -> list[dict[str, Any]]:
    audits = []
    for module in modules:
        for name in ("base_gate", "base_up", "base_down"):
            setattr(module, name, PackedResidualWeight.from_training(getattr(module, name)))
        for name in ("branch_gate", "branch_up", "branch_down"):
            setattr(module, name, PackedInt4Weight.from_training(getattr(module, name)))
        base_bytes = sum(getattr(module, name).payload_bytes() for name in ("base_gate", "base_up", "base_down"))
        branch_bytes = sum(getattr(module, name).payload_bytes() for name in ("branch_gate", "branch_up", "branch_down"))
        audits.append(
            {
                "base_payload_bytes": base_bytes,
                "branch_payload_bytes": branch_bytes,
                "total_payload_bytes": base_bytes + branch_bytes,
            }
        )
    return audits


def shadow_names(model: nn.Module) -> list[str]:
    return [
        name
        for name, _ in list(model.named_parameters()) + list(model.named_buffers())
        if "shadow" in name or "log_scale" in name
    ]
