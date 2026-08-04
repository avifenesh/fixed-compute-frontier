#!/usr/bin/env python3
"""Algebraic admission checks for the 2DM cycle-factor FFN."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F


D, M, NARROW, SAMPLES, SEED = 3, 12, 8, 32, 20260727


def cycle_indices(width: int, group: int = 4) -> torch.Tensor:
    if width % group:
        raise ValueError("width must divide fixed cycle length")
    index = torch.arange(width).reshape(-1, group)
    return index.roll(-1, dims=-1).flatten()


def parameter_ledger(d: int = D, m: int = M, narrow: int = NARROW) -> dict[str, int]:
    return {
        "cycle": 2 * d * m,
        "self_product": 2 * d * m,
        "narrow_swiglu": 3 * d * narrow,
        "full_swiglu": 3 * d * m,
    }


def sampled_rank(kind: str) -> dict[str, object]:
    generator = torch.Generator().manual_seed(SEED)
    inputs = torch.randn(SAMPLES, D, generator=generator, dtype=torch.float64)
    count = 2 * D * M
    theta = torch.randn(count, generator=generator, dtype=torch.float64)
    route = cycle_indices(M)

    def function(parameters: torch.Tensor) -> torch.Tensor:
        if kind in {"cycle", "self_product"}:
            w = parameters[: D * M].view(M, D)
            v = parameters[D * M :].view(D, M)
            hidden = F.linear(inputs, w)
            value = hidden[..., route] if kind == "cycle" else hidden
            return F.linear(F.silu(hidden) * value, v).flatten()
        gate = parameters[: D * NARROW].view(NARROW, D)
        up = parameters[D * NARROW : 2 * D * NARROW].view(NARROW, D)
        down = parameters[2 * D * NARROW :].view(D, NARROW)
        return F.linear(F.silu(F.linear(inputs, gate)) * F.linear(inputs, up), down).flatten()

    jacobian = torch.autograd.functional.jacobian(function, theta, vectorize=True)
    singular = torch.linalg.svdvals(jacobian)
    tolerance = float(singular.max() * max(jacobian.shape) * torch.finfo(singular.dtype).eps)
    rank = int((singular > tolerance).sum())
    return {"parameters": count, "outputs": int(jacobian.shape[0]), "rank": rank,
            "nullity": count - rank, "tolerance": tolerance,
            "smallest_counted": float(singular[rank - 1])}


def quadratic_spectra() -> dict[str, object]:
    generator = torch.Generator().manual_seed(SEED + 1)
    rows = torch.randn(M, D, generator=generator, dtype=torch.float64)
    route = cycle_indices(M)
    cycle_eigenvalues = []
    self_eigenvalues = []
    for index in range(M):
        left, right = rows[index], rows[route[index]]
        cycle_form = 0.25 * (torch.outer(left, right) + torch.outer(right, left))
        self_form = 0.5 * torch.outer(left, left)
        cycle_eigenvalues.append(torch.linalg.eigvalsh(cycle_form))
        self_eigenvalues.append(torch.linalg.eigvalsh(self_form))
    cycle_values = torch.stack(cycle_eigenvalues)
    self_values = torch.stack(self_eigenvalues)
    return {
        "cycle_all_indefinite": bool(((cycle_values[:, 0] < -1e-10) & (cycle_values[:, -1] > 1e-10)).all()),
        "self_all_psd": bool((self_values[:, 0] >= -1e-10).all()),
        "cycle_min_eigenvalue": float(cycle_values.min()),
        "cycle_max_eigenvalue": float(cycle_values.max()),
        "self_min_eigenvalue": float(self_values.min()),
    }


def run() -> dict[str, object]:
    ledger = parameter_ledger()
    route = cycle_indices(M)
    indegree = torch.bincount(route, minlength=M)
    ranks = {kind: sampled_rank(kind) for kind in ("narrow_swiglu", "self_product", "cycle")}
    spectra = quadratic_spectra()
    checks = {
        "equal_2dm_parameters": ledger["cycle"] == ledger["self_product"] == ledger["narrow_swiglu"],
        "candidate_is_two_thirds_full_swiglu": 3 * ledger["cycle"] == 2 * ledger["full_swiglu"],
        "fixed_point_free_permutation": int(torch.unique(route).numel()) == M and bool((route != torch.arange(M)).all()),
        "one_in_one_out": int(indegree.min()) == int(indegree.max()) == 1,
        "cycle_quadratics_indefinite": spectra["cycle_all_indefinite"],
        "self_quadratics_psd": spectra["self_all_psd"],
        "cycle_full_sampled_rank": ranks["cycle"]["nullity"] == 0,
        "cycle_rank_exceeds_narrow_swiglu": ranks["cycle"]["rank"] > ranks["narrow_swiglu"]["rank"],
    }
    return {"schema": "cycle-factor-ffn-stage0-v1", "ledger": ledger,
            "route": route.tolist(), "ranks": ranks, "quadratic_spectra": spectra,
            "checks": checks, "pass": all(checks.values())}


if __name__ == "__main__":
    payload = run()
    Path("results/cycle-factor-ffn-stage0.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
