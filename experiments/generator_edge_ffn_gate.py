#!/usr/bin/env python3
"""Stage-0 algebra, graph, and functional-rank gate for generator-edge FFN."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F


def edge_indices(width: int, *, duplicate: bool = False, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    if width <= 0 or width & (width - 1):
        raise ValueError("width must be a positive power of two")
    source = torch.arange(width, device=device)
    first = (source + 1) % width
    second = first if duplicate else (5 * source + 3) % width
    return first, second


def generator_edge_features(
    generated: torch.Tensor,
    *,
    duplicate: bool = False,
) -> torch.Tensor:
    first, second = edge_indices(generated.shape[-1], duplicate=duplicate, device=generated.device)
    activated = F.silu(generated)
    return torch.cat(
        (activated * generated[..., first], activated * generated[..., second]),
        dim=-1,
    )


def generator_edge_forward(
    hidden_states: torch.Tensor,
    generator_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    duplicate: bool = False,
) -> torch.Tensor:
    width, hidden_size = generator_weight.shape
    if hidden_states.shape[-1] != hidden_size:
        raise ValueError("hidden size mismatch")
    if down_weight.shape != (hidden_size, 2 * width):
        raise ValueError("down shape mismatch")
    generated = F.linear(hidden_states, generator_weight)
    return F.linear(generator_edge_features(generated, duplicate=duplicate), down_weight)


def parameter_ledger(hidden_size: int, width: int) -> dict[str, int]:
    if width % 2:
        raise ValueError("width must be even for self-product control")
    swiglu = 3 * hidden_size * width
    generator_edge = hidden_size * width + hidden_size * 2 * width
    self_product_width = 3 * width // 2
    self_product = 2 * hidden_size * self_product_width
    return {
        "hidden_size": hidden_size,
        "swiglu_width": width,
        "self_product_width": self_product_width,
        "swiglu_parameters": swiglu,
        "generator_edge_parameters": generator_edge,
        "duplicate_edge_parameters": generator_edge,
        "self_product_parameters": self_product,
    }


def _unpack_standard(theta: torch.Tensor, hidden_size: int, width: int):
    cursor = 0
    size = width * hidden_size
    gate = theta[cursor : cursor + size].view(width, hidden_size); cursor += size
    up = theta[cursor : cursor + size].view(width, hidden_size); cursor += size
    down = theta[cursor:].view(hidden_size, width)
    return gate, up, down


def _unpack_generator(theta: torch.Tensor, hidden_size: int, width: int):
    size = width * hidden_size
    generator = theta[:size].view(width, hidden_size)
    down = theta[size:].view(hidden_size, 2 * width)
    return generator, down


def _unpack_self_product(theta: torch.Tensor, hidden_size: int, width: int):
    control_width = 3 * width // 2
    size = control_width * hidden_size
    generator = theta[:size].view(control_width, hidden_size)
    down = theta[size:].view(hidden_size, control_width)
    return generator, down


def sampled_functional_rank(
    kind: str,
    hidden_size: int = 3,
    width: int = 4,
    samples: int = 16,
) -> dict[str, object]:
    ledger = parameter_ledger(hidden_size, width)
    parameter_count = ledger["swiglu_parameters"]
    generator = torch.Generator().manual_seed(20260727)
    theta = torch.randn(parameter_count, generator=generator, dtype=torch.float64)
    inputs = torch.randn(samples, hidden_size, generator=generator, dtype=torch.float64)

    def function(parameters: torch.Tensor) -> torch.Tensor:
        if kind == "swiglu":
            gate, up, down = _unpack_standard(parameters, hidden_size, width)
            output = F.linear(F.silu(F.linear(inputs, gate)) * F.linear(inputs, up), down)
        elif kind in {"generator_edge", "duplicate_edge"}:
            bank, down = _unpack_generator(parameters, hidden_size, width)
            output = generator_edge_forward(
                inputs, bank, down, duplicate=kind == "duplicate_edge"
            )
        elif kind == "self_product":
            bank, down = _unpack_self_product(parameters, hidden_size, width)
            generated = F.linear(inputs, bank)
            output = F.linear(F.silu(generated) * generated, down)
        else:
            raise ValueError(f"unknown kind: {kind}")
        return output.flatten()

    jacobian = torch.autograd.functional.jacobian(function, theta, vectorize=True)
    singular_values = torch.linalg.svdvals(jacobian)
    tolerance = float(singular_values.max() * max(jacobian.shape) * torch.finfo(jacobian.dtype).eps)
    rank = int((singular_values > tolerance).sum())
    return {
        "kind": kind,
        "parameters": parameter_count,
        "outputs": int(jacobian.shape[0]),
        "rank": rank,
        "nullity": parameter_count - rank,
        "tolerance": tolerance,
        "smallest_counted_singular_value": float(singular_values[rank - 1]) if rank else None,
        "largest_uncounted_singular_value": float(singular_values[rank]) if rank < len(singular_values) else None,
    }


def graph_checks(width: int) -> dict[str, object]:
    first, second = edge_indices(width)
    source = torch.arange(width)
    edges = torch.stack(
        (torch.cat((source, source)), torch.cat((first, second))), dim=1
    )
    unique_edges = torch.unique(edges, dim=0)
    out_degree = torch.bincount(edges[:, 0], minlength=width)
    in_degree = torch.bincount(edges[:, 1], minlength=width)
    return {
        "width": width,
        "edges": int(edges.shape[0]),
        "unique_edges": int(unique_edges.shape[0]),
        "out_degree_min": int(out_degree.min()),
        "out_degree_max": int(out_degree.max()),
        "in_degree_min": int(in_degree.min()),
        "in_degree_max": int(in_degree.max()),
        "first_route_is_permutation": int(torch.unique(first).numel()) == width,
        "second_route_is_permutation": int(torch.unique(second).numel()) == width,
    }


def run_stage0() -> dict[str, object]:
    ledger = parameter_ledger(384, 1024)
    graph = graph_checks(1024)
    ranks = {
        kind: sampled_functional_rank(kind)
        for kind in ("swiglu", "self_product", "duplicate_edge", "generator_edge")
    }
    checks = {
        "all_parameter_ledgers_equal": len({
            ledger["swiglu_parameters"],
            ledger["generator_edge_parameters"],
            ledger["duplicate_edge_parameters"],
            ledger["self_product_parameters"],
        }) == 1,
        "graph_has_2m_unique_edges": graph["edges"] == graph["unique_edges"] == 2048,
        "graph_is_two_regular": graph["out_degree_min"] == graph["out_degree_max"] == 2
        and graph["in_degree_min"] == graph["in_degree_max"] == 2,
        "candidate_rank_exceeds_swiglu": ranks["generator_edge"]["rank"] > ranks["swiglu"]["rank"],
    }
    return {
        "schema": "generator-edge-ffn-stage0-v1",
        "ledger": ledger,
        "graph": graph,
        "functional_ranks": ranks,
        "checks": checks,
        "stage0_cpu_pass": all(checks.values()),
    }


if __name__ == "__main__":
    result = run_stage0()
    output = Path("results/generator-edge-ffn-stage0.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
