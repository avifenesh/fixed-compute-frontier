#!/usr/bin/env python3
"""Stage-0 algebra and ledger for projection-coalesced attention/FFN."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Ledger:
    hidden_size: int
    baseline_width: int
    query_heads: int
    kv_heads: int
    head_dim: int
    kv_dim: int
    candidate_width: int
    baseline_input_rows: int
    candidate_input_rows: int
    baseline_output_columns: int
    candidate_output_columns: int
    baseline_dense_parameters: int
    candidate_dense_parameters: int
    candidate_to_baseline: float
    independent_swiglu_floor: int


def resource_ledger(
    hidden_size: int,
    baseline_width: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
) -> Ledger:
    if query_heads * head_dim != hidden_size:
        raise ValueError("query heads must span hidden_size")
    if query_heads % kv_heads:
        raise ValueError("query_heads must be divisible by kv_heads")
    kv_dim = kv_heads * head_dim
    numerator = hidden_size + 2 * kv_dim
    if numerator % 2:
        raise ValueError("input-shape-matched candidate width is not integral")
    candidate_width = baseline_width + numerator // 2
    baseline_input_rows = 2 * baseline_width + hidden_size + 2 * kv_dim
    candidate_input_rows = 2 * candidate_width
    baseline_output_columns = baseline_width + hidden_size
    baseline_dense_parameters = (
        3 * hidden_size * baseline_width
        + 2 * hidden_size * hidden_size
        + 2 * hidden_size * kv_dim
    )
    candidate_dense_parameters = 3 * hidden_size * candidate_width
    return Ledger(
        hidden_size=hidden_size,
        baseline_width=baseline_width,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        kv_dim=kv_dim,
        candidate_width=candidate_width,
        baseline_input_rows=baseline_input_rows,
        candidate_input_rows=candidate_input_rows,
        baseline_output_columns=baseline_output_columns,
        candidate_output_columns=candidate_width,
        baseline_dense_parameters=baseline_dense_parameters,
        candidate_dense_parameters=candidate_dense_parameters,
        candidate_to_baseline=candidate_dense_parameters / baseline_dense_parameters,
        independent_swiglu_floor=candidate_width - hidden_size - 2 * kv_dim,
    )


def repeat_kv(values: torch.Tensor, repeats: int) -> torch.Tensor:
    """Repeat `[B,Hkv,T,d]` into query-head order without learned work."""
    if repeats == 1:
        return values
    batch, kv_heads, tokens, head_dim = values.shape
    return (
        values[:, :, None, :, :]
        .expand(batch, kv_heads, repeats, tokens, head_dim)
        .reshape(batch, kv_heads * repeats, tokens, head_dim)
    )


def causal_gqa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Reference causal grouped-query attention returning `[B,T,D]`."""
    batch, tokens, hidden_size = query.shape
    if hidden_size != query_heads * head_dim:
        raise ValueError("invalid query width")
    kv_dim = kv_heads * head_dim
    if key.shape != (batch, tokens, kv_dim) or value.shape != key.shape:
        raise ValueError("invalid key/value shape")
    q = query.view(batch, tokens, query_heads, head_dim).transpose(1, 2)
    k = key.view(batch, tokens, kv_heads, head_dim).transpose(1, 2)
    v = value.view(batch, tokens, kv_heads, head_dim).transpose(1, 2)
    repeats = query_heads // kv_heads
    k = repeat_kv(k, repeats)
    v = repeat_kv(v, repeats)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    mask = torch.ones(tokens, tokens, dtype=torch.bool, device=query.device).tril()
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    probabilities = torch.softmax(scores.float(), dim=-1).to(query.dtype)
    output = torch.matmul(probabilities, v)
    return output.transpose(1, 2).contiguous().view(batch, tokens, hidden_size)


def coalesced_forward(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    *,
    zero_attention_local_products: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Frozen Stage-0 algebra, excluding normalization, RoPE, and residual."""
    hidden_size = hidden_states.shape[-1]
    kv_dim = kv_heads * head_dim
    width = gate_weight.shape[0]
    if up_weight.shape != gate_weight.shape:
        raise ValueError("gate/up shapes differ")
    if down_weight.shape != (hidden_size, width):
        raise ValueError("invalid down shape")
    if width < hidden_size + 2 * kv_dim:
        raise ValueError("width too small for frozen slices")
    gate = F.linear(hidden_states, gate_weight)
    up = F.linear(hidden_states, up_weight)
    query = gate[..., :hidden_size]
    key = up[..., hidden_size : hidden_size + kv_dim]
    value = up[..., hidden_size + kv_dim : hidden_size + 2 * kv_dim]
    attention = causal_gqa(query, key, value, query_heads, kv_heads, head_dim)
    local = F.silu(gate) * up
    if zero_attention_local_products:
        local = torch.cat(
            (torch.zeros_like(local[..., : hidden_size + 2 * kv_dim]),
             local[..., hidden_size + 2 * kv_dim :]),
            dim=-1,
        )
    joint = local.clone()
    joint[..., :hidden_size] = joint[..., :hidden_size] + attention
    output = F.linear(joint, down_weight)
    return output, {
        "gate": gate,
        "up": up,
        "query": query,
        "key": key,
        "value": value,
        "attention": attention,
        "local": local,
        "joint": joint,
    }


def dedicated_endpoint(
    hidden_size: int,
    independent_width: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    *,
    dtype: torch.dtype = torch.float64,
) -> dict[str, torch.Tensor]:
    """Construct candidate/reference weights for the exact containment floor."""
    kv_dim = kv_heads * head_dim
    width = hidden_size + 2 * kv_dim + independent_width
    generator = torch.Generator().manual_seed(20260727)
    random = lambda *shape: torch.randn(*shape, generator=generator, dtype=dtype) / math.sqrt(hidden_size)
    q_weight = random(hidden_size, hidden_size)
    k_weight = random(kv_dim, hidden_size)
    v_weight = random(kv_dim, hidden_size)
    o_weight = random(hidden_size, hidden_size)
    local_gate = random(independent_width, hidden_size)
    local_up = random(independent_width, hidden_size)
    local_down = random(hidden_size, independent_width)

    gate_weight = torch.zeros(width, hidden_size, dtype=dtype)
    up_weight = torch.zeros_like(gate_weight)
    down_weight = torch.zeros(hidden_size, width, dtype=dtype)
    gate_weight[:hidden_size] = q_weight
    up_weight[hidden_size : hidden_size + kv_dim] = k_weight
    up_weight[hidden_size + kv_dim : hidden_size + 2 * kv_dim] = v_weight
    local_start = hidden_size + 2 * kv_dim
    gate_weight[local_start:] = local_gate
    up_weight[local_start:] = local_up
    down_weight[:, :hidden_size] = o_weight
    down_weight[:, local_start:] = local_down
    return locals()


def run_stage0() -> dict[str, object]:
    ledger = resource_ledger(384, 1024, 6, 2, 64)
    endpoint = dedicated_endpoint(384, ledger.independent_swiglu_floor, 6, 2, 64)
    hidden = torch.randn(2, 7, 384, generator=torch.Generator().manual_seed(7), dtype=torch.float64)
    actual, pieces = coalesced_forward(
        hidden,
        endpoint["gate_weight"], endpoint["up_weight"], endpoint["down_weight"],
        6, 2, 64,
    )
    reference_attention = F.linear(pieces["attention"], endpoint["o_weight"])
    reference_local = F.linear(
        F.silu(F.linear(hidden, endpoint["local_gate"]))
        * F.linear(hidden, endpoint["local_up"]),
        endpoint["local_down"],
    )
    reference = reference_attention + reference_local
    endpoint_error = float((actual - reference).abs().max())

    separate = F.linear(pieces["local"], endpoint["down_weight"])
    separate = separate + F.linear(
        pieces["attention"], endpoint["down_weight"][:, :384]
    )
    fused_down_error = float((actual - separate).abs().max())
    payload = {
        "schema": "coalesced-attention-ffn-stage0-v1",
        "ledger": asdict(ledger),
        "checks": {
            "input_rows_exactly_matched": ledger.baseline_input_rows == ledger.candidate_input_rows,
            "candidate_dense_projection_nonincrease": ledger.candidate_dense_parameters <= ledger.baseline_dense_parameters,
            "independent_floor_positive": ledger.independent_swiglu_floor > 0,
            "dedicated_endpoint_max_abs_error": endpoint_error,
            "fused_down_max_abs_error": fused_down_error,
            "dedicated_endpoint_pass": endpoint_error < 1e-10,
            "fused_down_pass": fused_down_error < 1e-10,
        },
    }
    payload["stage0_pass"] = all(
        value for key, value in payload["checks"].items() if key.endswith("pass") or key in {
            "input_rows_exactly_matched",
            "candidate_dense_projection_nonincrease",
            "independent_floor_positive",
        }
    )
    return payload


if __name__ == "__main__":
    result = run_stage0()
    output = Path("results/coalesced-attention-ffn-stage0.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
