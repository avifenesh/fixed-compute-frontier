#!/usr/bin/env python3
"""T18-Flash2: two-head 192d fused soft-record operator."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import budget_neutral_soft_record_t18 as t18
from experiments import raw_prose_equality_plane_t10 as t10


HEADS = 2
HEAD_DIM = t18.HIDDEN // HEADS


class Flash2RecordMemory(t18.DenseRecordMemory):
    def forward(self, hidden: Tensor) -> Tensor:
        batch, tokens, width = hidden.shape
        if width != t18.HIDDEN:
            raise RuntimeError(f"expected hidden width {t18.HIDDEN}, found {width}")
        query = hidden.view(batch, tokens, HEADS, HEAD_DIM).transpose(1, 2)
        key = self.keys.view(self.slots, HEADS, HEAD_DIM).permute(1, 0, 2)
        value = self.values.view(self.slots, HEADS, HEAD_DIM).permute(1, 0, 2)
        key = key.unsqueeze(0).expand(batch, -1, -1, -1)
        value = value.unsqueeze(0).expand(batch, -1, -1, -1)
        if hidden.is_cuda:
            with torch.nn.attention.sdpa_kernel(
                torch.nn.attention.SDPBackend.FLASH_ATTENTION
            ):
                output = F.scaled_dot_product_attention(
                    query, key, value, dropout_p=0.0, is_causal=False
                )
        else:
            output = F.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=False
            )
        return output.transpose(1, 2).contiguous().view(batch, tokens, width)


class Flash2RecordBlock(nn.Module):
    def __init__(self, depth_scale: float) -> None:
        super().__init__()
        small = t10.core.small
        self.attention_norm = small.RMSNorm()
        self.attention = small.CausalAttention(depth_scale)
        self.ffn_norm = small.RMSNorm()
        self.memory = Flash2RecordMemory()

    def reset_parameters(self) -> None:
        self.attention_norm.weight.data.fill_(1.0)
        self.ffn_norm.weight.data.fill_(1.0)
        self.attention.reset_parameters()
        self.memory.reset_parameters()

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden))
        return hidden + self.memory(self.ffn_norm(hidden))


class Flash2SoftRecordInterpreterLM(t18.SoftRecordInterpreterLM):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        small = t10.core.small
        self.token = nn.Embedding(small.VOCAB, t18.HIDDEN)
        depth_scale = 1.0 / math.sqrt(2 * small.LAYERS)
        self.blocks = nn.ModuleList(
            [small.Block(depth_scale) for _ in range(small.LAYERS - 2)]
            + [Flash2RecordBlock(depth_scale), t18.SmallSwiGLUBlock(depth_scale)]
        )
        self.final_norm = small.RMSNorm()
        self.reset_parameters()

    @property
    def record_block(self) -> Flash2RecordBlock:
        block = self.blocks[-2]
        assert isinstance(block, Flash2RecordBlock)
        return block


def build_candidate(
    seed: int, device: torch.device
) -> Flash2SoftRecordInterpreterLM:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    return Flash2SoftRecordInterpreterLM().to(device)


def explicit_two_head(hidden: Tensor, keys: Tensor, values: Tensor) -> Tensor:
    batch, tokens, width = hidden.shape
    query = hidden.view(batch, tokens, HEADS, HEAD_DIM).transpose(1, 2)
    key = keys.view(-1, HEADS, HEAD_DIM).permute(1, 0, 2)
    value = values.view(-1, HEADS, HEAD_DIM).permute(1, 0, 2)
    scores = torch.einsum("bhtd,hsd->bhts", query, key) / math.sqrt(HEAD_DIM)
    probability = torch.exp(scores - scores.max(dim=-1, keepdim=True).values)
    probability = probability / probability.sum(dim=-1, keepdim=True)
    output = torch.einsum("bhts,hsd->bhtd", probability, value)
    return output.transpose(1, 2).contiguous().view(batch, tokens, width)


def bounded_reference(device: torch.device) -> dict[str, object]:
    torch.manual_seed(10_117)
    hidden = torch.randn(2, 3, t18.HIDDEN, device=device, requires_grad=True)
    memory = Flash2RecordMemory(slots=7, hidden=t18.HIDDEN).to(device)
    output = memory(hidden)
    gradient = torch.randn_like(output)
    output.backward(gradient)
    actual = (
        output.detach().clone(),
        hidden.grad.detach().clone(),
        memory.keys.grad.detach().clone(),
        memory.values.grad.detach().clone(),
    )
    hidden_ref = hidden.detach().clone().requires_grad_(True)
    keys_ref = memory.keys.detach().clone().requires_grad_(True)
    values_ref = memory.values.detach().clone().requires_grad_(True)
    reference = explicit_two_head(hidden_ref, keys_ref, values_ref)
    reference.backward(gradient)
    errors = {
        "forward": float((actual[0] - reference.detach()).abs().max().item()),
        "input_gradient": float((actual[1] - hidden_ref.grad).abs().max().item()),
        "key_gradient": float((actual[2] - keys_ref.grad).abs().max().item()),
        "value_gradient": float((actual[3] - values_ref.grad).abs().max().item()),
    }

    null_memory = Flash2RecordMemory(slots=5, hidden=t18.HIDDEN).to(device)
    query = F.normalize(
        torch.randn(1, 1, t18.HIDDEN, device=device), dim=-1
    )
    with torch.no_grad():
        null_memory.keys[0].copy_(query[0, 0] * 200.0)
        null_memory.values[0].zero_()
        null_memory.keys[1:].copy_(-query[0, 0] * 200.0)
        null_memory.values[1:].normal_()
        null_output = null_memory(query)
    return {
        "maximum_errors": errors,
        "null_output_max_abs": float(null_output.abs().max().item()),
    }
