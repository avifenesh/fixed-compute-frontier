import pytest
import torch
import torch.nn.functional as F

from experiments.triangular_microdepth_lm_screen import scratch_config
from experiments.triangular_value_flow_attention import (
    TriangularValueFlowAttention,
    rq_decomposition,
)
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import LlamaAttention


def make_source(seed=7, device="cpu", dtype=torch.float32):
    torch.manual_seed(seed)
    return LlamaAttention(scratch_config(), layer_idx=0).to(device, dtype=dtype)


def embeddings(batch, tokens, device="cpu", dtype=torch.float32):
    angles = torch.randn(batch, tokens, 32, device=device, dtype=torch.float32)
    cosine = angles.cos().to(dtype)
    sine = angles.sin().to(dtype)
    return torch.cat((cosine, cosine), -1), torch.cat((sine, sine), -1)


def strict_lower_values(weight, module):
    output = weight.reshape(module.hidden_size, module.query_heads, module.head_dim)
    representative = output[
        :module.head_dim, ::module.num_key_value_groups, :
    ].permute(1, 0, 2)
    values = []
    for start in range(0, module.head_dim, module.block_size):
        block = representative[
            :, start:start + module.block_size, start:start + module.block_size
        ]
        values.extend(block[:, row, :row] for row in range(1, module.block_size))
    return torch.cat(values, dim=-1)


def test_rq_orientation_sign_and_upper_triangle():
    torch.manual_seed(11)
    matrix = torch.randn(9, 9, dtype=torch.float64)
    upper, orthogonal = rq_decomposition(matrix)
    torch.testing.assert_close(upper @ orthogonal, matrix, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        orthogonal @ orthogonal.T,
        torch.eye(9, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.count_nonzero(torch.tril(upper, diagonal=-1)) == 0
    assert torch.all(torch.diagonal(upper) >= 0)


def test_gqa_value_output_canonicalization_preserves_arbitrary_attention_flow():
    source = make_source(13)
    control = TriangularValueFlowAttention(source, "canonical_value_control")
    batch, queries, tokens = 2, 5, 7
    hidden = torch.randn(batch, tokens, control.hidden_size)
    logits = torch.randn(batch, control.query_heads, queries, tokens)
    probabilities = logits.softmax(-1)

    def run(value_weight, output_weight):
        values = F.linear(hidden, value_weight).view(
            batch, tokens, control.kv_heads, control.head_dim
        )
        heads = []
        for query_head in range(control.query_heads):
            kv_head = query_head // control.num_key_value_groups
            heads.append(torch.einsum(
                "bqt,btd->bqd", probabilities[:, query_head], values[:, :, kv_head]
            ))
        return F.linear(torch.cat(heads, dim=-1), output_weight)

    raw = run(source.v_proj.weight, source.o_proj.weight)
    canonical = run(control.value_weight, control.output_weight)
    torch.testing.assert_close(canonical, raw, atol=3e-5, rtol=3e-5)


def test_exact_zero_blocks_direction_count_and_no_added_state():
    source = make_source(17)
    control = TriangularValueFlowAttention(source, "canonical_value_control")
    expected = control.kv_heads * control.blocks * (
        control.block_size * (control.block_size - 1) // 2
    )
    assert control.coefficient_values().numel() == expected
    assert torch.count_nonzero(control.coefficient_values()) == 0
    assert list(control.buffers()) == []
    assert sum(p.numel() for p in control.parameters()) == sum(
        p.numel() for p in source.parameters()
    )
    assert len(list(control.parameters())) == len(list(source.parameters()))


def test_candidate_starts_exactly_on_control_and_has_new_coefficient_gradient():
    control = TriangularValueFlowAttention(
        make_source(19), "canonical_value_control"
    )
    candidate = TriangularValueFlowAttention(
        make_source(19), "triangular_value_flow"
    )
    activation = torch.randn(2, 7, candidate.query_heads, candidate.head_dim)

    def loss(module):
        flowed = module.apply_value_flow(activation)
        output = F.linear(flowed.flatten(-2), module.output_weight)
        return output.square().mean()

    torch.testing.assert_close(
        candidate.apply_value_flow(activation),
        control.apply_value_flow(activation),
        atol=0.0,
        rtol=0.0,
    )
    loss(control).backward()
    loss(candidate).backward()
    control_gradient = strict_lower_values(control.output_weight.grad, control)
    candidate_gradient = strict_lower_values(candidate.output_weight.grad, candidate)
    assert torch.isfinite(candidate_gradient).all()
    assert torch.count_nonzero(candidate_gradient - control_gradient) > 0


@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
def test_real_llama_candidate_initial_output_matches_control(implementation):
    config = scratch_config()
    config._attn_implementation = implementation
    torch.manual_seed(23)
    control = TriangularValueFlowAttention(
        LlamaAttention(config, layer_idx=0), "canonical_value_control"
    )
    torch.manual_seed(23)
    candidate = TriangularValueFlowAttention(
        LlamaAttention(config, layer_idx=0), "triangular_value_flow"
    )
    hidden = torch.randn(2, 9, config.hidden_size)
    position = embeddings(2, 9)
    control_output = control(
        hidden, position_embeddings=position, attention_mask=None
    )[0]
    candidate_output = candidate(
        hidden, position_embeddings=position, attention_mask=None
    )[0]
    torch.testing.assert_close(candidate_output, control_output, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA BF16 cache test")
def test_bf16_cache_and_output_match_control_at_initialization():
    config = scratch_config()
    config._attn_implementation = "eager"
    torch.manual_seed(29)
    control = TriangularValueFlowAttention(
        LlamaAttention(config, layer_idx=0).to("cuda", dtype=torch.bfloat16),
        "canonical_value_control",
    ).cuda()
    torch.manual_seed(29)
    candidate = TriangularValueFlowAttention(
        LlamaAttention(config, layer_idx=0).to("cuda", dtype=torch.bfloat16),
        "triangular_value_flow",
    ).cuda()
    hidden = torch.randn(1, 5, config.hidden_size, device="cuda", dtype=torch.bfloat16)
    position = embeddings(1, 5, device="cuda", dtype=torch.bfloat16)
    caches = [DynamicCache(), DynamicCache()]
    outputs = []
    for module, cache in zip((control, candidate), caches):
        outputs.append(module(
            hidden,
            position_embeddings=position,
            attention_mask=None,
            past_key_values=cache,
            cache_position=torch.arange(5, device="cuda"),
        )[0])
    for field in ("keys", "values"):
        left = getattr(caches[0].layers[0], field)
        right = getattr(caches[1].layers[0], field)
        assert left.dtype == right.dtype == torch.bfloat16
        assert left.shape == right.shape
        assert left.numel() * left.element_size() == right.numel() * right.element_size()
        torch.testing.assert_close(right, left, atol=0.0, rtol=0.0)
    torch.testing.assert_close(outputs[1], outputs[0], atol=0.0, rtol=0.0)
