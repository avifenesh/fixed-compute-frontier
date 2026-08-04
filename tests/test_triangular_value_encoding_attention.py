import pytest
import torch
import torch.nn.functional as F

from experiments.triangular_microdepth_lm_screen import scratch_config
from experiments.triangular_value_encoding_attention import (
    TriangularValueEncodingAttention,
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


def test_gqa_canonicalization_preserves_arbitrary_attention_output():
    source = make_source(41)
    control = TriangularValueEncodingAttention(source, "canonical_value_control")
    hidden = torch.randn(2, 7, control.hidden_size)
    probabilities = torch.randn(2, control.query_heads, 5, 7).softmax(-1)

    def run(value_weight, output_weight):
        values = F.linear(hidden, value_weight).view(
            2, 7, control.kv_heads, control.head_dim
        )
        heads = []
        for query_head in range(control.query_heads):
            kv_head = query_head // control.num_key_value_groups
            heads.append(torch.einsum(
                "bqt,btd->bqd", probabilities[:, query_head], values[:, :, kv_head]
            ))
        return F.linear(torch.cat(heads, -1), output_weight)

    torch.testing.assert_close(
        run(control.value_weight, control.output_weight),
        run(source.v_proj.weight, source.o_proj.weight),
        atol=3e-5,
        rtol=3e-5,
    )


def test_exact_slots_counts_and_no_added_state():
    source = make_source(43)
    control = TriangularValueEncodingAttention(source, "canonical_value_control")
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


def test_packed_raw_control_preserves_source_v_and_o_exactly():
    source = make_source(45)
    raw = TriangularValueEncodingAttention(source, "packed_raw_control")
    assert torch.equal(raw.value_weight, source.v_proj.weight)
    assert torch.equal(raw.output_weight, source.o_proj.weight)
    assert torch.count_nonzero(raw.coefficient_values()) > 0


def test_candidate_initial_encoding_exact_and_gradient_gains_nonlinear_path():
    control = TriangularValueEncodingAttention(
        make_source(47), "canonical_value_control"
    )
    candidate = TriangularValueEncodingAttention(
        make_source(47), "triangular_value_encoding"
    )
    values = torch.randn(2, candidate.kv_heads, 7, candidate.head_dim)
    torch.testing.assert_close(
        candidate.apply_value_encoding(values), values, atol=0.0, rtol=0.0
    )

    def loss(module):
        encoded = module.apply_value_encoding(values)
        repeated = encoded.repeat_interleave(module.num_key_value_groups, dim=1)
        output = F.linear(
            repeated.transpose(1, 2).flatten(-2), module.output_weight
        )
        return output.square().mean()

    loss(control).backward()
    loss(candidate).backward()
    control_gradient = strict_lower_values(control.output_weight.grad, control)
    candidate_gradient = strict_lower_values(candidate.output_weight.grad, candidate)
    assert torch.isfinite(candidate_gradient).all()
    assert torch.count_nonzero(candidate_gradient - control_gradient) > 0


def test_nonzero_slot_has_independent_closed_form_effect():
    candidate = TriangularValueEncodingAttention(
        make_source(49), "triangular_value_encoding"
    )
    coefficient = 0.5
    with torch.no_grad():
        candidate.output_weight[1, 0] = candidate.tau * coefficient
    values = torch.zeros(1, candidate.kv_heads, 1, candidate.head_dim)
    values[:, 0, 0, 0] = -2.0
    encoded = candidate.apply_value_encoding(values)
    expected = values.clone()
    expected[:, 0, 0, 1] += coefficient * (-2.0 * 2.0)
    torch.testing.assert_close(encoded, expected, atol=0.0, rtol=0.0)
    assert not torch.equal(encoded, values)


@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
def test_real_llama_candidate_initial_output_matches_control(implementation):
    config = scratch_config()
    config._attn_implementation = implementation
    torch.manual_seed(53)
    control = TriangularValueEncodingAttention(
        LlamaAttention(config, layer_idx=0), "canonical_value_control"
    )
    torch.manual_seed(53)
    candidate = TriangularValueEncodingAttention(
        LlamaAttention(config, layer_idx=0), "triangular_value_encoding"
    )
    hidden = torch.randn(2, 9, config.hidden_size)
    position = embeddings(2, 9)
    torch.testing.assert_close(
        candidate(hidden, position_embeddings=position, attention_mask=None)[0],
        control(hidden, position_embeddings=position, attention_mask=None)[0],
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA BF16 cache test")
def test_bf16_initial_cache_and_output_match_control():
    config = scratch_config()
    config._attn_implementation = "eager"
    torch.manual_seed(59)
    control = TriangularValueEncodingAttention(
        LlamaAttention(config, layer_idx=0).to("cuda", dtype=torch.bfloat16),
        "canonical_value_control",
    ).cuda()
    torch.manual_seed(59)
    candidate = TriangularValueEncodingAttention(
        LlamaAttention(config, layer_idx=0).to("cuda", dtype=torch.bfloat16),
        "triangular_value_encoding",
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
        assert left.shape == right.shape
        assert left.dtype == right.dtype == torch.bfloat16
        assert left.numel() * left.element_size() == right.numel() * right.element_size()
        torch.testing.assert_close(right, left, atol=0.0, rtol=0.0)
    torch.testing.assert_close(outputs[1], outputs[0], atol=0.0, rtol=0.0)
