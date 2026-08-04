import pytest
import torch

from experiments.gauge_slot_attention import (
    GaugeSlotAttention,
    coefficients_from_packed,
    decode_packed_key,
)
from experiments.triangular_microdepth_lm_screen import scratch_config
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
)


def make_source(seed: int = 7, device: str = "cpu", dtype=torch.float32):
    torch.manual_seed(seed)
    return LlamaAttention(scratch_config(), layer_idx=0).to(device, dtype=dtype)


def position_embeddings(
    batch: int,
    tokens: int,
    *,
    device: str = "cpu",
    dtype=torch.float32,
    offset: int = 0,
):
    generator = torch.Generator().manual_seed(101 + offset)
    half = torch.randn(batch, tokens, 32, generator=generator).to(device, dtype=dtype)
    return (
        torch.cat((torch.cos(half), torch.cos(half)), dim=-1),
        torch.cat((torch.sin(half), torch.sin(half)), dim=-1),
    )


def test_canonical_chart_and_packed_export_have_exact_source_scalar_count():
    source = make_source()
    candidate = GaugeSlotAttention(source, "gauge_slot_g2")
    source_count = sum(parameter.numel() for parameter in source.parameters())
    assert candidate.chart_parameter_count() == source_count
    assert candidate.packed_serialized_values().numel() == source_count


def test_canonicalization_preserves_raw_rope_scores_and_implicit_pivots():
    source = make_source(11)
    control = GaugeSlotAttention(source, "canonical_bilinear_control")
    hidden = torch.randn(3, 5, source.q_proj.in_features)
    source_query = source.q_proj(hidden).view(3, 5, 6, 64).transpose(1, 2)
    source_key = source.k_proj(hidden).view(3, 5, 2, 64).transpose(1, 2)
    chart_query = torch.nn.functional.linear(
        hidden, control.query_weight
    ).view(3, 5, 6, 64).transpose(1, 2)
    chart_key = torch.nn.functional.linear(
        hidden, control.materialize_key_weight()
    ).view(3, 5, 2, 64).transpose(1, 2)
    source_key = source_key.repeat_interleave(3, dim=1)
    chart_key = chart_key.repeat_interleave(3, dim=1)
    embeddings = position_embeddings(3, 5)
    source_query, source_key = apply_rotary_pos_emb(
        source_query, source_key, *embeddings
    )
    chart_query, chart_key = apply_rotary_pos_emb(
        chart_query, chart_key, *embeddings
    )
    torch.testing.assert_close(
        torch.sum(chart_query * chart_key, dim=-1),
        torch.sum(source_query * source_key, dim=-1),
        atol=3e-5,
        rtol=3e-5,
    )

    physical = control.materialize_key_weight().reshape(2, 64, -1)
    for kv_head in range(control.kv_heads):
        for pair in range(control.pairs):
            column = int(control.pivot_columns[kv_head, pair])
            assert physical[kv_head, pair, column] == 0.0
            torch.testing.assert_close(
                physical[kv_head, pair + control.pairs, column],
                physical.new_tensor(control.tau),
            )


def test_packed_export_is_self_contained_and_decodes_both_payloads():
    candidate = GaugeSlotAttention(make_source(23), "gauge_slot_g2")
    with torch.no_grad():
        values = torch.linspace(-0.7, 0.8, candidate.pivot_indices.numel())
        candidate.key_packed.reshape(-1)[candidate.pivot_indices] = values
    artifact = candidate.export_packed()
    decoded_key = decode_packed_key(
        artifact["packed_key_weight"], artifact["pivot_indices"], artifact["tau"]
    )
    decoded_coefficients = coefficients_from_packed(
        artifact["packed_key_weight"],
        artifact["pivot_indices"],
        artifact["kv_heads"],
        artifact["pairs"],
    )
    torch.testing.assert_close(decoded_key, candidate.materialize_key_weight())
    torch.testing.assert_close(
        decoded_coefficients, candidate.curvature_coefficients()
    )


def test_both_storage_funded_coefficients_receive_finite_gradients():
    candidate = GaugeSlotAttention(make_source(29), "gauge_slot_g2")
    hidden = torch.randn(2, 7, candidate.hidden_size)
    key = torch.nn.functional.linear(hidden, candidate.materialize_key_weight())
    key = key.view(2, 7, candidate.kv_heads, candidate.head_dim)
    even = key[..., :candidate.pairs]
    odd = key[..., candidate.pairs:]
    coefficients = candidate.curvature_coefficients()[None, None]
    transformed = torch.cat((
        even + coefficients[..., 0] * odd.square(),
        odd + coefficients[..., 1] * even.square(),
    ), dim=-1)
    transformed.square().mean().backward()
    assert candidate.key_packed.grad is not None
    coefficient_gradients = candidate.key_packed.grad.reshape(-1)[
        candidate.pivot_indices
    ].reshape(candidate.kv_heads, candidate.pairs, 2)
    assert torch.isfinite(coefficient_gradients).all()
    assert torch.count_nonzero(coefficient_gradients[..., 0]) > 0
    assert torch.count_nonzero(coefficient_gradients[..., 1]) > 0


@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
def test_real_llama_prefill_candidate_starts_exactly_on_canonical_control(
    implementation,
):
    config = scratch_config()
    config._attn_implementation = implementation
    torch.manual_seed(103)
    control = GaugeSlotAttention(
        LlamaAttention(config, layer_idx=0), "canonical_bilinear_control"
    )
    torch.manual_seed(103)
    candidate = GaugeSlotAttention(
        LlamaAttention(config, layer_idx=0), "gauge_slot_g2"
    )
    hidden = torch.randn(2, 9, config.hidden_size)
    embeddings = position_embeddings(2, 9)
    control_output = control(
        hidden, position_embeddings=embeddings, attention_mask=None
    )[0]
    candidate_output = candidate(
        hidden, position_embeddings=embeddings, attention_mask=None
    )[0]
    torch.testing.assert_close(candidate_output, control_output, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA BF16 cache test")
def test_bf16_cached_candidate_matches_control_in_dtype_shape_and_values():
    config = scratch_config()
    config._attn_implementation = "eager"
    torch.manual_seed(107)
    control = GaugeSlotAttention(
        LlamaAttention(config, layer_idx=0).to("cuda", dtype=torch.bfloat16),
        "canonical_bilinear_control",
    ).to("cuda")
    torch.manual_seed(107)
    candidate = GaugeSlotAttention(
        LlamaAttention(config, layer_idx=0).to("cuda", dtype=torch.bfloat16),
        "gauge_slot_g2",
    ).to("cuda")
    control_cache = DynamicCache()
    candidate_cache = DynamicCache()
    hidden = torch.randn(1, 5, config.hidden_size, device="cuda", dtype=torch.bfloat16)
    embeddings = position_embeddings(
        1, 5, device="cuda", dtype=torch.bfloat16
    )
    control_output = control(
        hidden,
        position_embeddings=embeddings,
        attention_mask=None,
        past_key_values=control_cache,
        cache_position=torch.arange(5, device="cuda"),
    )[0]
    candidate_output = candidate(
        hidden,
        position_embeddings=embeddings,
        attention_mask=None,
        past_key_values=candidate_cache,
        cache_position=torch.arange(5, device="cuda"),
    )[0]
    control_key = control_cache.layers[0].keys
    candidate_key = candidate_cache.layers[0].keys
    assert control_key.dtype == candidate_key.dtype == torch.bfloat16
    assert control_key.shape == candidate_key.shape
    assert control_key.numel() * control_key.element_size() == (
        candidate_key.numel() * candidate_key.element_size()
    )
    torch.testing.assert_close(candidate_key, control_key, atol=0.0, rtol=0.0)
    torch.testing.assert_close(candidate_output, control_output, atol=0.0, rtol=0.0)
