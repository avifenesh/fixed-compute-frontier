import torch
import pytest

from experiments.gauge_g1_physical_attention import (
    PhysicalScaleG1Attention,
)
from experiments.triangular_microdepth_lm_screen import scratch_config
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
)


def make_source(seed: int = 7):
    torch.manual_seed(seed)
    return LlamaAttention(scratch_config(), layer_idx=0)


def test_chart_and_physical_serialization_have_exact_source_parameter_count():
    source = make_source()
    candidate = PhysicalScaleG1Attention(source, "physical_scale_g1")
    source_count = sum(parameter.numel() for parameter in source.parameters())
    assert candidate.chart_parameter_count() == source_count
    assert candidate.physical_serialized_values().numel() == source_count


def test_gauge_initialization_preserves_all_pairwise_raw_rope_scores():
    source = make_source(11)
    control = PhysicalScaleG1Attention(source, "polar_bilinear_control")
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
    angles = torch.randn(3, 5, 32)
    cosine = torch.cat((torch.cos(angles), torch.cos(angles)), dim=-1)
    sine = torch.cat((torch.sin(angles), torch.sin(angles)), dim=-1)
    source_query, source_key = apply_rotary_pos_emb(
        source_query, source_key, cosine, sine
    )
    chart_query, chart_key = apply_rotary_pos_emb(
        chart_query, chart_key, cosine, sine
    )
    source_scores = torch.sum(source_query * source_key, dim=-1)
    chart_scores = torch.sum(chart_query * chart_key, dim=-1)
    torch.testing.assert_close(chart_scores, source_scores, atol=2e-5, rtol=2e-5)


def test_candidate_starts_exactly_on_bilinear_control_slice():
    source_a = make_source(17)
    source_b = make_source(17)
    control = PhysicalScaleG1Attention(source_a, "polar_bilinear_control")
    candidate = PhysicalScaleG1Attention(source_b, "physical_scale_g1")
    torch.testing.assert_close(candidate.query_weight, control.query_weight)
    torch.testing.assert_close(
        candidate.materialize_key_weight(), control.materialize_key_weight()
    )
    torch.testing.assert_close(
        candidate.curvature_coefficients(),
        torch.zeros_like(candidate.curvature_coefficients()),
        atol=0.0,
        rtol=0.0,
    )


def test_coefficient_is_recoverable_from_existing_physical_pivot_weights():
    source = make_source(23)
    candidate = PhysicalScaleG1Attention(source, "physical_scale_g1")
    with torch.no_grad():
        candidate.polar[..., 1] = torch.linspace(-0.7, 0.8, candidate.polar[..., 1].numel()).reshape_as(candidate.polar[..., 1])
    physical = candidate.materialize_key_weight().reshape(2, 64, -1)
    recovered = torch.empty_like(candidate.curvature_coefficients())
    tau_squared = candidate.tau * candidate.tau
    for kv_head in range(candidate.kv_heads):
        for pair in range(candidate.pairs):
            column = int(candidate.pivot_columns[kv_head, pair])
            even = physical[kv_head, pair, column]
            odd = physical[kv_head, pair + candidate.pairs, column]
            radius_squared = even * even + odd * odd
            recovered[kv_head, pair] = (
                (radius_squared - tau_squared)
                / (radius_squared + tau_squared)
            )
    torch.testing.assert_close(
        recovered, candidate.curvature_coefficients(), atol=2e-6, rtol=2e-6
    )


def test_candidate_backward_is_finite_and_reaches_polar_scale():
    source = make_source(29)
    candidate = PhysicalScaleG1Attention(source, "physical_scale_g1")
    hidden = torch.randn(2, 7, source.q_proj.in_features)
    key = torch.nn.functional.linear(hidden, candidate.materialize_key_weight())
    key = key.view(2, 7, 2, 64)
    even = key[..., :candidate.pairs]
    odd = key[..., candidate.pairs:]
    coefficients = candidate.curvature_coefficients()[None, None]
    transformed = odd + coefficients * even.square()
    loss = transformed.square().mean()
    loss.backward()
    assert candidate.polar.grad is not None
    assert torch.isfinite(candidate.polar.grad).all()
    assert torch.count_nonzero(candidate.polar.grad[..., 1]) > 0


def position_embeddings(batch: int, tokens: int, offset: int = 0):
    generator = torch.Generator().manual_seed(101 + offset)
    half = torch.randn(batch, tokens, 32, generator=generator)
    return (
        torch.cat((torch.cos(half), torch.cos(half)), dim=-1),
        torch.cat((torch.sin(half), torch.sin(half)), dim=-1),
    )


@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
def test_real_llama_forward_matches_control_for_prefill(implementation):
    config = scratch_config()
    config._attn_implementation = implementation
    torch.manual_seed(103)
    source = LlamaAttention(config, layer_idx=0)
    control = PhysicalScaleG1Attention(source, "polar_bilinear_control")
    hidden = torch.randn(2, 9, config.hidden_size)
    embeddings = position_embeddings(2, 9)
    source_output = source(
        hidden,
        position_embeddings=embeddings,
        attention_mask=None,
    )[0]
    control_output = control(
        hidden,
        position_embeddings=embeddings,
        attention_mask=None,
    )[0]
    torch.testing.assert_close(control_output, source_output, atol=3e-5, rtol=3e-5)


def test_real_llama_cached_decode_matches_control():
    config = scratch_config()
    config._attn_implementation = "eager"
    torch.manual_seed(107)
    source = LlamaAttention(config, layer_idx=0)
    control = PhysicalScaleG1Attention(source, "polar_bilinear_control")
    source_cache = DynamicCache()
    control_cache = DynamicCache()
    prefill = torch.randn(1, 5, config.hidden_size)
    prefill_embeddings = position_embeddings(1, 5)
    source_prefill = source(
        prefill,
        position_embeddings=prefill_embeddings,
        attention_mask=None,
        past_key_values=source_cache,
        cache_position=torch.arange(5),
    )[0]
    control_prefill = control(
        prefill,
        position_embeddings=prefill_embeddings,
        attention_mask=None,
        past_key_values=control_cache,
        cache_position=torch.arange(5),
    )[0]
    torch.testing.assert_close(control_prefill, source_prefill, atol=3e-5, rtol=3e-5)
    decode = torch.randn(1, 1, config.hidden_size)
    decode_embeddings = position_embeddings(1, 1, offset=5)
    source_decode = source(
        decode,
        position_embeddings=decode_embeddings,
        attention_mask=None,
        past_key_values=source_cache,
        cache_position=torch.tensor([5]),
    )[0]
    control_decode = control(
        decode,
        position_embeddings=decode_embeddings,
        attention_mask=None,
        past_key_values=control_cache,
        cache_position=torch.tensor([5]),
    )[0]
    torch.testing.assert_close(control_decode, source_decode, atol=3e-5, rtol=3e-5)
