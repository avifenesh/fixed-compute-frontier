import pytest
import torch

from experiments.gauge_zero_g1_attention import GaugeZeroG1Attention
from experiments.triangular_microdepth_lm_screen import scratch_config
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


def make_source(seed: int = 7, device: str = "cpu", dtype=torch.float32):
    torch.manual_seed(seed)
    return LlamaAttention(scratch_config(), layer_idx=0).to(device, dtype=dtype)


def embeddings(batch, tokens, device="cpu", dtype=torch.float32):
    half = torch.randn(batch, tokens, 32, device=device, dtype=dtype)
    return (
        torch.cat((half.cos(), half.cos()), dim=-1),
        torch.cat((half.sin(), half.sin()), dim=-1),
    )


def test_dense_parameter_values_and_tensors_exactly_match_source():
    source = make_source()
    candidate = GaugeZeroG1Attention(source, "gauge_zero_g1")
    assert sum(p.numel() for p in candidate.parameters()) == sum(
        p.numel() for p in source.parameters()
    )
    assert len(list(candidate.parameters())) == len(list(source.parameters()))
    assert candidate.dense_serialized_values().numel() == sum(
        p.numel() for p in source.parameters()
    )


def test_full_gauge_preserves_raw_rope_scores_and_sets_exact_zero_coefficients():
    source = make_source(11)
    control = GaugeZeroG1Attention(source, "canonical_bilinear_control")
    hidden = torch.randn(3, 5, source.q_proj.in_features)
    source_q = source.q_proj(hidden).view(3, 5, 6, 64).transpose(1, 2)
    source_k = source.k_proj(hidden).view(3, 5, 2, 64).transpose(1, 2)
    chart_q = torch.nn.functional.linear(hidden, control.query_weight).view(
        3, 5, 6, 64
    ).transpose(1, 2)
    chart_k = torch.nn.functional.linear(hidden, control.key_weight).view(
        3, 5, 2, 64
    ).transpose(1, 2)
    source_k = source_k.repeat_interleave(3, dim=1)
    chart_k = chart_k.repeat_interleave(3, dim=1)
    position = embeddings(3, 5)
    source_q, source_k = apply_rotary_pos_emb(source_q, source_k, *position)
    chart_q, chart_k = apply_rotary_pos_emb(chart_q, chart_k, *position)
    torch.testing.assert_close(
        torch.einsum("bhid,bhjd->bhij", chart_q, chart_k),
        torch.einsum("bhid,bhjd->bhij", source_q, source_k),
        atol=3e-5,
        rtol=3e-5,
    )
    torch.testing.assert_close(
        control.curvature_coefficients(),
        torch.zeros_like(control.curvature_coefficients()),
        atol=0.0,
        rtol=0.0,
    )
    physical = control.key_weight.reshape(2, 64, -1)
    for kv_head in range(control.kv_heads):
        for pair in range(control.pairs):
            column = pair
            assert physical[kv_head, pair, column] >= 0.0
            assert physical[kv_head, pair + control.pairs, column] == 0.0
    assert list(control.buffers()) == []


def test_candidate_starts_exactly_on_control_and_pivot_gradient_contains_new_path():
    source_a = make_source(17)
    source_b = make_source(17)
    control = GaugeZeroG1Attention(source_a, "canonical_bilinear_control")
    candidate = GaugeZeroG1Attention(source_b, "gauge_zero_g1")
    hidden = torch.randn(2, 7, candidate.hidden_size)

    def loss(module):
        key = torch.nn.functional.linear(hidden, module.key_weight).view(
            2, 7, module.kv_heads, module.head_dim
        )
        even = key[..., :module.pairs]
        odd = key[..., module.pairs:]
        if module.arm == "gauge_zero_g1":
            coefficient = module.curvature_coefficients()[None, None]
            odd = odd + coefficient * even.square()
        return odd.square().mean()

    loss(control).backward()
    loss(candidate).backward()
    control_gradient = control.key_weight.grad.reshape(
        control.kv_heads, control.head_dim, control.hidden_size
    )[:, control.pairs:, :control.pairs].diagonal(dim1=-2, dim2=-1)
    candidate_gradient = candidate.key_weight.grad.reshape(
        candidate.kv_heads, candidate.head_dim, candidate.hidden_size
    )[:, candidate.pairs:, :candidate.pairs].diagonal(dim1=-2, dim2=-1)
    assert torch.isfinite(candidate_gradient).all()
    assert torch.count_nonzero(candidate_gradient - control_gradient) > 0


@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
def test_real_llama_candidate_initial_prefill_matches_control(implementation):
    config = scratch_config()
    config._attn_implementation = implementation
    torch.manual_seed(103)
    control = GaugeZeroG1Attention(
        LlamaAttention(config, layer_idx=0), "canonical_bilinear_control"
    )
    torch.manual_seed(103)
    candidate = GaugeZeroG1Attention(
        LlamaAttention(config, layer_idx=0), "gauge_zero_g1"
    )
    hidden = torch.randn(2, 9, config.hidden_size)
    position = embeddings(2, 9)
    control_output = control(hidden, position_embeddings=position, attention_mask=None)[0]
    candidate_output = candidate(hidden, position_embeddings=position, attention_mask=None)[0]
    torch.testing.assert_close(candidate_output, control_output, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA BF16 cache test")
def test_bf16_cache_dtype_bytes_and_initial_values_match_control():
    config = scratch_config()
    config._attn_implementation = "eager"
    torch.manual_seed(107)
    control = GaugeZeroG1Attention(
        LlamaAttention(config, layer_idx=0).to("cuda", dtype=torch.bfloat16),
        "canonical_bilinear_control",
    ).cuda()
    torch.manual_seed(107)
    candidate = GaugeZeroG1Attention(
        LlamaAttention(config, layer_idx=0).to("cuda", dtype=torch.bfloat16),
        "gauge_zero_g1",
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
    control_key = caches[0].layers[0].keys
    candidate_key = caches[1].layers[0].keys
    assert control_key.dtype == candidate_key.dtype == torch.bfloat16
    assert control_key.shape == candidate_key.shape
    assert control_key.numel() * control_key.element_size() == (
        candidate_key.numel() * candidate_key.element_size()
    )
    torch.testing.assert_close(candidate_key, control_key, atol=0.0, rtol=0.0)
    torch.testing.assert_close(outputs[1], outputs[0], atol=0.0, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA BF16 export test")
def test_one_step_bf16_dense_export_reproduces_coefficient_output_and_cache():
    config = scratch_config()
    config._attn_implementation = "eager"
    source = make_source(131, device="cuda")
    source.config._attn_implementation = "eager"
    candidate = GaugeZeroG1Attention(
        source, "gauge_zero_g1"
    ).cuda()
    hidden = torch.randn(2, 7, config.hidden_size, device="cuda")
    position = embeddings(2, 7, device="cuda", dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=3e-4, weight_decay=0.1)
    candidate.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        training_output = candidate(
            hidden, position_embeddings=position, attention_mask=None
        )[0]
        loss = training_output.float().square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    candidate.eval()

    training_coefficient = candidate.curvature_coefficients().to(torch.bfloat16)
    assert torch.count_nonzero(training_coefficient) > 0
    artifact = candidate.export_dense()
    query_weight = artifact["query_weight"].to(torch.bfloat16)
    key_weight = artifact["key_weight"].to(torch.bfloat16)
    value_weight = artifact["value_weight"].to(torch.bfloat16)
    output_weight = artifact["output_weight"].to(torch.bfloat16)
    coefficient_indices = torch.tensor([
        (kv_head * candidate.head_dim + candidate.pairs + pair)
        * candidate.hidden_size
        + pair
        for kv_head in range(candidate.kv_heads)
        for pair in range(candidate.pairs)
    ], device="cuda")
    serving_coefficient = (
        key_weight.reshape(-1)[coefficient_indices]
        .reshape(candidate.kv_heads, candidate.pairs)
        / artifact["tau"]
    )
    torch.testing.assert_close(
        serving_coefficient, training_coefficient, atol=0.0, rtol=0.0
    )

    candidate.to(torch.bfloat16)
    hidden_bf16 = hidden.to(torch.bfloat16)
    with torch.inference_mode():
        training_cache = DynamicCache()
        reloaded_reference = candidate(
            hidden_bf16,
            position_embeddings=position,
            attention_mask=None,
            past_key_values=training_cache,
            cache_position=torch.arange(7, device="cuda"),
        )[0]

    input_shape = hidden_bf16.shape[:-1]
    head_shape = (*input_shape, -1, candidate.head_dim)
    serving_query = torch.nn.functional.linear(
        hidden_bf16, query_weight
    ).view(head_shape).transpose(1, 2)
    serving_key = torch.nn.functional.linear(
        hidden_bf16, key_weight
    ).view(head_shape).transpose(1, 2)
    serving_value = torch.nn.functional.linear(
        hidden_bf16, value_weight
    ).view(head_shape).transpose(1, 2)
    query_even = serving_query[..., :candidate.pairs].float()
    query_odd = serving_query[..., candidate.pairs:].float()
    key_even = serving_key[..., :candidate.pairs].float()
    key_odd = (
        serving_key[..., candidate.pairs:].float()
        + serving_coefficient[None, :, None].float() * key_even.square()
    )
    cos = position[0][..., :candidate.pairs].unsqueeze(1).float()
    sin = position[1][..., :candidate.pairs].unsqueeze(1).float()
    serving_query = torch.cat((
        cos * query_even - sin * query_odd,
        sin * query_even + cos * query_odd,
    ), dim=-1).to(torch.bfloat16)
    serving_key = torch.cat((
        cos * key_even - sin * key_odd,
        sin * key_even + cos * key_odd,
    ), dim=-1).to(torch.bfloat16)
    serving_cache = DynamicCache()
    serving_key, serving_value = serving_cache.update(
        serving_key,
        serving_value,
        candidate.layer_idx,
        {
            "sin": position[1],
            "cos": position[0],
            "cache_position": torch.arange(7, device="cuda"),
        },
    )
    serving_attention, _ = eager_attention_forward(
        candidate,
        serving_query,
        serving_key,
        serving_value,
        None,
        dropout=0.0,
        scaling=candidate.scaling,
    )
    serving_output = torch.nn.functional.linear(
        serving_attention.reshape(*input_shape, -1).contiguous(),
        output_weight,
    )

    training_key = training_cache.layers[0].keys
    exported_key = serving_cache.layers[0].keys
    assert training_key.dtype == exported_key.dtype == torch.bfloat16
    assert training_key.shape == exported_key.shape
    assert training_key.numel() * training_key.element_size() == (
        exported_key.numel() * exported_key.element_size()
    )
    torch.testing.assert_close(exported_key, training_key, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        serving_output, reloaded_reference, atol=0.0, rtol=0.0
    )
