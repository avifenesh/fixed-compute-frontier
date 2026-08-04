import pytest
import torch
import torch.nn.functional as F
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention

from experiments.triangular_value_encoding_attention import (
    TriangularValueEncodingAttention,
    apply_rotary_fp32_one_store,
)
from experiments.triangular_value_encoding_cache_h100 import (
    BLOCK,
    GROUPS,
    HALF,
    HEAD_DIM,
    HIDDEN,
    K_DIM,
    KV_HEADS,
    Q_DIM,
    QKV_DIM,
    QUERY_HEADS,
    TAU,
    V_DIM,
    cache_write_epilogue,
    downstream_error,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0),
    reason="H100 required",
)


def full_config():
    return LlamaConfig(
        hidden_size=HIDDEN,
        intermediate_size=11008,
        num_hidden_layers=1,
        num_attention_heads=QUERY_HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        attention_bias=False,
    )


@torch.no_grad()
def test_nonzero_export_matches_cache_kernel_exactly():
    torch.manual_seed(701)
    source = LlamaAttention(full_config(), layer_idx=0)
    module = TriangularValueEncodingAttention(
        source, "triangular_value_encoding", block_size=BLOCK
    ).to("cuda", dtype=torch.bfloat16)
    del source

    sentinel_count = 0
    with torch.no_grad():
        # One exact power-of-two coefficient per nontrivial target exercises
        # every KV head and 16x16 block without reduction-order ambiguity.
        for kv_head in range(KV_HEADS):
            representative_head = kv_head * GROUPS
            for block_start in range(0, HEAD_DIM, BLOCK):
                for target in range(1, BLOCK):
                    coefficient = (1 + sentinel_count % 7) / 64.0
                    module.output_weight[
                        block_start + target,
                        representative_head * HEAD_DIM + block_start,
                    ] = TAU * coefficient
                    sentinel_count += 1
    assert sentinel_count == KV_HEADS * (HEAD_DIM // BLOCK) * (BLOCK - 1)
    assert torch.count_nonzero(module.coefficient_values()) == sentinel_count

    rows = 65
    generator = torch.Generator(device="cuda").manual_seed(709)
    hidden = torch.randn(
        rows, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    angles = torch.randn(
        rows, HALF, device="cuda", dtype=torch.float32, generator=generator
    )
    cosine_half = angles.cos().to(torch.bfloat16)
    sine_half = angles.sin().to(torch.bfloat16)
    cosine = torch.cat((cosine_half, cosine_half), -1)[None]
    sine = torch.cat((sine_half, sine_half), -1)[None]
    qkv = module.project_qkv_packed(hidden).contiguous()
    raw_qkv = qkv.clone()
    key_cache = torch.empty(rows, K_DIM, device="cuda", dtype=torch.bfloat16)
    value_cache = torch.empty(rows, V_DIM, device="cuda", dtype=torch.bfloat16)
    cache_write_epilogue(
        qkv,
        module.output_weight,
        cosine_half,
        sine_half,
        key_cache,
        value_cache,
        candidate=True,
    )

    query = raw_qkv[:, :Q_DIM].view(
        1, rows, QUERY_HEADS, HEAD_DIM
    ).transpose(1, 2)
    key = raw_qkv[:, Q_DIM:Q_DIM + K_DIM].view(
        1, rows, KV_HEADS, HEAD_DIM
    ).transpose(1, 2)
    query, key = apply_rotary_fp32_one_store(query, key, cosine, sine)
    value = raw_qkv[:, Q_DIM + K_DIM:].view(
        1, rows, KV_HEADS, HEAD_DIM
    ).transpose(1, 2)
    encoded = module.apply_value_encoding(value)
    expected_qkv_q = query.transpose(1, 2).reshape(rows, Q_DIM)
    expected_qkv_k = key.transpose(1, 2).reshape(rows, K_DIM)
    expected_cache_v = encoded.transpose(1, 2).reshape(rows, V_DIM)

    torch.testing.assert_close(qkv[:, :Q_DIM], expected_qkv_q, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        qkv[:, Q_DIM:Q_DIM + K_DIM], expected_qkv_k, atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        qkv[:, Q_DIM + K_DIM:], expected_cache_v, atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(key_cache, expected_qkv_k, atol=0.0, rtol=0.0)
    torch.testing.assert_close(value_cache, expected_cache_v, atol=0.0, rtol=0.0)
    raw_value = raw_qkv[:, Q_DIM + K_DIM:]
    assert torch.count_nonzero(value_cache - raw_value) > 0

    # Dense trained-like coefficients exercise the tensor-core reduction at
    # the largest admitted row shape. Different legal reduction trees may
    # disagree in a tiny number of final BF16 values, so bind that drift and
    # propagate it through attention mixing, O projection, and an NLL proxy.
    dense_index = 0
    for kv_head in range(KV_HEADS):
        representative_head = kv_head * GROUPS
        for block_start in range(0, HEAD_DIM, BLOCK):
            for target in range(1, BLOCK):
                for source_coordinate in range(target):
                    coefficient = ((dense_index % 31) - 15) / 256.0
                    module.output_weight[
                        block_start + target,
                        representative_head * HEAD_DIM + block_start + source_coordinate,
                    ] = TAU * coefficient
                    dense_index += 1
    assert dense_index == KV_HEADS * (HEAD_DIM // BLOCK) * BLOCK * (BLOCK - 1) // 2

    dense_rows = 2048
    dense_hidden = torch.randn(
        dense_rows,
        HIDDEN,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    dense_angles = torch.randn(
        dense_rows,
        HALF,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    dense_cosine = dense_angles.cos().to(torch.bfloat16)
    dense_sine = dense_angles.sin().to(torch.bfloat16)
    dense_qkv = module.project_qkv_packed(dense_hidden).contiguous()
    dense_raw = dense_qkv.clone()
    dense_key_cache = torch.empty(
        dense_rows, K_DIM, device="cuda", dtype=torch.bfloat16
    )
    dense_value_cache = torch.empty(
        dense_rows, V_DIM, device="cuda", dtype=torch.bfloat16
    )
    cache_write_epilogue(
        dense_qkv,
        module.output_weight,
        dense_cosine,
        dense_sine,
        dense_key_cache,
        dense_value_cache,
        candidate=True,
    )
    dense_value = dense_raw[:, Q_DIM + K_DIM:].view(
        1, dense_rows, KV_HEADS, HEAD_DIM
    ).transpose(1, 2)
    dense_expected = module.apply_value_encoding(dense_value).transpose(
        1, 2
    ).reshape(dense_rows, V_DIM)
    difference = (dense_value_cache.float() - dense_expected.float()).abs()
    assert float(difference.max()) <= 0.015625
    assert float(difference.mean()) <= 1e-8
    assert float((difference != 0).float().mean()) <= 1e-6
    propagated = downstream_error(
        dense_value_cache, dense_expected, module.output_weight, seed=719
    )
    assert propagated["output_projection_max_abs"] <= 0.015625
    assert propagated["output_projection_mean_abs"] <= 1e-7
    assert propagated["proxy_nll_abs_difference"] <= 1e-6
