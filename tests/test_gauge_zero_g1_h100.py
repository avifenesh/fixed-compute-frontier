import pytest
import torch

from experiments.gauge_zero_g1_attention import GaugeZeroG1Attention
from experiments.gauge_zero_g1_h100 import (
    HIDDEN,
    QKV_DIM,
    benchmark_cell,
    bootstrap_median_ratio,
    rope_inplace,
)
from experiments.triangular_microdepth_lm_screen import scratch_config
from transformers.models.llama.modeling_llama import LlamaAttention


def test_bootstrap_median_ratio_is_exact_for_constant_scaling():
    result = bootstrap_median_ratio(
        [1.01, 2.02, 3.03, 4.04],
        [1.0, 2.0, 3.0, 4.0],
        seed=17,
    )
    assert result["median_ratio"] == pytest.approx(1.01)
    assert result["lower_95"] == pytest.approx(1.01)
    assert result["upper_95"] == pytest.approx(1.01)


@pytest.mark.skipif(
    not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0),
    reason="H100 kernel smoke",
)
def test_h100_kernel_is_bf16_exact_and_ledger_is_equal():
    cell = benchmark_cell(rows=1, trials=3, warmup=1, seed=43)
    assert cell["correctness"]["candidate_bit_exact_bf16"]
    assert cell["correctness"]["control_bit_exact_bf16"]
    assert cell["correctness"]["candidate_max_abs"] == 0.0
    assert cell["correctness"]["control_max_abs"] == 0.0
    assert cell["logical"]["resident_weight_bytes_each"] == (
        QKV_DIM * HIDDEN * 2
    )
    assert cell["logical"]["pivot_metadata_bits"] == 0
    assert set(cell["candidate_over_control"]) == {
        "median_ratio",
        "lower_95",
        "upper_95",
    }


@pytest.mark.skipif(
    not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0),
    reason="H100 prototype-to-serving bridge",
)
@pytest.mark.parametrize(
    ("arm", "g1"),
    [("canonical_bilinear_control", False), ("gauge_zero_g1", True)],
)
def test_h100_kernel_bit_exact_with_lm_projection_and_rounding_contract(arm, g1):
    config = scratch_config()
    torch.manual_seed(211)
    source = LlamaAttention(config, layer_idx=0).to(
        "cuda", dtype=torch.bfloat16
    )
    module = GaugeZeroG1Attention(source, arm).cuda().eval()
    with torch.no_grad():
        coefficient = torch.linspace(
            -0.02,
            0.02,
            module.kv_heads * module.pairs,
            device="cuda",
            dtype=torch.bfloat16,
        ).reshape(module.kv_heads, module.pairs)
        module.coefficient_physical_weights().copy_(coefficient)

    rows = 7
    generator = torch.Generator(device="cuda").manual_seed(223)
    hidden = torch.randn(
        1,
        rows,
        module.hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    angles = torch.randn(
        1,
        rows,
        module.pairs,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    cosine_half = angles.cos().to(torch.bfloat16)
    sine_half = angles.sin().to(torch.bfloat16)
    position = (
        torch.cat((cosine_half, cosine_half), dim=-1),
        torch.cat((sine_half, sine_half), dim=-1),
    )
    with torch.inference_mode():
        query, key, value = module.project_qkv_rope(hidden, position)
        weight = torch.cat((
            module.query_weight,
            module.key_weight,
            module.v_proj.weight,
        ), dim=0).contiguous()
        serving = hidden.squeeze(0) @ weight.T
        rope_inplace(
            serving,
            weight,
            cosine_half.squeeze(0),
            sine_half.squeeze(0),
            g1=g1,
            hidden=module.hidden_size,
            query_heads=module.query_heads,
            kv_heads=module.kv_heads,
            head_dim=module.head_dim,
            tau=module.tau,
        )
    q_width = module.query_heads * module.head_dim
    k_width = module.kv_heads * module.head_dim
    prototype = torch.cat((
        query.transpose(1, 2).reshape(rows, q_width),
        key.transpose(1, 2).reshape(rows, k_width),
        value.transpose(1, 2).reshape(rows, k_width),
    ), dim=-1)
    torch.testing.assert_close(serving, prototype, atol=0.0, rtol=0.0)
