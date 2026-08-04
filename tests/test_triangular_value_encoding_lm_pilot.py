import torch
import pytest
from transformers.models.llama.modeling_llama import LlamaAttention

from experiments.triangular_microdepth_lm_screen import scratch_config
from experiments.triangular_value_encoding_attention import (
    TriangularValueEncodingAttention,
    serving_value_encoding,
)
from experiments.triangular_value_encoding_lm_pilot import coefficient_diagnostics


def make_module(seed, arm):
    torch.manual_seed(seed)
    return TriangularValueEncodingAttention(
        LlamaAttention(scratch_config(), layer_idx=0), arm, block_size=16
    )


def test_initial_candidate_and_control_coefficients_export_exactly():
    control = make_module(1301, "canonical_value_control")
    candidate = make_module(1301, "triangular_value_encoding")
    for module in (control, candidate):
        diagnostics = coefficient_diagnostics([module])
        assert diagnostics["bf16_export_coefficient_bit_exact"]
        assert diagnostics["coefficient_bf16_nonzero_fraction"] == 0.0
        assert diagnostics["coefficient_bf16_abs_max"] == 0.0
    assert torch.equal(control.value_weight, candidate.value_weight)
    assert torch.equal(control.output_weight, candidate.output_weight)


def test_all_arms_keep_identical_learned_tensor_budget():
    modules = [
        make_module(1303, arm)
        for arm in (
            "packed_raw_control",
            "canonical_value_control",
            "triangular_value_encoding",
        )
    ]
    assert len({sum(p.numel() for p in module.parameters()) for module in modules}) == 1
    assert len({len(list(module.parameters())) for module in modules}) == 1
    assert all(list(module.buffers()) == [] for module in modules)


def test_nonzero_coefficients_preserve_export_order_and_values():
    module = make_module(1307, "triangular_value_encoding")
    index = 0
    with torch.no_grad():
        for kv_head in range(module.kv_heads):
            representative = kv_head * module.num_key_value_groups
            for start in range(0, module.head_dim, module.block_size):
                for target in range(1, module.block_size):
                    for source in range(target):
                        coefficient = ((index % 15) - 7) / 128.0
                        module.output_weight[
                            start + target,
                            representative * module.head_dim + start + source,
                        ] = module.tau * coefficient
                        index += 1
    diagnostics = coefficient_diagnostics([module])
    assert diagnostics["bf16_export_coefficient_bit_exact"]
    assert diagnostics["coefficient_bf16_nonzero_fraction"] > 0.8
    assert diagnostics["coefficient_bf16_abs_max"] == 7 / 128


@pytest.mark.skipif(
    not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0),
    reason="H100 required",
)
def test_cuda_bf16_serving_forward_has_surrogate_gradients():
    module = make_module(1311, "triangular_value_encoding").cuda()
    with torch.no_grad():
        module.output_weight[1, 0] = module.tau * 0.25
    values = torch.randn(
        1,
        module.kv_heads,
        65,
        module.head_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    encoded = module.apply_value_encoding(values)
    token_major = values.detach().transpose(1, 2).contiguous().view(
        65, module.kv_heads, module.head_dim
    )
    expected = serving_value_encoding(
        token_major,
        module.output_weight.detach().to(torch.bfloat16),
        query_groups=module.num_key_value_groups,
        block_size=module.block_size,
        tau=module.tau,
    ).view(1, 65, module.kv_heads, module.head_dim).transpose(1, 2)
    assert torch.equal(encoded, expected)
    encoded.float().square().mean().backward()
    assert values.grad is not None and torch.isfinite(values.grad.float()).all()
    assert module.output_weight.grad is not None
    assert torch.count_nonzero(module.output_weight.grad) > 0
