import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from experiments.partner_feedback_swiglu_h100 import (  # noqa: E402
    decode_lambda,
    make_weights,
    partner_activation,
    torch_reference,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_partner_kernel_matches_reference():
    torch.manual_seed(5)
    device = torch.device("cuda")
    _, up_weight, _, chart, _ = make_weights(
        64, 256, torch.bfloat16, device, 0.08, 4.0
    )
    coefficient = decode_lambda(up_weight, chart, 4.0)
    assert abs(coefficient - 0.08) <= 0.002
    gate = torch.randn(7, 256, device=device, dtype=torch.bfloat16)
    up = torch.randn(7, 256, device=device, dtype=torch.bfloat16)
    expected = torch_reference(gate, up, coefficient)
    actual = partner_activation(gate, up, up_weight, chart, 4.0)
    error = torch.linalg.vector_norm(actual.float() - expected, dim=-1) / torch.linalg.vector_norm(
        expected, dim=-1
    ).clamp_min(1e-12)
    assert float(error.max()) <= 0.01

