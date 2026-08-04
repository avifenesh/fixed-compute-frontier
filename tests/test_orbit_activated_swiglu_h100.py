import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from experiments.orbit_activated_swiglu_h100 import (  # noqa: E402
    decode_coefficients,
    make_weights,
    orbit_activation,
    torch_orbit_reference,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_kernel_matches_reference_and_decodes_pivots() -> None:
    torch.manual_seed(3)
    device = torch.device("cuda")
    dimension, hidden = 64, 256
    gate_weight, up_weight, _, encoded, chart_abs = make_weights(
        dimension, hidden, torch.bfloat16, device, 0.035, 8.0
    )
    del gate_weight
    decoded = decode_coefficients(up_weight, chart_abs, 8.0)
    assert float((decoded - encoded).abs().max()) <= 0.001
    gate = torch.randn(5, hidden, device=device, dtype=torch.bfloat16)
    up = torch.randn(5, hidden, device=device, dtype=torch.bfloat16)
    expected = torch_orbit_reference(gate, up, decoded)
    actual = orbit_activation(gate, up, up_weight, chart_abs, 8.0)
    error = torch.linalg.vector_norm(actual.float() - expected, dim=-1) / torch.linalg.vector_norm(
        expected, dim=-1
    ).clamp_min(1e-12)
    assert float(error.max()) <= 0.01

