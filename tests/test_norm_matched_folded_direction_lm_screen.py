import torch

from experiments.norm_matched_folded_direction_lm_screen import norm_matched_rotation


def test_rotation_preserves_base_norm_and_is_orthogonal():
    torch.manual_seed(5)
    base = torch.randn(19, 13)
    factor = torch.randn(19, 13)
    rotated, perpendicular, _ = norm_matched_rotation(base, factor)
    torch.testing.assert_close(
        torch.linalg.vector_norm(rotated),
        torch.linalg.vector_norm(base),
        rtol=2e-6,
        atol=2e-6,
    )
    cosine = torch.sum(base * perpendicular).abs() / (
        torch.linalg.vector_norm(base) * torch.linalg.vector_norm(perpendicular)
    )
    assert float(cosine) < 2e-6


def test_parallel_factor_component_is_removed_exactly_enough():
    torch.manual_seed(7)
    base = torch.randn(11, 17)
    factor = 3.25 * base
    rotated, perpendicular, projection = norm_matched_rotation(base, factor)
    torch.testing.assert_close(perpendicular, torch.zeros_like(perpendicular), atol=2e-6, rtol=0)
    torch.testing.assert_close(rotated, base, atol=2e-6, rtol=2e-6)
    assert abs(float(projection) - 3.25) < 2e-6


def test_zero_base_is_safe():
    base = torch.zeros(3, 4)
    factor = torch.randn(3, 4)
    rotated, perpendicular, projection = norm_matched_rotation(base, factor)
    assert torch.equal(rotated, base)
    assert torch.equal(perpendicular, factor)
    assert float(projection) == 0.0

