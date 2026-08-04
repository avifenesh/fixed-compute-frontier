import importlib.util
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "triangular_microdepth_lm_screen.py"
)
SPEC = importlib.util.spec_from_file_location("triangular_microdepth_lm_screen", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_candidate_shape_closes_exact_ffn_parameter_budget() -> None:
    baseline = 3 * MODULE.HIDDEN_SIZE * MODULE.BASELINE_INTERMEDIATE_SIZE
    candidate = (
        2 * MODULE.HIDDEN_SIZE * MODULE.CANDIDATE_WIDTH
        + MODULE.COUPLING_PARAMETERS
        + MODULE.BIAS_COUNT
    )
    assert MODULE.CANDIDATE_WIDTH == 1528
    assert candidate == baseline
    assert 2 * MODULE.HIDDEN_SIZE * MODULE.PlainSiluMLP.WIDTH == baseline
    assert len(MODULE.BIAS_INDICES) == len(set(MODULE.BIAS_INDICES)) == 796


def test_zero_coupling_arms_have_identical_activations() -> None:
    torch.manual_seed(9)
    linear = MODULE.TriangularMicrodepthMLP("linear_reparam", 0.03)
    torch.manual_seed(9)
    triangular = MODULE.TriangularMicrodepthMLP("triangular", 0.03)
    hidden = torch.randn(3, 5, MODULE.HIDDEN_SIZE)
    torch.testing.assert_close(
        linear.activation(hidden), triangular.activation(hidden), rtol=1e-6, atol=2e-7
    )


def test_triangular_coupling_has_nonzero_gradient_at_zero() -> None:
    torch.manual_seed(11)
    module = MODULE.TriangularMicrodepthMLP("triangular", 0.03)
    hidden = torch.randn(2, 4, MODULE.HIDDEN_SIZE)
    module(hidden).square().mean().backward()
    gradient = module.packed_coupling.grad
    assert gradient is not None
    assert torch.count_nonzero(gradient).item() > 0
    assert torch.isfinite(gradient).all()


def test_linear_control_coupling_is_functionally_live() -> None:
    torch.manual_seed(13)
    module = MODULE.TriangularMicrodepthMLP("linear_reparam", 0.03)
    hidden = torch.randn(2, 3, MODULE.HIDDEN_SIZE)
    module(hidden).sum().backward()
    gradient = module.packed_coupling.grad
    assert gradient is not None
    assert torch.count_nonzero(gradient).item() > 0


def test_one_hop_ablation_removes_recursive_dependence() -> None:
    torch.manual_seed(15)
    module = MODULE.TriangularMicrodepthMLP("triangular", 0.03)
    with torch.no_grad():
        module.packed_coupling.normal_(std=0.1)
    hidden = torch.randn(2, 3, MODULE.HIDDEN_SIZE)
    full = module.activation(hidden)
    module.ablation_mode = "one_hop"
    one_hop = module.activation(hidden)
    assert not torch.equal(full, one_hop)


def test_decision_requires_both_superiority_controls() -> None:
    source = MODULE.decide.__code__.co_consts
    flattened = " ".join(value for value in source if isinstance(value, str))
    assert "triangular_at_least_0p1_percent_better_than_baseline" in flattened
    assert "triangular_at_least_0p1_percent_better_than_linear_control" in flattened
    assert "triangular_at_least_0p1_percent_better_than_plain_silu" in flattened
    assert "paired_interval_favors_triangular_vs_plain_silu" in flattened
    assert "zeroing_h_hurts_by_at_least_0p05_percent" in flattened
    assert "removing_multihop_hurts_by_at_least_0p02_percent" in flattened
