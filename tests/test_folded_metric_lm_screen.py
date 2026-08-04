import importlib.util
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "folded_metric_lm_screen.py"
)
SPEC = importlib.util.spec_from_file_location("folded_metric_lm_screen", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_tensor_adam_matches_first_step_sign_normalization():
    value = torch.zeros(2, 3)
    gradient = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
    first = torch.zeros_like(value)
    second = torch.zeros_like(value)
    MODULE.adam_tensor_step(value, gradient, first, second, 1, 0.01)
    expected = -0.01 * gradient / (gradient.abs() + 1e-8)
    torch.testing.assert_close(value, expected)


def test_deterministic_orthonormal_columns():
    first = MODULE.deterministic_orthonormal(11, 4, 9, torch.device("cpu"))
    second = MODULE.deterministic_orthonormal(11, 4, 9, torch.device("cpu"))
    assert torch.equal(first, second)
    torch.testing.assert_close(first.T @ first, torch.eye(4), atol=1e-6, rtol=1e-6)


def test_controller_keeps_factors_outside_model_parameters():
    linear = torch.nn.Linear(7, 9, bias=False)
    model_parameters = sum(p.numel() for p in linear.parameters())
    controller = MODULE.FoldedMetricController({"x": linear.weight}, learn_a=True, rank=3)
    assert sum(p.numel() for p in linear.parameters()) == model_parameters
    assert controller.factor_scalars == 3 * (9 + 7)
    assert not any("metric" in key or "factor" in key for key in linear.state_dict())


def fake_arm(loss, early_loss, per_batch, factor_scalars=0, adaptive=False):
    diagnostics = {}
    if factor_scalars:
        diagnostics["1525"] = {"median_a_change_norm": 0.1 if adaptive else 0.0}
    return {
        "model_parameters": 100,
        "optimizer_factor_scalars": factor_scalars,
        "evaluations": {
            "0": {"loss": 10.0},
            "305": {"loss": early_loss},
            "1525": {"loss": loss, "per_batch_loss": per_batch},
        },
        "metric_update_diagnostics": diagnostics,
        "served_reload_check": {
            "logits_bitwise_equal": True,
            "loss_absolute_difference": 0.0,
            "state_dict_contains_metric_factors": False,
            "disk_serialization_checked": True,
            "reloaded_parameter_count": 100,
        },
        "train": {"nonfinite": False, "max_loss": 10.0, "max_preclip_gradient_norm": 1.0},
    }


def test_decision_requires_all_three_controls():
    results = {
        "baseline": fake_arm(5.0, 6.0, [5.00, 5.02, 4.98, 5.01]),
        "lr_matched": fake_arm(4.99, 5.99, [4.99, 5.01, 4.97, 5.00]),
        "fixed_metric": fake_arm(4.98, 5.98, [4.98, 5.00, 4.96, 4.99], 10),
        "learned_metric": fake_arm(4.96, 5.97, [4.96, 4.98, 4.94, 4.97], 10, adaptive=True),
    }
    decision = MODULE.decide(results, integrity_valid=True)
    assert decision["advance_to_second_seed"]


def test_incomplete_result_is_not_decided():
    decision = MODULE.decide({"baseline": {}}, integrity_valid=True)
    assert not decision["complete"]
    assert decision["missing_arms"] == ["fixed_metric", "learned_metric", "lr_matched"]
