import importlib.util
import math
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "block_algebra_swiglu_lm_screen.py"
)
SPEC = importlib.util.spec_from_file_location(
    "block_algebra_swiglu_lm_screen", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_block_products_match_frozen_formulas():
    gate = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    up = torch.tensor([[5.0, 6.0, 7.0, 8.0]])
    baseline = MODULE.block_product(gate, up, "baseline")
    split = MODULE.block_product(gate, up, "split_complex")
    complex_result = MODULE.block_product(gate, up, "complex")
    assert torch.equal(baseline, torch.tensor([[5.0, 12.0, 21.0, 32.0]]))
    expected_split = torch.tensor(
        [[17.0, 16.0, 53.0, 52.0]]
    ) / math.sqrt(2.0)
    expected_complex = torch.tensor(
        [[-7.0, 16.0, -11.0, 52.0]]
    ) / math.sqrt(2.0)
    torch.testing.assert_close(split, expected_split)
    torch.testing.assert_close(complex_result, expected_complex)


def test_block_products_preserve_shape_and_dtype():
    gate = torch.randn(2, 5, 8, dtype=torch.bfloat16)
    up = torch.randn_like(gate)
    for arm in MODULE.VALID_ARMS:
        result = MODULE.block_product(gate, up, arm)
        assert result.shape == gate.shape
        assert result.dtype == gate.dtype


def test_odd_width_and_unknown_arm_are_rejected():
    gate = torch.randn(2, 3)
    up = torch.randn_like(gate)
    with pytest.raises(ValueError, match="even"):
        MODULE.block_product(gate, up, "complex")
    with pytest.raises(ValueError, match="unknown"):
        MODULE.block_product(torch.randn(2, 4), torch.randn(2, 4), "bad")


def fake_arm(loss, per_batch, parameters=100, rms=1.0):
    return {
        "total_parameters": parameters,
        "trainable_parameters": parameters,
        "evaluations": {
            "1525": {"loss": loss, "per_batch_loss": per_batch}
        },
        "activation_diagnostics": {
            "initial": {"activation_rms_layer_median": rms}
        },
        "train": {
            "nonfinite": False,
            "max_preclip_gradient_norm": 1.0,
            "max_loss": 10.0,
        },
    }


def test_decision_requires_effect_size_and_both_controls():
    results = {
        "baseline": fake_arm(4.0, [4.0, 4.02, 3.98, 4.01]),
        "split_complex": fake_arm(3.99, [3.99, 4.01, 3.97, 4.0]),
        "complex": fake_arm(3.97, [3.97, 3.99, 3.95, 3.98]),
    }
    decision = MODULE.decide(results, integrity_valid=True)
    assert decision["gates"]["complex_at_least_0p1_percent_better_than_baseline"]
    assert decision["gates"]["complex_at_least_0p1_percent_better_than_split_control"]
    assert decision["advance_to_fused_h100_gate"]


def test_incomplete_results_do_not_decide():
    decision = MODULE.decide({"baseline": {}}, integrity_valid=True)
    assert not decision["complete"]
    assert decision["missing_arms"] == ["complex", "split_complex"]
