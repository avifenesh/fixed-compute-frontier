import importlib.util
from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")
MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "cross_view_gradient_shadow_gate.py"
)
SPEC = importlib.util.spec_from_file_location("cross_view_gradient_shadow_gate", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_axis_filter_preserves_norm():
    direction = torch.randn(5, 9)
    weights = torch.linspace(0.2, 1.0, 5)
    filtered = MODULE.apply_axis_weights(direction, weights, rows=True)
    torch.testing.assert_close(
        torch.linalg.vector_norm(filtered), torch.linalg.vector_norm(direction)
    )


def test_cross_reliability_recovers_clean_rows():
    generator = torch.Generator().manual_seed(7)
    state = MODULE.DiagonalCrossViewState((4, 128), 9, torch.device("cpu"))
    signal = torch.randn(4, 128, generator=generator)
    noise_scale = torch.tensor([0.05, 0.2, 2.0, 5.0])[:, None]
    for _ in range(200):
        first = signal + noise_scale * torch.randn(4, 128, generator=generator)
        second = signal + noise_scale * torch.randn(4, 128, generator=generator)
        state.update(first, second, beta=0.95)
    weights = state.weights()["cross"]
    assert weights[0] > weights[1] > weights[2] > weights[3]


def test_hypothetical_adam_first_step_is_sign_direction():
    gradient = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
    direction = MODULE.hypothetical_adam_direction(gradient, {}, 1)
    expected = gradient / (gradient.abs() + 1e-8)
    torch.testing.assert_close(direction, expected)


def make_step(cross, identity, auto, shuffled):
    return {
        "families": {
            family: {
                "cos_identity": identity,
                "cos_scalar": identity,
                "cos_cross": cross,
                "cos_shuffled_cross": shuffled,
                "cos_auto_inv_sqrt": auto,
                "cos_auto_inv": auto - 0.001,
                "cos_auto_sqrt": auto - 0.002,
                "cos_auto_direct": auto - 0.003,
                "nmse_scalar": 0.8,
                "nmse_cross": 0.78,
                "nmse_shuffled": 0.82,
                "update_angle": 0.001,
                "max_norm_error": 1e-7,
                "raw_ratio_clipped_fraction": 0.1,
                "fallback_fraction": 0.0,
            }
            for family in MODULE.FAMILIES
        }
    }


def test_summary_requires_directional_effect_beyond_controls():
    passing = [make_step(0.30, 0.28, 0.285, 0.27) for _ in range(216)]
    decision = MODULE.summarize(passing, block_size=20)
    assert decision["advance_to_lm_screen"]
    failing = [make_step(0.284, 0.28, 0.285, 0.27) for _ in range(216)]
    assert not MODULE.summarize(failing, block_size=20)["advance_to_lm_screen"]


def test_summary_rejects_nonfinite_score():
    records = [make_step(0.30, 0.28, 0.285, 0.27) for _ in range(216)]
    records[100]["families"]["q"]["cos_cross"] = float("nan")
    decision = MODULE.summarize(records, block_size=20)
    assert not decision["gates"]["all_scores_finite"]
    assert not decision["advance_to_lm_screen"]


def test_summary_uses_median_geometry_not_outlier_mean():
    records = [make_step(0.30, 0.28, 0.285, 0.27) for _ in range(216)]
    for record in records:
        for family in MODULE.FAMILIES:
            record["families"][family]["update_angle"] = 0.0
    records[0]["families"]["q"]["update_angle"] = 1.0
    decision = MODULE.summarize(records, block_size=20)
    assert decision["families"]["q"]["mean_cosines"]["update_angle"] == 0.0
    assert not decision["gates"]["nontrivial_geometry_in_5_of_7_families"]
