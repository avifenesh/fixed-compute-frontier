import numpy as np

from experiments.gauge_embedded_curved_keys_learning import (
    ARMS,
    initial_parameters,
    make_restart_starts,
    make_world,
    model_logits,
    run_world,
)


def test_restart_starts_are_deterministic_and_shareable_across_arms():
    first = make_restart_starts(123)
    second = make_restart_starts(123)
    assert first.shape == (3, 10)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[:, 8:], 0.0)


def test_all_arms_have_same_raw_parameter_count_and_finite_logits():
    world = make_world(5, samples=128)
    parameters = initial_parameters()
    assert parameters.size == 10
    for arm in ARMS:
        logits = model_logits(
            parameters, arm, world["query_types"], world["keys"]
        )
        assert logits.shape == (128,)
        assert np.isfinite(logits).all()


def test_constant_and_centered_terms_cancel_between_same_position_keys():
    world = make_world(7, samples=128)
    parameters = initial_parameters()
    parameters[8:] = (0.6, -0.4)
    constant = model_logits(
        parameters, "constant_only", world["query_types"], world["keys"]
    )
    bilinear = model_logits(
        parameters, "bilinear", world["query_types"], world["keys"]
    )
    np.testing.assert_allclose(constant, bilinear, atol=1e-12)
    centered = model_logits(
        parameters, "centered_full", world["query_types"], world["keys"]
    )
    full = model_logits(
        parameters, "full_geck_g2", world["query_types"], world["keys"]
    )
    np.testing.assert_allclose(centered, full, atol=1e-12)


def test_single_frozen_learning_world_passes():
    result = run_world(41)
    assert result["pass"]
    assert all(result["gates"].values())
    for teacher in ("nonlinear_teacher", "bilinear_teacher"):
        for fit in result[teacher].values():
            assert fit["optimization_success"]
            assert fit["finite"]
