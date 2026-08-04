import numpy as np

from experiments.gauge_embedded_curved_keys_natural import (
    ARMS,
    ITEMS,
    WEIGHT_DECAY,
    initial_parameters,
    make_world,
    model_logits,
    objective_and_gradient,
)


def test_all_arms_have_same_raw_parameter_count_and_finite_logits():
    world = make_world(5)
    parameters = initial_parameters()
    assert parameters.size == 10
    subset = slice(0, 32)
    for arm in ARMS:
        logits = model_logits(
            parameters,
            arm,
            world["query_types"][subset],
            world["keys"][subset],
            world["angles"][subset],
        )
        assert logits.shape == (32, ITEMS)
        assert np.isfinite(logits).all()


def test_analytic_gradient_matches_centered_finite_difference_for_every_arm():
    world = make_world(7)
    subset = slice(0, 19)
    parameters = initial_parameters() + np.array([
        0.03, -0.02, 0.01, 0.04, -0.03,
        0.02, -0.01, 0.05, 0.31, -0.27,
    ])
    epsilon = 1e-6
    for arm in ARMS:
        objective, analytic = objective_and_gradient(
            parameters,
            arm,
            world["query_types"][subset],
            world["keys"][subset],
            world["angles"][subset],
            world["magnitude_labels"][subset],
        )
        numeric = np.empty_like(parameters)
        for index in range(parameters.size):
            delta = np.zeros_like(parameters)
            delta[index] = epsilon
            plus = objective_and_gradient(
                parameters + delta,
                arm,
                world["query_types"][subset],
                world["keys"][subset],
                world["angles"][subset],
                world["magnitude_labels"][subset],
            )[0]
            minus = objective_and_gradient(
                parameters - delta,
                arm,
                world["query_types"][subset],
                world["keys"][subset],
                world["angles"][subset],
                world["magnitude_labels"][subset],
            )[0]
            numeric[index] = (plus - minus) / (2.0 * epsilon)
        assert np.isfinite(objective)
        np.testing.assert_allclose(analytic, numeric, atol=2e-7, rtol=2e-6)


def test_zero_curvature_is_exact_bilinear_containment():
    world = make_world(11)
    parameters = initial_parameters()
    parameters[:8] += np.linspace(-0.2, 0.2, 8)
    parameters[8:] = 0.0
    subset = slice(0, 64)
    bilinear = model_logits(
        parameters,
        "bilinear",
        world["query_types"][subset],
        world["keys"][subset],
        world["angles"][subset],
    )
    for arm in ("one_curvature", "centered_full", "full_geck_g2"):
        candidate = model_logits(
            parameters,
            arm,
            world["query_types"][subset],
            world["keys"][subset],
            world["angles"][subset],
        )
        np.testing.assert_array_equal(candidate, bilinear)


def test_weight_decay_is_frozen_and_positive():
    assert WEIGHT_DECAY == 1e-4
