import numpy as np

from experiments.dihedral_monomial_prefix_t30_separation_preflight import (
    IDENTITY,
    RHO,
    TAU,
    WIDTH,
    apply_group,
    canonical_word,
    centered_basis,
    compose,
    fixed_points,
    fold_word,
    group_elements,
    normalized_mse,
    optimal_diagonal,
    permutation_matrix,
    run,
)


def test_canonical_words_and_distinct_actions():
    groups = group_elements()
    assert all(fold_word(canonical_word(group)) == group for group in groups)
    assert len({permutation_matrix(group).tobytes() for group in groups}) == 2 * WIDTH


def test_generators_do_not_commute():
    assert compose(TAU, RHO) != compose(RHO, TAU)
    assert not np.array_equal(
        permutation_matrix(compose(TAU, RHO)),
        permutation_matrix(compose(RHO, TAU)),
    )


def test_centered_basis_has_zero_mean_and_identity_covariance():
    values = centered_basis()
    assert np.allclose(values.mean(axis=0), 0.0, atol=1e-12, rtol=0.0)
    assert np.allclose(
        values.T @ values / len(values), np.eye(WIDTH), atol=1e-12, rtol=0.0
    )


def test_fixed_point_census_for_odd_dihedral_group():
    histogram: dict[int, int] = {}
    for group in group_elements():
        fixed = fixed_points(group)
        histogram[fixed] = histogram.get(fixed, 0) + 1
    assert histogram == {WIDTH: 1, 0: WIDTH - 1, 1: WIDTH}


def test_relaxed_diagonal_formula_for_every_group():
    inputs = centered_basis()
    errors = []
    for group in group_elements():
        target = apply_group(inputs, group)
        observed = inputs * optimal_diagonal(group)[None, :]
        error = normalized_mse(target, observed)
        assert abs(error - (1.0 - fixed_points(group) / WIDTH)) <= 1e-12
        errors.append(error)
    assert abs(float(np.mean(errors)) - 108.0 / 109.0) <= 1e-12


def test_constructive_routed_witness_is_exact():
    inputs = centered_basis()
    for group in group_elements():
        observed = apply_group(inputs, fold_word(canonical_word(group)))
        assert np.array_equal(observed, apply_group(inputs, group))


def test_identity_and_state_ledger():
    assert fold_word(()) == IDENTITY
    assert WIDTH * 2 + 2 == 220


def test_frozen_preflight_passes():
    result = run()
    assert result["status"] == "PASS"
    assert all(result["gates"].values())
