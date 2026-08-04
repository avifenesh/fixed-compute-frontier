from __future__ import annotations

import unittest

import numpy as np

from experiments.sparse_precision_dense_mixer_gate import (
    apply_operator,
    best_global_sparse_error,
    dense_operator,
    haar_orthogonal,
    make_permutations,
    neumann_apply,
    neumann_relative_bound,
    required_neumann_order,
    required_rank_fraction,
    theoretical_rank_fraction_lower_bound,
)


class SparsePrecisionDenseMixerGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.width = 48
        self.permutations = make_permutations(self.width, 3, seed=123)
        self.operator = dense_operator(self.permutations)

    def test_permutation_average_is_symmetric_stochastic_contraction(self) -> None:
        np.testing.assert_allclose(self.operator, self.operator.T, atol=1e-15)
        np.testing.assert_allclose(self.operator.sum(axis=1), 1.0, atol=1e-15)
        self.assertLessEqual(
            np.max(np.abs(np.linalg.eigvalsh(self.operator))), 1.0 + 1e-12
        )

    def test_gather_application_matches_materialized_operator(self) -> None:
        rng = np.random.default_rng(5)
        values = rng.normal(size=(self.width, 4))
        np.testing.assert_allclose(
            apply_operator(self.permutations, values),
            self.operator @ values,
            atol=1e-14,
        )

    def test_neumann_error_obeys_uniform_relative_bound(self) -> None:
        rho = 0.5
        order = required_neumann_order(rho, 0.01)
        rng = np.random.default_rng(6)
        values = rng.normal(size=(self.width, 3))
        exact = np.linalg.solve(np.eye(self.width) - rho * self.operator, values)
        approximate = neumann_apply(self.permutations, values, rho, order)
        error = np.linalg.norm(exact - approximate) / np.linalg.norm(exact)
        self.assertLessEqual(error, neumann_relative_bound(rho, order) + 1e-12)
        self.assertLessEqual(neumann_relative_bound(rho, order), 0.01)
        if order:
            self.assertGreater(neumann_relative_bound(rho, order - 1), 0.01)

    def test_inverse_is_full_rank_and_requires_nearly_full_rank_approximation(self) -> None:
        rho = 0.5
        eigenvalues = np.linalg.eigvalsh(self.operator)
        mixer_values = 1.0 / (1.0 - rho * eigenvalues)
        rank, fraction = required_rank_fraction(np.abs(mixer_values), 0.10)
        self.assertEqual(np.count_nonzero(mixer_values), self.width)
        self.assertGreaterEqual(
            fraction, theoretical_rank_fraction_lower_bound(rho, 0.10) - 1e-12
        )
        self.assertGreaterEqual(rank, int(0.90 * self.width))

    def test_haar_rotation_preserves_spectrum_but_destroys_sparse_precision(self) -> None:
        rho = 0.5
        precision = np.eye(self.width) - rho * self.operator
        interaction = np.eye(self.width) - precision
        budget = np.count_nonzero(np.abs(interaction) > 1e-15)
        orthogonal = haar_orthogonal(self.width, seed=7)
        rotated = orthogonal @ precision @ orthogonal.T
        rotated_interaction = np.eye(self.width) - rotated
        np.testing.assert_allclose(
            np.linalg.eigvalsh(precision), np.linalg.eigvalsh(rotated), atol=1e-12
        )
        self.assertLess(best_global_sparse_error(interaction, budget), 1e-12)
        self.assertGreater(best_global_sparse_error(rotated_interaction, budget), 0.25)


if __name__ == "__main__":
    unittest.main()
