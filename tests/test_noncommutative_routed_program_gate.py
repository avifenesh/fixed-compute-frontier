from __future__ import annotations

import unittest

import numpy as np

from experiments.noncommutative_routed_program_gate import (
    additive_operator,
    apply_rotation,
    expert_matrices,
    make_rotation_experts,
    rotation_matrix,
    sequential_operator,
)


class NoncommutativeRoutedProgramGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.first, self.second, self.angles = make_rotation_experts(12, 4, seed=9)
        self.matrices = expert_matrices(self.first, self.second, self.angles)

    def test_fast_rank_two_application_matches_dense_rotation(self) -> None:
        rng = np.random.default_rng(10)
        values = rng.normal(size=(12, 5))
        actual = apply_rotation(
            values, self.first[0], self.second[0], float(self.angles[0])
        )
        expected = rotation_matrix(
            self.first[0], self.second[0], float(self.angles[0])
        ) @ values
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_each_expert_and_composition_are_orthogonal(self) -> None:
        identity = np.eye(12)
        for matrix in self.matrices:
            np.testing.assert_allclose(matrix.T @ matrix, identity, atol=1e-12)
        composed = sequential_operator((0, 1, 3, 2), self.matrices)
        np.testing.assert_allclose(composed.T @ composed, identity, atol=1e-12)

    def test_additive_routing_is_order_invariant(self) -> None:
        left = additive_operator((0, 1, 2, 1), self.matrices)
        right = additive_operator((1, 2, 1, 0), self.matrices)
        np.testing.assert_allclose(left, right, atol=1e-12)

    def test_sequential_routing_is_order_sensitive(self) -> None:
        left = sequential_operator((0, 1, 2, 1), self.matrices)
        right = sequential_operator((1, 2, 1, 0), self.matrices)
        self.assertGreater(np.linalg.norm(left - right), 1e-3)


if __name__ == "__main__":
    unittest.main()
