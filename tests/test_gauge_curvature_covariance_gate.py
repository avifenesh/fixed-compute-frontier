from __future__ import annotations

import unittest

import numpy as np

from experiments.gauge_curvature_covariance_gate import (
    bilinear_gradients,
    bilinear_logits,
    covariance_examples,
    covariance_features,
    fit_logistic,
    orthogonal,
    square_gradients,
    square_logits,
)


class GaugeCurvatureCovarianceGateTests(unittest.TestCase):
    def test_paired_bags_have_exact_zero_linear_mean(self) -> None:
        rng = np.random.default_rng(7)
        rotation = orthogonal(rng, 16)
        mean, covariance, labels = covariance_examples(rng, rotation, 64)
        self.assertEqual(mean.shape, (64, 16))
        self.assertEqual(covariance.shape, (64, 16, 16))
        self.assertEqual(labels.sum(), 32)
        self.assertEqual(float(np.max(np.abs(mean))), 0.0)

    def test_zero_mean_linear_and_post_features_cannot_leak_class(self) -> None:
        rng = np.random.default_rng(11)
        mean, _, labels = covariance_examples(rng, np.eye(16), 256)
        result = fit_logistic(
            mean[:96],
            labels[:96],
            mean[96:160],
            labels[96:160],
            mean[160:],
            labels[160:],
        )
        self.assertLessEqual(abs(float(result["accuracy"]) - 0.5), 0.08)

    def test_gfqv_scalar_readout_spans_off_diagonal_covariance(self) -> None:
        rng = np.random.default_rng(13)
        _, covariance, _ = covariance_examples(rng, orthogonal(rng, 16), 32)
        rows, columns = np.triu_indices(16, k=1)
        coefficients = rng.normal(size=rows.size)
        gate = np.zeros((16, 16))
        for row, column, coefficient in zip(rows, columns, coefficients, strict=True):
            gate[row, column] = coefficient / 2.0
            gate[column, row] = coefficient / 2.0
        gfqv_quadratic = np.einsum("nij,ij->n", covariance, gate, optimize=True)
        direct = covariance_features(covariance, diagonal=False) @ coefficients
        np.testing.assert_allclose(gfqv_quadratic, direct, atol=1e-11, rtol=1e-11)

    def test_square_projection_gradient_matches_finite_difference(self) -> None:
        rng = np.random.default_rng(17)
        mean, covariance, labels = covariance_examples(rng, np.eye(4), 8)
        params = {
            "projection": rng.normal(scale=0.2, size=(4, 4)) * (1.0 - np.eye(4)),
            "readout": rng.normal(size=4),
            "linear": rng.normal(size=4),
            "bias": np.array([0.1]),
        }
        gradient = square_gradients(mean, covariance, labels, params)["projection"]
        row, column = 0, 1
        epsilon = 1e-6

        def loss() -> float:
            logits = square_logits(mean, covariance, params)
            return float(np.mean(np.logaddexp(0.0, logits) - labels * logits))

        original = params["projection"][row, column]
        params["projection"][row, column] = original + epsilon
        positive = loss()
        params["projection"][row, column] = original - epsilon
        negative = loss()
        params["projection"][row, column] = original
        self.assertAlmostEqual(
            gradient[row, column], (positive - negative) / (2.0 * epsilon), places=6
        )

    def test_bilinear_gradient_matches_finite_difference(self) -> None:
        rng = np.random.default_rng(19)
        mean, covariance, labels = covariance_examples(rng, np.eye(4), 8)
        params = {
            "left": rng.normal(scale=0.2, size=(4, 2)),
            "right": rng.normal(scale=0.2, size=(4, 2)),
            "readout": rng.normal(size=2),
            "linear": rng.normal(size=4),
            "bias": np.array([-0.2]),
        }
        gradient = bilinear_gradients(mean, covariance, labels, params)["left"]
        row, column = 1, 0
        epsilon = 1e-6

        def loss() -> float:
            logits = bilinear_logits(mean, covariance, params)
            return float(np.mean(np.logaddexp(0.0, logits) - labels * logits))

        original = params["left"][row, column]
        params["left"][row, column] = original + epsilon
        positive = loss()
        params["left"][row, column] = original - epsilon
        negative = loss()
        params["left"][row, column] = original
        self.assertAlmostEqual(
            gradient[row, column], (positive - negative) / (2.0 * epsilon), places=6
        )


if __name__ == "__main__":
    unittest.main()
