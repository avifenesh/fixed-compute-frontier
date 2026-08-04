import itertools
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from experiments.projection_shared_coupling_algebra import (
    coupling,
    inverse_coupling,
    one_way_jacobian,
)


def test_identity_and_inverse_are_exact_to_roundoff():
    rng = np.random.default_rng(11)
    a = rng.normal(size=16)
    b = rng.normal(size=16)
    alpha = rng.normal(size=16)
    beta = rng.normal(size=16)
    zero_a, zero_b = coupling(a, b, np.zeros(16), np.zeros(16))
    np.testing.assert_array_equal(zero_a, a)
    np.testing.assert_array_equal(zero_b, b)

    a_prime, b_prime = coupling(a, b, alpha, beta)
    recovered_a, recovered_b = inverse_coupling(
        a_prime, b_prime, alpha, beta
    )
    np.testing.assert_allclose(recovered_a, a, rtol=0, atol=2e-15)
    np.testing.assert_allclose(recovered_b, b, rtol=0, atol=2e-15)


def test_one_way_coupling_has_unit_determinant_and_exponential_jacobians():
    alpha = np.asarray([0.25, -0.5, 1.5, 2.0])
    matrices = [
        one_way_jacobian(np.asarray(signs), alpha)
        for signs in itertools.product((-1.0, 1.0), repeat=4)
    ]
    assert len({matrix.tobytes() for matrix in matrices}) == 16
    for matrix in matrices:
        assert np.isclose(np.linalg.det(matrix), 1.0, rtol=0, atol=2e-15)


def test_rmsnorm_gain_can_be_folded_into_multiple_immediate_consumers():
    rng = np.random.default_rng(13)
    normalized = rng.normal(size=(9, 12))
    gain = rng.normal(size=12)
    for output_width in (12, 36, 48):
        weight = rng.normal(size=(12, output_width))
        expected = (normalized * gain) @ weight
        actual = normalized @ (gain[:, None] * weight)
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_linear_projection_after_coupling_is_not_an_ordinary_linear_map():
    # Selecting b' gives f(a,b)=b+|a|.  Its slope in a flips across zero, so no
    # single ordinary linear projection can match all four samples.
    samples = np.asarray([[1.0, 0.0], [-1.0, 0.0], [2.0, 1.0], [-2.0, 1.0]])
    target = samples[:, 1] + np.abs(samples[:, 0])
    fitted, *_ = np.linalg.lstsq(samples, target, rcond=None)
    error = np.max(np.abs(samples @ fitted - target))
    assert error >= 1.0
