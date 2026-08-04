from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from experiments.inplace_hinge_reduction_algebra import dense, midpoint_hinge, run


def test_exact_dense_endpoint() -> None:
    a = np.arange(-16, 17, dtype=np.float64)
    b = np.arange(16, -17, -1, dtype=np.float64)
    np.testing.assert_array_equal(midpoint_hinge(a, b, 0.0), dense(a, b))


def test_partial_history_is_strictly_more_informative() -> None:
    a = np.array([1.0, 2.0])
    b = -a
    np.testing.assert_array_equal(dense(a, b), np.zeros(2))
    np.testing.assert_array_equal(midpoint_hinge(a, b, 1.0), np.array([1.0, 2.0]))


def test_midpoint_has_two_gradient_directions() -> None:
    result = run()
    assert result["side_gradient_span"]["midpoint_rank"] == 2
    assert result["side_gradient_span"]["final_only_rank"] == 1
    assert result["mode_bit_funding"]["extra_stored_parameters"] == 0
    assert result["mode_bit_funding"]["exact_to_float_tolerance"]
