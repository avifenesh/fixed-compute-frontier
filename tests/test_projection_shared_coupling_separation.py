from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

from experiments.projection_shared_coupling_separation import run


def test_coupling_exactly_separates_from_linear_and_bounded_self_hinge():
    result = run()
    assert result["coupling_exact"]
    assert result["linear_gap"] > 0.2
    assert result["bounded_self_hinge_mse"] > 0.1
    assert result["coupling_separates_from_bounded_self_hinge"]
