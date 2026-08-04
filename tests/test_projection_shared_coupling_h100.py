from pathlib import Path
import sys

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

sys.path.insert(0, str(Path(__file__).parents[1]))

from experiments.projection_shared_coupling_h100 import HIDDEN, ROWS, WIDTHS


def test_frozen_projection_shared_coupling_shape_grid():
    assert HIDDEN == 4096
    assert ROWS == (1, 8, 64, 256, 1024)
    assert WIDTHS == (6144, 14336)
    assert HIDDEN % 2 == 0
