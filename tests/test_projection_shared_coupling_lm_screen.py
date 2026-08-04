from pathlib import Path
import sys

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

sys.path.insert(0, str(Path(__file__).parents[1]))

from experiments.projection_shared_coupling_lm_screen import BOUND, VALID_ARMS


def test_frozen_arms_and_bound():
    assert VALID_ARMS == (
        "raw_baseline",
        "folded_null",
        "self_hinge",
        "one_way",
        "two_way",
    )
    assert BOUND == 0.25
