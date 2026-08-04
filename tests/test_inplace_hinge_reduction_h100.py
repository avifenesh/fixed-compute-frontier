import importlib.util
from pathlib import Path
import sys

import pytest


pytest.importorskip("torch")
pytest.importorskip("triton")


def _load_module():
    path = Path(__file__).parents[1] / "experiments" / "inplace_hinge_reduction_h100.py"
    sys.path.insert(0, str(path.parent))
    specification = importlib.util.spec_from_file_location(
        "inplace_hinge_reduction_h100", path
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_frozen_configuration_grid() -> None:
    module = _load_module()
    grid = module.configurations()
    assert len(grid) == 8
    assert {configuration["block_n"] for configuration in grid} == {64, 128}
    assert {configuration["num_warps"] for configuration in grid} == {4, 8}
    assert {configuration["num_stages"] for configuration in grid} == {3, 4}
