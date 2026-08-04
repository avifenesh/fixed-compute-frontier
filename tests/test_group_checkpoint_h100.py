import importlib.util
from pathlib import Path
import sys

import pytest


pytest.importorskip("torch")
pytest.importorskip("triton")


def _load_module():
    path = Path(__file__).parents[1] / "experiments" / "group_checkpoint_h100.py"
    sys.path.insert(0, str(path.parent))
    specification = importlib.util.spec_from_file_location("group_checkpoint_h100", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_frozen_group_and_configuration_grid() -> None:
    module = _load_module()
    grid = module.configurations()
    assert module.GROUP_WIDTH == 64
    assert len(grid) == 8
    assert {configuration["block_n"] for configuration in grid} == {64, 128}
    assert all(configuration["block_n"] % module.GROUP_WIDTH == 0 for configuration in grid)
