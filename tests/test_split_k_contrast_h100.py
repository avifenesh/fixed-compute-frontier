import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from experiments.split_k_contrast_h100 import configurations


def test_frozen_configuration_grid() -> None:
    grid = configurations()
    assert len(grid) == 8
    assert {configuration["block_n"] for configuration in grid} == {64, 128}
    assert {configuration["num_warps"] for configuration in grid} == {4, 8}
    assert {configuration["num_stages"] for configuration in grid} == {3, 4}
    assert all(configuration["block_m"] == 64 for configuration in grid)
    assert all(configuration["block_k"] == 32 for configuration in grid)

