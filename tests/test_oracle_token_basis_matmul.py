import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments" / "oracle_token_basis_matmul.py"
SPEC = importlib.util.spec_from_file_location("oracle_token_basis_matmul", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_cost_ratio_counts_both_contractions():
    assert MODULE.ideal_cost_ratio(64, 256, 1024) == 0.3125


def test_minimum_rank_from_energy():
    energy = torch.tensor([81.0, 16.0, 3.0])
    assert MODULE.minimum_rank_from_row_energy(energy, 0.5) == 1
    assert MODULE.minimum_rank_from_row_energy(energy, 0.2) == 2


def test_exact_rank_two_world():
    if not torch.cuda.is_available():
        return
    generator = torch.Generator(device="cuda").manual_seed(7)
    c = torch.randn(16, 2, generator=generator, device="cuda")
    basis = torch.randn(2, 12, generator=generator, device="cuda")
    weight = torch.randn(12, 20, generator=generator, device="cuda")
    x = c @ basis
    y = x @ weight
    metrics = MODULE.sequence_rank_metrics(x, y, errors=(1e-4,))
    row = metrics["error_0p0001"]
    assert row["input_basis_rank"] == 2
    assert row["output_oracle_rank"] == 2

