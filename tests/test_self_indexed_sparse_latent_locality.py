import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments" / "self_indexed_sparse_latent_locality.py"
SPEC = importlib.util.spec_from_file_location("self_indexed_sparse_latent_locality", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_selected_layer_starts_are_three_adjacent_safe_pairs():
    assert MODULE.selected_layer_starts(24) == [5, 11, 17]
    assert all(layer + 1 < 24 for layer in MODULE.selected_layer_starts(24))


def test_recall_at_k():
    predicted = torch.tensor([[1, 2, 3], [9, 8, 7]], dtype=torch.int32)
    truth = torch.tensor([[1, 4, 3], [8, 6, 5]], dtype=torch.int32)
    recall = MODULE.recall_at_k(predicted, truth)
    torch.testing.assert_close(recall, torch.tensor([2 / 3, 1 / 3]))


def test_graph_predict_uses_source_conditioned_edges():
    if not torch.cuda.is_available():
        return
    source_ids = torch.tensor([[0, 1], [1, 2]], dtype=torch.int32)
    magnitudes = torch.ones(2, 2, dtype=torch.float16)
    neighbors = torch.tensor([[3], [4], [5], [0], [0], [0]], dtype=torch.int32, device="cuda")
    weights = torch.ones(6, 1, dtype=torch.float16, device="cuda")
    predicted = MODULE.graph_predict(
        source_ids, magnitudes, neighbors, weights, address_count=6, active_k=2
    )
    assert set(predicted[0].tolist()) == {3, 4}
    assert set(predicted[1].tolist()) == {4, 5}

