from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from bounded_persistent_meta_learner_t70_smoke import StateTokenLearner  # noqa: E402


def test_causal_suffix_write_depends_on_observed_value() -> None:
    torch.manual_seed(1)
    model = StateTokenLearner(num_keys=4, num_values=3)
    state = model.fresh_state(1)
    key = torch.tensor([2])
    _, state_a = model.forward_segment(state, key, torch.tensor([0]))
    _, state_b = model.forward_segment(state, key, torch.tensor([1]))
    assert not torch.allclose(state_a, state_b)


def test_segment_shapes_and_gradients_cross_boundary() -> None:
    torch.manual_seed(2)
    model = StateTokenLearner(num_keys=4, num_values=3)
    state = model.fresh_state(5)
    keys = torch.tensor([0, 1, 2, 3, 0])
    values = torch.tensor([0, 1, 2, 0, 1])
    _, written_state = model.forward_segment(state, keys, values)
    logits, next_state = model.forward_segment(written_state, keys, None)
    logits.sum().backward()
    assert logits.shape == (5, 3)
    assert next_state.shape == (5, model.state_slots, model.d_model)
    assert model.value_embedding.weight.grad is not None
    assert model.value_embedding.weight.grad[:3].abs().sum().item() > 0
