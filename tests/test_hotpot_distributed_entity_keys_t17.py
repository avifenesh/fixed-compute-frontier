from __future__ import annotations

import inspect

import torch

from experiments import hotpot_distributed_entity_keys_t17 as experiment


def test_key_training_cannot_accept_questions_or_answers() -> None:
    assert tuple(inspect.signature(experiment.train_keys).parameters) == (
        "canonical",
        "train_views",
        "assignments",
        "device",
    )
    source = inspect.getsource(experiment.train_keys)
    assert "question" not in source
    assert "answer" not in source


def test_frozen_physical_ledger() -> None:
    assert experiment.LAYERS == (7, 8, 9)
    assert experiment.KEY_SEED == 9_323
    assert experiment.KEY_STEPS == 800
    assert experiment.KEY_BATCH == 256
    assert experiment.KEY_LR == 0.003
    assert experiment.TEMPERATURE == 0.05
    assert experiment.KEY_SCALARS == 923_520


def test_physical_scores_use_each_rows_assigned_layer() -> None:
    vectors = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]],
            [[0.0, 4.0], [5.0, 0.0], [0.0, 6.0]],
        ]
    )
    keys = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    assignments = torch.tensor([0, 1, 2])
    scores = experiment.physical_scores(vectors, keys, assignments)
    assert torch.equal(scores, torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))


def test_selected_interfaces_equal_actual_ffn_gate_inputs() -> None:
    model = experiment.t10.core.build_model(941, torch.device("cpu")).eval()
    tokens = torch.tensor([[11, 12, 13, 14]], dtype=torch.long)
    captured: list[torch.Tensor] = []
    hooks = []
    for layer in experiment.LAYERS:
        hooks.append(
            model.blocks[layer].gate.register_forward_pre_hook(
                lambda _module, arguments: captured.append(arguments[0].detach().clone())
            )
        )
    with torch.inference_mode():
        model.hidden(tokens)
    for hook in hooks:
        hook.remove()
    with torch.inference_mode():
        direct = experiment.selected_ffn_interfaces(model, tokens)
    assert len(captured) == len(experiment.LAYERS)
    assert torch.equal(direct, torch.stack(captured, dim=1))


def test_t16_integrity_anchors() -> None:
    assert experiment.T16_RESULT_SHA256 == (
        "f3bb7cc9b2182334e5ec9faa1dbe6b779a38491e7fb06baa1ef0b51db620a917"
    )
    assert experiment.T16_PROJECTION_SHA256 == (
        "70861caae201e145e156bcadbe2330990f7a09ab98b91f912623b9cd2d29b08c"
    )
