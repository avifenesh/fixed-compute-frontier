from __future__ import annotations

import torch

from experiments.cayley_program_tree import exact_ledger
from experiments.cayley_program_tree_lm_pilot import build_candidate


def test_candidate_model_matches_frozen_parameter_ledger() -> None:
    model, modules = build_candidate(torch.device("cpu"))
    ffn_parameters = sum(module.parameter_count() for module in modules)
    assert ffn_parameters == 12 * exact_ledger()["candidate_parameters"]
    assert ffn_parameters == 14_154_996
    assert sum(parameter.numel() for parameter in model.parameters()) == 37_752_948


def test_all_layers_are_program_trees() -> None:
    model, modules = build_candidate(torch.device("cpu"))
    assert len(modules) == 12
    assert all(layer.mlp is module for layer, module in zip(model.model.layers, modules, strict=True))
