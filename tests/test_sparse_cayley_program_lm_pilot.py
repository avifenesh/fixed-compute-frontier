from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "sparse_cayley_program_lm_pilot.py"
)
SPEC = importlib.util.spec_from_file_location("sparse_cayley_lm", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parameter_match_and_model_reduction() -> None:
    device = torch.device("cpu")
    full, _ = MODULE.build_arm(device, "full_swiglu", MODULE.SEED)
    narrow, _ = MODULE.build_arm(device, "narrow_swiglu", MODULE.SEED)
    candidate, modules = MODULE.build_arm(device, "dynamic_cayley", MODULE.SEED)
    full_total = sum(parameter.numel() for parameter in full.parameters())
    candidate_total = sum(parameter.numel() for parameter in candidate.parameters())
    narrow_ffn = sum(
        sum(parameter.numel() for parameter in layer.mlp.parameters())
        for layer in narrow.model.layers
    )
    candidate_ffn = sum(
        sum(parameter.numel() for parameter in layer.mlp.parameters())
        for layer in candidate.model.layers
    )
    assert candidate_ffn == narrow_ffn == 12 * 36_864
    assert candidate_total <= 0.70 * full_total
    assert len(modules) == 12


def test_static_and_dynamic_models_have_equal_parameters() -> None:
    dynamic, _ = MODULE.build_arm(torch.device("cpu"), "dynamic_cayley", MODULE.SEED)
    static, _ = MODULE.build_arm(torch.device("cpu"), "static_cayley", MODULE.SEED)
    assert sum(p.numel() for p in dynamic.parameters()) == sum(
        p.numel() for p in static.parameters()
    )
