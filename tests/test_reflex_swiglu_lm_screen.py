import hashlib
import json
from argparse import Namespace

import torch
import torch.nn as nn

from experiments.reflex_swiglu_lm_screen import (
    DATASET,
    DATASET_CONFIG,
    DATASET_REVISION,
    MODEL,
    MODEL_REVISION,
    ScreenMLP,
    decide,
    effective_alpha,
    validate_data_ledger,
    validate_experiment_protocol,
    validate_serving_ledger,
)


class DummyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 6, bias=False)
        self.up_proj = nn.Linear(4, 6, bias=False)
        self.down_proj = nn.Linear(6, 4, bias=False)
        self.act_fn = nn.SiLU()


def test_all_arms_share_exact_zero_endpoint_and_parameter_count():
    torch.manual_seed(3)
    reference = DummyMLP()
    state = reference.state_dict()
    x = torch.randn(2, 4)
    expected = reference.down_proj(reference.act_fn(reference.gate_proj(x)) * reference.up_proj(x))
    counts = []
    for arm in ("baseline", "reflex", "reflex_detach", "gate_bias", "gate_temperature"):
        original = DummyMLP()
        original.load_state_dict(state)
        module = ScreenMLP(original, arm)
        counts.append(sum(parameter.numel() for parameter in module.parameters()))
        torch.testing.assert_close(module(x), expected, rtol=0.0, atol=0.0)
    assert len(set(counts)) == 1


def test_reflex_shared_gradients_match_at_zero_and_beta_gradient_is_live():
    torch.manual_seed(5)
    state = DummyMLP().state_dict()
    plain_source = DummyMLP(); plain_source.load_state_dict(state)
    reflex_source = DummyMLP(); reflex_source.load_state_dict(state)
    plain = ScreenMLP(plain_source, "baseline")
    reflex = ScreenMLP(reflex_source, "reflex")
    x = torch.randn(3, 4)
    plain(x).square().sum().backward()
    reflex(x).square().sum().backward()
    for name in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"):
        torch.testing.assert_close(
            dict(plain.named_parameters())[name].grad,
            dict(reflex.named_parameters())[name].grad,
            rtol=0.0,
            atol=0.0,
        )
    assert reflex.beta.grad is not None
    assert float(reflex.beta.grad.abs().max()) > 0.0


def test_effective_alpha_is_bounded_and_unit_slope_at_zero():
    beta = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    alpha = effective_alpha(beta)
    alpha.backward()
    assert beta.grad == 1.0
    extremes = effective_alpha(torch.tensor([-100.0, 100.0]))
    assert bool((extremes.abs() <= 2.0).all())


def test_decision_requires_reflex_to_beat_baseline_and_controls():
    def arm(name, terminal, mid):
        return {
            "arm": name,
            "total_parameters": 100,
            "trainable_parameters": 100,
            "train": {
                "steps": 640,
                "nonfinite": False,
                "max_preclip_gradient_norm": 2.0,
                "max_loss": 4.0,
            },
            "activation_diagnostics": {"alpha_saturation_fraction": 0.0},
            "evaluations": {
                "0": {"loss": 3.0, "per_batch_loss": [3.0, 3.0, 3.0]},
                "160": {"loss": mid, "per_batch_loss": [mid, mid, mid]},
                "640": {"loss": terminal, "per_batch_loss": [terminal, terminal, terminal]},
            },
        }

    results = {
        "baseline": arm("baseline", 3.0, 3.1),
        "reflex": arm("reflex", 2.99, 3.09),
        "reflex_detach": arm("reflex_detach", 2.98, 3.08),
        "gate_bias": arm("gate_bias", 2.99, 3.09),
        "gate_temperature": arm("gate_temperature", 2.995, 3.095),
    }
    decision = decide(
        results,
        data_protocol_valid=True,
        serving_gate_pass=True,
        experiment_protocol_valid=True,
    )
    assert decision["advance_to_replication"]
    results["reflex_detach"]["activation_diagnostics"]["alpha_saturation_fraction"] = 0.02
    assert not decide(
        results,
        data_protocol_valid=True,
        serving_gate_pass=True,
        experiment_protocol_valid=True,
    )["advance_to_replication"]
    results["reflex_detach"]["activation_diagnostics"]["alpha_saturation_fraction"] = 0.0
    results["gate_bias"]["evaluations"]["640"]["loss"] = 2.97
    assert not decide(
        results,
        data_protocol_valid=True,
        serving_gate_pass=True,
        experiment_protocol_valid=True,
    )["advance_to_replication"]


def test_data_ledger_verifies_content_not_only_manifest_hash(tmp_path):
    train = tmp_path / "train.bin"
    validation = tmp_path / "validation.bin"
    train.write_bytes(bytes(range(16)))
    validation.write_bytes(bytes(range(16, 32)))
    manifest = {
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "dataset": DATASET,
        "dataset_config": DATASET_CONFIG,
        "dataset_revision": DATASET_REVISION,
        "sequence_length": 3,
        "dtype": "uint16",
        "split_rule": "validation iff uint64_be(sha256(id)[:8]) mod 10 == 0",
        "train_sequences": 2,
        "validation_sequences": 2,
        "files": {
            "train": {"bytes": 16, "sha256": hashlib.sha256(train.read_bytes()).hexdigest()},
            "validation": {"bytes": 16, "sha256": hashlib.sha256(validation.read_bytes()).hexdigest()},
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert validate_data_ledger(path, train, validation, 3)["valid"]
    train.write_bytes(b"x" * 16)
    try:
        validate_data_ledger(path, train, validation, 3)
    except ValueError:
        pass
    else:
        raise AssertionError("mutated token file was accepted")


def test_serving_ledger_enforces_key_cell_ratios(tmp_path):
    path = tmp_path / "serving.json"
    payload = {
        "all_gates_pass": True,
        "device": "NVIDIA H100 80GB HBM3",
        "cells": [
            {"batch": batch, "same_width_over_baseline_median": 1.01, "equal_parameter_over_baseline_median": 1.015}
            for batch in (1, 8)
        ],
    }
    path.write_text(json.dumps(payload))
    assert validate_serving_ledger(path)["valid"]
    payload["cells"][0]["same_width_over_baseline_median"] = 1.03
    path.write_text(json.dumps(payload))
    try:
        validate_serving_ledger(path)
    except ValueError:
        pass
    else:
        raise AssertionError("slow serving result was accepted")


def test_exact_protocol_validator_accepts_only_frozen_h100_defaults():
    args = Namespace(
        sequence_length=512,
        micro_batch_size=32,
        gradient_accumulation=2,
        eval_batch_size=32,
        eval_batches=128,
        steps=640,
        eval_steps=[40, 160, 640],
        warmup_steps=32,
        shared_lr=1e-4,
        beta_lr=5e-4,
        weight_decay=0.1,
        gradient_clip=1.0,
        seed=101,
        compile=False,
        save_checkpoints=True,
        strict_protocol=True,
    )
    train_file = Namespace(sequence_count=40_960)
    validation_file = Namespace(sequence_count=4_096)
    assert validate_experiment_protocol(
        args,
        ("baseline", "reflex", "reflex_detach", "gate_bias", "gate_temperature"),
        train_file,
        validation_file,
    )["valid"]
    args.steps = 639
    try:
        validate_experiment_protocol(
            args,
            ("baseline", "reflex", "reflex_detach", "gate_bias", "gate_temperature"),
            train_file,
            validation_file,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("modified training horizon was accepted")
