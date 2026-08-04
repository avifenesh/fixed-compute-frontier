from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from experiments import shared_score_heterogeneous_attention_t84 as t84
from experiments import shared_score_heterogeneous_attention_t84_controller as controller
from experiments.shared_score_heterogeneous_attention_t84_decoder import decode_gate


def test_frozen_protocol_hashes() -> None:
    for path, expected in t84.EXPECTED_HASHES.items():
        assert t84.sha256_file(path) == expected


def test_mixer_payload_is_exact_and_orthogonal() -> None:
    mixer = t84.load_mixer()
    assert mixer.shape == (14, 14)
    assert abs(mixer.T @ mixer - __import__("numpy").eye(14)).max() < 1e-14


def test_parameter_counts_and_paths() -> None:
    for architecture in t84.ARCHITECTURES:
        if architecture == "soft_time_matched":
            model = t84.build_model(architecture, 843001, r_attention=2, r_ffn=3)
        else:
            model = t84.build_model(architecture, 843001)
        expected = 17090 if architecture == "deepsets_conditioned" else 17190
        assert t84.parameter_count(model) == expected


def transformer_shapes(*, tropical: bool = False) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {"input.weight": (32, 14)}
    for block in range(2):
        prefix = f"block.{block}"
        shapes[f"{prefix}.norm_attn.weight"] = (32,)
        if tropical:
            for name in ("query_trop.W", "key_trop.W", "value_trop.W"):
                shapes[f"{prefix}.attn.{name}"] = (8, 8)
            shapes[f"{prefix}.attn.out.weight"] = (32, 32)
            ffn_width = 109
        else:
            for name in ("q", "k", "v", "o"):
                shapes[f"{prefix}.attn.{name}.weight"] = (32, 32)
            ffn_width = 64
        shapes[f"{prefix}.norm_ffn.weight"] = (32,)
        shapes[f"{prefix}.ffn.up.weight"] = (ffn_width, 32)
        shapes[f"{prefix}.ffn.down.weight"] = (32, ffn_width)
    shapes["final_norm.weight"] = (32,)
    for label in ("ys", "ym", "y"):
        shapes[f"readout.{label}.weight"] = (2, 32)
        shapes[f"readout.{label}.bias"] = (2,)
    return shapes


def test_complete_named_parameter_grammar_and_shapes() -> None:
    for architecture in t84.ARCHITECTURES:
        model = t84.build_model(
            architecture,
            843001,
            r_attention=2 if architecture == "soft_time_matched" else 1,
            r_ffn=3 if architecture == "soft_time_matched" else 1,
        )
        actual = {name: tuple(parameter.shape) for name, parameter in model.named_parameters()}
        if architecture == "deepsets_conditioned":
            expected = {
                "phi1.weight": (74, 28),
                "phi2.weight": (74, 74),
                "pool_proj.weight": (32, 162),
                "norm.weight": (32,),
                "ffn.up.weight": (64, 32),
                "ffn.down.weight": (32, 64),
                "final_norm.weight": (32,),
                **{
                    f"readout.{label}.{kind}": shape
                    for label in ("ys", "ym", "y")
                    for kind, shape in (("weight", (2, 32)), ("bias", (2,)))
                },
            }
        else:
            expected = transformer_shapes(tropical=architecture == "tropical_official")
        assert actual == expected


def test_probe_parameter_grammar() -> None:
    surface = controller.SurfaceLinear(141)
    query = controller.ProbeMLP(14)
    assert {name: tuple(value.shape) for name, value in surface.named_parameters()} == {
        "linear.weight": (2, 141),
        "linear.bias": (2,),
    }
    assert {name: tuple(value.shape) for name, value in query.named_parameters()} == {
        "fc1.weight": (64, 14),
        "fc1.bias": (64,),
        "fc2.weight": (2, 64),
        "fc2.bias": (2,),
    }


def test_training_batches_are_balanced_homogeneous_and_paired() -> None:
    first = t84.training_batch(842100, 0)
    second = t84.training_batch(842100, 0)
    assert first[2] == second[2]
    assert first[0].shape == (128, first[2] + 1, 14)
    assert first[0].equal(second[0])
    pairs = first[1]["ys"] * 2 + first[1]["ym"]
    assert pairs.bincount(minlength=4).tolist() == [32, 32, 32, 32]


def test_mask_has_no_padding_and_query_excludes_self() -> None:
    for n in range(4, 9):
        mask = t84.causal_mask(n)
        assert mask.shape == (n + 1, n + 1)
        assert mask[n, :n].all()
        assert not mask[n, n]
        assert mask.any(dim=-1).all()


def test_trace_free_forward_dispatches_only_required_reducers(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"soft": 0, "max": 0}
    original_soft = t84.soft_reduce
    original_max = t84.maxplus_reduce

    def counted_soft(*args, **kwargs):
        calls["soft"] += 1
        return original_soft(*args, **kwargs)

    def counted_max(*args, **kwargs):
        calls["max"] += 1
        return original_max(*args, **kwargs)

    monkeypatch.setattr(t84, "soft_reduce", counted_soft)
    monkeypatch.setattr(t84, "maxplus_reduce", counted_max)
    tokens = __import__("torch").zeros((2, 9, 14), dtype=__import__("torch").float32)
    mask = t84.causal_mask(8)

    soft = t84.build_model("soft_standard", 843001).eval()
    assert set(soft(tokens, mask)) == {"ys", "ym", "y"}
    assert calls == {"soft": 2, "max": 0}

    calls.update(soft=0, max=0)
    candidate = t84.build_model("candidate", 843001).eval()
    assert set(candidate(tokens, mask)) == {"ys", "ym", "y"}
    assert calls == {"soft": 2, "max": 2}


def test_sattolo_is_exact_derangement() -> None:
    permutation = t84.exact_sattolo(850000 + 100 * 842100 + 10)
    assert sorted(permutation.tolist()) == list(range(10_000))
    assert all(index != donor for index, donor in enumerate(permutation.tolist()))


def synthetic_record(candidate_error: float, control_error: float | None = None) -> dict:
    error = candidate_error if control_error is None else control_error
    metrics = {
        name: {"ys": error, "ym": error, "y": error}
        for name in controller.SLICE_NAMES
    }
    diagnostics = {
        "lesions": {
            branch: {
                name: {"ys": 0.4, "ym": 0.4, "y": 0.4}
                for name in controller.OOD_NAMES
            }
            for branch in ("soft", "max")
        },
        "route_shuffles": {
            branch: {
                name: {"errors": {"ys": 0.4, "ym": 0.4, "y": 0.4}}
                for name in controller.OOD_NAMES
            }
            for branch in ("soft", "max")
        },
        "health": {"passed": True},
    }
    return {"metrics": metrics, "diagnostics": diagnostics}


def test_gate_family_has_exactly_106_statements() -> None:
    candidate = [synthetic_record(0.05) for _ in range(16)]
    gates = controller.candidate_gates(candidate)
    for name in controller.CONTROL_ORDER:
        control = [synthetic_record(0.05, 0.10) for _ in range(16)]
        gates.extend(controller.control_gates(name, candidate, control))
    assert len(gates) == 106
    assert len({gate["name"] for gate in gates}) == 106


def test_relative_gate_rejects_19_percent_and_accepts_20_percent() -> None:
    control = [synthetic_record(0.10, 0.10) for _ in range(16)]
    candidate_19 = [synthetic_record(0.081) for _ in range(16)]
    candidate_20 = [synthetic_record(0.080) for _ in range(16)]
    gate_19 = next(
        gate
        for gate in controller.control_gates("soft_standard", candidate_19, control)
        if gate["name"].endswith("macro.relative_20pct")
    )
    gate_20 = next(
        gate
        for gate in controller.control_gates("soft_standard", candidate_20, control)
        if gate["name"].endswith("macro.relative_20pct")
    )
    assert gate_19["passed"] is False
    assert gate_20["passed"] is True


@pytest.mark.parametrize(
    ("gate", "expected"),
    [
        ({"integrity_valid": False, "required_control_valid": True, "resource_valid": True, "control_floor_established": True, "all_decidable_gates_pass": True}, "PROTOCOL INVALID"),
        ({"integrity_valid": True, "required_control_valid": False, "resource_valid": True, "control_floor_established": True, "all_decidable_gates_pass": True}, "PROTOCOL INVALID"),
        ({"integrity_valid": True, "required_control_valid": True, "resource_valid": False, "control_floor_established": True, "all_decidable_gates_pass": True}, "RESOURCE-INCONCLUSIVE"),
        ({"integrity_valid": True, "required_control_valid": True, "resource_valid": True, "control_floor_established": False, "all_decidable_gates_pass": True}, "CONTROL FLOOR NOT ESTABLISHED"),
        ({"integrity_valid": True, "required_control_valid": True, "resource_valid": True, "control_floor_established": True, "all_decidable_gates_pass": False}, "STOP-REJECT"),
        ({"integrity_valid": True, "required_control_valid": True, "resource_valid": True, "control_floor_established": True, "all_decidable_gates_pass": True}, "CONTINUE"),
    ],
)
def test_decoder_priority(gate: dict[str, bool], expected: str) -> None:
    assert decode_gate(gate) == expected


def test_decoder_complete_32_input_word_mapping() -> None:
    keys = (
        "integrity_valid",
        "required_control_valid",
        "resource_valid",
        "control_floor_established",
        "all_decidable_gates_pass",
    )
    for values in itertools.product((False, True), repeat=5):
        gate = dict(zip(keys, values))
        if not gate["integrity_valid"] or not gate["required_control_valid"]:
            expected = "PROTOCOL INVALID"
        elif not gate["resource_valid"]:
            expected = "RESOURCE-INCONCLUSIVE"
        elif not gate["control_floor_established"]:
            expected = "CONTROL FLOOR NOT ESTABLISHED"
        elif not gate["all_decidable_gates_pass"]:
            expected = "STOP-REJECT"
        else:
            expected = "CONTINUE"
        assert decode_gate(gate) == expected
