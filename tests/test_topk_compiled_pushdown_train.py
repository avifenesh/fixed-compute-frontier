import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.topk_compiled_pushdown_reference import ACTIONS
from experiments.topk_compiled_pushdown_train import (
    ARMS,
    AcquisitionModel,
    earliest_minimum_epoch,
    learning_rate,
    parameter_shapes,
    smoke_test,
    verify_manifest_sources,
    verify_preflight_receipt,
    verify_runtime,
)
from experiments.topk_compiled_pushdown_evaluator import _parameter_shapes


def test_all_arms_have_exact_parameter_count() -> None:
    for vocabulary_size in (4, 5, 6):
        expected = 193 * vocabulary_size + 17_588
        for arm in ARMS:
            shapes = parameter_shapes(vocabulary_size, arm)
            assert sum(math.prod(shape) for shape in shapes.values()) == expected
            model = AcquisitionModel(
                vocabulary_size=vocabulary_size, arm=arm, model_seed=1701
            )
            assert sum(parameter.numel() for parameter in model.parameters()) == expected


def test_trainer_and_sealed_evaluator_agree_on_checkpoint_schema() -> None:
    for vocabulary_size in (4, 5, 6):
        for arm in ARMS:
            assert parameter_shapes(vocabulary_size, arm) == _parameter_shapes(
                "D1", arm, vocabulary_size
            )


def test_common_initialization_is_name_addressed_across_arms() -> None:
    left = AcquisitionModel(vocabulary_size=4, arm="S32", model_seed=1701)
    right = AcquisitionModel(vocabulary_size=4, arm="MLP", model_seed=1701)
    for name in ("E", "W_e", "W_h", "W_r", "b_h", "W_y", "U_y", "b_y"):
        torch.testing.assert_close(getattr(left, name), getattr(right, name))


def test_every_arm_has_finite_forward_and_backward() -> None:
    sequence = [0, 2, 3, 2, 1]
    for arm in ARMS:
        model = AcquisitionModel(vocabulary_size=4, arm=arm, model_seed=1701)
        result = smoke_test(model, [sequence])
        assert result["status"] == "pass"
        assert math.isfinite(result["loss"])
        assert math.isfinite(result["gradient_norm"])


def test_learning_rate_hits_frozen_endpoints() -> None:
    assert learning_rate(1) > 0.0
    assert math.isclose(learning_rate(313), 3e-4)
    assert math.isclose(learning_rate(6_260), 3e-5)


def test_checkpoint_selection_uses_global_minimum_tolerance() -> None:
    losses = [2.0] * 20
    losses[0] = 1.0
    losses[1] = 1.0 - 9e-9
    losses[2] = 1.0 - 18e-9
    assert earliest_minimum_epoch(losses) == 2


def test_root_pop_is_masked_before_softmax_and_merges_are_exact() -> None:
    model = AcquisitionModel(vocabulary_size=4, arm="S32", model_seed=1701)
    with torch.no_grad():
        model.W_a.zero_()
        model.T.zero_()
    root = ()
    state = {(0, root): torch.tensor(0.4), (1, root): torch.tensor(0.6)}
    candidates, _ = model.expand_beam(
        state=state,
        h=torch.zeros(64),
        time_index=0,
        chronological_parent=root,
    )
    assert len(candidates) == 12
    for weight in candidates.values():
        torch.testing.assert_close(weight, torch.tensor(1.0 / 12.0))
    torch.testing.assert_close(
        torch.stack(list(candidates.values())).sum(), torch.tensor(1.0)
    )


def test_beam_ties_use_lexicographic_structural_keys() -> None:
    candidates = {
        (2, ()): torch.tensor(1.0),
        (0, ()): torch.tensor(1.0),
        (1, ()): torch.tensor(1.0),
    }
    selected = AcquisitionModel.select_beam(candidates, 2)
    assert list(selected) == [(0, ()), (1, ())]


def test_hard3_is_hard_forward_and_soft_backward() -> None:
    model = AcquisitionModel(vocabulary_size=2, arm="HARD3", model_seed=1701)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.E[0, 0] = 1.0
        model.W_e[0, 0] = 1.0
        model.W_v[0, 0] = 1.0
        model.U_y[1, 0] = 1.0
    loss = model.run_hard3([0, 1])
    payload = torch.tanh(torch.tensor(1.0))
    expected = F.cross_entropy(
        torch.tensor([[0.0, payload]]), torch.tensor([1])
    )
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert abs(float(model.T.grad[0, 3, 4])) > 1e-8


def test_link_chronology_uses_complete_structural_key_order() -> None:
    high_stack = ((2, 0),)
    low_stack = ((0, 0),)
    state = {
        (0, high_stack): torch.tensor(0.5),
        (2, low_stack): torch.tensor(0.5),
    }
    result = AcquisitionModel.next_chronological_parent(
        state=state,
        pushed={high_stack, low_stack},
        current=(),
    )
    assert result == low_stack


def test_sequence_loss_excludes_bos_and_includes_eos_target() -> None:
    model = AcquisitionModel(vocabulary_size=3, arm="MLP", model_seed=1701)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.b_y.copy_(torch.tensor([0.0, 1.0, 2.0]))
    loss = model.sequence_loss([0, 1, 2])
    expected = F.cross_entropy(model.b_y.unsqueeze(0), torch.tensor([1]))
    expected = expected + F.cross_entropy(
        model.b_y.unsqueeze(0), torch.tensor([2])
    )
    torch.testing.assert_close(loss, expected)


def test_runtime_must_match_frozen_manifest() -> None:
    manifest = {
        "environment": {
            "python": "0.0.0",
            "numpy": "0.0.0",
            "torch_installed": "0.0.0",
            "torch_required": "0.0.0",
        }
    }
    try:
        verify_runtime(manifest)
    except ValueError as error:
        assert "runtime mismatch" in str(error)
    else:
        raise AssertionError("mismatched runtime was accepted")


def test_training_manifest_requires_preflight_sources(tmp_path) -> None:
    old_labels = {
        "candidate",
        "evaluator",
        "evaluator_tests",
        "preregistration",
        "reference",
        "reference_tests",
        "training",
        "training_tests",
    }
    manifest = {"source_hashes": {label: {} for label in old_labels}}
    try:
        verify_manifest_sources(project_root=tmp_path, manifest=manifest)
    except ValueError as error:
        assert "source hashes incomplete" in str(error)
    else:
        raise AssertionError("training accepted a manifest without preflight sources")


def test_training_requires_the_hash_bound_passing_preflight_receipt(
    tmp_path: Path,
) -> None:
    source_hashes = {
        label: {"path": f"{label}.py", "sha256": label}
        for label in ("preflight", "preflight_tests", "training", "locked_evaluation")
    }
    receipt_path = tmp_path / "preflight.json"
    receipt = {
        "schema": "cpkv-topk-001-preflight-v1",
        "lineage": "CPKV-TOPK-001",
        "status": "pass",
        "seed": 1701,
        "base_sample_count": 512,
        "projection_limit_hours": 5.7,
        "families": {
            family: {
                "status": "pass",
                "projected_all_arm_cpu_hours": 5.0,
                "projection_limit_hours": 5.7,
                "actual_freeze_limit_hours": 6.0,
            }
            for family in ("D1", "A1", "A2")
        },
        "source_hashes": source_hashes,
    }

    def manifest_for_receipt() -> dict:
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        return {
            "source_hashes": source_hashes,
            "preflight_receipt": {
                "path": receipt_path.name,
                "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            },
        }

    verify_preflight_receipt(
        project_root=tmp_path, manifest=manifest_for_receipt()
    )
    receipt["status"] = "fail"
    try:
        verify_preflight_receipt(
            project_root=tmp_path, manifest=manifest_for_receipt()
        )
    except ValueError as error:
        assert "contract mismatch" in str(error)
    else:
        raise AssertionError("training accepted a failed preflight receipt")


class ScalarAcquisitionModel(AcquisitionModel):
    def expand_beam(
        self,
        *,
        state,
        h,
        time_index,
        chronological_parent,
    ):
        candidates = {}
        pushed = set()
        base_logits = self.W_a @ h
        for (q, stack), parent_weight in state.items():
            top_symbol = stack[-1][0] if stack else -1
            valid, probabilities = self.transition_probabilities(
                base_logits, q, top_symbol
            )
            for local_index, action_index in enumerate(valid):
                action = ACTIONS[action_index]
                if action.kind == "push":
                    parent = chronological_parent if self.arm == "LINK32" else stack
                    next_stack = parent + ((action.symbol, time_index),)
                    pushed.add(next_stack)
                elif action.kind == "pop":
                    next_stack = stack[:-1]
                else:
                    next_stack = stack
                key = (action.next_q, next_stack)
                contribution = parent_weight * probabilities[local_index]
                candidates[key] = candidates.get(key, 0.0) + contribution
        return candidates, pushed

    def expand_and_select_beam(
        self,
        *,
        state,
        h,
        time_index,
        chronological_parent,
        beam_size,
    ):
        candidates, pushed = self.expand_beam(
            state=state,
            h=h,
            time_index=time_index,
            chronological_parent=chronological_parent,
        )
        return self.select_beam(candidates, beam_size), pushed


def _assert_candidate_maps_close(vector_candidates, scalar_candidates) -> None:
    assert list(vector_candidates) == list(scalar_candidates)
    torch.testing.assert_close(
        torch.stack(list(vector_candidates.values())),
        torch.stack(list(scalar_candidates.values())),
        rtol=2e-6,
        atol=2e-7,
    )


def _assert_exhaustive_beam_expansions_match(
    vector: AcquisitionModel,
    scalar: ScalarAcquisitionModel,
    arm: str,
) -> None:
    beam = 3 if arm == "DIRECT3" else 32
    sequences = [
        [0, *bits, 1]
        for length in range(1, 7)
        for bits in (
            [2 + ((value >> shift) & 1) for shift in reversed(range(length))]
            for value in range(1 << length)
        )
    ]
    for tokens in sequences:
        vector_zero = vector.E.new_zeros(64)
        scalar_zero = scalar.E.new_zeros(64)
        vector_h = vector_zero
        scalar_h = scalar_zero
        vector_r = vector_zero
        scalar_r = scalar_zero
        vector_state = {(0, ()): vector.E.new_tensor(1.0)}
        scalar_state = {(0, ()): scalar.E.new_tensor(1.0)}
        vector_payloads = {}
        scalar_payloads = {}
        vector_chronology = scalar_chronology = ()
        for time_index, token in enumerate(tokens[:-1]):
            vector_h = vector.common_hidden(token, vector_h, vector_r)
            scalar_h = scalar.common_hidden(token, scalar_h, scalar_r)
            vector_raw_payload = vector.W_v @ vector_h
            scalar_raw_payload = scalar.W_v @ scalar_h
            vector_payloads[time_index] = vector_raw_payload / torch.clamp(
                torch.linalg.vector_norm(vector_raw_payload), min=1.0
            )
            scalar_payloads[time_index] = scalar_raw_payload / torch.clamp(
                torch.linalg.vector_norm(scalar_raw_payload), min=1.0
            )
            vector_candidates, vector_pushed = vector.expand_beam(
                state=vector_state,
                h=vector_h,
                time_index=time_index,
                chronological_parent=vector_chronology,
            )
            scalar_candidates, scalar_pushed = scalar.expand_beam(
                state=scalar_state,
                h=scalar_h,
                time_index=time_index,
                chronological_parent=scalar_chronology,
            )
            _assert_candidate_maps_close(vector_candidates, scalar_candidates)
            assert vector_pushed == scalar_pushed
            vector_state, optimized_pushed = vector.expand_and_select_beam(
                state=vector_state,
                h=vector_h,
                time_index=time_index,
                chronological_parent=vector_chronology,
                beam_size=beam,
            )
            scalar_state = scalar.select_beam(scalar_candidates, beam)
            assert optimized_pushed == scalar_pushed
            assert list(vector_state) == list(scalar_state)
            torch.testing.assert_close(
                torch.stack(list(vector_state.values())),
                torch.stack(list(scalar_state.values())),
                rtol=2e-6,
                atol=2e-7,
            )
            vector_chronology = vector.next_chronological_parent(
                state=vector_state,
                pushed=vector_pushed,
                current=vector_chronology,
            )
            scalar_chronology = scalar.next_chronological_parent(
                state=scalar_state,
                pushed=scalar_pushed,
                current=scalar_chronology,
            )
            assert vector_chronology == scalar_chronology
            vector_weights = torch.stack(list(vector_state.values()))
            scalar_weights = torch.stack(list(scalar_state.values()))
            vector_reads = torch.stack(
                [
                    vector_payloads[stack[-1][1]] if stack else vector_zero
                    for _, stack in vector_state
                ]
            )
            scalar_reads = torch.stack(
                [
                    scalar_payloads[stack[-1][1]] if stack else scalar_zero
                    for _, stack in scalar_state
                ]
            )
            vector_r = (vector_weights.unsqueeze(1) * vector_reads).sum(dim=0)
            scalar_r = (scalar_weights.unsqueeze(1) * scalar_reads).sum(dim=0)
            torch.testing.assert_close(
                vector.output_logits(vector_h, vector_r),
                scalar.output_logits(scalar_h, scalar_r),
                rtol=2e-6,
                atol=2e-7,
            )


def test_vectorized_beam_matches_scalar_candidates_losses_and_gradients() -> None:
    for arm in ("S32", "DIRECT3", "LINK32"):
        vector = AcquisitionModel(vocabulary_size=4, arm=arm, model_seed=1701)
        scalar = ScalarAcquisitionModel(
            vocabulary_size=4, arm=arm, model_seed=1701
        )
        _assert_exhaustive_beam_expansions_match(vector, scalar, arm)

        sequences = [
            [0, *bits, 1]
            for length in range(1, 7)
            for bits in (
                [2 + ((value >> shift) & 1) for shift in reversed(range(length))]
                for value in range(1 << length)
            )
        ]
        vector_loss = torch.stack(
            [vector.sequence_loss(tokens) for tokens in sequences]
        ).sum()
        scalar_loss = torch.stack(
            [scalar.sequence_loss(tokens) for tokens in sequences]
        ).sum()
        vector_loss.backward()
        scalar_loss.backward()
        torch.testing.assert_close(vector_loss, scalar_loss, rtol=2e-6, atol=2e-5)
        for (name, vector_parameter), (_, scalar_parameter) in zip(
            vector.named_parameters(), scalar.named_parameters(), strict=True
        ):
            torch.testing.assert_close(
                vector_parameter.grad,
                scalar_parameter.grad,
                rtol=3e-4,
                atol=3e-5,
                msg=lambda message, name=name: f"{name}: {message}",
            )
