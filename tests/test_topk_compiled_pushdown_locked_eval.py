import math
import itertools
from pathlib import Path

import torch

from experiments.topk_compiled_pushdown_locked_eval import (
    EVALUATION_CELL_KEYS,
    SHARD_SCHEMA,
    MetricAccumulator,
    Trace,
    _select,
    _write_once_text,
    accumulate_trace,
    analytic_next_distribution,
    analytic_map_continuation,
    analytic_outcome_distribution,
    causal_history_metrics,
    completion_outcome,
    continuation_decision,
    control_trace,
    delayed_positions,
    hard3_trace,
    model_continuation,
    merge_evaluation_shard_records,
    teacher_forced_trace,
    teacher_forced_decision,
    weighted_stack_trace,
)
from experiments.topk_compiled_pushdown_train import AcquisitionModel


def test_fp64_selection_uses_structural_tie_break() -> None:
    selected = _select({(2, ()): 1.0, (0, ()): 1.0, (1, ()): 1.0}, 2)
    assert list(selected) == [(0, ()), (1, ())]
    assert math.isclose(sum(selected.values()), 1.0)


def test_locked_output_publication_is_atomic_and_never_overwrites(
    tmp_path: Path,
) -> None:
    output = tmp_path / "shard-00.json"
    _write_once_text(output, "first\n")
    try:
        _write_once_text(output, "second\n")
    except FileExistsError:
        pass
    else:
        raise AssertionError("write-once publication overwrote an existing shard")
    assert output.read_text(encoding="utf-8") == "first\n"


def test_locked_trace_matches_training_forward_loss_at_fp32_tolerance() -> None:
    tokens = [0, 2, 3, 2, 1]
    model = AcquisitionModel(vocabulary_size=4, arm="S32", model_seed=1701)
    trace = teacher_forced_trace(model, tokens, evaluated_arm="S32")
    locked_logits = torch.stack(trace.logits)
    targets = torch.tensor(tokens[1:])
    locked_loss = torch.nn.functional.cross_entropy(
        locked_logits, targets, reduction="sum"
    )
    training_loss = model.sequence_loss(tokens)
    torch.testing.assert_close(
        locked_loss.float(), training_loss, rtol=2e-5, atol=2e-5
    )


def test_zero_read_changes_only_stack_read_path() -> None:
    tokens = [0, 2, 3, 2, 1]
    model = AcquisitionModel(vocabulary_size=4, arm="S32", model_seed=1701)
    clean = teacher_forced_trace(model, tokens, evaluated_arm="S32")
    zero = teacher_forced_trace(
        model, tokens, evaluated_arm="S32", intervention="ZERO-READ"
    )
    assert len(clean.logits) == len(zero.logits) == len(tokens) - 1
    assert any(
        not torch.allclose(left, right)
        for left, right in zip(clean.logits, zero.logits, strict=True)
    )


def test_delayed_position_contracts() -> None:
    assert delayed_positions(
        "D1",
        [0, 2, 3, 5, 4, 1],
        ["BOS", "EOS", "open0", "open1", "close0", "close1"],
    ) == {2: 4, 3: 5}
    assert delayed_positions(
        "A1", [0, 2, 3, 3, 2, 1], ["BOS", "EOS", "0", "1"]
    ) == {2: 2, 3: 3}
    assert delayed_positions(
        "A2", [0, 2, 3, 4, 1], ["BOS", "EOS", "a", "b", "c"]
    ) == {2: 3, 3: 4}


def test_conditional_suffix_nll_scores_the_full_suffix_after_the_prompt() -> None:
    tokens = [0, 2, 3, 2, 1]
    trace = Trace(
        logits=[torch.zeros(4, dtype=torch.float64) for _ in tokens[:-1]],
        stats={},
    )
    accumulator = MetricAccumulator()
    accumulate_trace(
        accumulator,
        trace,
        tokens,
        family="A1",
        token_names=["BOS", "EOS", "0", "1"],
    )
    result = accumulator.result()
    assert result["conditional_suffix_tokens"] == 2
    assert math.isclose(result["conditional_suffix_nll"], 2.0 * math.log(4.0))


def test_executed_state_aggregates_totals_but_keeps_true_maxima() -> None:
    accumulator = MetricAccumulator()
    accumulator.add_stats(
        {
            "tokens": 4,
            "maximum_stack_depth": 7,
            "logical_peak_auxiliary_bytes": 100,
        }
    )
    accumulator.add_stats(
        {
            "tokens": 5,
            "maximum_stack_depth": 3,
            "logical_peak_auxiliary_bytes": 80,
        }
    )
    assert accumulator.stats == {
        "tokens": 9,
        "maximum_stack_depth": 7,
        "logical_peak_auxiliary_bytes": 100,
    }


def test_causal_history_bound_holds_on_learned_operator() -> None:
    model = AcquisitionModel(vocabulary_size=4, arm="S32", model_seed=1701)
    metrics = causal_history_metrics(model, [0, 2, 3, 2, 1])
    assert metrics["prefixes"] == 4
    for error, bound in zip(
        metrics["read_errors"], metrics["bounds"], strict=True
    ):
        assert error <= bound + 1e-6


def test_analytic_conditionals_preserve_latent_multiplicity() -> None:
    d1 = analytic_next_distribution(
        "D1", ["open0"], low=1, high=2
    )
    assert d1 == {"close0": 0.5, "open0": 0.25, "open1": 0.25}
    a1 = analytic_next_distribution("A1", ["0"], low=1, high=1)
    assert a1 == {"0": 0.75, "1": 0.25}
    a2 = analytic_outcome_distribution("A2", ["a"], low=1, high=1)
    assert a2 == {"a:1,b:1,c:1": 1.0}


def test_completion_validity_and_outcomes() -> None:
    assert completion_outcome(
        "D1", ["open0", "close0", "EOS"], low=1, high=2
    ) == (True, "length:2", 2)
    assert completion_outcome(
        "A1", ["0", "1", "0", "EOS"], low=1, high=2
    ) == (True, "length:3", 3)
    assert completion_outcome(
        "A2", ["a", "b", "c", "EOS"], low=1, high=2
    ) == (True, "a:1,b:1,c:1", 3)


def test_incremental_decoder_matches_independent_whole_sequence_references() -> None:
    tokens = [0, 2, 3, 2, 1]
    for arm in ("S32", "DIRECT3", "LINK32"):
        model = AcquisitionModel(vocabulary_size=4, arm=arm, model_seed=1701)
        expected = weighted_stack_trace(
            model, tokens, beam_size=3 if arm == "DIRECT3" else 32
        )
        actual = teacher_forced_trace(model, tokens, evaluated_arm=arm)
        for left, right in zip(expected.logits, actual.logits, strict=True):
            torch.testing.assert_close(left, right)
        assert expected.stats == actual.stats
        assert actual.stats["action_projections"] == actual.stats["tokens"]
    hard = AcquisitionModel(vocabulary_size=4, arm="HARD3", model_seed=1701)
    expected_hard = hard3_trace(hard, tokens)
    actual_hard = teacher_forced_trace(hard, tokens, evaluated_arm="HARD3")
    for left, right in zip(expected_hard.logits, actual_hard.logits, strict=True):
        torch.testing.assert_close(left, right)
    for arm in ("MLP", "RNN32"):
        model = AcquisitionModel(vocabulary_size=4, arm=arm, model_seed=1701)
        expected = control_trace(model, tokens)
        actual = teacher_forced_trace(model, tokens, evaluated_arm=arm)
        for left, right in zip(expected.logits, actual.logits, strict=True):
            torch.testing.assert_close(left, right)


def test_incremental_continuation_stops_on_eos() -> None:
    model = AcquisitionModel(vocabulary_size=4, arm="MLP", model_seed=1701)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.b_y[1] = 10.0
    continuation = model_continuation(
        model,
        "MLP",
        [0, 2],
        eos_id=1,
        token_names=["BOS", "EOS", "0", "1"],
        uniforms=None,
        horizon=10,
    )
    assert continuation == [1]


def test_analytic_conditionals_match_explicit_small_enumerators() -> None:
    generated = {"D1": [], "A1": [], "A2": []}
    for n in range(1, 3):
        for bits in itertools.product((0, 1), repeat=n):
            opens = [f"open{bit}" for bit in bits]
            closes = [f"close{bit}" for bit in reversed(bits)]
            generated["D1"].append((opens + closes, 2.0 ** (-n)))
            for parity in (0, 1):
                for center in itertools.product((0, 1), repeat=parity):
                    content = [str(bit) for bit in bits]
                    content += [str(bit) for bit in center]
                    content += [str(bit) for bit in reversed(bits)]
                    generated["A1"].append(
                        (content, 0.5 * 2.0 ** (-(n + parity)))
                    )
    for branch in (0, 1):
        for n in range(1, 3):
            for m in range(1, 3):
                counts = (n, n, m) if branch == 0 else (m, n, n)
                content = (
                    ["a"] * counts[0]
                    + ["b"] * counts[1]
                    + ["c"] * counts[2]
                )
                generated["A2"].append((content, 1.0))

    for family, examples in generated.items():
        prefixes = {
            tuple(content[:length])
            for content, _ in examples
            for length in range(len(content) + 1)
        }
        for prefix in prefixes:
            matches = [
                (content, weight)
                for content, weight in examples
                if list(prefix) == content[: len(prefix)]
            ]
            denominator = sum(weight for _, weight in matches)
            brute_next = {}
            brute_outcomes = {}
            for content, weight in matches:
                symbol = "EOS" if len(prefix) == len(content) else content[len(prefix)]
                brute_next[symbol] = brute_next.get(symbol, 0.0) + weight / denominator
                if family == "A2":
                    outcome = (
                        f"a:{content.count('a')},b:{content.count('b')},"
                        f"c:{content.count('c')}"
                    )
                else:
                    outcome = f"length:{len(content)}"
                brute_outcomes[outcome] = (
                    brute_outcomes.get(outcome, 0.0) + weight / denominator
                )
            actual_next = analytic_next_distribution(
                family, list(prefix), low=1, high=2
            )
            actual_outcomes = analytic_outcome_distribution(
                family, list(prefix), low=1, high=2
            )
            assert set(actual_next) == set(brute_next)
            assert set(actual_outcomes) == set(brute_outcomes)
            for symbol, probability in brute_next.items():
                assert math.isclose(actual_next[symbol], probability, abs_tol=1e-12)
            for outcome, probability in brute_outcomes.items():
                assert math.isclose(
                    actual_outcomes[outcome], probability, abs_tol=1e-12
                )
            continuation_weights = {}
            for content, weight in matches:
                continuation = tuple(content[len(prefix) :] + ["EOS"])
                continuation_weights[continuation] = (
                    continuation_weights.get(continuation, 0.0) + weight
                )
            maximum = max(continuation_weights.values())
            brute_map = list(
                min(
                    continuation
                    for continuation, weight in continuation_weights.items()
                    if math.isclose(weight, maximum, abs_tol=1e-15)
                )
            )
            assert analytic_map_continuation(
                family,
                list(prefix),
                low=1,
                high=2,
                horizon=10,
            ) == brute_map


def _metric(nll: float, accuracy: float) -> dict:
    return {
        "token_nll": nll,
        "delayed_accuracy": accuracy,
        "early_nll": 1.0,
        "early_accuracy": 0.7,
        "conditional_suffix_nll": 1.0,
        "early_length_slices": {"10": {"nll": 1.0}},
    }


def _passing_decision_ledgers() -> tuple[dict, dict]:
    teacher_cells = {}
    continuation_cells = {}
    for family in ("D1", "A1", "A2"):
        for split in ("E1", "E2"):
            for seed in range(1701, 1711):
                arms = {
                    control: _metric(1.0, 0.5)
                    for control in ("DIRECT3", "HARD3", "MLP", "RNN32", "LINK32")
                }
                arms.update(
                    {
                        "S32": _metric(0.9, 0.6),
                        "K1": _metric(0.91, 0.59),
                        "K2": _metric(0.91, 0.59),
                        "K3": _metric(0.91, 0.59),
                    }
                )
                teacher_cells[f"{family}/{split}/{seed}"] = {
                    "arms": arms,
                    "kl_s32_to_k3": {"nat_per_token": 0.005},
                    "causal_k3_history": {
                        "mean_delta": 0.005,
                        "fraction_delta_le_0_01": 0.995,
                        "maximum_bound_excess": 0.0,
                    },
                    "interventions": {
                        "S32/ZERO-READ": _metric(1.1, 0.51),
                        "S32/CHRONO-LINK": _metric(1.1, 0.51),
                        "K3/ZERO-READ": _metric(1.1, 0.51),
                        "K3/CHRONO-LINK": _metric(1.1, 0.51),
                    },
                }
                continuation_arms = {}
                for arm in ("DIRECT3", "HARD3", "MLP", "RNN32", "LINK32"):
                    continuation_arms[arm] = {
                        "grammar_valid_rate": 0.5,
                        "premature_eos_rate": 0.2,
                        "late_eos_rate": 0.2,
                        "no_eos_rate": 0.2,
                        "stratified_outcome_tv": 0.2,
                        "greedy_grammar_valid_rate": 0.5,
                        "greedy_analytic_map_rate": 0.5,
                        "conditional_suffix_nll": 1.0,
                    }
                continuation_arms["S32"] = {
                    "grammar_valid_rate": 0.7,
                    "premature_eos_rate": 0.1,
                    "late_eos_rate": 0.1,
                    "no_eos_rate": 0.1,
                    "stratified_outcome_tv": 0.1,
                    "greedy_grammar_valid_rate": 0.7,
                    "greedy_analytic_map_rate": 0.7,
                    "conditional_suffix_nll": 0.8,
                }
                continuation_arms["K3"] = {
                    "grammar_valid_rate": 0.68,
                    "premature_eos_rate": 0.11,
                    "late_eos_rate": 0.11,
                    "no_eos_rate": 0.11,
                    "stratified_outcome_tv": 0.11,
                    "greedy_grammar_valid_rate": 0.68,
                    "greedy_analytic_map_rate": 0.68,
                    "conditional_suffix_nll": 0.82,
                }
                continuation_cells[f"{family}/{split}/{seed}"] = {
                    "arms": continuation_arms
                }
    return teacher_cells, continuation_cells


def test_frozen_decision_ledgers_accept_only_material_retained_gain() -> None:
    teacher_cells, continuation_cells = _passing_decision_ledgers()
    teacher_decision = teacher_forced_decision({"cells": teacher_cells})
    diagnostic = teacher_decision["cells"]["D1/E1"]
    assert teacher_decision["pass"] is True, {
        "compression": diagnostic["compression"],
        "kl": diagnostic["recurrent_kl"],
        "interventions": diagnostic["interventions"],
    }
    assert continuation_decision({"cells": continuation_cells})["pass"] is True


def test_kl_gate_uses_the_across_seed_mean_not_the_median() -> None:
    teacher_cells, _ = _passing_decision_ledgers()
    for offset, seed in enumerate(range(1701, 1711)):
        teacher_cells[f"D1/E1/{seed}"]["kl_s32_to_k3"]["nat_per_token"] = (
            0.009 if offset < 6 else 0.020
        )
    decision = teacher_forced_decision({"cells": teacher_cells})
    cell = decision["cells"]["D1/E1"]
    assert math.isclose(cell["recurrent_kl"]["mean_nat_per_token"], 0.0134)
    assert cell["recurrent_kl"]["pass"] is False
    assert decision["pass"] is False


def test_continuation_gate_rejects_unretained_conditional_suffix_nll() -> None:
    _, continuation_cells = _passing_decision_ledgers()
    for seed in range(1701, 1711):
        continuation_cells[f"A1/E2/{seed}"]["arms"]["K3"][
            "conditional_suffix_nll"
        ] = 0.9
    decision = continuation_decision({"cells": continuation_cells})
    cell = decision["cells"]["A1/E2"]
    assert cell["metrics"]["conditional_suffix_nll"]["pass"] is False
    assert decision["pass"] is False


def _complete_shard_records() -> list[dict]:
    teacher_cells, continuation_cells = _passing_decision_ledgers()
    records = []
    for index, key in enumerate(EVALUATION_CELL_KEYS):
        records.append(
            {
                "schema": SHARD_SCHEMA,
                "lineage": "CPKV-TOPK-001",
                "status": "complete",
                "shard_index": index,
                "cell_key": key,
                "manifest_sha256": "manifest",
                "checkpoint_freeze_receipt_sha256": "receipt",
                "locked_evaluator_sha256": "evaluator",
                "teacher_forced": {"cells": {key: teacher_cells[key]}},
                "continuations": {"cells": {key: continuation_cells[key]}},
            }
        )
    return records


def test_shard_merge_requires_exact_bound_cells_and_frozen_order() -> None:
    teacher, continuations = merge_evaluation_shard_records(
        list(reversed(_complete_shard_records())),
        manifest_sha256="manifest",
        receipt_sha256="receipt",
        locked_evaluator_sha256="evaluator",
    )
    assert tuple(teacher["cells"]) == EVALUATION_CELL_KEYS
    assert tuple(continuations["cells"]) == EVALUATION_CELL_KEYS


def test_shard_merge_rejects_missing_duplicate_or_rebound_shards() -> None:
    records = _complete_shard_records()
    for corrupted in (
        records[:-1],
        [*records[:-1], records[0]],
        [
            *records[:17],
            {**records[17], "manifest_sha256": "different"},
            *records[18:],
        ],
    ):
        try:
            merge_evaluation_shard_records(
                corrupted,
                manifest_sha256="manifest",
                receipt_sha256="receipt",
                locked_evaluator_sha256="evaluator",
            )
        except ValueError:
            pass
        else:
            raise AssertionError("corrupt shard set was accepted")
