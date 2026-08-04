from types import SimpleNamespace

from experiments import topk_compiled_pushdown_preflight as preflight
from experiments.topk_compiled_pushdown_preflight import (
    PREFLIGHT_SOURCE_PATHS,
    executed_state,
    stratified_sample,
)


def test_stratified_sample_covers_every_length() -> None:
    sequences = [[0, *([2] * length), 1] for length in range(1, 8)]
    sample = stratified_sample(sequences, base_count=2)
    assert {len(sequence) - 2 for sequence in sample} == set(range(1, 8))
    assert sample[:2] == sequences[:2]


def test_preflight_source_hash_labels_are_exact() -> None:
    assert set(PREFLIGHT_SOURCE_PATHS) == {
        "preflight",
        "preflight_tests",
        "training",
        "locked_evaluation",
    }


def test_executed_state_aggregates_peaks_by_maximum(monkeypatch) -> None:
    traces = iter(
        (
            SimpleNamespace(
                stats={
                    "tokens": 4,
                    "maximum_retained_configurations": 10,
                    "maximum_stack_depth": 3,
                    "logical_peak_auxiliary_bytes": 1_000,
                }
            ),
            SimpleNamespace(
                stats={
                    "tokens": 6,
                    "maximum_retained_configurations": 8,
                    "maximum_stack_depth": 5,
                    "logical_peak_auxiliary_bytes": 900,
                }
            ),
        )
    )
    monkeypatch.setattr(
        preflight, "teacher_forced_trace", lambda *args, **kwargs: next(traces)
    )
    result = executed_state(object(), [[0, 1], [0, 1]], "S32")
    assert result == {
        "tokens": 10,
        "maximum_retained_configurations": 10,
        "maximum_stack_depth": 5,
        "logical_peak_auxiliary_bytes": 1_000,
    }
