import math

from experiments.topk_compiled_pushdown_reference import (
    K_BEAM,
    PersistentArena,
    S_BEAM,
    TrackedTupleState,
    action_probabilities,
    run_oracle,
    step_arena,
    step_tracked_tuple,
)


def test_root_pop_is_masked_and_normalized() -> None:
    actions = action_probabilities(time_index=0, token=0, q=0, top_symbol=-1)
    assert all(action.kind != "pop" for action, _ in actions)
    assert math.isclose(sum(probability for _, probability in actions), 1.0)


def test_tracked_state_has_declared_beam_bounds() -> None:
    state = TrackedTupleState(
        total={(0, ()): 1.0},
        covered={(0, ()): 1.0},
        k_state={(0, ()): 1.0},
    )
    for time_index, token in enumerate((0, 1, 1, 0)):
        state = step_tracked_tuple(state, time_index=time_index, token=token)
        assert len(state.total) <= S_BEAM
        assert len(state.k_state) <= K_BEAM
        assert 0.0 < sum(state.covered.values()) <= 1.0 + 1e-12


def test_two_representations_and_history_bound_exhaustively() -> None:
    result = run_oracle(max_length=6)
    assert result["status"] == "pass"
    assert result["checked_sequences"] == 127
    assert result["x4"]["top3_history_mass"] == 0.75
    assert math.isclose(result["x4"]["read_error"], 1.0 / math.sqrt(12.0))


def test_arena_does_not_allocate_rejected_push_candidates() -> None:
    arena = PersistentArena()
    state = step_arena(
        arena,
        {(0, 0): 1.0},
        time_index=0,
        token=0,
        beam_size=1,
    )
    assert len(state) == 1
    assert len(arena.nodes) <= 2  # root plus at most the retained push
