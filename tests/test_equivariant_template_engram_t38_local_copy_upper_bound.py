from experiments.equivariant_template_engram_t38_local_copy_upper_bound import (
    add_fractions,
    merge_counts,
    oracle_counts,
    self_check,
)


def test_frozen_self_check() -> None:
    assert self_check()["pass"] is True


def test_copyability_is_literal_filtered_and_window_monotone() -> None:
    tokens = tuple(range(10)) + (1, 8, 1)
    eligible = oracle_counts(tokens, frozenset({0, 2, 3, 4, 5, 6, 7, 8, 9}))
    literal = oracle_counts(tokens, frozenset(range(10)))
    assert eligible["copyable_8"] <= eligible["copyable_16"] <= eligible["copyable_32"]
    assert eligible["copyable_32"] > literal["copyable_32"] == 0


def test_merge_and_fraction_denominator_include_all_events() -> None:
    merged = merge_counts(
        (
            {"events": 10, "unrestricted_repeat_32": 4, "copyable_8": 1, "copyable_16": 2, "copyable_32": 3},
            {"events": 10, "unrestricted_repeat_32": 6, "copyable_8": 3, "copyable_16": 4, "copyable_32": 5},
        )
    )
    result = add_fractions(merged)
    assert result["copyable_32_fraction"] == 8 / 20
    assert result["unrestricted_repeat_32_fraction"] == 10 / 20
