from __future__ import annotations

import itertools

from experiments.equivariant_template_engram_t38_census import (
    Document,
    KeyStats,
    OutcomeCounts,
    Record,
    admit_records,
    canonicalize,
    compile_head,
    decode_operation,
    encode_operation,
    key_order_digest,
    permute_bindings,
    run_algebra_self_check,
    score_advantage,
    wilson_lower,
)


def rename(values: tuple[int, ...], permutation: dict[int, int]) -> tuple[int, ...]:
    return tuple(permutation.get(value, value) for value in values)


def test_exhaustive_canonicalization_and_equivariance_up_to_four_symbols() -> None:
    literal_ids = frozenset({0})
    vocabulary = tuple(range(4))
    eligible = (1, 2, 3)
    vocab_size = 4

    for length in range(1, 5):
        contexts = tuple(itertools.product(vocabulary, repeat=length))
        templates = {
            context: canonicalize(context, literal_ids)[0] for context in contexts
        }
        for permutation_values in itertools.permutations(eligible):
            permutation = dict(zip(eligible, permutation_values, strict=True))
            for context in contexts:
                template, bindings = canonicalize(context, literal_ids)
                renamed_context = rename(context, permutation)
                renamed_template, renamed_bindings = canonicalize(
                    renamed_context, literal_ids
                )
                assert renamed_template == template
                assert renamed_bindings == rename(bindings, permutation)

                for target in vocabulary:
                    operation = encode_operation(
                        target, bindings, literal_ids, vocab_size
                    )
                    if operation < 0:
                        continue
                    renamed_target = permutation.get(target, target)
                    assert (
                        decode_operation(
                            operation, renamed_bindings, literal_ids, vocab_size
                        )
                        == renamed_target
                    )

        # Converse: equal normal forms iff some eligible permutation maps x to y.
        permutations = [
            dict(zip(eligible, values, strict=True))
            for values in itertools.permutations(eligible)
        ]
        for left in contexts:
            for right in contexts:
                same_template = templates[left] == templates[right]
                same_orbit = any(rename(left, permutation) == right for permutation in permutations)
                assert same_template == same_orbit


def test_copy_indices_and_literal_payload_examples() -> None:
    literals = frozenset({0, 4, 5})
    vocab_size = 10

    template, bindings = canonicalize((1, 0, 2, 0, 1), literals)
    assert template == (-1, 1, -2, 1, -1)
    assert bindings == (1, 2)
    operation = encode_operation(1, bindings, literals, vocab_size)
    assert operation == vocab_size
    assert decode_operation(operation, bindings, literals, vocab_size) == 1

    operation = encode_operation(2, bindings, literals, vocab_size)
    assert operation == vocab_size + 1
    assert decode_operation(operation, bindings, literals, vocab_size) == 2

    operation = encode_operation(4, bindings, literals, vocab_size)
    assert operation == 4
    assert decode_operation(operation, bindings, literals, vocab_size) == 4

    assert encode_operation(3, bindings, literals, vocab_size) == -1
    assert decode_operation(vocab_size + 2, bindings, literals, vocab_size) is None
    assert decode_operation(3, bindings, literals, vocab_size) is None


def test_deterministic_admission_order_and_document_gate() -> None:
    candidates = {}
    for key, winner_count, total, documents in (
        ((1,), 20, 20, {"a", "b", "c", "d"}),
        ((2,), 25, 30, {"a", "b", "c", "d"}),
        ((3,), 25, 30, {"a", "b", "c"}),
        ((4,), 25, 31, {"a", "b", "c", "d"}),
    ):
        candidates[key] = KeyStats(
            total=total,
            winner=7,
            winner_count=winner_count,
            documents=set(documents),
        )

    first = admit_records(candidates, n=8, mode="static_template")
    second = admit_records(
        dict(reversed(tuple(candidates.items()))), n=8, mode="static_template"
    )
    assert list(first) == list(second)
    assert (3,) not in first
    expected = [(4,), (2,), (1,)]
    # Equal winning count sorts by larger total before digest.
    assert list(first) == expected
    assert [record.order_digest for record in first.values()] == [
        key_order_digest(key) for key in expected
    ]

    lexical = admit_records(candidates, n=8, mode="lexical")
    assert (3,) in lexical
    assert len(lexical) == 4


def test_repair_minus_harm_is_all_event_difference() -> None:
    candidate = {
        "prose": {"events": 100, "correct": 15},
        "math": {"events": 200, "correct": 40},
        "code": {"events": 50, "correct": 5},
    }
    baseline = {
        "prose": {"events": 100, "correct": 10},
        "math": {"events": 200, "correct": 30},
        "code": {"events": 50, "correct": 5},
    }
    by_source, aggregate = score_advantage(candidate, baseline)
    assert by_source == {"prose": 0.05, "math": 0.05, "code": 0.0}
    assert abs(aggregate - (0.05 + 0.05) / 3.0) < 1e-15


def test_wilson_lower_is_conservative_and_monotone_for_fixed_total() -> None:
    assert wilson_lower(0, 20) == 0.0
    assert 0.0 < wilson_lower(19, 20) < 19 / 20
    assert wilson_lower(20, 20) < 1.0
    assert wilson_lower(20, 20) > wilson_lower(19, 20)


def test_compiler_generalizes_copy_to_unseen_binding_while_concrete_key_abstains() -> None:
    literals = frozenset({0})
    vocab_size = 64
    train = []
    for doc_index, first in enumerate((1, 4, 7, 10)):
        context = (first, 0, first + 1, 0, first + 2, 0, first, 0)
        tokens = tuple((*context, first) * 4)
        train.append(Document("code", f"train-{doc_index}", "train", tokens))

    t38 = compile_head(train, 8, "t38", literals, vocab_size)
    lexical = compile_head(train, 8, "lexical", literals, vocab_size)

    unseen_context = (20, 0, 21, 0, 22, 0, 20, 0)
    template, bindings = canonicalize(unseen_context, literals)
    assert template in t38
    assert unseen_context not in lexical
    record = t38[template]
    assert decode_operation(record.outcome, bindings, literals, vocab_size) == 20
    assert bindings not in record.train_bindings


def test_binding_control_is_identity_only_below_arity_two_and_deranges_otherwise() -> None:
    document = Document("code", "doc", "adjudication", ())
    assert permute_bindings((), document, 8) == ()
    assert permute_bindings((11,), document, 8) == (11,)
    for size in range(2, 9):
        bindings = tuple(range(size))
        shifted = permute_bindings(bindings, document, 8)
        assert sorted(shifted) == list(bindings)
        assert all(left != right for left, right in zip(bindings, shifted, strict=True))


def test_runtime_exhaustive_check_covers_all_vocabularies_and_partitions() -> None:
    receipt = run_algebra_self_check()
    assert receipt["pass"] is True
    assert receipt["literal_partitions_checked"] == sum(1 << size for size in range(1, 5))
    assert receipt["bottom_operations_checked"] > 0
    assert receipt["deterministic_compiler_replay_checked"] is True
    assert receipt["repair_minus_harm_checked"] is True


def test_outcome_ties_choose_smaller_encoding_and_bottom_forces_abstention() -> None:
    counts = OutcomeCounts(5)
    counts.add(3)
    assert counts.summary() == (2, 3, 1)
    bottom_tie = OutcomeCounts(5)
    bottom_tie.add(-1)
    assert bottom_tie.summary() == (2, -1, 1)
