from __future__ import annotations

import collections

import torch

from experiments import raw_self_query_functional_record_t22a as t22a


class Tokenizer:
    @staticmethod
    def encode(text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [100 + len(text.strip())]


def layout() -> t22a.SpecialLayout:
    return t22a.SpecialLayout(1, 2, 3, 4, 5, 100)


def test_skeleton_encoding_erases_target_slot_and_keeps_offsets() -> None:
    encoded = t22a.encode_skeleton(
        ((-2, "born"), (1, "in")),
        Tokenizer(),
        layout(),
        layout().read,
        layout().answer,
    )
    assert encoded[0] == layout().read
    assert encoded[-1] == layout().answer
    assert encoded.count(layout().mask) == 6
    assert 104 in encoded  # born
    assert 102 in encoded  # in


def test_equality_dataset_is_raw_exact_and_balanced_sampling_is_deterministic() -> None:
    skeleton = ((-1, "born"), (1, "in"))
    collisions = {
        skeleton: {(0, "1970"), (1, "1970"), (2, "1980")}
    }
    dataset = t22a.build_equality_dataset(collisions, Tokenizer(), layout())
    assert len(dataset.positive) == 1
    assert len(dataset.negative) == 2
    assert dataset.positive[0].target == 0
    assert all(row.target == 1 for row in dataset.negative)
    assert t22a.balanced_equality_rows(dataset, 9, 4) == t22a.balanced_equality_rows(
        dataset, 9, 4
    )


def test_unary_packing_places_only_missing_word_tokens_under_loss() -> None:
    row = t22a.EncodedProbe(
        document_index=7,
        query_ids=(1, 2, 3),
        target_ids=(11, 12),
        skeleton=((-1, "born"),),
        target_word="1970",
        structured=True,
    )
    inputs, targets, mask, documents = t22a.pack_unary_rows(
        [row], eos=0, device=torch.device("cpu")
    )
    assert inputs[0, :4].tolist() == [1, 2, 3, 11]
    assert torch.nonzero(mask[0]).flatten().tolist() == [2, 3]
    assert targets[0, 2:4].tolist() == [11, 12]
    assert documents.tolist() == [7]


def test_equality_packing_injects_two_records_and_uses_binary_target() -> None:
    row = t22a.EqualityExample(3, 8, (4, 2, 5), 0)
    inputs, lengths, pairs, positions, targets = t22a.pack_equality_rows(
        [row], eos=0, device=torch.device("cpu")
    )
    assert inputs[0, :3].tolist() == [4, 2, 5]
    assert lengths.tolist() == [3]
    assert pairs.tolist() == [[3, 8]]
    assert positions.tolist() == [[0, 2]]
    assert targets.tolist() == [0]


def test_frozen_update_schedules_match_declared_exposure() -> None:
    one = collections.Counter(
        t22a.phase_for_step("functional_record_1x", step)
        for step in range(1, t22a.ONE_X_STEPS + 1)
    )
    two = collections.Counter(
        t22a.phase_for_step("dense_probe_2x", step)
        for step in range(1, t22a.TWO_X_STEPS + 1)
    )
    assert one == {"natural": 19_000, "unary": 8_000, "equality": 2_000}
    assert two == {"natural": 19_000, "unary": 16_000, "equality": 4_000}


def test_record_and_physical_budgets_are_unchanged() -> None:
    logical_bits = 2_405 * t22a.CODE_DIMS * 4
    assert logical_bits == 2_116_400
    assert t22a.t19b.EXPECTED_WRITES == 743_670
    assert t22a.t19b.EXPECTED_WRITES <= t22a.t19b.WRITE_BUDGET
    assert t22a.t21a.sha256_file(t22a.STAGE0_RESULT) == t22a.STAGE0_SHA256
