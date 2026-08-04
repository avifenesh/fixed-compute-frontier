import pytest

from experiments.scale_referenced_whole_record_t32_stage0 import (
    ACTIVE_CELLS,
    RECORD_CELLS,
    SCALES,
    TOKEN_SLOTS,
    decode_scaled_digit,
    digit_amplitude,
    bf16_scalar,
    numerical_replay,
    pack_record,
    path_has_snapshot_revision,
    resource_ledger,
    unpack_record,
)


BOUNDARY_IDS = (0, 1, 63, 64, 4_095, 4_096, 49_151, 65_535)


@pytest.mark.parametrize("length", [0, 1, 127, 128])
def test_boundary_record_roundtrip(length):
    tokens = tuple(
        BOUNDARY_IDS[index % len(BOUNDARY_IDS)] if index < length else 0
        for index in range(TOKEN_SLOTS)
    )
    record = pack_record(tokens, length)
    assert len(record) == RECORD_CELLS
    assert all(0 <= cell < 64 for cell in record[:381])
    assert record[381:] == (1, 0, 0)
    assert unpack_record(record) == (tokens, length)


def test_last_token_and_length_share_auxiliary_remainders_exactly():
    tokens = (0,) * 127 + (65_535,)
    record = pack_record(tokens, 128)
    assert unpack_record(record) == (tokens, 128)
    assert tuple(record[3 * index + 2] % 4 for index in range(8)) == (3,) * 8
    assert tuple(record[3 * index + 2] % 4 for index in range(8, 12)) == (0, 0, 0, 2)


def test_nonzero_padding_is_rejected():
    with pytest.raises(ValueError, match="nonzero token"):
        pack_record((7,) + (0,) * 127, 0)


@pytest.mark.parametrize("index", [382, 383])
def test_nonzero_reserved_cell_is_rejected(index):
    record = list(pack_record((0,) * TOKEN_SLOTS, 0))
    record[index] = 1
    with pytest.raises(ValueError, match="reserved"):
        unpack_record(record)


def test_late_auxiliary_value_is_rejected():
    record = list(pack_record((0,) * TOKEN_SLOTS, 0))
    record[3 * 12 + 2] = 1
    with pytest.raises(ValueError, match="auxiliary"):
        unpack_record(record)


def test_every_digit_decodes_at_every_registered_scale():
    for digit in range(64):
        for scale in SCALES:
            value = bf16_scalar(scale * digit_amplitude(digit))
            reference = bf16_scalar(scale)
            assert decode_scaled_digit(value, reference) == digit


def test_complete_numerical_replay_including_perturbations():
    replay = numerical_replay()
    assert replay["clean_failures"] == []
    assert replay["perturbed_failures"] == []
    assert replay["minimum_clean_margin_in_scale_units"] >= 1.0
    assert replay["minimum_perturbed_margin_in_scale_units"] > 0.0


def test_resource_ledger_is_exact_and_below_five_percent():
    ledger = resource_ledger()
    assert ACTIVE_CELLS == 382
    assert ledger["payload_entries"] == 918_710
    assert ledger["total_entries"] == 1_680_736
    assert ledger["fraction_of_model"] < 0.05


def test_snapshot_revision_path_predicate_is_structural():
    revision = "abc123"
    assert path_has_snapshot_revision(
        "/cache/repo/snapshots/abc123/tokenizer.json", revision
    )
    assert not path_has_snapshot_revision(
        "/cache/repo/snapshots/wrong/tokenizer.json", revision
    )
