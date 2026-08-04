import math

import numpy as np
import pytest

from experiments.packed_token_evidence_plane_t27_stage0 import (
    PLANE_CELLS,
    TAPE_TOKENS,
    VOCAB_CAPACITY,
    amplitude,
    amplitude_to_nibble,
    bf16_roundtrip_float32,
    binary_code_to_token,
    channel_ledger,
    match_positions,
    nibbles_to_token,
    pack_tokens,
    read_unique_successor,
    silu_square_pair,
    token_binary_code,
    token_to_nibbles,
    unpack_tokens,
)


@pytest.mark.parametrize("token_id", [0, 1, 15, 16, 255, 256, 4095, 4096, 49_151, 65_535])
def test_token_radix_roundtrip(token_id):
    assert nibbles_to_token(token_to_nibbles(token_id)) == token_id


def test_pack_shape_range_and_roundtrip():
    tape = tuple((index * 1_229 + 17) % VOCAB_CAPACITY for index in range(TAPE_TOKENS))
    cells = pack_tokens(tape)
    assert len(cells) == PLANE_CELLS
    assert min(cells) >= 0
    assert max(cells) <= 15
    assert unpack_tokens(cells) == tape


def test_bf16_amplitude_roundtrip():
    levels = tuple(amplitude(nibble) for nibble in range(16))
    rounded = bf16_roundtrip_float32(levels)
    assert np.array_equal(rounded, np.asarray(levels, dtype=np.float32))
    assert tuple(amplitude_to_nibble(value) for value in rounded) == tuple(range(16))


def test_silu_pair_is_square_on_complete_nibble_difference_domain():
    levels = tuple(amplitude(nibble) for nibble in range(16))
    for left in levels:
        for right in levels:
            difference = float(left - right)
            assert math.isclose(
                silu_square_pair(difference),
                difference * difference,
                rel_tol=0.0,
                abs_tol=1e-10,
            )


def test_unique_missing_ambiguous_and_virtual_padding_reads():
    tape = tuple([101, 202, 303, 101, 202, 404] + list(range(49)))
    assert match_positions(tape, (101, 202)) == (0, 3)
    assert read_unique_successor(tape, (101, 202)) == {
        "status": "ambiguous",
        "positions": (0, 3),
        "value": None,
    }
    assert read_unique_successor(tape, (60_000, 60_001))["status"] == "missing"

    end_tape = tuple(range(1, TAPE_TOKENS + 1))
    assert read_unique_successor(end_tape, (54, 55)) == {
        "status": "ok",
        "positions": (53,),
        "value": 0,
    }


@pytest.mark.parametrize("token_id", [0, 1, 2, 255, 256, 49_151, 65_535])
def test_binary_code_roundtrip(token_id):
    assert binary_code_to_token(token_binary_code(token_id)) == token_id


def test_channel_ledger_fits_width():
    ledger = channel_ledger()
    assert ledger == {
        "two_token_squared_distance_upper_bound": 880,
        "unique_successor_selection_upper_bound": 220,
        "four_nibble_token_decoder_upper_bound": 192,
        "available_swiglu_width": 1024,
    }
    assert max(value for key, value in ledger.items() if key != "available_swiglu_width") <= 1024
