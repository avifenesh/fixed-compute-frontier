from __future__ import annotations

from experiments import xor_functional_record_t24_stage0 as t24


def test_hash_positions_are_distinct_deterministic_and_bounded() -> None:
    skeleton = ((-2, "was"), (-1, "born"), (1, "in"))
    first = t24.hash_positions(skeleton)
    assert first == t24.hash_positions(skeleton)
    assert len(first) == len(set(first)) == 3
    assert all(0 <= value < t24.LOGICAL_SLOTS for value in first)


def test_gf2_multi_payload_solver_roundtrips() -> None:
    rows = [0b00111, 0b01101, 0b11001]
    payloads = [0b0000011, 0b1010101, 0b1110000]
    assert t24.gf2_rank(rows) == len(rows)
    solution = t24.solve_full_row_rank(rows, payloads, columns=5)
    for row, expected in zip(rows, payloads, strict=True):
        actual = 0
        for column in range(5):
            if (row >> column) & 1:
                actual ^= solution[column]
        assert actual == expected


def test_record_layout_and_read_identity() -> None:
    skeletons = [
        ((-1, "born"), (1, "in")),
        ((-1, "member"), (1, "of")),
        ((-2, "released"), (1, "by")),
    ]
    payloads = [3, 65, 17]
    record, rank = t24.encode_record(skeletons, payloads)
    assert rank == len(skeletons)
    assert len(record) == 220
    assert record[-2:] == [0, 0]
    assert [t24.read_record(record, key) for key in skeletons] == payloads


def test_preregistration_is_sealed() -> None:
    assert t24.sha256_file(t24.PREREGISTRATION) == t24.PREREGISTRATION_SHA256
