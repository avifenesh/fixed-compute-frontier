import math

from experiments.t32r_proof_first_audit import (
    BASELINE_FFN_WIDTH,
    FROZEN_SCAN_MULTIPLIES,
    NUM_RECORDS,
    RECORD_LENGTH,
    T32_DIGITS_PER_RECORD,
    VOCAB_SIZE,
    audit_payload,
    binary_entropy,
    fano_record_bits,
    typed_plane_ledger,
    width_unit_bytes,
)


def test_fano_bound_matches_exact_entropy_and_decreases_with_error() -> None:
    exact = RECORD_LENGTH * math.log2(VOCAB_SIZE)
    assert fano_record_bits(0.0) == exact
    bounds = [fano_record_bits(error) for error in (0.0, 0.01, 0.05, 0.10)]
    assert all(left > right for left, right in zip(bounds, bounds[1:]))
    assert abs(bounds[-1] - 1735.3566198093047) < 1e-9
    assert binary_entropy(0.0) == 0.0
    assert binary_entropy(1.0) == 0.0


def test_direct_typed_plane_exact_resident_ledger() -> None:
    ledger = typed_plane_ledger()
    assert ledger.token_bytes == NUM_RECORDS * RECORD_LENGTH * 2 == 615_680
    assert ledger.length_bytes == NUM_RECORDS * 2 == 4_810
    assert ledger.scanner_bytes == 54_466
    assert ledger.position_bytes == 8_192
    assert ledger.total_bytes == 683_148
    assert width_unit_bytes() == 23_040
    assert ledger.width_removed == 30
    assert ledger.candidate_width == BASELINE_FFN_WIDTH - 30 == 994
    assert ledger.slack_bytes == 8_052


def test_direct_typed_plane_dominates_frozen_record_bytes() -> None:
    payload = audit_payload()
    t32_record_bytes = NUM_RECORDS * T32_DIGITS_PER_RECORD * 2
    typed_record_bytes = NUM_RECORDS * (RECORD_LENGTH + 1) * 2
    assert t32_record_bytes == 1_832_610
    assert typed_record_bytes == 620_490
    assert typed_record_bytes < t32_record_bytes
    assert payload["resident"]["t32_record_bytes_per_token"] == 5.953125


def test_arithmetic_break_even_is_ten_processed_tokens() -> None:
    ledger = typed_plane_ledger()
    assert ledger.saved_multiplies_per_token == 345_600
    assert 9 * ledger.saved_multiplies_per_token < FROZEN_SCAN_MULTIPLIES
    assert 10 * ledger.saved_multiplies_per_token >= FROZEN_SCAN_MULTIPLIES
    assert ledger.arithmetic_break_even_tokens == 10


def test_low_rank_map_can_be_injective_on_a_finite_set() -> None:
    points = ((0, 0), (1, 0), (0, 1))
    projection = (1, 2)
    values = [sum(x * p for x, p in zip(point, projection)) for point in points]
    assert len(set(values)) == len(points)

