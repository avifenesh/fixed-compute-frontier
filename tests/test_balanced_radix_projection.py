from __future__ import annotations

import pytest
import torch

from experiments.balanced_radix_projection import (
    BINARY_RADIX,
    RADIX,
    TILE_K,
    decode_binary_tile,
    decode_balanced_tile,
    matched_ffn_ledger,
    pack_two_binary_weights,
    pack_two_ternary_weights,
    packed_two_binary_projection,
    packed_two_projection,
)


def test_all_balanced_digits_decode_exactly() -> None:
    low = torch.arange(-TILE_K, TILE_K + 1, dtype=torch.int32)
    high = torch.arange(-TILE_K, TILE_K + 1, dtype=torch.int32)
    grid_low, grid_high = torch.meshgrid(low, high, indexing="ij")
    packed = grid_low + RADIX * grid_high
    decoded_low, decoded_high = decode_balanced_tile(packed)
    torch.testing.assert_close(decoded_low, grid_low, rtol=0, atol=0)
    torch.testing.assert_close(decoded_high, grid_high, rtol=0, atol=0)


@pytest.mark.parametrize("input_k", [32, 64, 384])
def test_random_packed_projection_equals_two_independent_dots(input_k: int) -> None:
    generator = torch.Generator().manual_seed(1709 + input_k)
    activation = torch.randint(-1, 2, (7, input_k), generator=generator, dtype=torch.int8)
    low_weight = torch.randint(-1, 2, (19, input_k), generator=generator, dtype=torch.int8)
    high_weight = torch.randint(-1, 2, (19, input_k), generator=generator, dtype=torch.int8)
    actual_low, actual_high = packed_two_projection(
        activation, low_weight, high_weight
    )
    expected_low = activation.to(torch.int32) @ low_weight.to(torch.int32).T
    expected_high = activation.to(torch.int32) @ high_weight.to(torch.int32).T
    torch.testing.assert_close(actual_low, expected_low, rtol=0, atol=0)
    torch.testing.assert_close(actual_high, expected_high, rtol=0, atol=0)


def test_extreme_weights_fit_signed_int8() -> None:
    positive = torch.ones(5, 32, dtype=torch.int8)
    negative = -positive
    packed_max = pack_two_ternary_weights(positive, positive)
    packed_min = pack_two_ternary_weights(negative, negative)
    assert int(packed_max.max()) == 66
    assert int(packed_min.min()) == -66


def test_binary_radix64_touching_points_decode_by_parity() -> None:
    # At +/-32, high must be even because both binary-weight dots have the
    # parity of the count of nonzero activations.
    low = torch.tensor([-32, 32, -32, 32], dtype=torch.int32)
    high = torch.tensor([-2, -2, 2, 2], dtype=torch.int32)
    decoded_low, decoded_high = decode_binary_tile(low + BINARY_RADIX * high)
    torch.testing.assert_close(decoded_low, low, rtol=0, atol=0)
    torch.testing.assert_close(decoded_high, high, rtol=0, atol=0)


@pytest.mark.parametrize("input_k", [32, 64, 384])
def test_binary_radix64_projection_is_exact(input_k: int) -> None:
    generator = torch.Generator().manual_seed(911 + input_k)
    activation = torch.randint(-1, 2, (9, input_k), generator=generator, dtype=torch.int8)
    low_weight = 2 * torch.randint(0, 2, (23, input_k), generator=generator, dtype=torch.int8) - 1
    high_weight = 2 * torch.randint(0, 2, (23, input_k), generator=generator, dtype=torch.int8) - 1
    packed = pack_two_binary_weights(low_weight, high_weight)
    assert int(packed.min()) >= -65 and int(packed.max()) <= 65
    actual_low, actual_high = packed_two_binary_projection(
        activation, low_weight, high_weight
    )
    expected_low = activation.to(torch.int32) @ low_weight.to(torch.int32).T
    expected_high = activation.to(torch.int32) @ high_weight.to(torch.int32).T
    torch.testing.assert_close(actual_low, expected_low, rtol=0, atol=0)
    torch.testing.assert_close(actual_high, expected_high, rtol=0, atol=0)


def test_decoding_only_after_full_k_is_not_valid() -> None:
    # Each K=32 low digit is legal, but two tiles sum to 64 and carry into the
    # high digit.  This is the exact failure the kernel must avoid.
    packed_full = torch.tensor([2 * TILE_K], dtype=torch.int32)
    decoded_low, decoded_high = decode_balanced_tile(packed_full)
    assert (int(decoded_low), int(decoded_high)) == (-1, 1)
    assert (int(decoded_low), int(decoded_high)) != (2 * TILE_K, 0)


def test_ffn_physical_ledger_is_exactly_matched() -> None:
    ledger = matched_ffn_ledger(hidden_size=384, baseline_width=1024)
    assert ledger.candidate_width == 1536
    assert ledger.candidate_resident_weight_bytes == ledger.baseline_resident_weight_bytes
    assert ledger.candidate_int8_scalar_products == ledger.baseline_int8_scalar_products
    assert (
        ledger.candidate_independent_ternary_weight_digits
        == 3 * ledger.baseline_independent_ternary_weight_digits // 2
    )
