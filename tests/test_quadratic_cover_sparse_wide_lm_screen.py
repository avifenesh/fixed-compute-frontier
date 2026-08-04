import torch

from experiments.quadratic_cover_sparse_wide_lm_screen import (
    D, M, WIDTH, MaskedLinear, degree_two_support_counts,
    quadratic_cover_codebook, quadratic_cover_masks, random_2of4_mask,
    serving_ledger,
)


def test_codebook_has_exact_uniform_degree_two_coverage():
    codebook = quadratic_cover_codebook()
    assert len(codebook) == WIDTH == 1792
    assert set(degree_two_support_counts(codebook).values()) == {640}
    for coordinate in range(4):
        assert sum(coordinate in gate for gate, _ in codebook) == WIDTH // 2
        assert sum(coordinate in up for _, up in codebook) == WIDTH // 2


def test_materialized_cover_masks_are_exactly_two_of_four():
    gate, up = quadratic_cover_masks(layer_index=3)
    for mask in (gate, up):
        assert mask.shape == (WIDTH, D)
        assert torch.all(mask.reshape(WIDTH, D // 4, 4).sum(dim=-1) == 2)


def test_random_mask_is_exactly_two_of_four_and_reproducible():
    first = random_2of4_mask(17, 20, seed=9)
    second = random_2of4_mask(17, 20, seed=9)
    assert torch.equal(first, second)
    assert torch.all(first.reshape(17, 5, 4).sum(dim=-1) == 2)


def test_serving_ledger_is_exact_and_below_dense():
    ledger = serving_ledger()
    assert ledger["baseline_ffn_bytes_per_layer"] == 2_359_296
    assert ledger["candidate_value_bytes_per_layer"] == 2_064_384
    assert ledger["candidate_metadata_bytes_per_layer"] == 258_048
    assert ledger["candidate_ffn_bytes_per_layer"] == 2_322_432
    assert ledger["candidate_nonzero_macs_per_token"] == 1_032_192
    assert ledger["width_ratio"] == 1.75
    assert ledger["nonzero_mac_ratio"] == 0.875
    assert ledger["fits_bytes"] and ledger["fits_nonzero_macs"]


def test_masked_linear_blocks_masked_gradients():
    mask = random_2of4_mask(8, 12, seed=11)
    layer = MaskedLinear(8, 12, mask, std=0.1)
    inputs = torch.randn(5, 12)
    layer(inputs).square().mean().backward()
    assert torch.all(layer.weight.grad[~mask] == 0)
    assert torch.any(layer.weight.grad[mask] != 0)


def test_sparse_nonzero_count_matches_three_projection_formula():
    gate, up = quadratic_cover_masks(layer_index=0)
    assert int(gate.sum() + up.sum()) == WIDTH * D
    assert serving_ledger()["candidate_nonzero_macs_per_token"] == 3 * D * WIDTH // 2
    assert serving_ledger()["baseline_matrix_macs_per_token"] == 3 * D * M

