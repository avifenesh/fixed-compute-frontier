import math
import hashlib
import json

import numpy as np
import pytest

import experiments.radial_trust_cayley_b2_spec as b2_spec

from experiments.radial_trust_cayley_b2_spec import (
    AFFINE_MULTIPLIERS,
    D,
    E,
    MEAN_MULTIPLIER_F32_BITS,
    N,
    RESIDUAL_SCALE_F32_BITS,
    TOPOLOGY_SEEDS,
    bf16_rne_from_fp64,
    build_packed_descriptors,
    counter_range_order,
    decode_descriptor,
    energy_loop_order,
    evaluation_row,
    fp32_from_bf16_bits,
    generate_trace,
    iter_candidate_layer_draws,
    leaf_depth_counts,
    memory_pair_order,
    normalize_rows_fp64,
    operation_ledger,
    path_from_bits,
    resident_ledger,
    route_coordinate,
    timing_chunk_order,
    tensor_sha256,
    tuning_row,
    verify_packed_descriptors,
)


def test_exact_scalar_and_resident_budget() -> None:
    ledger = resident_ledger()
    assert E == 14_332
    assert ledger.candidate_layer_scalars == 176_137_214
    assert ledger.dense_layer_scalars == 176_160_768
    assert ledger.dense_layer_scalars - ledger.candidate_layer_scalars == 23_554
    assert ledger.candidate_layer_allocated_bytes == 352_274_432
    assert ledger.dense_layer_allocated_bytes == 352_321_536
    assert ledger.topology_bytes == 147_456
    assert ledger.candidate_panel_bytes == 4_227_440_640
    assert ledger.dense_panel_bytes == 4_227_858_432
    assert ledger.candidate_minus_dense_panel_bytes == -417_792


def test_packed_topology_is_formula_derived_signed_involution() -> None:
    descriptors = build_packed_descriptors()
    assert descriptors.shape == (3, 3, D)
    assert descriptors.nbytes == 147_456
    assert not np.any(descriptors >> np.uint32(24))
    verify_packed_descriptors(descriptors)
    for descriptor in descriptors.reshape(-1)[::997]:
        partner, pair, sign = decode_descriptor(int(descriptor))
        assert 0 <= partner < D
        assert 0 <= pair < D // 2
        assert sign in (0, 1)


def test_descriptors_match_independent_scalar_formula_exhaustively() -> None:
    descriptors = build_packed_descriptors()
    for flat_index, (seed, multiplier) in enumerate(
        zip(TOPOLOGY_SEEDS, AFFINE_MULTIPLIERS, strict=True)
    ):
        permutation = []
        for position in range(D):
            left_q = 6 * (2 * seed + 1)
            left_offset = 97 * seed + 17
            right_q = 6 * (2 * ((53 * seed + 7) % 257) + 1)
            right_offset = 193 * seed + 29
            first = (position + left_q * position**2) % D
            second = (multiplier * first + left_offset) % D
            permutation.append((second + right_q * second**2 + right_offset) % D)
        basis, matching = divmod(flat_index, 3)
        for pair in range(D // 2):
            positive = permutation[2 * pair]
            negative = permutation[2 * pair + 1]
            assert int(descriptors[basis, matching, positive]) == (
                negative | (pair << 12)
            )
            assert int(descriptors[basis, matching, negative]) == (
                positive | (pair << 12) | (1 << 23)
            )
            assert decode_descriptor(descriptors[basis, matching, positive]) == (
                negative,
                pair,
                0,
            )
            assert decode_descriptor(descriptors[basis, matching, negative]) == (
                positive,
                pair,
                1,
            )


def test_incomplete_tree_has_only_the_frozen_depths() -> None:
    assert leaf_depth_counts() == {12: 1025, 13: 6142}
    expected = 12 * 1025 / 4096 + 13 * 6142 / 8192
    assert math.isclose(expected, 52223 / 4096)
    short_nodes, short_leaf = path_from_bits([1] * 12)
    assert len(short_nodes) == 12
    assert short_leaf >= N
    long_nodes, long_leaf = path_from_bits([0] * 13)
    assert len(long_nodes) == 13
    assert long_leaf >= N


def test_route_coordinate_uses_u32_wrap_then_mask() -> None:
    for layer in (0, 5, 11):
        for depth in (0, 7, 12):
            for node in (0, 1, N - 1):
                seed = 815 + 104_729 * layer + 7_919
                expected = (
                    (
                        node * 1_103_515_247
                        + seed
                        + depth * 12_345
                    )
                    & 0xFFFFFFFF
                ) & 4095
                assert route_coordinate(node, depth, layer) == expected


def test_bf16_recipe_is_ties_to_even_and_roundtrips_bits() -> None:
    fp32_bits = np.array(
        [
            0x3F808000,  # halfway between 1.0 and the next BF16: lower is even
            0x3F818000,  # halfway with odd lower BF16: round upward
            0xBF808000,
            0x00000000,
            0x80000000,
        ],
        dtype="<u4",
    )
    values = fp32_bits.view("<f4").astype(np.float64)
    packed = bf16_rne_from_fp64(values)
    assert packed.tolist() == [0x3F80, 0x3F82, 0xBF80, 0x0000, 0x8000]
    reconstructed = fp32_from_bf16_bits(packed)
    assert reconstructed.view("<u4").tolist() == [
        0x3F800000,
        0x3F820000,
        0xBF800000,
        0x00000000,
        0x80000000,
    ]


def test_scalar_normalization_and_row_maps() -> None:
    source = np.array([[3.0, 4.0], [-1.0, 1.0]], dtype=np.float64)
    normalized = normalize_rows_fp64(source)
    np.testing.assert_allclose(
        np.sqrt(np.mean(normalized * normalized, axis=1)),
        np.ones(2),
        rtol=0,
        atol=2e-16,
    )
    assert evaluation_row(9, 999, 8, 7, 11) == (
        ((1000 * 9 + 999) * 8 + 7 + 683 * 11) & 8191
    )
    assert tuning_row(-200, 8, 7, 11) == (
        (-200 * 8 + 7 + 683 * 11) % 2048
    )


def test_frozen_trace_generators_and_artifact_hashes() -> None:
    gaussian_fp64, gaussian_bf16 = generate_trace(
        "gaussian", rows=3, width=8, seed=9_000_815
    )
    rademacher_fp64, rademacher_bf16 = generate_trace(
        "rademacher", rows=3, width=8, seed=9_100_815
    )
    assert gaussian_fp64.shape == rademacher_fp64.shape == (3, 8)
    assert gaussian_bf16.shape == rademacher_bf16.shape == (3, 8)
    assert gaussian_bf16.dtype == rademacher_bf16.dtype == np.dtype("<u2")
    assert tensor_sha256(gaussian_fp64) == (
        "9fcdb9e231e807d1140b3aa2baf2b65c692f7618dd9fb7d8115b82d0186c3184"
    )
    assert tensor_sha256(gaussian_bf16) == (
        "4f49d53a2b3eec77022788e9c2a899a59bd109b493bb47a72287e58a53202e55"
    )
    assert tensor_sha256(rademacher_fp64) == (
        "dd7000df144b85950b8540a014c04af73fdeba601a63615cb4303c965fece43d"
    )
    assert tensor_sha256(rademacher_bf16) == (
        "7c7b8e701f8d4452e8ecc3066b0f56d8f8f150fd507fef1604ec6b077092e82e"
    )


def test_frozen_trace_wrapper_has_only_four_exact_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    sentinel = (np.empty((0, 0)), np.empty((0, 0), dtype="<u2"))

    def fake_generate_trace(family, *, rows, width, seed):
        calls.append((family, rows, width, seed))
        return sentinel

    monkeypatch.setattr(b2_spec, "generate_trace", fake_generate_trace)
    assert b2_spec.generate_frozen_trace("gaussian") is sentinel
    assert b2_spec.generate_frozen_trace("rademacher") is sentinel
    assert b2_spec.generate_frozen_trace("gaussian", tuning=True) is sentinel
    assert b2_spec.generate_frozen_trace("rademacher", tuning=True) is sentinel
    assert calls == [
        ("gaussian", 8192, 4096, 9_000_815),
        ("rademacher", 8192, 4096, 9_100_815),
        ("gaussian", 2048, 4096, 8_000_815),
        ("rademacher", 2048, 4096, 8_100_815),
    ]


def test_candidate_draw_order_packs_one_payload_allocation() -> None:
    draws = dict(iter_candidate_layer_draws(0, width=8, nodes=5))
    assert tuple(draws) == ("basis", "payload", "threshold")
    assert draws["basis"].shape == (3, 3, 4)
    assert draws["payload"].shape == (5, 2, 3, 8)
    assert draws["threshold"].shape == (5,)
    assert draws["basis"].dtype == np.dtype("<u2")
    assert draws["payload"].dtype == np.dtype("<u2")
    assert tensor_sha256(draws["basis"]) == (
        "6d5044390fc5f202bbd709ed0a011c9073287c3f07c93c03ba39cbc2e2b5425a"
    )
    assert tensor_sha256(draws["payload"]) == (
        "641525b5fd3e679699a4be25fab367d4201c2d49a2ef028ecaab838f631f8bcd"
    )
    assert tensor_sha256(draws["threshold"]) == (
        "01d448afd928065458cf670b60f5a594d735af0172c8d67f22a81680132681ca"
    )
    expected_bytes = draws["basis"].tobytes(order="C")
    assert tensor_sha256(draws["basis"]) == hashlib.sha256(expected_bytes).hexdigest()

    rng = np.random.Generator(np.random.PCG64(815))
    raw_basis = np.empty((3, 3, 4), dtype=np.float64)
    for basis in range(3):
        for matching in range(3):
            raw_basis[basis, matching] = rng.normal(0.0, 1.0, size=(4,))
    expected_basis = bf16_rne_from_fp64(np.tanh(raw_basis) * np.float64(1 / 12))
    expected_payload = np.empty((5, 2, 3, 8), dtype="<u2")
    expected_payload[:, :, 0] = bf16_rne_from_fp64(
        rng.normal(1.0, 0.02, size=(5, 2, 8))
    )
    expected_payload[:, :, 1] = bf16_rne_from_fp64(
        rng.normal(1.0, 0.02, size=(5, 2, 8))
    )
    expected_payload[:, :, 2] = bf16_rne_from_fp64(
        rng.normal(0.0, 0.15, size=(5, 2, 8))
    )
    assert np.array_equal(draws["basis"], expected_basis)
    assert np.array_equal(draws["payload"], expected_payload)
    assert np.count_nonzero(draws["threshold"]) == 0


def test_operation_ledger_matches_paper_max_depth_totals() -> None:
    ledger = operation_ledger(13)
    assert ledger.sparse_applications == 104
    assert ledger.multiplications_before_silu_expansion == 1_757_197
    assert ledger.additions_before_final_output == 1_490_957
    assert ledger.silu_evaluations == 53_248
    assert ledger.square_roots == 13
    assert ledger.scalar_divisions == 13
    assert ledger.final_output_subtractions == 4096
    assert ledger.descriptor_requested_bytes == 5_111_808
    assert ledger.coefficient_requested_bytes == 2_555_904
    residual_scale = np.array([RESIDUAL_SCALE_F32_BITS], dtype="<u4").view("<f4")[0]
    mean_multiplier = np.array([MEAN_MULTIPLIER_F32_BITS], dtype="<u4").view("<f4")[0]
    assert residual_scale == np.float32(0.2773500978946686)
    assert mean_multiplier == np.float32(2**-12)


def test_all_frozen_orders_are_permutations_and_deterministic() -> None:
    timing = timing_chunk_order(3, 7)
    assert timing == timing_chunk_order(3, 7)
    assert len(timing) == len(set(timing)) == 32
    counters = counter_range_order(4)
    energy = energy_loop_order(4)
    assert len(counters) == len(set(counters)) == 16
    assert len(energy) == len(set(energy)) == 16
    assert counters != energy
    assert set(memory_pair_order(0)) == {"candidate", "dense"}
    timing_golden = timing_chunk_order(0, 0)
    counters_golden = counter_range_order(0)
    energy_golden = energy_loop_order(0)
    assert timing_golden[:5] == (
        ("gaussian", 8, "moe"),
        ("rademacher", 4, "dense"),
        ("gaussian", 8, "candidate"),
        ("gaussian", 4, "moe"),
        ("gaussian", 2, "candidate"),
    )
    assert timing_golden[-5:] == (
        ("rademacher", 4, "monarch"),
        ("gaussian", 2, "dense"),
        ("gaussian", 1, "monarch"),
        ("rademacher", 1, "monarch"),
        ("rademacher", 1, "candidate"),
    )
    assert counters_golden[:5] == (
        ("rademacher", 4, "dense"),
        ("rademacher", 1, "dense"),
        ("gaussian", 2, "dense"),
        ("rademacher", 4, "candidate"),
        ("rademacher", 8, "candidate"),
    )
    assert energy_golden[:5] == (
        ("rademacher", 2, "candidate"),
        ("rademacher", 1, "dense"),
        ("gaussian", 8, "candidate"),
        ("rademacher", 2, "dense"),
        ("gaussian", 2, "dense"),
    )
    assert [memory_pair_order(process) for process in range(5)] == [
        ("dense", "candidate"),
        ("candidate", "dense"),
        ("dense", "candidate"),
        ("candidate", "dense"),
        ("candidate", "dense"),
    ]
    schedule_hash = lambda sequence: hashlib.sha256(
        json.dumps(sequence, separators=(",", ":"), ensure_ascii=True).encode(
            "ascii"
        )
    ).hexdigest()
    assert schedule_hash(timing_golden) == (
        "26120b614b32184dd60fe51842c0c76fe9de32233f176868885c54ebc42556ae"
    )
    assert schedule_hash(counters_golden) == (
        "7e1bb49d0cf624aff66011f2575d606575e2e6d2dd26cf8a110448cec60c9f52"
    )
    assert schedule_hash(energy_golden) == (
        "2a6b94edb9c53f60ddee9d79f444f0e0d22f94f1ef35282a91ee821e8fa5d306"
    )
    assert schedule_hash(tuple(memory_pair_order(p) for p in range(5))) == (
        "6f5e367025b81ad16849ccfaf180e52cc68e585629b1172e7c216768a70a6bc4"
    )
