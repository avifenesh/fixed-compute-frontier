import math

import numpy as np
import pytest

import experiments.radial_trust_cayley_b2_fp64_oracle as oracle
from experiments.radial_trust_cayley_b2_spec import (
    build_packed_descriptors,
    fp32_from_bf16_bits,
    iter_candidate_layer_draws,
)


WIDTH = 8
NODES = 5


def small_artifacts() -> oracle.OracleLayerArtifacts:
    draws = dict(iter_candidate_layer_draws(0, width=WIDTH, nodes=NODES))
    return oracle.OracleLayerArtifacts(
        basis_bf16=draws["basis"],
        payload_bf16=draws["payload"],
        threshold_bf16=draws["threshold"],
    )


def zero_bf16(shape: tuple[int, ...]) -> np.ndarray:
    return np.zeros(shape, dtype="<u2")


def bf16_bits(values: np.ndarray) -> np.ndarray:
    fp32 = np.asarray(values, dtype="<f4")
    raw = fp32.view("<u4")
    bias = np.uint32(0x7FFF) + ((raw >> np.uint32(16)) & np.uint32(1))
    return ((raw + bias) >> np.uint32(16)).astype("<u2")


def test_formula_topology_matches_packed_descriptor_words() -> None:
    topology = oracle.build_formula_topology(WIDTH)
    descriptors = build_packed_descriptors(WIDTH)
    assert topology.partner.shape == (3, 3, WIDTH)
    for basis in range(3):
        for matching in range(3):
            for output in range(WIDTH):
                word = int(descriptors[basis, matching, output])
                assert int(topology.partner[basis, matching, output]) == word & 0xFFF
                assert int(topology.pair[basis, matching, output]) == (word >> 12) & 0x7FF
                assert int(topology.sign[basis, matching, output]) == (word >> 23) & 1


def test_frozen_width_route_coordinate_literals() -> None:
    assert oracle.route_coordinate(0, 0, 0, 4096) == 542
    assert oracle.route_coordinate(1, 0, 0, 4096) == 141
    assert oracle.route_coordinate(0, 12, 0, 4096) == 1226
    assert oracle.route_coordinate(0, 0, 11, 4096) == 1585
    assert oracle.route_coordinate(7165, 12, 11, 4096) == 400


def test_sparse_basis_is_skew_and_cayley_polynomials_are_transposes() -> None:
    artifacts = small_artifacts()
    weights = oracle.fp64_from_bf16_bits(artifacts.basis_bf16)
    topology = oracle.build_formula_topology(WIDTH)
    rng = np.random.Generator(np.random.PCG64(17))
    left = rng.normal(size=(3, WIDTH))
    right = rng.normal(size=(3, WIDTH))
    for basis in range(3):
        a_left = oracle.apply_sparse_skew_fp64(left, weights, topology, basis)
        a_right = oracle.apply_sparse_skew_fp64(right, weights, topology, basis)
        np.testing.assert_allclose(
            np.sum(a_left * right, axis=1),
            -np.sum(left * a_right, axis=1),
            rtol=0,
            atol=2e-15,
        )
        c_left = oracle.cayley_neumann_fp64(left, weights, topology, basis)
        ct_right = oracle.cayley_neumann_fp64(
            right, weights, topology, basis, transpose=True
        )
        np.testing.assert_allclose(
            np.sum(c_left * right, axis=1),
            np.sum(left * ct_right, axis=1),
            rtol=0,
            atol=4e-15,
        )


def test_zero_down_payload_gives_zero_output_and_exact_forced_paths() -> None:
    artifacts = small_artifacts()
    payload = artifacts.payload_bf16.copy()
    payload[:, :, 2] = 0
    zero_down = oracle.OracleLayerArtifacts(
        basis_bf16=artifacts.basis_bf16,
        payload_bf16=payload,
        threshold_bf16=artifacts.threshold_bf16,
    )
    hidden = np.arange(2 * WIDTH, dtype=np.float64).reshape(2, WIDTH) / 8.0
    bits = np.array([[0, 0, 1], [1, 1, 1]], dtype=np.uint8)
    result = oracle.run_forced_oracle(
        hidden,
        zero_down,
        bits,
        layer=0,
        width=WIDTH,
        nodes=NODES,
    )
    np.testing.assert_array_equal(result.output, np.zeros_like(hidden))
    np.testing.assert_array_equal(result.final_hidden, hidden)
    assert [(step.node, step.bit) for step in result.traces[0]] == [
        (0, 0),
        (1, 0),
        (3, 1),
    ]
    assert [(step.node, step.bit) for step in result.traces[1]] == [(0, 1), (2, 1)]
    assert [step.coordinate for step in result.traces[0]] == [6, 6, 5]
    assert [step.coordinate for step in result.traces[1]] == [6, 5]
    assert all(step.raw_delta_rms == 0.0 for trace in result.traces for step in trace)
    assert all(step.trusted_delta_rms == 0.0 for trace in result.traces for step in trace)


def test_zero_basis_matches_manual_identity_basis_recurrence() -> None:
    one = bf16_bits(np.ones((NODES, 2, WIDTH), dtype=np.float32))
    payload = zero_bf16((NODES, 2, 3, WIDTH))
    payload[:, :, 0] = one
    payload[:, :, 1] = one
    payload[:, :, 2] = one
    artifacts = oracle.OracleLayerArtifacts(
        basis_bf16=zero_bf16((3, 3, WIDTH // 2)),
        payload_bf16=payload,
        threshold_bf16=zero_bf16((NODES,)),
    )
    hidden = np.array([[0.25, -0.5, 1.0, -1.5, 2.0, -2.5, 0.75, -0.125]])
    bits = np.array([[1, 0, 0]], dtype=np.uint8)
    expected = hidden.copy()
    for _ in range(2):
        activation = expected / (1.0 + np.exp(-expected)) * expected
        raw_delta = activation
        trusted = raw_delta / np.sqrt(1.0 + np.mean(raw_delta * raw_delta, axis=1))[:, None]
        expected = expected + oracle.RESIDUAL_SCALE * trusted
    result = oracle.run_forced_oracle(
        hidden,
        artifacts,
        bits,
        layer=3,
        width=WIDTH,
        nodes=NODES,
    )
    np.testing.assert_allclose(result.final_hidden, expected, rtol=0, atol=5e-16)
    np.testing.assert_allclose(result.output, expected - hidden, rtol=0, atol=5e-16)
    assert [(step.node, step.bit) for step in result.traces[0]] == [(0, 1), (2, 0)]


def test_trusted_delta_rms_is_strictly_below_one_for_nonzero_updates() -> None:
    hidden = fp32_from_bf16_bits(
        bf16_bits(np.linspace(-2.0, 2.0, WIDTH, dtype=np.float32))[None, :]
    ).astype(np.float64)
    result = oracle.run_forced_oracle(
        hidden,
        small_artifacts(),
        np.array([[0, 1, 0]], dtype=np.int8),
        layer=11,
        width=WIDTH,
        nodes=NODES,
    )
    assert result.traces[0]
    assert [step.coordinate for step in result.traces[0]] == [1, 1, 7]
    assert all(math.isfinite(step.raw_delta_rms) for step in result.traces[0])
    assert all(0.0 <= step.trusted_delta_rms < 1.0 for step in result.traces[0])


def _independent_dense_bases(basis_bf16: np.ndarray) -> np.ndarray:
    weights = (
        basis_bf16.astype("<u4") << np.uint32(16)
    ).view("<f4").astype(np.float64)
    matrices = np.zeros((3, WIDTH, WIDTH), dtype=np.float64)
    seeds = (21, 124, 233, 93, 160, 191, 18, 31, 38)
    multipliers = (61, 359, 169, 373, 131, 115, 113, 19, 29)
    for flat, (seed, multiplier) in enumerate(zip(seeds, multipliers, strict=True)):
        permutation = []
        for position in range(WIDTH):
            left_q = 6 * (2 * seed + 1)
            left_offset = 97 * seed + 17
            right_q = 6 * (2 * ((53 * seed + 7) % 257) + 1)
            right_offset = 193 * seed + 29
            first = (position + left_q * position**2) % WIDTH
            second = (multiplier * first + left_offset) % WIDTH
            permutation.append((second + right_q * second**2 + right_offset) % WIDTH)
        basis, matching = divmod(flat, 3)
        for pair in range(WIDTH // 2):
            positive = permutation[2 * pair]
            negative = permutation[2 * pair + 1]
            weight = weights[basis, matching, pair]
            matrices[basis, positive, negative] += weight
            matrices[basis, negative, positive] -= weight
    return matrices


def _independent_cayley(vector: np.ndarray, matrix: np.ndarray, transpose: bool) -> np.ndarray:
    term = vector.copy()
    total = vector.copy()
    sign = 1.0 if transpose else -1.0
    for _ in range(4):
        term = sign * (matrix @ term)
        total = total + term
    return 2.0 * total - vector


def _independent_nonzero_expected(
    hidden: np.ndarray,
    artifacts: oracle.OracleLayerArtifacts,
    bits: np.ndarray,
) -> np.ndarray:
    matrices = _independent_dense_bases(artifacts.basis_bf16)
    payload = (
        artifacts.payload_bf16.astype("<u4") << np.uint32(16)
    ).view("<f4").astype(np.float64)
    output = np.empty_like(hidden)
    for token in range(hidden.shape[0]):
        state = hidden[token].copy()
        node = 0
        depth = 0
        while node < NODES:
            bit = int(bits[token, depth])
            basis = depth % 3
            projected = _independent_cayley(state, matrices[basis], False)
            gate, up, down = payload[node, bit]
            gate_value = gate * projected
            activation = gate_value / (1.0 + np.exp(-gate_value))
            local_delta = down * activation * (up * projected)
            raw_delta = _independent_cayley(local_delta, matrices[basis], True)
            rms = math.sqrt(
                math.fsum(float(value * value) for value in raw_delta) / WIDTH
            )
            trusted = raw_delta / math.hypot(1.0, rms)
            state = state + oracle.RESIDUAL_SCALE * trusted
            node = 2 * node + 1 + bit
            depth += 1
        output[token] = state - hidden[token]
    return output


def test_asymmetric_nonzero_end_to_end_matches_independent_dense_calculation() -> None:
    artifacts = small_artifacts()
    hidden = np.array(
        [
            [0.5, -1.25, 0.75, 2.0, -0.125, 1.5, -2.25, 0.375],
            [-0.8, 0.2, 1.1, -1.7, 2.4, -0.3, 0.6, 1.9],
        ],
        dtype=np.float64,
    )
    bits = np.array([[0, 0, 1], [1, 0, 1]], dtype=np.uint8)
    result = oracle.run_forced_oracle(
        hidden,
        artifacts,
        bits,
        layer=5,
        width=WIDTH,
        nodes=NODES,
    )
    expected = _independent_nonzero_expected(hidden, artifacts, bits)
    np.testing.assert_allclose(result.output, expected, rtol=0, atol=3e-15)
    assert not np.array_equal(result.output[0], result.output[1])


def test_radial_normalization_is_stable_when_naive_square_would_overflow() -> None:
    artifacts = oracle.OracleLayerArtifacts(
        basis_bf16=bf16_bits(np.full((3, 3, WIDTH // 2), 2.0**43)),
        payload_bf16=bf16_bits(np.ones((1, 2, 3, WIDTH))),
        threshold_bf16=zero_bf16((1,)),
    )
    result = oracle.run_forced_oracle(
        np.ones((1, WIDTH)),
        artifacts,
        np.array([[0]], dtype=np.uint8),
        layer=0,
        width=WIDTH,
        nodes=1,
    )
    step = result.traces[0][0]
    assert math.isfinite(step.raw_delta_rms) and step.raw_delta_rms > 1e154
    assert math.isfinite(step.trusted_delta_rms)
    assert 0.0 < step.trusted_delta_rms <= 1.0
    assert np.isfinite(result.output).all()
    assert np.count_nonzero(result.output) > 0


@pytest.mark.parametrize(
    "bits",
    [
        np.array([[0]], dtype=np.uint8),
        np.array([[0, 2, 0]], dtype=np.uint8),
        np.array([[0.0, 1.0, 0.0]], dtype=np.float64),
    ],
)
def test_malformed_or_incomplete_forced_paths_are_rejected(bits: np.ndarray) -> None:
    with pytest.raises(ValueError):
        oracle.run_forced_oracle(
            np.ones((1, WIDTH)),
            small_artifacts(),
            bits,
            layer=0,
            width=WIDTH,
            nodes=NODES,
        )


def test_nonfinite_bf16_and_wrong_payload_layout_are_rejected() -> None:
    artifacts = small_artifacts()
    invalid_basis = artifacts.basis_bf16.copy()
    invalid_basis[0, 0, 0] = np.uint16(0x7F80)
    with pytest.raises(ValueError, match="NaN or infinity"):
        oracle.run_forced_oracle(
            np.ones((1, WIDTH)),
            oracle.OracleLayerArtifacts(
                invalid_basis, artifacts.payload_bf16, artifacts.threshold_bf16
            ),
            np.array([[1, 0, 0]], dtype=np.uint8),
            layer=0,
            width=WIDTH,
            nodes=NODES,
        )
    with pytest.raises(ValueError, match="payload_bf16"):
        oracle.run_forced_oracle(
            np.ones((1, WIDTH)),
            oracle.OracleLayerArtifacts(
                artifacts.basis_bf16,
                artifacts.payload_bf16[:, :, :, ::-1],
                artifacts.threshold_bf16,
            ),
            np.array([[1, 0, 0]], dtype=np.uint8),
            layer=0,
            width=WIDTH,
            nodes=NODES,
        )


def test_frozen_wrapper_has_no_dimension_or_residual_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    sentinel = object()

    def fake_run(hidden, artifacts, forced_bits, **kwargs):
        calls.append((hidden, artifacts, forced_bits, kwargs))
        return sentinel

    monkeypatch.setattr(oracle, "run_forced_oracle", fake_run)
    hidden = bf16_bits(np.ones((1, 4096), dtype=np.float32))
    artifacts = object()
    bits = object()
    assert oracle.run_frozen_forced_oracle(hidden, artifacts, bits, layer=7) is sentinel
    assert len(calls) == 1
    decoded, observed_artifacts, observed_bits, kwargs = calls[0]
    np.testing.assert_array_equal(decoded, np.ones((1, 4096), dtype=np.float64))
    assert observed_artifacts is artifacts
    assert observed_bits is bits
    assert kwargs == {
        "layer": 7,
        "width": 4096,
        "nodes": 7166,
        "residual_scale": oracle.RESIDUAL_SCALE,
    }
    with pytest.raises(ValueError, match="hidden_bf16"):
        oracle.run_frozen_forced_oracle(
            np.ones((1, 4096), dtype=np.float64), artifacts, bits, layer=7
        )
    with pytest.raises(ValueError, match="hidden_bf16"):
        oracle.run_frozen_forced_oracle(hidden[:, ::-1], artifacts, bits, layer=7)
