import ctypes
import math
import struct

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import experiments.radial_trust_cayley_b2_staged_reference as staged
from experiments.radial_trust_cayley_b2_spec import (
    build_packed_descriptors,
    iter_candidate_layer_draws,
)
from experiments.radial_trust_cayley_b2_trace_checker import (
    RouteStep,
    check_route_steps,
)


WIDTH = 8
NODES = 5
CPU = torch.device("cpu")


def small_artifacts() -> staged.TorchLayerArtifacts:
    draws = dict(iter_candidate_layer_draws(0, width=WIDTH, nodes=NODES))
    return staged.TorchLayerArtifacts(
        basis=staged.torch_bf16_from_raw_bits(draws["basis"], device=CPU),
        payload=staged.torch_bf16_from_raw_bits(draws["payload"], device=CPU),
        threshold=staged.torch_bf16_from_raw_bits(draws["threshold"], device=CPU),
    )


def test_raw_bf16_decode_is_exact_and_requires_c_order() -> None:
    bits = np.array([0x3F80, 0xBF80, 0x0001, 0x8000], dtype="<u2")
    tensor = staged.torch_bf16_from_raw_bits(bits, device=CPU)
    assert tensor.dtype == torch.bfloat16
    assert tensor.view(torch.uint16).tolist() == bits.tolist()
    with pytest.raises(ValueError, match="C-order"):
        staged.torch_bf16_from_raw_bits(bits[::-1], device=CPU)
    invalid = bits.copy()
    invalid[0] = np.uint16(0x7F80)
    with pytest.raises(ValueError, match="NaN or infinity"):
        staged.torch_bf16_from_raw_bits(invalid, device=CPU)


def test_formula_topology_matches_every_small_packed_descriptor() -> None:
    topology = staged.build_formula_topology(WIDTH, device=CPU)
    descriptors = build_packed_descriptors(WIDTH)
    for basis in range(3):
        for matching in range(3):
            for output in range(WIDTH):
                word = int(descriptors[basis, matching, output])
                assert int(topology.partner[basis, matching, output]) == word & 0xFFF
                assert int(topology.pair[basis, matching, output]) == (word >> 12) & 0x7FF
                assert bool(topology.sign[basis, matching, output]) == bool((word >> 23) & 1)


def test_formula_topology_matches_every_target_width_descriptor() -> None:
    topology = staged.build_formula_topology(4096, device=CPU)
    words = torch.from_numpy(build_packed_descriptors().astype(np.int64, copy=False))
    assert torch.equal(topology.partner, words.bitwise_and(0xFFF))
    assert torch.equal(topology.pair, words.bitwise_right_shift(12).bitwise_and(0x7FF))
    assert torch.equal(topology.sign, words.bitwise_right_shift(23).bitwise_and(1).bool())


def test_target_width_route_literals_and_residual_bits() -> None:
    cases = (
        (0, 0, 0, 542),
        (1, 0, 0, 141),
        (0, 12, 0, 1226),
        (0, 0, 11, 1585),
        (7165, 12, 11, 400),
    )
    for node, depth, layer, expected in cases:
        observed = staged.route_coordinates(
            torch.tensor([node]), depth, layer, 4096
        )
        assert observed.item() == expected
    assert struct.unpack("<I", struct.pack("<f", staged.RESIDUAL_SCALE_F32))[0] == (
        0x3E8E00D5
    )


def test_zero_down_payload_has_zero_output_and_checker_valid_routes() -> None:
    artifacts = small_artifacts()
    payload = artifacts.payload.clone()
    payload[:, :, 2] = 0
    zero_down = staged.TorchLayerArtifacts(
        artifacts.basis, payload, artifacts.threshold
    )
    hidden = torch.arange(2 * WIDTH, dtype=torch.float32).reshape(2, WIDTH).to(
        torch.bfloat16
    )
    bits = torch.tensor([[0, 0, 1], [1, 1, 1]], dtype=torch.uint8)
    result = staged.run_staged_reference(
        hidden,
        zero_down,
        layer=0,
        width=WIDTH,
        nodes=NODES,
        forced_bits=bits,
    )
    assert torch.equal(result.output, torch.zeros_like(hidden))
    assert torch.equal(result.final_hidden, hidden)
    expected_nodes = [[0, 1, 3], [0, 2]]
    expected_bits = [[0, 0, 1], [1, 1]]
    for token, trace in enumerate(result.traces):
        assert [step.node for step in trace] == expected_nodes[token]
        semantic = tuple(
            RouteStep(
                step.depth,
                step.node,
                step.bit,
                step.coordinate,
                step.payload_edge,
            )
            for step in trace
        )
        checked = check_route_steps(
            semantic,
            layer=0,
            width=WIDTH,
            nodes=NODES,
            expected_bits=expected_bits[token],
        )
        assert checked.length == len(expected_nodes[token])
        assert all(step.trust_scale == 1.0 for step in trace)


def test_natural_zero_tie_routes_right_and_stops_at_exact_leaf() -> None:
    artifacts = staged.TorchLayerArtifacts(
        basis=torch.zeros((3, 3, WIDTH // 2), dtype=torch.bfloat16),
        payload=torch.zeros((NODES, 2, 3, WIDTH), dtype=torch.bfloat16),
        threshold=torch.zeros((NODES,), dtype=torch.bfloat16),
    )
    result = staged.run_staged_reference(
        torch.zeros((1, WIDTH), dtype=torch.bfloat16),
        artifacts,
        layer=11,
        width=WIDTH,
        nodes=NODES,
    )
    assert [(step.node, step.bit) for step in result.traces[0]] == [(0, 1), (2, 1)]
    assert [step.coordinate for step in result.traces[0]] == [1, 0]
    assert torch.equal(result.output, torch.zeros((1, WIDTH), dtype=torch.bfloat16))


def test_fixed_pairwise_reduction_uses_the_declared_tree() -> None:
    values = torch.tensor(
        [[1.0, 2.0**24, -2.0**24, 3.0, 5.0, -5.0, 7.0, 9.0]],
        dtype=torch.float32,
    )
    observed = staged.fixed_pairwise_sum(values)
    first = values[:, :4] + values[:, 4:]
    second = first[:, :2] + first[:, 2:]
    expected = second[:, 0] + second[:, 1]
    assert torch.equal(observed, expected)


def test_matching_order_is_protected_by_bf16_cancellation_literal() -> None:
    basis = torch.zeros((3, 3, WIDTH // 2), dtype=torch.bfloat16)
    basis[0, 0, 0] = 2.0**24
    basis[0, 1, 2] = -(2.0**24)
    basis[0, 2, 0] = 1.0
    artifacts = staged.TorchLayerArtifacts(
        basis=basis,
        payload=torch.zeros((NODES, 2, 3, WIDTH), dtype=torch.bfloat16),
        threshold=torch.zeros((NODES,), dtype=torch.bfloat16),
    )
    observed = staged.sparse_apply_staged(
        torch.ones((1, WIDTH), dtype=torch.bfloat16),
        artifacts,
        staged.build_formula_topology(WIDTH, device=CPU),
        0,
        transpose_sign=False,
    )
    assert observed[0, 0].view(torch.uint16).item() == 0x3F80
    frozen_scalar = _fma(1.0, 1.0, _fma(-(2.0**24), 1.0, 2.0**24))
    reordered_scalar = _fma(
        -(2.0**24), 1.0, _fma(1.0, 1.0, 2.0**24)
    )
    assert _bf16_word(frozen_scalar) == 0x3F80
    assert _bf16_word(reordered_scalar) == 0x0000


_LIBM = ctypes.CDLL("libm.so.6")
_LIBM.fmaf.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
_LIBM.fmaf.restype = ctypes.c_float
_LIBM.expf.argtypes = (ctypes.c_float,)
_LIBM.expf.restype = ctypes.c_float
_LIBM.sqrtf.argtypes = (ctypes.c_float,)
_LIBM.sqrtf.restype = ctypes.c_float


def _f32(value: float) -> float:
    return ctypes.c_float(value).value


def _fadd(left: float, right: float) -> float:
    return _f32(_f32(left) + _f32(right))


def _fmul(left: float, right: float) -> float:
    return _f32(_f32(left) * _f32(right))


def _fdiv(left: float, right: float) -> float:
    return _f32(_f32(left) / _f32(right))


def _fma(left: float, right: float, addend: float) -> float:
    return float(_LIBM.fmaf(_f32(left), _f32(right), _f32(addend)))


def _bf16_word(value: float) -> int:
    raw = struct.unpack("<I", struct.pack("<f", _f32(value)))[0]
    return ((raw + 0x7FFF + ((raw >> 16) & 1)) & 0xFFFFFFFF) >> 16


def _bf16_value(word: int) -> float:
    return struct.unpack("<f", struct.pack("<I", int(word) << 16))[0]


def _bf16_round(value: float) -> float:
    return _bf16_value(_bf16_word(value))


def _independent_matching_bases(
    basis_words: np.ndarray,
) -> list[list[list[tuple[int, float] | None]]]:
    operators: list[list[list[tuple[int, float] | None]]] = [
        [[None for _ in range(WIDTH)] for _ in range(3)] for _ in range(3)
    ]
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
            weight = _bf16_value(int(basis_words[basis, matching, pair]))
            operators[basis][matching][positive] = (negative, weight)
            operators[basis][matching][negative] = (positive, -weight)
    return operators


def _independent_sparse(
    vector: list[float],
    matchings: list[list[tuple[int, float] | None]],
    *,
    negate: bool,
) -> list[float]:
    output = []
    for coordinate in range(WIDTH):
        first = matchings[0][coordinate]
        assert first is not None
        first_partner, first_weight = first
        total = _fmul(
            -first_weight if negate else first_weight,
            vector[first_partner],
        )
        for matching in (1, 2):
            term = matchings[matching][coordinate]
            assert term is not None
            partner, weight = term
            total = _fma(
                -weight if negate else weight,
                vector[partner],
                total,
            )
        output.append(_bf16_round(total))
    return output


def _independent_cayley(
    vector: list[float],
    matchings: list[list[tuple[int, float] | None]],
    *,
    transpose: bool,
) -> list[float]:
    source = vector[:]
    term = vector[:]
    total = vector[:]
    for _ in range(4):
        term = _independent_sparse(term, matchings, negate=not transpose)
        total = [_fadd(left, right) for left, right in zip(total, term, strict=True)]
    return [
        _bf16_round(_fma(2.0, accumulator, -original))
        for accumulator, original in zip(total, source, strict=True)
    ]


def _independent_staged_expected(
    hidden_words: np.ndarray,
    basis_words: np.ndarray,
    payload_words: np.ndarray,
    threshold_words: np.ndarray,
    forced_bits: np.ndarray | None,
) -> tuple[np.ndarray, tuple[tuple[tuple[int, int, int, int, int], ...], ...]]:
    operators = _independent_matching_bases(basis_words)
    initial = [[_bf16_value(int(word)) for word in row] for row in hidden_words]
    final = []
    all_traces = []
    residual_scale = struct.unpack("<f", struct.pack("<I", 0x3E8E00D5))[0]
    for token, initial_row in enumerate(initial):
        state = initial_row[:]
        node = 0
        depth = 0
        trace = []
        while node < NODES:
            basis = depth % 3
            projected = _independent_cayley(
                state, operators[basis], transpose=False
            )
            route_seed = 815 + 104_729 * 5 + 7_919
            wrapped_coordinate = (
                node * 1_103_515_247 + route_seed + depth * 12_345
            ) & 0xFFFFFFFF
            route_coordinate = wrapped_coordinate & (WIDTH - 1)
            if forced_bits is None:
                threshold = _bf16_value(int(threshold_words[node]))
                route_value = _fadd(projected[route_coordinate], -threshold)
                bit = int(route_value >= 0.0)
            else:
                bit = int(forced_bits[token, depth])
            payload = payload_words[node, bit]
            gate = [_bf16_value(int(word)) for word in payload[0]]
            up = [_bf16_value(int(word)) for word in payload[1]]
            down = [_bf16_value(int(word)) for word in payload[2]]
            local = []
            for value_index in range(WIDTH):
                gate_value = _fmul(gate[value_index], projected[value_index])
                exponential = float(_LIBM.expf(_f32(-gate_value)))
                denominator = _fadd(1.0, exponential)
                silu = _fmul(gate_value, _fdiv(1.0, denominator))
                up_value = _fmul(up[value_index], projected[value_index])
                activation = _fmul(silu, up_value)
                local.append(_bf16_round(_fmul(down[value_index], activation)))
            raw = _independent_cayley(local, operators[basis], transpose=True)
            work = [_fmul(value, value) for value in raw]
            stride = WIDTH // 2
            while stride:
                for reduction_index in range(stride):
                    work[reduction_index] = _fadd(
                        work[reduction_index], work[reduction_index + stride]
                    )
                stride //= 2
            mean = _fmul(work[0], 0.125)
            denominator = float(_LIBM.sqrtf(_fadd(1.0, mean)))
            trust_scale = _fdiv(1.0, denominator)
            trusted = [_bf16_round(_fmul(value, trust_scale)) for value in raw]
            state = [
                _bf16_round(_fma(residual_scale, delta, value))
                for value, delta in zip(state, trusted, strict=True)
            ]
            trace.append(
                (
                    node,
                    bit,
                    route_coordinate,
                    2 * node + bit,
                    struct.unpack("<I", struct.pack("<f", trust_scale))[0],
                )
            )
            node = 2 * node + 1 + bit
            depth += 1
        final.append(state)
        all_traces.append(tuple(trace))
    output = np.array(
        [
            [_bf16_word(_fadd(value, -original)) for value, original in zip(row, start)]
            for row, start in zip(final, initial, strict=True)
        ],
        dtype="<u2",
    )
    return output, tuple(all_traces)


def test_asymmetric_nonzero_matches_independent_scalar_and_literal_bits() -> None:
    draws = dict(iter_candidate_layer_draws(0, width=WIDTH, nodes=NODES))
    hidden_words = np.array(
        [
            [0x3F00, 0xBFA0, 0x3F40, 0x4000, 0xBE00, 0x3FC0, 0xC010, 0x3EC0],
            [0xBF4D, 0x3E4D, 0x3F8D, 0xBFD9, 0x401A, 0xBE9A, 0x3F1A, 0x3FF3],
        ],
        dtype="<u2",
    )
    bits = np.array([[0, 0, 1], [1, 0, 1]], dtype=np.uint8)
    expected_words, expected_traces = _independent_staged_expected(
        hidden_words,
        draws["basis"],
        draws["payload"],
        draws["threshold"],
        bits,
    )
    literal_words = np.array(
        [
            [0x3B80, 0x3D00, 0x3CC0, 0x3D40, 0x3C68, 0xBD90, 0xBC80, 0xBC60],
            [0xBC40, 0x3A80, 0xBD20, 0x3D20, 0xBE10, 0xBCC0, 0xBC40, 0x3E38],
        ],
        dtype="<u2",
    )
    literal_traces = (
        (
            (0, 0, 3, 0, 0x3F7C901F),
            (1, 0, 3, 2, 0x3F7A8766),
            (3, 1, 2, 7, 0x3F7EA016),
        ),
        (
            (0, 1, 3, 1, 0x3F7E22E6),
            (2, 0, 2, 4, 0x3F746BD3),
        ),
    )
    assert expected_words.tolist() == literal_words.tolist()
    assert expected_traces == literal_traces
    artifacts = staged.TorchLayerArtifacts(
        basis=staged.torch_bf16_from_raw_bits(draws["basis"], device=CPU),
        payload=staged.torch_bf16_from_raw_bits(draws["payload"], device=CPU),
        threshold=staged.torch_bf16_from_raw_bits(draws["threshold"], device=CPU),
    )
    result = staged.run_staged_reference(
        staged.torch_bf16_from_raw_bits(hidden_words, device=CPU),
        artifacts,
        layer=5,
        width=WIDTH,
        nodes=NODES,
        forced_bits=torch.from_numpy(bits),
    )
    assert result.output.dtype == result.final_hidden.dtype == torch.bfloat16
    assert result.output.view(torch.uint16).numpy().tolist() == literal_words.tolist()
    observed_traces = tuple(
        tuple(
            (
                step.node,
                step.bit,
                step.coordinate,
                step.payload_edge,
                struct.unpack("<I", struct.pack("<f", step.trust_scale))[0],
            )
            for step in trace
        )
        for trace in result.traces
    )
    assert observed_traces == literal_traces

    topology = staged.build_formula_topology(WIDTH, device=CPU)
    one_sparse = staged.sparse_apply_staged(
        staged.torch_bf16_from_raw_bits(hidden_words, device=CPU),
        artifacts,
        topology,
        0,
        transpose_sign=False,
    )
    one_cayley = staged.cayley_staged(
        staged.torch_bf16_from_raw_bits(hidden_words, device=CPU),
        artifacts,
        topology,
        0,
        transpose=False,
    )
    assert one_sparse.dtype == one_cayley.dtype == torch.bfloat16


def test_nonzero_natural_route_matches_independent_scalar_and_literals() -> None:
    draws = dict(iter_candidate_layer_draws(0, width=WIDTH, nodes=NODES))
    hidden_words = np.array(
        [
            [0x3F00, 0xBFA0, 0x3F40, 0x4000, 0xBE00, 0x3FC0, 0xC010, 0x3EC0],
            [0xBF00, 0x3FA0, 0xBF40, 0xC000, 0x3E00, 0xBFC0, 0x4010, 0xBEC0],
        ],
        dtype="<u2",
    )
    expected_words, expected_traces = _independent_staged_expected(
        hidden_words,
        draws["basis"],
        draws["payload"],
        draws["threshold"],
        None,
    )
    literal_words = np.array(
        [
            [0xBCA0, 0xBD00, 0x3C40, 0x3E40, 0x3CB8, 0xBE00, 0x3D00, 0xBC00],
            [0x3CC0, 0x3E00, 0xBC80, 0xBD80, 0x3BE0, 0xBD80, 0x3EF8, 0xBDD4],
        ],
        dtype="<u2",
    )
    literal_traces = (
        (
            (0, 1, 3, 1, 0x3F7202C6),
            (2, 1, 2, 5, 0x3F7FA900),
        ),
        (
            (0, 0, 3, 0, 0x3F731A12),
            (1, 0, 3, 2, 0x3F7B10BA),
            (3, 0, 2, 6, 0x3F63B51D),
        ),
    )
    assert expected_words.tolist() == literal_words.tolist()
    assert expected_traces == literal_traces
    artifacts = staged.TorchLayerArtifacts(
        basis=staged.torch_bf16_from_raw_bits(draws["basis"], device=CPU),
        payload=staged.torch_bf16_from_raw_bits(draws["payload"], device=CPU),
        threshold=staged.torch_bf16_from_raw_bits(draws["threshold"], device=CPU),
    )
    result = staged.run_staged_reference(
        staged.torch_bf16_from_raw_bits(hidden_words, device=CPU),
        artifacts,
        layer=5,
        width=WIDTH,
        nodes=NODES,
    )
    assert result.output.view(torch.uint16).numpy().tolist() == literal_words.tolist()
    observed_traces = tuple(
        tuple(
            (
                step.node,
                step.bit,
                step.coordinate,
                step.payload_edge,
                struct.unpack("<I", struct.pack("<f", step.trust_scale))[0],
            )
            for step in trace
        )
        for trace in result.traces
    )
    assert observed_traces == literal_traces


def test_asymmetric_nonzero_reference_is_deterministic_and_route_checked() -> None:
    artifacts = small_artifacts()
    hidden = torch.tensor(
        [[0.5, -1.25, 0.75, 2.0, -0.125, 1.5, -2.25, 0.375]],
        dtype=torch.bfloat16,
    )
    bits = torch.tensor([[0, 0, 1]], dtype=torch.uint8)
    first = staged.run_staged_reference(
        hidden, artifacts, layer=5, width=WIDTH, nodes=NODES, forced_bits=bits
    )
    second = staged.run_staged_reference(
        hidden, artifacts, layer=5, width=WIDTH, nodes=NODES, forced_bits=bits
    )
    assert torch.equal(first.output, second.output)
    assert torch.count_nonzero(first.output) > 0
    assert first.traces == second.traces
    assert all(math.isfinite(step.trust_scale) for step in first.traces[0])
    assert all(0.0 < step.trust_scale <= 1.0 for step in first.traces[0])


def test_forced_bits_and_artifact_layout_are_strict() -> None:
    artifacts = small_artifacts()
    hidden = torch.zeros((1, WIDTH), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="only zero or one"):
        staged.run_staged_reference(
            hidden,
            artifacts,
            layer=0,
            width=WIDTH,
            nodes=NODES,
            forced_bits=torch.tensor([[0, 2, 0]], dtype=torch.uint8),
        )
    with pytest.raises(ValueError, match="payload"):
        staged.run_staged_reference(
            hidden,
            staged.TorchLayerArtifacts(
                artifacts.basis,
                artifacts.payload[:, :, :, ::2],
                artifacts.threshold,
            ),
            layer=0,
            width=WIDTH,
            nodes=NODES,
            forced_bits=torch.tensor([[0, 0, 1]], dtype=torch.uint8),
        )


def test_frozen_wrapper_forwards_only_exact_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    sentinel = object()

    def fake_run(hidden, artifacts, **kwargs):
        calls.append((hidden, artifacts, kwargs))
        return sentinel

    monkeypatch.setattr(staged, "run_staged_reference", fake_run)
    hidden = torch.zeros((1, 4096), dtype=torch.bfloat16)
    artifacts = object()
    bits = torch.zeros((1, 13), dtype=torch.uint8)
    assert staged.run_frozen_staged_reference(
        hidden, artifacts, layer=7, forced_bits=bits
    ) is sentinel
    assert calls == [
        (
            hidden,
            artifacts,
            {
                "layer": 7,
                "width": 4096,
                "nodes": 7166,
                "forced_bits": bits,
                "residual_scale": staged.RESIDUAL_SCALE_F32,
            },
        )
    ]


def test_full_target_shape_zero_payload_cpu_smoke() -> None:
    artifacts = staged.TorchLayerArtifacts(
        basis=torch.zeros((3, 3, 2048), dtype=torch.bfloat16),
        payload=torch.zeros((7166, 2, 3, 4096), dtype=torch.bfloat16),
        threshold=torch.zeros((7166,), dtype=torch.bfloat16),
    )
    hidden = torch.zeros((1, 4096), dtype=torch.bfloat16)
    result = staged.run_frozen_staged_reference(hidden, artifacts, layer=0)
    assert torch.equal(result.output, torch.zeros_like(hidden))
    assert len(result.traces[0]) == 12
    assert all(step.bit == 1 for step in result.traces[0])
