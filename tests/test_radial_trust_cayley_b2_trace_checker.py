from dataclasses import replace

import pytest

import experiments.radial_trust_cayley_b2_trace_checker as checker


def test_splitmix64_full_values_and_bits_have_frozen_goldens() -> None:
    cases = {
        (0, 0, 0): 3_246_858_695_411_730_098,
        (1, 0, 0): 6_951_516_134_914_417_455,
        (8191, 11, 12): 10_844_285_056_177_008_235,
        (683, 7, 5): 7_469_085_564_731_718_367,
        (4096, 11, 0): 3_192_007_008_824_378_521,
    }
    for arguments, expected in cases.items():
        assert checker.splitmix64_value(*arguments) == expected
        assert checker.uniform_route_bit(*arguments) == expected & 1


def test_diagnostic_route_bit_sequences_and_leaf_depths_are_exact() -> None:
    assert checker.diagnostic_route_bits("collapsed", row=0, layer=0) == (0,) * 13
    cases = {
        (0, 0): ("0111100101001", 12072),
        (1, 0): ("111010111101", 7868),
        (8191, 11): ("110010011000", 7319),
        (683, 7): ("111111110010", 8177),
    }
    for (row, layer), (text, leaf) in cases.items():
        bits = checker.diagnostic_route_bits("uniform", row=row, layer=layer)
        assert "".join(str(bit) for bit in bits) == text
        trace = checker.expected_diagnostic_trace("uniform", row=row, layer=layer)
        checked = checker.check_frozen_route_steps(
            trace, layer=layer, expected_bits=bits
        )
        assert (checked.length, checked.leaf) == (len(text), leaf)


def test_target_width_route_coordinates_and_payload_edges_are_literal() -> None:
    assert checker.route_coordinate(7165, 12, 11, 4096) == 400
    trace = checker.expected_diagnostic_trace("collapsed", row=0, layer=0)
    assert trace[0] == checker.RouteStep(0, 0, 0, 542, 0)
    assert trace[1] == checker.RouteStep(1, 1, 0, 198, 2)
    assert trace[-1] == checker.RouteStep(12, 4095, 0, 1627, 8190)
    checked = checker.check_frozen_route_steps(
        trace, layer=0, expected_bits=(0,) * 13
    )
    assert (checked.length, checked.leaf) == (13, 8191)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("depth", 2, "depth"),
        ("node", 9, "node"),
        ("bit", 2, "binary"),
        ("coordinate", 0, "coordinate"),
        ("payload_edge", 99, "payload edge"),
    ],
)
def test_checker_rejects_each_corrupted_trace_field(
    field: str, value: int, message: str
) -> None:
    trace = list(checker.expected_diagnostic_trace("collapsed", row=0, layer=0))
    trace[1] = replace(trace[1], **{field: value})
    with pytest.raises(ValueError, match=message):
        checker.check_frozen_route_steps(trace, layer=0)


def test_checker_rejects_early_late_and_wrong_forced_routes() -> None:
    trace = checker.expected_diagnostic_trace("collapsed", row=0, layer=0)
    with pytest.raises(ValueError, match="before reaching"):
        checker.check_frozen_route_steps(trace[:-1], layer=0)
    with pytest.raises(ValueError, match="after the leaf"):
        checker.check_frozen_route_steps(trace + (trace[-1],), layer=0)
    wrong_bits = (1,) + (0,) * 12
    with pytest.raises(ValueError, match="forced route"):
        checker.check_frozen_route_steps(
            trace, layer=0, expected_bits=wrong_bits
        )


def test_generic_small_trace_matches_hand_derived_path() -> None:
    steps = checker.expected_route_steps(
        (0, 0, 1), layer=0, width=8, nodes=5
    )
    assert steps == (
        checker.RouteStep(0, 0, 0, 6, 0),
        checker.RouteStep(1, 1, 0, 6, 2),
        checker.RouteStep(2, 3, 1, 5, 7),
    )
    checked = checker.check_route_steps(
        steps,
        layer=0,
        width=8,
        nodes=5,
        expected_bits=(0, 0, 1),
    )
    assert (checked.length, checked.leaf) == (3, 8)


def test_invalid_uniform_counter_and_nonbinary_bit_sequences_are_rejected() -> None:
    with pytest.raises(ValueError):
        checker.splitmix64_value(8192, 0, 0)
    with pytest.raises(ValueError):
        checker.splitmix64_value(0, 12, 0)
    with pytest.raises(ValueError, match="zero or one"):
        checker.expected_route_steps((0, 2), layer=0, width=8, nodes=5)
    with pytest.raises(ValueError, match="exact integers"):
        checker.expected_route_steps((0.5, 0, 1), layer=0, width=8, nodes=5)
    valid = checker.expected_route_steps((0, 0, 1), layer=0, width=8, nodes=5)
    with pytest.raises(ValueError, match="exact integers"):
        checker.check_route_steps(
            valid,
            layer=0,
            width=8,
            nodes=5,
            expected_bits=(0.5, 0, 1),
        )
    with pytest.raises(ValueError, match="diagnostic route"):
        checker.diagnostic_route_bits("collapsed", row=8192, layer=0)
    with pytest.raises(ValueError, match="diagnostic route"):
        checker.diagnostic_route_bits("collapsed", row=0, layer=12)
