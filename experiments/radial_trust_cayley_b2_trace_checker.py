#!/usr/bin/env python3
"""Independent route generator/checker for the frozen B2 tree.

The checker defines route semantics only.  It deliberately does not choose a
CUDA diagnostic-buffer shape, dtype, padding sentinel, or storage layout.
"""

from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import Sequence


FROZEN_WIDTH = 4096
FROZEN_NODES = 7166
FROZEN_LAYERS = 12
ROUTE_MULTIPLIER = 1_103_515_247
ROUTE_DEPTH_STRIDE = 12_345
MASK64 = (1 << 64) - 1


@dataclass(frozen=True)
class RouteStep:
    depth: int
    node: int
    bit: int
    coordinate: int
    payload_edge: int


@dataclass(frozen=True)
class RouteCheck:
    length: int
    leaf: int


def _exact_bit(raw_bit: object) -> int:
    try:
        bit = operator.index(raw_bit)
    except TypeError as error:
        raise ValueError("route bits must be exact integers zero or one") from error
    if bit not in (0, 1):
        raise ValueError("route bits must be exact integers zero or one")
    return bit


def route_coordinate(node: int, depth: int, layer: int, width: int) -> int:
    if node < 0 or depth < 0 or not 0 <= layer < FROZEN_LAYERS:
        raise ValueError("invalid route coordinate input")
    if width <= 0 or width & (width - 1):
        raise ValueError("route width must be a positive power of two")
    route_seed = 815 + 104_729 * layer + 7_919
    wrapped = (
        node * ROUTE_MULTIPLIER + route_seed + depth * ROUTE_DEPTH_STRIDE
    ) & 0xFFFFFFFF
    return wrapped & (width - 1)


def splitmix64_value(row: int, layer: int, depth: int) -> int:
    if not 0 <= row < 8192 or not 0 <= layer < FROZEN_LAYERS or depth < 0:
        raise ValueError("invalid uniform-route counter")
    value = (
        0xD1B54A32D192ED03 + row + (layer << 13) + (depth << 17)
    ) & MASK64
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & MASK64
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & MASK64
    value ^= value >> 31
    return value


def uniform_route_bit(row: int, layer: int, depth: int) -> int:
    return splitmix64_value(row, layer, depth) & 1


def diagnostic_route_bits(
    regime: str,
    *,
    row: int,
    layer: int,
    nodes: int = FROZEN_NODES,
) -> tuple[int, ...]:
    if (
        regime not in ("collapsed", "uniform")
        or not 0 <= row < 8192
        or not 0 <= layer < FROZEN_LAYERS
        or nodes <= 0
    ):
        raise ValueError("invalid diagnostic route request")
    node = 0
    depth = 0
    bits: list[int] = []
    while node < nodes:
        bit = 0 if regime == "collapsed" else uniform_route_bit(row, layer, depth)
        bits.append(bit)
        node = 2 * node + 1 + bit
        depth += 1
        if depth > 63:
            raise ValueError("tree depth exceeds checker bound")
    return tuple(bits)


def expected_route_steps(
    bits: Sequence[int],
    *,
    layer: int,
    width: int,
    nodes: int,
) -> tuple[RouteStep, ...]:
    if nodes <= 0 or not bits:
        raise ValueError("route requires a nonempty tree and bit sequence")
    node = 0
    steps: list[RouteStep] = []
    for depth, raw_bit in enumerate(bits):
        if node >= nodes:
            raise ValueError("route bits continue after the leaf")
        bit = _exact_bit(raw_bit)
        steps.append(
            RouteStep(
                depth=depth,
                node=node,
                bit=bit,
                coordinate=route_coordinate(node, depth, layer, width),
                payload_edge=2 * node + bit,
            )
        )
        node = 2 * node + 1 + bit
    if node < nodes:
        raise ValueError("route bits ended before reaching a leaf")
    return tuple(steps)


def check_route_steps(
    steps: Sequence[RouteStep],
    *,
    layer: int,
    width: int,
    nodes: int,
    expected_bits: Sequence[int] | None = None,
    require_frozen_depth: bool = False,
) -> RouteCheck:
    if nodes <= 0 or not steps:
        raise ValueError("route trace must be nonempty")
    if expected_bits is not None and len(expected_bits) != len(steps):
        raise ValueError("expected bits and trace have different lengths")
    node = 0
    for depth, step in enumerate(steps):
        if node >= nodes:
            raise ValueError("trace continues after the leaf")
        if step.depth != depth:
            raise ValueError("trace depth is not contiguous")
        if step.node != node:
            raise ValueError("trace node does not follow the prior bit")
        if step.bit not in (0, 1):
            raise ValueError("trace bit is not binary")
        if expected_bits is not None:
            expected_bit = _exact_bit(expected_bits[depth])
            if step.bit != expected_bit:
                raise ValueError("trace bit differs from the forced route")
        expected_coordinate = route_coordinate(node, depth, layer, width)
        if step.coordinate != expected_coordinate:
            raise ValueError("trace coordinate differs from the frozen formula")
        if step.payload_edge != 2 * node + step.bit:
            raise ValueError("trace payload edge differs from node and bit")
        node = 2 * node + 1 + step.bit
    if node < nodes:
        raise ValueError("trace ended before reaching a leaf")
    if require_frozen_depth and len(steps) not in (12, 13):
        raise ValueError("frozen B2 trace length must be 12 or 13")
    return RouteCheck(length=len(steps), leaf=node)


def expected_diagnostic_trace(
    regime: str, *, row: int, layer: int
) -> tuple[RouteStep, ...]:
    bits = diagnostic_route_bits(regime, row=row, layer=layer)
    return expected_route_steps(
        bits,
        layer=layer,
        width=FROZEN_WIDTH,
        nodes=FROZEN_NODES,
    )


def check_frozen_route_steps(
    steps: Sequence[RouteStep],
    *,
    layer: int,
    expected_bits: Sequence[int] | None = None,
) -> RouteCheck:
    return check_route_steps(
        steps,
        layer=layer,
        width=FROZEN_WIDTH,
        nodes=FROZEN_NODES,
        expected_bits=expected_bits,
        require_frozen_depth=True,
    )
