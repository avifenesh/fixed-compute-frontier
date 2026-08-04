"""Exact two-involution factorization for proposed permutation transport."""

from __future__ import annotations

from dataclasses import dataclass
from typing import MutableSequence, Sequence, TypeVar


T = TypeVar("T")


def validate_permutation(permutation: Sequence[int]) -> None:
    size = len(permutation)
    seen = bytearray(size)
    for target in permutation:
        if type(target) is not int or not 0 <= target < size or seen[target]:
            raise ValueError("expected a bijection over 0..N-1")
        seen[target] = 1


def is_involution(permutation: Sequence[int]) -> bool:
    try:
        validate_permutation(permutation)
    except ValueError:
        return False
    return all(permutation[permutation[node]] == node for node in range(len(permutation)))


def compose(left: Sequence[int], right: Sequence[int]) -> tuple[int, ...]:
    """Return ``left o right`` for source-to-destination permutations."""

    validate_permutation(left)
    validate_permutation(right)
    if len(left) != len(right):
        raise ValueError("permutations must have equal size")
    return tuple(left[right[node]] for node in range(len(left)))


def factor_two_involutions(
    permutation: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return involutions ``r, q`` such that ``permutation = r o q``.

    On each cycle ``(a0 a1 ... ak-1)``, ``r`` reverses the cycle coordinate:
    ``r(ai) = a(-i mod k)``.  This gives ``r p r = p^-1``.  Therefore
    ``q = r p`` is also an involution and ``p = r q``.
    """

    validate_permutation(permutation)
    size = len(permutation)
    reversal = list(range(size))
    visited = bytearray(size)

    for start in range(size):
        if visited[start]:
            continue
        cycle: list[int] = []
        node = start
        while not visited[node]:
            visited[node] = 1
            cycle.append(node)
            node = permutation[node]
        cycle_size = len(cycle)
        for index, cycle_node in enumerate(cycle):
            reversal[cycle_node] = cycle[-index % cycle_size]

    second = [reversal[permutation[node]] for node in range(size)]
    result = (tuple(reversal), tuple(second))
    if not is_involution(result[0]) or not is_involution(result[1]):
        raise AssertionError("factorization did not produce involutions")
    if compose(*result) != tuple(permutation):
        raise AssertionError("factorization does not reconstruct permutation")
    return result


def apply_involution_in_place(values: MutableSequence[T], involution: Sequence[int]) -> None:
    if len(values) != len(involution) or not is_involution(involution):
        raise ValueError("values and involution must describe the same slots")
    for left, right in enumerate(involution):
        if left < right:
            values[left], values[right] = values[right], values[left]


def apply_permutation_out_of_place(
    values: Sequence[T], permutation: Sequence[int]
) -> list[T]:
    if len(values) != len(permutation):
        raise ValueError("values and permutation must describe the same slots")
    validate_permutation(permutation)
    output = list(values)
    for source, destination in enumerate(permutation):
        output[destination] = values[source]
    return output


def apply_factored_permutation_in_place(
    values: MutableSequence[T], permutation: Sequence[int]
) -> None:
    reversal, second = factor_two_involutions(permutation)
    apply_involution_in_place(values, second)
    apply_involution_in_place(values, reversal)


@dataclass(frozen=True)
class TransportLedger:
    node_count: int
    feature_width: int
    value_bytes: int
    index_bytes: int
    value_payload_bytes: int
    permutation_index_bytes: int
    packed_visit_bitset_bytes: int
    direct_out_of_place_peak_bytes: int
    ideal_cycle_peak_optimistic_bytes: int
    factored_execution_peak_bytes: int
    factored_end_to_end_optimistic_bytes: int
    factored_optimistic_saving_vs_direct_bytes: int
    one_dense_stage_value_traffic_assumption_bytes: int
    two_dense_stage_value_traffic_assumption_bytes: int
    extra_dense_stage_value_traffic_assumption_bytes: int
    one_index_stream_traffic_assumption_bytes: int
    two_index_stream_traffic_assumption_bytes: int
    one_dense_stage_total_traffic_assumption_bytes: int
    two_dense_stage_total_traffic_assumption_bytes: int


@dataclass(frozen=True)
class PermutationValueTraffic:
    """Exact logical row traffic when fixed points are skipped."""

    node_count: int
    cycle_count: int
    fixed_point_count: int
    row_bytes: int
    direct_out_of_place_bytes: int
    ideal_cycle_bytes: int
    canonical_two_involution_bytes: int
    canonical_extra_vs_cycle_bytes: int


def transport_ledger(
    node_count: int,
    feature_width: int,
    *,
    value_bytes: int = 2,
    index_bytes: int = 2,
) -> TransportLedger:
    """Return an optimistic logical ledger, excluding the learned proposer.

    ``factored_execution_peak_bytes`` applies only after two factors have been
    materialized.  The end-to-end figure additionally assumes the proposed
    permutation buffer can be overwritten while using only a packed visited
    bitset.  The implementation above does not achieve that bound: it also
    materializes a cycle list and uses one byte per visited node.  Similarly,
    the ideal cycle figure assumes serial or one-block execution with one
    feature-row temporary.  Parallel cycle descriptors, allocator space,
    construction/validation traffic, proposer state, and kernel workspace are
    excluded and must be measured separately.  The traffic fields model kernels
    that stream every row in one or two dense stages.  They are assumptions,
    not permutation-independent lower bounds; use
    :func:`permutation_value_traffic` when fixed points are skipped.
    """

    for name, value in (
        ("node_count", node_count),
        ("feature_width", feature_width),
        ("value_bytes", value_bytes),
        ("index_bytes", index_bytes),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if node_count > 1 << (8 * index_bytes):
        raise ValueError("index_bytes cannot address every node")

    values = node_count * feature_width * value_bytes
    indices = node_count * index_bytes
    bitset = (node_count + 7) // 8
    one_feature = feature_width * value_bytes
    direct_peak = 2 * values + indices
    ideal_cycle_peak = values + indices + bitset + one_feature
    factored_execution_peak = values + 2 * indices
    factored_end_to_end_peak = factored_execution_peak + bitset
    direct_traffic = 2 * values
    factored_traffic = 4 * values
    return TransportLedger(
        node_count=node_count,
        feature_width=feature_width,
        value_bytes=value_bytes,
        index_bytes=index_bytes,
        value_payload_bytes=values,
        permutation_index_bytes=indices,
        packed_visit_bitset_bytes=bitset,
        direct_out_of_place_peak_bytes=direct_peak,
        ideal_cycle_peak_optimistic_bytes=ideal_cycle_peak,
        factored_execution_peak_bytes=factored_execution_peak,
        factored_end_to_end_optimistic_bytes=factored_end_to_end_peak,
        factored_optimistic_saving_vs_direct_bytes=(
            direct_peak - factored_end_to_end_peak
        ),
        one_dense_stage_value_traffic_assumption_bytes=direct_traffic,
        two_dense_stage_value_traffic_assumption_bytes=factored_traffic,
        extra_dense_stage_value_traffic_assumption_bytes=(
            factored_traffic - direct_traffic
        ),
        one_index_stream_traffic_assumption_bytes=indices,
        two_index_stream_traffic_assumption_bytes=2 * indices,
        one_dense_stage_total_traffic_assumption_bytes=direct_traffic + indices,
        two_dense_stage_total_traffic_assumption_bytes=(
            factored_traffic + 2 * indices
        ),
    )


def permutation_value_traffic(
    permutation: Sequence[int],
    feature_width: int,
    *,
    value_bytes: int = 2,
) -> PermutationValueTraffic:
    """Count exact logical value bytes for the canonical factorization.

    The direct out-of-place executor reads and writes every row.  The ideal
    cycle executor skips fixed points.  Across the two canonical involutions,
    each cycle of ``permutation`` contributes exactly two fixed-point
    incidences, so the factors move ``2 * (N - cycle_count)`` rows in total.
    Each moved row contributes one read and one write.
    """

    validate_permutation(permutation)
    for name, value in (
        ("feature_width", feature_width),
        ("value_bytes", value_bytes),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    size = len(permutation)
    visited = bytearray(size)
    cycles = 0
    fixed_points = 0
    for start in range(size):
        if visited[start]:
            continue
        cycles += 1
        node = start
        cycle_size = 0
        while not visited[node]:
            visited[node] = 1
            cycle_size += 1
            node = permutation[node]
        if cycle_size == 1:
            fixed_points += 1

    row_bytes = feature_width * value_bytes
    direct = 2 * size * row_bytes
    ideal_cycle = 2 * (size - fixed_points) * row_bytes
    canonical_factors = 4 * (size - cycles) * row_bytes
    return PermutationValueTraffic(
        node_count=size,
        cycle_count=cycles,
        fixed_point_count=fixed_points,
        row_bytes=row_bytes,
        direct_out_of_place_bytes=direct,
        ideal_cycle_bytes=ideal_cycle,
        canonical_two_involution_bytes=canonical_factors,
        canonical_extra_vs_cycle_bytes=canonical_factors - ideal_cycle,
    )
