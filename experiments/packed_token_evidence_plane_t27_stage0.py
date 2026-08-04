from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


VOCAB_CAPACITY = 65_536
TAPE_TOKENS = 55
NIBBLES_PER_TOKEN = 4
PLANE_CELLS = TAPE_TOKENS * NIBBLES_PER_TOKEN
PAD_TOKEN = 0
SEED = 27_001
RANDOM_TAPES = 1_000


def token_to_nibbles(token_id: int) -> tuple[int, int, int, int]:
    if not 0 <= int(token_id) < VOCAB_CAPACITY:
        raise ValueError(f"token_id outside [0,{VOCAB_CAPACITY - 1}]: {token_id}")
    value = int(token_id)
    return tuple((value >> (4 * k)) & 0xF for k in range(4))  # type: ignore[return-value]


def nibbles_to_token(nibbles: Sequence[int]) -> int:
    if len(nibbles) != NIBBLES_PER_TOKEN:
        raise ValueError(f"expected four nibbles, got {len(nibbles)}")
    value = 0
    for k, nibble in enumerate(nibbles):
        nibble = int(nibble)
        if not 0 <= nibble < 16:
            raise ValueError(f"nibble outside [0,15]: {nibble}")
        value |= nibble << (4 * k)
    return value


def pack_tokens(tokens: Sequence[int]) -> tuple[int, ...]:
    if len(tokens) != TAPE_TOKENS:
        raise ValueError(f"expected {TAPE_TOKENS} tokens, got {len(tokens)}")
    return tuple(n for token in tokens for n in token_to_nibbles(int(token)))


def unpack_tokens(cells: Sequence[int]) -> tuple[int, ...]:
    if len(cells) != PLANE_CELLS:
        raise ValueError(f"expected {PLANE_CELLS} cells, got {len(cells)}")
    return tuple(
        nibbles_to_token(cells[i : i + NIBBLES_PER_TOKEN])
        for i in range(0, PLANE_CELLS, NIBBLES_PER_TOKEN)
    )


def amplitude(nibble: int) -> int:
    nibble = int(nibble)
    if not 0 <= nibble < 16:
        raise ValueError(f"nibble outside [0,15]: {nibble}")
    return 2 * nibble - 15


def amplitude_to_nibble(value: float) -> int:
    rounded = int(round(float(value)))
    if rounded < -15 or rounded > 15 or rounded % 2 == 0:
        raise ValueError(f"not an odd amplitude level: {value}")
    return (rounded + 15) // 2


def bf16_roundtrip_float32(values: Iterable[float]) -> np.ndarray:
    """Round float32 values to BF16 (RNE), then expand them back to float32."""

    array = np.asarray(tuple(values), dtype=np.float32)
    bits = array.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    bf16_as_float32_bits = rounded & np.uint32(0xFFFF0000)
    return bf16_as_float32_bits.view(np.float32)


def silu(value: float) -> float:
    value = float(value)
    if value >= 0:
        return value / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return value * exp_value / (1.0 + exp_value)


def silu_square_pair(difference: float) -> float:
    difference = float(difference)
    return difference * silu(difference) + (-difference) * silu(-difference)


def _virtual_token(tape: Sequence[int], position: int) -> int:
    if 0 <= position < TAPE_TOKENS:
        return int(tape[position])
    if position in (TAPE_TOKENS, TAPE_TOKENS + 1):
        return PAD_TOKEN
    raise IndexError(position)


def match_positions(tape: Sequence[int], query_pair: Sequence[int]) -> tuple[int, ...]:
    if len(tape) != TAPE_TOKENS:
        raise ValueError(f"expected {TAPE_TOKENS} tokens, got {len(tape)}")
    if len(query_pair) != 2:
        raise ValueError("query_pair must contain exactly two token IDs")
    q0, q1 = (int(query_pair[0]), int(query_pair[1]))
    token_to_nibbles(q0)
    token_to_nibbles(q1)
    return tuple(
        position
        for position in range(TAPE_TOKENS)
        if _virtual_token(tape, position) == q0
        and _virtual_token(tape, position + 1) == q1
    )


def read_unique_successor(tape: Sequence[int], query_pair: Sequence[int]) -> dict[str, object]:
    positions = match_positions(tape, query_pair)
    if not positions:
        return {"status": "missing", "positions": positions, "value": None}
    if len(positions) != 1:
        return {"status": "ambiguous", "positions": positions, "value": None}
    position = positions[0]
    return {
        "status": "ok",
        "positions": positions,
        "value": _virtual_token(tape, position + 2),
    }


def token_binary_code(token_id: int) -> tuple[int, ...]:
    token_to_nibbles(token_id)
    return tuple((int(token_id) >> bit) & 1 for bit in range(16))


def binary_code_to_token(code: Sequence[int]) -> int:
    if len(code) != 16:
        raise ValueError(f"expected 16 bits, got {len(code)}")
    value = 0
    for bit, item in enumerate(code):
        item = int(item)
        if item not in (0, 1):
            raise ValueError(f"not a bit: {item}")
        value |= item << bit
    return value


def bipolar_score(left: Sequence[int], right: Sequence[int]) -> int:
    if len(left) != 16 or len(right) != 16:
        raise ValueError("both codes must contain 16 bits")
    return sum((2 * int(a) - 1) * (2 * int(b) - 1) for a, b in zip(left, right))


def channel_ledger() -> dict[str, int]:
    return {
        "two_token_squared_distance_upper_bound": 2 * 2 * 4 * 55,
        "unique_successor_selection_upper_bound": 55 * 4,
        "four_nibble_token_decoder_upper_bound": 4 * 48,
        "available_swiglu_width": 1_024,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_stage0() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    preregistration = root / "results/packed-token-evidence-plane-t27-stage0-preregistration.md"
    source = Path(__file__).resolve()
    test = root / "tests/test_packed_token_evidence_plane_t27_stage0.py"

    all_token_roundtrips = all(
        nibbles_to_token(token_to_nibbles(token_id)) == token_id
        for token_id in range(VOCAB_CAPACITY)
    )

    boundary_tapes = [
        tuple([0] * TAPE_TOKENS),
        tuple([VOCAB_CAPACITY - 1] * TAPE_TOKENS),
        tuple(range(TAPE_TOKENS)),
        tuple((VOCAB_CAPACITY - 1 - i) for i in range(TAPE_TOKENS)),
        tuple((i * 1_229 + 17) % VOCAB_CAPACITY for i in range(TAPE_TOKENS)),
    ]
    boundary_pack_roundtrips = all(unpack_tokens(pack_tokens(tape)) == tape for tape in boundary_tapes)
    boundary_cells = tuple(cell for tape in boundary_tapes for cell in pack_tokens(tape))

    levels = tuple(amplitude(nibble) for nibble in range(16))
    levels_bf16 = bf16_roundtrip_float32(levels)
    bf16_level_roundtrip = all(
        float(value) == float(level) and amplitude_to_nibble(float(value)) == nibble
        for nibble, (level, value) in enumerate(zip(levels, levels_bf16))
    )

    differences = sorted({float(a - b) for a in levels for b in levels})
    square_errors = [abs(silu_square_pair(difference) - difference * difference) for difference in differences]
    nonzero_square_values = [silu_square_pair(difference) for difference in differences if difference]
    square_identity = max(square_errors) <= 1e-10
    square_margin = silu_square_pair(0.0) == 0.0 and min(nonzero_square_values) >= 4.0 - 1e-10

    rng = np.random.default_rng(SEED)
    unique_read_count = 0
    no_unique_pair_count = 0
    random_read_failures: list[dict[str, object]] = []
    for tape_index in range(RANDOM_TAPES):
        tape = tuple(int(value) for value in rng.integers(0, VOCAB_CAPACITY, size=TAPE_TOKENS))
        pairs: dict[tuple[int, int], list[int]] = {}
        for position in range(TAPE_TOKENS):
            pair = (_virtual_token(tape, position), _virtual_token(tape, position + 1))
            pairs.setdefault(pair, []).append(position)
        unique = [(pair, positions[0]) for pair, positions in pairs.items() if len(positions) == 1]
        if not unique:
            no_unique_pair_count += 1
            continue
        pair, position = unique[tape_index % len(unique)]
        result = read_unique_successor(tape, pair)
        expected = _virtual_token(tape, position + 2)
        if result != {"status": "ok", "positions": (position,), "value": expected}:
            random_read_failures.append(
                {"tape_index": tape_index, "position": position, "expected": expected, "actual": result}
            )
        else:
            unique_read_count += 1

    repeated_tape = tuple([101, 202, 303, 101, 202, 404] + list(range(49)))
    repeated_result = read_unique_successor(repeated_tape, (101, 202))
    repeated_ambiguity = repeated_result == {
        "status": "ambiguous",
        "positions": (0, 3),
        "value": None,
    }

    binary_roundtrip = True
    correct_scores: set[int] = set()
    one_bit_scores: set[int] = set()
    for token_id in range(VOCAB_CAPACITY):
        code = token_binary_code(token_id)
        if binary_code_to_token(code) != token_id:
            binary_roundtrip = False
            break
        correct_scores.add(bipolar_score(code, code))
        for bit in range(16):
            neighbor = list(code)
            neighbor[bit] ^= 1
            one_bit_scores.add(bipolar_score(code, neighbor))

    ledger = channel_ledger()
    ledger_exact = ledger == {
        "two_token_squared_distance_upper_bound": 880,
        "unique_successor_selection_upper_bound": 220,
        "four_nibble_token_decoder_upper_bound": 192,
        "available_swiglu_width": 1_024,
    }
    ledger_within_width = all(
        value <= ledger["available_swiglu_width"]
        for key, value in ledger.items()
        if key != "available_swiglu_width"
    )

    gates = {
        "all_65536_token_roundtrips": all_token_roundtrips,
        "boundary_pack_roundtrips": boundary_pack_roundtrips
        and len(pack_tokens(boundary_tapes[0])) == PLANE_CELLS
        and min(boundary_cells) >= 0
        and max(boundary_cells) <= 15,
        "bf16_level_roundtrip": bf16_level_roundtrip,
        "silu_square_identity": square_identity and square_margin,
        "random_unique_successor_reads": no_unique_pair_count == 0
        and unique_read_count == RANDOM_TAPES
        and not random_read_failures,
        "repeated_pair_is_ambiguous": repeated_ambiguity,
        "binary_code_roundtrip_and_margin": binary_roundtrip
        and correct_scores == {16}
        and one_bit_scores == {14},
        "channel_ledger": ledger_exact and ledger_within_width,
    }

    result = {
        "experiment": "packed-token-evidence-plane-t27-stage0",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "domain": {
            "vocab_capacity": VOCAB_CAPACITY,
            "tape_tokens": TAPE_TOKENS,
            "plane_cells": PLANE_CELLS,
            "logical_bits": PLANE_CELLS * 4,
            "seed": SEED,
            "random_tapes": RANDOM_TAPES,
        },
        "numeric": {
            "amplitude_levels": levels,
            "bf16_levels": tuple(float(value) for value in levels_bf16),
            "difference_domain": differences,
            "max_silu_square_absolute_error": max(square_errors),
            "minimum_nonzero_squared_distance": min(nonzero_square_values),
        },
        "reads": {
            "unique_read_count": unique_read_count,
            "no_unique_pair_count": no_unique_pair_count,
            "random_read_failures": random_read_failures,
            "repeated_pair_result": repeated_result,
        },
        "output_code": {
            "correct_scores": sorted(correct_scores),
            "one_bit_neighbor_scores": sorted(one_bit_scores),
            "minimum_margin": min(correct_scores) - max(one_bit_scores),
        },
        "channel_ledger": ledger,
        "gates": gates,
        "integrity": {
            "preregistration_sha256": _sha256(preregistration),
            "source_sha256": _sha256(source),
            "test_sha256": _sha256(test),
        },
        "claim_boundary": (
            "Primitive identity only; no natural sufficiency, physical Transformer, "
            "language quality, or smarter-model claim."
        ),
    }
    return result


def main() -> None:
    print(json.dumps(run_stage0(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
