#!/usr/bin/env python3
"""Pure-CPU proof regressions for the T32R pre-run capability audit.

This module does not train a model, access a corpus, or use a GPU.  It makes the
closed-form information and resident-byte comparisons executable so later
changes cannot silently alter the paper argument.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass


VOCAB_SIZE = 49_152
RECORD_LENGTH = 128
NUM_RECORDS = 2_405
T32_DIGITS_PER_RECORD = 381
T32_LEVEL_BITS = 6
BF16_BYTES = 2
UINT16_BYTES = 2
LAYERS = 10
HIDDEN = 384
BASELINE_FFN_WIDTH = 1_024
SCANNER_BF16_ENTRIES = 27_233
POSITION_BF16_ENTRIES = 128 * 32
FROZEN_SCAN_MULTIPLIES = 3_230_144


def binary_entropy(error: float) -> float:
    """Binary entropy in bits, with the continuous endpoint convention."""

    if not 0.0 <= error <= 1.0:
        raise ValueError("error must lie in [0, 1]")
    if error in (0.0, 1.0):
        return 0.0
    return -error * math.log2(error) - (1.0 - error) * math.log2(1.0 - error)


def fano_record_bits(error: float, *, vocab: int = VOCAB_SIZE, length: int = RECORD_LENGTH) -> float:
    """Lower bound for arbitrary-coordinate recovery from persistent state."""

    if vocab < 2:
        raise ValueError("vocab must be at least two")
    return length * (
        math.log2(vocab)
        - binary_entropy(error)
        - error * math.log2(vocab - 1)
    )


def width_unit_bytes(*, layers: int = LAYERS, hidden: int = HIDDEN) -> int:
    """Resident BF16 bytes removed by one SwiGLU-width unit in all layers."""

    return layers * 3 * hidden * BF16_BYTES


def width_unit_multiplies(*, layers: int = LAYERS, hidden: int = HIDDEN) -> int:
    """Multiplies removed per processed token by one width unit."""

    return layers * 3 * hidden


@dataclass(frozen=True)
class TypedPlaneLedger:
    token_bytes: int
    length_bytes: int
    scanner_bytes: int
    position_bytes: int
    total_bytes: int
    width_removed: int
    candidate_width: int
    slack_bytes: int
    saved_multiplies_per_token: int
    arithmetic_break_even_tokens: int


def typed_plane_ledger() -> TypedPlaneLedger:
    token_bytes = NUM_RECORDS * RECORD_LENGTH * UINT16_BYTES
    length_bytes = NUM_RECORDS * UINT16_BYTES
    scanner_bytes = SCANNER_BF16_ENTRIES * BF16_BYTES
    position_bytes = POSITION_BF16_ENTRIES * BF16_BYTES
    total_bytes = token_bytes + length_bytes + scanner_bytes + position_bytes
    unit_bytes = width_unit_bytes()
    width_removed = math.ceil(total_bytes / unit_bytes)
    funded_bytes = width_removed * unit_bytes
    saved_multiplies = width_removed * width_unit_multiplies()
    return TypedPlaneLedger(
        token_bytes=token_bytes,
        length_bytes=length_bytes,
        scanner_bytes=scanner_bytes,
        position_bytes=position_bytes,
        total_bytes=total_bytes,
        width_removed=width_removed,
        candidate_width=BASELINE_FFN_WIDTH - width_removed,
        slack_bytes=funded_bytes - total_bytes,
        saved_multiplies_per_token=saved_multiplies,
        arithmetic_break_even_tokens=math.ceil(FROZEN_SCAN_MULTIPLIES / saved_multiplies),
    )


def audit_payload() -> dict[str, object]:
    source_entropy = RECORD_LENGTH * math.log2(VOCAB_SIZE)
    t32_logical_bits = T32_DIGITS_PER_RECORD * T32_LEVEL_BITS
    t32_physical_bits = T32_DIGITS_PER_RECORD * BF16_BYTES * 8
    t32_table_bytes = NUM_RECORDS * T32_DIGITS_PER_RECORD * BF16_BYTES
    typed = typed_plane_ledger()
    return {
        "domain": {
            "vocab_size": VOCAB_SIZE,
            "record_length": RECORD_LENGTH,
            "num_records": NUM_RECORDS,
        },
        "information": {
            "source_entropy_bits_fixed_length": source_entropy,
            "t32_logical_bits": t32_logical_bits,
            "t32_logical_over_entropy": t32_logical_bits / source_entropy,
            "t32_physical_bits": t32_physical_bits,
            "t32_physical_over_entropy": t32_physical_bits / source_entropy,
            "fano_lower_bits": {
                str(error): fano_record_bits(error)
                for error in (0.0, 0.01, 0.05, 0.10)
            },
        },
        "resident": {
            "t32_record_table_bytes": t32_table_bytes,
            "t32_record_bytes_per_token": t32_table_bytes / (NUM_RECORDS * RECORD_LENGTH),
            "typed_plane": asdict(typed),
        },
        "finite_projection_counterexample": {
            "ambient_dimension": 2,
            "projection_rank": 1,
            "points": [[0, 0], [1, 0], [0, 1]],
            "projection": [1, 2],
            "projected_values": [0, 1, 2],
            "finite_set_is_injective": True,
        },
        "scope": {
            "uses_gpu": False,
            "uses_corpus": False,
            "trains_model": False,
            "establishes_latency": False,
            "establishes_learnability": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    indent = None if args.compact else 2
    print(json.dumps(audit_payload(), indent=indent, sort_keys=True))


if __name__ == "__main__":
    main()

