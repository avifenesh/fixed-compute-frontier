#!/usr/bin/env python3
"""Sealed-word decoder for T84 effective v4.

This module deliberately accepts booleans, not metrics.  The audited evaluator
must first hash a complete stage and reduce every currently decidable frozen
statement to these fields.  The decoder emits exactly one allowed word.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Literal, TypedDict


Decision = Literal[
    "CONTINUE",
    "STOP-REJECT",
    "CONTROL FLOOR NOT ESTABLISHED",
    "RESOURCE-INCONCLUSIVE",
    "PROTOCOL INVALID",
]


class GateInput(TypedDict):
    integrity_valid: bool
    required_control_valid: bool
    resource_valid: bool
    control_floor_established: bool
    all_decidable_gates_pass: bool


def decode_gate(gate: GateInput) -> Decision:
    if not gate["integrity_valid"] or not gate["required_control_valid"]:
        return "PROTOCOL INVALID"
    if not gate["resource_valid"]:
        return "RESOURCE-INCONCLUSIVE"
    if not gate["control_floor_established"]:
        return "CONTROL FLOOR NOT ESTABLISHED"
    if not gate["all_decidable_gates_pass"]:
        return "STOP-REJECT"
    return "CONTINUE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("gate_input", type=Path)
    parser.add_argument("vmhwm_output", type=Path)
    return parser.parse_args()


def linux_vmhwm_bytes() -> int:
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmHWM:"):
            fields = line.split()
            if len(fields) != 3 or fields[2] != "kB":
                raise RuntimeError("unexpected decoder VmHWM format")
            return int(fields[1]) * 1024
    raise RuntimeError("decoder VmHWM missing")


def write_diagnostic_new(path: Path, vmhwm: int) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    data = json.dumps(
        {"decoder_absolute_linux_vmhwm_bytes": vmhwm},
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.rename(temporary, path)


def main() -> None:
    args = parse_args()
    raw = json.loads(args.gate_input.read_text(encoding="utf-8"))
    expected = {
        "integrity_valid",
        "required_control_valid",
        "resource_valid",
        "control_floor_established",
        "all_decidable_gates_pass",
    }
    if set(raw) != expected or not all(type(raw[key]) is bool for key in expected):
        raise RuntimeError("gate input must contain exactly five boolean fields")
    word = decode_gate(raw)
    write_diagnostic_new(args.vmhwm_output, linux_vmhwm_bytes())
    print(word)


if __name__ == "__main__":
    main()
