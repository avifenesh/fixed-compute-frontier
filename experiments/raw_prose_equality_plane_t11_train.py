#!/usr/bin/env python3
"""Matched T11 pilot using the sealed T10 controls and schedule."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

LOCAL_ROOT = Path(__file__).resolve().parents[1]
if str(LOCAL_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_ROOT))

from experiments import raw_prose_equality_plane_t10_train as base
from experiments import raw_prose_equality_plane_t11 as t11


ROOT = t11.ROOT
OUTPUT = ROOT / "results/raw-prose-equality-plane-t11-training-pilot.json"
PROTOCOL = ROOT / "results/raw-prose-equality-plane-t11-training-pilot-protocol.md"
REPRESENTATION = t11.OUTPUT

# Bind the sealed training logic to the new writer.  Every schedule and control
# constant remains defined by the T10 harness; only these dynamic module/path
# globals change in this process.
base.t10 = t11
base.OUTPUT = OUTPUT
base.PROTOCOL = PROTOCOL
base.REPRESENTATION = REPRESENTATION

MODEL_SEED = base.MODEL_SEED
MUON_LEARNING_RATE = base.MUON_LEARNING_RATE
CHECKPOINTS_1X = base.CHECKPOINTS_1X
STEPS_2X = base.STEPS_2X
QuerySplit = base.QuerySplit
build_query_split = base.build_query_split


def run(device: torch.device) -> dict[str, object]:
    result = base.run(device)
    result["schema"] = "raw-prose-equality-plane-t11-training-pilot-v1"
    result["integrity"]["base_t10_training_harness_sha256"] = t11.sha256_file(
        Path(base.__file__)
    )
    result["integrity"]["source_sha256"] = t11.sha256_file(Path(__file__))
    result["claim_boundary"] = (
        "Single-seed minimal-isolation pilot. Passing admits a sealed three-seed "
        "run; it does not establish unrestricted QA or production superiority."
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    result = run(torch.device(arguments.device))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "all_gates_pass": result["all_gates_pass"],
                "failed_gates": sorted(
                    name for name, passed in result["gates"].items() if not passed
                ),
                "best_2x_control": result["best_2x_control"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
