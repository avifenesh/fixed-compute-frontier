#!/usr/bin/env python3
"""Sealed three-unseen-seed replication of the passed T11 pilot."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

LOCAL_ROOT = Path(__file__).resolve().parents[1]
if str(LOCAL_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_ROOT))

from experiments import raw_prose_equality_plane_t11 as t11
from experiments import raw_prose_equality_plane_t11_train as pilot


ROOT = t11.ROOT
OUTPUT = ROOT / "results/raw-prose-equality-plane-t11-replication.json"
PREREGISTRATION = (
    ROOT / "results/raw-prose-equality-plane-t11-replication-preregistration.md"
)
MODEL_SEEDS = (6_703, 6_997, 7_307)


def relative_nll_delta(candidate: float, baseline: float) -> float:
    return candidate / baseline - 1.0


def run(device: torch.device, output: Path) -> dict[str, object]:
    if not PREREGISTRATION.exists():
        raise RuntimeError("missing frozen replication preregistration")
    seed_results: dict[str, object] = {}
    previous_seed = pilot.base.MODEL_SEED
    try:
        for seed in MODEL_SEEDS:
            pilot.base.MODEL_SEED = seed
            pilot.MODEL_SEED = seed
            result = pilot.run(device)
            seed_path = output.with_name(f"{output.stem}-seed-{seed}{output.suffix}")
            seed_path.parent.mkdir(parents=True, exist_ok=True)
            seed_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            seed_results[str(seed)] = result
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "all_gates_pass": result["all_gates_pass"],
                        "failed_gates": sorted(
                            name
                            for name, passed in result["gates"].items()
                            if not passed
                        ),
                        "seed_artifact": str(seed_path),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        pilot.base.MODEL_SEED = previous_seed
        pilot.MODEL_SEED = previous_seed

    deltas: list[float] = []
    direct_margins: list[float] = []
    equality_margins: list[float] = []
    failed_or_retried_steps = 0
    for result in seed_results.values():
        candidate = result["arms"]["compiler_muon_1x"]
        baseline = result["arms"]["muon_labels_1x"]
        for checkpoint in pilot.CHECKPOINTS_1X:
            key = str(checkpoint)
            candidate_checkpoint = candidate["checkpoints"][key]
            baseline_checkpoint = baseline["checkpoints"][key]
            deltas.append(
                relative_nll_delta(
                    candidate_checkpoint["natural"]["nll"],
                    baseline_checkpoint["natural"]["nll"],
                )
            )
            direct_margins.append(
                candidate_checkpoint["direct_heldout"]["minimum_margin"]
            )
            equality_margins.append(
                candidate_checkpoint["equality_heldout"]["minimum_margin"]
            )
        failed_or_retried_steps += sum(
            arm["resource_ledger"]["failed_or_retried_steps"]
            for arm in result["arms"].values()
        )

    gates = {
        "all_three_unseen_seeds_ran": set(seed_results)
        == {str(seed) for seed in MODEL_SEEDS},
        "every_seed_passed_every_pilot_gate": all(
            result["all_gates_pass"] for result in seed_results.values()
        ),
        "worst_natural_delta_within_0p5pct": max(deltas) <= 0.005,
        "all_direct_margins_positive": min(direct_margins) > 0.0,
        "all_equality_margins_positive": min(equality_margins) > 0.0,
        "no_failed_or_retried_steps": failed_or_retried_steps == 0,
        "finite_aggregate": all(
            math.isfinite(value)
            for value in deltas + direct_margins + equality_margins
        ),
    }
    return {
        "schema": "raw-prose-equality-plane-t11-replication-v1",
        "status": "pass" if all(gates.values()) else "fail",
        "device": str(device),
        "torch_version": torch.__version__,
        "model_seeds": MODEL_SEEDS,
        "pilot_seed_excluded": 6_401,
        "seed_results": seed_results,
        "aggregate": {
            "worst_natural_nll_relative_delta": max(deltas),
            "best_natural_nll_relative_delta": min(deltas),
            "minimum_direct_margin": min(direct_margins),
            "minimum_equality_margin": min(equality_margins),
            "failed_or_retried_steps": failed_or_retried_steps,
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "integrity": {
            "source_sha256": t11.sha256_file(Path(__file__)),
            "pilot_source_sha256": t11.sha256_file(Path(pilot.__file__)),
            "writer_source_sha256": t11.sha256_file(Path(t11.__file__)),
            "preregistration_sha256": t11.sha256_file(PREREGISTRATION),
            "representation_sha256": t11.sha256_file(t11.OUTPUT),
        },
        "claim_boundary": (
            "Three unseen seeds establish controlled-scale reproducibility only. "
            "They do not establish unrestricted QA, frontier-training superiority, "
            "or production capability."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    result = run(torch.device(arguments.device), arguments.output)
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
                "aggregate": result["aggregate"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
