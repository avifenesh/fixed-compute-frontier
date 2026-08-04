#!/usr/bin/env bash
set -euo pipefail

baseline_profiler=/workspace/cutlass/build/tools/profiler/cutlass_profiler
hinge_profiler=/workspace/cutlass/build_hinge/tools/profiler/cutlass_profiler
output_root=/workspace/fixed-compute-frontier/results/cutlass-inplace-hinge-gate
kernel=cutlass3x_sm90_tensorop_i64x128x32gemm_s8_s8_s32_s8_s8_128x128x128_2x1x1_0_tnn_align16_warpspecialized_cooperative_epi_tma

mkdir -p "${output_root}"

run_case() {
  local profiler=$1
  local arm=$2
  local rows=$3
  "${profiler}" \
    --operation=Gemm \
    --kernels="${kernel}" \
    --m="${rows}" \
    --n=4096 \
    --k=4096 \
    --verification-enabled=false \
    --warmup-iterations=20 \
    --profiling-iterations=200 \
    --workspace-count=1 \
    --output="${output_root}/m${rows}-${arm}" \
    --verbose=false
}

for rows in 1 8 64 256 1024 4096; do
  run_case "${baseline_profiler}" baseline "${rows}"
  run_case "${hinge_profiler}" hinge "${rows}"
done

python3 - "${output_root}" <<'PY'
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
records = []
for path in sorted(root.glob("*.gemm.csv")):
    with path.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    records.append(
        {
            "case": path.name.removesuffix(".gemm.csv"),
            "m": int(row["m"]),
            "n": int(row["n"]),
            "k": int(row["k"]),
            "runtime_ms": float(row["Runtime"]),
            "gbytes_per_second": float(row["GB/s"]),
            "gflops": float(row["GFLOPs"]),
            "operation": row["Operation"],
            "disposition": row["Disposition"],
            "status": row["Status"],
        }
    )

by_case = {record["case"]: record for record in records}
comparisons = []
for rows in (1, 8, 64, 256, 1024, 4096):
    baseline = by_case[f"m{rows}-baseline"]["runtime_ms"]
    hinge = by_case[f"m{rows}-hinge"]["runtime_ms"]
    comparisons.append(
        {
            "m": rows,
            "baseline_ms": baseline,
            "hinge_ms": hinge,
            "hinge_over_baseline": hinge / baseline,
        }
    )

decision = {
    "key_cells_at_most_1_01": all(
        item["hinge_over_baseline"] <= 1.01
        for item in comparisons
        if item["m"] in (256, 1024)
    ),
    "all_cells_at_most_1_02": all(
        item["hinge_over_baseline"] <= 1.02 for item in comparisons
    ),
}
payload = {
    "schema": "cutlass-inplace-hinge-gate-v1",
    "claim": "Active alpha=1 midpoint hinge in native CUTLASS SM90a S8 TMA/WGMMA mainloop.",
    "shape": {"rows": [1, 8, 64, 256, 1024, 4096], "n": 4096, "k": 4096},
    "records": records,
    "comparisons": comparisons,
    "decision": decision,
}
(root / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
