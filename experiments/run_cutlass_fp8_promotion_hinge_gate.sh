#!/usr/bin/env bash
set -euo pipefail

profiler=/workspace/cutlass/build_fp8_base/tools/profiler/cutlass_profiler
baseline_library=/workspace/cutlass/build_fp8_base/tools/library
hinge_library=/workspace/cutlass/fp8_hinge_runtime
output_root=/workspace/fixed-compute-frontier/results/cutlass-fp8-promotion-hinge-gate
kernel=cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_128x128x128_2x1x1_0_tnn_align16_warpspecialized_cooperative_epi_tma

mkdir -p "${output_root}"

run_case() {
  local arm=$1
  local rows=$2
  local libraries=$3
  LD_LIBRARY_PATH="${libraries}:${baseline_library}" "${profiler}" \
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

for rows in 64 256 1024 4096; do
  run_case baseline "${rows}" "${baseline_library}"
  run_case hinge "${rows}" "${hinge_library}"
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
for rows in (64, 256, 1024, 4096):
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
    "key_cells_at_most_1_005": all(
        item["hinge_over_baseline"] <= 1.005
        for item in comparisons
        if item["m"] in (256, 1024)
    ),
    "all_cells_at_most_1_01": all(
        item["hinge_over_baseline"] <= 1.01 for item in comparisons
    ),
}
payload = {
    "schema": "cutlass-fp8-promotion-hinge-gate-v1",
    "claim": "Active alpha=1 midpoint hinge reuses an accurate-FP8 promotion boundary.",
    "shape": {"rows": [64, 256, 1024, 4096], "n": 4096, "k": 4096},
    "records": records,
    "comparisons": comparisons,
    "decision": decision,
}
(root / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
