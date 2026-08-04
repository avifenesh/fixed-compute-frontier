#!/usr/bin/env bash
set -euo pipefail

profiler=/workspace/cutlass/build/tools/profiler/cutlass_profiler
output_root=/workspace/fixed-compute-frontier/results/cutlass-interference-gate
w8_kernel=cutlass3x_sm90_tensorop_i64x128x32gemm_s8_s8_s32_s8_s8_128x128x128_2x1x1_0_tnn_align16_warpspecialized_cooperative_epi_tma
w4_kernel=cutlass_tensorop_s4_i16864gemm_s4_128x128_256x4_tn_align32

mkdir -p "${output_root}"
rm -f "${output_root}"/*.gemm.csv

run_case() {
  local label=$1
  local kernel=$2
  local rows=$3
  local columns=$4
  local reduction=$5

  "${profiler}" \
    --operation=Gemm \
    --kernels="${kernel}" \
    --m="${rows}" \
    --n="${columns}" \
    --k="${reduction}" \
    --verification-enabled=false \
    --warmup-iterations=50 \
    --profiling-iterations=200 \
    --workspace-count=1 \
    --output="${output_root}/${label}" \
    --verbose=false
}

for rows in 256 512 2048 4096; do
  run_case "m${rows}-w8-single" "${w8_kernel}" "${rows}" 14336 4096
  run_case "m${rows}-w4-routed" "${w4_kernel}" "${rows}" 14336 4096
  run_case "m${rows}-w4-dual-concatenated" "${w4_kernel}" "${rows}" 28672 4096
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
for rows in (256, 512, 2048, 4096):
    prefix = f"m{rows}"
    w8 = by_case[f"{prefix}-w8-single"]["runtime_ms"]
    routed = by_case[f"{prefix}-w4-routed"]["runtime_ms"]
    dual = by_case[f"{prefix}-w4-dual-concatenated"]["runtime_ms"]
    comparisons.append(
        {
            "m": rows,
            "w8_single_ms": w8,
            "w4_routed_ms": routed,
            "w4_dual_concatenated_ms": dual,
            "dual_w4_over_w8": dual / w8,
            "routed_w4_over_w8": routed / w8,
            "dual_w4_over_routed_w4": dual / routed,
        }
    )

payload = {
    "claim": "Raw CUTLASS ceiling only; excludes scales, routing, two-accumulator combination, activation, and quantization quality.",
    "shape": {"baseline_n": 14336, "candidate_concatenated_n": 28672, "k": 4096},
    "records": records,
    "comparisons": comparisons,
}
(root / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
