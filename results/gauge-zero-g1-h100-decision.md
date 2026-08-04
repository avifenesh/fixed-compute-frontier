# Gauge-zero G1 H100 formal-v1 decision

Status: **failed one uncertainty gate; not admitted**  
Date: 2026-07-27

The formal v1 result is
`results/gauge-zero-g1-h100.json` (SHA-256
`b92660a47e9aef6bd6e2f62dcb0ce28111f2f76dffc1a155712306e0ad063064`).
All frozen source and runtime integrity checks passed.

## What passed

- Candidate and control were independently BF16 bit-exact in every row cell.
- Dense QKV weight bytes, output bytes, parameter values, and KV coordinates
  were unchanged. The deterministic `pivot_column = pair_index` rule required
  zero atlas bits.
- Every point-estimate latency ratio was within 1.02:
  `1.01445, 1.00335, 1.00000, 1.01031, 0.99815, 0.99985` for rows
  `1, 8, 32, 128, 512, 2048`.
- Every bootstrap upper bound was within the frozen 1.03 full-grid gate.

## What failed

The frozen rows-1-and-8 bootstrap upper bound had to be at most 1.02. Rows 8
passed at 1.01346, but rows 1 reached 1.02946. Its median ratio was 1.01445;
the failure is uncertainty around a point estimate inside the envelope, not a
measured median regression beyond 2%.

The package therefore remains failed. Thresholds are not changed and the
10M-token LM pilot is not admitted from this result.

## Allowed next step

A separately preregistered, independent-seed confirmation may repeat the
unchanged one-row implementation with enough trials to resolve the confidence
interval. It must keep the 1.02 bootstrap-upper threshold. Kernel or estimator
changes require a new version and cannot retroactively rescue formal v1.

## Independent confirmation outcome

That allowed confirmation subsequently passed. Its immutable result is
`results/gauge-zero-g1-h100-confirmation.json` (SHA-256
`cda86ea01fed65ca46ace94b56df67bae6b6e3275b424693b987b1c974c8b25b`).
With 4,000 trials and seed 6,829, the one-row median ratio was `1.00956` and
the bootstrap 95% interval was `[1.00668, 1.01435]`, below the unchanged 1.02
upper threshold. Candidate and control remained BF16 bit-exact and required
zero pivot metadata.

Formal v1 is still recorded as failed; the confirmation resolves its only
uncertain boundary and admits the mechanism to a small LM quality pilot. It
does not admit a production executor or establish a capability gain.
