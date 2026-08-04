# Phase-elastic FFN v2: residual-precision base

## Why this refinement exists

V1 passed the storage, stability, full-quality, and branch-causality gates:
the full path beat dense BF16 by 0.2811%, and its optional branch improved over
base-only by 0.3746%.  It failed only the protected prefill endpoint: W8
base-only was 0.0938% worse than dense, beyond the frozen 0.05% margin.

V2 changes one allocation, not the direction.  The always-on width-1,024 base
receives a per-coordinate ternary residual in addition to INT8.  The optional
W4 branch shrinks from width 1,984 to 1,472 to pay for it.  An independently
measured pretrained endpoint showed that the same INT8-plus-ternary format
recovered 91.67% of the INT8 base's NLL tax.  V2 tests whether that precision
exchange closes the prefill gap while retaining useful decode-only capacity.

## Exact per-layer ledger

- BF16 reference: 2,359,296 bytes.
- base INT8 plus ternary codes: 1,474,560 bytes.
- base's two BF16 scale tables: 9,728 bytes.
- width-1,472 W4 branch codes: 847,872 bytes.
- branch BF16 scales: 6,656 bytes.
- total: 2,338,816 bytes; slack: 20,480 bytes.

The base-only path retains 1.0x reference matrix coefficients.  The full path
is 2.4375x.  As in v1, these are quality/storage ledgers; no runtime result is
inferred.

## Frozen protocol and decision

Everything else is identical to v1: model, seed 815, data order, 305 steps,
optimizer, stochastic branch presence, evaluation batches, and thresholds.
V2 advances only if:

1. full NLL beats dense by at least 0.10%;
2. base-only is noninferior to dense within 0.05%;
3. full beats base-only by at least 0.10%;
4. all paired intervals, packed bytes, code bounds, path-use, stability, and
   integrity gates pass.

A pass authorizes an untouched 50M-token replication of v2, not a kernel yet.

