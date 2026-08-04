# Causal write/read plane T22a-v3 — distinct-title count correction

Status: **FROZEN AFTER CPU PREFLIGHT, BEFORE ANY T22 GPU TRAINING OR RESULT**  
Date: 2026-07-31

This file supersedes only the distinct-title arithmetic in
`causal-write-read-plane-t22a-preregistration-v2.md`, SHA-256
`66604efec55201e274d831bd74c97ae890082cc1e9c6bff818f73c1ec910ba17`.
The root T22a preregistration remains SHA-256
`e6517016b796caef7d4ba8d4e628efa38d5aa16dd3b508d96f84d01205554445`.

V2 incorrectly treated 208 development support-title mentions as 208 distinct
titles.  The executable raw-data preflight established the exact counts:

- 208 support-title mentions but **205 distinct sealed support titles**;
- 208 router selections but **205 distinct router-selected titles**;
- the union contains **207 distinct writer-held-out documents**;
- the writer-training partition contains **2,198 documents**;
- held-out read evaluation contains `207 * 8 = 1,656` probes.

The router/support mismatch remains exactly one development row, and the union
still adds exactly the two non-support router titles `Card game` and
`Parker Brothers`.  No QA-training routed title overlaps the 207 held-out
documents.

V2's corrected mechanism is unchanged: writer specialization receives zero
held-out gradients; the frozen writer processes every held-out raw document
exactly once; the dense control may train on all documents; and the router's
real error remains priced.  All model, data, seed, optimizer, schedule,
compute, capability, causality, natural-quality, physical-ledger, and decision
thresholds from the root preregistration and v2 remain frozen.

The isolation gate uses the corrected exact counts: writer train = 2,198,
writer held-out = 207, compile minimum = maximum = 1, and total held-out
compile passes = 207.

