# Causal write/read plane T22a-v4 — router-mismatch count correction

Status: **FROZEN AFTER CPU PREFLIGHT, BEFORE ANY T22 GPU TRAINING OR RESULT**  
Date: 2026-07-31

This file supersedes only the router/support mismatch count stated in
`causal-write-read-plane-t22a-preregistration-v3.md`, SHA-256
`ad563abdc209e2ac465a21028d8975059cbe0bb351d3308015f42ce60663ddc5`.
All distinct-title counts and all protocol terms in v3 remain unchanged.

The exhaustive 104-row preflight found exactly **two** rows on which the
unchanged label-blind longest-nonoverlapping-title router differs from the
sealed supporting-title pair:

1. `5ac0d9a35542992a796ded90`: routed `Hungry Hungry Hippos` and
   `Parker Brothers`; support pair `Hungry Hungry Hippos` and `Parcheesi`.
2. `5ae0847455429945ae959394`: routed `Card game` and `Hanafuda`; support
   pair `Hanafuda` and `Okey`.

The two added router titles are therefore exactly `Parker Brothers` and
`Card game`.  The support/router union remains 207 distinct held-out raw
documents, the writer-training partition remains 2,198, and held-out read
evaluation remains 1,656 probes.

The implementation must assert `router_support_mismatches == 2`.  It must not
replace either routed pair with evaluator support labels.  Both errors remain
priced in the unchanged 75% development-accuracy and 10-point superiority
gates.  No seed, model, data, writer, probe, control, optimizer, schedule,
compute ledger, threshold, physical ledger, or decision rule changes.

