# Radial-trust Cayley program tree — 10M-token language pilot

Status: **frozen before GPU execution of this candidate**  
Date: 2026-07-30

## Claim under test

Edge B: retain full-SwiGLU language quality and its resident FFN parameter
budget while activating only an `O(D log D)` conditional path.  At the frozen
width this is 8.203125% of dense SwiGLU's ideal multiply-like FFN work.

The unbounded v1 recurrence is closed.  This candidate differs at one
mathematically forced point only: every raw edge update `d` is transformed by

`d / sqrt(1 + mean(d^2))`

before the `1/sqrt(9)` residual addition.  The radial map adds no learned state,
has a full-rank Jacobian for every finite input, and proves internal RMS at
most four for RMS-one input.  All payloads, initialization, topology, routes,
depth, basis count, and Neumann order remain v1-exact.

## Frozen artifacts

- Module: `cf64ab47717ec0cb8a8720fbd768adf48ca9785f0169253966f5a1539ff70226`
- Pilot: `79cdccbbaffb11428f5fc5e1368c9f690a56538f74d7cafdb8c5914c7f806ca5`
- Stage-0 result:
  `973a2a210dce047544347c7b2cecbca1805c4f707553bcc2753c47d3740cfa6e`
- Stage-0 passed all six machine gates and three tests.
- Frozen train SHA-256:
  `1871a8a790e2b2ae5273f1a46bc9fa2e39cbe95d5cb7537bde5a2b3e35cb464a`
- Frozen validation SHA-256:
  `889188f0e4c43cd49b2933ac462e8bd367520f15fe08d324f169f7eca962ce7b`

## Exact resource ledger

Each candidate FFN has 1,179,583 learned scalars, 65 fewer than the
1,179,648-scalar width-1,024 SwiGLU.  Across twelve layers the complete model
may not exceed the baseline model's parameter count.

The hard path activates nine edge payloads and the frozen sparse-Cayley
operators.  The v1 multiply-like upper bound is 96,768 operations per FFN,
8.203125% of dense SwiGLU.  Radial trust adds per depth one `D`-element square,
one reduction, one broadcast scale, and one scalar square root.  These costs
are excluded from the 8.203125% number and must be charged by later fused
hardware measurement.  They preserve `O(D log D)` scaling.

## Frozen protocol

- Reuse the completed full-SwiGLU arm only if environment, data, seed, schedule,
  batch, token count, and all 128 evaluation batches match exactly.
- Seed 815.
- 305 optimizer steps; 30 warmup steps.
- Sequence 512; microbatch 8; accumulation 8; global batch 64.
- AdamW `3e-4`, betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay `0.1`.
- Gradient clip 1.0; route-balance coefficient 0.01.
- 9,994,240 prediction tokens.
- Terminal evaluation: the same 128 batches of 32 sequences.

## Promotion gates

All gates must pass:

1. Protocol, artifact, data, environment, schedule, and finite-training gates.
2. Candidate FFN and total model parameter counts do not exceed full SwiGLU.
3. Candidate terminal NLL is no more than 0.5% worse than full SwiGLU, with
   paired upper 95% interval inside that margin.
4. The disclosed ideal active ratio remains below 9% before trust overhead.
5. Initial and terminal MLP activations are finite.
6. Median route perplexity is at least 1.5, maximum global bit load is below
   90%, and depths 6–8 visit at least 25% of reachable nodes.
7. On identical first-32 terminal batches, forcing every tree to the all-zero
   path worsens NLL by at least 0.5%.

Failure closes this exact radial-trust tree and schedule; it is not tuned.
A pass is not yet a breakthrough.  It authorizes exact-byte top-1 MoE,
structured-matrix, additive-memory, frozen-route, longer-training, and fused
hardware controls.  Only retained quality plus verified resource advantage
against those controls can support the edge-B claim.
