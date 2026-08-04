# Radial-trust Cayley B2 typed implementation map

Date: 2026-08-01  
Status: **CPU IMPLEMENTATION MAP; NO CUDA OR GPU RUN AUTHORIZED**

Frozen parent:
`results/radial-trust-cayley-b2-physical-preregistration.md`, SHA-256
`31ed1d8e6cffcd9b189dc4736e55bb5ef175a8bdf3fbcc514dc8ae13891e256e`.

This document only extracts the already-frozen graph into auditable blocks. It
does not change an equation, tolerance, schedule, workload, control, or gate.
Sections 3 and 6 of the frozen parent already published the required symbolic
dependency and traffic sheet before implementation began. This map makes the
code boundaries and independent checks explicit.

## 1. Frozen scalar identities

```text
D=4096, M=14336, N=7166, E=14332
K=3 bases, q=3 matchings/basis, p=4 Neumann steps
L in {12,13}; E[L]=52223/4096=12.749755859375
residual scale FP32 bits=0x3e8e00d5
mean multiplier FP32 bits=0x39800000
```

For a path of depth `L`:

| Quantity | Symbolic value | `L=12` | `L=13` |
|---|---:|---:|---:|
| sparse `A` applications | `8L` | 96 | 104 |
| radial reductions | `L` | 12 | 13 |
| radial normalizations | `L` | 12 | 13 |
| requested descriptor bytes | `8L*3*D*4` | 4,718,592 | 5,111,808 |
| requested coefficient bytes | `8L*3*D*2` | 2,359,296 | 2,555,904 |
| selected payload bytes | `L*3*D*2` | 294,912 | 319,488 |
| selected threshold bytes | `2L` | 24 | 26 |
| named multiply/square/scale operations before SiLU expansion | `L(33D+1)` | 1,622,028 | 1,757,197 |
| named add/subtract operations before final output subtraction and SiLU expansion | `L(28D+1)` | 1,376,268 | 1,490,957 |
| preceding row plus final `h-h_initial` | `L(28D+1)+D` | 1,380,364 | 1,495,053 |
| named SiLU evaluations | `LD` | 49,152 | 53,248 |
| square roots and scalar divisions | `L` each | 12 each | 13 each |

Expanding each SiLU contributes `LD` each of negate, exponential, addition,
division, and multiplication. Conversion, comparison, descriptor decoding,
integer addressing, loads/stores, and synchronization remain separately typed;
they are not hidden in these arithmetic totals.

At `L=13`, the unique learned address set is:

```text
selected payloads       319,488 bytes
all three basis weights  36,864 bytes
selected thresholds          26 bytes
learned subtotal        356,378 bytes
shared topology         147,456 bytes
learned plus static     503,834 bytes
```

Unique bytes, requested bytes, cache traffic, and HBM traffic are four distinct
quantities. Only counters may establish physical cache or HBM traffic.

## 2. Dependency graph

```text
packed artifacts + one layer's BF16 input h + layer/depth/node
  -> U0 descriptor decode and gather
  -> U1 one sparse A_k application
  -> U2 four-step C_k(h)
  -> exact route coordinate and threshold comparison
  -> selected gate/up/down payload
  -> U3 diagonal payload SwiGLU
  -> U2T four-step C_k^T(delta_local)
  -> U4 radial trust reduction and normalization
  -> BF16 residual update + next node
  -> repeat the complete edge 12 or 13 times
  -> U6 BF16 layer output h_final-h_initial

12 independent (layer input, layer artifact) pairs
  -> enqueue U6_0(input_0),...,U6_11(input_11) in ascending order
  -> one-stream serialization, never output-to-next-input chaining
  -> U7 collection of 12 distinct layer outputs
```

The route comparison precedes payload selection. Within one U6 call, the next
edge depends on the prior edge's BF16 residual result. Neither dependency may
be speculated across, reordered, or replaced with a soft route. Between U6
calls, U7 has a frozen stream-order dependency but no data dependency: every
layer reads its own workload slice and no layer output becomes another input.

## 3. Artifact and trace cards

### A0: immutable topology and learned arrays

- **Input:** exact formula seeds and per-layer `PCG64` construction seed.
- **Output:** shared descriptor `[3,3,4096]` little-endian `uint32`; per-layer
  basis coefficients `[3,3,2048]` BF16; payload `[7166,2,3,4096]` BF16 in one
  C-order allocation; thresholds `[7166]` BF16.
- **Meaning:** formula topology plus the exact frozen natural-route artifact.
- **Invariant:** reserved descriptor bits are zero; each matching is a
  fixed-point-free involution; every pair shares one weight with opposite sign;
  base pointers are 256-byte aligned in the serving process.
- **Independent check:** a separately written scalar formula generator compares
  every one of the 36,864 raw packed words; a separate RNG reconstruction hashes
  every packed learned array.
- **Controls:** all-zero threshold is exact and consumes no RNG; corrupt one
  reserved bit, partner, pair, sign, draw-order step, or allocation stride and
  require rejection.
- **Kill:** any word/hash/layout mismatch invalidates the artifact before CUDA.
- **Does not prove:** learnability, useful routes, latency, or capability.

### A1: frozen BF16 input traces

- **Input:** `(family, tuning/evaluation)` only.
- **Output:** exact frozen row count by `4096`, with binary64 source and BF16-RN
  packed representations and hashes.
- **Meaning:** Gaussian or Rademacher workload rows, not language states.
- **Invariant:** shapes and seeds cannot be caller-overridden.
- **Independent check:** small golden artifacts plus exact wrapper-call tests for
  all four family/split branches; final target artifacts are generated only in
  the pinned CPython/NumPy environment and then frozen by hash.
- **Controls:** wrong seed, row count, width, dtype, or tuning flag must change a
  golden or fail the contract.
- **Kill:** a target artifact generated outside the pinned environment or without
  a frozen hash cannot enter correctness or measurement.
- **Does not prove:** that these traces approximate a language-model hidden-state
  distribution.

## 4. Numerical operator cards

### U0: packed descriptor decode and gather

- **Input:** one basis/matching ID, output coordinate `j`, immutable descriptor,
  BF16 coefficient row, and BF16 vector `v[4096]`.
- **Output:** `(partner:uint12, pair:uint11, sign:bit)` and one signed BF16 term.
- **Meaning:** one edge of a signed perfect matching.
- **Formula:** decode bits `0..23`; XOR recurrence transpose sign when required;
  gather `v[partner]` and `w[pair]`.
- **Cost:** one 4-byte descriptor and one 2-byte coefficient request per output
  per matching, plus the partner-vector request and integer decode/address work.
- **Independent check:** exhaustive host formula decoder; CUDA may share constants
  but not index-generation code.
- **Controls:** positive, negative, corrupted-reserved-bit, wrong-pair, and
  non-involution cases.
- **Kill:** any decoded-field or gathered-term mismatch kills the kernel ID.
- **Does not prove:** coalescing, cache reuse, or low HBM traffic.

### U1: one three-matching sparse `A_k v`

- **Input:** BF16 `v[4096]`, one basis's descriptors and coefficients.
- **Output:** BF16-RN `A_k v[4096]`.
- **Meaning:** one learned skew sparse linear application.
- **Formula:** per coordinate, one explicit FP32 RN multiply followed by two
  ordered FP32 RN FMAs in matching order `0,1,2`, then BF16-RN.
- **Cost:** `3D` multiplies and `2D` additions in the mathematical graph;
  `3D*4` descriptor bytes and `3D*2` coefficient bytes requested.
- **Invariant:** fixed matching order, exact transpose sign rule, BF16 boundary
  after every application.
- **Independent check:** scalar staged-rounding reference with separately
  generated topology.
- **Controls:** zero vector, basis vector, random vector, shuffled matching order,
  transpose inner-product identity, and sign corruption.
- **Kill:** numerical mismatch kills the kernel ID. Incorrect symbolic requested-
  byte arithmetic or violation of the frozen counter protocol is a harness
  failure. A greater than 20% predicted/observed traffic or latency-model error
  is an instrumentation warning, and unfavorable valid measured traffic is
  scientific evidence; neither is relabeled as a harness failure.
- **Does not prove:** the four-step recurrence or complete edge is faster.

### U2 and U2T: four-step Cayley approximants

- **Input:** BF16 source `v[4096]` and basis ID; U2T flips the recurrence sign.
- **Output:** BF16-RN `C_k(v)` or `C_k^T(v)`.
- **Meaning:** four serial sparse Neumann powers and the final `2*sum-source`.
- **Formula:** exactly four U1 applications with a BF16 term after each power;
  FP32 ordered accumulator; exact final FP32 RN FMA; BF16-RN output.
- **Cost per call:** four sparse applications, `13D` multiplications and `13D`
  additions/subtractions; four inter-power dependencies.
- **Invariant:** no term, accumulator output, or basis output survives a frozen
  precision boundary at higher precision than allowed.
- **Independent check:** staged PyTorch target-width reference plus CPU FP64
  diagnostic; only the staged reference is the serving oracle.
- **Controls:** zero, small-radius, large-radius, alternating-sign, transpose
  identity, reordered powers, and omitted BF16 boundary.
- **Kill:** staged-reference mismatch or nondeterminism kills the kernel ID.
- **Does not prove:** approximation quality outside the B1 conditioning domain or
  useful learned routing.

### U3: selected diagonal SwiGLU payload

- **Input:** BF16 `z,g,u,d[4096]`, with payload address selected by the exact route.
- **Output:** BF16-RN local delta `[4096]`.
- **Meaning:** coordinatewise gate/up/down transformation for one selected edge.
- **Formula:** FP32 `g*z`, FP32 `u*z`, frozen precise SiLU sequence, two ordered
  FP32 products, then BF16-RN.
- **Cost:** `4D` named payload multiplications, `D` SiLU evaluations, and
  `3D*2` selected payload bytes.
- **Invariant:** no runtime payload regeneration, alternate `exp`, fast math, or
  higher-precision retained local delta.
- **Independent check:** scalar libdevice-compatible sequence with frozen special
  values and target-shape staged reference.
- **Controls:** zero, symmetric signs, extreme finite BF16, NaN rejection,
  shuffled payload, and identity payload.
- **Kill:** special-function, rounding, finiteness, or payload-address mismatch
  kills the kernel ID.
- **Does not prove:** conditional population gives language capability.

### U4: radial trust

- **Input:** BF16 raw delta `[4096]`.
- **Output:** BF16-RN trusted delta `[4096]` and diagnostic FP32 scale.
- **Meaning:** `delta/sqrt(1+mean(delta^2))`.
- **Formula:** FP32 squares; fixed 12-level pairwise reduction at strides
  `2048,...,1`; exact mean multiplier; precise sqrt and division; FP32 scaling;
  BF16-RN output.
- **Cost:** `D` squares, `D-1` reduction additions, one mean multiply, one scalar
  add, one sqrt, one division, and `D` scalings.
- **Invariant:** reduction order is independent of CTA size; trusted BF16 RMS is
  at most `1+2^-7`; NaN or infinity fails.
- **Independent check:** exact ordered scalar reduction and radial Jacobian
  checks from B1.
- **Controls:** zero, one-hot, constant, alternating, overflow-near, radius
  boundaries, shuffled reduction, and trust-disabled ablation.
- **Kill:** invariant, ordered-reference, or determinism failure stops B2.
- **Does not prove:** that trust improves language optimization or capability.

### U5: one complete routed edge

- **Input:** BF16 hidden state, `(layer,depth,node)`, topology, basis, threshold,
  and selected payload arrays.
- **Output:** next BF16 hidden state, exact route bit, next node, coordinate,
  radius, and trust diagnostic in untimed mode.
- **Meaning:** U2 route, U3 payload, U2T, U4, and exact BF16 residual update.
- **Formula:** route coordinate uses unsigned 32-bit wrap then `&4095`; route is
  ordered `z[c]-theta>=0`; residual FMA uses FP32 bits `0x3e8e00d5`.
- **Cost:** eight sparse applications, one payload transform, one reduction and
  normalization, route integer/comparison work, and the residual work.
- **Invariant:** payload read depends on the route; next edge consumes the BF16
  residual; timed mode stores no route or diagnostics.
- **Independent check:** CPU forced-route oracle, independent node/coordinate
  trace checker, and staged target-width reference.
- **Controls:** forced left/right, collapsed, uniform, natural, threshold tie,
  short/long leaf, shuffled payload, and identity-basis replay.
- **Kill:** any bit/node/coordinate mismatch, invalid depth, or numerical mismatch
  kills the kernel ID.
- **Does not prove:** complete-path speed, route entropy, or useful semantics.

### U6: one complete 12/13-edge FFN layer

- **Input:** BF16 batch `[B,4096]` plus one layer artifact.
- **Output:** BF16 `[B,4096]` equal to `h_final-h_initial`.
- **Meaning:** one CTA owns one token through its full tree path.
- **Cost:** the depth-indexed row in Section 1 plus final `D` subtractions;
  critical path contains `L` serial U5 edges.
- **Invariant:** path ends exactly when the selected child is `>=7166`; natural
  paths have depth 12 or 13 only; every compiled ID implements one bitwise
  numerical operator.
- **Independent check:** clear target-shape graph, CPU forced-route oracle,
  staged BF16 reference, and independent route checker.
- **Correctness matrix:** before tuning, every compiled candidate ID uses natural
  routes at every listed `B`, both tuning families, `r=0,...,7`, inside the
  complete 12-layer panel. Forced collapsed/uniform paths use every candidate ID
  only at `B in {1,8}`, both tuning families, `r=0,...,7`, and all 12 layers.
  After selection, natural routes use the selected arm/B configuration at
  evaluation `p=0`, both families, `r=0,...,31`, and every layer; collapsed and
  uniform rechecks use selected candidate primary-B configurations only, while
  I0 uses its frozen natural trace. Repeatability is the separate Gaussian,
  natural, `B=1,p=0,r=0` complete-panel cell repeated 100 times.
- **Timing matrix:** U6 is only the candidate slice of the natural one-layer
  diagnostic: both families and primary `B={1,2,4,8}`. No one-layer high-batch,
  collapsed, uniform, or I0 timing is inferred.
- **Kill:** correctness, repeatability, conditioning census, spills/resource
  limit, or frozen implementation-universe failure stops or removes the exact ID
  as specified by the parent. A slow but valid U6 is evidence, not yet a B2
  decision without U7 and matched controls.
- **Does not prove:** 12-layer amortization, dense-relative Pareto gain, or
  language capability.

### U7: serial matched panel of 12 independent layer calls

- **Input:** 12 distinct BF16 layer-input batches `[12,B,4096]` (or 12 distinct
  input pointers selected by the frozen row map) and all 12 layer artifacts.
- **Output:** 12 distinct BF16 layer outputs `[12,B,4096]`, one per input/layer.
- **Meaning:** the complete B2 serial matched panel under one-stream layer order;
  it is not an output chain or an end-to-end transformer simulation.
- **Cost composition:** persistent bytes sum over 12 layers plus one shared
  descriptor; serial critical paths add; reusable transient workspace takes the
  measured maximum while every actual graph/workspace byte is charged; physical
  traffic, latency, energy, and memory are measured rather than inferred.
- **Invariant:** call `ell` reads only input pointer `ell` and writes distinct
  output pointer `ell`; no output is consumed by a later call. Calls execute in
  ascending layer order on one stream and never concurrently. Graph replay
  contains no allocation, compilation, host routing, profiler, or extra
  synchronization; allocation snapshots reconcile exactly.
- **Independent check:** every layer output and route is checked against that
  layer's independently selected input rows; 100 identical complete-panel
  replays are bitwise identical.
- **Correctness matrix:** natural pre-tuning panels cover every compiled/configured
  arm tuple, every listed `B`, both tuning families, and `r=0,...,7`. Forced
  candidate panels and post-selection/repeatability checks use only the distinct
  cells frozen in U6 and the parent; there is no implied cross-product.
- **Timing matrix:** natural primary measures one layer and U7 for all four arms;
  natural high-batch measures U7 only; collapsed, uniform, and I0 candidate
  diagnostics measure U7 only at primary `B`; cold timing is candidate/dense at
  primary `B`. These schedules remain disjoint.
- **Controls:** strongest frozen dense schedules, exact-byte top-1 MoE, lower-byte
  Monarch family, and identity-basis diagnostic within the preceding matrices.
- **Kill:** the exact Section 14 dense-relative physical gate decides B2; an
  invalid harness yields no scientific result.
- **Does not prove:** learning from prose, better held-out knowledge/reasoning,
  production-model quality, or transfer beyond the frozen H100 stratum.

## 5. Composition rules

1. **Route error:** any route-bit mismatch is invalid; it is not averaged into a
   numerical error budget because it changes all downstream addresses.
2. **Numerical error:** after exact routes agree, every coordinate must pass the
   frozen BF16 predicate and each output must pass frozen NRMSE. Per-block error
   tolerances cannot be loosened to manufacture end agreement.
3. **Traffic:** symbolic requested and unique bytes are predictions of address
   demand only. L1/L2/HBM bytes come from the frozen counter protocol.
4. **State/workspace:** persistent allocations add; sequentially reusable
   transient storage uses actual peak snapshots; code, graphs, handles,
   metadata, common buffers, and unexplained residuals are all charged.
5. **Critical path:** four U1 recurrences form U2; U2 and U2T plus U3/U4 form
   one serial U5; `L` U5 blocks form U6. Twelve independent U6 calls form U7 by
   frozen one-stream serialization, not by output-to-input data chaining.

## 6. Remaining implementation order

```text
CPU spec PASS
  -> CPU FP64 forced-route oracle and independent trace checker
  -> target-shape staged BF16 reference in the pinned runtime
  -> runtime invariant/snapshot recorder and manifest schema
  -> custom CUDA U0 through U5, each correctness-audited
  -> U6/U7 plus frozen controls
  -> complete independent code audit
  -> only then: local-idle check or isolated H100 acquisition
```

The first empirically unresolved B2 fact remains physical: whether the exact
complete operator beats the matched dense control under the frozen serving
vector. B2 intentionally leaves every language and acquisition claim outside
its scope.
