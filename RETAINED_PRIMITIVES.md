# Retained primitives after the permutation-graph screen

Status: **primitive library; Ghost Gradient capability branch closed**  
Date: 2026-07-27

This file preserves mechanisms that are algebraically correct or operationally
useful without carrying forward the claims that failed. A retained primitive is
not evidence that a model becomes smarter.

## The retained set

| Primitive | What it can buy | What it cannot buy | Re-admission condition |
|---|---|---|---|
| Gauge Ghost Gradient: richer TVE Jacobian with an ordinary forward | Keeps the served Transformer, parameters, attention graph, and KV cache ordinary while changing the training vector field; it consistently helps an under-scaled optimizer at low LR | A capability gain: its bracketed tested optimum was `6.089153` NLL versus an already-better canonical endpoint at `5.918237`; the best ghost had 2.888% regret, became unstable at high LR, and trained about 1.5x slower | **Closed for capability.** Retain only as an example of a virtual Jacobian acting like a preconditioner; do not run the sealed seeds, scrambled control, fused backward, or scale test |
| Triangular value encoding from V/O gauge-null coordinates | Converts 960 formerly function-null coordinates per 37.8M layer into nonlinear pre-pooling value features with no new parameters, KV width, state, or metadata; five untouched 100M-token FineWeb-Edu seeds improved NLL by 0.077%-0.141%, with a 0.106866% mean gain | A strict capability/cost win or scale/domain generality; the deterministic mechanism screen falsified forward-path necessity on its seed, and full-head growth was worse than block-16 | Retain only as the virtual training Jacobian and algebraic source for Ghost Gradient unless a future result independently restores a need for nonlinear serving |
| Exact discrete state plus atomic commit | Structural-invariant preservation and protection against corrupt state; reversibility only for reversible updates | Selection of the semantically correct update | Name a useful semantic predicate that is cheaper to verify than to solve and charge every proposal, rejection, and retry |
| Learned proposal plus exact verification | A potential search-cost advantage on certificate problems where checking a correct answer is genuinely cheaper than finding it | General language correctness from a structural checker | Beat the strongest same-information exact solver under the full served-resource ledger |
| Disjoint matchings and involutions | Conflict-free parallel transport without a full output-value buffer | New learned information or higher semantic accuracy | A target-GPU benchmark must beat direct gather, block-local gather, and cycle-following in a frozen service cell |
| Beneš/butterfly switching | A fixed-depth, conflict-free route for every permutation; a useful sparse-mixer topology | Semantic route correctness, free feature mixing, or a novel architecture by itself | Only revisit as a matched conditional-butterfly experiment, not as an exact-validator claim |
| Parent pointers, DFS intervals, and approximate ancestry summaries | Explicit choices among state bytes, query work, update work, and approximation | A free memory reduction relative to the strongest exact representation | Pick a workload whose measured query/update mix favors the representation |
| Weighted union-find with group potentials | Exact monotone equivalence and relative-group queries in near-optimal state | A new model algebra or an edge over a matched RAM controller | Reuse only as an explicit typed tool; charge entity mapping, capacity policy, and pointer behavior |
| Private finite-field fingerprints | For constant or inverse-polynomial epsilon, exponentially smaller one-sided-error equality state than deterministic exact streaming | General semantic memory, exactness, or an edge over randomized digital recurrence | Use only for declared equality/checksum semantics; charge epsilon, RNG, field arithmetic, and adversarial-input assumptions |
| Packed Boolean presence plane | Sixteen times as many exact Boolean channels as BF16 scalars per byte; word-parallel monotone OR updates | More information bits, free semantic feature production, or an edge over bit-addressable recurrence | Use only when typed flags already exist; charge mask construction, lost analog state, and physical kernel behavior |
| Gauge-fixed value chart plus cyclic quadratic writer | Converts `r^2-r` redundant V/O basis directions into identifiable pre-cache quadratic features at the same logical value arithmetic and cache width | A better learned model: it lost 0/5 learned-routing worlds to GLU on NLL | **Closed for this writer.** Reopen only with a different algebraic mechanism and a new reason the failed inductive bias should change; do not tune this screen to rescue it |
| Shared feature straight-line circuit | Full-rank structured transforms can have linear or `D log D` circuit work; a matched feature DAG won 5/5 hierarchy worlds | A dense-projection replacement: a 25% Haar component broke mixed noninferiority in 5/5 worlds, and Haar error was 58.86x dense | **Closed for the fixed ancestry DAG.** Reopen only with independent evidence of reusable subexpressions in real jointly trained QKV/FFN transforms; do not add stages or routing to rescue this screen |
| Full-rank conditional Cayley tree | At the same `Theta(D^2)` resident FFN budget, a balanced tree activates `Theta(D log D)` work while a selected Cayley-diagonal edge has a provably rank-`D` update; the deterministic `D=4096` ledger is 118.15x below dense FFN MACs while an optimistic equal-active narrow MoE has local rank at most 121 | Language quality, arbitrary dense-map expressivity, independent information per path, broad tree-routing novelty, or physical GPU speed | **Active candidate, not a result.** Require the frozen language gate, exact-byte top-1/hierarchical MoE, BTT/Monarch, PEER, route-causality, multi-seed, and fused serving controls; close it if the compositional restriction loses quality |
| Exact temporal-innovation projection | For a supplied `k`-sparse change, updates an arbitrary dense `W x` by reading `k` columns; the exact RMSNorm scale can be folded around the cached projection | An exact sparse stack: generic dense fan-out makes the next change dense with probability one; frozen SmolLM2 exact changes were 100%, and quantized events stayed 81%-99% dense | **Closed as a free LLM transform.** Reuse only when an external subsystem already guarantees sparse innovations; charge cached state, indices, dense output traffic, gathers, and request-support divergence |
| Online-softmax cardinality theorem | Identifies the unique universally mean-neutral chunk router, `Z_p=sum exp(p s)`, and proves that ordinary `(m,l,o,n)` state cannot synthesize an anchor-sensitive case | Boundary-continuous attention or a capability win; raw peak, Gaussian-corrected peak, and confidence lift all failed the causal screen | Reopen only with an explicitly paid power-sum state plus a construction that removes or covers the hard boundary before LM or kernel work |
| Dense-exception scaling law | Distinguishes a real scaling break from a smaller quadratic coefficient: feature rank must satisfy `r(D)=o(D)`, while full-dense token/layer fallback rates must vanish; power-law exception spectra change regime above exponent `1/2` | A model candidate or novelty claim for structured-plus-residual layers; all quality, selector, residency, and physical-kernel questions remain | Require protected causal noninferiority and a decreasing exception fraction across at least three widths, plus a randomized/isotropic boundary; do not admit a single-width cap |
| Fatal-control-first testing | Rejects impossible superiority claims before training or GPU rental | A positive architecture result | Apply to every future pre-candidate |
| Static symmetric group-128 W4 FFN codebook | A verified `4.134259` resident bits/weight ledger with exact code, scale, topology, and tail accounting | A quality-preserving substrate for conditional decoders: SmolLM2-135M NLL degraded by `8.43619%`, 42.18x the frozen limit | **Closed as a no-recovery base.** Any data-aware quantizer, outlier path, mixed precision, fine-tuning, or decoder-funded recovery is a new costed hypothesis |
| One gauge-funded quadratic key feature per RoPE pair | Adds a sign-invariant second-order content address at unchanged learned-parameter count, dense Q/K matmuls, and KV width; the teacher-free primary passed every direct gate in 5/5 worlds | A retroactive pass of the rejected 3/5 package, a uniquely superior scale chart, LM quality, or free inference | Re-test the narrower chart-agnostic G1 claim on unseen worlds with a signed-linear no-harm task; then count coefficient access and the fused H100 epilogue before any LM screen |
| One learned address map plus channel-group relation transports | Shows that one content address can causally feed different relation transports to different channel groups; successful seeds used the wide path strongly | A reliable architecture edge: the sole weight-RMS successor produced one healthy seed and four under-scaled wide-map seeds, failing 7/10 gates | **Closed for AFTA.** Retain only the relation-transport algebra; do not rescue the `1x64 + 8x16` allocation with another optimizer, constraint, schedule, or task mix |
| Norm-funded invertible triangular chart | Adds pairwise cross terms before all following projections with no new matrix weight/MAC and reuses one redundant RMSNorm scale; the map is exactly baseline-containing at zero and invertible for bounded coefficient | A learned capability edge: after 1,200 steps its coefficients collapsed near zero and causal/mispairing ablations moved accuracy by less than 0.07 points | **Closed before formal screening.** Retain the gauge and cost proof only; do not retry fixed coordinate-coupling charts |
| Static heterogeneous feature-order SwiGLU | At identical three-matrix ledgers, RMS-matched `SiLU(g)` and `g*tanh(g)` channels provide quadratic- and cubic-leading GLU features; the even gate is a tied 2-for-1 SiLU pair and improved 10M-token NLL by 0.316% in development | A converged capability edge: two untouched seeds gained only 0.017%-0.026% at 10M and reversed to losses of 0.069%-0.086% at 50M, making the preregistered 4/5 gate impossible | **Closed for the fixed 50/50 recipe.** Retain the feature-order algebra and early-optimization boundary; do not tune ratios, scale, STE, or schedule to rescue it |
| Margin-damped token-conditioned feature binding | Chooses one of eight block-local value permutations from existing gate values; `tanh^2(top1-top2)` makes the route correction vanish quadratically at winner ties, with no learned router, selector bytes, state, or new matmul | A useful FFN: the route was high-entropy and strongly causal under identity/misroute ablations, but variance-normalized binding lost 0.681% NLL to SwiGLU and every nontrivial cross-mixing arm lost | **Closed as an FFN replacement.** Retain only the boundary-continuous sparse-choice primitive; use it only when alternatives already have independent semantics, and always include amplitude/fixed-relation controls |
| Phase-elastic packed precision | Reallocates one BF16 FFN's bytes into an INT8-plus-ternary width-1,024 base and a decode-only W4 width-1,472 branch; a shared tokenwise phase boundary exactly trains base prompt caches feeding full decode. Fresh-seed 50M exact-packed mixed-cache NLL improved 0.619%, actual cached NLL improved 0.629%, and the branch itself added 0.404% over base | A completed same-cost serving win: full decode has 2.4375x matrix coordinates, training was 1.60x slower, and the physical kernel was blocked when the Vast H100 exited and the account lacked replacement credit | **Retained as the leading candidate.** Resume only at the physical gate: fused in-mainloop INT8+ternary/W4 decode, strongest BF16 control, numerical replay, prefill/decode/end-to-end latency, workspace, and peak VRAM. Do not rerun quality screens unless the packed algebra changes |

## Why structural verification does not amplify capability

Let the desired semantic permutation be `p*`, the learned proposal be `p`, and
the verifier accept every bijection. For a non-identity required update,

\[
\operatorname{success}(p)=\mathbf{1}[p=p^*].
\]

Rejecting a non-bijection and retaining the old state is still a trajectory
error. Every wrong bijection passes. The verifier therefore improves state
safety, not exact task success. A Beneš network removes even the rejection case:
every switch setting is structurally a permutation, but most settings remain
semantically wrong.

A proposal/verifier path can create a real edge only when the checked relation
`R(x, y)` is sufficiently close to the actual target and verification is
asymmetrically cheap:

\[
C_{proposal} + K C_{verify} + C_{commit} < C_{best\ exact\ solver},
\]

while success, model bytes, state, latency, and retries remain inside the
frozen contract. This is possible for some certificate problems. It is not
established for general next-token modeling.

The workload family hit a three-way boundary:

1. A structural predicate checks safety but not semantics.
2. Local matching constraints admit exact augmenting-path or assignment
   controls.
3. Making the predicate hard enough to avoid those controls reduces the method
   to ordinary learned certificate generation and verification.

## Two-involution transport: exact result, systems-only value

Every finite permutation `p` can be written as two involutions. On each cycle
`(a0 ... ak-1)`, define `r(ai) = a(-i mod k)` and `q = r p`. Then
`r p r = p^-1`, `q^2 = identity`, and `p = r q`. The executable reference is
[`experiments/permutation_involution_factorization.py`](experiments/permutation_involution_factorization.py).

For `N=64`, feature width `d=16`, BF16 values, and `uint16` indices, the logical
payload is 2,048 bytes:

| Executor | Logical peak bytes | Dense-stage traffic model | Main caveat |
|---|---:|---:|---|
| Direct out-of-place gather | 4,224 | 4,224 | Pays one second value buffer |
| In-place cycle following | 2,216 optimistic | at most 4,224 before descriptors | Serial/irregular cycles; fixed points reduce value traffic |
| Two materialized involutions | 2,304 after factorization | 8,448 for two dense stages | Canonical factors can skip fixed points |
| Factorization end to end | 2,312 optimistic lower bound | not represented | Factor construction and its workspace are excluded |

The current Python factorizer does **not** achieve the 2,312-byte abstract
bound. It uses a byte per visited node, a Python cycle list, and simultaneous
objects whose physical allocation is much larger. The number is retained only
as a lower-bound target for a different packed implementation.

For row width `R`, permutation cycle count `c`, and fixed-point count `f`, an
executor that skips fixed points moves `2(N-f)R` value bytes with ideal cycles
and `4(N-c)R` with the two canonical involutions. The factors tie for cycles of
length one or two and pay an extra `2R(k-2)` for every cycle of length `k >= 3`.
They are potentially latency-nondominated only if parallel disjoint swaps
overcome that logical traffic disadvantage on real hardware. At this frozen
shape the whole 2 KiB tile fits easily on chip, so block-local direct
permutation is the first control.

## Beneš switch network: exact ledger and closure

For `N = 2^k`, a Beneš network has `2k - 1` dependent stages with `N/2`
disjoint 2x2 switches per stage. At `N=64`, `d=16`, BF16:

- stages: `11`;
- switches and route decisions: `11 * 32 = 352`;
- bit-packed route: `44` bytes;
- information floor: `log2(64!) = 295.995...` bits, or `37` whole bytes;
- ordinary byte-index route: `uint8[64] = 64` bytes;
- packed six-bit destination indices: `48` bytes;
- feature tile: `64 * 16 * 2 = 2,048` bytes.

Bit packing saves 20 bytes against byte indices but only 4 bytes against packed
six-bit indices, and an enumerative/Lehmer permutation code reaches the 37-byte
information floor at additional decode cost. A normal `uint8[11,32]` tensor
consumes 352 bytes, and BF16 route logits consume 704 bytes, unless a custom
streamed/packed implementation is used.

A one-block fused kernel could load and store the value tile once, for 4,096
bytes of HBM value traffic, but it still has 11 serial switch stages and ten
inter-stage barriers, plus an initial cooperative-load barrier and possibly a
final store barrier. Without fusion, eleven stages approach 45,056 bytes of
value traffic. Direct gather has the same one-load/one-store HBM lower bound
with one transport stage.

Adding shared nonlinear or invertible pair coupling makes the construction a
real sparse model layer, but not a new validation algebra. At the illustrative
shape `d=16`, hidden width 32, a shared additive coupling plus a local route MLP
costs about 3,233 parameters, 1,092,608 MACs, and 352 hard decisions. A matched
three-layer small Transformer core is about 9,840 parameters and 983,040 MACs
before secondary operations. The switch model is therefore not compute-free;
it needs direct capability controls.

The late collision check closes the broad novelty claim. Neural
Shuffle-Exchange already uses learned nonlinear switches for `O(N log N)`
sequence processing; learned butterfly factorizations cover sparse structured
transforms; ButterflyFlow covers invertible butterfly layers; and Dimension
Mixer explicitly covers nonlinear butterfly MLP and attention variants:

- [Neural Shuffle-Exchange Networks](https://openreview.net/forum?id=HylPsErlIS)
- [Learning Fast Algorithms for Linear Transforms Using Butterfly Factorizations](https://proceedings.mlr.press/v97/dao19a.html)
- [ButterflyFlow](https://proceedings.mlr.press/v162/meng22a.html)
- [Dimension Mixer](https://proceedings.mlr.press/v280/sapkota25a.html)

The only defensible future experiment is narrow: can conditional, weight-tied
butterfly wiring beat fixed butterfly, random matching, continuous 2x2 gating,
and a matched Transformer on capability per served resource? That is preserved
as a benchmark question, not admitted as candidate 004.

## Mandatory controls if any primitive is reopened

1. Identical proposer with direct gather and ideal cycle-following commit.
2. Strongest exact solver over the same visible semantic constraints.
3. No-op/rejection counted as an error whenever the gold state changes.
4. Fixed butterfly, random matching, learned soft 2x2 gates, and a matched
   ordinary sparse/Transformer layer.
5. Exact physical accounting for route logits, packing, construction, retries,
   barriers, descriptors, register/shared-memory pressure, and all traffic.

## Gauge-to-curvature result

The value/output factorization contains an exact `GL(r)` basis redundancy.
Fixing an identity value minor removes `r^2` dense projection operations.  The
retained cyclic-mask writer spends the resulting logical budget on

\[
v_j=u_j+z_j\sum_{k\ne j+1\pmod r}z_kH_{kj}.
\]

For `r>=3`, the forbidden edges form no fixed points and no two-cycles, so all
squares and at least one orientation of every cross monomial remain.  Under a
full-row-rank output block, the construction adds `r^2-r` identifiable
vector-valued quadratic directions while matching dense value multiplication,
addition, and cache-width ledgers.  The zero-diagonal predecessor failed its
preference gate; it reached 85.50% median accuracy on the zero-mean covariance
task, versus 86.04% for an exact-ledger square writer, and collapsed to 49.29%
when the covariance was axis-aligned.

The later learned-Q/K H100 gate rejected the cyclic writer as an architecture:
it lost all five worlds to paper-style GLU, with `-27.79%` median relative
excess-NLL improvement.  GLU also used 84 rather than 96 combined K+V cache
scalars per token/layer.  The cyclic result is therefore retained as algebra
only, not as a pre-candidate.  Details and executable evidence are in
[`results/gauge-to-curvature-stage0-decision.md`](results/gauge-to-curvature-stage0-decision.md)
and
[`results/gauge-curvature-learned-routing-decision.md`](results/gauge-curvature-learned-routing-decision.md).

No further GPU should be rented for the cyclic writer under this claim.  Any
other retained primitive needs a new paper/CPU argument and frozen fatal
controls before another rental.

## Capped scalar microdepth boundary

A fused H100 scalar-program screen retained groups of at most eight dependent
scalar operations as a plausible execution primitive, but the matched
50M-token language screen rejected using that primitive to replace SwiGLU's
gate matrix.  The full triangular arm reached `5.590961680` NLL versus
`5.527706977` for SwiGLU.  Zeroing its couplings caused only a `0.00387%`
relative loss increase, while removing all paths longer than one hop slightly
improved the result.

Retain only the executor boundary: a successor may use a group-local chain of
length at most eight if it preserves the complete SwiGLU endpoint and obtains
its coefficients without adding served weight storage.  Do not reuse the
wider two-projection SiLU recipe.

Evidence:
[`results/triangular-microdepth-lm-screen-decision.md`](results/triangular-microdepth-lm-screen-decision.md).

The first exact-endpoint successor encoded one coefficient in every neuron's
up/down scale orbit.  Its BF16 codec passed without a duplicate tensor, but
the deployed kernel required `B*M` strided pivot loads and seven dependent
steps.  Full-FFN median latency was `1.0336x`, `1.0329x`, `1.0274x`, and
`1.1016x` baseline at token counts 1, 8, 32, and 128.  This closes
per-feature coefficient decoding under the current claim.  A successor must
make the already-computed activation amplitude expose the scale coordinate
directly.

Evidence:
[`results/orbit-activated-swiglu-h100-decision.md`](results/orbit-activated-swiglu-h100-decision.md).

A parallel partner-feedback successor removed both costs.  It needed only one
layer-scalar pivot read per kernel program, no extra workspace, and no serial
feature dependency; full-path H100 overhead was `0.04%` to `0.42%`.  The
matched 50M-token screen nevertheless rejected it.  Full partner feedback
reached `5.535271119` NLL versus `5.535243854` raw SwiGLU and
`5.535192754` for the scale-insensitive partner-gate control.  Replacing the
trained partner path by self-feedback improved loss.  Retain the gauge-capacity
theorem and executor, not this architecture.

Evidence:
[`results/partner-feedback-swiglu-lm-decision.md`](results/partner-feedback-swiglu-lm-decision.md).

## Projection reuse boundary

A shared pair of width-1,344 generator banks can supply GQA Q/K/V slices,
1,344 SwiGLU products, and a common down/output matrix while using 1.56% fewer
dense projection weights and MACs than the ordinary width-1,024 block.  The
algebra exactly contains arbitrary parallel GQA plus an independent width-704
SwiGLU.  It trained stably and beat the sequential Transformer by 0.105% NLL.

It is not retained as an architecture.  An independent parallel attention and
SwiGLU control reached 5.452698 NLL, versus 5.527355 for the coalesced model,
and ran slightly faster during training.  This establishes the boundary:
reusing generator computation is not enough when it forces attention
selection, token transport, and local-memory features to share output columns.

Retain the independent parallel block as the baseline and require the next
reuse mechanism to give every constructed interaction an independent readout.
Do not repair this branch with different slices or partial sharing.

Evidence:
[`results/coalesced-attention-ffn-lm-decision.md`](results/coalesced-attention-ffn-lm-decision.md).

## Reused-generator boundary

At a matched `3DM` FFN ledger, a degree-two generator graph had full sampled
functional rank while ordinary SwiGLU retained one scale nullity per unit.
The algebra was real, but the unfused executor materialized `2M` gathered edge
features and ran 1.052x to 1.229x the parallel SwiGLU baseline.  Close fixed
graph reuse under this claim; do not pay for a language screen before an
independently justified online/fused executor exists.

The exact-budget self-product control survived three small-model screens: a
width-`1.5M` bank followed by `SiLU(a)*a` repeatedly improved NLL by
0.43%-0.68% over SwiGLU. Fair split and packed SwiGLU fusion then closed it as
a fixed-served-cost architecture. It was faster for decode and batch-1/8
prefill, but batch-32 prefill was 1.0295x split and 1.0424x packed SwiGLU; the
strict memory envelope also failed. Retain the quality/shape observation, not
the architecture. Equal dense MACs did not fund 50% more scalar nonlinear
work at large prefill.

Evidence:
[`results/generator-edge-ffn-h100-decision.md`](results/generator-edge-ffn-h100-decision.md).
[`results/self-product-ffn-scale-fused-h100-decision.md`](results/self-product-ffn-scale-fused-h100-decision.md).

The single-point latency rematch at width 2,560 made the self-product model
2.564% smaller overall and 2.5%-5.3% faster in decode and small/medium
prefill, but it remained 3.16% slower than packed SwiGLU at batch-32 prefill.
Replacing the sigmoid with a minimax hard gate changed that failure by only
about 0.1 percentage point.  The large-prefill tax is therefore the wider
intermediate/down shape and its materialization, not the transcendental.
Do not tune another width or activation approximation under this claim.

A width-`M`, degree-one cycle then tested whether each learned generator row
could serve once as a gate and once as a neighboring value at `2DM` cost.
It had full sampled functional rank and indefinite quadratic features, but at
10M tokens it reached 6.324109 NLL versus 6.294818 for full SwiGLU, 6.320083
for exact-cost narrow SwiGLU, and 6.316748 for self-product.  Relation
ablations were strongly causal, so this closes arbitrary same-token static
factor reuse rather than an ignored path.  Re-admit reuse only when the second
operand has independent semantics, such as causal temporal state.

Evidence:
[`results/self-product-ffn-latency-matched-h100.json`](results/self-product-ffn-latency-matched-h100.json).
[`results/hard-gated-self-product-h100.json`](results/hard-gated-self-product-h100.json).
[`results/cycle-factor-ffn-decision.md`](results/cycle-factor-ffn-decision.md).

## Nonlinear reduction placement boundary

An associative dual-number reduction can make partial dot-product state
functional rather than disposable:

\[
(p,q)\star(g,u)=
\bigl(p+g+apg,\ q+u+a(pu+qg)\bigr).
\]

It is associative and commutative, recovers ordinary summation exactly at
`a=0`, and increased sampled local function rank from 92 to 93 on the frozen
Stage-0 witness. It is therefore a real algebraic extension rather than an
optimizer or parameter-count illusion.

The frozen H100 implementation closed the obvious executor. Replacing the
GEMM K-reduction with a custom four-block fused reducer was 1.113x-1.174x the
fastest ordinary full-FFN latency at rows 1-32 and 2.867x-6.394x at rows
128-2,048. It saved activation memory and passed correctness, but forfeiting
the packed cuBLAS path dominated the ledger.

Retain the **placement rule**, not this executor: alter algebra only where the
production kernel already exposes a compact online reduction state. Do not
force a custom matmul mainloop merely to preserve intermediate products. The
next live branch is attention's existing online-softmax state, where a new
state transition may cost scalar/vector updates without new QKV/FFN matmuls,
KV width, or persistent activation tensors.

Evidence:
[`results/associative-nonlinear-reduction-stage0-preregistration.md`](results/associative-nonlinear-reduction-stage0-preregistration.md).
[`results/nonlinear-reduction-swiglu-h100-decision.md`](results/nonlinear-reduction-swiglu-h100-decision.md).

## Nonpairwise evidence boundary

Power-evidence attention showed that an attention router can use a statistic
shared across more than one token. In the matched learned screen, staggered PEA
improved mean overall accuracy by 6.151 percentage points over ordinary
attention. Removing its learned `p1` path reduced anchor accuracy in every seed,
by 6.533 points on average. This is causal evidence that cross-token routing
power was learned and used.

Hard chunks are closed. PEA's anchor advantage over cheaper IPA was only 3.687
points, below the frozen 5-point gate; staggering recovered only 0.239 boundary
points; and an affected head's origin-8 attention was distorted by 28.6%-32.0%.
Neither candidate passed. Do not repair the branch with more offsets, overlap,
or learned chunk boundaries.

Retain the narrower principle: **one token's evidence may alter the retrieval
weight of a related token**, but the relation must not introduce internal hard
boundaries. A successor must preserve an ordinary-head path, keep QKV/FFN
matrix counts and KV width fixed, and prove a relational gain against a matched
global-temperature control before a language-model run.

Evidence:
[`results/pea-ipa-learning-decision.md`](results/pea-ipa-learning-decision.md).

## Projection-codec boundary

Radix-64 packing can recover two independent binary-weight projections from
one ternary-activation INT8 dot stream exactly.  Parity resolves the K=32
endpoint ambiguity, and a width-1.5M packed SwiGLU has the same resident INT8
weight bytes and issued scalar products as an ordinary width-M SwiGLU.

It is not retained as an executor.  Decoding must occur after every K=32 tile;
ordinary accumulation across K creates unrecoverable carries.  On H100 at
K=4096 the exact packed kernel was 0.946x-0.960x the speed of a fused dual-INT8
kernel despite halving its INT8 products.  Native INT4 and binary-popcount
controls are stronger still because they preserve the long reduction and use
no more representation bits.

Retain the boundary: **do not interrupt a production long reduction to decode
an execution codec**.  A live successor must either preserve the native
mainloop and move conditionality to operand selection, or compute a second
useful algebra over already-loaded operands on an otherwise idle pipeline.

Evidence:
[`results/balanced-radix-projection-decision.md`](results/balanced-radix-projection-decision.md).

## Native-precision boundary

Two W4 bases have the same raw payload-bit count as one W8 matrix, and legacy
warp-MMA instruction extents make their nominal instruction counts look equal.
That is not a Hopper execution identity.  H100's native WGMMA integer path is
S8; the CUTLASS S4 kernel is a legacy SM80 warp-MMA path.  At `K=4096` and
`N=14336`, a concatenated dual-S4 ceiling was 74.999x to 87.769x the S8 WGMMA
time over 256 to 4,096 rows, before candidate-only routing, scale, combination,
or activation costs.

Retain the admission rule: **a low-bit algebra must match the fastest native
instruction family, not an abstract scalar-product or legacy-instruction
count.**  Precision decomposition is not a candidate until its complete
native kernel beats the strongest directly routed low-bit control and matches
the production baseline across the serving envelope.

Evidence:
[`results/interference-expert-h100-decision.md`](results/interference-expert-h100-decision.md).

## Partial-reduction state boundary

Splitting one dense W8 reduction into two INT32 accumulator banks exposes the
ordinary projection `s=A+B` and a tied contrast `d=A-B` with unchanged WGMMA
instructions, weights, and operand reads.  The algebra is real: the stacked
linear statistics can double sampled information rank, and a minimum
baseline-containing quadratic correction doubled the frozen local Hessian
rank.

The two-bank H100 placement is closed.  Its null path was bit-exact, used
Tensor Cores, and was effectively free from rows 1 through 256, but the extra
accumulator raised the best kernel from 64 to 96 registers per thread and cost
7.33% at 1,024 rows.  This failed the frozen 5% full-envelope limit before a
fused FFN increased accumulator lifetime further.

Retain the narrower rule: **compress a useful partial-reduction fact before
reusing the production accumulator.**  A successor may retain a few bits from
one checkpoint, but not a second full accumulator, and must keep the native
long reduction, avoid global intermediates, and pass decode plus large
prefill before training.

Evidence:
[`results/split-k-contrast-h100-decision.md`](results/split-k-contrast-h100-decision.md).

A per-output sign checkpoint did not compress physical state under the tested
compiler.  Keeping `sign(A)` live across the second half of K still raised the
selected kernel from 64 to 96 registers and cost 6.43% at 1,024 rows.  Logical
bit width is irrelevant when the live tensor has accumulator shape.

Retain the refined boundary: reduce checkpoint **cardinality**, not only its
numeric precision.  The next admissible executor may keep one statistic per
fixed feature group or a proven packed ballot word; it may not call an
unpacked lane predicate a one-bit resource.

Evidence:
[`results/mid-reduction-sign-h100-decision.md`](results/mid-reduction-sign-h100-decision.md).

Reducing checkpoint cardinality alone also failed.  One first-half sign per
fixed 64-feature group needed 24 extra registers because summarizing the live
accumulator tile raised peak state, then cost 8.20% at 1,024 rows.  Close both
export forms: full-lane predicates and group reductions.

Retain the stronger algebraic shift: **mutate and forget**.  A nonlinear
in-place map of the one live accumulator at a single checkpoint can make
reduction order functional, then resume the same WGMMA loop with no historical
state.  It must prove that the transient mutation itself does not raise the
kernel's register or latency envelope.

Evidence:
[`results/group-checkpoint-h100-decision.md`](results/group-checkpoint-h100-decision.md).

The first mutate-and-forget algebra survived both exactness and function-class
gates.  At one K midpoint, `A <- A + alpha*abs(A)` followed by the ordinary
second half gives `A+B+alpha|A|`; at `alpha=1` this is `B+2ReLU(A)`.  Its two
regional gradients span two directions, while a final-only hinge of `A+B`
spans one.  A paired up/down sign gauge can fund an on/off mode bit without a
stored parameter.

The weak Triton executor was misleadingly free.  In the native S8 CUTLASS
mainloop, resources stayed identical but the required new WGMMA drain cost
1.80% at rows 256 and the selected kernel did not cover decode.  Close the
exact-width S8 executor, not the algebra.

Retain the new placement rule: **spend an existing mandatory accumulator
synchronization, never create one.**  Accurate FP8 promotion and synchronous
weight-only decode kernels are the next admissible sites.  Any successor must
compare against an equally synchronized final-hinge control.

Evidence:
[`results/inplace-hinge-cutlass-h100-decision.md`](results/inplace-hinge-cutlass-h100-decision.md).

Accurate FP8 supplied the existing synchronization that S8 lacked, but did not
make a per-output nonlinearity free.  The FP8 candidate kept identical
resources and dependency-barrier count, yet its 128 extra selects plus 128 adds
cost 3.02%-4.22% across rows 64-4,096.  Close all full-output-tile mutation
placements tested so far.

Retain the higher-level cost law: **share cheap nonlinear work before a group
of large projections; do not repeat it over every projected feature.**  A
width-preserving input coupling can be computed once for Q/K/V or gate/up in
`O(d)`, while a midpoint accumulator map costs `O(m)` per projection and is
not amortized.  The successor must preserve the ordinary model at a null
setting and fund its coefficients from an exact parameter gauge.

Evidence:
[`results/inplace-hinge-fp8-promotion-h100-decision.md`](results/inplace-hinge-fp8-promotion-h100-decision.md).

Projection-shared nonlinear charts passed the systems boundary: replacing a
fused RMS gain with two elementwise shears kept 32 registers, zero spills, and
the same shared memory, while full norm-plus-projection latency stayed within
0.284% at the important H100 cells.  Moving `O(d)` work before a projection
group is therefore viable.

The first learning forms are closed.  Two-way coupling lost to a simpler
self-hinge at 10M tokens.  Self-hinge then beat raw RMS gain at 10M on a second
seed but lost by 0.001277 NLL at 50M.  Zero ablation showed the feature was
used; folded-null showed removing gain was harmful.  Do not replace an
optimization-useful gauge merely because it is functionally redundant at
inference.

Retain the next cost trade: **overparameterize only during training, quotient
the gauge exactly at export.**  Train with both RMS gain and nonlinear chart;
fold the gain into all immediate weight-matrix columns, delete it, and serve
only the chart in the original `d` parameter slots.

Evidence:
[`results/projection-shared-self-hinge-lm-replication-decision.md`](results/projection-shared-self-hinge-lm-replication-decision.md).
