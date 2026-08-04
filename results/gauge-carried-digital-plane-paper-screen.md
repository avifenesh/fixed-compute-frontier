# Gauge-carried digital plane — storage exists, useful inference path does not

Status: **REJECT AS AN ACTIVE-GOAL CANDIDATE; ADMIT NO MODEL OR GPU RUN**  
Date: 2026-07-31  
Scope: exact parameter symmetries used to carry corpus-derived discrete state

## Decision

A compiler can encode bits in the choice of representative inside a trained
model's parameter-symmetry orbit without changing the real-arithmetic function.
That is a genuine lossless storage channel.  It is not, by itself, a way to
make the served model know or reason about the encoded facts.

The obstruction is exact: if the ordinary forward graph remains invariant to
the gauge choice, every prompt receives the same logits before and after the
write.  If a new path makes the logits depend on the gauge choice, the path has
broken the symmetry.  The orbit coordinate then becomes functional model state
and the read path must be charged in weights, operations, traffic, workspace,
and critical depth.  Which matched carrier can preserve the same base function
and payload at equal cost is a separate comparison; equal byte count alone
does not prove absorption by explicit state.

Therefore this direction does not supply the missing raw-prose-to-useful-plane
mechanism under the frozen serving contract.  Do not implement or benchmark it.

## 1. The clean algebra

Let a finite served model have parameters `theta` and forward function

\[
f_\theta:q\mapsto z,
\]

where `z` is the complete logit sequence.  Let `G` be an exact parameter
symmetry group:

\[
f_{g\cdot\theta}(q)=f_\theta(q)
\quad\text{for every }g\in G\text{ and every }q.
\]

Examples in exact real arithmetic include:

- for SwiGLU, a joint permutation of the gate/up output coordinates and the
  corresponding input coordinates of the down projection;
- a value/output head basis change
  `(W_V,W_O) -> (W_V C,C^{-1}W_O)` for invertible `C`, applied consistently
  inside each value-sharing/output group;
- a query/key basis change in the positional-encoding regimes where it
  commutes with the score computation.

Suppose a raw-prose compiler selects a group element `g(D)` from corpus `D` and
writes

\[
\theta_D=g(D)\cdot\theta.
\]

The checkpoint representative can reveal `g(D)` to a white-box decoder.  But
the unmodified served graph satisfies

\[
f_{\theta_D}(q)=f_\theta(q),
\]

so its held-out answer distribution, ranking, calibration, and output-token
distribution are identical for every corpus payload.  Gauge-related hidden
coordinates need not themselves be identical.

### No-use theorem

**Theorem.** An exact gauge-orbit writer followed only by a gauge-invariant
served graph has zero causal effect on all model-output metrics.

**Proof.** Pointwise equality of the complete logits for every prompt implies
equality of every deterministic function of those logits and equality of every
sampled-output distribution under the same sampler and random seed.  Therefore
the intervention `g(D)` cannot change the evaluation metric. `QED`

**Plain version.** The compiler has written on the model's coordinate system,
not on anything the model's normal computation can observe.

## 2. Why adding a reader removes the claimed free edge

Let an added read path be `R(q,theta_D)` and let the new output be

\[
F(q;\theta_D)=H(f_{\theta_D}(q),R(q,\theta_D)).
\]

For the payload to help, there must exist `g_1,g_2` in the admitted codebook
and a query `q` such that

\[
F(q;g_1\cdot\theta)\ne F(q;g_2\cdot\theta).
\]

So `F` is not gauge invariant.  The orbit coordinate is now a functional
coordinate of the architecture.  Its reader requires at least one of:

1. extra executed operations or memory traffic that inspect the checkpoint;
2. extra persistent state such as canonical reference weights, keys, or an
   index;
3. removal of some existing served computation to fund the reader; or
4. offline decoding into ordinary functional weights.

Cases 1 and 2 fail the componentwise serving ledger unless an equal amount is
removed.  Case 3 is a new reallocating architecture whose removed capability
and strongest matched reader control must be tested.  Case 4 is an ordinary
compiler or fine-tuning pipeline; the gauge channel has supplied no served
operator advantage.

This is not an impossibility theorem for every reallocating self-reading
architecture.  It closes the claim that gauge storage itself is a free digital
plane.

## 3. Capacity is real and bounded; the discrete lane is smaller than the frozen plane

An arbitrary permutation of `m` distinguishable hidden units can select at
most

\[
B_{perm}(m)=\left\lfloor\log_2(m!)\right\rfloor
\]

fixed-length payload bits, assuming a decoder also knows the canonical
representative.  If the parameter has a stabilizer subgroup `S`, the orbit
bound is instead `floor(log2(m!/|S|))`.  For `m=1024`, `log2(m!)` is
`8,769.006`, so the fixed-length ceiling is 8,769 bits (`1.0704 KiB`) per FFN.
Ten independent width-1,024 FFNs carry at most 87,690 bits (`10.7044 KiB`).
Pair-swap codebooks carry less.  This is below the frozen 683,148-byte exact
raw plane; it is not a theorem that a 10.7-KiB payload cannot be useful.  Under
the no-use theorem, however, none of those bits affects the normal forward
outputs.

Continuous `GL(r)` gauges have many real coordinates, but they do not evade the
finite checkpoint bound.  A `B`-bit checkpoint has at most `2^B` physical
states, so it cannot carry more than `B` total recoverable bits including the
base model; additional function-preserving capacity is bounded by the number
of physically distinguishable representatives in the orbit.  Exact payload
recovery, exact real-arithmetic functional invariance, and machine-close
deployed outputs are three different claims.  Quantized matrix inversion,
conditioning, canonical-reference data, rounding, and reduction order can
shrink the reliable physical codebook even when the real algebra is exact.
Those are secondary problems; observability already rejects the candidate.

## 4. Mandatory carrier controls

Equal checkpoint bytes do **not** prove that explicit state can reproduce the
gauge payload while retaining the same base function: explicit state may have
to displace useful base-weight bits.  A future symmetry-breaking reader would
therefore need three separate controls under the complete ledger:

- the identical gauge carrier with the strongest legal reader, which tests
  whether the proposed reader rather than the encoding creates the gain;
- explicit discrete state funded by removing weights, which measures the base
  capability sacrificed for direct readability; and
- alternative redundant carriers with the same payload, preserved base
  function tolerance, and reader budget.

If a payload is decoded offline into served functional weights, a direct
compiler is an additional control.  No complete reader or carrier comparison
exists here, so no online capability advantage is established.  This missing
comparison is not needed for the narrower no-use theorem.

## 5. W0–W5 audit

### W0 — structural statistic

Absent.  The carrier can encode any already-produced bit string, but it does
not define which raw observable autonomously extracts titles, relations, or
answer-bearing facts.

### W1 — positive construction

Passes only as storage: a compiler can map a bit string to a neuron
permutation.  It fails as a model construction because the normal logits are
provably unchanged.

### W2 — failure theorem

The no-use theorem is the exact boundary.  Every query fails to observe every
payload while the served graph remains gauge invariant.

### W3 — strong-control absorption

Unresolved.  Equal-byte explicit state is not automatically equivalent because
it may displace base weights.  The identical gauge carrier, explicitly funded
state, and alternative redundant carriers must be compared once a concrete
reader exists.

### W4 — complete bytes and work

Not supplied for any useful reader.  White-box extraction needs a canonical
reference/key or a deterministic canonicalizer and scans model weights; an
online inference reader must charge the corresponding state, traffic, work,
and depth.

### W5 — effect-size path

Exactly zero without a reader.  Not non-vacuously bounded and multiply
uncertain with one:
raw-prose extraction, code reliability, query decoding, integration, and
reader reasoning all remain empirical.

## 6. Prior-art and novelty boundary

Function-preserving payload storage is already an active watermarking lane.
[Functional Invariants to Watermark Large Transformers](https://arxiv.org/abs/2310.11446)
uses dimension permutations and scale/unscale transformations to make
functionally equivalent copies.  [TransMark](https://doi.org/10.1016/j.csi.2026.104180)
encodes payloads by FFN neuron permutations and claims exact functional
invariance.  A 2025 [complete Transformer gauge characterization](https://neurips.cc/virtual/2025/136893)
describes per-head query/key and value/output groups and their RoPE boundary.

The present no-use result is not a novelty claim over those works.  It states
why their white-box storage mechanism cannot become the active project's
useful online knowledge plane without paying for and controlling a
symmetry-breaking reader.

## Retain

Retain two design laws:

1. **Redundant checkpoint coordinates can carry offline metadata but not
   online capability through an invariant graph.**
2. **A useful plane needs an observable read primitive, not only information
   capacity.**  The primitive's physical cost and matched-control exposure are
   part of the idea, not later implementation details.

The next candidate must therefore begin with a read operator that has a proved
resource separation, then ask whether raw prose can compile into its typed
state with one isolated empirical uncertainty.
