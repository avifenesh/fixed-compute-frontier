# Staggered Power-Evidence Attention — Stage 0 preregistration

Status: frozen before execution  
Date: 2026-07-27

## Claim boundary

Can a statically selected attention head route sharply between neighborhoods
while reading softly inside the selected neighborhood, with exact neutrality
to IID chunk cardinality and without another QK, PV, QKV, or FFN matmul?

For semantic chunk `b`, define

\[
Z_{1,b}=\sum_j e^{s_j},\quad
Z_{p,b}=\sum_j e^{p s_j},\quad
\mu_b=\frac{\sum_j e^{s_j}v_j}{Z_{1,b}},
\]

and

\[
y_p=\frac{\sum_b Z_{p,b}\mu_b}{\sum_b Z_{p,b}}.
\]

`p=1` is exactly ordinary attention for every partition. `p=2` is the
power-evidence head: router temperature two, local reader temperature one.
Global-temperature-two is the matched control and uses temperature two for
both router and local reader.

This Stage 0 establishes algebra, a score-locked capability witness, and the
boundary-staggering repair. It does not establish learned-language quality or
GPU noninferiority.

## Frozen mechanisms

- Sequence length 128, chunk width 16.
- Ordinary and global-temperature-two controls.
- PEA2, raw peak with alpha one, and confidence lift with alpha one.
- Boundary staggering averages two PEA2 head outputs with deterministic chunk
  origins 0 and 8. At `p=1`, both partitions remain exactly ordinary.
- A deployable architecture uses a static per-head `p in {1,2}` bitmap. The
  architecture-search family contains the exact all-`p=1` baseline; a forced
  nonempty `p=2` subset does not. The bitmap is not claimed free until a later
  gauge-codec/export gate.

## Frozen algebra gates

1. Direct PEA2 weights and the two-level state construction agree within
   `1e-12` over random unequal chunks.
2. PEA1 agrees with ordinary attention within `1e-12` for origins 0 and 8 and
   unequal chunk sizes.
3. PEA2 preserves ordinary within-chunk conditional ratios within `1e-12`;
   global temperature two changes them by at least `1e-3`.
4. Changing a third score changes a cross-chunk PEA2 odds ratio by at least
   `0.1`, while ordinary pairwise IIA changes by at most `1e-12`.
5. Shifted `l2` execution remains finite and agrees with direct execution
   within `1e-12` under row shifts through `+/-500`.
6. The missing-statistic witness has identical maximum and `Z1` but unequal
   `Z2` for exponentiated score sets `(1,0.6,0.2)` and `(1,0.5,0.3)`.

## Exact IID cardinality gate

For several fixed eight-score multisets, exhaust all `8!` permutations into
unequal chunks of sizes `(1,3,4)`. Exact permutation averaging must give each
chunk mean PEA2 router share `n/8` and every token position mean final weight
`1/8`, each within `1e-12`. This is deterministic, not a Monte-Carlo estimate.

## Score-locked anchor-to-payload gate

All 128 scores are zero except a value-zero anchor with score
`Delta in {0.5,1,2,3,4}`. The desired payload score remains zero. Record target
payload probability and exact sign-prediction accuracy over all `2^8`
assignments of eight independent `+/-1` payload labels.
Prediction is the sign of the scalar attention output. Outputs with absolute
value at most `1e-14` receive half credit, equal to expected accuracy under an
independent fair random tie-break. This tie rule and tolerance are frozen;
ties occur materially for controls with equal payload coefficients.

Conditions:

1. anchor and target payload share an origin-0 chunk;
2. anchor is position 31 and payload position 32, deliberately split by origin
   0 but co-chunked by origin 8;
3. the arithmetic mean of origin-0 and origin-8 PEA2 head outputs.

Fatal capability gates:

- same-chunk PEA2 target probability exceeds ordinary and global temperature
  at every Delta, with a minimum ordinary-relative ratio of at least 1.05;
- staggered PEA2 target probability is never below ordinary, with minimum
  ratio at least 1.01;
- same-chunk PEA2 exact label accuracy exceeds global temperature at every
  Delta;
- staggered PEA2 exact label accuracy exceeds ordinary at every Delta;
- the earlier peak/confidence causal result remains a mandatory Pareto
  control: PEA2 must pass exact IID neutrality, which those arms failed. The
  exact bound artifact is `results/peak-lifted-attention-causal-control.json`,
  SHA-256 `8091612564cc9e2e373e0d02021c7005864284becc7adf7adf95ac59a9238b78`;
  all six prior gates must remain false and `live_variant` must remain null.

## Resource boundary

For a `p=2` head, shifted score weight `w=exp(s-m)` is already present. PEA2
adds `l2=sum w*w`, one score multiply/accumulation, one `l2/l1` scalar division
per semantic chunk, and one width-`d_v` rescale of the local value numerator;
it adds no second per-score exponential, QK/PV/QKV/FFN matmul, KV field, or
persistent cache. When a semantic chunk is a native PV unit, the outer merge
can replace the ordinary merge schedule without an additional scalar merge
exponential; unaligned chunks may require extra merge work and are not credited
with that property. This logical ledger does **not** claim only one extra live
scalar in monolithic prefill: unless semantic chunks align with native PV
units, a current-chunk value accumulator may coexist with the outer output
accumulator. That is a later fatal kernel gate.

Passing Stage 0 admits only a tiny learned structured task. It does not admit
an LM run or FlashAttention patch.
