# T32R rank-32 boundary scan — frozen algebra

Status: **PAPER PASS TO CPU REFERENCE ONLY**  
Date: 2026-07-31

## Operator

The tokenizer emits two typed document handles in occurrence order. The normal
residual stream receives one shared learned handle marker; the handle indices
address the fixed T32 record table outside the residual stream.

After block 3, let `h in R^384` be the assistant-boundary hidden state. For
each document `j in {A,B}`, unpack its exact IDs `t[j,0:128]` and gather the
ordinary shared input embeddings `E[t]`. With one shared
`P in R^(384 x 32)`, define

\[
q=P^Th,\qquad u_{j,i}=P^TE[t_{j,i}]+p_i,
\]

where `p_i` is a fixed sinusoidal position code.

Apply a learned depthwise three-tap filter

\[
c_{j,i}=a_-\odot u_{j,i-1}+a_0\odot u_{j,i}+a_+\odot u_{j,i+1},
\]

with zero boundary padding. Split 32 channels into four heads of width eight.
For learned diagonal scales `d_q,d_k,d_v`, each head computes ordinary
single-query attention over its document:

\[
z_{j,i}=(d_q\odot q)^T(d_k\odot c_{j,i})/\sqrt8,
\quad
\alpha_{j}=softmax(z_j),
\quad
s_j=\sum_i\alpha_{j,i}(d_v\odot c_{j,i}).
\]

Preserve document roles with full `R_A,R_B in R^(32 x 32)`, then inject

\[
h'=h+\sigma(g^Tq+b)O(R_As_A+R_Bs_B),
\]

where `O in R^(384 x 32)`. Blocks 4--10 process `h'` normally. The scan runs
once during prefill; later decode tokens reuse the enriched boundary KV.

No scan runs without exactly two valid typed handles.

## Exact boundary

Before projection, the expanded sequence equals the embeddings of an explicit
128-token context position by position:

\[
E[U(Pack(t))_i]=E[t_i].
\]

The only lossy map is the declared shared rank-32 projection and subsequent
bounded attention. There is no learned document encoder and no stronger
compiler.

## Attention approximation bound

For one head over `n=128` positions, suppose the desired position has score at
least `Delta` above every other position. Then

\[
\alpha_*\ge\frac1{1+(n-1)e^{-\Delta}}.
\]

If every value has norm at most `M`, the selected-value error is bounded by

\[
\left\|\sum_i\alpha_iv_i-v_*\right\|
\le2M(n-1)e^{-\Delta}.
\]

Thus the finite softmax can approximate a selected local feature to error
`epsilon` whenever

\[
\Delta\ge\log(2M(n-1)/\epsilon).
\]

This does not prove that natural training creates the margin. The CPU reference
must verify the bound on controlled vectors; later training must measure
natural margins.

## Role separation

The role path can distinguish document order. An explicit witness is
`R_A=I,R_B=-I`: the pre-output summary is `s_A-s_B`, and swapping the records
negates it. Learned full role maps strictly contain this witness. This is why
the two summaries are not simply averaged.

## Rank limitation

`P^T:R^384 -> R^32` has a kernel of dimension at least 352. Therefore the scan
cannot preserve arbitrary token-embedding distinctions. Natural adequacy at
rank 32 is the sole empirical representation hypothesis. A pass on exact T32
storage cannot be transferred to this projection.

## Exact resource ledger

| component | entries |
|---|---:|
| fixed record digits | 916,305 |
| shared projection `P` | 12,288 |
| diagonal Q/K/V | 96 |
| depthwise three-tap filter | 96 |
| two role maps | 2,048 |
| summary output `O` | 12,288 |
| scalar gate | 33 |
| shared handle marker | 384 |
| total | **943,538** |
| entries removed by width 1024 -> 940 | **967,680** |
| matched slack | **24,142** |

## Exact multiplication ledger

Per request with two full records:

| operation | multiplies |
|---|---:|
| two embedding projections | 3,145,728 |
| boundary-query projection | 12,288 |
| depthwise context | 24,576 |
| diagonal Q/K/V | 16,416 |
| score and weighted-value products | 16,384 |
| role maps | 2,048 |
| summary output | 12,288 |
| gate dot and gated residual | 416 |
| total | **3,230,144** |

This is below the frozen 3.7M scan envelope. It reads 196,608 bytes of BF16
token embeddings plus 1,524 bytes of record digits, before cache effects and
intermediate traffic. FLOPs do not predict latency for these small operations;
the H100 gate remains separate.

## Training boundary

The record table is deterministic and fixed. Gradients train `P`, the scan,
role maps, gate, output, shared normal embeddings, and backbone. The scan is
applied only at an explicit assistant-generation boundary, so it cannot leak a
future assistant target into an earlier causal position.

Raw-document reconstruction may train the generic reader only if the target
token is excluded from the query input and the matched controls receive the
same derived examples and training work. Natural QA labels are not admitted at
the primitive stage.

## CPU admission

Implement the exact equations with deterministic NumPy and an independent
reference. Require shape/determinism, no-handle identity, role-swap separation,
the softmax concentration bound, exact accounting, finite gradients by analytic
formula where applicable, and zero GPU/model/data access. Passing admits only
a frozen target-H100 block microbenchmark and a separate learnability preflight.
