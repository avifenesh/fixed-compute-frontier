# Projection-coalesced attention/FFN — Stage 0 preregistration

## Question

Can one learned pair of expanded feature projections do both token selection
and token-local feature construction, so that the separate Q/K/V/O matrices
can be reinvested into wider nonlinear features without increasing served
cost?

This branch is admitted only because it changes which information the dense
projections preserve and exchange.  It is not another activation-only
superset of SwiGLU.

## Frozen algebra

For normalized hidden states `X in R^(T x D)`, compute only

```
G = X Wg^T,  U = X Wu^T                  (T x M')
Q = G[..., 0:D]
K = U[..., D:D+Kdim]
V = U[..., D+Kdim:D+2*Kdim]
A = causal_attention(Q, K, V)             (T x D)
Z = SiLU(G) * U                           (T x M')
Z[..., 0:D] += A
Y = Z Wdown^T                             (T x D)
```

`Kdim` is the total non-repeated GQA key/value width.  RoPE, Q/K
normalization, causal masking, and grouped repetition are unchanged from the
baseline attention core.  The attention output is scattered into the first
`D` expanded channels, so the first `D` columns of `Wdown` are simultaneously
the attention output projection and SwiGLU value vectors.

The block is parallel: attention and the local feature branch see the same
pre-normalized residual state and their joint output is added through one
residual connection.

## Frozen width and resource ledger

Baseline dense projection parameters/MACs per token:

```
P_base = 3 D M + 2 D^2 + 2 D Kdim
```

The input-side baseline projection emits

```
2 M + D + 2 Kdim
```

channels for gate, up, Q, K, and V.  Freeze the coalesced width to match that
input GEMM output exactly:

```
M' = M + (D + 2 Kdim) / 2
P_candidate = 3 D M'
P_base - P_candidate = D (D/2 - Kdim)
```

For the scratch LM (`D=384`, `M=1024`, six query heads, two KV heads,
head size 64, hence `Kdim=128`):

| quantity | baseline | candidate |
|---|---:|---:|
| combined input rows | 2,688 | 2,688 |
| output-side input columns | 1,408 (`M+D`) | 1,344 |
| dense weights/MACs per token | 1,572,864 | 1,548,288 |
| ratio | 1.0 | 0.984375 |
| nonlinear feature width | 1,024 | 1,344 |
| KV cache elements/token/layer | 256 | 256 |

The candidate has no router, expert indices, recurrent state, extra learned
tensor, or extra workspace in the abstract graph.  Elementwise work grows by
320 SiLU/multiply pairs per token and adds a scatter-add of 384 values.  These
operations must be charged in the H100 gate.

At a representative `D=4096, M=11008, Kdim=1024` GQA shape, the same rule
gives `M'=14080`: the combined input projection remains 28,160 rows, the
output-side width falls from 15,104 to 14,080, and nonlinear width grows by
27.9%.  This is an analytic projection ledger, not a latency claim.

## Expressivity statement and floor

This candidate does **not** contain a sequential Transformer block and must
earn back both parallelization and weight-tying constraints.

It does contain arbitrary parallel GQA plus an independent SwiGLU of width

```
M_independent = M' - D - 2 Kdim.
```

Construction: reserve `G[0:D]` for Q and set its paired U rows to zero;
reserve the next `Kdim` U rows for K and the next `Kdim` U rows for V while
setting their paired G rows to zero; use `Wdown[:,0:D]` as arbitrary `Wo`;
use all remaining rows as an ordinary independent SwiGLU.  At the scratch
shape this guaranteed independent width is 704.  Training is useful only if
the attention-bearing rows also become useful local features rather than
behaving as this dedicated construction.

## Prior-art boundary checked before execution

- Parallel attention/FFN blocks are established, including PaLM.
- Sharing Q/K/V projections is an active, separately populated direction.
- FFNs as parameter attention and multi-head FFNs are established.
- Fusing kernels or concatenating independent projection matrices does not
  change the algebra or the ledger.

The distinct hypothesis tested here is cross-sublayer **activation reuse**:
Q/K/V are slices of the same full-strength banks used by every SwiGLU feature,
and `Wo` is a slice of the same down matrix.  No novelty claim is made until a
broader collision search and a positive causal result both exist.

## Stage 0 gates

All must pass before LM training:

1. Shapes, causal masking, GQA repetition, gradients, and fused-down equality
   pass deterministic tests.
2. The integer parameter/MAC ledger above is reproduced by code.
3. The arbitrary-attention plus independent-width-704 construction matches a
   separately evaluated reference numerically.
4. No future token changes an earlier output.

## Frozen matched LM screen

If Stage 0 passes, train from scratch on the already frozen 50M-token stream:

1. `sequential_baseline`: ordinary serial Llama attention then SwiGLU,
   `M=1024`.
2. `parallel_baseline`: independent Q/K/V/O and SwiGLU, `M=1024`, but one
   shared pre-norm and parallel residual branches.
3. `coalesced_1024`: the frozen algebra without reinvesting the saved
   projection budget.
4. `coalesced_1344`: the frozen input-shape-matched candidate.

Same tokenizer, data order, initialization scale, optimizer, schedule, token
count, attention core, RoPE, head counts, precision, and seed.  Evaluate at
10M and 50M prediction tokens.

Promotion requires all of:

- `coalesced_1344` terminal validation NLL at least 0.05% below
  `sequential_baseline`, with the paired 95% interval wholly below zero;
- terminal NLL at least 0.025% below both `parallel_baseline` and
  `coalesced_1024`;
- early NLL noninferior to `sequential_baseline` within 0.05%;
- finite gradients/activations and no more than 1% of sampled activations
  outside four times the baseline RMS envelope;
- after training, zeroing local products on the `D+2*Kdim` attention-bearing
  channels hurts NLL by at least 0.01%, showing that dual use—not merely the
  704 independent channels—carries capability.

This screen may reject.  It cannot establish A-E dominance.

## Fatal interpretation

- If `parallel_baseline` loses badly, this formulation confounds projection
  reuse with lost within-layer serial depth; close it rather than tuning the
  sharing map.
- If `coalesced_1344` matches `parallel_baseline` but not the sequential
  baseline, projection reuse is viable but not a better model.
- If the dual-use ablation is harmless, the extra rows are acting as dedicated
  attention channels and the claimed reuse mechanism failed.
- No rescue by more layers, learned routing, distillation, or topology search
  is allowed under this branch.
