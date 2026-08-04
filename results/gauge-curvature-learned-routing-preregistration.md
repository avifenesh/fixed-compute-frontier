# Gauge-curvature learned-routing gate — preregistration

Status: **frozen before observing any trained result**  
Date: 2026-07-26  
Stage: two-layer learned-Q/K mechanism gate on one rented H100

## 1. Question and permitted conclusion

Can a gauge-funded, cache-neutral quadratic value writer learn an addressed
second-order statistic better than a standard Transformer and stronger
equal-resource nonlinear controls?

This is a kill gate, not an LLM-quality claim. A positive result only advances
one writer to a small language-model gate. A negative result rejects the writer
in its most favorable mechanism regime. No result here admits a model
candidate.

## 2. Frozen data

One scene has four bags and four queries. Every bag contains eight independent
Gaussian draws and their negatives, for 16 source tokens; the 64 sources are
globally shuffled. All four queries follow the sources in random order. There
are no positional embeddings.

Every scene has exactly two positive and two negative bags. Each scene also
gets four fresh orthonormal address vectors in `R^16`; a bag and its query share
one vector. Fresh addresses prevent ID memorization, while the fixed 2/2 class
count makes the unordered union of bags label-independent for any requested
address.

For world `w`, with a fixed Haar rotation `R_w`, content dimension `r=16`, and
`s=(1^8,-1^8)`, the primary covariance task is

\[
\Sigma_y=R_w\,\mathrm{diag}(1+0.15ys)R_w^\top,
\qquad y\in\{-1,+1\}.
\]

The two classes have equal trace, determinant, zero mean, and norm
distribution. Emitting `z,-z` makes every bag's raw linear content mean exactly
zero. Five independently rotated worlds use seeds
`[17011, 17027, 17041, 17053, 17077]`.

Each world has 8,192 training, 1,024 validation, and 2,048 test scenes. Splits
are independently generated. Each scene returns all four labels. The input
layout in `D=96` is fixed:

- dimensions `0:16`: raw content;
- dimensions `16:32`: address, scaled by 4;
- dimensions `32:34`: fixed source/query type code;
- dimensions `34:96`: zero at input.

The address, shuffle, and query order are independent of labels. A held-out
mean task, run only after a screen pass, replaces each pair by
`y*mu + epsilon, y*mu - epsilon`; its bag mean is exactly linear. An unpaired
covariance task is also confirmatory, not part of the screen.

## 3. Frozen model

All matrix projections are bias-free (the matched low-rank G2 control has one
explicitly ledgered 48-value gate offset). Models are two-layer, pre-RMSNorm
causal Transformers with:

- model width `D=96`;
- six query heads, three KV heads, head/key width `r=16`;
- learned dense Q and K projections over the full hidden state;
- ordinary residual SwiGLU blocks;
- a final RMSNorm and shared scalar classifier at the four query positions;
- no dropout, RoPE, learned input projection, or positional embedding.

Thus attention weights and the first layer's ordinary source MLP remain strong
alternative ways to compute nonlinear statistics. The test does not isolate
the candidate from standard-Transformer escape paths.

Training includes a common routing loss on final-layer query attention:

\[
L=L_{BCE}+0.05\left[-\log\sum_{i\in\mathrm{requested\ bag}}\alpha_{qi}\right].
\]

Q/K are still learned, not fixed or oracle. The auxiliary term only exposes
failure to learn the public address relation. If a claimed quality win has
more than a two-percentage-point target-bag-mass advantage over its comparator,
it must be repeated with the learned router frozen; otherwise it is rejected as
a routing win.

## 4. Writers and resource ledger

Let `z` be the first 16 normalized hidden coordinates and
`u=z+x_rest B`, independently per KV head. All masks store only their live
entries. The base SwiGLU width is 256.

| variant | value rule | attention params/layer | FFN width | attention+FFN params over 2 layers | cached K+V scalars/token/layer |
|---|---|---:|---:|---:|---:|
| `dense` | `x Wv` | 27,648 | 256 | 202,752 | 96 |
| `bda` | `u` | 26,880 | 256 | 201,216 | 96 |
| `cyclic` | `x_rest B + z*(alpha + z H)`, one cyclic zero/column | 27,648 | 256 | 202,752 | 96 |
| `square` | `u + (z H)^2`, diagonal omitted | 27,600 | 256 | 202,656 | 96 |
| `source_mlp` | `x_rest B + alpha*z + SiLU(x A) C`, rank 5 | 27,648 | 256 | 202,752 | 96 |
| `g2_matched` | `u*sigmoid((x A) C + b)`, rank 5 | 27,648 | 256 | 202,752 | 96 |
| `g2_full` | `(x Wv)*sigmoid(x Wg)` | 32,256 | 240 | 202,752 | 96 |
| `glu` | `(x Wa)*SiLU(x Wg)`, value width 12 | 27,648 | 256 | 202,752 | 84 |
| `g1` | elementwise query gate after SDPA | 36,864 | 224 | 202,752 | 96 |

Q/K/O are included in the attention counts. RMSNorm and classifier parameters
are common and omitted from the table. For cyclic, the reclaimed `3*16^2=768`
V-gauge pool is spent on 720 live `H` entries and 48 learned `alpha` values.
Its V path therefore has exactly the dense V parameter, multiplication, and
addition ledger. This is a **raw ledger**, not a claim of 768 new functional
degrees: the 48 `alpha` directions remain removable by per-channel V/O scaling,
so the cyclic writer adds at most 720 generic nonlinear quotient directions.
`source_mlp` spends the same 768 raw values on
`A:96->5`, `C:5->48`, and 48 learned pivot coefficients. Unlike a redundant
post-`C` output scale, those coefficients separately control the linear pivot
path, but the complete value/output map still has channel and low-rank
factorization gauges. No feature-density conclusion will use raw counts as
quotient dimensions. `g2_full` and `g1` pay for their
extra projections by shrinking SwiGLU, so their two-layer matrix-parameter and
matrix-MAC totals match dense. `glu` exactly preserves the dense V/O parameter
ledger while reducing value-cache width from 48 to 36.

Parameter equality does not pretend sigmoid and SiLU are free. Activation and
explicit elementwise operations, executed bytes, peak memory, examples/second,
and wall time are reported separately.

## 5. Initialization and training

Dense, BDA, cyclic, source-MLP, and matched-G2 begin from the same
well-conditioned dense V/O function through an exact per-head V/O gauge
transform. Cyclic starts at `alpha=1,H=0`; source-MLP uses
`alpha=1`, random `A`, and `C=0`;
matched G2 uses zero gate weights and a doubled initial output projection, so
its initial `sigmoid(0)=1/2` function still matches dense without charging an
extra factor-two multiply. Square is only **near-matched**: it uses a small
nonzero `H` because its Jacobian is zero at `H=0`, so `(zH)^2` makes its initial
function slightly different. Common Q/K, norms, classifier, and overlapping
MLP weights are identical for a given initialization seed.

The primary screen uses initialization seed `314159`. Confirmation seeds are
`271828` and `161803`. Every variant in a world sees identical examples and
minibatch indices.

- AdamW, betas `(0.9,0.95)`, weight decay `0.01`;
- batch size 128;
- 1,500 fixed steps, no per-variant early stopping;
- learning rate `2e-3`, 100-step linear warmup, cosine decay to `2e-4`;
- gradient norm clipping at 1.0;
- BF16 autocast with FP32 parameters and FP32 loss;
- validation every 150 steps for diagnostics; the final fixed-step checkpoint
  supplies test metrics.

A sixth, excluded debug world may run for at most 30 steps to verify shapes,
loss movement, determinism, gauge equivalence, and result persistence. Its
metrics cannot alter the frozen primary protocol.

## 6. Measurements and validity

Primary metric: held-out binary NLL over all four queries. Also report accuracy,
validation NLL versus step, gradient failures, target-bag attention mass,
bag-argmax routing accuracy, exact trainable parameters, logical matrix MACs,
explicit nonlinear operations, cache elements, peak allocated GPU memory,
examples/second, and wall time.

The analytic Gaussian likelihood supplies the finite-test Bayes NLL and
accuracy. Because every scene has exactly two positive bags, the oracle
enumerates all six valid 2-of-4 assignments, normalizes their joint
likelihoods, and reports each bag's posterior marginal; four independent
per-bag likelihood ratios are not the Bayes answer for this scene prior. The
harness is invalid if any of these holds:

- Bayes accuracy is below 85%;
- the best learned model is below 75% test accuracy;
- its mean target-bag mass is below 90% or bag-argmax routing is below 99%;
- an address-only, uniform-all-bags, or broken-address control exceeds 52%;
- paired content means differ from zero by more than `1e-6` before embedding;
- declared and actual parameter ledgers disagree.

## 7. Sequential rejection rule

After the one-seed/five-world screen, define the comparator as the eligible
equal-or-cheaper control with the lowest median NLL. Stop and reject cyclic
unless it beats that fixed comparator in at least four worlds and has positive
median relative excess-NLL improvement, where excess is measured above the
finite-test Bayes NLL.

Only a screen pass unlocks the other two seeds, the unpaired covariance task,
the exact-mean task, and the frozen-router diagnostic when required. Final
promotion requires all of:

1. cyclic wins at least 12 of 15 paired world/seed runs and at least four of
   five world means;
2. it closes at least 5% of the comparator-to-Bayes NLL gap;
3. a hierarchical paired 95% bootstrap interval over worlds then seeds excludes
   zero improvement;
4. mean-task accuracy falls by at most 0.5 percentage point and mean-task NLL
   rises by at most 1% versus dense;
5. the gain is not explained by routing and survives `source_mlp`, the ordinary
   source SwiGLU, canonical G2, paper-style GLU, and G1.

A tie, a win only against dense, a route-quality win, or a benefit that costs
more executed memory/compute than declared is rejection. If a cheaper control
dominates instead, it may be retained as a separate hypothesis, but it receives
no architecture claim from this synthetic task.

## 8. Source boundary

The control formulas follow the current primary descriptions in
[Gated Attention](https://arxiv.org/abs/2505.06708),
[GLU Attention](https://arxiv.org/abs/2507.00022), and
[Block-Diagonal Attention](https://arxiv.org/abs/2510.01718). The cyclic
composition and its decision thresholds are this experiment's hypotheses, not
claims attributed to those papers.
