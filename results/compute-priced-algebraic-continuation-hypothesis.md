# Compute-priced algebraic continuation (CPAC)

Status: pre-candidate; oracle evidence required before optimizer work  
Date: 2026-07-30

## The proposed training method

For every dense Transformer map `W`, train a materialized sum of fast matrix
atoms rather than committing to dense algebra from step zero:

`W_t = W_0 + A_1 + ... + A_k`.

An atom may be low-rank, block-sparse, Kronecker-factored, or eventually dense.
Every new atom is introduced at zero scale, so adding capacity initially leaves
the model function exactly unchanged. Periodically, an exact dense gradient is
measured and the next atom is selected by the amount of residual gradient it
captures per charged multiply-like operation. When no structured atom is
efficient enough, the layer promotes to an ordinary dense matrix. At export,
all atoms are summed/materialized into the ordinary dense weight. The served
architecture, parameters, matmuls, and numerical function are unchanged.

This is not an auxiliary objective, a teacher, a data curriculum, or a claim
that one fixed structure replaces language-model matrices. Dense algebra is a
per-layer fallback inside the method.

## Local mathematical criterion

Let `G = grad_W L` and let `A` be an available update direction. If the loss is
locally beta-smooth, then

`L(W - eta A) <= L(W) - eta <G,A> + beta eta^2 ||A||_F^2 / 2`.

Optimizing this bound over `eta` gives guaranteed predicted descent

`D(A) = <G,A>^2 / (2 beta ||A||_F^2)`.

The dense steepest direction gives `D(G) = ||G||_F^2 / (2 beta)`. Therefore
the fraction of dense local descent retained by `A` is exactly

`q(A) = <G,A>^2 / (||G||_F^2 ||A||_F^2) = cos^2(G,A)`.

If the structured training step costs fraction `c(A)` of the dense step, its
optimistic descent per compute is `q(A)/c(A)`. This supplies an oracle upper
screen before any optimizer can hide the absence of exploitable structure.

For an `m x n` map:

- rank `r`: ideal apply-cost ratio `r(m+n)/(mn)`;
- block-sparse support with `k` scalar coordinates: `k/(mn)`;
- one Kronecker atom `P (m1 x n1) kron Q (m2 x n2)`, where
  `m=m1*m2` and `n=n1*n2`: ideal ratio `1/m1 + 1/n2`;
- sums pay the sum of their component costs.

Factor discovery, dense-gradient refreshes, fixed attention work, logits,
normalization, memory traffic, and kernel inefficiency must be added later.
Consequently an oracle advantage near 1.5x is insufficient.

## Why this is not the already-known branch

- Progressive width/depth growth changes the whole model on a schedule.
- InRank grows low-rank cumulative updates only.
- Sparse-plus-low-rank pretraining fixes the structure or support policy.
- Structured optimizers precondition/project updates but normally execute the
  ordinary dense forward maps.

CPAC's only potentially distinct claim is **online selection among different
algebras using exact gradient energy per measured compute, with per-map dense
promotion and dense export**. If the oracle chooses low-rank almost everywhere,
the proposal reduces to prior art and is closed as non-distinct.

## Large-gain ceiling

At width `d=4096`, 32 layers, context 4096, FFN ratio 8/3, GQA KV ratio 1/4,
and vocabulary 49,152, internal learned projections account for roughly 82%
of forward multiply-like work under a simple ledger. If those projections
retain dense-quality descent at 25% of their dense cost, the whole-step ideal
ratio is approximately

`0.18 + 0.82*0.25 = 0.385`,

or a 2.60x ceiling before refresh and systems overhead. This is large enough
to survive a 1.5x final gate. A method that leaves the dense forward untouched
does not have this ceiling and is out of scope.

## Stage 0 fatal oracle

Measure exact from-zero LM gradients at widths 192, 384, and 768, at
initialization, early learning, and a later checkpoint. Sample Q/K/V/O and all
three FFN matrices from early, middle, and late layers. For every matrix:

1. compute the best truncated-SVD direction;
2. compute the best 16x16 block-sparse direction;
3. compute balanced Kronecker directions;
4. compute low-rank plus block-sparse residual directions;
5. select the least-cost direction reaching `cos^2 >= 0.90`, otherwise use
   dense fallback.

Also run size-matched isotropic random-gradient controls.

Advance to an executable optimizer only if all are true:

1. aggregate structured projection cost is at most 35% of dense matrix cost in
   at least seven of nine width/checkpoint cells while retaining at least 90%
   local descent;
2. mixed algebra costs at least 25% less than low-rank-only in at least six
   cells, proving a distinction from InRank;
3. the required structured fraction decreases with width at every checkpoint,
   satisfying the retained dense-exception scaling direction;
4. isotropic controls require at least 85% of dense cost;
5. after adding the frontier fixed-work ledger and one dense refresh per 64
   steps, the projected whole-step efficiency is at least 2.0x in the widest
   cell.

This remains only an optimistic admission test. A pass does not establish
trainability, kernel speed, final NLL, or novelty.

## Closure rule

If the oracle fails, do not tune atom counts, add another fixed structured
family, or claim memory savings. Retain only the measured gradient-spectrum
fact and return to a different training-compute lever.
