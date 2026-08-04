# Dense-exception scaling law — refinement of the key shift

Status: **theorem and deterministic gate complete; no candidate admitted**  
Date: 2026-07-26  
GPU: **not used or justified**

## Result

“Cheap structured bulk plus a capped dense exception” is not yet an
architecture.  It becomes a genuine break from dense scaling only when the
exception becomes proportionally smaller as model width grows.

For width `D`, ideal structured-bulk cost `C_bulk(D)`, and a rank-`r(D)` dense
correction, the logical projection work is

\[
C(D)=C_{bulk}(D)+2Dr(D).
\]

If `C_bulk(D)=O(D log D)`, then:

- `r(D)=Theta(D)` restores `Theta(D^2)` work;
- `r(D)=D^alpha`, `0<alpha<1`, gives `Theta(D^(1+alpha))` work;
- `r(D)=O(log D)` preserves `O(D log D)` work.

The rank cap at one width is therefore not the result.  The result is the
**exception exponent** across widths.

The same rule applies to other exception geometries:

| exception geometry | logical exceptional work | condition to break quadratic scaling |
|---|---:|---|
| feature rank `r(D)` | `2 D r(D)` | `r(D)=o(D)` |
| local dense block width `b(D)` | `D b(D)` before cross-block work | `b(D)=o(D)` |
| full-dense token fallback rate `p(D)` | `p(D)D^2` | `p(D)=o(1)` |
| full-dense layer fraction `q(D)` | `q(D)D^2` | `q(D)=o(1)` |

To retain near-linear `O(D log D)` work, token or layer fallback must be only
`O(log D / D)`.  At `D=4096` this is `0.00293`, under 0.3%.  A constant 5% or
25% dense fallback may improve the coefficient but does not change the scaling
law.

If the complete dense fallback weights remain resident, static learned bytes
remain `Theta(D^2)` even when execution is skipped.  That can at most support a
speed edge, not a smaller-model or denser-knowledge edge.

## Isotropic exception boundary

Let the target contain independently normalized structured and exception
components with weights `a=0.75` and `b=0.25`.  If the exception is isotropic,
for example `bQ` with Haar-orthogonal `Q`, all `D` exception singular values are
equal.  The best rank-`r` correction leaves relative total error

\[
\epsilon_r=sqrt{
\frac{b^2(1-r/D)}{a^2+b^2}
}.
\]

At `D=4096`:

| allowed total relative error | required exception rank | rank fraction | ideal `D log D + 2Dr` / dense work |
|---:|---:|---:|---:|
| 10% | 3,687 | 90.01% | 1.80x |
| 5% | 3,994 | 97.51% | 1.95x |
| 1% | 4,092 | 99.90% | 2.00x |

The correction becomes more expensive than the dense projection before it is
accurate.  A “small” 25% isotropic component is not a small exception in rank.
This explains the previous feature-DAG mixed result: a 25% Haar component
destroyed the structured advantage because it occupied almost every direction.

## Spectral phase transition

Suppose exception singular values follow

\[
\sigma_i\propto i^{-s}.
\]

Their squared-energy tail is `sum i^(-2s)`.  The critical point is `s=1/2`:

- `s<1/2`: tail energy grows polynomially with width; the needed exception
  remains proportional to `D`;
- `s=1/2`: energy grows logarithmically; for allowed exception-tail fraction
  `delta`, the required rank scales as roughly `D^(1-delta)`, technically
  sublinear but still nearly dense for a tight error budget;
- `s>1/2`: total exception energy is summable; for fixed error, required rank
  becomes sublinear and can approach a width-independent cap.

The deterministic finite-width gate confirms the transition for 5% total
error:

| spectral exponent `s` | rank at `D=256` | rank at `D=4096` | fraction at `D=4096` | ideal work ratio at `D=4096` |
|---:|---:|---:|---:|---:|
| 0.00 | 250 | 3,994 | 97.51% | 1.953x |
| 0.50 | 220 | 3,280 | 80.08% | 1.604x |
| 0.75 | 114 | 436 | 10.64% | 0.216x |
| 1.00 | 22 | 24 | 0.59% | 0.0146x |
| 1.50 | 4 | 4 | 0.10% | 0.0049x |

The useful world is not “the dense exception has small amplitude.”  It is:

> The protected, activation-weighted exception spectrum has a stable decay
> exponent strictly above one half, so fixed-error rank can approach a bounded
> cap, and that decay survives scale.

Spectral energy alone is not a quality test.  The Qwen recurrent-state result
already showed that better Frobenius reconstruction can worsen causal KL.
Future measurement must use protected causal loss while estimating the smallest
exception needed at each width.

## Generic capacity boundary

The space of real `D x D` matrices has dimension `D^2`.  A normal smooth
parameterization with fewer than `D^2` total continuous scalars cannot cover an
open set of generic matrices, regardless of whether its factors are called
diagonal, low-rank, hierarchical, routed, or deep.  Depth can reconstruct dense
maps by composing cheap factors, but generic coverage eventually pays the
missing degrees through total factors, parameters, work, or serial depth.

This does not prevent a language-model win.  Language is structured rather
than a random matrix distribution.  It means the empirical claim must be that
the required exception exponent shrinks on language while randomized or
Haar-like controls expose the boundary.

## Collision boundary

The broad key shift is established prior art:

- [DLoR](https://arxiv.org/abs/2605.05659) proves diagonal-plus-low-rank networks
  recover universal approximation through width or depth;
- [FOSL](https://openreview.net/forum?id=qUephWWa2B) combines sparse/folded bulk
  with a low-rank path in LLM projections;
- [structured FFN training](https://openreview.net/forum?id=WxLVYZbIew),
  [BLAST](https://openreview.net/forum?id=n0arS0DDot), and
  [structured-matrix search](https://openreview.net/forum?id=fc88ANWvdF)
  already learn block, low-rank, and broader efficient linear operators;
- [SLA](https://arxiv.org/abs/2509.24006) and
  [ELSAA](https://arxiv.org/abs/2607.20214) use cheap low-rank/global bulk plus
  sparse exact exceptions in attention.

Therefore “structured bulk plus dense residual” is not a novelty claim and is
not worth duplicating on a rented GPU.  The research contribution available to
this ledger is the stricter admission law below.

## New admission law

A capped-exception candidate is eligible for GPU training only when it declares
one exception geometry and freezes measurements at three or more widths.
Promotion requires all of:

1. protected quality is noninferior at every width;
2. the measured exception fraction decreases with width rather than merely
   fitting under a fixed cap once;
3. the fitted exception exponent makes total issued work subquadratic;
4. resident weights, selectors, metadata, state, and physical kernels fit the
   claimed A/B/C/D/E resource edge;
5. a randomized or isotropic control forces the exception back toward the
   dense boundary.

No current candidate supplies this evidence, so no H100 is justified.  The
next creative step must specify **why language training should produce an
exception spectrum with exponent above one half**.  Without that causal reason,
choosing another structured matrix is mechanism search inside an already
occupied family.

## Artifacts

- `experiments/dense_exception_scaling_law.py`
- `tests/test_dense_exception_scaling_law.py`
- `results/dense-exception-scaling-law.json`
