# Matmul escape search contract

Status: **INDEPENDENTLY AUDITED PAPER SEARCH BOUNDARY; NO CANDIDATE OR RUN ADMITTED**  
Date: 2026-07-31

## Question

How can a production language model replace the dense matrix-vector products
in Q/K/V/O and the FFN without merely paying for the same generic map in a
different notation?

The answer cannot start from a kernel.  It starts from the mathematical reason
the map is allowed to be cheaper.

## Exact universality boundary

The vector space of real `m x n` linear maps has dimension `mn`.  Consider a
fixed arithmetic architecture with `s` continuous learned scalars whose output
is linear in `x`.  Assume its parameter-to-matrix map is `C^1`, finite-piece
locally Lipschitz, or semialgebraic, as in ordinary finite neural circuits.  We
explicitly exclude space-filling maps and infinite-precision encodings.

If `s < mn`, the image of this regular parameterization has empty interior—and
under these assumptions measure zero—in `R^(m*n)`.  Therefore it cannot
represent every generic dense matrix exactly.

This gives the first no-go:

> Universal exact coverage requires at least `mn` independent learned degrees
> somewhere in stored weights, mutable state, or runtime-supplied data.
> Executed work or serial depth may exploit structure or change application
> cost, but cannot by itself restore absent matrix information.  Otherwise the
> proposal must restrict the map, domain, or observable, or accept approximation.

The computational boundary points the same way.  In exact online
data-structure/arithmetic-circuit models, with arbitrary query vectors,
polynomial preprocessing space, and sufficiently large fields, generic square
matvec has `Omega(n^2/log n)`-type lower bounds.  This does not cover
approximation, finite precision, batching, restricted queries/states, or
hardware-specific parallel cost.  Low corrupted VC dimension is one
sufficient structured escape for Boolean matrices; the non-Boolean extension
has additional threshold assumptions
([Anand, van den Brand, and McCarty](https://arxiv.org/abs/2502.21240)).

This is not a theorem that Transformers require dense matrices.  It is a
theorem about what the next proposal must explain.

## Four current proof routes—not exhaustive or disjoint

These are search buckets, not a completeness theorem, and candidates may
combine them.  Exact downstream quotient or gauge equivalence belongs under
Exit B.  Hardware, batching, amortization, and memory hierarchy remain
separately charged resource exchanges.

### Exit A: structured learned maps

Restrict the trained matrix to a family with a shorter executable description:
low-rank, sparse, butterfly, Kronecker, repeated rows/subexpressions, a small
alphabet plus a low-complexity row graph, or another proved circuit family.

To claim a changed asymptotic scaling law, the structure needed for protected
language quality must become proportionally cheaper with width.  A constant
dense exception changes a coefficient, not the asymptotic regime.  It may still
produce a breakthrough-size finite-width Pareto improvement and must be judged
on that narrower claim.

Existing boundary in this project:

- the fixed feature DAG represented its own hierarchy, but a 25% Haar component
  destroyed that frozen DAG at the tested width, teacher, and gate;
- the dense-exception theorem requires the protected exception fraction to
  decrease with width;
- projection coalescing failed when sharing deprived attention and local
  features of independent readouts.

A new structured-map proposal must explain **why jointly trained language
weights should have its structure**.  Merely selecting a new factorization is
not a mechanism.

### Exit B: restricted reachable states or observables

Require correctness only on the hidden states the model can actually reach.
Let `X` be that set.  Two linear maps `W` and `W'` are exactly equivalent on
`X` when

\[
(W-W')x=0\quad\text{for every }x\in X.
\]

If `span(X)=R^n`, then `W=W'` only when `W'` is another global linear map
required to reproduce `W`'s vector output exactly.  This does not rule out a
structured `W`, a nonlinear or conditional evaluator on `X`, or exact
end-to-end equivalence modulo a downstream quotient—for example,
softmax-invariant logit shifts or directions annihilated by the next map.  A
gain justified specifically by linear agreement on `X` needs
`dim(span(X))<n`; other gains need one of those separate mechanisms.

For approximate equivalence, let `mu=E[x]`, `x_c=x-mu`, and
`Sigma=E[x_c x_c^T]`, with finite second moment.  For `A=W-W'`, after matching
or charging the affine mean term, the exact mean-square identity is

\[
\mathbb E\|Ax_c\|_2^2
=\operatorname{tr}\!\left(A\Sigma A^T\right).
\]

Without centering, the error adds `||A mu||_2^2`.  For unconstrained rank
approximation, the activation-weighted spectrum is the singular spectrum of
`W Sigma^(1/2)`, not the unweighted spectrum of `W`.

This is an average identity under the baseline state distribution, not an
adversarial or on-policy guarantee.  Rare states need a supremum or tail bound,
and replacing `W` may shift the later reachable-state distribution.

A new reachable-state proposal therefore needs a proved invariant or a frozen
coverage/error theorem before measuring language loss.  “Hidden states look
manifold-like” is not a pre-run mechanism.

### Exit C: compute a different conditional/discrete operator

Stop trying to reproduce an arbitrary fixed matrix.  Let the input select a
small program, expert, state transition, graph route, table entry, or digital
predicate update.  This is the algebraic move behind MoE's capacity/active-work
separation and Mamba's input-conditioned state dynamics.

The candidate now needs a separating task witness:

1. the restricted dense baseline cannot realize the named conditional behavior
   inside the same resource vector;
2. the new operator has an explicit construction that can;
3. route creation, balance, indices, state, and irregular traffic are charged;
4. the witness corresponds to a useful language dependency rather than a
   teacher generated by the candidate topology.

Existing boundary in this project:

- conditional Cayley programs had a large local rank/work theorem but failed
  the language gate;
- packed Boolean state made exact flags cheap but did not create semantic
  features;
- frozen coordinate/token-routing and AFTA screens support—but do not prove—the
  hypothesis that a useful conditional unit must match the dependency
  structure.

This exit remains open in principle, but it must change the semantic operation,
not just replace a GEMM with gathers.

### Exit D: accept a proved epsilon approximation

Change the contract from exact equality to a declared approximation over a
declared domain.  For a linear projection this could be an operator-norm,
activation-weighted, probabilistic, or quantized bound; for attention it could
bound the difference of the normalized kernel sum over the reachable query/key
set.

An approximation is an algebraic exit only when all of the following are
written before measurement:

1. the input domain and norm or probability law;
2. the local error `epsilon` and failure probability;
3. the resource/error scaling law;
4. propagation through normalization, residual addition, nonlinearities, and
   later layers;
5. an end-output margin, KL, NLL, or decision guarantee that makes the local
   error safe;
6. adversarial and out-of-distribution states that expose the failure surface.

For a generic full-spectrum matrix, a tight uniform approximation can still
require essentially dense rank/circuit complexity.  The opening appears only
when the permitted error, input distribution, spectrum, precision, or random
failure budget removes distinctions that the end task provably does not need.

The frozen post-hoc Qwen3.5 top-k/tail-moment families failed their protected
causal-output gate even after receiving exact scores and prefix state.  This
does not close jointly trained sparse attention or all top-k/moment schemes.  A
successor needs a changed approximation object, training
mechanism/distribution, or new certified bound rather than silently relaxing
the same failed threshold.

## Resource exchanges are not an additional algebraic proof route

Precomputation, external memory, quantization, lookup tables, and compilation
can still be excellent systems choices.  They exchange one currency for
another; quantization can simultaneously instantiate Exit D:

```text
online MACs <-> persistent bytes <-> preprocessing <-> approximation error
           <-> irregular traffic <-> update cost
```

They produce a smarter fixed-cost model only when the complete exchange frees
enough serving resource to add a proved capability-bearing object.  The
reverse-projected raw-memory identity is an example: it changes the K/V
memory/compute point but is functionally equal to ordinary cross-attention.

## Pre-run packet for the next matmul proposal

Every proposal must fill this table before code:

| field | required answer |
|---|---|
| chosen exit | structured weights, restricted states, different operator, or proved epsilon approximation |
| end gain | capability and minimum effect, not kernel speed |
| separating witness | smallest family exhibiting the baseline obstruction |
| positive construction | explicit parameters/algorithm solving the witness |
| impossibility scope | exact restricted baseline that the witness excludes |
| approximation | epsilon, norm/domain, and propagated output bound |
| alternative currency | exact resource that pays for the gain |
| strongest control | baseline after receiving every legal compiler/system optimization |
| scale law | why the advantage grows or stays material with width/context |
| block cards | typed formulas, invariants, adversaries, resources, kill conditions |
| final unknown | the single important empirical fact left after microbenchmarks |

Automatic paper rejection occurs when:

- the candidate still claims arbitrary dense-map equivalence with fewer than
  `mn` effective degrees and no paid approximation/trade;
- its sole claim is exact replacement by another global linear `W'`, but the
  reachable states span the full input space and no downstream quotient is
  declared;
- the strongest matched control absorbs the claimed advantage after equal
  compiler, training, and resource allowances;
- the projected capability ceiling is only small polish.

If more than one independent semantic, optimization, or physical uncertainty
remains, hold the candidate and decompose it until the final experiment has one
important unresolved empirical hypothesis.  This is a hold rule, not an
impossibility theorem.

## Current decision

Return candidate generation to these current proof routes.  Do not
microbenchmark a matmul replacement until one proposal supplies both a
separating witness and a breakthrough-size end-to-end resource path.  The H100
remains irrelevant until then.

The first explicit Exit-C proposal in this reassessment was a learned
bit-sliced three-LUT residual.  It has an exact local circuit advantage for
structured Boolean functions, but its claimed one-instruction GPU mapping
holds only when one LUT function is shared across all 32 aligned bit lanes.
Independent learned gates require dynamic table-selection work or compiled
code, and learned wiring adds both persistent indices and irregular movement.
Most importantly, no raw-text feature acquisition or breakthrough-size
language effect is derived.  The primitive is retained on HOLD, with no code
or GPU authorization, in
[`learned-bitsliced-logic-residual-paper-audit.md`](learned-bitsliced-logic-residual-paper-audit.md).

The next Exit-C proposal replaced one attention component with an exact online
suffix-memory head.  It retains a plausible 4--10x state-byte advantage over a
BF16 dimension-128 KV head for literal suffix queries, but it does not retain
an exact continuation distribution in amortized constant time and exact
arbitrary-length copying still requires state linear in the information to be
copied.  Heavy cache, induction, suffix-decoding, and memory-grafting prior art
also absorbs the algorithmic object.  The broad breakthrough claim is rejected;
one whole-KV-group substitution remains on HOLD behind an effect-size paper
gate in
[`exact-dynamic-induction-head-paper-decision.md`](exact-dynamic-induction-head-paper-decision.md).

A third route changed the computation clock rather than the inner operator:
cheap updates on every surface token, with much deeper weight-shared work only
on sparse semantic events.  The exact budget identity permits event depth to
grow as `1/rho`, but router misses compound across event chains and false
positives consume the multiplier.  Hourglass, BLT, Mixture-of-Recursions, and
ANIRA already occupy the general hierarchy/adaptive-depth direction.  Only an
explicit persistent semantic clock remains conceptually open, and it still
bundles event discovery with recurrent-transition learning.  The general
proposal is therefore absorbed without a run in
[`multirate-semantic-depth-paper-reassessment.md`](multirate-semantic-depth-paper-reassessment.md).
