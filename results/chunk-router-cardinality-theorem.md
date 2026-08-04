# Cardinality-neutral chunk routing — uniqueness and minimum-state boundary

Status: **theorem retained; successor architecture not yet admitted**  
Date: 2026-07-27

## Setup

For a chunk of `n` attention scores `s=(s_1,...,s_n)`, let

\[
Z_1(s)=\sum_i e^{s_i}.
\]

A chunk-common lift `A_n(s)` gives the chunk unnormalized router mass

\[
W_n(s)=e^{A_n(s)}Z_1(s).
\]

The local value mean may remain ordinary softmax. The question is whether a
deterministic, permutation-invariant lift can be content-sensitive, compatible
with row shifts, and avoid giving chunks more expected router mass per token
only because they contain more IID draws.

## Strong neutrality is impossible

Row-shift compatibility requires

\[
A_n(s+c\mathbf 1)-A_n(s)=a c
\]

for one slope `a` shared by all chunks. Requiring the full distribution of
`A_n(X_1,...,X_n)` to be independent of `n` for every IID law forces a
constant lift.

A two-point IID law is sufficient. Degenerate laws fix the values at `(x,x)`
and `(y,y)`. The mixed pair occurs with probability `2p(1-p)`. If its lift is
new, the two-sample law gains a third atom; if it equals an endpoint, the atom
probabilities differ from the one-sample law. Hence a nonconstant universally
distribution-neutral anchor statistic does not exist, even with access to all
scores.

## Unique universal mean-neutral router

Use the weaker attention-relevant condition

\[
\mathbb E_F[W_n/n]\quad\text{is independent of }n
\]

for every IID law with the required finite exponential moment. Symmetry and
universal unbiasedness force the per-token routed mass to be the sample mean
of its one-token form. Together with row-shift compatibility, the unique family
is

\[
\boxed{
A_n(s)=b+\log\sum_i e^{(1+a)s_i}-\log\sum_i e^{s_i}
}
\]

and therefore

\[
\boxed{W_n=e^b Z_p,\qquad Z_p=\sum_i e^{p s_i},\ p=1+a.}
\]

This separates local reading from chunk routing:

\[
\mu_b=\frac{\sum_j e^{s_j}v_j}{Z_{1,b}},\qquad
y=\frac{\sum_b Z_{p,b}\mu_b}{\sum_b Z_{p,b}}.
\]

At `p=1`, this is ordinary attention. For `p>1`, one strong anchor can raise
the router mass of its neighborhood without changing the local payload ratios.
For every IID score law,

\[
\mathbb E[Z_p/n]=\mathbb E[e^{pX}],
\]

so variable chunk count does not create expected per-token router mass.

## Why ordinary FlashAttention state is insufficient

Ordinary block state stores `m=max s` and
`l_1=sum exp(s-m)`, which determine `Z_1`, but not `Z_p`. In exponentiated
coordinates, `(1,0.6,0.2)` and `(1,0.5,0.3)` have the same maximum and sum but
different sums of squares. Values cannot repair the missing score statistic:
set every value to zero and `o` becomes identical.

For `p=2`, the exact missing statistic is only

\[
l_2=\sum_i [e^{s_i-m}]^2.
\]

It needs no second exponential, QK/PV/FFN multiplication, KV field, persistent
cache, or activation-sized allocation. It does require one additional scalar
accumulator/reduction/register path. That price cannot honestly be called zero.

## Boundary still open

The theorem solves cardinality neutrality, not hard chunk boundaries. The
failed peak screen showed a 36%-41% payload loss when an anchor and its payload
crossed a boundary. Replacing the statistic by `Z_p` does not remove that
discontinuity. Power-evidence routing is therefore not admitted directly to
training or H100 work.

A successor must first show one of:

1. a same-head boundary-continuous or overlapping construction within the
   paid scalar-state ledger; or
2. a staggered multihead construction whose aggregate causal signal survives
   every relative anchor position without sacrificing ordinary-head capacity.

Only then should the one-extra-reduction hardware and language gates run.
