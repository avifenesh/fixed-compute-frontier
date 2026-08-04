# Multirate semantic depth: paper reassessment

Date: 2026-08-01  
Status: **ABSORBED AS A GENERAL DIRECTION; RETAIN A NARROW OPEN QUESTION; NO CODE OR GPU**

## Independent proposal

The status-quo assumption attacked here is that every surface token should pay
for the same feature width and depth.  Natural language operates at several
clocks: bytes/subwords change rapidly, while entities, propositions, goals,
and discourse state change less often.

A two-clock block would maintain a cheap fast state `f_t` at every token and an
expensive weight-shared slow state `s_k` only at selected events:

\[
f_t=F(f_{t-1},x_t,s_{k(t)}),
\]

\[
s_{k+1}=G^{R}(s_k,\operatorname{pool}(f_{a_k:b_k}))
\quad\text{when }e_t=1.
\]

`G^R` means `R` recurrent applications of the same block.  The intended edge
is more reasoning depth per semantic event without more model bytes or average
served work.

## Exact resource identity

Let events occur on fraction `rho` of tokens, let `c_f` be fast work per token,
and let one slow recurrence cost `c_s`.  Average work is

\[
C_{multi}=c_f+\rho R c_s.
\]

Under baseline budget `C`, the maximum funded recurrence is

\[
R\le \frac{C-c_f}{\rho c_s}.
\]

This is a real scaling axis: as event density falls, the architecture can spend
depth proportional to `1/rho` on each event.  Weight sharing keeps persistent
slow-block parameters independent of `R`.

It is a reallocation theorem, not a capability theorem.  The gain exists only
if the task's important computation is sparse in this clock and repeated use
of `G` is more valuable than the removed per-token computation.

## Router error boundary

Let a required event be missed with probability `beta`.  A task needing all
`K` event updates has success ceiling

\[
P(success)\le(1-\beta)^K
\]

before any reasoning errors.  To sustain target `tau`, the router needs

\[
\beta\le1-\tau^{1/K}.
\]

For `K=8` and `tau=0.8`, this requires `beta <= 0.0275`.  Semantic boundary
recall therefore cannot be treated as a forgiving implementation detail.

If `alpha` is the false-positive rate on non-events, actual slow-path density
is

\[
\rho'=\rho(1-\beta)+(1-\rho)\alpha,
\]

and the funded depth becomes

\[
R\le\frac{C-c_f}{\rho'c_s}.
\]

False positives directly consume the claimed reasoning multiplier.  Bursty or
adversarial text can make `rho'` approach one, creating either a latency-tail
violation or a forced depth reduction.

## Capability witness and its limit

The favorable witness contains long surface spans per semantic event and a
target requiring sequential composition across event states.  A slow recurrent
state performs one exact transition per event while the fast path absorbs
surface redundancy.  Compared with a uniform token model at the same average
work, it can devote more sequential transformations to the causal event chain.

This does not establish a separation from:

- a recurrent SSM updating once per token;
- hierarchical patch models;
- a token router assigning variable recurrent depth;
- fixed chunks with the same slow/fast work split; or
- an ordinary model trained to use cheap local layers and sparse global layers.

An exact lower bound against that complete control class is not yet supplied.

## Prior-art collision

The algebraic direction is already strongly represented:

- [Hourglass](https://arxiv.org/abs/2110.13711) downsamples and upsamples token
  representations to spend expensive Transformer work hierarchically.
- [Byte Latent Transformer](https://arxiv.org/abs/2412.09871) uses
  entropy-sized byte patches and reports better scaling at fixed inference
  cost by growing patch and model size together.
- [Mixture-of-Recursions](https://arxiv.org/abs/2507.10524) combines shared
  recurrent weights with token-specific depth and reports a better
  compute/parameter Pareto frontier.
- [ANIRA](https://arxiv.org/abs/2602.08864) isolates adaptive recurrent compute
  and finds that complexity-aligned allocation can emerge, while also showing
  that it does not imply algorithmic length generalization.

The equation `cheap frequent clock + expensive rare clock` is therefore not a
new frontier by itself.  Combining dynamic patches and recurrent depth is also
an obvious composition unless it changes the proof object or produces a new
control separation.

## What remains genuinely open

Current patching commonly follows local predictability/entropy, while adaptive
depth commonly routes individual token states.  The narrower unresolved object
is a **persistent semantic clock** whose events correspond to causal state
changes and whose slow recurrence carries an explicit event-to-event program.

That version is not admitted because it presently bundles two empirical
unknowns:

1. discovering high-recall semantic events from raw next-token training; and
2. learning a recurrent transition whose additional steps yield material
   natural reasoning rather than redundant refinement.

It also needs a componentwise serving policy for event bursts.  Average FLOPs
alone cannot pay for worse p95/p99 latency, batch divergence, mutable state, or
sequential prefill depth.

## Decision

Retain temporal sparsity as a valid scaling axis and BLT/MoR as evidence that
the axis matters.  Do not claim or test the general two-clock architecture.
Reopen only if the semantic clock itself receives:

1. a raw-observable or separately testable event definition;
2. a witness separating it from entropy patches and per-token adaptive depth;
3. an event-miss effect bound that supports a material end gain; and
4. a complete burst/tail serving ledger.

This is a useful map of an occupied direction, not yet the new method.
