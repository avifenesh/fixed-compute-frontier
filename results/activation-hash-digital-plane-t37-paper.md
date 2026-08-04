# T37 activation-hash digital plane — paper screen

Date: 2026-08-01  
Decision: **CLOSED ON PAPER; CAPACITY-ROBUSTNESS NECESSARY BOUND RETAINED; NO RUN ADMITTED**

## Candidate

T37 asks whether a small neural head can expose an exponentially large digital
address space without growing its continuous width.  A from-zero model processes
raw prose, normalizes a writer activation `z_w`, and assigns it the binary code

\[
c(z)=\operatorname{sign}(Rz)\in\{-1,+1\}^b,
\]

where the rows of `R` are random hyperplanes or rows reallocated from an existing
projection.  A packed payload such as token IDs, a span pointer, or a small logit
delta is written under `c(z_w)`.  At serving, the same model maps a natural query
to `z_q`, computes `c(z_q)`, reads one or a few buckets, and injects the payload.

The hoped-for exchange is attractive:

- the continuous head remains width `b`;
- the apparent address space has `2^b` cells;
- one bucket read replaces a dense FFN or attention read; and
- weight bytes removed elsewhere fund the table, keeping the complete serving
  vector fixed.

The nominal bit namespace is exponential.  All `2^b` sign codes are reachable
only when the hyperplane map has sufficient rank (in particular `b<=d` for a
generic linear map); otherwise the arrangement realizes fewer cells.  Even in
the favorable full-rank case, reliable semantic capacity does not follow from
the namespace size.

## P1. Hashing cannot create a semantic bridge

Let `S` be the latent task-relevant semantic identity, `Z` the normalized model
activation, and `C=c(Z)` any deterministic digital code.  Since

\[
S\longrightarrow Z\longrightarrow C
\]

is a Markov chain, the data-processing inequality gives

\[
I(S;C)\le I(S;Z).
\]

More operationally, every decision rule `g(C)` is also a decision rule
`g(c(Z))` over `Z`.  Therefore the Bayes error using the code cannot be lower
than the Bayes error using the full activation:

\[
\inf_g P[g(C)\ne S]\;\ge\;\inf_f P[f(Z)\ne S].
\]

This does not say a code cannot regularize finite-sample learning or reduce
serving work.  It says the hash cannot manufacture the missing equivalence
between a declarative raw-prose surface and a future question.  If the writer
and query geometry does not already identify the same fact or relation, a
post-hoc hash has no more information with which to repair it; equality is
possible when the code is sufficient for `S`.

For a learned hash, the conclusion changes only in name.  The learned map is
then the semantic quotient under investigation.  Its raw-only acquisition,
margin, supervision, parameters, and serving work must be proved or isolated;
the `b`-bit output is merely its encoding.

## P2. Exact random-hyperplane collision law

Let unit vectors `z_w,z_q` have angle `theta in [0,pi]`.  For an isotropic random
hyperplane, a single sign bit agrees with probability

\[
p(\theta)=1-\frac{\theta}{\pi}.
\]

With `b` independent hyperplanes, exact-bucket recall is

\[
r_b(\theta)=\left(1-\frac{\theta}{\pi}\right)^b.
\]

This equality is useful because it exposes the scaling tension rather than
hiding it inside a nearest-neighbor benchmark.

## P3. Capacity-robustness necessary bound

Assume `N-1` irrelevant writer keys are isotropic relative to a query and T37
probes exactly one `b`-bit bucket.

### False-collision bound

For a false key at random angle `Theta`, the collision probability averaged
over independent hyperplanes and the isotropic key is

\[
q_{b,d}=E_\Theta\left[\left(1-\frac{\Theta}{\pi}\right)^b\right].
\]

Since `E[1-Theta/pi]=1/2` and `u mapsto u^b` is convex for integer
`b>=1`, Jensen's inequality gives

\[
q_{b,d}\ge2^{-b}.
\]

The inequality can be very loose: in two dimensions the unsigned angle is
uniform on `[0,pi]`, so `q_{b,2}=1/(b+1)`.  Consequently the expected number of
false colliders is `(N-1)q_{b,d}`, not generally `(N-1)2^{-b}`.  Keeping it at
most `lambda` nevertheless requires the optimistic necessary condition

\[
b\ge \log_2\frac{N-1}{\lambda}.
\]

The true finite-dimensional requirement may be much stronger.  The bound also
ignores nonuniform learned codes, correlated facts, bucket metadata, and
selection among collisions.

### True-recall consequence

For any collision-safe `b` satisfying that necessary condition, a semantic
pair separated by fixed angle
`theta>0` has recall at most

\[
r_b(\theta)
\le
\left(1-\frac{\theta}{\pi}\right)^{\log_2((N-1)/\lambda)}
=
\left(\frac{N-1}{\lambda}\right)^{
  \log_2(1-\theta/\pi)},
\]

which tends to zero polynomially as `N` grows.  Conversely, maintaining recall
at least `r_0` requires

\[
\theta
\le
\pi\left(1-r_0^{1/b}\right)
\sim \frac{-\pi\log r_0}{b}.
\]

Because collision control makes `b=Omega(log N)`, the writer/query angle must
shrink as `O(1/log N)`.  A capped-size head does not naturally tolerate a fixed
amount of semantic paraphrase noise while its reliable memory grows.

### Concrete scale

For `N=10^6`, even the optimistic necessary condition for at most one expected
false collider is `b>=20`.  At a 30-degree
writer/query angle,

\[
r_{20}=(5/6)^{20}\approx0.0261.
\]

To retain `r_0=0.9` with 20 bits requires

\[
\theta\le\pi(1-0.9^{1/20})\approx0.0165\text{ radians}
\approx0.95\text{ degrees}.
\]

Thus `2^20` nominal addresses do not provide one million robust semantic
addresses unless the model has already learned an unusually tight shared gauge.

## P4. The standard repairs spend the claimed cap

### Multiple tables

With `L` independent tables, recall becomes

\[
1-(1-r_b(\theta))^L.
\]

Achieving target recall `r_0` needs

\[
L\ge
\frac{\log(1-r_0)}{\log(1-r_b(\theta))}.
\]

For the million-key, 20-bit, 30-degree example, 90% recall needs at least
88 tables.  Pointer replication, bucket reads, candidate merging, and random
traffic grow with `L`; they are not a capped head.

### Hamming multiprobe

Probing every bucket within radius `t` costs

\[
B(b,t)=\sum_{i=0}^{t}{b\choose i}.
\]

For a constant tolerated bit-error fraction `0<tau=t/b<=1/2`, this is
`2^{b H_2(tau)+o(b)}` probes.  Keeping `t` constant makes work polynomial in
`b`, but the tolerated error fraction vanishes.  Robustness again exchanges for
work rather than appearing for free.

### Dense and sparse tables

A dense direct table stores `2^b` payload cells; the exponential namespace is
paid in persistent bytes even when most cells are empty.  A sparse map stores
only `N` entries but also stores fingerprints/keys, collision structure, and
payloads, and its reliable capacity remains `Theta(N)` rather than `2^b` free
facts.  A minimal perfect hash eliminates collisions among writer keys but does
not map a noisy query code to the writer key; that step is approximate nearest
neighbor search again.

## P5. Complete resource ledger still required

Even an admitted learned-code successor must charge:

1. `bd` multiply-accumulates or the exact existing projection rows displaced;
2. code packing and sign/threshold instructions;
3. every dense cell or sparse key, pointer, fingerprint, and payload byte;
4. bucket probes, collision-list traffic, candidate scoring, and synchronization;
5. payload-to-residual expansion or decoder work; and
6. the backbone capacity removed to keep parameters, resident state, traffic,
   throughput, and tail latency noninferior.

Reusing existing query rows can fund item 1.  It does not fund the table or
make the query/write codes semantically equal.

## P6. Prior-art boundary

The components are occupied:

- [Reformer](https://arxiv.org/abs/2001.04451) uses random-projection LSH to
  bucket query/key vectors before attention.
- [kNN-LM](https://arxiv.org/abs/1911.00172) stores language-model hidden states
  with their next tokens and retrieves neighbors from the same learned geometry.
- [Product-Key Memory](https://arxiv.org/abs/1907.05242) provides large sparse
  learned lookup tables with structured exact nearest-neighbor access.
- [HashEvict](https://arxiv.org/abs/2412.16187) binarizes Gaussian projections of
  attention queries/keys and uses Hamming distance before attention.
- [Distributed SLIDE](https://arxiv.org/abs/2201.12667) indexes learned neuron
  weights in LSH tables and activates only retrieved neurons.
- [Neural Bloom Filters](https://arxiv.org/abs/1906.04304) already study learned,
  compressed approximate membership under one-shot writes.
- [Lngram](https://arxiv.org/abs/2605.24869) learns low-bit discrete symbols
  from hidden states and performs exact N-gram conditional-memory lookup over
  the resulting latent sequence, closely occupying the learned-address route.
- [MagicPIG](https://proceedings.iclr.cc/paper_files/paper/2025/hash/6d50d824ae819d5a961c1d8edc15e833-Abstract-Conference.html),
  [HATA](https://arxiv.org/abs/2506.02572),
  [Spotlight Attention](https://arxiv.org/abs/2508.19740), and
  [BinaryPC](https://openreview.net/forum?id=4spHlgHY9x) occupy current random,
  learned nonlinear, hash-aware, and data-aware binary retrieval points for
  attention/KV selection.

T37 therefore cannot claim novelty for hidden-state hashing, digital buckets,
sparse activation, or approximate associative lookup.  A contribution would
have to be the raw-only shared semantic quotient or a new physical
time-space-error point, neither of which the hash supplies.

## Decision

Close this random-hyperplane, exact-single-bucket T37 before CPU or GPU work.
The fixed-pair collision law and the necessary false-collision bound refute its
proposed reason for running: nominal fixed-width codes have exponentially many
names, but fixed semantic noise makes exact-bucket recall vanish as even the
optimistic collision-safe code length grows.  Multiprobe, replication, and
dense allocation buy recall with the state/work that the claim was supposed to
avoid.

This result does not rule out learned, few-bucket, structured, or
error-correcting quotients.  It says those mechanisms—not the exponential hash
namespace—must supply the robustness, and must be compared against Lngram and
the strongest current hashing controls.  T37 provides no isolated raw-prose
writer/query training law or matched ledger for such a successor.

Retain:

1. hashing is a resource transform, not a semantic compiler;
2. any learned hash must be evaluated as the missing raw-to-query quotient,
   before table integration;
3. code length, semantic angle, false colliders, probes, and table bytes form one
   inseparable ledger; and
4. a future capped head must expose a correction mechanism whose work grows with
   actual ambiguity, with a frozen expected and tail-cost bound.

No CPU benchmark, local GPU test, or rental is scientifically admitted.
