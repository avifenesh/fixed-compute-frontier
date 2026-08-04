# Epsilon fingerprint register — valid error/state trade, not candidate 004

Status: **retain as a classical randomized primitive**  
Date: 2026-07-26  
Candidate number: **none**

## Outcome

This pass found a real mathematical escape from deterministic state bounds, but
not a new model architecture or a served Pareto result. A fixed finite-field
recurrence compares two explicitly delimited token strings using 234 logical
state bits at a maximum length of one million bytes. Equal strings are never
rejected; an unequal pair is accepted with probability at most `5.4211e-14`
when the random field point is fresh and hidden.

The price is explicit: **randomness plus nonzero epsilon**. A randomized digital
recurrent cell can execute the identical update. The construction is therefore
a useful checksum/equality register, not a generally smarter Transformer/SSM
replacement.

## Operator and theorem

Let `enc` inject alphabet `Sigma` into a prime field `F_p`. For each request,
sample a private `r` uniformly from `F_p`. Each explicitly marked segment uses
the fixed update

\[
h \leftarrow rh + enc(a) \pmod p.
\]

For equal-length strings `x,y`, return equal iff `h_x=h_y`. If `x=y`, the result
is always equal. If `x != y`, their hash difference is a nonzero polynomial in
`r` of degree below `n`; a nonzero degree-`d` polynomial over a field has at
most `d` roots. Therefore

\[
\Pr_r[h_x=h_y]\le\frac{n-1}{p}.
\]

Choosing

\[
p > \max\left(|\Sigma|,\frac{n-1}{\epsilon}\right)
\]

gives false-positive probability below `epsilon` and zero false negatives.

The deterministic exact lower bound goes the other way. After reading the
first `n`-symbol string, a one-pass machine must distinguish all
`|Sigma|^n` prefixes: if two different prefixes share a state, appending one of
them after the separator forces the same state to both accept and reject. Exact
deterministic state is thus at least

\[
\left\lceil n\log_2|\Sigma|\right\rceil
\]

bits.

## Frozen ledger

For byte strings, `n=1,000,000`, and the 64-bit prime
`18,446,744,073,709,551,557`:

- fresh random point: 64 bits;
- two fingerprints: 128 bits;
- two length counters: 40 bits;
- phase: 2 bits;
- total logical request state: **234 bits = 30 packed bytes**;
- deterministic exact one-pass lower bound: **8,000,000 bits = 1,000,000 bytes**;
- state ratio: **34,188x**;
- one-sided collision bound: **5.4211e-14**, about `2^-44.07`.

The logical update is one modular Horner step plus a counter/phase update. On
real hardware this is not automatically one instruction: a 64-by-64 product
uses a 128-bit intermediate and modular reduction can require several
instructions. RNG, token encoding, allocation/alignment, and the physical
integer kernel remain outside the logical ledger and must be measured.

## Executed gate

The bounded executable register stores only the evaluation point, two hashes,
two counters, a phase, and static configuration; it never retains either token
stream. It accepts two equal million-token segments at the frozen boundary,
rejects the next token without state mutation, accepts empty segments, rejects
unequal lengths, and rejects invalid event sequences.

Separately, the small-field algebra gate exhaustively evaluated every random
point in `F_257` for 5,334 unequal binary-string pairs through length six:
1,370,838 field evaluations. The largest collision set had five points, exactly
the maximum
allowed for degree five. Equal-string, length, primality, ledger, and persisted
report checks are separate focused tests.

Artifacts:

- [Executable gate](../experiments/epsilon_fingerprint_register.py)
- [Machine-readable result](epsilon-fingerprint-register-stage0.json)
- [Focused tests](../tests/test_epsilon_fingerprint_register.py)

## Fatal controls and collision

This is the classical polynomial-fingerprint idea represented by Karp and
Rabin, not a newly discovered streaming algorithm. More importantly, its strict
space separation is **deterministic exact versus randomized bounded-error**. A
recurrent digital controller given the same finite-field instruction, random
point, state bits, and error budget is byte- and operation-identical.

The guarantee also requires input fixed independently of the secret random
point. Reusing a public point permits adversarial collisions. The register knows
only exact token-string equality between declared segments; semantic
equivalence, arbitrary later substring selection, and learned boundaries bring
back the canonicalizer/router problem. Standard KV retains much more
information, so replacing it with this register would discard capabilities, not
dominate them.

Primary references:

- [Karp–Rabin randomized fingerprints](https://doi.org/10.1147/rd.312.0249)
- [Modern equality randomness/error tradeoffs](https://eccc.weizmann.ac.il/report/2025/068/)
- [Neural Bloom Filters](https://proceedings.mlr.press/v97/rae19a.html)
- [Looped Transformers as programmable computers](https://proceedings.mlr.press/v202/giannou23a.html)

## Decision

The theorem and bounded streaming reference implementation pass. The method is
retained as a classical probabilistic equality/checksum primitive and as a
control for any future claim that fixed recurrent state cannot preserve
long-range global invariants. It is
not candidate 004, does not establish a general language-model gain, and does
not justify GPU rental.
