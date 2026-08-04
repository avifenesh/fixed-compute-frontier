# T81 coded-bracket screen — matrix-only preregistration

Date: 2026-08-02  
Status: **FROZEN BEFORE FIRST RUN**

## 1. Scope

This test evaluates only the finite-dimensional linear sketch implied by
Proposition T81.3. It does not simulate vector fields, learn a representation,
validate raw grounding, or support a smarter-model claim. CPU only; no local or
rented GPU is admitted.

The question is whether star-coded commutator measurements have a large
finite-constant advantage over strong matrix-level controls under three
separate cost ledgers:

1. experimental identities or resets;
2. active control components; and
3. quadratic control energy with fixed additive endpoint noise.

## 2. Frozen data family

For `k in {32,64}`, index the `N=k(k-1)/2` upper-triangle generator pairs. At
each anchor `p`, the number of nonzero partners is exactly
`d_p=min(d,k-p)` for `d in {1,2,4}`. Their locations are uniformly sampled
without replacement. Each nonzero pair has an `m=4` dimensional bracket vector
with a uniformly random direction and norm uniform on `[1.0,1.5]`; hence the
beta-min is `1.0`.

Endpoint noise is independent Gaussian with per-coordinate standard deviation
`sigma in {0.0,0.05,0.10}`. The primary development grid uses 64 fixed seeds
for `k=32`; the `k=64` confirmation grid uses 32 disjoint fixed seeds. Run the
development grid first and do not run confirmation if the seam closes there.
All method randomness is independently derived from the world seed.

## 3. Frozen methods

1. **Direct exhaustive pairs:** one noisy observation per pair. Repeats are
   averaged when the budget permits.
2. **Adaptive pairwise with known degree:** randomizes the untested partners in
   each star and stops after detecting `d_p` nonzero rows. This is a stronger
   pairwise control than always testing all pairs. Its detection threshold is
   beta-min/2.
3. **Star code, unnormalized:** fixes `V_p`, signs every possible partner with
   independent Rademacher coefficients, and uses simultaneous orthogonal
   matching pursuit (SOMP) with the known family degree. If the requested code
   count reaches the star width, use direct identity measurements instead.
4. **Star code, equal energy:** identical to method 3 but partner coefficients
   are divided by `sqrt(n_p)`, giving anchor-plus-partner energy approximately
   equal to a pair query. Endpoint noise is unchanged.
5. **Dense wedge code:** independent Rademacher `a,b` produce the physically
   realizable row `a wedge b`; SOMP receives the true global sparsity only for
   this matrix-level upper diagnostic.
6. **Generic dense Rademacher design:** an unphysical ordinary compressed-
   sensing control over all `N` pair coordinates, decoded with the true global
   sparsity. It is a ceiling on what unrestricted group queries buy.

Oracle degree is allowed because the frozen family supplies it to every
degree-aware method, including adaptive pairwise. This tests the theorem's
best-case identity-count seam. No claim about unknown-degree deployment is
permitted from this run.

## 4. Frozen sweep and scoring

For star designs, sweep a common requested measurement count per anchor
`r=2,...,24`, using `min(r,n_p)` actual measurements. For dense wedge and
generic designs, sweep total measurements over a fixed 24-point grid from
`ceil(0.04N)` through `N`. Direct controls report their actual stopping cost.

Execution is staged to keep the falsifier cheap:

1. run only direct/adaptive pairwise and the two star codes on the complete
   `k=32` development grid;
2. if the unnormalized star's median identity-loop gain is below 20%, stop and
   close without running dense diagnostics;
3. otherwise run dense wedge and generic controls on the same development
   worlds; and
4. run `k=64` confirmation only if the development decision remains retained.

For every cell report:

- exact global row-support recovery rate;
- mean pair-level precision, recall, and F1;
- coefficient normalized squared error;
- identity-loop count;
- active-component count; and
- quadratic actuation energy.

The primary `k=32` phase transition is the minimum identity-loop count whose
exact global support recovery has a two-sided Wilson 95% lower bound of at
least `0.90` and point estimate at least `0.95`. The smaller `k=64`
confirmation grid uses lower bound `0.85` with the same point estimate. If no
point passes, record failure. The rule is applied separately at every
`(k,d,sigma)` cell.

## 5. Decision gates

- **Retain the ideal experimental-batch seam:** implementable star code uses at
  most half the identity loops of adaptive known-degree pairwise in at least
  75% of nontrivial grid cells, never loses by more than 20%, and has pair F1
  at least `0.98` at its selected point.
- **Provisional:** relative identity-loop gain is 20% to less than 2x, with no
  single-digit or negative median cell.
- **Close:** median gain is below 20%, any required comparison is single-digit,
  or the result depends on the generic/unphysical decoder beating the
  implementable star code.
- **Energy-qualified physical gain:** the equal-energy star separately clears
  at least 20% lower energy at the same recovery criterion. Failure here does
  not erase a batch-only result, but forbids a physical-efficiency claim.

No matrix outcome satisfies the project's broad 20% intelligence gate. A
retained seam must later improve a complete active-discovery or control phase
by at least 20% against matched learned-world-model controls.

## 6. Integrity checks

- verify numerically that every star row equals the corresponding restricted
  wedge row;
- require exact noiseless reconstruction for a small full-rank identity case;
- verify independent world and sensing seeds;
- record software versions and wall time; and
- write a complete JSON result plus a human-readable decision artifact.
