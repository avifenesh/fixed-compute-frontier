# Title-incidence direct-record audit

Status: **exploratory algebra retained; exact direct route rejected**  
Date: 2026-07-31

## Question

Can arbitrary 220-dimensional document records be written directly into the
existing token-embedding rows so that averaging a title's token rows recovers
its record, eliminating per-document FFN lookup?

For 2,405 titles and the 4,150 vocabulary rows occurring in their canonical
tokenizations, define `A[i,j]` as token `j`'s normalized count in title `i`.
For token-row payloads `X` and desired document records `Z`, exact direct
encoding requires

`A X = Z`.

This is possible for every `Z` if and only if `A` has full row rank 2,405.

## Exact H100 result

- `A` shape: `2,405 x 4,150`.
- Nonzeros: 12,643.
- Duplicate title token multisets: zero.
- Exact float64 numerical rank: **2,379**.
- Nullity in document space: **26**.
- Largest singular value: 3.8504806377.
- Smallest singular value: `3.77e-17`.
- Effective condition number: `1.02e17`.

A bipartite maximum matching can assign a distinct active vocabulary row to
every title, so the sparse pattern is structurally full rank.  But the selected
2,405-square incidence matrix is numerically only rank 2,344 and has condition
number `1.92e19`.  Structural uniqueness therefore does not yield a robust
linear code.

## Decision

Do not claim exact arbitrary record access from additive title-token payloads.
The row space can carry 2,379/2,405 = 98.9189% of independent per-coordinate
document degrees, so it may be a useful approximate parameterization for
jointly learned records.  It cannot replace the exact router for an arbitrary
pre-existing table, and BF16 inversion of the matched square basis is not
credible.

Order-sensitive contextual computation could break the 26 linear
dependencies, but that is a learned nonlinear router and must be priced and
tested as such.  This audit does not reopen T15-T17 post-hoc analog routing.
