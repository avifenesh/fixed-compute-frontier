# Dihedral-monomial prefix operators T30 — Stage-0 preregistration

Status: **FROZEN BEFORE REFERENCE IMPLEMENTATION**  
Date: 2026-07-31

## Question

Does the `D_109` monomial-affine construction form the claimed exact closed
operator family, equal sequential and compiled title execution, support a
correct lazy physical frame, and fit the frozen codec/state/record ledger?

This test reads no corpus, language model, evaluator, checkpoint, or GPU.

## Frozen group and action

- width `m=109`;
- group elements `(s,r)` with `s in {-1,+1}` and `r in [0,108]`;
- index action `i -> s*i+r mod 109`;
- code `r` for `s=+1`, and `109+r` for `s=-1`, hence codes `[0,217]`;
- exactly two radix-16 cells store every code;
- operator `T(g,a,b)(h)=a*P_g(h)+b` elementwise.

Enumerate all 218 elements.  Verify code round trips, identity, inverse,
closure, and every ordered triple under the integer group law.  Compare the
formula's permutation with explicit index action.

## Frozen vector token codebook

Create four width-109 token operators.  At coordinate `i`, cycle the T29 scalar
coefficient/offset codebook by `(i+token_index) mod 4`.  Use group elements:

```text
token 0: (+1, 0)     identity
token 1: (+1, 1)     unit rotation
token 2: (-1, 0)     reflection
token 3: (+1, 17)    nontrivial rotation
```

All values are exact binary fractions.  Enumerate all token sequences of
length zero through five and incoming states made from:

1. all zeros;
2. all ones;
3. the linear ramp from -2 to 2;
4. differently valued markers `1` at coordinate zero and `2` at coordinate
   one.

Compare direct sequential execution, left fold, balanced fold, and an
independent materialized affine-matrix application within `1e-12` float64.
For every sequence of length zero through three, reconstruct `B` from the zero
input and reconstruct every monomial column from all 109 basis-vector probes;
require exact recovery of the declared group permutation, scales, and offsets.

## Frozen separation checks

Require:

- `tau*rho*tau = rho^-1`;
- `tau*rho != rho*tau` as materialized matrices and on the two-marker state;
- at least one ordered token pair has different forward/reverse homogeneous
  action;
- the 218 group actions on a vector with all distinct coordinates are all
  distinct;
- setting every group element to identity exactly reduces the implementation
  to T29's diagonal-affine composition.

## Frozen title-substitution worlds

Enumerate every prefix, document, and suffix sequence of length zero through
two, every title operator in the four-token codebook, and all four incoming
states.  Require agreement between expanded and compiled execution.

Also enumerate two title/document insertions with sequences of length zero
through one.  Include repeated title operators, reversed document order, and
wrong-document substitutions.  Every unequal exact document operator must
change at least one tested state/suffix.

## Frozen lazy-frame reference

Maintain logical state as `h=P_f u`.  For each frozen sequence and incoming
state, compare:

1. materializing every permutation;
2. updating only `(s,r)` frame metadata and applying frame-indexed scales and
   offsets to physical `u`;
3. materializing only at the final read.

Require exact group-frame equality and state agreement within `1e-12`.
Report physical vector reads/writes and arithmetic counts.  The reference must
perform exactly one multiply and one add per coordinate per token and no
physical permutation copy.

## Frozen codec and error bounds

Reuse T29's exact BF16 levels:

```text
A = {0, 1/16, 2/16, ..., 14/16, 1}
B = {-2, -3/2, -5/4, -1, -3/4, -1/2, -1/4, -1/8,
      0,  1/8,  1/4,  1/2,  3/4,  1,    3/2,   2}
```

Quantize folded document `A,B` values by nearest level with lower-index ties;
group codes remain exact.  Use every document sequence of length zero through
five, suffixes of length zero through two, and all four incoming states.

Verify zero clipping and the one-trigger bound

```text
suffix_scale * (epsilon_A * max_abs(h) + epsilon_B)
```

coordinatewise and in infinity norm.  Also enumerate every pair of quantized
documents of length zero through two and verify the recursively accumulated
two-trigger bound.  Any group-code error, clipping, or bound violation kills
the frozen codec.

## Frozen ledger

Assert exactly:

- logical state: 109 BF16 values plus one uint16 frame = 220 bytes;
- record: 109 A cells + 109 B cells + 2 group cells = 220 cells;
- logical record: 880 bits/document;
- documents: 2,405;
- payload entries: 529,100;
- full reassigned-entry ledger: 743,670;
- cap: 743,734; spare: 64;
- recurrent arithmetic: 109 multiplies and 109 additions per token in every
  arm;
- physical state vector traffic: 109 reads and 109 writes per token in the
  lazy reference.

## Mandatory gates

All group, fold, reconstruction, separation, title, lazy-frame, codec,
quantization-bound, and ledger checks must pass.  Unit tests must independently
cover identity, inverse, rotation/reflection noncommutation, reversed order,
empty documents, repeated titles, codec endpoints, invalid group codes, and a
lazy/materialized cross-check.

## Kill boundary

Failure closes this exact `D_109` operator or codec as specified.  Do not alter
the group, width, cells, codebook, sequence domain, coefficient levels, or
tolerance after seeing results.

Passing admits only a separately frozen multi-seed synthetic learnability
screen against the byte-matched diagonal, zero, shuffle, and state-cache
controls.  It does not admit natural training, quantized model integration,
H100 kernel work, or a production claim.
