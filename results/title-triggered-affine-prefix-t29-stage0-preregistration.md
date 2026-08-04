# Title-triggered affine prefix operators T29 — Stage-0 preregistration

Status: **FROZEN BEFORE REFERENCE IMPLEMENTATION**  
Date: 2026-07-31

## Question

Does the scalar/vector affine-prefix construction exactly equal sequential
recurrence and title-triggered virtual insertion, and do frozen four-bit BF16
levels obey the proved finite-precision error bound inside the exact served
entry/state ledger?

This is an algebra and codec test.  It reads no corpus, evaluator, model,
checkpoint, or GPU.

## Frozen scalar token codebook

Use the four exact float64 token transitions

```text
(a,b) = (0.25,   -0.50)
        (0.50,    0.25)
        (0.75,    0.50)
        (0.9375, -0.25)
```

The recurrence is `h <- a*h+b`.

Enumerate every token sequence of length zero through six.  Evaluate incoming
states `{-2,-1,0,1,2}`.

## Independent exact forms

For every sequence, compare:

1. direct sequential state updates;
2. a left fold of affine pairs;
3. a balanced recursive fold;
4. the pair reconstructed from two black-box evaluations,
   `B=T(0)` and `A=T(1)-T(0)`.

All final states and pairs must agree within `1e-12` float64 absolute error.

Check associativity for every ordered triple from the token codebook plus every
length-two folded pair.  Check identity `(1,0)` on both sides.

## Frozen title-substitution worlds

Enumerate every `P`, `D`, and `Q` sequence of length zero through two over the
same codebook and every frozen incoming state.  Let `t` range over all four
title-token transitions.

Require exact agreement between:

```text
scan(P + [t] + D + Q)
scan(P + [compose(D,t)] + Q)
```

where the composed title transition is executed once.  Also enumerate two
title/document insertions and require ordered composition to equal the fully
expanded sequence.

For every pair of documents with different exact summaries, substituting the
wrong document must change at least one tested final state/suffix.  This is a
sensitivity check, not a semantic score.

## Frozen four-bit codec

Use the 16 coefficient levels

```text
A = {0, 1/16, 2/16, ..., 14/16, 1}
```

and the 16 offset levels

```text
B = {-2, -3/2, -5/4, -1, -3/4, -1/2, -1/4, -1/8,
      0,  1/8,  1/4,  1/2,  3/4,  1,    3/2,   2}
```

Both affine identity coordinates `A=1` and `B=0` are therefore representable.

Quantize by nearest level with ties going to the lower index.  Values outside
the level range are clipped and counted.  Every level and every decoded index
must survive a CPU PyTorch BF16 round trip exactly.

For every document sequence of length zero through six, quantize its folded
pair.  For every incoming state and every suffix of length zero through three,
compare exact and quantized execution.  Let

```text
epsilon_A = abs(A_hat-A)
epsilon_B = abs(B_hat-B)
alpha_Q   = product(abs(a_q)) over the suffix
bound     = alpha_Q * (epsilon_A*abs(h_in) + epsilon_B)
```

The observed final-state error must not exceed `bound + 1e-12` in any case.
Report clipping, exact/quantized errors, bounds, suffix length, and the
worst-case witness.  Stage 0 does not impose a natural-quality error threshold;
any clipping is a kill because it invalidates the declared finite codec domain.

## Frozen vector and ledger checks

Generate deterministic width-110 vectors by cycling scalar codebook entries.
The vector sequential, left-fold, balanced-fold, and title-substitution results
must agree coordinatewise within `1e-12`.

Assert:

- memory state width: 110 scalars in every arm;
- stored document pair: `2*110 = 220` four-bit cells;
- logical payload: 880 bits/document;
- documents: 2,405;
- payload entries: `2,405*220 = 529,100`;
- title-code entries: 132,800;
- address/gate entries: 76,960;
- thresholds: 2,405;
- up constants: 2,405;
- total existing entries reassigned: 743,670;
- cap: 743,734;
- spare entries: 64;
- served recurrent update remains 110 multiplies and 110 additions per token in
  candidate and controls.

## Mandatory gates

All must pass:

1. zero exact-form, associativity, identity, order, or title-substitution
   mismatches above `1e-12`;
2. wrong-document sensitivity for every unequal-summary pair;
3. all 32 codec levels and all indices round-trip exactly through BF16;
4. zero quantizer clipping over the frozen sequence domain;
5. every quantized final-state error is inside the proved bound;
6. width-110 vector agreement within `1e-12`;
7. every ledger equation matches exactly and no state or per-token recurrent
   operation is added;
8. implementation tests include empty documents, repeated titles, reversed
   order, maximal/minimal levels, and a deliberately noncommuting dense-affine
   counterexample showing that the scalar elementwise representation cannot be
   generalized silently.

## Kill boundary

Failure closes this exact codec or affine substitution as specified.  Do not
change levels, ranges, state width, sequence domain, error tolerance, title
placement, or ledger after observing the result.

Passing admits only a separately frozen stronger-information test of an
unquantized online memory stream.  It does not admit natural training, H100
work, physical coefficient injection, or a production claim.
