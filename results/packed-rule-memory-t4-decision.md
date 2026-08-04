# Packed rule memory T4 — decision

Status: killed at sealed BF16 quick gate  
Date: 2026-07-30

T4 did not advance to language training.  The sealed quick result is
`packed-rule-memory-t4-quick.json`.  Structural tests passed 7/7, all 768
supports were recovered exactly, and the nibble table round-tripped all 24,576
support bits through 6,144 scalar values.  Actual BF16 execution failed:

- mean parity accuracy: 49.63%;
- worst-rule accuracy: 12.50%;
- best-rule accuracy: 100%;
- protected-copy accuracy: 100%.

The preregistered construction is closed.  No full training run, threshold
change, or post-result compiler-weight adjustment is admissible.

## Fatal trace

A read-only layer trace separated storage, decoding, and execution:

1. After the nibble-decoder FFN, decoded support signs matched the true support
   at 100%.
2. After rule-to-query attention, copied support signs still matched at 100%.
3. Decoded absolute amplitudes ranged from 0.2715 to 1.0078 in the traced
   batch.  They were not a constant-magnitude sign code.
4. The selector's observed reductions no longer matched the integer sum of the
   selected bits.  Examples included expected `6` observed as `14.0625`,
   expected `2` observed as `-15.875`, and expected `0` observed as `-16`.
5. The count-to-parity decoder therefore received the wrong statistic and
   returned chance-level parity.

The magnitude spread comes from BF16 cancellation in the shared piecewise-SiLU
decoder.  Sign recovery is robust because each integer codeword remains on the
correct side of zero.  Attention routing is not sign-only: softmax exponentiates
the unequal positive scores, so large-amplitude selected coordinates dominate
small-amplitude selected coordinates.  The intended uniform average becomes a
weighted, often nearly single-position reduction.

## Finding retained

Packing four rule bits into one ordinary BF16-safe scalar is representationally
viable: the model recovered every bit sign with a 130-channel shared decoder.
What fails is compositional use of those bits by dot-product attention.  A
usable packed neural memory requires a **symbol regeneration boundary** that
maps decoded values back to a constant-amplitude alphabet before attention.
Correct signs alone are insufficient.

This is not a small numerical polish.  It exposes a general interface law for
heterogeneous neural algebra:

> If a downstream softmax uses decoded symbols as query/key coordinates,
> decoding must preserve score magnitude as well as symbol identity; otherwise
> exponentially amplified amplitude error changes the discrete operation.

A successor may test an explicit error-correcting regeneration stage or an
algebra that consumes signs without exponentiating magnitudes.  It may not
reinterpret T4 as a pass or tune this sealed decoder.
