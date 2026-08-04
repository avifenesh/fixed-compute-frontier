# Regenerated packed memory T5 — decision

Status: killed at sealed BF16 quick gate  
Date: 2026-07-30

T5 added the preregistered smooth symbol-regeneration boundary to T4 and did
not advance to full training.  Six structural tests passed.  The sealed BF16
quick gates were:

| gate | result |
|---|---:|
| raw decoded signs | 100% |
| regenerated signs | 100% |
| maximum within-rule magnitude ratio | 1.2469 (fail; limit 1.05) |
| parity mean / minimum | 51.49% / 0% (fail) |
| protected copy | 100% |

The regenerated absolute values ranged from 0.02417 to 0.08105.  The ideal
clipped map is constant amplitude, but its two BF16 SiLU hinge approximations
and cancellation retained roughly 25% within-rule spread.  Softmax again
turned that spread into nonuniform selection.  Correct raw and canonical signs
therefore remained insufficient.

T4 and T5 jointly establish a wall for this smooth packed-memory interface:

1. BF16-safe integer codewords can store four support bits per scalar.
2. A shared 130-channel SwiGLU decoder can recover all bit signs.
3. A second 66-channel smooth clipping stage can recover all canonical signs.
4. Neither produces the constant score magnitude required by dot-product
   softmax, and the resulting discrete program remains unusable.

The exact T5 construction is closed.  No `a`, `beta`, attention-temperature,
or acceptance-threshold tuning is admissible.  Future packed memory would need
either a genuinely discrete regeneration primitive or a downstream algebra
whose semantics depend only on sign.  Adding either changes the architecture
and must be treated as a new research front, not a T5 repair.
