# Heterogeneous algebraic compilation — causal T1c preregistration

Frozen before result generation: 2026-07-30

## Question

Can a rule recovered from raw token sequences be compiled into the existing
weights of a causal attention-plus-SwiGLU decoder, without adding a module,
state, parameter, branch, or serving operation?

## Fixed causal decoder

- vocabulary: bit `0`, bit `1`, parity-query, copy-query;
- sequence: 32 bit tokens followed by one query token;
- hidden size 64, four causal softmax-attention heads, head size 16;
- one residual attention block and one residual SwiGLU block of width 128;
- learned token and absolute-position embeddings;
- bias-free dense Q/K/V/O, gate/up/down, and output matrices;
- 43,392 parameters in every arm;
- no normalization, recurrence, external memory, conditional kernel, or
  inference-time compiler.

The three T1b dense-mixing worlds are reused.  A parity query asks for the
induced observed-coordinate parity (degrees 18, 17, and 13).  A protected copy
query asks for the first observed bit.  Validation uses fresh latent bits.

## Candidate transition

Every arm begins byte-identical.  The candidate consumes 256 parity-query
sequences, recovers the observed parity mask with overdetermined `GF(2)`
elimination, then writes only ordinary model weights:

1. one causal head attends uniformly to recovered support positions for the
   parity query and to the zero-valued query position for the copy query;
2. one causal head copies position zero for the copy query and attends to the
   zero-valued query position for the parity query;
3. existing SwiGLU channels implement a finite interpolation from the selected
   sign average to parity, gated by the parity-query embedding;
4. the ordinary output matrix reads the parity feature plus copied sign.

The compiler takes no gradient step.  All solver work and parameter writes are
charged.

## Controls

Controls train the byte-identical causal decoder on fresh balanced parity/copy
queries:

- AdamW learning rates `{1e-3, 3e-3}`;
- Muon learning rates `{1e-3, 3e-3}` with RMS-matched adjustment;
- Rademacher rebirth followed by AdamW, multipliers `{0.5, 1.0}`, learning
  rates `{1e-3, 3e-3}`, signed hinge loss;
- batch 64, no weight decay, norm clipping 1.0;
- checkpoints at steps 4 (same example count), 8 (2x examples), 500, 2,000;
- best endpoint selected optimistically per control family and world.

## Frozen gates

Advance only if every world satisfies all conditions:

1. candidate full, parity, and protected accuracy are each at least `99%`;
2. recovered support exactly matches the induced T1b support;
3. candidate and controls have byte-identical initial state hashes;
4. best AdamW, Muon, and Rademacher-hinge parity accuracy is below `80%` at
   step 8 and below `80%` at step 2,000;
5. at least one control reaches `95%` protected copy accuracy;
6. the candidate has the same parameter count, graph, causal mask, and
   inference operations as controls;
7. source, tests, preregistration, and T1b result are hash-bound.

A pass establishes architectural transfer only.  Natural-language coexistence
and novelty remain unproved.
