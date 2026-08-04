# Heterogeneous algebraic compilation — T1 preregistration

Frozen before result generation: 2026-07-30

## Question

Can a charged finite-field solver compile an unknown high-order parity rule into
an otherwise unchanged residual-SwiGLU network, while tuned continuous
optimizers fail to acquire the same rule despite vastly more training work?

This is a fatal small-system test.  Passing advances the mechanism class; it is
not by itself a general-LM breakthrough.

## Fixed system

- input dimension: `32` sign bits plus one binary route bit;
- secret parity support: `24` of the 32 sign bits;
- hidden dimension: `68` (`2*32 + 4`), with zero-valued workspace inputs;
- residual SwiGLU depth: `6` (`ceil(log2(32)) + 1`);
- intermediate width: `64` in every block;
- one scalar output logit;
- no bias, normalization, conditional branch, sparse serving kernel, or
  candidate-only deployed state;
- three paired worlds: seeds `731, 947, 1213`, each with a fresh model
  initialization, secret support, and coordinate permutation.

For route `r=1`, the label is the product of the 24 selected signs.  For
`r=0`, the protected label is the first presented sign.  Both routes occur
equally often.  Validation examples are fresh.

## Candidate

Every arm begins from byte-identical model weights in its world.  The candidate
consumes a fixed prefix of `8d = 256` route-one examples, converts signs and
labels to bits, and performs overdetermined Gaussian elimination over `GF(2)`.
It compiles only if the system has rank 32 and every equation is consistent.

Compilation zeros the residual blocks, writes a balanced multiplication tree
using the exact identity

```text
[silu(z) - silu(-z)] v = z v,
```

and sets the ordinary output head to

```text
x_0 + r*parity(x) - r*x_0.
```

No candidate gradient step is taken.  Solver XOR work, parameter writes, data
exposure, CPU/GPU time, and evaluation work are reported separately.

On an identically shaped random-label prefix with `8d` equations, any
inconsistency must make the solver abstain without compiling.

## Controls

The same initialized model is trained on fresh balanced-route examples with:

1. AdamW learning rates `{3e-4, 1e-3, 3e-3, 1e-2}`;
2. PyTorch Muon learning rates `{1e-3, 3e-3, 1e-2, 3e-2}`, using
   `adjust_lr_fn=match_rms_adamw`;
3. the best learning rate for each optimizer is selected optimistically per
   world by held-out full-task accuracy;
4. batch size `64`, no weight decay, gradient clipping at norm `1.0`, and
   checkpoints at steps `{4, 8, 100, 500, 2000}`.

Four steps expose the same 256 examples as the candidate prefix; eight expose
twice as many.  The 2,000-step endpoint is a deliberately generous control,
not a matched-compute claim.  Dense forward/backward/optimizer work is charged.

## Frozen gates

Advance only if all conditions hold:

1. compiled held-out full accuracy and route-one parity accuracy are each at
   least `99%` in all three worlds;
2. every compiled protected route-zero accuracy is at least `99%`;
3. the best AdamW or Muon control is below `80%` parity accuracy at step 8 and
   below `80%` at step 2,000 in every world;
4. the best control nevertheless reaches at least `95%` protected route-zero
   accuracy, proving that failure is specific to composition rather than total
   optimization failure;
5. recovered support exactly matches the hidden support in every world;
6. all random-label systems are detected inconsistent, compilation abstains,
   and fresh random-label accuracy lies in `[45%, 55%]`;
7. source, test, preregistration, and result hashes are recorded.

Any partial gain, sub-percent endpoint difference, one-seed exception, solver
failure, or random-label compilation closes this exact lane.

