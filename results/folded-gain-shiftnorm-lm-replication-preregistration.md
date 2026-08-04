# Folded-gain ShiftNorm — frozen 50M replication

Status: **frozen before execution**  
Date: 2026-07-28

## Candidate

During training, each attention/FFN input norm has both scale and shift:

\[
u = g\odot(\operatorname{RMS}(x)+b).
\]

At export, fold `g` into the input columns of every immediate Q/K/V or gate/up
matrix and remove it.  The served norm is

\[
u_{serve}=\operatorname{RMS}(x)+b.
\]

The `d` shift values occupy exactly the `d` slots used by ordinary RMS gain.
The served model has identical parameter count, tensor shapes, normalized
activation width, attention/FFN matrices, head layout, and KV cache.  Runtime
replaces one pointwise gain multiply with one pointwise add.  Training spends
one extra `d`-vector and optimizer state per sublayer norm.

## Frozen arms

1. `raw_baseline`: ordinary learned RMS gain.
2. `direct_shift`: serve-equivalent shift norm trained without the temporary
   gain, testing whether the scaffold is necessary.
3. `scaffold_scale`: train `g*(z+gamma*z)`; an absorbable chart controlling
   training overparameterization without adding an affine shift.
4. `scaffold_bias`: the candidate `g*(z+b)`.

All arms clone one serialized base state.  Every 1-D norm/chart parameter has
zero weight decay; matrices use weight decay 0.1.  Scaffold arms have identical
training counts; all exported served counts equal raw baseline.

## Frozen protocol

- independent seed 813; retained H100;
- same 12-layer hidden-384, SwiGLU-1,024 scratch model and immutable data;
- 1,525 steps = 49,971,200 prediction tokens;
- sequence 512, microbatch 32, accumulation 2;
- 128 validation batches of 32 at 0, 305, and 1,525 steps;
- AdamW LR 3e-4, betas (0.9,0.95), 100-step warmup, cosine decay, clip 1.0;
- Torch 2.5.1+cu124, CUDA 12.4, Transformers 4.57.6;
- no checkpoint selection, arm-specific LR, or shift bound change.

## Frozen gates

1. Protocol/data/base hash, exact initialization, finite training, training
   count, and exported served count checks pass.
2. Candidate is no worse than every arm at 10M.
3. At 50M candidate NLL is at least 0.01% below raw and the best of direct
   shift and scaffold-scale; paired 95% intervals wholly favor candidate.
4. Mean absolute shift is at least 0.002 and maximum below 0.25; output RMS is
   within 5% of input RMS.
5. Removing the learned shift from the terminal candidate worsens NLL by at
   least 0.01%, with a paired interval wholly favoring the full model.
6. Exact multi-consumer export is rechecked after training in BF16 tolerance.

Passing is the first architecture result in this search that trades additional
training state for better capability at unchanged served parameter/compute
shape.  It still requires a production-kernel and larger-scale/multi-seed gate.

