# Training-time gauge scaffold — frozen 10M chart screen

Status: **frozen before execution**  
Date: 2026-07-28

## Cost trade under test

Spend extra parameters and optimizer state only during training so the served
model can contain both the optimization-useful RMS gain and a functional chart
without increasing served parameters.

Training computes

\[
y_j = W_j\,[g\odot C_\gamma(\operatorname{RMS}(x))].
\]

At export, for every immediate consumer set
`W'_j = W_j diag(g)`, remove `g`, and serve `W'_j C_gamma(RMS(x))`.  This is
exact for all Q/K/V consumers of an attention norm and both gate/up consumers
of an FFN norm.  Gamma occupies the original `d` norm slots.  Training has
`d` extra values per sublayer norm; serving does not.

## Frozen charts

All scaffold arms keep explicit gain `g` and a bounded
`gamma=0.25*tanh(theta/0.25)` initialized at zero:

1. `scaffold_null`: `C(z)=z`; theta is a null slot.
2. `scaffold_scale`: `C(z)=z+gamma*z`; a purely linear, absorbable chart and
   optimizer/reparameterization control.
3. `scaffold_bias`: `C(z)=z+gamma`; tests an input-independent shared shift.
4. `scaffold_hinge`: `C(z)=z+gamma*abs(z)`.
5. `scaffold_centered`: `C(z)=z+gamma*(abs(z)-sqrt(2/pi))`, testing the
   sign-dependent component after removing the Gaussian absolute-value mean.

`raw_baseline` is ordinary learned RMSNorm gain with no training scaffold.

All 1-D norm/chart parameters use zero weight decay in every arm; matrix
parameters retain weight decay 0.1.  Every arm is cloned from one serialized
base initialization before norm replacement.  Scaffold arms have equal
training parameter/optimizer counts.  After export every arm has the baseline
served parameter count and unchanged matrices, activation widths, heads, and
KV cache.

## Frozen protocol

- same 12-layer hidden-384, SwiGLU-1,024 scratch model and immutable token
  streams as prior screens;
- seed 812; retained H100; 305 steps = 9,994,240 prediction tokens;
- sequence 512, microbatch 32, accumulation 2;
- 128 validation batches of 32 at step 0 and 305;
- AdamW LR 3e-4, betas (0.9,0.95), 30-step warmup, cosine decay, clip 1.0;
- Torch 2.5.1+cu124, CUDA 12.4, Transformers 4.57.6;
- no arm-specific LR, coefficient bound, or checkpoint selection.

## Frozen selection and gate

Select the lower terminal NLL of `scaffold_hinge` and
`scaffold_centered`.  This is an explicitly exploratory family selection; a
passing winner must later replicate at 50M on a new seed.

Advance the selected nonlinear chart only if:

1. protocol/data/base-state hashes pass; all initial losses match within
   `2e-5`; training is finite;
2. scaffold training counts match each other, and every exported served count
   equals raw baseline;
3. winner NLL is at least 0.005% below raw baseline and the best of null,
   linear-scale, and shared-bias controls;
4. paired per-batch 95% intervals wholly favor the winner against raw and the
   best control;
5. mean absolute gamma is at least 0.002, maximum is below 0.25, positive-sign
   fraction remains `[0.1,0.9]`, and output RMS is within 5% of input RMS;
6. setting gamma to zero in the trained winner worsens NLL by at least 0.005%
   with a paired interval favoring the full chart;
7. terminal bias-only, centered, and uncentered hinge ablations are reported.

Passing means the training-only cost trade is worth a 50M replication.  It
does not yet establish a general model improvement or quantized export.

