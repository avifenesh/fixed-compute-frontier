# Factorized RMS gain — frozen 10M development screen

Status: **frozen before execution**  
Date: 2026-07-28

## Hypothesis

Train an ordinary RMS gain through two temporary factors, then collapse them
to one vector at export.  The served architecture, parameter count, operation,
and checkpoint tensor shape are exactly unchanged.

Candidate effective gain:

\[
s = g\odot(1+0.25\tanh(\theta/0.25)),\qquad u=s\odot RMS(x).
\]

At initialization `g=1, theta=0`.  Export stores only `s` as the ordinary RMS
gain.  Training spends one extra `d` vector and Adam state per sublayer norm.

## Frozen arms

1. `raw`: ordinary RMS gain, base LR.
2. `raw_norm_lr2`: ordinary RMS gain with exactly 2x LR on the 24 sublayer
   norm gains; all other parameters retain base LR.
3. `factor_add`: `s=g+delta`, both factors at base LR.  Under identical Adam
   histories this is the direct constant-two-update control.
4. `factor_multiply`: bounded multiplicative candidate above; its induced
   gain-space metric changes with the factors.

All arms clone one base state.  All 1-D parameters have zero weight decay;
matrices have 0.1.  Frozen protocol: seed 814, H100, 305 steps (9,994,240
tokens), sequence 512, microbatch 32, accumulation 2, 128 validation batches,
LR 3e-4, 30-step warmup, cosine decay, clip 1.0.

Advance multiplicative factorization to 50M only if terminal NLL is at least
0.005% below raw, raw-2x-LR, and additive factorization, with paired 95%
intervals favoring it against the best control.  Initial losses must match,
training must be finite, and actual collapse to one gain vector must preserve
validation NLL within `2e-6` while restoring the raw served parameter count.

