# Gauge-exposed partner-feedback SwiGLU — frozen 50M-token LM screen

Status: **frozen before execution**.

## Hypothesis

At identical raw/trainable/serialized parameters, dense matrix dimensions,
training tokens, and optimizer schedule, a parallel fixed-partner feedback pass
will improve language-model validation NLL over ordinary SwiGLU, a null carrier
control, and same-neuron feedback.

The candidate is

```text
g  = Gx
u  = Ux
z0 = SiLU(g) * u
q_i = SiLU(g_i + lambda * clip(z0_partner(i), -1, 1)) * u_i
y  = Vq
```

`partner(i)=i xor 1`.  Training replaces the canonical `U[0,0]` coordinate
with one scalar `beta`, so the trainable count stays unchanged.  Define

```text
lambda = 0.1 * tanh(beta / 0.1)
s = 1 + 4 * lambda
chart = BF16(1 / sqrt(384)) = 0.051025390625 exactly
```

The forward pass uses canonical `Ubar[0,0]=chart`, scales the complete row-zero
up activation by `s`, and inverse-scales channel zero before the unchanged down
projection.  At export this folds to `U[0]=s*Ubar[0]` and
`V[:,0]=Vbar[:,0]/s`; the H100 kernel recovers `lambda` from the folded pivot.

Every arm starts at `beta=0`, hence `lambda=0`, `s=1`, and exact ordinary
SwiGLU output.  The removed canonical coordinate and replacement beta contribute
one trainable scalar in total, not two.  Beta has zero weight decay in every
arm; all other parameters retain weight decay `0.1`.  In real arithmetic the
bounded map gives `0.6 < s < 1.4`; BF16 may round to a boundary.  Training may
reconstruct the canonical row/matrix and pay that temporary cost; no such
representation is served.

## Frozen arms

1. `raw_baseline`: status-quo ordinary SwiGLU with the complete trainable
   `up_proj.weight` and standard weight decay on every coordinate.  Its initial
   `U[0,0]` is set to the same chart value but remains an ordinary matrix
   coordinate thereafter.
2. `carrier_null`: canonical replacement-scalar wrapper, but activation ignores
   `lambda`; its row/down scaling cancels exactly.
3. `self_feedback`: the same scalar refines every gate from its own
   clipped `z0`, matching the cheap pointwise expressivity control.
4. `gauge_null`: cross-feature control using only the partner gate,
   `clip(SiLU(g_partner))`, rather than the partner's scale-sensitive SwiGLU
   activation.
5. `partner_feedback`: the candidate fixed adjacent partner map using
   `clip(z0_partner)`.

Every model has exactly the same trainable count and optimizer-state count.
Shared initialization is reset to the same seed for every arm.

## Frozen training protocol

- scratch Qwen-style causal LM: 12 layers, `D=384`, SwiGLU `M=1024`, six query
  heads, two KV heads, head dimension 64;
- immutable existing token stream and validation set from the block-algebra
  screen;
- seed 809; H100; Torch `2.5.1+cu124`; CUDA `12.4`; Transformers `4.57.6`;
- sequence 512; microbatch 32; accumulation 2;
- 1,525 steps = 49,971,200 prediction tokens per arm;
- evaluation: 128 batches of 32 at steps 0, 305, and 1,525;
- AdamW, LR `3e-4`, betas `(0.9,0.95)`, weight decay `0.1`, 100-step warmup,
  cosine decay, gradient clip 1.0;
- no checkpoint selection, second seed, or coefficient LR.

## Promotion gates

All integrity, exact-count, endpoint, finite-training, and stability checks
must pass.  In addition:

1. Candidate is no worse than all controls at 10M tokens.
2. Candidate terminal NLL is at least `0.05%` below raw status-quo SwiGLU and
   at least `0.025%` below the best of carrier-null, self-feedback, and
   gauge-null.
3. Paired document-block 95% intervals favor candidate versus raw SwiGLU and
   the best carrier/activation control.
4. Mean absolute terminal decoded `lambda` is at least `0.002`, no carrier
   scale is nonpositive, and maximum `max(|s|,1/|s|)` is below 2.
5. Setting candidate `lambda=0` after training worsens NLL by at least `0.05%`
   with a paired interval wholly favoring the full model.
6. Replacing partners by self-feedback after training worsens NLL by at least
   `0.02%` with a paired interval wholly favoring the learned partner model.
7. A deterministic shifted-partner permutation ablation is reported.  If all promotion
   gates pass, advancement is to a preregistered second fixed matching rather
   than claiming the adjacent matching is uniquely meaningful.

Passing means one architecture/seed is worth replication and production-like
kernel work.  Failing closes this exact scalar-carried adjacent-feedback
recipe.  The earlier routed re-pairing and per-neuron decoded carrier branches
remain closed regardless of outcome.
