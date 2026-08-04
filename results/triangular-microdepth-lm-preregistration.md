# Block-triangular microdepth FFN — frozen 50M-token screen

Status: **preregistered before any language-model result**  
Date: 2026-07-27

## Hypothesis

A width-1,528 two-projection SiLU FFN with learned lower-triangular recurrence
inside fixed groups of eight will beat ordinary width-1,024 SwiGLU, exact-budget
plain SiLU-1,536, and an absorbable linear-reparameterization control.  It has exactly
the baseline FFN parameter budget and nearly the same arithmetic, but adds
nonlinear depth in coefficient space rather than a third dense projection.

The claimed gain is better composition/generalization per served parameter and
matmul.  It is not additional independent stored information.

The exact depth witness belongs to the ReLU limiting variant: eight triangular
units implement four tent-map iterates and 16 linear pieces, while a shallow
width-eight one-dimensional ReLU network has at most nine.  SiLU approaches
that construction under temperature/weight scaling; this is not claimed as an
exact finite-width SiLU separation.  The SiLU language result must stand on its
own empirical gate.

## Frozen arms

1. `baseline`: ordinary Qwen-style width-1,024 SwiGLU.
2. `plain_silu`: the strongest simple exact-budget control, a width-1,536
   two-projection SiLU MLP with no coupling or biases.
3. `linear_reparam`: width-1,528 two-projection SiLU; the same packed
   lower-triangular parameters linearly remix activated coefficients.  This
   causal linear recurrence is exactly absorbable into the down projection,
   so it controls for width, biases, parameters, serial sparse arithmetic, and
   training reparameterization without nonlinear microdepth.
4. `triangular`: width-1,528 two-projection SiLU with
   `c_i = SiLU(a_i + sum_{j<i} H_ij c_j)` in groups of eight.

Every replacement FFN has `1,179,648` trainable parameters: `1,173,504` dense
weights, `5,348` packed coupling scalars, and `796` learned biases.  Every full
model must therefore have exactly the same parameter count.  The bias mask is
fixed: offsets `1,3,5,7` in every eight-unit group (764 scalars), plus offset
`0` in the first 32 groups.  This gives every group the threshold sites needed
by the ReLU limiting witness and removes mask ambiguity.

## Frozen training protocol

- H100, PyTorch `2.5.1+cu124`, CUDA `12.4`, Transformers `4.57.6`.
- Existing immutable scratch token files and manifest from the block-algebra
  screen; packed length 512.
- 12 layers, hidden 384, six attention heads, two KV heads, head width 64.
- Seed 277; AdamW `(0.9, 0.95)`, epsilon `1e-8`, weight decay `0.1`.
- Learning rate `3e-4`, 100 warmup steps, cosine decay, gradient clip 1.0.
- Microbatch 32, accumulation 2, 1,525 optimizer steps: 49,971,200 prediction
  tokens per arm.
- Fixed validation at step 305 and 1,525: 128 batches of 32 sequences.

## Frozen decision

Advance only if all are true:

- source/data/protocol integrity passes;
- total and trainable parameters are identical;
- initial activation RMS is within 25% of baseline;
- all runs remain finite and bounded;
- triangular is no worse than all three controls at 10M tokens;
- at 50M tokens triangular improves NLL by at least 0.1% versus both baseline
  and `linear_reparam`, and also versus the width-1,536 `plain_silu` control;
- paired document-block 95% intervals favor triangular versus all three
  controls;
- zeroing every trained `H` worsens NLL by at least 0.05%, with the paired 95%
  interval favoring the full candidate;
- replacing recursive predecessor coefficients with their uncoupled SiLU
  values (the `one_hop` ablation) worsens NLL by at least 0.02%, with the paired
  interval favoring the full candidate.  This prevents a gain from being
  credited to biases or only first-order local mixing.

One seed can reject this recipe.  It cannot establish an architecture claim.
A pass authorizes seed two and a target-shape full-FFN kernel gate; it does not
authorize scale-up.

## Pre-existing hardware boundary

The executor-only fused H100 grid was run before this hypothesis was frozen.
It showed the compiled scalar recurrence at group length eight within about
1.0-2.3% of a separately fused additive executor across FP16/BF16 and batches
1-128, while length 16 hit a large efficiency cliff.  Therefore group size
eight is fixed here.  This is a new no-router FFN mechanism, not a post-hoc
change to the older routed-program gate, whose full frozen grid failed.
