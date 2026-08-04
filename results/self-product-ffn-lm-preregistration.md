# Variance-matched self-product FFN — LM preregistration

## Frozen candidate

Inside the independent one-norm parallel attention/FFN block, replace
width-1,024 SwiGLU with

```
a = A x                                      # width R = 1,536
f = c * SiLU(a) * a
y = B f
```

`A` and `B` contain `2DR = 3DM = 1,179,648` learned scalars and dense MACs per
layer, exactly matching width-`M=1024` SwiGLU.  There is no bias, router,
mask, index, state, or auxiliary loss.

The fixed scale is frozen from standard-normal Gaussian moments using
200-point Gauss-Hermite quadrature:

```
E[SiLU(z)^2] = 0.3557755198173524
E[(SiLU(z) z)^2] = 1.1901360380807904
c = sqrt(E[SiLU(z)^2] / (1.5 E[(SiLU(z) z)^2]))
  = 0.44642046792894413
```

This matches the summed initial second moment of 1,536 self-product features
to 1,024 independent SwiGLU products under the declared Gaussian model.  The
scale is foldable into `B` after training; deployment has two dense projections
and one activation/product kernel.

## What is gained and lost

The candidate has 1,536 independently read nonlinear atoms rather than 1,024.
Its sampled functional Jacobian was full rank at the deterministic `D=3`
probe, while SwiGLU had one exact up/down scale nullity per unit.  It sacrifices
independent gate/value directions: each atom uses one learned direction twice.

Masked GLU, shared-gate GLUs, and power activations are adjacent prior art.
This experiment makes no novelty claim.  It asks whether exact-budget width
plus identifiable self-gating is a real language-model frontier point against
the strongest control discovered in this lab.

## Frozen arms

All arms keep identical parallel attention, one RMSNorm, residual structure,
head configuration, embeddings, and optimizer.

1. `parallel_swiglu`: width 1,024, ordinary independent gate/up/down.
2. `parallel_wide_silu`: width 1,536, `sqrt(2/3)*SiLU(Ax)` and one down
   projection.  Same exact FFN weights/MACs; isolates width from self-product.
3. `parallel_self_product`: the candidate above.

Every arm uses exactly the same total learned parameter count.  Full-strength
matrices use initializer standard deviation `1/sqrt(384)`.

## Frozen training protocol

- Existing block-algebra scratch tokenizer/data and manifest.
- 12 layers, hidden 384, six query heads, two KV heads, head size 64, RoPE,
  SDPA, BF16 autocast, TF32.
- H100 SXM, PyTorch 2.5.1+cu124, Transformers 4.57.6.
- Seed 1907.
- Sequence 512, microbatch 32, accumulation 2, 1,525 steps = 49,971,200
  prediction tokens per arm.
- Fused AdamW, beta `(0.9,0.95)`, epsilon `1e-8`, weight decay 0.1, peak LR
  `3e-4`, 100-step linear warmup then cosine decay, gradient clip 1.
- Fixed validation: 128 batches of 32 sequences at steps 0, 305, 1,525.

## Promotion gates

All must pass:

1. Candidate terminal NLL at least 0.05% below parallel SwiGLU, with a wholly
   favorable paired 95% interval.
2. Candidate terminal NLL at least 0.025% below wide SiLU, also with a wholly
   favorable paired interval.
3. Candidate at 10M tokens noninferior to parallel SwiGLU within 0.05%.
4. Identical total/trainable parameter counts and exact FFN dense ledger.
5. Finite training/activation diagnostics; at initialization and terminal no
   layer has more than 1% of sampled pre-down activations outside four times
   the corresponding parallel-SwiGLU median RMS.
6. Replacing `SiLU(a)*a` by `SiLU(a)` in the trained candidate hurts terminal
   NLL by at least 0.01%, with a wholly favorable paired interval.
7. A deployment-folded H100 rerun remains at most 1.02x parallel SwiGLU in all
   four previously frozen timing cells.
8. All terminal promotion comparisons for both scaled arms use fresh BF16
   evaluations after folding their scale into every down weight. Candidate vs
   wide-SiLU uses the folded wide-SiLU score. Folded and unfolded NLL must
   differ by at most 0.01% independently for each scaled arm.
9. The H100 artifact must embed and match the final LM source,
   LM preregistration, H100 source, H100 preregistration, and H100 test hashes.

This screen can reject or justify replication.  It cannot prove A-E dominance.

## Fatal interpretation

- If wide SiLU wins, width—not self-product algebra—is causal.
- If candidate beats wide SiLU but loses to SwiGLU, independent factor
  directions are more valuable than identifiable atom count.
- If the plain ablation does not hurt, the multiplication is not causal.
- No asymmetric mixing, learned powers, centering, clipping, or per-layer
  scaling rescue is allowed under this branch.
