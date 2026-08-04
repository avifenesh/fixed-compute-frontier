# Folded learned-metric optimizer — matched LM screen preregistration

Status: frozen after the first-batch control probe and before the first full arm.

## Hypothesis and claim boundary

For each selected dense weight, keep only the ordinary fused matrix `W` in the
model while maintaining training-only optimizer factors `A` and `B`. After the
ordinary AdamW update, add

`A_new @ B_new - A_old @ B_old`

directly to `W`. Under SGD this is exactly the effective update of the redundant
chart `W_base + A @ B`; with learned factors its leading geometry contains
left- and right-metric terms. The served function class is unchanged. A positive
result would mean the same ordinary dense architecture reached a better endpoint
because training used a richer coordinate chart. It would not prove a larger
inference function class or broad novelty over reparameterized optimizers.

## Frozen mechanism and controls

- Target every q, k, v, o, gate, up, and down projection in all 12 layers: 84
  matrices and 18,874,368 served weight scalars.
- Rank 16, scale 1, deterministic CPU-QR initialization of orthonormal `A` with
  factor seed 20,260,731, and zero `B`.
- Apply Adam to optimizer-only factors with the model AdamW betas, epsilon, and
  unmultiplied scheduled global LR. Do not decay `A` or `B`. Apply weight decay
  exactly once to the fused served `W` through ordinary AdamW.
- Arms, in order: ordinary `baseline`; `lr_matched`; fixed `A` with trained `B`
  (`fixed_metric`); trained `A` and `B` (`learned_metric`).
- The LR control uses fixed per-matrix multipliers equal to the learned arm's
  effective/base first-update norm. They are frozen in probe SHA-256
  `ca1266d9d4c7b5a1de573f5a4bc6f7148cec6af0d04645af738f720e80afc6f1`.
  Across 84 matrices the multipliers range from 1.0166728487 to 1.1452428068,
  with median 1.0463652324. The probe uses training batch zero, no validation,
  the 3e-6 warmup LR, high float32 matmul precision, and TF32 enabled.

## Frozen quality protocol

- Machine: the retained rented H100 SXM 80 GB.
- Base configuration/tokenizer: SmolLM2-135M revision
  `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, modified before random
  initialization to 12 layers, hidden size 384, intermediate size 1024, six
  query heads, two KV heads, and head dimension 64.
- All arms use initialization seed 223 and contain exactly 37,758,336 ordinary
  model parameters. Factors are optimizer state and never enter parameters or
  the model state dict.
- Data: the existing document-disjoint FineWeb-Edu stream. Train SHA-256 is
  `1871a8a790e2b2ae5273f1a46bc9fa2e39cbe95d5cb7537bde5a2b3e35cb464a`;
  validation SHA-256 is
  `889188f0e4c43cd49b2933ac462e8bd367520f15fe08d324f169f7eca962ce7b`.
- Sequence length 512; microbatch 32; accumulation 2; 1,525 steps; 49,971,200
  prediction tokens per arm.
- AdamW, LR `3e-4`, betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay 0.1,
  gradient clip 1.0, 100-step warmup, cosine decay to 10% of peak.
- Evaluate all 4,096 validation sequences at initialization, step 305
  (9,994,240 tokens), and step 1,525.
- Record effective/base update ratio, factor/base update ratio and cosine, and
  `A`/`B` norms at steps 1, 100, 305, and 1,525.

## Frozen decision

Advance only if every gate passes:

1. source, test, preregistration, algebra, probe, data, and protocol hashes
   validate, and all initial validation losses agree within `1e-7`;
2. all arms remain finite with maximum pre-clip gradient norm at most 100 and
   maximum training loss at most 20;
3. learned-metric terminal validation NLL is at least 0.1% below both baseline
   and the per-matrix LR-matched control;
4. learned-metric terminal NLL is at least 0.05% below fixed-metric;
5. all three paired 95% validation-batch interval upper bounds are below zero;
6. the learned-vs-baseline edge at 50M tokens is no more than 0.025 percentage
   points below its edge at 10M tokens, and learned `A` actually moves;
7. every model disk-serializes as a plain factor-free state dict and reloads
   into the ordinary model with bitwise-equal BF16 logits and exactly equal loss.

Failure closes this exact rank-16, equal-factor-LR, AdamW, 50M-token optimizer
recipe. Do not tune rank, scale, factor LR, target matrices, or training length
after seeing the result. A pass authorizes one independently seeded replication;
it is not a capability claim until that replication passes the same controls.

## Cost ledger

Serving cost is exactly unchanged: zero extra model parameters, checkpoint
weights, dense MACs, KV-cache bytes, or runtime branches. Training adds 1,302,528
factor scalars (3.45% of model parameters), their two Adam moments, and per
optimizer step 905,969,664 extra factor MACs for fixed `A` or 1,207,959,552 for
learned `A,B`. This experiment explicitly spends training-only cost to test for
a better fixed-serving-cost endpoint.
