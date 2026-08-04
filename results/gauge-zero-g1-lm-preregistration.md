# Gauge-zero G1 LM discovery preregistration

Status: frozen before the formal LM run  
Date: 2026-07-27

## Question and claim boundary

Can one gauge-canonicalized physical K weight be reused as a quadratic key
feature and improve validation NLL at the same learned parameter count, model
state, KV width, and dense-QKV shape, while staying within the separately
confirmed incremental H100 epilogue budget?

This first run is a one-seed discovery screen, not a general capability claim.
Validation-batch intervals measure checkpoint-conditional evaluation
uncertainty, not training-seed variance. A discovery pass triggers two frozen
fresh-seed replications; it does not by itself earn the word breakthrough.

## Frozen mechanism and numerical contract

For every norm-free split-half RoPE K pair, rotate deterministic pivot column
`pair_index` to `(radius, 0)` and apply the inverse-transpose rotation to every
sharing GQA Q pair. The zero odd pivot remains an ordinary dense K weight and
is reused as `c = weight / 0.125` in `odd += c * even^2`.

Ordinary attention is exactly contained at `c=0`. No learned scalar, parameter
tensor, optimizer slot, runtime index tensor, pivot metadata, KV coordinate, or
KV byte is added. The served contract is BF16 dense projection and BF16
coefficient; shear plus split-half RoPE in FP32; one BF16 final Q/K store.

The sealed H100 v3 prerequisite must pass and the terminal trained candidate
must reproduce its fused executor bit-exactly in all 12 layers.

## Frozen experiment

- arms, in order: raw Llama attention; deterministic canonical bilinear
  control; zero-pivot G1 candidate;
- scratch Llama: hidden 384, 12 layers, 6 Q heads, 2 KV heads, head dimension
  64, 37,758,336 learned scalars;
- identical initialization seed 2207, data batches, optimizer, schedule,
  clipping, BF16 autocast, and SDPA;
- one fused AdamW group, betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay
  0.1, peak LR `3e-4`, 50-step warmup and cosine decay;
- sequence 512, microbatch 32, accumulation 2, 305 optimizer steps:
  9,994,240 prediction tokens per arm;
- validation at steps 0, 61, and 305 with 64 batches of 32 sequences:
  1,048,576 paired prediction tokens per checkpoint;
- H100, Python 3.11.10, NumPy 2.1.2, Torch 2.5.1+cu124, CUDA 12.4,
  Transformers 4.57.6;
- source, attention source, tests, data manifest, initial preflight, H100 v3
  result/contract, this document, and exact runtime arguments are hash- or
  protocol-sealed before training;
- save and hash all three FP32 terminal checkpoints plus the candidate BF16
  dense-attention export.

## Frozen all-or-nothing discovery gates

Every gate must pass:

1. learned scalar/tensor counts, model buffer counts, serialized state values
   and bytes, and optimizer-state bytes are equal across arms; all metrics are
   finite and key nonfinite fractions are zero;
2. initial candidate/control NLL is exact; raw/control aggregate delta is at
   most `1e-4` and its paired 95% interval lies wholly inside `[-1e-4, 1e-4]`;
3. terminal canonical-control minus raw paired 95% upper bound is at most
   `0.05% * raw_NLL`;
4. candidate-minus-control and candidate-minus-raw paired 95% upper bounds are
   each at most `-0.025% * reference_NLL`—the confidence bound, not merely the
   point estimate, must clear the practical floor;
5. on the trained candidate with all weights fixed, disabling only the shear
   increases NLL by at least 0.01%, and the full-minus-disabled paired upper
   bound is at most `-0.01% * candidate_NLL`;
6. BF16 coefficient export is exact, some coefficients survive BF16, maximum
   absolute coefficient is at most 0.5, pivot metadata is zero, and every
   trained layer is bit-exact against the fused H100 export path;
7. all sealed integrity, runtime protocol, data ledger, checkpoint hashes, and
   BF16 export hash are valid.

No threshold may be changed after seeing the formal result. A failed gate
closes this exact G1 discovery configuration rather than inviting a post-hoc
threshold or seed change.

## Frozen continuation rule

Only if seed 2207 passes, run the same matched experiment at fresh seeds 3719
and 6673. Retain as a replicated architecture pre-candidate only if candidate
NLL beats canonical control in all three seeds, the mean relative improvement
is at least 0.025%, neither control violates raw noninferiority, and every
stability/export/cost invariant remains valid. Broader capability still needs
larger-model, longer-training, task, quantization, and production-serving tests.
