# Rank-three block-algebra SwiGLU — matched LM screen preregistration

Status: frozen before the first language-model arm.

## Hypothesis

Pairing FFN hidden coordinates as real complex numbers changes the activation
core from two coordinatewise products (real multiplication-tensor rank 2) to
complex multiplication (real rank 3).  This exposes 1.5 times as many leading
bilinear atoms with the same learned G/U/V matrices, widths, parameters, dense
matmuls, and activation output size.

The candidate product for activated gate `a` and value `u` is

`(a0*u0 - a1*u1, a0*u1 + a1*u0) / sqrt(2)`.

The scaling matches output variance under exchangeable independent
coordinates.  The exact equal-arithmetic control is split-complex product

`(a0*u0 + a1*u1, a0*u1 + a1*u0) / sqrt(2)`,

whose multiplication tensor is real-split/rank 2.  Complex must beat both this
control and ordinary SwiGLU.  A checkpoint-sidecar test is forbidden: the
diagonal-to-complex interpolation stays in the split/rank-2 orbit until blend
`0.4320408003`, so a safe tiny retrofit would not test the claimed mechanism.

## Frozen quality protocol

- Machine: the retained rented H100 SXM 80 GB.
- Base configuration/tokenizer: SmolLM2-135M revision
  `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, modified before random
  initialization to 12 layers, hidden size 384, intermediate size 1024, six
  query heads, two KV heads, and head dimension 64.
- Tied 49,152-token embeddings and all other Llama/SmolLM2 settings retained.
- Initialization seed 223 and standard deviation `1/sqrt(384)` for every arm.
- Arms, in order: ordinary `baseline`, fixed `split_complex`, fixed `complex`.
- No learned candidate-only parameter, routing, auxiliary loss, compile, or
  checkpoint selection.
- Data: document-disjoint FineWeb-Edu stream at the already frozen dataset
  revision and split rule; 97,656 train sequences and 4,096 validation
  sequences of width 513 uint16 tokens.
- Sequence length 512; microbatch 32; accumulation 2; 1,525 steps; 49,971,200
  prediction tokens per arm.  The final 56 train sequences are intentionally
  unused so every optimizer step is full.
- AdamW, LR `3e-4`, betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay 0.1,
  gradient clip 1.0, 100-step warmup, cosine decay to 10% of peak.
- Evaluate all 4,096 validation sequences at initialization, step 305
  (9,994,240 tokens), and step 1,525.

## Frozen decision

Advance to a fused H100 kernel only if every gate passes:

1. source, preregistration, test, data, and protocol integrity validate;
2. all arms have identical total and trainable parameter counts;
3. initial median activation RMS is within 25% of baseline in both algebra arms;
4. training remains finite, maximum pre-clip gradient norm is at most 100, and
   maximum step loss is at most 20;
5. complex terminal validation NLL is at least 0.1% below baseline;
6. complex terminal validation NLL is at least 0.1% below split-complex;
7. paired 95% validation-batch intervals for both comparisons are below zero.

Failure closes this exact fixed-complex/split scaling, 12-layer, 50M-token,
one-seed recipe; do not tune its scaling, learning rate, grouping, or duration.
It does not by itself close every two-dimensional nonsplit algebra or
quaternion/C4.  A pass authorizes a fused target-scale H100 benchmark.  Only a
quality pass plus at most 1% fused latency overhead authorizes a second-seed
confirmation, and no capability claim survives until that independent seed.
The fused implementation must also reproduce direct-product validation NLL
within 0.01% relative, because Gauss's three-multiply form has different BF16
rounding.

## Claim boundary

The algebra gate proves a larger real multiplication-tensor rank at fixed dense
budget.  This screen tests whether that rank buys language capability.  It does
not claim novelty over hypercomplex neural networks broadly, and eager PyTorch
throughput is not a serving result.
