# Self-product FFN 100M-scale preregistration

This scale test was frozen after three 37.75M-parameter screens and before any
result from the larger architecture.

## Fixed architecture

- Hidden size 640, 16 layers, 10 query heads, 2 KV heads, head dimension 64.
- SwiGLU intermediate width 1,792.
- Wide-SiLU and self-product width 2,688.
- Candidate: `B [0.44642046792894413 * SiLU(Ax) * (Ax)]`.
- Parallel attention/FFN layout for all arms.
- Arms: parallel SwiGLU, parameter-matched wide SiLU, self-product.
- Every arm must have identical total/trainable parameters and exactly
  `3 * 640 * 1792 = 3,440,640` FFN dense parameters/MAC coefficients per layer.
- The candidate performs 2,688 SiLUs and pointwise products per layer/token,
  versus 1,792 for SwiGLU. Total scalar-operation counts are not equal;
  fixed served cost requires the BF16-resident H100 gates below.

## Fixed training

- Seed 271828.
- Frozen FineWeb-Edu revision and document-disjoint split rule.
- 195,200 training sequences, 3,050 optimizer steps, microbatch 32,
  accumulation 2, sequence length 512: 99,942,400 prediction tokens per arm.
- 128 fixed validation batches of 32 sequences at steps 0, 610, and 3,050.
- AdamW, learning rate 3e-4, 200-step warmup then cosine decay, weight decay
  0.1, gradient clip 1.0.
- Both scaled arms use fresh BF16 evaluation after folding their scale into
  every down weight.

## Promotion gates

All must pass:

1. Candidate folded terminal NLL at least 0.05% below SwiGLU and 0.025% below
   folded wide SiLU; both paired 95% intervals wholly favorable.
2. Candidate at 20M tokens noninferior to SwiGLU within 0.05%.
3. Plain-SiLU ablation hurts at least 0.01% with a wholly favorable interval.
4. Folded/unfolded NLL differs by at most 0.01% for both scaled arms.
5. All training and activation diagnostics are finite.
6. At initialization and terminal, candidate median activation RMS,
   worst-layer p99, and worst-layer absolute maximum are each no greater than
   SwiGLU's corresponding value.
7. A scale-matched, BF16-resident H100 test has candidate median CUDA-event
   and wall latency no greater than SwiGLU in every prefill and cached-decode
   cell, with bootstrap upper ratio at most 1.02. Candidate decoder-core and
   end-to-end allocated/reserved peak increments must not exceed SwiGLU.
8. The untouched seed-31415 confirmation artifact says
   `advance_to_scale_test=true`, and all data/code/result hashes close.

Passing establishes a replicated 37.75M result that survives a roughly 100M
parameter / 100M-token scale test at fixed served parameters, dense MACs, and
measured H100 service cost—not equal total scalar operations.
It still does not prove billion-parameter scaling or downstream task gains.
