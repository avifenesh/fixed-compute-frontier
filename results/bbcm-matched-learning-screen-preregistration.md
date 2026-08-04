# BBCM exact-budget matched learning screen preregistration

Status: frozen after exact-format endpoint and allocation-control review, before execution.

## Question

Under the same served FFN byte and selected-matrix MAC ceiling, does a shared INT8 base plus four routed ternary residuals learn a better language model than both continued dense BF16 and four independent W4 experts?

This is an upcycling screen, not a from-scratch architecture verdict. FP32 master weights and Adam state make training more expensive; only exported codes, BF16 scales, router, and active inference algebra count in the serving ledger.

## Frozen model, data, and optimization

- `HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.
- Existing document-disjoint train/validation manifest SHA-256 `955cc3d3be7992840da95881450e4483b2dee4a9567b062b17e1ebe061c817a4`.
- One H100; PyTorch 2.5.1+cu124; CUDA 12.4; Transformers 4.57.6.
- Train FFNs and candidate routers only. Attention, embeddings, norms, and LM head are frozen.
- 320 updates, sequence 512, microbatch 32, accumulation 2: 10,485,760 prediction targets per arm. Evaluate all fixed 2,097,152 validation targets at steps 0, 16, 80, and 320.
- AdamW `(0.9,0.95)`, epsilon `1e-8`, cosine decay after 16 warmup steps, weight decay `0.1` on weight masters, no decay on scales/router, clip `1.0`.
- LR `1e-4` for dense BF16 and W4 masters; `5e-5` for BBCM base and delta masters because both contribute to one selected weight; router LR `5e-4`.
- Seed 211 and identical train sequence order.

## Frozen arms

1. Continued dense BF16, width 1,536.
2. BBCM, width 1,520: signed INT8 base plus four ternary `{-1,0,1}` deltas.
3. Independent K4-W4, width 1,520: four symmetric `{-7,...,7}` expert matrices.

Per layer, the BF16 FFN ceiling is 5,308,416 bytes and 2,654,208 matrix MAC/token. BBCM is 5,293,896 bytes; K4-W4 is 5,286,664 bytes. Each conditional arm uses 2,628,864 selected-matrix-plus-router MAC/token. Unused bytes are inert slack.

Before the arms, separately evaluate a dense BF16 width-1,520 endpoint to expose the width cut. On the first eight training-only sequences, score neuron `i` as `mean(abs(SiLU(g_i)*u_i)) * norm(down[:,i])`; retain the top 1,520. Freeze and hash the exact masks, and reuse them in BBCM and W4.

## Frozen exact fake quantization

FP32 masters are fake-quantized on every forward. Positive per-output scales are learned, rounded to BF16 with a straight-through estimator before code assignment, and decoded with the rounded value. Initial INT8 and ternary scales use the exact endpoint algorithms. Physical bit packing is not claimed or tested here.

All four experts start identical. For steps 1-16, freeze the router and assign exact-count balanced routes using a deterministic step/layer-seeded random permutation of flattened tokens. This avoids position-modulo specialization. Thereafter use the same tiny-random BF16-rounded affine router in both conditional arms, hard top-1 with no drops, and load-balance coefficient `0.01`. The selected expert output is multiplied by `1 + p_selected - stopgrad(p_selected)`: exactly one in the forward pass, biased sparse router gradient in backward.

## Frozen diagnostics

- Validation NLL and paired per-batch intervals.
- Three count-preserving, seeded random token-permutation routing controls at step 320; compare against the best (lowest-NLL) shuffled control.
- Forced most-used global expert NLL.
- Route counts, minimum load, and normalized per-layer entropy.
- Exact decoded pairwise route-weight correlation and route-delta/mean variance ratio.
- Export code ranges, ternary alphabet, BF16 scale semantics, exact byte/MAC ledger, training memory, and gradients.

Teacher-output and first-order counterfactual oracles are deferred: implementing them changes memory and evaluation machinery materially. Therefore a negative routing result can reject this router, but cannot prove that no better router exists.

## Frozen promotion gates

BBCM passes this breakthrough screen only if all are true:

1. Exact serving byte/MAC ledgers, matching mask hashes, valid exported code/scale semantics, finite training, and pre-clip norm at most 100.
2. At steps 80 and 320, BBCM is below continued BF16; terminal relative gain is at least 0.05%, with paired 95% upper bound below zero.
3. BBCM beats K4-W4 by at least 0.02%, with paired 95% upper bound below zero.
4. Learned routing beats the best of three count-preserving seeded shuffled controls by at least 0.02%, with paired 95% upper bound below zero.
5. No global route is below 5%, median normalized layer entropy is at least 0.85, and decoded route weights are no longer identical.

If BBCM misses the main quality gates, this exact `8+4x2` upcycling direction is closed. A 640-step extension is allowed only if its terminal gap to BF16 is positive but at most `0.0005` NLL, at least halves from step 0, it significantly beats W4, and routes remain live. A positive one-seed result authorizes replication and the remaining shared allocations—not a serving/runtime claim.
