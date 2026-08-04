# Phase-elastic v3: phase-faithful semantic screen

## Why this experiment exists

The v2 architecture passed its untouched 50M homogeneous replication: its
base-only endpoint beat BF16 dense by 0.1791%, and its full endpoint beat BF16
dense by 0.6315%.  That experiment trained each layer's optional branch with an
independent Bernoulli switch.  It therefore did not train or evaluate the
deployed state transition in which a base-only prompt cache is consumed by a
full-path decoder.

V3 changes only the training switch.  Every layer receives the same tokenwise
phase boundary.  In zero-based indexing, positions `< P` use the base and
positions `>= P` use base plus branch.  Thus a server can prefill `P` tokens
base-only, process the last prompt token at position `P` through the full path,
and use its logits to predict the first generated token.  No token is inserted
twice and absolute positions continue across the boundary.

This is a semantic quality gate.  Packed-kernel cost remains a later gate.

## Frozen experiment

- fresh seed 6673;
- arms in order: BF16 dense baseline, phase-faithful v3 candidate;
- the v2 codebooks and byte allocation are unchanged;
- scratch parallel-attention model `D=384,M=1024,L=12`;
- identical batches, sequence 512, microbatch 32, accumulation 2;
- 305 AdamW steps = 9,994,240 prediction tokens loaded per arm;
- ordinary full-sequence next-token loss for both arms, so no altered loss
  weighting can explain a gain;
- candidate phase boundaries sampled independently per sequence from first-full
  positions `{64,128,256,384,448,480,496}` and shared by every layer;
- peak LR `3e-4`, 30-step warmup, cosine decay, betas `(0.9,0.95)`, weight
  decay 0.1, clip 1.0;
- 64 fixed validation batches;
- H100, Torch `2.5.1+cu124`, CUDA 12.4, Transformers 4.57.6.

## Frozen evaluation

Mixed-mode suffix NLL is evaluated at first-full positions
`{256,384,448,480,496}`.  For each boundary, loss is reported for:

- all suffix logits beginning at `P`;
- first generated-token logit at `P`;
- decode offsets 2-8;
- decode offsets 9-32;
- decode offsets 33 and later, when present.

The primary mixed-mode score gives each boundary equal weight, then averages
paired validation batches.  Dense is scored on exactly the same token indices.
Homogeneous base and full NLL are diagnostics, not the deployed metric.

One-pass masked logits and K/V are also compared with actual two-stage cached
execution: base-only prefill for positions `< P`, followed by full-path
incremental execution from `P`.  This check uses the same weights and tokens in
both FP32 with TF32 disabled and the deployed BF16 autocast path.

Before any terminal training result was observed, an untrained-model smoke test
showed the expected reduction-shape distinction: FP32 paths agreed near
`1e-5`, while BF16 one-pass versus incremental execution differed by up to
`0.092` in logits and `0.065` in K/V.  The frozen semantic gate therefore uses
a strict FP32 algebra tolerance and separately calibrated BF16 tolerances.  An
off-by-one or cache-state error is orders of magnitude larger than either.

## Frozen pass

V3 advances to a long-horizon, multi-seed replication only if:

1. cached two-stage execution agrees with one-pass phase masking: FP32/TF32-off
   maximum difference `<= 3e-5`; BF16 max logit difference `<= 0.125`; BF16
   max K/V difference `<= 0.10`;
2. aggregate mixed-mode suffix NLL beats dense by at least 0.10%, with a wholly
   favorable paired 95% interval;
3. aggregate first-token NLL is noninferior to dense within 0.05%, including
   the paired 95% upper confidence bound;
4. no evaluated boundary's all-suffix NLL is worse than dense by more than
   0.20%;
5. all byte, code, strictly-positive scale, phase-use, finiteness, data, and
   protocol checks pass.

A pass is evidence that the quality effect survives the real phase transition.
It is not yet evidence of equal latency, broad capability, or production-ready
quantization.
