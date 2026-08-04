# TVE 360M second-domain continuation design

This is a build design, not yet a preregistration. It is intentionally a
continued-pretraining gate rather than a 360M scratch run: 100M tokens is 2.65
tokens per parameter for the 37.8M pilot but only 0.28 tokens per parameter at
360M, so a scratch comparison would mostly retest early optimization in a
severely undertrained model.

## Base and domain

- Base: `HuggingFaceTB/SmolLM2-360M` revision
  `f8027fd0eaeea54caa13c31d31b9fdc459c38b49`.
- Exact geometry: 32 layers, hidden 960, Q15/KV5, head 64, intermediate 2,560,
  tied 49,152-token embeddings, context 8,192.
- Second domain: official peS2o v2 train/validation splits at dataset revision
  `636a503e44a3ca1b58e01fb61eab0825cd574de0`.
- Tokenizer: the pinned SmolLM2 tokenizer. Train and validation remain official
  split-disjoint.

## First scale gate

Continue the same pretrained checkpoint for 99,942,400 tokens per arm:

1. packed raw native continuation;
2. function-preserving gauge-canonical continuation, TVE off;
3. block-16 TVE continuation using the admitted custom backward.

Use FP32 master parameters with BF16 autocast, sequence 512, microbatch 8,
accumulation 8, 3,050 steps, AdamW, learning rate 3e-5, weight decay 0.1, 50
warmup steps, cosine decay, and gradient clipping at 1.0. Evaluate 64 paired
batches at 10M, 25M, 50M, and 100M tokens. The first run uses one frozen data
order and is a scale/domain rejection gate only.

Candidate must beat both raw and gauge-canonical terminal NLL with paired upper
bounds beyond a frozen practical margin, retain a bit-exact BF16 serving bridge,
use no new state/cache/metadata, and remain within 5 percent of raw training
throughput after custom-autograd integration. Endpoint disabling must hurt.

## Claim boundary

A positive first run proves only that the mechanism transfers to one exact 360M
pretrained checkpoint and a second continued-pretraining domain. It does not
estimate training-seed variation because the base checkpoint and dropout-free
training are deterministic. Confirmation requires at least three frozen data
orders or independently trained bases, then the protected capability suite and
native full-attention serving gate in the capability/resource contracts.
