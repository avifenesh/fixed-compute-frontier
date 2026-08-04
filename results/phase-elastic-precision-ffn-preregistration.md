# Phase-elastic precision FFN: frozen 10M quality screen

## Hypothesis

Prefill and decode have opposite rooflines.  Prefill is a large, compute-bound
matrix multiplication; low-batch decode repeatedly streams weights and is
strongly bandwidth-bound.  A single fixed FFN shape therefore leaves a
different resource idle in each phase.

The candidate stores two full-fan-in branches inside the ordinary BF16 FFN
byte budget:

- a width-1,024 signed-W8 base branch, used in both prefill and decode;
- a width-1,984 signed-W4 residual branch, used only for decode.

Each quantized matrix has one learned BF16 scale per output row.  Per layer:

- dense BF16 reference: 2,359,296 bytes;
- W8 base codes/scales: 1,184,512 bytes;
- W4 residual codes/scales: 1,151,488 bytes;
- candidate total: 2,336,000 bytes, leaving 23,296 bytes slack.

The base-only prefill path executes the ordinary 1,179,648 dense coefficients
per token.  The full decode path executes 3,465,216 coefficients, 2.9375 times
the reference arithmetic, while streaming no more learned FFN payload.  These
are logical ledgers, not a latency claim.  A packed mixed-input H100 gate is
authorized only after quality passes.

## Training

All FFN weights are trained from scratch with straight-through symmetric
quantization.  This is not post-training quantization and does not reuse the
failed frozen W4 endpoint claim.

The W8 branch starts from the ordinary dense initialization.  The W4 residual
uses ordinary gate/up initialization and a down projection scaled by
`0.05 * sqrt(1024/1984)`, so the full path starts close to the base.  During
training, each layer independently drops its residual branch with probability
1/2, without inverse-probability rescaling.  This makes the same weights learn
both base-only and full trajectories without a second forward pass.  Terminal
evaluation deterministically scores both paths.

Frozen protocol: H100, scratch parallel-attention model `D=384,M=1024,L=12`,
seed 815, the existing document-disjoint token stream, sequence 512,
microbatch 32, accumulation 2, 305 steps (9,994,240 prediction tokens), AdamW
`3e-4`, 30-step warmup, weight decay 0.1, clip 1.0, and 128 fixed validation
batches.  The dense comparison is the frozen seed-815 full-SwiGLU endpoint
from the cycle-factor screen.

## Decision

Advance only if all of the following hold:

1. full decode-path NLL beats dense BF16 by at least 0.10%, with a wholly
   favorable paired 95% interval;
2. base-only prefill NLL is noninferior to dense within 0.05%, including the
   paired upper bound;
3. disabling the W4 branch worsens candidate NLL by at least 0.10%, with a
   wholly favorable paired interval;
4. exact code/scale bytes fit, quantized codes and scales are finite, both
   paths are stable, and protocol integrity passes.

A quality pass authorizes a physical mixed W8/W4 H100 decode/prefill gate.  It
does not itself establish unchanged latency, throughput, or energy.

