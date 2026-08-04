# Phase-elastic v2: untouched 50M replication

## Scope

This is the first fresh-initialization, long-horizon replication of the
phase-elastic precision FFN selected at seed 815 after its 10M screen passed
every frozen quality and storage gate.

The architecture is unchanged: a width-1,024 INT8-plus-ternary base active in
both phases, plus a width-1,472 W4 branch active only for decode.  Each layer
independently drops the branch on half of training forwards.  Packed FFN bytes
remain 2,338,816 per layer versus 2,359,296 for BF16 dense.

## Frozen experiment

- fresh seed 3719 for initialization and training randomness;
- arms in order: ordinary BF16 dense baseline, phase-elastic v2;
- scratch parallel-attention model `D=384,M=1024,L=12`;
- identical token batches, sequence 512, microbatch 32, accumulation 2;
- 1,525 AdamW steps = 49,971,200 prediction tokens;
- peak LR `3e-4`, 100-step warmup, cosine decay, betas `(0.9,0.95)`, weight
  decay 0.1, clip 1.0;
- 128 fixed validation batches at initialization, 10M, and 50M;
- H100, Torch `2.5.1+cu124`, CUDA 12.4, Transformers 4.57.6.

The baseline is retrained at the fresh seed; no seed-815 quality number is used
as its terminal reference.

## Frozen pass

At 50M tokens the candidate must:

1. improve full-path NLL over dense by at least 0.25%, with a wholly favorable
   paired 95% interval;
2. keep base-only NLL noninferior to dense within 0.05%, including its paired
   upper bound;
3. improve full over base-only by at least 0.10%, with a wholly favorable
   interval;
4. retain at least half of any positive full-path relative edge observed at
   10M, and retain an absolute edge of at least 0.10%;
5. pass exact byte, code, scale, path-use, finiteness, data, and protocol gates.

A pass establishes a replicated quality/storage pre-candidate.  It authorizes
physical H100 gates but does not establish latency or a general capability
claim.

