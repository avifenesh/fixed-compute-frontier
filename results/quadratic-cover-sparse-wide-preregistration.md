# Quadratic-cover sparse-wide SwiGLU: frozen 10M screen

## Claim under test

Replace each dense width-1,024 SwiGLU by a width-1,792 SwiGLU whose gate,
up, and down matrices obey the hardware-native 2:4 pattern.  The candidate is
not claimed fast from an abstract sparsity count; this screen tests quality
first.  A physical sparse H100 kernel is authorized only after a quality pass.

The candidate exchanges dense fan-in for 75% more nonlinear features.  For
BF16 values and four metadata bits per group of four weights, its per-layer
resident FFN payload is

`3 * 384 * 1792 * (2 BF16 values * 16 bits + 4 metadata bits) / 4`

or 2,322,432 bytes, versus 2,359,296 bytes for dense width 1,024.  It executes
1,032,192 nonzero multiply-accumulates per token, 87.5% of the dense
baseline's 1,179,648.  Runtime, activation traffic, workspace, and padding are
not inferred from these counts.

## Mask algebra

For each consecutive group of four input coordinates, a gate mask `A` and an
up mask `B` each select two coordinates.  The leading SwiGLU term contains the
quadratic support `(a^T x)(b^T x)`.  A frozen 1,792-row code assigns `(A,B)` so
that:

- every coordinate occurs in exactly 896 gate masks and 896 up masks;
- every one of the four squares and six unordered cross terms is supported by
  exactly 640 rows, or 5/14 of all rows;
- each row remains exactly 2:4 sparse in both projections.

The primary control uses independent random 2:4 gate/up masks at the identical
width, nonzero count, down-mask construction, initialization scales, and
serving ledger.  It isolates the mask code from the generic sparse-wide idea.

Gate and up nonzero weights use `sqrt(2)` times the baseline initialization
standard deviation, restoring each projection's variance after 50% masking.
Down nonzero weights use `sqrt(2*1024/1792)` times the baseline standard
deviation, restoring aggregate FFN output variance rather than granting the
wider arm a larger residual update.

## Frozen run

- H100; Torch `2.5.1+cu124`; CUDA `12.4`; Transformers `4.57.6`
- scratch parallel-attention model: `D=384`, dense `M=1024`, `L=12`
- arms: independent-random sparse-wide, quadratic-cover sparse-wide
- frozen dense comparison: the seed-815 full-SwiGLU arm from the cycle-factor
  screen
- seed 815; same token stream, optimizer batches, and validation batches
- sequence 512, microbatch 32, accumulation 2
- 305 optimizer steps = 9,994,240 prediction tokens
- AdamW `3e-4`, 30-step warmup, betas `(0.9,0.95)`, weight decay `0.1`, clip
  `1.0`
- 128 fixed validation batches at initialization and step 305

Training uses dense masked shadows because the run is a quality screen.  The
served ledger counts only compressed nonzeros and metadata.  No latency claim
is allowed from this emulator.

## Frozen decisions

The sparse-wide direction advances only if at least one sparse arm beats dense
full SwiGLU by 0.05% relative NLL with a wholly favorable paired 95% interval.

The quadratic-cover refinement advances over generic sparse-wide only if it
also beats the independent-random sparse arm by 0.05% with a wholly favorable
paired interval.  A 10M pass authorizes one untouched 50M replication.  Only a
50M pass authorizes a packed sparse H100 serving gate.

