# T32R streaming H100 gate — positional-buffer residency erratum

Status: **FROZEN BEFORE IMPLEMENTATION OR TIMING**  
Date: 2026-07-31

The rank-32 algebra treated its deterministic sinusoidal position code as zero
parameters. That is correct for trainable-parameter counting but incomplete for
the physical HBM contract: a cached BF16 `128 x 32` code occupies 4,096 entries
or 8,192 bytes.

The H100 implementation must charge this buffer. The physical resident ledger
is therefore:

```text
trainable/fixed record-reader entries   943,538
BF16 positional buffer                    4,096
candidate matched slack                   20,046
total replacing dense FFN entries        967,680
```

No shape, algorithm, timing threshold, or scientific condition changes. The
source and timings did not exist when this correction was frozen. Baseline and
candidate total resident tensor bytes must match exactly, counting parameters
and persistent buffers.
