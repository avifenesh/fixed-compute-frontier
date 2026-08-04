# Phase-elastic v3: exact packed-artifact replay

## Purpose

The phase-faithful 10M screen passed every frozen gate and improved mixed-mode
suffix NLL by 0.5662%.  Its evaluation still used fake-quantized training
modules containing FP32 shadows.  This replay tests the served discrete object,
not another training proxy.

## Frozen replay

- deterministically recreate only the seed-6673 phase-faithful candidate with
  the already frozen 305-step stream;
- verify its pre-export phase grid reproduces the recorded discovery;
- quantize scales to BF16, encode the base as signed INT8 plus four packed
  ternary values per byte, and encode the branch as two signed INT4 values per
  byte;
- replace every FFN training weight module, deleting all shadow and log-scale
  parameters;
- serialize the whole model state, delete the model, construct an independent
  packed skeleton, strictly reload the artifact, and evaluate it;
- repeat the five-boundary mixed-mode grid and cached semantic checks.

The unpacking PyTorch module is a correctness oracle, not a latency kernel.  It
materializes decoded weights transiently and therefore cannot establish served
VRAM or speed.  Its persistent buffers must, however, match the exact byte
ledger.

## Frozen pass

1. pre-export aggregate suffix loss reproduces the discovery within `1e-7`;
2. every post-reload bucket loss equals its pre-export value within `1e-7`;
3. post-reload packed weights retain at least 90% of the discovery's relative
   suffix gain over the recorded dense arm, with a favorable paired interval;
4. post-reload first-token NLL remains noninferior within 0.05%, including the
   paired upper confidence bound;
5. every layer's persistent packed FFN payload is exactly 2,338,816 bytes;
6. no parameter or buffer name contains `shadow` or `log_scale` after reload;
7. strict artifact reload, INT8 `[-127,127]`, W4 `[-7,7]`, ternary code bounds,
   positive scales, cache semantics, data, finiteness, and protocol checks pass.

A pass authorizes the 50M phase-faithful replication.  It does not authorize a
same-cost serving claim before a physical packed kernel is measured.
