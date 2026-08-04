# Projection-coalesced attention/FFN — unfused H100 diagnostic

This is an exploratory whole-model timing diagnostic, not a preregistered
latency claim.  It was run on the retained Vast H100 SXM 80 GB instance with
PyTorch 2.5.1+cu124 and Transformers 4.57.6 after five warmups.  Each cell is
the median of 15 synchronized BF16 inference forwards of the 12-layer scratch
model, with no KV cache and no compilation.  Arms were not randomized or
interleaved.

| batch x tokens | sequential baseline | parallel baseline | coalesced 1024 | coalesced 1344 |
|---|---:|---:|---:|---:|
| 1 x 1 | 6.809 ms | 5.992 ms | 5.072 ms | 5.063 ms |
| 1 x 512 | 7.265 ms | 6.444 ms | 5.399 ms | 5.469 ms |
| 8 x 512 | 7.904 ms | 7.001 ms | 5.908 ms | 6.360 ms |
| 32 x 512 | 11.476 ms | 10.510 ms | 9.447 ms | 9.992 ms |

At these exact cells, the width-1,344 candidate is 12.9% to 25.6% faster than
the sequential baseline and 4.9% to 15.5% faster than the independent parallel
control.  The timing includes the current unfused `clone`, slice, and scatter
operations.

Interpretation is deliberately narrow: the candidate is not obviously killed
by GPU execution at the scratch shape.  This does not measure autoregressive
decode with a populated KV cache, an optimized baseline server, compiled
graphs, H200, tensor parallelism, traffic, energy, or confidence intervals.
