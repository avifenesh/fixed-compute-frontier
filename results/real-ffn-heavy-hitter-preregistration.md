# Real FFN heavy-hitter prerequisite

Status: frozen before H100 execution  
Date: 2026-07-27

## Candidate

For an FFN with `M` learned feature detectors, do not evaluate all `M`
dot-products.  Precompute a sparse-recovery measurement matrix of the detector
bank, evaluate `r = O(k log M)` measurements, recover the identities of the
`k` heavy neurons, and execute only their exact key/value rows.

This is useful only if trained FFN responses are strongly compressible.  The
first gate therefore measures the native SwiGLU activations of the immutable
checkpoint `HuggingFaceTB/SmolLM2-135M` at revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.

## Frozen workload

- layers: `0`, `14`, `29`
- dataset: streamed WikiText-103 training text
- requested tokens: `4096`
- sequence length: `256`
- measured active counts: `16, 32, 64, 96, 128, 256`
- candidate operating point: `k=96`, or `6.25%` of the 1,536-neuron FFN

Two selectors are measured:

1. an oracle using the exact SwiGLU activation times the down-projection column
   norm; and
2. a gate-only selector using `abs(SiLU(gate))`, which is the prerequisite for
   recovery from a linear sketch of gate preactivations.

For each token, the selected neurons reconstruct the actual down-projection
output and report relative L2 error.

## Promotion rule

At `k=96`, every tested layer must satisfy all four conditions:

- oracle median output error at most `5%`;
- oracle p90 output error at most `10%`;
- gate-selected median output error at most `7.5%`;
- gate-selected p90 output error at most `15%`.

Failure rejects sketch-decoder and custom-kernel work for native SmolLM2
SwiGLU.  It does not reject training a deliberately sparse FFN from scratch,
which would be a different hypothesis and must beat Spark/MoC/memory-layer
controls.  Passing permits only a learned-sketch recovery gate; it does not
establish model quality or GPU speed.
