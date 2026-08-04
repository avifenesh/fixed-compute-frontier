# Gauge-zero G1 H100 one-row confirmation preregistration

Status: frozen after formal v1 failed only its one-row uncertainty gate and
before this confirmation is run  
Date: 2026-07-27

Formal v1 is immutable at SHA-256
`b92660a47e9aef6bd6e2f62dcb0ce28111f2f76dffc1a155712306e0ad063064`.
It passed BF16 correctness, all median ratios, the full-grid 1.03 bootstrap
upper gate, rows-8 decode, weight bytes, and zero-metadata accounting. It
failed only because the rows-1 bootstrap upper ratio was 1.02946 against 1.02,
although its median ratio was 1.01445.

This is a narrow independent-seed uncertainty resolution, not a retroactive
rewrite of formal v1 and not a kernel change.

Frozen contract:

- exact same hashed H100 source, attention source, tests, runtime, shape,
  normal `torch.mm`, fused RoPE control, G1 kernel, cold-L2 method, and BF16
  oracle as formal v1;
- rows: 1 only;
- trials: 4,000 matched randomized cold-weight trials;
- warmups: 50;
- seed: 6,829, unused by formal v1;
- bootstrap: the same 5,000 matched median-ratio resamples;
- thresholds unchanged: median ratio at most 1.02 and bootstrap 95% upper
  ratio at most 1.02;
- candidate and control must remain independently BF16 bit-exact and pivot
  metadata must remain zero bits.

All gates must pass. Failure closes this executor under the 2% one-row claim.
A pass resolves only formal v1's uncertainty and permits the LM quality pilot;
it still does not establish a fastest production baseline or model-quality
gain.
