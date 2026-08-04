# Orbit-activated SwiGLU carrier-chain — H100 decision

Decision: **reject the per-neuron decoded group-of-eight executor before
language training.**

The algebra and BF16 carrier remained valid, but the frozen full-FFN latency
gate failed.

| Tokens | Baseline median | Candidate median | Ratio |
|---:|---:|---:|---:|
| 1 | 141.728 us | 146.496 us | 1.03364x |
| 8 | 144.800 us | 149.568 us | 1.03293x |
| 32 | 150.624 us | 154.752 us | 1.02741x |
| 128 | 165.920 us | 182.784 us | 1.10164x |

The frozen key-cell limit was `1.02x`; the full-grid limit was `1.05x`.
Both failed.  This benchmark used the same separate gate/up GEMMs in both
arms, so it is a matched feasibility result, not parity with a more optimized
production coalesced-projection baseline.

## Correctness retained

- Zero-carrier output matched baseline exactly in all four cells.
- Nonzero fused output differed from the Torch reference by only `0.1679%`
  maximum row-relative error.
- BF16 carrier maximum absolute decode error was `0.0004883`; no material
  coefficient changed sign or collapsed to zero.
- Carrier scales stayed in `[0.7200,1.2799]`, with maximum reciprocal/direct
  condition factor `1.3889`.
- Candidate folded-BF16 export error exceeded ordinary folded-BF16 export by
  only `0.00503%` row-relative.
- No duplicate coefficient tensor or extra activation-sized workspace was
  used.

## Why this executor failed

The kernel grid is token by feature-group.  It issues one strided semantic
`U`-pivot load per token-feature: `B*M`, not `M`, load instructions.  At
`B=128,M=14336` that is `1,835,008` pivot loads.  It also executes seven serial
predecessor steps per group.  Cache can reduce physical HBM transactions but
cannot remove those instructions or dependencies.

Do not train, tune, or merely cache this exact carrier-chain recipe.  The
gauge-activation principle remains open only if the redundant scales affect
the computation through activations already in registers, without per-feature
decoding or a serial chain.

Result SHA-256:
`94401000859831f245ac87678a6c1ef6bcfa03f469f38e4dd28fab7761e17a39`.

