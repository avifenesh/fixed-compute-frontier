# Hard-gated self-product: frozen H100 refinement

## Algebraic change

The retained self-product activation is

`f(z) = z * SiLU(z) = z^2 sigmoid(z)`.

Replace only its sigmoid gate with

`q(z) = clamp(1/2 + a z, 0, 1)`

and serve

`h(z) = z^2 q(z)`.

The gate obeys `q(-z) = 1-q(z)`, so for any symmetric zero-mean preactivation distribution both `f` and `h` have mean `E[z^2]/2`. The frozen slope `a = 0.14709223807891283` numerically minimizes `sup_z z^2 |q(z)-sigmoid(z)|`; its cutoff is `1/(2a) = 3.3992276311` and the measured supremum error is `0.373444651`. No slope or width sweep is allowed.

For standard-normal preactivations, 200-point Gauss-Hermite quadrature gives second moments `E[f(z)^2] = 1.1901360381` and `E[h(z)^2] = 1.0678826085`. The frozen deployment-fold scale is therefore `0.47128177407143074`, preserving the prior self-product candidate's initial second moment after its scale `0.44642046792894413`.

This changes the expensive transcendental sigmoid into BF16 multiply, add, clamp, square, and multiply. Width and dense projections remain the preregistered latency-matched values: `D=640`, SwiGLU `M=1792`, candidate `R=2560`, `L=16`.

## Frozen serving protocol and decision

- same H100/runtime, seed 271828, 5 warmups, 30 randomized repetitions, cells, clocks, and 5,000 bootstrap resamples as `self-product-ffn-latency-matched-h100-preregistration.md`
- compare against both split and packed fused SwiGLU
- candidate median latency ratio must be `<=1.00` and bootstrap upper 95% bound `<=1.02` for every cell and clock
- exhaustive finite-BF16 activation equivalence and full-model eager/fused semantic equivalence must pass
- static parameter bytes must be strictly lower than both baselines
- peak allocated bytes and allocated increment must be no greater than both baselines in every prefill cell and scope
- CUDA allocator reserved bytes are reported but diagnostic: the preceding isolated measurement showed allocator-bin reservation can rise while both live allocation and static weights fall

All gates must pass to authorize a language-model quality trial. A serving pass alone is not a capability result.
