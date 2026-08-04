# BBCM matched learning screen decision

Decision: close exact `INT8 base + K4 ternary deltas` upcycling as a fixed-cost breakthrough candidate. Do not run the preregistered 640-step extension.

## Fixed-cost result

After 320 identical FFN-only updates on 10,485,760 prediction targets:

- continued BF16 M1536: NLL `2.8138115313`;
- exact-budget BBCM M1520: NLL `2.8277117815`, `0.4940%` worse;
- exact-budget independent K4-W4 M1520: NLL `2.9928197786`.

BBCM minus BF16 is `+0.01390025` NLL with paired 95% interval `[+0.01349016,+0.01431034]`. The gap is already positive at step 80. The extension gate is false.

The activation-aware M1520 BF16 endpoint alone costs `0.94498%` relative NLL: `2.8563108183` versus `2.8295718897`, paired interval `[+0.02609104,+0.02738681]`. The small proxy pays disproportionate metadata/alignment overhead: `16/1536=1.04%` width here versus `16/14336=0.112%` at the target geometry. The actual frozen proxy still fails its required comparison, so this closes the exact Smol upcycling recipe without proving target-scale impossibility.

## What did work

BBCM is much better than naive K4-W4: terminal advantage `0.165108` NLL, paired interval `[-0.167335,-0.162881]`. This validates shared high-precision structure as a better upcycling codec when route weights are extremely correlated.

Learned routing is also real, not a route-count artifact:

- learned BBCM NLL: `2.8277117815`;
- best of three count-preserving shuffled routes: `2.8291998319`;
- learned minus shuffled: `-0.00148805`, paired interval `[-0.00164763,-0.00132847]`.

That routing gain is only about `0.0526%` relative NLL and `10.7%` of the remaining dense gap, far too small to repay the fixed-budget damage. Global routes are balanced (minimum global load `23.80%`, median normalized layer entropy `0.9832`), but layers 0 and 29 each have a route below `0.1%`; the frozen global/median gate does not imply every layer is healthy.

## Load-bearing diagnosis

The selected matrices barely specialize:

- mean pairwise route-weight correlation: `0.99998948`;
- minimum correlation: `0.99998683`;
- mean route-delta/route-mean variance ratio: `7.89e-6`.

The representation stays safely inside the high-correlation regime where a shared base should beat W4, but it creates too little conditional variation to become a better model. Increasing delta/router learning rates or extending this one seed would tune around a decisive miss and is not authorized.

## Scope

This closes the exact SmolLM2 upcycling candidate, not every possible conditional-weight model. It does not establish a runtime result: codes were fake-quantized and decoded, not physically packed or fused. The useful retained finding is narrower:

> Conditional routing can extract a small statistically clear gain from almost-identical quantized matrices, but storing full route-specific bitplanes is a poor exchange. The next architecture should create conditional functions by reinterpreting one shared matrix or activation basis, without paying dense per-route storage or width.

Result SHA-256: `f3c4f8b328e8498570a95520d31ce77824d38282829f54a8ce54c3885718dfd2`.
