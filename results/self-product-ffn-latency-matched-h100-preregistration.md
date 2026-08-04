# Latency-matched self-product FFN: frozen H100 serving gate

## Question

Can the repeatedly favorable self-product nonlinearity pay for a smaller FFN that is no slower and uses less served state than ordinary SwiGLU?

This is a new, single-point cost rematch of the retained self-product result. It is not a repair of the failed equal-dense-MAC claim and it is not a width sweep.

## Frozen shapes

- hidden size `D = 640`
- layers `L = 16`
- SwiGLU intermediate width `M = 1792`
- self-product and wide-SiLU width `R = 2560`
- attention heads `10`, KV heads `2`, head dimension `64`
- BF16 resident weights and activations

The candidate width was selected before this run from the frozen worst prior packed-baseline ratio at width 2688: `floor_to_128(2688 / 1.0424) = 2560`. No other width will be measured in this gate.

Per layer, the baseline FFN has `3DM = 3,440,640` dense weights/MAC coefficients. The candidate has `2DR = 3,276,800`, which is 4.7619% lower. Its activation kernel evaluates 2,560 SiLUs and multiplies per token rather than 1,792, which is 42.8571% more elementwise work. The full measured static parameter ledger, rather than this analytic estimate, is authoritative.

## Implementations

- split SwiGLU: separate gate and up projections, in-place fused activation, down projection
- packed SwiGLU: one packed `D -> 2M` projection, strided in-place fused activation, down projection
- wide SiLU control: one `D -> R` projection, in-place fused SiLU, down projection
- self-product candidate: one `D -> R` projection, in-place fused `SiLU(z) * z`, down projection

The candidate must be compared against both SwiGLU implementations. The wide-SiLU arm is diagnostic only.

## Frozen timing protocol

- device must contain `H100`
- runtime: Torch `2.5.1+cu124`, CUDA `12.4`, Transformers `4.57.6`, Triton `3.1.0`
- seed `271828`
- 5 warmups, 30 randomized-order repetitions
- prefill cells: `(batch, tokens) = (1,512), (8,512), (32,512)`
- decode cells: `(batch, cached context + one token) = (1,512+1), (8,512+1)`
- both wall-clock and CUDA-event timing
- 5,000 bootstrap resamples of the median ratio

For every cell, clock, and baseline, the candidate must have median ratio `<= 1.00` and bootstrap 95% upper bound `<= 1.02`.

## Frozen correctness and memory gates

- fused activation must be bitwise equal to the PyTorch BF16 expression for every finite BF16 value (and the fixed SwiGLU pairing)
- fused and eager full-model prefill logits, next-token logits, and KV caches must meet the already frozen tolerances (`atol=0.03`, `rtol=0.02`)
- candidate static parameter bytes must be strictly lower than both SwiGLU baselines
- for every prefill cell and both decoder-core and service-last-token scopes, candidate peak allocated/reserved bytes and their increments must be no greater than both baselines

The serving gate passes only if all semantic, latency, static-size, and peak-memory checks pass. A pass authorizes one long-horizon language-model comparison at the same frozen shapes. It does not by itself establish a quality gain.
