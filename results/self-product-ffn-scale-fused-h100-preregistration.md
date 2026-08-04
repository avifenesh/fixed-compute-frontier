# Self-product scale fused H100 preregistration

## Fixed question

Was the eager H100 failure caused by materialized activation temporaries, or
does the self-product algebra still increase served cost after both candidate
and SwiGLU receive fair in-place activation fusion?

## Fixed implementation

- Same D640/M1792/R2688/L16 models and resident BF16 weights.
- Baseline Triton kernel overwrites gate with `SiLU(gate) * up`; it materializes
  the two projection banks but no third activation output.
- A second exact SwiGLU baseline concatenates gate/up weights into one
  `D -> 2M` input projection and applies a strided in-place Triton kernel. The
  candidate must pass latency and peak gates against both the split and packed
  baseline, which is equivalent to dominating their measured Pareto envelope.
- Candidate Triton kernel overwrites its sole generator bank with
  `generator^2 * sigmoid(generator)`.
- Wide-SiLU control also uses an in-place unary kernel.
- Kernels add no parameters or persistent buffers. Each fused model must match
  the corresponding eager folded prompt logits, every prompt K/V cache tensor,
  next-token logits, and every updated next-token K/V cache tensor within the
  frozen `atol=0.03`, `rtol=0.02` BF16 tolerance. These checks are embedded in
  the result and are mandatory for a serving pass.
- The candidate self-product and wide-SiLU unary kernels must also be bitwise
  equal to eager PyTorch over all 65,280 finite BF16 inputs. The fused SwiGLU
  kernel must be bitwise equal over all finite gate inputs paired by the frozen
  rotation 7,919. This exhaustive activation ledger is embedded and mandatory.
- The failed eager result/source/preregistration/test hashes are immutable
  dependencies.
- The untouched confirmation result, scale data manifest, and complete future
  scale-admission code/preregistration/test are hash-frozen before this run.

## Frozen gate

- Same resident-BF16 H100/runtime, three prefill cells, two fixed-context
  cached-decode cells, five warmups, 30 interleaved samples, wall/CUDA clocks,
  5,000-replicate bootstrap, and isolated decoder/service memory accounting as
  the eager test.
- Candidate median ratio must be at most 1.0 and bootstrap upper 95% ratio at
  most 1.02 in every latency cell/clock.
- Candidate absolute and incremental allocated/reserved peaks must be no
  greater than the fused SwiGLU baseline in every prefill scope.
- Static BF16 parameters/buffers must be equal across the candidate, controls,
  split SwiGLU, and packed SwiGLU.

This test may rescue the serving implementation. It cannot erase the eager
failure. If it fails, no scale LM training and the branch closes.
