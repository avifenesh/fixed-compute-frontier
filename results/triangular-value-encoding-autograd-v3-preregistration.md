# TVE hybrid-autograd H100 gate v3

## Disclosed prior failure

V2 is a formal NO-GO. Its custom single-forward operator was 5.4x-5.7x faster
than the duplicate serving-plus-surrogate path and value-gradient errors stayed
inside the frozen contract, but active weight-gradient RMSE was `0.414`, `4.303`,
and `2.176` at the tail, 37.8M, and 360M geometries. The frozen limit was `0.05`.
No v2 tolerance is changed or reinterpreted.

The error came from using a different reduction tree for the token-summed
physical weight gradient. V3 changes the algorithm: serving-exact forward and
`dV` remain custom Triton, while `dO` uses the same BF16 torch matmul and
scatter algebra as the reference. This is a hybrid implementation, not a
tolerance rescue.

## Frozen untouched cases

- Tail: seed `7411`, shape `[2,2,65,64]`, query groups 3.
- 37.8M training: seed `8537`, shape `[32,2,512,64]`, query groups 3.
- SmolLM2-360M training: seed `9661`, shape `[8,5,512,64]`, query groups 3.
- BF16 values/upstream, FP32 physical output weights, block size 16, tau 0.125.

## Correctness gates for every case

1. Hybrid forward is bit-exact to the serving kernel/current reference path.
2. Active physical slot count is exactly 960 or 2,400 as specified.
3. The complete dense physical weight gradient is bit-exact to the reference;
   support outside active slots is therefore also exactly zero.
4. Value-gradient relative L2 error is at most `0.5%`, cosine similarity is at
   least `0.99998`, and maximum absolute error is at most `0.0625`.
5. All tensors and metrics are finite.
6. The new hybrid signed two-token closed-form unit test passes unchanged.

## Operator-cost gate

For the exact 37.8M and 360M training geometries, compare forward plus backward
against the existing duplicate serving-forward plus PyTorch-surrogate path.
Use 10 warmups and 40 randomized paired trials with CUDA events and no outlier
removal. In both cases:

- median paired hybrid/current time is at most `0.75`;
- hybrid peak allocated bytes do not exceed current peak bytes.

Passing admits the hybrid operator only to the separately frozen full-model
integration experiment. It does not prove full-model training throughput,
trajectory equivalence, 360M efficacy, or served inference cost.
