# TVE replay-autograd H100 gate v4

## Disclosed prior failure

V3 is a formal NO-GO. It passed every forward, value-gradient, finite-value,
support, signed-example, timing, and memory gate. It ran at `0.367x` and
`0.344x` the current duplicate path at the 37.8M and 360M geometries. However,
the complete dense physical weight gradient was not bit-exact in any of the
three frozen cases, so v3 is rejected.

The remaining mismatch is caused by replacing PyTorch autograd's broadcasted
matmul VJP and reduction tree with an algebraically equivalent manual matmul.
V4 changes the algorithm, not the tolerance: the serving-exact forward and
Triton value gradient remain, while physical `dO` is obtained by replaying only
the weight-dependent PyTorch surrogate graph inside backward. Values are
detached during replay, so this graph computes no duplicate value gradient.
The dense weight gradient must still be bit-exact.

## Frozen regression cases

The same cases that rejected v3 are deliberately reused as exact regression
tests:

- Tail: seed `7411`, shape `[2,2,65,64]`, query groups 3.
- 37.8M training: seed `8537`, shape `[32,2,512,64]`, query groups 3.
- SmolLM2-360M training: seed `9661`, shape `[8,5,512,64]`, query groups 3.
- BF16 values/upstream, FP32 physical output weights, block size 16, tau 0.125.

## Correctness gates for every case

1. Replay forward is bit-exact to the serving kernel/current reference path.
2. Active physical slot count is exactly 960 or 2,400 as specified.
3. The complete dense physical weight gradient and its active subset are
   bit-exact to the PyTorch reference; support outside active slots is zero.
4. Value-gradient relative L2 error is at most `0.5%`, cosine similarity is at
   least `0.99998`, and maximum absolute error is at most `0.0625`.
5. All tensors and metrics are finite.
6. The bound signed two-token closed-form unit test passes unchanged.

No correctness threshold from v2 or v3 is relaxed.

## Operator-cost gate

For the exact 37.8M and 360M training geometries, compare forward plus backward
against the current serving-forward plus full PyTorch-surrogate path. Use 10
warmups and 40 randomized paired trials with CUDA events and no outlier
removal. In both cases:

- median paired replay/current time is at most `0.75`;
- replay peak allocated bytes do not exceed current peak bytes.

Passing admits replay autograd only to the separately frozen full-model
integration experiment. It does not prove full-model throughput, trajectory
equivalence, 360M efficacy, or served inference cost.
