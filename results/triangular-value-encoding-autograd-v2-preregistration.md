# TVE custom-autograd H100 gate v2

## Disclosed development result

The first isolated custom-backward test was a formal NO-GO. Forward and the
closed-form signed two-token case passed, but 2 of 960 active weight-gradient
slots exceeded the frozen elementwise tolerance. The failure was consistent
with reduction-order drift: weight-gradient relative L2 error was 0.2542%,
cosine similarity 0.99999678, and RMSE 0.03134. Value-gradient relative L2 was
0.3485% with cosine 0.99999392. No tolerance from that run is changed or
reinterpreted.

V2 tests a newly specified numerical contract on untouched seeds and exact
training geometries. It treats the current BF16 PyTorch surrogate as a numerical
reference, not mathematical truth; tensor-core reduction order is allowed to
differ while algebra, support, scale, and direction remain tightly bounded.

## Frozen cases

- Tail case: seed 3181, values `[2, 2, 65, 64]`, query groups 3.
- 37.8M training case: seed 4261, values `[32, 2, 512, 64]`, query groups 3.
- SmolLM2-360M training case: seed 5347, values `[8, 5, 512, 64]`, query groups 3.
- All values are BF16, physical output weights are FP32, block size is 16, and
  tau is 0.125.

## Correctness gates for every case

1. Custom forward is bit-exact to the serving kernel.
2. Gradient support outside the 960 or 2,400 reused strict-lower physical slots
   is exactly zero.
3. Value-gradient relative L2 error is at most 0.5%, cosine similarity is at
   least 0.99998, and maximum absolute error is at most 0.0625.
4. Active weight-gradient relative L2 error is at most 0.5%, cosine similarity
   is at least 0.99998, and RMSE is at most 0.05.
5. All tensors and metrics are finite.

The existing exact signed two-token closed-form test must also pass unchanged.

## Exploratory operator-cost gate

For the two exact training geometries, time forward plus backward for the
current duplicate serving-plus-surrogate path and the custom single-forward
path. Use 10 warmups and 40 randomized paired trials with CUDA events and no
outlier removal. The custom median must be no more than 75% of the current path
in both geometries and must not increase peak allocated bytes.

This cost gate admits the operator for integration work only. It does not prove
full-model training overhead, optimizer-step equivalence, loss-trajectory
equivalence, or served inference cost.
