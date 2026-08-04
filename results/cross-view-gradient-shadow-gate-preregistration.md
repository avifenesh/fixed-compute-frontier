# Cross-view gradient filter — causal shadow preregistration

Status: frozen before the first H100 shadow run.

## Question

Can agreement between two independent microbatch gradients identify a useful
matrix-update direction that scalar magnitude changes and ordinary single-view
gradient energy cannot explain?

This is a no-candidate-update gate. The model follows ordinary AdamW. A
candidate filter built only from prior microbatch pairs is scored on the next
pair. Failure forbids a full candidate training arm.

## Algebra

For a matrix row (or column) group `i`, let `Ga` and `Gb` be gradients from two
disjoint size-32 halves evaluated at identical weights. Define

`C_i = E <Ga_i,Gb_i>` and
`T_i = E (||Ga_i||^2 + ||Gb_i||^2)/2`.

The diagonal coefficient minimizing symmetric cross-view prediction error is
`q_i=C_i/T_i`. The best scalar is `q=sum(C)/sum(T)`, and its excess population
risk is

`sum_i T_i (q_i-q)^2 >= 0`.

Under independent zero-mean gradient noise, `q_i` is signal energy divided by
signal-plus-noise energy. The filter uses the smaller matrix axis, fixed before
data. Empirical-Bayes shrinkage pulls every local ratio toward the scalar using
lagged EMA residual variance. The first 40 steps are statistics-only. All
ratios are clipped to `[0,1]`.

For a later averaged two-half update, half reliability `q` maps to
`a=2q/(1+q)`. Any eventual candidate would apply `a` to the Adam data update
and renormalize it to the exact original Frobenius norm before adding the
unchanged decoupled decay. Thus a common scalar cancels and only direction can
change. This shadow gate does not modify updates.

## Frozen protocol

- Retained H100 SXM 80 GB; high float32 matmul precision and TF32 enabled.
- The same 12-layer, width-384, FFN-1024, 37,758,336-parameter scratch model,
  seed 223, fixed FineWeb-Edu token files, and ordinary AdamW schedule used by
  the preceding 50M-token screens.
- 256 macrosteps. View A reads size-32 batches 0 through 255. View B reads
  size-32 batches 1,525 through 1,780. The sequence ranges are disjoint and
  separated by 40,608 sequences; the run must verify at least one inserted EOS
  document boundary in that gap. This prevents both views from containing
  chunks of the same source document. The independent-noise identity remains a
  population-sampling assumption, so the causal held-out score—not the identity
  alone—is the evidence gate.
- Both gradients are computed at identical weights with independent forward and
  backward calls. Filter statistics use unclipped gradients. The ordinary model
  update uses their mean and the frozen global clip.
- EMA beta `0.95`; filter seed 20,260,801; fixed smaller-axis selection.
- Score steps 41 through 256. Every score uses statistics through the previous
  macrostep, then current statistics are incorporated.
- Score symmetric held-out normalized MSE of raw half-gradient prediction and
  held-out cosine after applying the filter to hypothetical Adam half-updates
  formed from the frozen prior moments.
- Diagnose the candidate update angle using the real globally clipped mean
  gradient and frozen prior Adam moments; prediction statistics and half-view
  scores remain unclipped to avoid coupling the two views.
- Controls: causal global scalar, identity, four lagged energy-only diagonal
  filters (`T^-1/2`, `T^-1`, `T^1/2`, `T`), and a fixed-permutation shuffled
  cross-view estimator.
- Confidence intervals use a deterministic 5,000-draw moving-block bootstrap
  over macrosteps with block length 20. Matrices are not treated as independent
  samples.

## Frozen decision

Advance to a matched LM screen only if every gate passes:

1. aggregate held-out normalized MSE falls by at least 0.5% versus the causal
   scalar and its block-bootstrap difference interval is entirely below zero;
2. MSE improves in at least five of seven matrix families, with no family worse
   by more than 0.5%;
3. aggregate symmetric pseudo-Adam cosine rises by at least 0.002, its blocked
   interval versus identity is above zero, and cross-view beats identity in at
   least five of seven families;
4. cross-view cosine beats the best energy-only control in at least five of
   seven families with blocked interval above zero, and beats shuffled cross
   view in at least six of seven;
5. the hypothetical norm-preserved update has median `1-cos` from baseline at
   least `1e-4` in five of seven families, maximum relative norm error at most
   `1e-6`, at most 25% of raw local ratios hit a boundary, and filter fallback
   rate at most 1%;
6. all losses, gradients, statistics, ratios, and scores remain finite.

Passing proves only a causal directional prediction signal and authorizes a
four-arm LM screen: AdamW, dynamic-norm shadow of the rejected folded metric,
norm-preserved energy-only filter, and norm-preserved cross-view filter. It is
not a capability, compute-efficiency, or novelty result. Failure closes this
diagonal beta-0.95 cross-view filter without a full training run.

## Cost boundary

The proposed deployed model cost remains exactly ordinary dense serving. This
shadow run adds five FP32 statistics for each of 26,112 row/column groups
(522,240 bytes), 26,112 int64 permutation entries (208,896 bytes), a temporary
all-model first-view gradient snapshot (about 151 MB), and a second-view target
snapshot (75,497,472 bytes). Scoring also creates transient filter tensors. A
production candidate needs only optimizer statistics and one separated-gradient
snapshot during training; none enters the model state dict or inference graph.
