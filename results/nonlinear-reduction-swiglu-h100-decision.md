# Associative nonlinear K-reduction H100 decision

## Decision

**Close the frozen fused split-K executor before language-model training. Retain
the algebra, but do not inject it into a custom GEMM mainloop under the current
fixed-service-cost claim.**

The associative dual-number reducer passed its correctness, exact-ledger, and
activation-memory gates. It failed both latency gates. The failure is not a
small kernel-tuning ambiguity: the candidate was slower than the fastest
ordinary packed SwiGLU path at every frozen row count.

## Frozen experiment

- H100 SXM 80 GB; BF16 inputs, weights, activations, and outputs with FP32
  accumulation.
- `D=4096`, `M=14336`, four 1,024-wide K blocks, `alpha=0.05`.
- Full FFN timing includes the nonlinear up/gate computation and the down
  projection.
- Controls: the same custom fused kernel with the nonlinear update disabled,
  ordinary split cuBLAS SwiGLU, and ordinary packed-input cuBLAS SwiGLU.
- 100 raw samples per cell and 5,000 paired-bootstrap resamples.

| Rows | candidate ms | custom ordinary ms | packed ordinary ms | candidate / fastest ordinary |
|---:|---:|---:|---:|---:|
| 1 | 0.158752 | 0.163344 | 0.139936 | 1.1345 |
| 8 | 0.160000 | 0.164640 | 0.143744 | 1.1131 |
| 32 | 0.173152 | 0.172320 | 0.147536 | 1.1736 |
| 128 | 0.475440 | 0.326176 | 0.165808 | 2.8674 |
| 512 | 1.659248 | 1.571712 | 0.293376 | 5.6557 |
| 2,048 | 6.622464 | 6.778720 | 1.035744 | 6.3939 |

The candidate also failed the custom-disabled envelope at rows 128 and 512.
Its one-call incremental activation peak was lower than both ordinary arms in
every cell: one `M`-wide activation instead of the ordinary `2M`-wide pair.

Correctness passed. Relative L2 error was `0.00349783` for the disabled target
path against ordinary SwiGLU and `0.000114762` for the associative target path
against the reference, both below their preregistered tolerances. The static
ledger was exactly 176,160,768 BF16 scalars / 352,321,536 bytes.

## Retained boundary

The reducer remains a valid algebraic primitive:

\[
(p,q)\star(g,u)=
\bigl(p+g+apg,\ q+u+a(pu+qg)\bigr).
\]

It is associative and commutative, exactly contains ordinary SwiGLU at
`a=0`, and added one sampled local function direction in the Stage-0 witness.
What failed is its placement: forcing an altered reduction into the K loop
forfeits the highly optimized GEMM path. The next candidate must modify a
reduction whose state is already exposed in a production kernel, so its new
algebra is paid by a few scalar/vector operations rather than a replacement
matmul executor.

This result closes only the frozen executor, shape, blocking, coefficient, and
hardware. It does not prove that the algebra is useless under every possible
implementation.

## Integrity

- result SHA-256:
  `cc9965510ebbfb6f0e42a4d976972488fc41727c7a790685098d140472b23012`
- log SHA-256:
  `5eaa98807a2b0855ae063341f6f2f56ecd17133cd955d26811604eb3020474c9`
- source SHA-256:
  `b19ce2e5b4dd28eb69c602109eac02a55bcfb2ac58bd7ee444969365b0d6a3c3`
- preregistration SHA-256:
  `9a76d66108e0ec1c8425fb150ce0ed9b27599ee75535ab402a559fcdc625ccd0`

An independent review recomputed every median and bootstrap interval, checked
all timing vectors, correctness gates, dtype/byte ledgers, and reconstructed
the final `h100_pass=false` decision with no discrepancy.
