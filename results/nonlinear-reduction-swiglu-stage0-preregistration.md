# Nonlinear-reduction SwiGLU — Stage 0 preregistration

## Fixed question

Can a dense FFN retain and nonlinearly update the partial accumulators that its
input GEMM already computes, thereby obtaining new identifiable functions
without adding a dense MAC, a learned parameter, a persistent byte, an output
feature, or a third accumulator?

This stage tests algebra and accounting only.  It makes no language-quality,
kernel-speed, or novelty claim.

## Frozen algebra

Partition the input axis into `k` consecutive blocks.  For packed SwiGLU gate
and up weights, keep the ordinary two output accumulators `p` and `q`:

```
p = 0
q = 0
for b in 0..k-1:
    p = p + G_b x_b
    q = q + U_b x_b
    q = q * (1 + alpha_b * clamp(p, -2, 2))
z = SiLU(p) * q
y = C z
```

At `alpha_b = 0`, this reduces exactly to ordinary packed SwiGLU.  A future
kernel must mutate `p` and `q` only at K-block boundaries while both remain in
registers; it may not materialize block partials.

For stability, `alpha_b = 0.25*tanh(r[j_b]-1)`.  Here `r` is the existing
post-attention RMSNorm weight and `j_b` is one selected coordinate per block.
The selected norm coordinates are gauge-fixed to one at conversion time while
the corresponding gate/up input columns are multiplied by their old norm
weights.  This preserves the pretrained endpoint and makes every alpha zero.
No alpha tensor is stored.

## Frozen probe

- Double precision, seed 43.
- `D=4`, `M=3`, output width 4, `k=2`, selected norm coordinates `(0,2)`.
- 64 deterministic Gaussian probe inputs.
- One raw parameter vector contains the existing RMSNorm, gate, up, and down
  tensors only: `D + 3DM = 40` scalars in both arms.
- Compute the sampled functional Jacobian of all outputs with respect to all 40
  raw parameters at the exact shared endpoint (`r=1`, hence `alpha=0`).
- Rank tolerance is `max(shape) * eps(float64) * largest_singular_value`.

## Frozen gates

Stage 0 passes only if all gates pass:

1. Baseline and candidate outputs are exactly equal at alpha zero within
   `atol=rtol=1e-12` on the 64 probes.
2. Gauge conversion of a separate non-unit norm vector preserves ordinary
   SwiGLU and yields candidate alpha zero within `atol=rtol=1e-12`.
3. Both arms expose exactly 40 raw parameters, `3DM` dense projection weights
   and MACs, two live packed-input accumulators, and an `M`-wide down input.
4. The candidate sampled functional-Jacobian rank is strictly larger than the
   baseline rank.  Singular values and the tolerance are reported.
5. A directional finite-difference derivative along every broken norm/column
   gauge direction agrees with the candidate analytic JVP within `1e-6`
   relative error.  Because relative error is undefined at the baseline's
   exact zero, its analytic and finite-difference derivative norms and their
   maximum absolute discrepancy must each be at most `1e-8`.  The candidate
   derivative norm must be at least `1e-6`.
6. All values, derivatives, and singular values are finite.

Passing authorizes only a fused H100 mainloop-boundary test against both the
same kernel with recurrence disabled and the strongest packed SwiGLU path.  It
does not authorize language training.  Failure closes this exact recurrence.
