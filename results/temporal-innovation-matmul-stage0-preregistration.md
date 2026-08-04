# Temporal-innovation dense projection — Stage 0 preregistration

Status: **frozen before model download or activation measurement**  
Date: 2026-07-26  
Stage: CPU-only algebra and frozen-model slack gate; no GPU rental

## Claim under test

For an arbitrary learned dense projection `W`, consecutive outputs need not be
recomputed from zero:

\[
y_t = W x_t = y_{t-1} + W\Delta x_t,
\qquad \Delta x_t=x_t-x_{t-1}.
\]

If `Delta x_t` has `k` nonzero coordinates, the update reads `k` columns of
`W`, not all `D` columns.  The learned matrix remains arbitrary and full rank;
unlike the rejected feature-DAG branch, this identity does not restrict the
matrix's degrees of freedom.

The proposed edge is decode compute and weight traffic, not more learned bytes.
It is useful only if language-model activations have sparse temporal innovation
or can be event-quantized with negligible distortion.  Prefill remains dense
and parallel.  This gate does not claim an H100 speedup.

## Exact algebra and the propagation boundary

The executable self-test uses a random dense `W` and a 5%-sparse innovation.
Incremental and from-scratch outputs must agree to relative L2 error below
`1e-6` in FP32.

RMSNorm does not invalidate the identity.  For

\[
r_t=\operatorname{rsqrt}(D^{-1}\lVert x_t\rVert_2^2+\epsilon),
\qquad z_t=W(\gamma\odot x_t),
\]

we have

\[
W\operatorname{RMSNorm}(x_t)
=\frac{r_t}{r_{t-1}}W\operatorname{RMSNorm}(x_{t-1})
+r_t W(\gamma\odot\Delta x_t).
\]

The normalized incremental path must also agree below `1e-6`.

There is a hard exact-composition boundary.  For any fixed nonzero innovation
and a matrix whose entries come from a continuous distribution, every element
of `W Delta x` is nonzero with probability one: each zero output is one
measure-zero hyperplane.  A sparse event therefore fans out into a dense change
after one generic dense projection.  SiLU and other smooth nonlinearities with
nonzero derivative do not restore exact zeros.  Exact temporal sparsity cannot
propagate through a stack of unrestricted dense maps.

Consequently the broad exact replacement is already rejected algebraically.
The empirical gate asks only whether a deadband/quantized escape has enough
pre-existing slack to deserve joint training.

## Prior-art collision

This is not presented as a novel family.  Sigma-Delta networks already send
quantized activation changes between layers; Delta Activation training and
event-driven recurrent networks explicitly promote temporal sparsity.  Recent
language-model work also trains activation sparsity.  Their existence is a
collision check, not evidence that the method works for this workload.

## Frozen model and panel

- checkpoint: `HuggingFaceTB/SmolLM2-360M` at revision
  `f8027fd0eaeea54caa13c31d31b9fdc459c38b49`;
- checkpoint identity must match the existing baseline capture:
  `model.safetensors` SHA-256
  `7aaff6661428bed033abba9522bec81938678642cca3181fe752b6ca9e1e540f`;
- CPU only, FP32 model execution, deterministic seed `260726`;
- four fixed 192-token lanes: prose, dialogue, code, and deterministic random
  token IDs;
- one batched causal forward pass with no generation;
- sampled layers are frozen as the first, quarter, middle, three-quarter, and
  last decoder layers after the checkpoint config is read;
- 16 evenly spaced adjacent-token transitions per lane and projection.

The inspected linear projections are `q_proj`, `k_proj`, `v_proj`, `o_proj`,
`gate_proj`, `up_proj`, and `down_proj`.  A forward pre-hook observes each
projection's actual input.  The attention group is Q/K/V/O; the FFN group is
gate/up/down.  In particular, the input to `down_proj` measures whether sparse
innovation survives the FFN's dense up/gate fan-out and nonlinearity.

## Frozen measurements

For every sampled projection and transition, report:

1. exact changed-coordinate fraction (`Delta x != 0`);
2. relative projected-delta error after retaining the largest 10%, 25%, and
   50% input changes by magnitude;
3. for symmetric per-coordinate 8-, 6-, and 4-bit quantization, the fraction
   of coordinates whose integer code changes between adjacent tokens;
4. relative output L2 distortion of each quantized activation under the real
   projection weight.

Quantizer scale is an oracle scale fitted from the entire four-lane activation
tensor seen by that module.  This intentionally favors the candidate.  It is
not a deployable calibration claim.  Metrics are aggregated by projection,
layer, lane, and the attention/FFN groups; median and p90 are binding.

## Frozen decision

The exact route has `free_exact_reuse=true` only if both attention and FFN
groups have median changed-coordinate fraction at most `0.10` and p90 at most
`0.20`.  Continuous dense activations are expected to fail; the threshold is
retained to prevent changing the claim after observation.

A bit-width has `event_quantized_slack=true` only if, in both groups:

- median changed-code fraction is at most `0.10` and p90 at most `0.20`;
- median relative projected-output distortion is at most `0.01` and p90 at
  most `0.02`.

Additionally, the raw top-10% innovation path must have median relative
projected-delta error at most `0.05` and p90 at most `0.10` in both groups.
This is a second, independent check that the favorable code-change count is
not just a quantizer artifact.

Possible decisions:

- `reject_exact_and_pretrained_slack`: exact composition is impossible and no
  measured quantized path passes;
- `joint_training_hypothesis_only`: exact composition is impossible but at
  least one measured quantized path passes all slack gates.

Neither outcome promotes a model candidate or justifies GPU rental.  A passing
slack result would justify a separately preregistered, equal-budget tiny
language-model training gate.  A failure closes temporal reuse as a free
transformation of ordinary dense LLMs; it does not prove that explicit
event-sparsity training is impossible.

## Physical ledger required in the result

The result must report the optimistic logical weight-read fraction and all
additional per-request state needed to keep previous projection inputs and
outputs.  It must state that arbitrary column gathers are not Tensor-Core GEMM,
that supports differ across requests, and that only a direct H100/H200 kernel
comparison can establish latency or cost.  No logical operation count may be
called a speedup.
