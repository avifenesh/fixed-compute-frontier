# Scale-funded GECK-G1 teacher-free energy retrieval — preregistration

## Fixed claim under test

Can one square per RoPE pair, funded by an existing full-gauge pivot
coordinate, learn sign-invariant record-specific content addressing that an
ordinary head cannot learn, without adding a learned scalar, a Q/K matmul, or
a KV-cache coordinate? Centering is retained as a control, not used by the
primary candidate.

This gate tests an attention-head capability and its structural controls. It
does not establish language-model quality or an inference-cost win.

## Frozen algebra and parameter contract

Every arm is one 8-dimensional head with four ordinary RoPE pairs, base
10,000. Query input has two one-hot types. Key input is `(x, y, 0)`, so the
third coordinate is identically zero.

Every arm trains exactly 40 FP32 scalars / 160 serialized bytes:

- physical Q content map: `8 x 2 = 16`;
- physical K content map: `8 x 2 = 16`;
- four physical K pivot pairs, trained in polar coordinates: `4 x 2 = 8`.

For pivot pair `p = tau exp(s) (sin(phi), cos(phi))`, where
`tau = 1/sqrt(8)`, compilation applies the exact ordinary-attention gauge

`K_c = exp(-s) R_phi K`, `Q_c = exp(s) R_phi Q`.

This preserves every ordinary norm-free RoPE logit. The primary arm then uses

`u' = u`,

`v' = v + tanh(s) * u^2`.

The nonlinear-off slice is
`s=0`, which exactly contains the bilinear arm for arbitrary `phi`.

The primary adds one square, one coefficient multiply, and one add per RoPE
pair per token. It adds no trainable parameter, dense matmul, or KV-cache
value. The later kernel gate must count coefficient access or preparation from
the existing pivot weights; no uncredited sidecar constant is assumed here.

## Frozen teacher-free task

Each bag has four records with iid standard-Gaussian `(x,y)` content. Query
type 0 asks for the record with the largest `x^2`; type 1 asks for the largest
`y^2`. The target is the hard record index and is computed directly before
model initialization. No model or soft teacher produces labels.

For every split, the joint Cartesian product of query type, target slot, and
target-coordinate sign is exactly balanced. The winning record is randomly
placed in its scheduled slot; all other records are randomly permuted.

Per formal world:

- 8,192 train bags;
- 2,048 validation bags;
- 8,192 untouched Gaussian test bags;
- 8,192 four-record unit-variance Laplace bags;
- 8,192 eight-record Gaussian bags with new RoPE offsets;
- 8,192 eight-record unit-variance Laplace bags combining both shifts.

Development seed 53 was used to choose the protocol and is excluded. Formal
world seeds are 89, 127, 167, 211, and 257.

## Frozen arms

1. ordinary bilinear RoPE;
2. gauge-linear control, foldable into a linear K map;
3. position-only exact-mean-square control;
4. centered rotation-funded one-curvature GECK;
5. centered scale-funded one-curvature GECK;
6. uncentered scale-funded one-curvature GECK, the primary candidate;
7. centered two-curvature GECK-G2;
8. uncentered GECK-G2.

All arms receive the same four paired Q/K starts and start at identical logits.
The single-curvature arms are retrained controls, not post-hoc ablations.

## Frozen optimization and selection

Runtime is Python 3.11.10, NumPy 2.1.2, PyTorch 2.5.1+cu124, CUDA 12.4, and
NVIDIA H100 80GB HBM3, with `CUBLAS_WORKSPACE_CONFIG=:4096:8` and deterministic
algorithms enabled.

Each paired run uses AdamW for exactly 4,000 updates, batch 256, learning rate
`3e-3`, 100-step linear warmup then cosine decay, weight decay `1e-3`, and
per-arm/per-restart gradient clipping at 1.0. All arms receive the same
minibatch order. Each arm selects its restart by validation NLL only.

## Frozen counterfactuals

- Independently flip every record coordinate sign; labels remain unchanged.
- Evaluate all 24 record-position permutations and align predictions back to
  record identity. A bag is the statistical cluster.
- Move the complete vector of quadratic features to another record using a
  label-independent nonzero cyclic shift per bag; raw linear content and
  positions stay fixed.
- Set the primary scale curvature coefficient to zero at evaluation.
- Evaluate the three separately generated OOD sets above.

The paired test-bag bootstrap uses 2,000 replicates, chunks of 100, seed
`world_seed * 1,000,000 + 43`, and the original bag as the resampling unit.
Within every replicate it compares the primary against the best of the three
nonquadratic controls, recomputing that maximum for accuracy and minimum for
NLL.

## Per-world pass gates

Every formal world must satisfy every gate:

- All structural balance/label checks pass; all arms have 40 scalars and 160
  FP32 bytes; initial logits match within `1e-7`; training completes with only
  finite parameters, losses, and reported metrics.
- Primary test accuracy is at least 90%, each query type at least 88%, and NLL
  at most 0.25.
- The paired-bootstrap 95% lower bound is at least +0.40 accuracy and +0.50
  nats NLL versus the best nonquadratic control.
- In each of the four paired restarts, primary test accuracy beats the best
  same-restart nonquadratic control by at least 0.40.
- Gauge-linear and position-only controls each stay within 0.04 accuracy and
  0.03 NLL of bilinear.
- Uncentered scale-funded G1 is no worse than centered scale-funded G1,
  centered G2, or full G2 by more than 0.02 accuracy or 0.03 NLL. Being better
  is allowed.
- On both eight-record shifts, centered scale-funded G1 NLL is no worse than
  the matched centered rotation-funded G1 by more than 0.10.
- Independent sign flips reduce accuracy by no more than 0.02 and preserve at
  least 95% of predictions.
- Across all position permutations, accuracy is at least 90%, aligned
  prediction agreement at least 95%, and the largest target-slot accuracy gap
  at most 0.03.
- Quadratic-feature derangement reduces accuracy by at least 0.50 and ends no
  more than 0.05 above the best nonquadratic test accuracy.
- Zeroing scale curvature reduces accuracy by at least 0.50.
- On four-record Laplace, primary accuracy is at least 90%; on each eight-record
  set it is at least 85%. Every OOD NLL is at most 0.50, and every OOD accuracy
  advantage over the best nonquadratic arm is at least 0.40.

All five worlds must pass. Passing admits a standard-task no-harm screen and a
fused H100 epilogue benchmark. It does not yet admit a language-model claim.
