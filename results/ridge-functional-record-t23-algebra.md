# Ridge functional record T23 — algebra and boundary

Status: **DERIVATION ONLY; NOT PREREGISTERED OR ADMITTED**  
Date: 2026-07-31

## Evidence that forces the change

T21a and T22a isolate two different failures.

- A freely optimized 220-cell document code trained through ordinary
  autoregressive likelihood received nonzero gradients but reduced document
  NLL by only 3.95% versus zero and 0.86% versus shuffle.  It tied dense-1x at
  60.58% held-out QA.
- A one-pass same-model writer trained through forced withheld reads improved
  held-out probe NLL by only 0.051% versus zero, 0.002% versus shuffle, and was
  0.038% worse than a random code.  It therefore failed acquisition before
  semantic transfer.

The missing quadrant is a record that is **fitted to forced raw reads** without
requiring a neural forward writer to amortize the entire set-to-code map.

Per-document gradient descent would fill that quadrant, but GradMem already
establishes iterative optimized memory tokens as a strong baseline.  It would
also leave the central representation question obscured by optimizer steps.
T23 instead asks whether the record can be the exact solution of a small,
permutation-invariant algebraic problem whose features come from the same
from-zero model.

## Document-local operator

Split the existing ten-block Transformer at a frozen middle boundary.  For a
raw-only corrupted view of document `d`, the first half of the same model emits
a normalized query feature

`k_i in R^22`.

The withheld raw token supplies a target feature

`v_i in R^10`

through the same trainable embedding/state system.  No parser, teacher,
generated question, answer annotation, support field, or external encoder is
used.  Collect `m` separating raw views as

`K_d = [k_1,...,k_m] in R^(22 x m)` and
`V_d = [v_1,...,v_m] in R^(10 x m)`.

The continuous document program is the unique ridge solution

`C_d* = V_d K_d^T (K_d K_d^T + lambda I)^(-1) in R^(10 x 22)`.

It contains exactly 220 scalars.  The stored program is

`C_d = 4 Q16(tanh(s C_d*))`,

with one frozen global `s`, the existing 16-level alphabet, and no per-document
scale or optimized parameter.

At a read site, the same first-half model emits `q in R^22`; the document
program returns `r = C_d q in R^10`, which is injected into reserved hidden
coordinates before the remaining ordinary blocks.  Training backpropagates
through the solve and the straight-through quantizer into the shared model, but
the held-out document compile performs only the frozen forward features,
22-by-22 solve, and quantization.

## Exact property

For `lambda > 0`, define

`J(C) = ||C K - V||_F^2 + lambda ||C||_F^2`.

`J` is strictly convex.  Setting its derivative to zero gives

`C (K K^T + lambda I) = V K^T`,

so the stated `C*` is the unique global minimizer.  The compiler therefore
does not learn to imitate an unknown code.  It computes the best linear
query-to-value function available in the same-model feature basis.

Three useful consequences follow.

1. **Permutation invariance.**  Reordering raw views multiplies both `K` and
   `V` by the same column permutation, leaving `V K^T` and `K K^T` unchanged.
2. **Exact orthogonal recall.**  If the keys are orthonormal and `lambda -> 0`,
   then `C* k_i = v_i` for every fitted view.
3. **Compositional interpolation.**  For a new query in the fitted span,
   `q = K a`, the unregularized read is
   `C* q = V P_K a`, where
   `P_K = K^T (K K^T)^(-1) K` is the projector onto the row space of `K`.
   This equals `V a` only when `a` lies in that row space (including the square
   invertible case).  A natural question can therefore compose only the value
   combinations identifiable from the fitted key geometry; the held-out gate
   must test this rather than assume it.

These properties are absent from a pooled hidden vector.  They also differ
from T20a's energy-optimal HOSVD of next-token learning operators: T23 fits the
causal raw query-to-withheld-target function directly, and the key covariance
appears in the inverse rather than being ranked by gradient energy.

## Quantization and finite capacity

The exact theorem applies before quantization.  If each stored coefficient has
error at most `epsilon`, then for a unit query

`||(C_quant - C*) q||_2 <= ||C_quant - C*||_F <= sqrt(220) epsilon`.

This is only a bound, not an assurance that the fixed 16-level alphabet is
adequate.  A mandatory continuous-versus-quantized control must measure the
loss caused by finite coding.  Failure of the continuous operator closes the
feature/read law; success continuous but failure quantized isolates the finite
alphabet.

The rank is at most 22.  The mechanism does not promise lossless storage of all
roughly 100 content tokens per document.  It promises the best regularized
10-output function on a learned 22-dimensional query span.  Held-out raw probes
and natural QA, not reconstruction of fitted views, decide whether that is the
right compression.

## Why the compiler is not a stronger model

The solve has no language parameters and cannot answer a question.  It only
aggregates key/value features emitted by the same served model into a finite
coefficient table.  Its extra cost occurs once during offline compilation and
must be charged against an equal-work dense-training control.  The compiler
then disappears.

This avoids the false comparison “if the compiler is smarter, deploy it”: the
compiler is a 22-by-22 ridge solve, not a predictor.  Any capability must still
come from the shared model's learned feature basis and ordinary reader.

## Physical serving path

T23 is virtual until proved otherwise.  A possible ordinary-graph realization
uses two already available SwiGLU roles:

1. the existing title-addressed FFN channels retrieve the flattened 220-cell
   record into reserved hidden coordinates;
2. a later shared SwiGLU block uses one existing intermediate channel per
   coefficient to form the 220 products between record coefficients and the
   appropriate query coordinate, while its down projection sums them into ten
   outputs.

This reassigns existing channels; it adds no parameter, token, layer, KV state,
or issued multiply.  It is not yet evidence that natural quality survives the
reassignment.  Physical export is forbidden unless the virtual behavioral
gates pass first, and the export must then demonstrate exact semantic
equivalence and unchanged serving resources.

## Mandatory diagnostic order

1. **Continuous operator oracle:** same split/read law without quantization.
   It must show causal held-out raw-probe gain over zero, shuffle, additive
   outer-product memory, and random code.
2. **Quantized autonomous compiler:** the one-solve, 220-cell, 16-level record
   must retain the causal gain on documents that supplied no model gradients.
3. **Behavioral transfer:** correct records must substantially improve sealed
   title-disjoint natural QA over all controls, and zero/shuffle must remove the
   gain.
4. **Physical export:** only after all virtual gates pass.

The additive outer-product control is `C_add = V K^T` with matched features and
capacity.  It isolates whether whitening/error correction from
`(K K^T + lambda I)^(-1)` is doing useful work rather than the experiment merely
benefiting from a matrix-shaped record.

## Failure interpretation

- Continuous correct versus zero/shuffle fails: close this linear functional
  feature/read law.
- Continuous passes but quantized fails: the fixed four-bit coefficient
  alphabet, not acquisition, is the boundary.
- Raw probes pass but natural QA fails: close the raw-mask-to-natural-query
  feature alignment.
- Virtual behavior passes but physical export fails: close this ordinary-graph
  realization.
- All pass: replicate at unseen seeds and corpus split before any broader model
  claim.

No post-result tuning of dimensions, regularization, scale, masks, steps,
injection layer, router, or thresholds is allowed inside a frozen run.

## Prior-art boundary

- Fast-weight programmers and DeltaNet learn online key/value matrix updates;
  delta-rule correction is an iterative approximation to a regression memory.
- `delta-mem` keeps an online state matrix and uses its readout during serving.
- GradMem optimizes memory tokens per context with test-time gradient descent.
- Latent Context Compilation emits portable buffer tokens that remain serving
  inputs.

T23 is not novel merely because it uses ridge regression.  The potentially new
combination is narrower: a same-model raw-only closed-form document operator,
quantized to a fixed digital plane, behaviorally validated on unseen natural
queries, and then physically absorbed into otherwise identical ordinary-model
weights so no memory artifact or extra operation remains at serving.  Novelty
must not be claimed before behavioral and physical evidence exists.

Primary comparisons:

- Linear Transformers Are Secretly Fast Weight Programmers:
  <https://arxiv.org/abs/2102.11174>
- GradMem: <https://arxiv.org/abs/2603.13875>
- delta-mem: <https://arxiv.org/abs/2605.12357>
- Latent Context Compilation: <https://arxiv.org/abs/2602.21221>
- Beyond Perplexity: <https://arxiv.org/abs/2607.00368>
