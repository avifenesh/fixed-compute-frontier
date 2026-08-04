# Functional residual operator T20a — preregistration

Status: **frozen before implementation result**  
Date: 2026-07-31

## Key shift

T19d proved that predictive residual information fits the aggregate write
budget but an entropy stream requires sequential decoding.  T19e then gave
every lexical feature a collision-free coordinate and scored only 56.7308%,
identical to query-only.  The missing representation must therefore preserve
what the prose asks the model to *learn*, not merely its words or pooled state.

For a frozen raw-trained LM, let:

- `h_t in R^384` be the final hidden state before target token `y_t`;
- `E in R^(V x 384)` be the tied token matrix;
- `p_t` be the predicted vocabulary distribution;
- `r_t = E[y_t] - p_t E` be the negative hidden-space loss gradient.

For document `d`, define its normalized first-order learning operator

`G_d = normalize_F(sum_t h_t outer r_t)`.

If a virtual hidden-space linear map were trained one step on the document,
`G_d` is its update direction.  Unlike a word set or mean hidden state, it
binds predictive context to the correction requested by the raw target.

## Raw-only 220-cell projection

The compiler reads only each document's raw `text`, plus the sealed T12
checkpoint that was trained from scratch on natural text and the same raw
corpus.  It uses at most the first 128 next-token transitions and no question,
answer, support annotation, parser, external model, or teacher.

Across the 2,405 normalized document operators, form

`C_left = sum_d G_d G_d^T` and `C_right = sum_d G_d^T G_d`.

Let `A` be the top 20 eigenvectors of `C_left` and `B` the top 11 eigenvectors
of `C_right`, with deterministic sign canonicalization.  The document payload
is

`M_d = normalize_F(A^T G_d B) in R^(20 x 11)`.

This is a separable HOSVD projection learned only from raw prediction error.
Flattening gives exactly 220 cells.  Payloads are round-tripped through BF16.

The fixed random control replaces `A,B` with seed-20,001 Gaussian QR bases of
the same shapes and repeats the entire payload/query/reader protocol.

## Privileged query sufficiency screen

For a question, the two longest nonoverlapping raw titles are removed.  The
remaining property text is converted into the same normalized operator `G_q`
and projected to `Q = normalize_F(A^T G_q B)`.  This query-side gradient
calculation is a privileged upper bound: a passing result must later train the
ordinary forward graph to emit `Q`; T20a does not claim that it already does.

For flattened `Q` and two title-addressed payloads `M_a,M_b`, the reader uses
the exact T19b symmetric feature algebra:

- `min(Q*M_a, Q*M_b)`;
- `max(Q*M_a, Q*M_b)`;
- `M_a*M_b`;
- `abs(M_a-M_b)`;
- the five corresponding dot/min/max scalars.

One float64 logistic reader is fitted from zeros on the 146 training QA rows,
with training-only standardization, mean BCE, L2 0.01, LBFGS at most 200
iterations, threshold 0.5, and no sweep or retry.  It is evaluated on the 104
development rows.

Three controls are frozen:

1. **random subspace:** identical 20x11 capacity and reader, random raw-blind
   orthonormal projections;
2. **query-only:** the same reader sees only flattened HOSVD `Q`;
3. **document shuffle:** the fitted HOSVD reader is evaluated after fixed
   seed-20,005 permutation of development document payloads, without refit.

## Ordinary-graph realization and physical ledger

The interaction is bilinear, not a new served operator.  If a hidden region
holds flattened `Q` and another holds `M`, an existing SwiGLU can place one in
its gate input and one in its up input; their elementwise product is `Q*M`, and
the existing down projection performs the learned reduction.  A later
joint-training experiment must realize this inside the unchanged Transformer.

T20a also writes the HOSVD payload with T19b's already demonstrated layout:

| write | entries |
|---|---:|
| title token codes | 132,800 |
| per-document address/gate constants | 81,770 |
| `2,405 * 220` operator payload | 529,100 |
| **total** | **743,670** |

This is 64 entries below the frozen 743,734 isolation cap.  No parameter,
tensor, layer, token, KV state, or served matrix shape is added.

## Gates

All gates are required:

1. Frozen checkpoint/corpus/train/evaluator/preregistration hashes match; the
   compiler sees only raw documents and the same-model checkpoint.
2. All document and query operators, covariance matrices, eigenspaces,
   projections, payloads, reader weights, and logits are finite and nonzero.
3. `A` and `B` are orthonormal within 1e-4; every payload has unit FP32 norm;
   minimum BF16 cosine is at least 0.9999.
4. Exactly two titles are found in every QA row without support annotations;
   both answer classes occur in train and development.
5. Candidate/control state schemas and parameter counts are identical; the
   compiler-specific physical write is exactly 743,670 and at most 743,734.
6. HOSVD candidate training accuracy is at least 95%.
7. HOSVD development accuracy is at least 75% and at least 10 absolute points
   above T19b's 60.5769%.
8. HOSVD development accuracy beats both the random-subspace and query-only
   controls by at least 10 absolute points.
9. Shuffling development document payloads reduces HOSVD accuracy by at least
   10 absolute points.

## Decision rule

- **Fail:** close this frozen first-order HOSVD statistic.  Do not tune ranks,
  layers, contexts, normalization, checkpoint, residual sign, reader, or
  subspace.  Use the controls to decide whether the failure is functional
  signal, compression, or query dependence before proposing a successor.
- **Pass:** admit one same-shape, from-zero joint writer/forward-reader
  experiment with matched dense training controls.  T20a alone cannot prove a
  smarter model because its query operator is privileged.

No post-result rescue is allowed.
