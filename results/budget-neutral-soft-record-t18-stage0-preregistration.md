# Budget-neutral soft record T18 — Stage-0 preregistration

Status: frozen before execution  
Date: 2026-07-31

## Purpose

T12-T17 close post-hoc exact analog addressing.  T18 replaces two ordinary
SwiGLU sublayers with a jointly trainable soft entity-record sublayer plus a
smaller SwiGLU, so query interpretation and record use can be learned as part
of the served architecture.  Stage 0 verifies only the operator, resource
ledger, and raw-only compiler boundary.

No accuracy, language-quality, or Pareto claim can follow from Stage 0.

## Exact served matrix ledger

Frozen constants:

- hidden width `H = 384`;
- baseline SwiGLU width `F = 1,024`;
- raw entities `N = 2,405`;
- record slots `S = N + 1 = 2,406`, including one learned null key with a zero
  value;
- residual candidate SwiGLU width `G = 444`.

Two baseline SwiGLUs contain and execute

`6HF = 2,359,296`

matrix scalars/MACs per token.  Candidate record keys/values plus its residual
SwiGLU contain and execute

`2SH + 3HG = 2,359,296`.

The equality must be exact.  Attention, embeddings, normalizations, output
projection, layer count, hidden width, context, precision, and KV shape are
unchanged.

## Frozen operator

At block 8's normalized pre-FFN state `h`:

`p = softmax(h K^T / sqrt(H))`

`record(h) = p V`.

This replaces block 8's ordinary SwiGLU residual.  Block 9 retains ordinary
attention and uses a width-444 SwiGLU instead of width 1,024.

`K,V` each have shape `2,406 x 384`.  Row zero is the null slot; its value is
fixed to zero whenever the raw compiler writes records.  The remaining 2,405
rows correspond to the sorted canonical title set.

## Frozen raw-only compiler API

The Stage-0 compiler accepts only:

- the candidate served model;
- the fixed tokenizer;
- rows with exactly `document_id`, `title`, and `text`.

It groups raw prose by title, writes the normalized mean title-token embedding
as the initial key, and writes the mean of at most 512 raw-prose token
embeddings as the initial value.  This simple writer is an interface contract,
not the final research compiler.  It must reject any question, answer,
supporting annotation, label, or extra field.

The null key is zero and the null value is exactly zero after compilation.

## Frozen Stage-0 execution

- seed: 10,103;
- device: retained H100/H200;
- reference batch: 2 sequences of 128 tokens;
- loss: ordinary next-token cross-entropy over the candidate logits;
- one backward pass, no optimizer update;
- explicit dense reference for record scores, probabilities, and values;
- null-route causal control on a bounded standalone operator;
- actual candidate/baseline parameter counts and matrix-MAC counts;
- peak allocated HBM and materialized activation ledger.

## Admission gates

All must pass:

1. candidate and baseline total parameter counts are exactly equal;
2. replaced matrix parameter and per-token MAC ledgers are both exactly
   2,359,296;
3. candidate logits have the baseline shape and finite values;
4. loss and every candidate gradient are finite after one backward pass;
5. record forward output, key gradient, value gradient, and input gradient match
   the explicit dense reference within `2e-6` on the bounded contract;
6. a dominant null key with zero value suppresses record output below `1e-5`;
7. the raw compiler accepts all 2,405 titles, rejects an extra label field, and
   leaves the null value exactly zero;
8. compilation changes no tensor shape or parameter count;
9. source, preregistration, corpus, manifest, and checkpoint anchors are
   emitted;
10. the materialized and theoretical activation/non-matrix operation costs are
    reported, not treated as free.

Passing admits a from-scratch matched training preregistration.  It does not
admit training automatically: the activation/softmax/kernel ledger may reveal
a served-cost blocker that must be designed out first.
