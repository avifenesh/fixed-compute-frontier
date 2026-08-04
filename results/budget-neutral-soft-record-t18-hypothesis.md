# Budget-neutral soft entity-record operator T18 — algebraic hypothesis

Status: **design boundary; no capability result**  
Date: 2026-07-31

## Key reframe

The failed compiler asked a raw-document vector to decide whether a natural
question expressed the same property.  T18 removes that responsibility.

The compiler performs only:

`raw document for entity e -> dense record v_e`.

The served model performs:

`natural contextual state h -> soft mixture of entity records -> ordinary reasoning`.

No exact categorical route is assumed.  Close entities may share mass.  The
only success criterion is a substantial held-out knowledge/reasoning gain over
matched training at unchanged served resources and protected language quality.

## Exact small-model matrix ledger

For the frozen model width `H=384`, SwiGLU width `F=1,024`, and `N=2,405`
candidate entities:

- one ordinary SwiGLU has `3HF = 1,179,648` matrix parameters/MACs per token;
- two ordinary SwiGLUs have `6HF = 2,359,296`;
- dense record keys plus values have `2NH = 1,847,040`;
- a residual SwiGLU of width 444 has `3H*444 = 511,488`;
- record operator plus residual SwiGLU total `2,358,528`, 768 fewer matrix
  parameters/MACs than the two replaced SwiGLUs.

The served matrix budget can therefore be matched without increasing model
parameters or matrix MACs:

`two 384->1024->384 SwiGLUs`

become

`384->2405 key scores -> 2405->384 record mixture + 384->444->384 SwiGLU`.

Existing normalization parameters can be retained one-for-one.  This ledger
does not price softmax exponentials/reductions, extra activation traffic,
kernel launches, or loss of nonlinear depth.  Those are fatal controls, not
free details.

## Why the compiler is not a stronger served model

The compiler is allowed to process each raw document once during training.  It
cannot see a future query and is absent at serving.  Its output is only the
fixed value row `v_e`; query interpretation remains in the served model.

The possible gain is amortization: spend document-side computation once, store
its result inside weights, and reuse it across requests.  If the records do not
improve held-out QA after all training/serving costs are matched, the mechanism
has no value.

## From-zero candidate

A valid candidate must be trained from scratch, not surgically declared from
the T12 checkpoint:

1. candidate and control receive identical natural tokens, raw documents,
   router/control examples, total token presentations, and optimizer budget;
2. the candidate compiler sees only title plus raw prose and emits one record
   per entity;
3. the record operator and reader train jointly so the address/record contract
   is part of the architecture rather than post-hoc geometry;
4. the control retains the two ordinary SwiGLUs and receives any
   compiler-derived training targets, preventing privileged preprocessing from
   being the candidate's advantage;
5. the compiler is discarded and only the matched served models are compared.

## Required gates before a production claim

1. exact parameter and matrix-MAC equality;
2. measured H100/H200 latency, HBM traffic, workspace, and peak-memory
   noninferiority at production-relevant batches/contexts;
3. protected natural NLL no worse than +0.5%;
4. at least +10 absolute points on a newly frozen document-disjoint natural QA
   evaluator, with every seed positive;
5. causal record-use ablation of at least 8 points;
6. candidate beats matched dense training, twice-trained dense control, and a
   same-budget supervised/preprocessed control;
7. no external lookup, extra KV state, extra served parameter, or hidden
   compiler execution;
8. replication on unseen seeds and a larger model/data scale.

Only those gates can move the result toward “smarter production LLM.”

## Immediate next step

Before another H100 training run, implement and test the operator/reference
ledger:

- exact forward/backward reference;
- matrix-parameter/MAC counter;
- softmax and activation-memory ledger;
- conversion of two baseline FFNs into one record-plus-small-FFN block;
- compiler API that accepts only raw title/prose rows;
- fatal null-record and shuffled-record controls.

No synthetic accuracy task is needed: T11 already established controlled
digital memory.  The first learned evaluation after the operator contracts
must use real prose and natural questions.
