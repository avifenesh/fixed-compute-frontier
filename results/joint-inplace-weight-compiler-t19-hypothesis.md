# Joint in-place weight compiler T19 — hypothesis boundary

Status: **point-zero redesign; no result**  
Date: 2026-07-31

## Surviving mechanism

The served model is exactly an ordinary dense Transformer.  No record lookup,
extra head, adapter, projection, KV state, external index, sparse kernel, or
changed tensor shape exists at serving.

A training-only compiler reads raw prose and emits structured writes into a
fixed subset of the Transformer's existing FFN/embedding weights.  The reader
and compiler train jointly from zero so natural query states and compiler
writes share an intentional protocol.  At the end of training, writes are
merged, the compiler is discarded, and the served graph is byte-for-byte the
ordinary model class.

## Why this differs from ordinary gradient training

For an example `x`, SGD contributes a gradient that is averaged with unrelated
examples in shared continuous coordinates.  The proposed compiler emits an
example-conditioned allocation and payload:

`C_theta(x) -> (address code, sparse existing-weight indices, quantized payload)`.

Its objective includes collision/interference cost across the compiled corpus,
so distinct facts can occupy disjoint or error-correcting digital cells while
the reader learns how to activate those cells.  The compiler's one-time work is
amortized and absent at serving.

The research question is not whether a hypernetwork can memorize.  It is
whether structured in-place writes yield substantially more held-out natural
knowledge/reasoning than the strongest equal-served-size dense training control
after the control receives the same raw corpus and compiler-derived targets.

## Non-negotiable constraints inherited from T10-T18

1. No post-hoc cosine, output metric, learned entity projection, or exact analog
   title router.
2. No new served parameter, operation, workspace, KV scalar, tensor shape, or
   model class.
3. Compiler input contains only raw document ID/title/text; no questions,
   answers, support annotations, or latent schema.
4. Natural query compatibility is trained from corpus-derived alternate views,
   not assumed from ordinary LM geometry.
5. Every compiler-derived target/view is also given to the strongest dense
   control, so preprocessing is not the candidate's information advantage.
6. The candidate must beat both equal-step and equal-total-training-compute
   controls; extra compiler training cost is reported separately.
7. Final evidence uses a newly frozen evaluator and unseen seeds.  T12's 104
   questions are development-only.

## First mathematical gate before training

Construct a bounded model of the actual write budget and prove/test:

- exact served state-dict identity between candidate export and baseline class;
- a frozen write subset no larger than T11's demonstrated 2.0333% isolation
  budget for the first gate;
- compiler writes are BF16-representable with positive read margins;
- collision-aware allocation stores more independent raw facts than the rank
  of a matched gradient update under the same write subset;
- removing/shuffling compiled writes destroys the added behavior;
- giving the dense control the exact compiler-derived views/targets does not
  reproduce the candidate at equal served resources.

Only after that bounded gate passes should the H100 train a real-prose model.

## Production admission target

The full goal still requires, jointly:

- from-zero raw-prose compilation;
- at least +10 absolute points on fresh document-disjoint natural knowledge and
  reasoning, with every seed positive;
- protected natural NLL within +0.5%;
- identical served model class, parameters, precision, graph, KV, workspace,
  and latency;
- stronger matched training/preprocessing controls;
- unseen-seed and larger-scale replication.

Until those hold, the result is not a smarter production LLM.
