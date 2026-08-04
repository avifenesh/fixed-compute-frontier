# Hotpot contextual address T13 — preregistration

Status: frozen before execution  
Date: 2026-07-31

## Purpose

T12 established exact natural multi-token entity addressing but closed static
input embeddings as an insufficient semantic key space.  T13 tests the direct
implementation rethink: does the contextual vector already multiplied by an
existing FFN gate contain a reliable zero-extra-serving-compute interface for
natural questions and raw prose?

This is a privileged feasibility diagnostic.  It does not compile a plane,
modify weights, answer questions, or establish a held-out capability gain.

## Frozen state

- checkpoint:
  `hotpot-semantic-address-t12-shared-base.pt`, 146,340,367 bytes, SHA-256
  `a2900585a6e9afe4a9fba4f55afca30df5bd79c335d646cda5b9e940b72d693b`;
- terminal model-state SHA-256:
  `53b3117727045ad31b52efd719b55fb251a557b2256008c924318c3cc244de57`;
- model: 36,577,152 parameters, 10 layers, hidden width 384, six heads,
  SwiGLU width 1,024;
- tokenizer: `HuggingFaceTB/SmolLM2-135M` at
  `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`;
- candidate corpus SHA-256:
  `a02d2bdc714bae0c6e42d8527333de01506120845095aec936a301634fb7d00a`;
- T12 result SHA-256:
  `e7c260ec7cbf1aef33e1f8e15213a6ee53f58741ab6971b1446e8e6186b0853a`.

No optimizer, update, learned probe, projection, adapter, or new parameter is
permitted.  The checkpoint must remain byte-identical before and after T13.

## Frozen contextual interface

For a token sequence, run the checkpoint through blocks 0 through 8 and the
attention sublayer of block 9.  The interface vector is block 9's normalized
FFN input, exactly the tensor already multiplied by `gate.weight` and
`up.weight` during ordinary inference.  The final FFN, final norm, and logits
are not needed for the diagnostic.

- Raw page prose is tokenized without special tokens, truncated to 512 tokens,
  and encoded in consecutive non-overlapping 128-token chunks.  All real-token
  interface vectors are L2-normalized and retained.
- The full natural question is tokenized without special tokens and truncated
  to the model's 128-token context.  Its final real-token interface vector is
  L2-normalized and used as the query.
- Padding is appended only for batching and is discarded.  Causality makes it
  unable to affect preceding real-token states.

There is no layer choice, pooling choice, learned metric, or representation
sweep.

## Frozen diagnostic

For each question, the diagnostic uses the two gold supporting-page identities
only to test representational compatibility:

1. compute the maximum cosine similarity between the question vector and any
   contextual token vector in each supporting page;
2. take the minimum of the two page maxima;
3. fit one scalar threshold and one direction on the same 146
   document-disjoint training questions used in T12;
4. report accuracy on the same 104 T12 evaluation questions.

The exact procedure is run on both the sealed trained checkpoint and its
untouched seed-8,209 initialization.  Frozen references are 55.7692% for the
privileged lexical diagnostic and 60.5769% for T12's trained static diagnostic.

## Admission gates

All must pass:

1. every checkpoint, model-state, source-data, corpus, manifest, T12 result,
   and preregistration integrity check passes;
2. the interface implementation is numerically identical to the tensor seen
   by the existing final-block FFN gate;
3. the checkpoint is byte-identical before and after the diagnostic;
4. trained contextual accuracy is at least 70%;
5. trained contextual accuracy is at least 10 percentage points above the
   privileged lexical reference;
6. trained contextual accuracy exceeds both the random-initialization
   contextual diagnostic and T12's trained static diagnostic;
7. all values are finite and no training or retry occurs.

Passing admits a separately preregistered compiler experiment on a newly
frozen holdout.  It does not itself prove autonomous extraction, question
answering, a strict Pareto gain, or a smarter production LLM.

Failure closes this untrained final-FFN contextual-cosine interface at the
frozen scale.  No post-result layer, pooling, metric, chunking, prompt, or
threshold sweep belongs to T13.

## Holdout accounting

The T12 aggregate evaluation result was already observed before this protocol
was written.  Therefore the 104 questions are a development set for T13 even
though individual evaluation labels do not affect weights or protocol
parameters.  No result on them may be presented as fresh held-out evidence.
