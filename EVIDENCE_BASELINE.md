# Evidence Baseline

Updated: 2026-07-22

## Current boundary

The audit found no demonstrated architecture that adds broad intelligence while
holding the entire serving envelope fixed: graph, precision, hardware, batch,
context, memory, latency, and generated-token count.

The strongest clean evidence for better capability at the same served graph is
offline: better data, objectives, optimization, and post-training. These are not
free; they move cost into training. Exact runtime work can create real B/D gains,
but does not itself create intelligence.

## Robust patterns, with the moved cost exposed

1. **Dataset quality at fixed recipe.** DataComp-LM changes the dataset while
   holding a 6.9B model, 138B training tokens, and the training recipe fixed,
   producing large score differences. Filtering, extraction, deduplication, and
   filter-model training are offline costs. Evidence:
   <https://arxiv.org/abs/2406.11794> and
   <https://github.com/mlfoundations/dclm>.
2. **Post-training at fixed deployed graph.** Tulu 3 improves the same Llama
   base through SFT, preference learning, and RLVR. Teacher/judge calls, rollout
   compute, and possibly longer answers move total cost. Evidence:
   <https://arxiv.org/abs/2411.15124> and
   <https://github.com/allenai/open-instruct/blob/main/docs/tulu3.md>.
3. **Auxiliary training objectives discarded at inference.** Multi-token
   prediction improved 13B HumanEval and MBPP results while allowing auxiliary
   heads to be omitted from ordinary inference. Extra training heads and losses
   are the moved cost; the gain is not established as universal. Evidence:
   <https://proceedings.mlr.press/v235/gloeckle24a.html>.
4. **Compute-optimal sizing.** Chinchilla demonstrated that more training data
   and fewer served weights can beat a larger undertrained model at comparable
   training compute. The precise optimum is disputed, but overlarge undertrained
   models are a real source of slack. Evidence:
   <https://arxiv.org/abs/2203.15556> and critique
   <https://arxiv.org/abs/2404.10102>.
5. **Exact runtime improvements.** FlashAttention removes excess HBM traffic
   from conventional attention implementations; PagedAttention reduces KV
   memory-management waste. Both preserve the same mathematical model. Their gains are
   hardware/workload dependent and create a runtime budget rather than
   intelligence. Evidence: <https://arxiv.org/abs/2205.14135> and
   <https://arxiv.org/abs/2309.06180>.

## Claims not accepted without stronger controls

- Active MoE parameters are not equivalent to dense-model serving cost; expert
  residency, traffic, dispatch, collectives, and imbalance remain.
- KV compression or linear/hybrid attention is not weight-held knowledge
  density; it is a memory/runtime edge unless saved budget is reinvested and
  remeasured.
- Speculative decoding is not free intelligence; it is a lossless runtime trade
  involving draft/verification work and workload-dependent acceptance.
- More route identities, heads, experts, or feature combinations are not more
  independent stored knowledge.

## Consequence

Published evidence informs where slack has existed; it does not select our next
candidate. Any future entry must reproduce a gain under this repository's full
budget ledger rather than inherit a paper's headline metric.
