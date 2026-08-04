# Target Workload and Baseline v0

## Training scale ladder

Use the open, standardized DataComp-LM/OpenLM ladder so candidates cannot pick
a favorable private baseline.

| Gate | Parameters | Train tokens | Train FLOPs | Baseline estimate | Use |
|---|---:|---:|---:|---:|---|
| G3 screen | 412M | 8.2B | 2.0e19 | 26 H100-hours | Reject/rank; three seeds |
| G4 research | 1.4B | 28B | 2.4e20 | 240 H100-hours | Mandatory confirmation |
| Later target | 6.9B | 138B | 5.7e21 | 3,700 H100-hours | Deployment-scale proof |

These are DCLM baseline estimates, not promises about rental hardware. The
official source is <https://github.com/mlfoundations/dclm> and the paper is
<https://arxiv.org/abs/2406.11794>.

Freeze at each tier:

- GPT-NeoX 50K tokenizer and its exact hash;
- packed 2,048-token training sequences;
- data pool, processed shards, split, ordering seeds, and exposure count;
- optimizer/search budget, checkpoint selection, and evaluation prompts;
- exact training-FLOP budget as the primary comparison.

If a candidate changes training cost, publish two views:

1. stop both at the tier's fixed total training FLOPs and report tokens seen;
2. allow the full fixed token count and report the candidate's extra training
   FLOPs separately.

## Strong ordinary baseline frontier

A candidate must beat all applicable controls:

1. Exact DCLM/OpenLM dense model and recipe at the official tier.
2. Best ordinary dense point from a small, equal-budget width/depth/GQA sweep.
3. A dense control given the same additional data, objective, optimizer tuning,
   teacher information, and/or training FLOPs used by the candidate.
4. Public strong models near the size only as external deployment anchors; they
   are not causal controls because their data and training budgets differ.

All served learned bytes count, including inactive experts, auxiliary heads,
routers, tables, codebooks, and helpers.

## Fixed generation cells

For diagnostic timing, use real fixed prompts with exact token lengths, greedy
decoding, and EOS disabled.

| Cell | Input tokens | Output tokens | Static batch / online concurrency |
|---|---:|---:|---|
| Short interactive | 128 | 128 | 1, 8, 32 |
| Normal chat | 512 | 128 | 1, 8, 32 |
| Prefill-heavy | 2,048 | 128 | 1, 8, 32 |
| Decode-heavy | 512 | 512 | 1, 8, 32 |
| Extended context | 8,192 | 256 | 1, 8 |

The 8K cell is a separate lane because baseline pretraining is 2K. It becomes a
binding claim only when baseline and candidate receive the identical declared
context-extension stage.

## Online service trace

Use 10,000 requests per run with open-loop Poisson arrivals:

- 60% `512 -> 128`;
- 25% `2,048 -> 128`;
- 15% `512 -> 512`;
- offered load at 25%, 50%, 75%, and 90% of the baseline's maximum
  SLO-valid goodput.

Primary service metrics are goodput, p50/p95/p99 TTFT and TPOT, end-to-end
latency, failures/rejections, realized batching, peak HBM, exact KV/state,
measured traffic/work, joules, and gross cost per completed request.

The default reference SLO lanes are:

- interactive: 500 ms TTFT and 30 ms TPOT;
- conversational: 2 s TTFT and 100 ms TPOT.

These are reference lanes, not claims that every model/workload must target the
same product. Any changed SLO creates a separate workload stratum.

## Hardware and attention baseline

- Development uses one rented H100 stratum; the exact PCIe/SXM/NVL SKU is
  frozen and reported.
- H200 replication is a separate stratum. Candidate and baseline always run on
  the same physical SKU; results are never pooled across H100 and H200.
- H100/H200 use the fastest mature exact-attention backend available for each
  frozen shape after a preregistered baseline autotune (for example FA3/cuDNN
  SDPA). A candidate is not compared with naive materialized attention.
- Prefill and single-token decode are benchmarked separately because their
  attention/KV bottlenecks differ.

## Quality baseline

At 400M, use the corrected current DCLM Core evaluation plus the private packs
in `CAPABILITY_CONTRACT.md`. At 1B and later, add DCLM Extended, MMLU where not
at floor, OLMES base-model suites, and RULER at every legitimately supported
context length. Public tasks are development evidence; locked private packs
decide the claim.

## Promotion

Promote from 400M only when three seeds beat the strongest dense control on the
preregistered quality vector without a protected regression. Believe the result
only after the 1B confidence interval remains positive and the full resource
contract passes. A 7B run is reserved for a result that has already survived
both gates; it is confirmation, not exploration.
