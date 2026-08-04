# Causal write/read plane T22a — decision

Status: **FAIL; CLOSE THE FROZEN ONE-PASS SAME-MODEL WRITER**  
Date: 2026-07-31

## Decision

Do not export this record into the ordinary graph.  Do not tune the code width,
level count, amplitude, hidden-state slice, masking rate, learning rate, number
of read steps, writer exposure, title router, QA schedule, or thresholds.

T22a fails at **record acquisition**, before the semantic-query-transfer gate.
The same-model one-pass writer did not put causally readable information about
writer-held-out prose into its 220-cell record.  A natural-QA failure therefore
cannot be attributed to an otherwise successful compressed memory that merely
lacked the right question interface.

This closes only the frozen forward write map

`C(X) = 4 Q16(tanh(g(X)_last[64:284]))`

under the preregistered forced-read objective.  It does not close digital
planes, all forward writers, iterative loss-driven writes, or the fixed-served-
compute research objective.

## Causal evidence

The learned writer compiled every one of the 207 writer-held-out documents
exactly once after specialization, with zero gradient presentations.  On the
1,656 sealed held-out probes:

| condition | probe NLL | token accuracy |
|---|---:|---:|
| learned record, correct | 7.162270 | 3.9251% |
| learned record, zero | 7.165928 | 4.0459% |
| learned record, shuffled | 7.162430 | 4.0459% |
| fixed random record | 7.159545 | 4.5894% |

The preregistered effects were:

- correct versus zero: **0.0510%**, below the required 20%;
- correct versus shuffle: **0.0022%**, below the required 20%;
- learned versus random: **-0.0381%**, below the required +10%.

The record is therefore nearly causally invisible, and a same-size random code
is slightly better.  The result is not a marginal miss.

## Natural-QA result

All arms fitted the 146 QA-training rows to effectively zero loss.  On the 104
title-disjoint development rows:

| arm / ablation | accuracy |
|---|---:|
| learned writer, correct | 58.6538% |
| learned writer, zero | 56.7308% |
| learned writer, shuffled | 58.6538% |
| fixed random writer | 59.6154% |
| zero-record reader | 54.8077% |
| compute-matched dense causal | 50.0000% |

The candidate was **0.9615 points below** the best control.  Zeroing its record
cost only **1.9231 points**, while shuffling cost exactly **0 points**.  It
failed the 75% accuracy, +10-point superiority, and both 10-point causal-drop
gates.

## Contracts that passed

- Raw-only writer inputs and frozen data hashes matched.
- Every arm began from the identical common-base state and finished without a
  failed or retried step.
- The learned code used all and only the 16 frozen levels and survived BF16
  roundtrip exactly.
- The 207 held-out records each received one compile pass and no gradients.
- Candidate and dense specialization each charged exactly 6,144,000 sequence-
  token positions; the one-pass corpus compile charged another 307,840.
- The prospective physical write used exactly 743,670 existing entries, below
  the 743,734 cap, with one served state schema and parameter count.
- Protected natural NLL passed: 8.020629 for the candidate versus 10.077588 for
  the dense control, a 20.41% improvement rather than degradation.

The two full experiments shared the H100 while running, so their elapsed
training times are not standalone throughput measurements.  This does not
affect the capability or issued-work gates.

## What was learned

Changing from optional autoregressive conditioning to forced withheld reads
was necessary but not sufficient.  It removed the teacher-forced-prefix
shortcut, yet a single amortized final-state slice still failed to behave as a
content-sensitive write operator on unseen documents.

The next admissible architecture must change the **write dynamics**, not widen
or retune this record.  It must provide error correction or an identifiable
state update while retaining all hard constraints: same from-zero model, raw
prose only, no stronger compiler, no per-document optimized parameters in the
claimed autonomous write path, and no additional served parameters, tokens,
KV state, layers, or FLOPs.

## Evidence hashes

- Result JSON SHA-256:
  `be7ce8c4d31250910e806770606fdb25381283584e20bd529d90008767bdd10b`
- Run log SHA-256:
  `6033bfcb3aaad8fc16dda6ae16471a8d3ad40231df60db5069dedb85c2a1fa51`
- Executed source SHA-256:
  `c59457f73c4b7770afcbf6c82b5254ee0cda2cea233a8bdc7b859c41fa37f222`
- Authoritative preregistration SHA-256:
  `f3dd0a9634efd1aca50c6c267c9980f8a14487737e8b0e4a5b20e1df3a7ea3e9`

