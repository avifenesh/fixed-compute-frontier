# Heterogeneous family discovery T6 — decision

Status: all preregistered 37M screen gates passed; advance to adjacent scale  
Date: 2026-07-30

T6 is the first clean full result in the heterogeneous-compilation lane.  The
sealed artifact `heterogeneous-family-discovery-t6.json` passes every gate in
both paired model seeds with source, tests, preregistration, predecessor, and
data hashes intact.

## Acquisition result

The training-time portfolio received only 96 input/output examples and a task
ID for each of 320 tasks.  It searched 12,870 supports and four count algebras,
recovered the exact support and family for all 256 structured tasks, and
abstained on all 64 random-label decoys.

After compilation and 1,000 mixed steps, every candidate task in parity,
majority, exact-half, and modulo-3 remained at 100% accuracy in both seeds.
Worst-task accuracy was also 100%; protected copy was 100%; independent random
labels stayed at 50.38% and 51.04%.

Ordinary gradient controls received all 30,720 charged prefix examples plus
twice the mixed-step budget.  Their step-2,000 mean accuracies were:

| seed | parity | majority | exact-half | modulo-3 | copy |
|---|---:|---:|---:|---:|---:|
| 2039 | 49.77% | 63.78% | 73.33% | 66.39% | 100% |
| 2281 | 48.35% | 62.85% | 71.86% | 67.35% | 100% |

Thus the result is a qualitative acquisition change, not a sub-percent
endpoint polish: one same-size candidate acquires 256 exact procedures across
four automatically selected families that 2x ordinary training does not.

## Language and stability

Candidate step-1,000 natural validation NLL improved over its paired 1x
control:

| seed | ordinary 1x | compiler 1x | relative delta |
|---|---:|---:|---:|
| 2039 | 6.27540 | 6.15664 | -1.892% |
| 2281 | 6.35602 | 6.16778 | -2.962% |

The compiler avoids 480 ineffective gradient prefix updates whose first and
last losses remain near chance.  All arms stayed finite with maximum loss below
10.91.  Candidate maximum pre-clip gradient norms were 7.72 and 8.28.

## Deployment and capacity accounting

Every arm exports the same 36,577,152-parameter, vocabulary-49,152 causal graph.
No solver, sidecar, adapter, expert, retrieval store, or extra token row remains
at inference.  The shared interpreter uses 28 hidden coordinates, two existing
heads, and 110 existing SwiGLU channels.  It has 5,853 fixed nonzero entries,
including 5,120 task-description entries.  Hard isolation freezes 5,983,580
total entries, mostly zeros; reducing this capacity reservation remains open.

## Claim boundary and prior art

Program synthesis from examples, neural compilation, Tracr/ALTA, Transformer
Programs, and compiled neural program libraries are prior art.  T6 must not be
described as inventing those broad ingredients.

The narrower result supported here is automatic family selection plus
abstention, compiled into an existing same-size causal LM that co-trains on a
real FineWeb stream, with paired 1x/2x controls and no deployed module or graph
change.

This passes the project contract's **qualitative acquisition** mechanism at the
37M screen.  It is not yet the final frontier proof because the admission ladder
requires an adjacent scale, three seeds, a stronger optimizer envelope, private
untouched slices, and a full resource ledger.  T6 therefore advances rather
than completes the research goal.
