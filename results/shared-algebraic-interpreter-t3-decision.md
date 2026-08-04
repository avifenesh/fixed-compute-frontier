# Shared algebraic interpreter T3 — decision

Status: preregistered protocol failed; large density separation retained  
Date: 2026-07-30

The sealed result is `shared-algebraic-interpreter-t3.json`.  Its aggregate
gate is false because one ordinary-control seed exceeded the frozen pre-clip
gradient envelope.  The exact protocol is closed and its threshold must not be
changed after measurement.

## What passed

The candidate recovered all 256 independently sampled 16-of-32 supports from
64 examples per rule.  One shared in-place interpreter then achieved 100%
accuracy on every rule at steps 250, 500, and 1,000 in both paired model seeds.
Protected-copy accuracy was 100% throughout.

Ordinary gradient training received the complete 16,384-example charged prefix
and twice the mixed-step budget.  At step 2,000 it remained at chance:

| seed | compiled 1x mean/min | ordinary 2x mean/min | copy |
|---|---:|---:|---:|
| 1181 | 100% / 100% | 50.47% / 32.81% | 100% |
| 1429 | 100% / 100% | 50.24% / 39.06% | 100% |

The candidate also passed every language-noninferiority checkpoint.  At step
1,000 it was better than its paired 1x control, not merely within tolerance:

| seed | ordinary 1x NLL | compiled 1x NLL | relative delta |
|---|---:|---:|---:|
| 1181 | 6.22007 | 6.19328 | -0.431% |
| 1429 | 6.20327 | 6.17345 | -0.481% |

The exported graph has 36,577,152 parameters and vocabulary 49,152 in every
arm.  Rule IDs, position-specific input symbols, queries, and result symbols
reuse 324 token rows absent from both sealed natural streams.  No adapter,
sidecar, expert, retrieval store, vocabulary growth, or runtime solver exists
in the candidate checkpoint.

## Density accounting

All 256 rules use the same 40 hidden coordinates, two attention heads, and 53
active decoder channels.  Rule descriptions occupy exactly 8,192 scalar
entries, or 32 per rule.  The compiled checkpoint has 9,902 fixed nonzero
entries total, so only 1,710 fixed nonzeros implement the shared interpreter
beyond the rule table.

Hard block isolation freezes 6,343,336 entries, 99.844% of which are zeros.
That is the construction's main remaining capacity cost.  Amortized frozen
entries are 24,779 per rule, versus roughly 2.6 million frozen entries for the
single-rule T2 circuit.  The rule-density improvement is therefore about two
orders of magnitude, but it is not yet a proof that those zero constraints can
be removed safely.

## Why the preregistered claim fails

Seed 1429's ordinary controls reached a maximum pre-clip gradient norm of
1,247.54; the frozen limit was 100.  Loss remained finite (maximum 10.93),
global clipping kept training finite, and the candidate maxima were only 7.82
and 8.85.  Nevertheless, the gate applies to every arm, so it fails.  Seed 1181
passed the same envelope with a maximum of 11.41.

This does not explain away the failure: T3 is not a clean all-gates frontier
result.  It also remains a synthetic fixed-family test, and the compiler is
told to solve over GF(2).  Those facts prevent a general breakthrough claim.

## Finding retained and next question

T3 establishes a qualitative, same-size optimization separation and a large
amortization result: a short rule table plus one shared neural interpreter can
install 256 exact capabilities that twice the ordinary gradient budget does
not acquire, while improving paired natural validation NLL.

The next admissible step is not another optimizer tweak.  It must remove a
research assumption:

1. discover which algebraic family explains each charged slice rather than
   being told GF(2); and
2. replace millions of hard zero constraints with a low-interference merged
   circuit or prove that such packing cannot preserve the language function.

