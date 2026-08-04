# Held-out forced-read record oracle T23a — decision

Status: **VALID FAIL; CLOSE THIS VIRTUAL READ INTERFACE**  
Date: 2026-07-31

## Decision

T23a localizes the T22 failure at the continuous record/read interface, before
one-pass writer amortization and before four-bit quantization.  Do not improve,
widen, distill, or error-correct the writer against this interface.  Even a
stronger document-local continuous oracle could not write a record that the
frozen reader used on disjoint held-out target positions.

This is a wall for the exact rule that injects a 220-cell vector into one
title-position hidden-state slice and asks an ordinary Transformer reader to
discover the code convention through forced masked-token training.  It is not
a theorem against digital planes, weight compilation, or every possible
write/read algebra.

## Decisive evidence

All mandatory integrity gates passed on the corrected from-zero rerun.  The
continuous oracle optimized one record per held-out document for 40 balanced
rounds using 80 raw-only compile views per document.  Model weights were frozen,
compile and evaluation target pairs were disjoint, and the 207 held-out
documents never entered meta-reader gradients.

On 1,656 sealed held-out probes:

| condition | probe NLL | token accuracy |
|---|---:|---:|
| optimized continuous record | 7.156023 | 4.1667% |
| zero record | 7.152845 | 4.0459% |
| shuffled optimized record | 7.160149 | 3.9251% |
| fixed random continuous record | 7.158069 | 3.8647% |

The optimized continuous record was **0.0444% worse than zero**, versus the
required 20% improvement.  Its two additional correct tokens over zero are far
below the required ten-point accuracy gain.  Because the continuous oracle
failed, neither quantization nor the one-pass writer can be the first causal
bottleneck.

The discrete arms agreed:

- STE-Q16 correct NLL was 7.182554 versus 7.179266 for zero, a 0.0458% loss;
- matched one-pass NLL was 7.180094 versus 7.179266 for zero, a 0.0115% loss;
- every Q16 level index survived BF16 exactly.

The reader specialization also damaged protected natural modeling: continuous
NLL rose from 5.412746 to 5.975954 (**10.4052%**), and Q16 NLL rose to 5.952670
(**9.9750%**), against a 0.5% allowance.

## Interpretation

T22 showed that a one-pass final-state slice did not autonomously write useful
records.  T23 now shows that replacing that writer with per-document iterative
optimization still does not create useful held-out reads.  Therefore the next
move is not a better compiler.  The read contract itself must be identifiable:
the architecture must define how a stored relation is applied to a query,
rather than hoping a generic residual injection learns an arbitrary codebook.

The admissible successor must first demonstrate a constructive read identity
on unseen documents and unseen query/target pairs.  Only after that identity
passes may an autonomous raw-prose writer be trained against it.  Any successor
that merely changes code width, levels, injection coordinates, writer depth,
or optimizer reopens a closed lane.

## Integrity correction

The first execution was retained as invalid because its implementation tested
bit-exact floating-point level values, while the preregistration specified
BF16-stable integer level indices.  The frozen amendment changed only that
integrity test and reported component-level diagnostics.  Training, records,
controls, metrics, thresholds, and classification were exactly reproduced in
the authoritative rerun.

## Evidence hashes

- Authoritative result JSON SHA-256:
  `30bf897b34267d0516dc01453018c85357f08979b851d32a137034b0d65e26f7`
- Authoritative run log SHA-256:
  `8b2647a18dbcc3597516b60828de1238dd65d1c1942c741dac294dacc7a5df50`
- Executed source SHA-256:
  `6fecb58619dbfb07a5b79fe1a17b66cc2b1a140d9543048c3895a1f02775980c`
- Frozen preregistration SHA-256:
  `1d7d69ac456b83402d979cb9e45789554e1c6285fb85fb87d8de8ffa36016f3f`
- Frozen integrity amendment SHA-256:
  `c5590e10b415d1ab619358f82d206ae881763a48c09877afa703d4cb36bdc3a4`
- Invalid first result SHA-256:
  `e5fe78e44c0ed5c6737acb6258d067e1460d11fd41a2beb8db280b60b0d27536`
- Invalid first run log SHA-256:
  `716bd4b3bbe9568b5de68d8107e87012616c8372779c4b130a0edf9019ec76f2`
