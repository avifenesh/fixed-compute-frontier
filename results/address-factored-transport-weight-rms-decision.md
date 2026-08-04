# Address-factored transport: weight-RMS successor decision

## Decision

**FAIL. Close AFTA.**

The sole permitted successor did not make address-factored transport attention
reliable. The frozen result is complete, independently reconstructed, and not
an execution artifact. Per the preregistration, do not rescue this branch with
SVD constraints, longer training, changed head allocation, task reweighting, or
relaxed thresholds.

## Exact outcome

Only three of ten conjunctive gates passed: integrity, no-shortcut, and strict
ordinary-export integrity. The following seven failed:

| gate | candidate result | required result |
|---|---:|---:|
| shared gain | `-1.6772` points vs best noncandidate | at least `+5` points and every-seed win |
| hard shared gain | `+0.5151` points | at least `+5` points |
| independent protection | `-3.4717` points vs `split12` | mean no worse than `-1` point and every seed no worse than `-2` |
| ordinary protection | seed `1506` was `-59.6680` points | every seed no worse than `-2` points |
| overall gain | `-0.0822` points vs `split12` | at least `+2` points |
| causal use | two seeds had negative wide-path ablation drops | every seed at least `+5` points |
| score guard | 16 of 20 task-seed comparisons were below `0.5x` | every comparison in `[0.5x, 2.0x]` |

Mean accuracies were:

| arm | shared | hard shared | independent | ordinary | overall |
|---|---:|---:|---:|---:|---:|
| `hybrid_afta` | 0.742334 | 0.630884 | 0.431079 | 0.670190 | 0.614535 |
| `split12` | 0.752539 | 0.625732 | 0.465796 | 0.627734 | 0.615356 |
| `split12_temp2` | 0.759106 | 0.618579 | 0.468726 | 0.759521 | 0.662451 |

The candidate beat the strongest shared-address competitor only in seed `2555`.
It lost in the other four seeds, with shared-accuracy margins of `-6.65`,
`-21.64`, `-5.29`, and `-45.21` points. The training-only radial constraint
therefore did not remove the original basin bifurcation; it yielded one healthy
wide map and four under-scaled ones.

## What survives

The Stage-0 theorem and successful learned seeds showed a real mechanism: one
content address can feed different relation transports to different channel
groups, and that wide path can be causally used. What failed is the stronger
architecture claim that this factorization provides a dependable capability
edge under matched served resources.

Retain relation-specific transport as an algebraic primitive. Do not retain the
one-wide-plus-eight-narrow AFTA allocation as a model candidate.

## Integrity

- all 255 expected evaluations were present and unique;
- all source, preregistration, dependency, initialization, stream, and ledger
  hashes/checks matched;
- all 25 exports loaded strictly into the ordinary served class;
- maximum export logit error was exactly zero;
- the complete ten-gate reconstruction matched the stored `pass=false`;
- result SHA-256:
  `2541f868e801d9ec8b26891d9830768f0cc3a519e1791698e9284bfa9f85530a`.

