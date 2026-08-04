# Block-triangular microdepth FFN — 50M-token decision

Decision: **reject the two-projection triangular recipe and do not spend a
second seed or an integrated serving kernel on it.**

The run was healthy and exactly budget matched, but the nonlinear recurrence
did not recover the capability lost by removing SwiGLU's multiplicative gate.

## Frozen terminal result

| Arm | Validation NLL | Relative to SwiGLU |
|---|---:|---:|
| SwiGLU, width 1,024 | **5.527706977** | — |
| Plain SiLU, width 1,536 | 5.590641335 | 1.13855% worse |
| Postactivation linear recurrence | 5.590996005 | 1.14415% worse |
| Preactivation triangular recurrence | 5.590961680 | 1.14432% worse |

The ordering was already decisive at 10M tokens: `6.293972343` for SwiGLU
versus `6.331617702` for the candidate.  The candidate therefore failed both
the frozen early noninferiority gate and the terminal `0.1%` improvement gate.

## What the ablations identify

| Candidate evaluation | NLL |
|---|---:|
| Full recurrence | 5.590961680 |
| All `H` couplings zeroed | 5.591177888 |
| Only direct one-hop predecessors retained | **5.590952978** |

Zeroing `H` worsened mean paired block loss by `0.00021621`, with a nominal
95% interval `[-0.00024597, -0.00018645]` for full-minus-zero.  The couplings
therefore learned a real but tiny correction.  Removing paths longer than one
hop instead improved NLL by `0.00000870`; its interval
`[-0.00000638, 0.00002379]` spans zero.  The proposed compositional microdepth
mechanism did not emerge.

The postactivation recurrence was functionally absorbable and finished within
`0.000355` NLL of plain SiLU.  This validates it as a useful negative control:
serial parameterization alone supplied no gain.

## Exact conclusion

The experiment does not say that short scalar dependencies can never help an
FFN.  It says that spending the SwiGLU gate matrix on 50% more shallow SiLU
features plus group-local triangular couplings is a losing exchange.  Almost
all of the loss is explained by the missing multiplicative gate; learned
one-hop correction is roughly 290 times too small to close the terminal gap,
and learned multi-hop composition contributes no measurable value.

The retained hardware primitive remains a capped group of at most eight
serial scalar operations.  Any successor must preserve ordinary SwiGLU at an
exact endpoint and fund the scalar couplings without reducing its three dense
projections.

## Validity

- All four arms had exactly `37,758,336` total and trainable parameters.
- Each arm consumed `49,971,200` prediction tokens from the same immutable
  scratch stream under the same optimizer schedule.
- Training and gradients remained finite and bounded.
- Twelve focused algebra/model tests and full-model BF16 forward/backward
  smoke passed on the retained H100 before the frozen run.
- Source, test, preregistration, algebra-result, data, and protocol hashes were
  checked before training.

Result SHA-256:
`891f384a98414a73a69ebb92e74a829870d6c768db684381db05d1567c6bd8bb`.

