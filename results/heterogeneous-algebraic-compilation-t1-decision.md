# Heterogeneous algebraic compilation — T1 decision

Status: all preregistered T1 gates passed; advance to adversarial T1b  
Date: 2026-07-30

## Outcome

The finite-field solver recovered the exact hidden 24-of-32 parity support in
all three seeded and permuted worlds.  The compiled, otherwise unchanged
78,404-parameter residual-SwiGLU model achieved:

| world | candidate parity | candidate protected | best AdamW parity at 2,000 steps | best Muon parity at 2,000 steps |
|---|---:|---:|---:|---:|
| 731 | 100% | 100% | 50.907% | 50.981% |
| 947 | 100% | 100% | 51.225% | 50.539% |
| 1213 | 100% | 100% | 50.313% | 49.916% |

The candidate used 256 labeled route-one examples, about 132k–135k recorded
GF(2) XOR-bit operations, parameter writes, and zero dense gradient steps.
Each control endpoint used 128,000 fresh examples and 2,000 dense training
steps.  The controls reached 99.927%–100% on the protected linear route, so
their chance parity accuracy is a composition-specific failure rather than a
failure to optimize anything.

Every overdetermined random-label system was inconsistent, the solver
abstained, and fresh random-label accuracy was 49.847%–50.476%.  All 30 frozen
gates passed.

## What was proved

This is a qualitative capability separation on the exact small system.  It
demonstrates that changing the identification algebra can dominate giving a
general continuous optimizer hundreds of times more examples.  It also answers
the “stronger compiler” objection in this bounded setting: Gaussian elimination
does not model the task distribution generally; it recovers a 32-bit finite-
field rule that the same dense SwiGLU model can execute after the solver is
deleted.

## What was not proved

This is not yet a general-LM breakthrough and not a broad novelty claim.
Gaussian elimination for parity, analytical neural solutions, and compiled
neural modules are prior art.  The test exposes binary coordinates and only
permutes them.  It does not show automatic discovery under an unknown mixed
representation, raw sequential recovery, protected integration into a trained
LM, or a natural-language gain.

Current parity theory also makes initialization a mandatory adversarial
control: discrete Rademacher initialization can make almost-full parity
learnable where Gaussian perturbations do not
(<https://arxiv.org/abs/2412.04910>).  Although its positive result uses a
different two-layer ReLU regime and much larger polynomial width, the control
must be attacked rather than assumed irrelevant.

## Decision

Advance to T1b, not T2.  T1b must add data-independent Rademacher rebirth
controls, replace coordinate permutation with an unknown invertible `GF(2)`
mixing, and recover the rule from consecutive windows of a generated
recurrence.  Failure on any of those closes the claim that the mechanism is
more than a clean-coordinate demonstration.

Artifacts:

- result: `results/heterogeneous-algebraic-compilation-t1.json`
- source SHA-256:
  `e1ecb03c260aa1554d860fbbb724772fa4b83293a1aa85e349c527dd7c707a8d`
- test SHA-256:
  `2d5e5aced9c74f13f4e2f580e6477d39f63ddc31b07eb200e1a2b464ebf90ea1`
- preregistration SHA-256:
  `d4ddcaa47cf73820ab621ff18ddc8fc8fbd0a81cba96fc21a9d44bcb5af127eb`

