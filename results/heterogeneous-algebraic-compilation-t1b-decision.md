# Heterogeneous algebraic compilation — adversarial T1b decision

Status: all 45 frozen gates passed; advance to causal-architecture gate  
Date: 2026-07-30

## Result

The candidate recovered the exact induced parity mask and reached 100% full,
parity, and protected accuracy in all six cells: three worlds crossed with two
views (unknown dense `GF(2)` mixing and consecutive recurrence windows).

The secret mixing matrices had densities `0.4873`, `0.4980`, and `0.4961` and
were not permutations.  Their induced observed parities had degrees 18, 17,
and 13.  Every system had rank 32.

The best data-independent Rademacher-rebirth controls remained at chance on the
compositional route after 2,000 AdamW steps and 128,000 examples:

| world | view | best BCE parity | best hinge parity | best protected |
|---|---|---:|---:|---:|
| 731 | mixed | 50.397% | 51.661% | 100% |
| 731 | recurrence | 51.512% | 51.906% | 100% |
| 947 | mixed | 50.245% | 49.951% | 100% |
| 947 | recurrence | 50.888% | 49.975% | 100% |
| 1213 | mixed | 50.399% | 50.544% | 100% |
| 1213 | recurrence | 50.729% | 52.161% | 100% |

Controls covered three Rademacher scales, two learning rates, BCE and signed
hinge objectives.  The one numerically bad hinge endpoint does not create the
result: its parity accuracy was still chance, and stable controls in the same
cell also learned the protected route.

## Finding

The large separation is not explained by standard Gaussian initialization,
exposed feature coordinates, independent training examples, or failure of the
controls to learn a simple rule.  In this exact family, a narrow solver changes
the sample/optimization problem: approximately 135k finite-field XOR-bit
operations identify a compact rule, while continuous training does not acquire
the rule after 128,000 examples per run.

This is still not a general-LM breakthrough.  The serving network receives a
flattened fixed context; it does not yet establish that the rule can be placed
inside an existing causal attention/FFN decoder.  Compiled symbolic neural
modules and hand-constructed parity Transformers are prior art.  The surviving
novelty seam is automatic charged discovery plus compilation into a fixed
pre-existing parameter budget with protected coexistence.

## Decision

Advance to a causal-architecture gate.  It must recover the hidden rule from
raw token sequences and program ordinary causal attention plus SwiGLU weights,
with no added module or deployed state.  The protected query must still work,
and tuned gradient controls must receive 2x and long-horizon budgets.

Integrity:

- source SHA-256:
  `2e2c789d81ad0737a44c724ad73d41350f02f49ba0108dedde5e3c49dee6d501`
- test SHA-256:
  `af293ec425b9387154f2a2105d8141b2e21416e1839f149cf2334125fe522782`
- preregistration SHA-256:
  `0fad365959686762ddcfcd3d8530f843a13e92b2cd508f46052b6a4716263ac2`
- predecessor result SHA-256:
  `e74fc24340ce11ac967ed36a26afcfcac234ba566fc4e80c6697793439cd64cd`

