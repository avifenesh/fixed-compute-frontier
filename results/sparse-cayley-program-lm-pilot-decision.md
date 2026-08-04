# Sparse Cayley program language pilot decision

## Decision: closed

The preregistered 9,994,240-prediction-token experiment was protocol-valid,
but the dynamic Cayley program did not recover the quality lost by replacing a
dense FFN.

Terminal validation NLL:

| Arm | NLL |
|---|---:|
| Full SwiGLU | 6.292342 |
| Equal-parameter narrow SwiGLU | 6.533284 |
| Static Cayley | 7.294289 |
| Dynamic Cayley | 7.288349 |

The dynamic program was 0.0814% better than the static transform, but 11.56%
worse than the equal-parameter narrow SwiGLU.  Its paired delta against the
narrow control was +0.755066 NLL with 95% interval
`[+0.748049, +0.762083]`.  Gap recovery was negative (`-3.1338`).

Routing did not collapse, so this is not explained by an inactive router.  The
reported fixed-route percentage in the machine decision compared 32 fixed
batches against the 128-batch dynamic mean and is invalid.  On the same first
32 terminal batches, dynamic routing scored 7.271986 and the fixed route scored
7.282005: fixing the route was only 0.1378% worse.  This corrected comparison
strengthens the conclusion that input-conditioned routing contributed very
little.

No promotion quality gate passed.  The exact `E4,L2,B2,alpha=0.25,T=4`
recipe is closed; it will not be tuned.

## Evidence integrity

- Result SHA-256:
  `99ed8c7baebbcdbe8003abb6249b14d88b44b62c2d994112bb268a9218e7b5ff`
- Log SHA-256:
  `0464e3c21fcf7c0a56fb9dc9eb97686b1bdbdd91e93cb7a55d77b534f38079d2`
