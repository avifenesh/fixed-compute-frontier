# TVE replay-autograd v4 decision: exact but too slow

## Formal decision: NO-GO

V4 repaired the numerical failure: forward and complete dense/active physical
weight gradients were bit-exact in the tail, 37.8M, and 360M cases. Value
gradients passed the unchanged error contract and the signed closed form passed.

It failed the frozen operator-time gate:

| Geometry | Replay/current time | Required | Peak replay/current bytes |
| --- | ---: | ---: | ---: |
| 37.8M | 0.8761x | <= 0.75x | 193,071,616 / 253,889,024 |
| 360M | 0.8594x | <= 0.75x | 157,028,352 / 192,008,192 |

Replay saves memory but not enough time. Do not admit it to full-model TVE
integration under the frozen claim.

The mechanism result also makes the original goal obsolete: the nonlinear TVE
forward was unnecessary on the causal seed. Future kernel work should target an
identity-forward Ghost Gradient operator, where only the virtual VJP is added
during training and all inference machinery is deleted. The fast v3 reduction
can be reconsidered only as a different approximate training algorithm with a
new, trajectory-level noninferiority test; its failed exactness gate must not be
reinterpreted.

Bound result: `results/triangular-value-encoding-autograd-v4.json`, SHA-256
`36bcfff814bf9a50ce1f58a7ee1e4fc292379e646d6ccabeb3b3c28c01307972`.
