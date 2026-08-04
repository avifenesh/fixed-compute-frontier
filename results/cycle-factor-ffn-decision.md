# Cycle-factor FFN decision

## Verdict

Close arbitrary static factor reuse before replication or kernel work.

The candidate was genuinely smaller and algebraically distinct, but its language model was worse:

| arm | parameters | 10M-token NLL |
|---|---:|---:|
| full SwiGLU | 37,753,728 | **6.2948176** |
| exact-2DM narrow SwiGLU | 33,035,136 | 6.3200835 |
| width-M self-product | 33,035,136 | 6.3167478 |
| width-M cycle factor | 33,035,136 | **6.3241085** |

Cycle factor lost 0.4653% to full SwiGLU, 0.0637% to the exact-cost narrow control, and 0.1165% to self-product. Every paired interval was wholly unfavorable.

This was not a dead-mechanism result. Replacing the learned cycle relation with self edges after training raised NLL to 13.40294; changing to the second neighbor raised it to 11.21010. The weights strongly co-adapted to the relation, but arbitrary static adjacency was a worse inductive bias.

Stage 0 remains a valid algebraic result: at the equal 72-parameter witness, cycle factor had full sampled functional rank 72 versus 64 for narrow SwiGLU, and its quadratic forms were indefinite rather than self-product's positive-semidefinite squares. Functional rank did not predict useful language features.

Result SHA-256: `c66dfe6f5a6e3a1c4032932b9cbd2c491812af2e86a314e9cc6d46e83aba4029`.

## Retained boundary

Do not tune cycle length, signs, fixed permutations, or a same-token mixing angle. Re-admit factor reuse only when the reused operand already has independent semantics. The immediate successor is temporal reuse: the previous token's generated value is meaningful before the factor graph is imposed.
