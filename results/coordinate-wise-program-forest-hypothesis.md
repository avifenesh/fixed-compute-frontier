# Coordinate-wise program forest — retained successor hypothesis

Status: **not admitted; do not execute before the token-path tree is decided**  
Date: 2026-07-30

## Reason it exists

The active Cayley tree selects one joint path for an entire token.  That may be
the wrong conditional unit: a hidden state can carry many simultaneous
features, while one token-level path forces every transformed coordinate to
share one program address.

## Candidate algebra

Keep the same payload tensor `[node, bit, gate/up/down, coordinate]`, resident
parameter budget, bases, and depth.  Replace the token-level node state
`node[t]` by a coordinate-level state `node[t, i]`.  At every depth, transformed
coordinate `i` chooses its own bit and gathers three scalars from its own tree
node.  The Cayley basis then mixes all coordinate updates before the next
depth.

This changes one selected vector program into a Cartesian product of scalar
programs.  It still reads `3D` learned payload scalars per depth and performs
`O(D log D)` basis/payload work, but it can activate multiple feature programs
inside one token instead of treating the token as one indivisible expert.

## What is not free

- route comparisons and threshold reads rise from `O(log D)` to
  `O(D log D)`;
- route workspace is one small integer per token-coordinate;
- payload reads become highly irregular and may be physically worse than the
  already difficult token-path gather;
- Cartesian route count is not independent learned information;
- per-coordinate learned trees, adaptive scalar nonlinearities, product
  routing, and conditional decision trees all have broad prior-art collisions.

## Re-admission gate

Only consider this successor if the frozen token-path tree fails specifically
through insufficient conditional capacity rather than OOM, nonfinite training,
or hopeless physical execution.  Before GPU rental it must pass:

1. an exact byte/work/workspace ledger including route-state integers;
2. a finite-difference local-rank and route-causality witness;
3. a mixed rotated-target fatal control that the fixed feature DAG failed;
4. a direct comparison with token-level routing and equal-byte narrow MoE;
5. a prior-art check focused on feature-wise MoE, KAN/adaptive activations, and
   product-key routing.

This note preserves the structural successor without treating it as a result
or silently tuning the active tree.
