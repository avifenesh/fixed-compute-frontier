# Tile-routed cross-gate matched LM screen preregistration

Decision frozen before observing any learning result.

## Claim under test

One SwiGLU layer computes only one fixed matching between its gate and up
features.  A 256-channel tile can instead select one of four cyclic matchings
using four gate values already computed inside that tile:

```
g = Gx
u = Ux
route[tile] = argmax(g[tile, 0:4])
u_route = cyclic_shift_within_tile(u, route[tile])
u_eff = u + alpha[layer] * (u_route - u)
z = SiLU(g) * u_eff
y = Vz
```

`alpha[layer] = 2*tanh(beta[layer]/2)` and every beta starts at zero, so all
arms are exactly the original checkpoint at step 0.  Hard routes are used in
the forward pass and a softmax straight-through estimator is used only for the
route gradient.  There is no router matrix, expert matrix, or stored
permutation table.

## Frozen arms

1. `baseline`: original pairing; beta is present but has zero effect.
2. `static_local`: every tile uses cyclic shift 1; controls for the blend and
   local memory movement without conditional computation.
3. `static_balanced`: tile `t` always uses route `t mod 4`; controls for a
   heterogeneous but token-invariant local wiring pattern with balanced
   aggregate route counts.
4. `global_routed`: one four-way route, read from the first four gate values,
   is applied to every tile in a token; controls for ordinary token routing.
5. `tile_routed`: every 256-channel tile chooses its own four-way route from
   its first four gate values.
6. `tile_routed_reindexed`: the same tile-routed arm in a fixed random logical
   reindexing of each FFN hidden dimension.  Gate/up activations are permuted
   before tile routing and exactly inverse-permuted before the unchanged down
   matmul.  At zero blend this preserves both values and BF16 accumulation
   order bit-for-bit; at nonzero blend it is the channel-serialization gauge
   control.

Every arm has one scalar beta per FFN layer and therefore identical parameter
and optimizer-state counts.  All shared parameters start from the same model
revision and see identical batches in identical order.

## Frozen protocol

- Model: `HuggingFaceTB/SmolLM2-135M`, revision
  `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.
- Existing immutable FineWeb-Edu token files and manifest from the Reflex
  screen.
- H100, PyTorch `2.5.1+cu124`, Transformers `4.57.6`.
- Sequence length 512; microbatch 32; accumulation 2.
- 320 optimizer steps = 10,485,760 prediction tokens per arm.
- Evaluation: all 4,096 fixed validation sequences at steps 0, 16, 80, 320.
- AdamW: shared LR `1e-4`, beta LR `5e-4`, weight decay `0.1` on shared
  parameters only, betas `(0.9, 0.95)`, clip norm 1.0.
- Warmup 16 steps, then cosine decay to 10% of peak.
- Seed 131.  No compilation and no checkpoint selection.
- Tile width 256, four cyclic shifts `(0, 1, 2, 3)`, route softmax temperature
  1.0.  These values will not be tuned from this result.

## Frozen advance gates

All integrity, endpoint, finite-training, parameter-count, and route-accounting
gates must pass.  At step 320, all of the following must also hold:

1. Both tile-routed arms are at least 0.05% lower NLL than `baseline`.
2. The paired 95% intervals for both tile-routed arms minus baseline are
   wholly below zero.
3. The worse tile-routed arm is at least 0.025% lower NLL than the best of
   `static_local`, `static_balanced`, and `global_routed`.
4. The paired 95% interval against that best control is wholly below zero.
5. For both tile-routed arms, at least 75% of exact `(layer, tile)` cells use
   two routes above 5% over validation tokens, and median within-cell
   normalized route entropy is at least 0.50.  Aggregate balance alone cannot
   pass this gate.
6. For both tile-routed arms, mean absolute effective alpha is at least
   `1e-4` and no alpha is saturated above 1.9 in magnitude.
7. The two tile-routed terminal losses differ by less than 0.025% relative;
   otherwise any gain is checkpoint channel-serialization luck.

Passing this screen means only that conditional tile-local feature pairing has
a real matched-learning signal.  It does not yet prove target-scale quality,
exact zero-byte deployment, or a <=1% fused H100 serving overhead including
route calculation.  Failing any quality gate closes this frozen recipe; no
post-hoc LR, shift, tile-size, or step extension is allowed.

Before launch, a separate frozen integrity manifest must contain and validate
the SHA-256 hashes of this preregistration, the experiment source, and its unit
test.  Runtime self-reporting alone is not accepted.
