# Tile-routed cross-gate matched LM screen — decision

Decision: **reject this routing/rebinding recipe; do not tune, extend, or fuse it.**

The frozen six-arm SmolLM2-135M screen completed at 320 optimizer steps and
10,485,760 training tokens per arm.  The candidate preserved the exact
pretrained endpoint, routed nontrivially, and learned live nonsaturated blend
coefficients.  It nevertheless failed the capability gates by a wide margin.

## Terminal validation loss

| Arm | Loss | Relative improvement over baseline |
|---|---:|---:|
| Baseline | 2.817755731 | — |
| Static local shift | **2.817692781** | 0.002234% |
| Static balanced shifts | 2.817742348 | 0.000475% |
| Global routed shift | 2.817726139 | 0.001050% |
| Tile routed | 2.817732856 | 0.000812% |
| Tile routed, reindexed gauge | 2.817683168 | 0.002575% |

The preregistered requirement was that both tile arms improve on baseline by at
least 0.05%.  The worse tile arm reached only 0.000812%, about 61.6 times below
that threshold.  It was also 0.001422% worse than the best cheap control,
whereas promotion required at least a 0.025% improvement over that control.

The paired 95% interval for ordinary tile routing versus baseline crossed zero:
`[-8.813e-5, 4.238e-5]` NLL.  The reindexed arm's interval was below zero, but
its effect was still tiny and the paired tile-versus-best-control interval
crossed zero: `[-3.659e-5, 1.167e-4]` NLL.  Thus neither the effect-size nor the
control-separation gates passed.

## The mechanism was active

This is not a dead-router result.  Both tile arms had 97.22% of layer/tile
cells use at least two routes above 5%, with median normalized route entropies
0.8940 and 0.8845.  Their mean absolute learned blend coefficients were
0.001148 and 0.001045; maximum absolute coefficients were 0.003804 and
0.005540, with no saturation.  All losses and gradients remained finite.

The logical reindex control agreed with the ordinary tile arm to 0.001763%
relative terminal loss, inside the frozen 0.025% gauge-robustness bound.  The
failure therefore does not reduce to an unlucky serialized feature ordering.

## What this closes

The tested construction conditionally re-paired already-computed gate and up
features while keeping the three dense FFN matrices unchanged.  Active routing
did not produce a meaningful language-learning advantage over a fixed shift.
This closes local permutation routing as the sought fixed-cost mechanism under
the frozen upcycling protocol.  More routes, a longer run, router tuning, or a
custom fused kernel would be post-hoc continuation without evidence of useful
capability.

The retained lesson is narrower and constructive: the next activation-side
algebra must create genuinely new inter-feature products, not merely select a
matching among existing projected scalars.  It must also have an exact dense
endpoint and be tested against an equal-cost static control.

## Integrity and cost boundary

- Exact endpoint: bitwise-equal initial logits and loss; maximum loss
  difference `0.0`.
- Identical parameter count: `134,515,038` in every arm, including 30 scalar
  blend parameters reserved in the baseline.
- Frozen source, preregistration, and test hashes all validated at runtime.
- Eight focused tests passed on the rented H100 environment.
- Result SHA-256:
  `d9808b6159f59ce977688007714d1d5b6188b3bb38884606a22d2b34386d8baf`.

The eager Python throughput numbers are diagnostic only, not serving claims:
111,143 tokens/s for baseline, 76,636 for static local, and 60,308 for tile
routing.  Because the candidate failed capability, no fused serving benchmark
is authorized.
