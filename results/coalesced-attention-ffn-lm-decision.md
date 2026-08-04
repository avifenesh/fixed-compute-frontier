# Projection-coalesced attention/FFN — LM decision

## Verdict

**Close the projection-coalescing branch under its claim.**

The candidate is a real Pareto improvement over the ordinary sequential
scratch Transformer, but it is dominated by the preregistered independent
parallel control.  Since the control uses the same dense projection ledger as
the sequential baseline and is also faster in the exploratory H100 cells, the
candidate is not a new fixed-cost frontier point.

Machine-readable result SHA256:
`9d7e7999ac42978743a88baf503cd1ec8763af6b48f7757694371f458b12ff1e`.

## Frozen terminal results

| arm | 10M NLL | 50M NLL | dense projection params/MACs per layer |
|---|---:|---:|---:|
| sequential baseline | 6.314061720 | 5.533173133 | 1,572,864 |
| independent parallel baseline | **6.133308645** | **5.452698179** | 1,572,864 |
| coalesced width 1,024 | 6.220147058 | 5.550511498 | 1,179,648 |
| coalesced width 1,344 | 6.211749576 | 5.527354565 | **1,548,288** |

The width-1,344 candidate:

- improves 0.10516% over sequential, with paired candidate-minus-baseline NLL
  `-0.005819`, 95% interval `[-0.009336, -0.002301]`;
- improves 0.41720% over the width-1,024 tied control, with interval wholly
  favorable;
- is 1.36916% **worse** than the independent parallel control, with paired
  difference `+0.074656`, 95% interval `[+0.073255, +0.076058]`.

The dominant failure is far larger than seed noise or the promotion margin.
Do not run another seed.

## Causal interpretation

1. **Parallelization did not cause a capability loss.**  It improved terminal
   NLL by 1.4546% over the ordinary sequential block at the same dense
   projection parameters/MACs.
2. **Projection independence matters.**  At width 1,024, tying Q/K/V/O to the
   FFN costs 1.7938% NLL versus the parallel control.
3. **Budget reinvestment works but is insufficient.**  Expanding the tied
   width from 1,024 to 1,344 recovers 0.4172%, leaving most of the independence
   loss.
4. **The shared channels are active.**  Zeroing local products on the 640
   attention-bearing channels raises NLL from 5.52735 to 7.21448.  This
   ablation is intentionally broad: it proves the rows carry local capability,
   not that every individual row needs both roles.
5. **The failure is not numerical.**  Training and diagnostics remained
   finite.  The candidate did fail the conservative worst-layer activation
   envelope: 1.42% of sampled initial values and 4.39% of one terminal layer
   exceeded four times the corresponding sequential median RMS, against a 1%
   gate.  The parallel control also has a heavy worst-layer tail, so this is a
   secondary stability warning rather than the causal rejection.

## Resource interpretation

The candidate used 1.56% fewer dense projection weights/MACs per layer and
0.78% fewer total model parameters than the independent parallel control.
Measured training throughput was 324,087 tokens/s versus 327,480 tokens/s for
parallel and 300,673 tokens/s for sequential.  In the earlier unfused
inference diagnostic it was 12.9% to 25.6% faster than sequential, but 4.9% to
15.5% faster than parallel only at some scratch cells; those measurements were
not a production serving claim.

The independent parallel control therefore dominates the candidate on both
capability and measured training throughput while differing by less than 1%
in total parameter count.

## Retained knowledge

- The exact ledger and width-704 containment construction remain valid.
- Dense attention and FFN projections can technically share full-strength
  activation banks, and the resulting model trains stably.
- At this scale, representation independence is worth more than a 31.25%
  increase in tied nonlinear width.
- The next candidate must use independent readouts from reused generators; it
  must not force attention selection, attention transport, and local memory to
  occupy the same columns.
- The independent parallel block becomes the required architecture baseline
  for subsequent FFN algebra screens.  Its gain is known prior art (including
  PaLM), so it is evidence and a baseline, not our claimed discovery.

No repair by a different slice map, learned routing, partial sharing, or more
width is allowed under this branch.
