# TVE direct-native five-seed decision

Decision: **PASS for the frozen 37.8M / 100M-token mechanism confirmation.**

This is not a strict A-E capability or deployment pass. It admits triangular
value encoding (TVE) to mechanism, native-cost, and model-scale gates.

## Primary result

Five untouched seeds compared TVE directly with the packed raw transformer on
the pinned FineWeb-Edu corpus, with identical architecture size, data order,
optimizer, prediction tokens, state bytes, and validation batches.

| Seed | raw NLL | TVE NLL | TVE - raw | relative NLL gain |
|---:|---:|---:|---:|---:|
| 8,821 | 5.08536223 | 5.08017271 | -0.00518952 | 0.102048% |
| 9,901 | 5.09130772 | 5.08434324 | -0.00696448 | 0.136792% |
| 11,003 | 5.09306635 | 5.08912825 | -0.00393810 | 0.077323% |
| 12,109 | 5.10539649 | 5.10146355 | -0.00393295 | 0.077035% |
| 13,217 | 5.09410487 | 5.08691192 | -0.00719295 | 0.141202% |

The raw mean was `5.0938475341`; the TVE mean was `5.0884039328`.
The ratio-of-means relative NLL improvement was `0.1068662%`, equivalent to a
`0.5428812%` perplexity reduction.

Using the seed as the independent unit, the mean paired TVE-minus-raw NLL was
`-0.0054436013`. Its two-sided 95% Student-t interval with 4 degrees of freedom
was:

```text
[-0.0074055172, -0.0034816855]
```

The interval is entirely favorable and all five individual seed means favor
TVE. TVE also favored raw at every frozen intermediate checkpoint (10M, 25M,
and 50M prediction tokens).

## Path-use ablation

Disabling encoding in each terminal TVE checkpoint increased loss. The
full-minus-disabled seed means were:

```text
[-0.0778106004, -0.0891591758, -0.0899130106,
 -0.0839163810, -0.0907429755]
```

The seed-clustered 95% interval was
`[-0.0930729325, -0.0795439248]`. This proves the trained models use the TVE
path. It does not distinguish useful forward representation from co-adapted or
optimizer-mediated effects; the frozen backward-only control owns that test.

## Integrity

- Every frozen decision gate passed.
- All five initial candidate/raw intervals were within the declared `0.02%`
  equivalence band.
- Parameter, tensor, buffer, state, optimizer, and metadata ledgers matched.
- All values were finite; coefficient and serving-bridge gates passed.
- Fifteen remote checkpoints/exports had unique expected paths and matching
  byte counts and SHA-256 hashes.
- Independent recomputation from the raw 64-batch vectors matched the embedded
  result exactly.
- Result SHA-256:
  `652ce5a231ec721596a33dbc96c638e348fbb92ca2ce6c2412068436215ebd71`.

## Interpretation boundary

Established at this scale:

1. The 10M-token discovery gain did not disappear by 100M tokens.
2. The direct complete recipe beat the raw transformer in five new
   initializations.
3. The effect is small but larger than the frozen practical floor.
4. The nonlinear path is used after training.

Not established:

1. that forward representational capacity, rather than the changed training
   signal, causes the gain;
2. a second corpus or a pretrained 360M checkpoint;
3. full-model training-cost equivalence with the custom backward;
4. native complete-attention latency or service cost;
5. protected capability slices, 400M multi-seed replication, or strict A-E
   dominance.

The next decision is therefore not another architecture idea. It is the
backward-only mechanism falsifier, followed by native-cost and 360M
second-domain gates if the mechanism survives.
