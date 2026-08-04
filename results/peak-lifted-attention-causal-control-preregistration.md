# Peak-lifted attention causal-boundary control — preregistration

Status: frozen before execution  
Date: 2026-07-27

## Question

Does the Stage-0 peak-lift direction survive the strongest cheap structural
objection: a causal partial chunk has fewer chances to produce a large maximum,
so fixed chunks may create a position-modulo-chunk prior rather than useful
content routing?

This is a mechanism screen, not a language-quality or latency claim.

## Frozen arms

For token scores `s_j` in chunk `b`, all arms preserve the ordinary conditional
softmax inside the chunk and add a chunk-common lift to the logits.

1. `ordinary`: zero lift.
2. `raw_peak`: `0.5 max_b(s)`.
3. `normal_corrected_peak`: `0.5(max_b(s)-kappa_n)`, where `kappa_n` is a
   separately Monte-Carlo-calibrated expected maximum of `n` standard-normal
   scores. This is an intentionally favorable oracle control for the peak
   idea, not yet a deployment recipe.
4. `confidence`: `0.5(n-l_b)/(n-1)` for `n>1`, zero for `n=1`, with
   `l_b=sum exp(s_j-max_b(s))`.
5. `global_temperature`: `1.5 s`, included on the anchor task only.

Chunk width is 64. Fixed origins 0 and 32 and query-relative backwards chunks
are evaluated. Random-score trials use held-out samples, never the samples used
to estimate `kappa_n`.

## Frozen probes

### Neutral IID fairness

For causal lengths `3C+r`, `r=1..C`, use 4,096 held-out rows for standard
normal, standardized Laplace, standardized Student-t(5), and an all-equal
control. For each chunk, compare candidate probability mass with ordinary
probability mass on the identical score row. Exchangeability makes expected
ordinary mass exactly its fair token-count share `n/N`; the paired difference
is used as a control variate. Record the relative paired mean, its standard
error, the five-standard-error absolute bound `abs(mean)+5*SE`, the latest-
chunk deviation, and the range across `r`. The factor five is frozen as a
conservative multiplicity guard across the full grid.

The raw operator fails as a general causal mechanism if any IID family has
more than 5% maximum five-SE chunk-share distortion. The oracle correction is
not admitted unless its five-SE bound stays within 5% on every nonconstant
held-out family and within 1% on the all-equal control. Confidence is reported
as the hardware-native count-aware control under the same thresholds.

### Anchor-to-payload capability witness

Use 16,384 rows of four full chunks. Insert one score-4 anchor and a score--0.5
payload in one chunk, and a score-3.8 distractor anchor in another chunk.
Record the target payload probability relative to ordinary attention. Repeat
over all 64 anchor positions with the payload eight positions later modulo the
sequence, under fixed origins 0 and 32. Identical random score rows are reused
across origins. Report mean and minimum uplift separately when the anchor and
payload share a chunk and when they cross a chunk boundary.

The mechanism signal requires at least 10% mean payload-probability uplift
over ordinary, must beat global temperature, and must retain nonnegative
minimum crossing-boundary uplift under the better of the two fixed origins.
This is only a synthetic causal witness.

## Interpretation rule

- If raw peak fails IID fairness, fixed raw peak is rejected even if the anchor
  witness passes.
- If normal correction passes only the distribution used for calibration, the
  failure is structural: no distribution-free correction has been shown.
- If confidence passes fairness and the anchor witness, it replaces raw peak
  as the live algebraic variant; otherwise the entire chunk-statistic family is
  paused before language training.
- Query-relative chunking may be retained only as a decode-oriented systems
  option; it cannot rescue a statistic that fails token-count fairness.

No threshold or arm may change after the result is viewed.
