# 002 — Attention Evidence Scaling

Status: **closed theoretical negative result**  
Date: 2026-07-23  
Primary attempted edge: D/E

No GPU experiment was run. Independent review rejected the causal and resource
argument before implementation.

## Claim tested

Online softmax already computes the maximum score and normalization sum. The
candidate proposed using the resulting maximum attention probability to scale
the ordinary head output:

\[
p_{\max}=1/\ell, \qquad O'=(1+p_{\max})O.
\]

The intended gain was to make attention concentration affect downstream
computation without adding learned bytes, KV bytes, output width, or another
attention scan.

## What survived

- For a nonempty, dropout-free row after the global online-softmax reduction,
  `p_max = 1/l` is mathematically valid and is cheap to obtain inside the fused
  kernel.
- Concentration contains information that the normalized value mean can omit.
- The current primary-source search found no exact precedent for this specific
  `p_max` output modulation under the complete fixed-resource claim.
- Reusing an on-chip statistic remains a legitimate search class, but not a
  capability gain by itself.

## What failed

1. The duplicate-value argument was overstated. Duplicating one value among
   other values generally changes the ordinary weighted mean; it is invariant
   only in restricted constructions such as an all-identical value set.
2. Scaling does not expose `p_max` independently. The map
   `(O, p_max) -> (1 + p_max)O` is non-injective and confounded with ordinary
   value/output magnitude. If conflicting values cancel to `O = 0`, the
   candidate still outputs zero for every concentration.
3. `p_max` is concentration, not correctness or evidence confidence. Q/K scale,
   context length, temperature, and repeated keys can change it without a
   corresponding change in correctness. The model could use it as an arbitrary
   adaptive amplitude gate.
4. The candidate does not contain the baseline. Every head is forced through a
   variable scale, so diffuse averaging heads can regress and `W_O` cannot
   globally undo the query-dependent factor.
5. The proposed whole-model toy did not isolate the mechanism. A learnable
   transformer can encode duplication or conflict through values, positions,
   residual paths, other heads, or later layers even when one local attention
   read aliases.
6. The constant-scale control was vacuous because a constant is absorbable into
   the output projection. It did not control for a generic dynamic gate.
7. The proposed three-seed, two-standard-deviation rule was not a valid paired
   decision procedure.
8. The hardware gate stopped at an attention block and therefore could not
   establish end-to-end TTFT, TPOT, sustainable throughput, energy/token, and
   GPU-dollar parity.

## Knowledge retained

1. A statistic being available inside a kernel does not mean it can be added to
   the model's representation for free.
2. An independent scalar cannot be losslessly added to a fixed-width continuous
   head output without sacrificing another degree of freedom, increasing width,
   or using a lossy encoding.
3. Modulating an existing vector is not the same as exposing a new feature.
4. Attention concentration must not be described as confidence without a
   separate causal calibration result.
5. Operator-level aliasing is not automatically a whole-transformer lower
   bound; the rest of the network may route around it.
6. A resource claim must finish at the full server and workload trace, not at a
   fused-kernel microbenchmark.

## Branch closure

This branch must not continue by substituting another scalar function of
`p_max`, entropy, or LSE into the same output-rescaling claim. Replacing a value
channel with a dedicated statistic would be a materially different allocation
claim with an explicit capacity donor and requires a fresh admission argument.
