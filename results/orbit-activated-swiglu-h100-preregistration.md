# Orbit-activated SwiGLU — fused H100 preregistration

Status: **frozen before execution**.

This is a serving gate, not a language-capability result.

## Compared paths

Both paths use BF16 `G,U,V` with `D=4096`, `M=14336` and execute the same three
dense matrix multiplications.

1. `baseline`: fused ordinary SwiGLU activation.
2. `orbit`: group-of-eight orbit-activated recurrence.  Its Triton kernel reads
   each coefficient from the deterministic semantic pivot of the actual packed
   `U` matrix.  Passing a duplicate coefficient tensor is forbidden.

The frozen executor issues one logical pivot load per token and hidden feature,
or `B*M` load instructions.  Cache reuse may reduce physical HBM transactions,
but the ledger must not count only `M` reads.  This first feasibility gate also
uses separate gate/up GEMMs in both arms; passing it does not yet establish
parity with a production executor that coalesces those projections.

The full captured path is `x@G.T`, `x@U.T`, fused activation, `z@V.T`.
Batch/token counts are `1,8,32,128`.  Each cell uses at least 40 warmups and 200
random-order CUDA-event samples inside captured CUDA graphs.

## Correctness gates

1. With zero encoded coefficient, orbit output matches the fused baseline with
   maximum row-relative error at most `0.2%` in every cell.
2. A nonzero carrier test must contain negative-clipped, unclipped, and
   positive-clipped predecessor values and match a Torch float32 reference
   within `1%` maximum row-relative error.
3. Coefficients must be decoded from BF16 `U` pivots with maximum absolute
   error at most `0.001` for encoded values in `[-0.035,0.035]`.
4. The candidate allocates no persistent coefficient tensor and no additional
   activation-sized workspace.
5. Report coefficient absolute-error percentiles, sign flips among coefficients
   with magnitude at least `0.001`, zero-collapse rate, clip-region occupancy,
   and `max(max(|s|,1/|s|))`.  The frozen carrier range must keep this condition
   factor below `1.5`.
6. Compare folded-BF16 candidate output to its canonical-float32 reference and
   also report ordinary folded-BF16 SwiGLU error against ordinary canonical
   float32.  Candidate excess row-relative error over that base export error
   must be below `0.5%`.  This is still not a substitute for BF16 validation
   NLL in the later language screen.

## Latency gates

- orbit/baseline median at batches 1 and 8: at most `1.02x`;
- orbit/baseline median over the full grid: at most `1.05x`;
- report absolute distributions, not only ratios.

Passing authorizes the matched language screen.  Failure closes this exact
group-of-eight executor before training.  A launch-only or activation-only
timing cannot satisfy this protocol.

The endpoint contains ordinary SwiGLU on the open dense set where every
deterministic `U` pivot is nonzero.  No global-containment claim is made for
the measure-zero excluded charts.
