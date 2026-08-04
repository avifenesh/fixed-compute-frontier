# Nonlinear-reduction SwiGLU — fused H100 fatal gate

## Admission and claim

Stage 0 established that the associative dual-number reducer adds one robust
functional direction at an exact ordinary-SwiGLU endpoint, is independent of
K-block order/reduction tree, and has much tighter path-gain tails than the
serial reducer.  This gate asks whether its reduction can execute without an
output workspace or unacceptable H100 latency.  It does not test language
quality or novelty.

The clean deployment chart uses selected post-attention RMSNorm slots only as
coefficient carriers.  At conversion, their old scales are folded into the
corresponding gate/up columns; normalization uses scale one at those positions
thereafter.  `alpha=0.25*tanh(carrier-1)` is derived at model load and passed
as scalar kernel metadata.  There is no alpha tensor or persistent byte.

## Frozen implementation

- H100, PyTorch `2.5.1+cu124`, CUDA `12.4`, and isolated Triton `3.6.0` from
  `/workspace/triton36`; resident BF16 weights/inputs/outputs and FP32
  accumulators.  Triton 3.1.0 is rejected because its Hopper MMA-layout
  conversion aborts on the associative combine.
- Target FFN shape `D=4096`, `M=14336`, four consecutive K blocks of 1,024.
- One packed resident gate/up weight tensor plus one down tensor in every arm:
  exactly `3DM` BF16 weights.
- The candidate combines each block partial `(g,u)` with running state `(p,q)`
  as `p'=p+g+alpha*p*g`, `q'=q+u+alpha*(p*u+q*g)`.  It has two mathematical
  `M`-wide states but may require two additional physical temporary accumulator
  fragments during a block combine; register pressure and occupancy are part
  of this fatal test, not assumed free.
- It writes only the final `M`-wide BF16 `SiLU(p)*q` tensor.  The custom control
  uses an ordinary two-accumulator mainloop and the same fused output.
- The nonzero timing coefficient is `alpha=0.05`.
- Strong ordinary controls are split G/U GEMMs with fused in-place SwiGLU and
  one packed `D -> 2M` GEMM with fused in-place SwiGLU.  The faster median in
  each cell defines the ordinary envelope.
- Full FFN timing includes the input path, activation, and ordinary down GEMM.
- Token-row cells: `1, 8, 32, 128, 512, 2048`.
- Each path is warmed and CUDA-graph captured.  Run 30 graph warmups and 100
  deterministic randomized-order CUDA-event samples per cell.  Report medians
  and 5,000 paired bootstrap ratio intervals.  All raw timing vectors are
  persisted so medians and intervals can be recomputed.

## Correctness and resource gates

All must pass:

1. A separate `rows=16,D=256,M=256,k=4` probe agrees with independent FP32
   blockwise references for ordinary and nonzero associative outputs:
   relative L2 error at most 1% and all values finite.
2. The ordinary custom output agrees with ordinary packed SwiGLU on the
   target shape at relative L2 error at most 1%.  The nonzero associative
   target output must independently agree with the blockwise FP32 reference at
   the same tolerance and all target values must be finite.
3. Every arm has exactly `3DM` resident BF16 weight scalars; the candidate has
   no persistent tensor absent from controls and writes only an `M`-wide FFN
   activation before down projection.  Exact resident weight bytes and all
   input/intermediate/output dtypes are recorded.
4. Candidate median/custom-control median is at most 1.01 and its paired upper
   95% bootstrap ratio is at most 1.02 in every cell.
5. Candidate median/fastest-ordinary median is at most 1.00 and its paired
   upper 95% bootstrap ratio is at most 1.02 in every cell.
6. Candidate peak allocated bytes during one uncaptured full-FFN call are no
   greater than both ordinary controls in every cell.

Passing authorizes a matched language screen with carrier-null baseline,
associative dual reduction, a same-arithmetic non-associative control, and an
alpha-zero ablation.  That screen must log carrier alpha, individual-factor and
all up-path gain quantiles/sign flips, activation RMS, and gradient p99.
Failure closes this executor before training.
