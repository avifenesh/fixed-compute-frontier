# Interference-expert H100 decision

## Decision

**Close the two-INT4-basis interference expert before language training.**

The abstract instruction-count equality does not survive Hopper's native
execution paths.  H100 has an SM90a WGMMA path for signed INT8, while the
signed INT4 candidate is generated as a legacy SM80 warp-MMA kernel.  The
candidate therefore cannot claim that two INT4 bases cost one native INT8
projection merely because `mma.sync.m16n8k64.s4` covers twice the K extent of
`mma.sync.m16n8k32.s8`.

## Frozen raw-kernel gate

The gate used CUTLASS 3.5.1 built with CUDA 12.4 for `sm90a` on an H100 SXM
80 GB.  It compared:

- one native SM90a S8 WGMMA projection, `D=4096 -> M=14336`;
- one legacy S4 projection of the same shape, the strongest ordinary routed
  two-W4-expert control before dispatch;
- one S4 projection with `N=28672`, representing two concatenated W4 bases
  before their combination.

All measurements are raw GEMM ceilings.  They exclude quantization scales,
routing, activation quantization, the candidate's second accumulator bank,
base combination, and SwiGLU.  Verification was disabled because the two
formats intentionally compute different quantized functions.

| Rows | W8 WGMMA | routed W4 | dual W4 | dual W4 / W8 |
|---:|---:|---:|---:|---:|
| 256 | 0.031087 ms | 1.171680 ms | 2.331460 ms | 74.999x |
| 512 | 0.050116 ms | 2.335330 ms | 4.093960 ms | 81.690x |
| 2,048 | 0.186412 ms | 8.190220 ms | 16.361200 ms | 87.769x |
| 4,096 | 0.376053 ms | 16.349000 ms | 32.142100 ms | 85.472x |

The selected SM90 WGMMA kernel requires a 256-row cluster-aligned problem, so
this table is a prefill/throughput gate rather than a decode benchmark.  That
limitation does not rescue the proposal: its claimed native equal-work path
is absent, and the measured legacy path fails before every candidate-only
cost is added.  A future dequantizing W4A8/W4A16 kernel would be a different
precision and execution proposal, not evidence for native dual-W4A4 parity.

## Algebraic boundary

Two four-bit bases can parameterize two correlated effective matrices at the
same raw payload-bit count as one W8 matrix.  They do not contain the W8
function class plus independent conditional capacity, require either extra
scale tables or shared scales, and are dominated by directly routing one W4
expert whenever both bases must be computed.

Do not run `experiments/interference_expert_lm_pilot.py`; it was intentionally
left unexecuted after the fatal hardware review.

## Evidence

- raw summary SHA-256:
  `639136dbb634a52b46142ea03772635dc9f34ebfdba1d1df33a6d21aec23c1cc`;
- runner SHA-256:
  `c224820b2a3fa1726bdc101225ec7e9159d9553665e13c3619ea96d27774f11e`;
- raw CSVs and summary:
  `results/cutlass-interference-gate/`.

