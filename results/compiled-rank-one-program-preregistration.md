# Compiled ordered rank-one program — preregistration

Status: **frozen before H100 execution**  
Date: 2026-07-27

## Candidate and exact claim

For a retrieved ordered program of `L` shared singleton experts,

\[
x_i=x_{i-1}+c_i v_i,\qquad
c_i=s_i\phi(u_i^\top x_{i-1}+b_i),
\]

precompute `G_ij = u_i^T v_j`.  Inference can evaluate the exact same nonlinear
program as

\[
a_i=u_i^\top x_0+b_i,\qquad
c_i=s_i\phi\!\left(a_i+\sum_{j<i}G_{ij}c_j\right),\qquad
x_L=x_0+\sum_i c_i v_i.
\]

Thus all `u_i^T x_0` projections and the final value accumulation remain wide,
parallel operations.  The sequential part is only an `L`-scalar triangular
recurrence.  This is exact for any pointwise scalar activation; it is not a
linearization or approximation.

The capability claim is narrow: an additive PEER-style singleton mixture is
invariant to expert order, whereas this program is generically order-sensitive
and nests `L` nonlinear transformations while reading the same `L` pairs of
rank-one expert vectors.

## Different currency

The candidate explicitly pays for:

1. `L(L-1)/2` cached cross scalars per stored program;
2. an `L`-step scalar dependency chain;
3. discrete ordered program IDs and their routing/training problem;
4. offline recompilation of cross terms when expert vectors change.

It does not claim that route count is free learned information.  Programs can
only reuse and compose the shared expert bank; arbitrary independent functions
still require arbitrary independent parameter bits.

## Frozen serving shape

- hidden dimension `D = 4096`;
- program length `L in {4, 8, 16, 32}`;
- BF16 expert vectors and cached cross terms;
- target bank `N = 16,384` shared singleton experts;
- target table `P = 1,000,000` ordered programs;
- H100 SXM, decode-like batches `B in {1, 8, 32, 128}`.

At `L=8`, a physical program record is 16 bytes of `uint16` IDs plus 56 bytes
of BF16 strict-lower-triangle cross terms: 72 bytes.  The shared expert bank is
268,500,992 bytes including BF16 bias and scale metadata, and one million
compiled records are 72,000,000 bytes.  This
excludes router keys and runtime alignment, which must be reported separately.

The active expert payload is 131,104 bytes/token, identical to an
additive mixture using the same eight singleton experts.  The candidate adds 28
scalar multiply-accumulates to 65,536 vector MACs, before routing and indexing.

## Frozen gates

### G0 — algebra

- float64 compiled/direct maximum relative error below `1e-12` for ReLU, tanh,
  and SiLU;
- additive forward/reverse order distance below `1e-12` on the fixed witness;
- sequential forward/reverse order distance above `1e-3`.

Failure closes the candidate immediately.

### G1 — unoptimized H100 executor

For every `(B,L)` cell, compare additive, direct sequential, and compiled
execution with identical selected `U,V,b,s` tensors.  Report eager and
`torch.compile` separately, randomized interleaving, at least 200 timed samples,
and numerical error against FP32 direct execution.

Promotion requires:

- compiled maximum relative error `<= 2e-2` in BF16 and `<= 2e-3` in FP16;
- compiled median latency lower than direct sequential for every `L >= 8` cell;
- compiled median latency no more than `1.15x` additive at `B <= 8, L <= 16`.

The 15% margin is only a prototype promotion gate.  The final service gate
remains the repository-wide 2% noninferiority contract and requires a fused
kernel plus real program-bank gathers.

### G2 — learnability/capability

Using identical data, parameterized expert bank, active expert count, optimizer
budget, and router-visible information, compare:

1. additive singleton experts;
2. direct sequential singleton experts;
3. compiled sequential singleton experts (same forward function as 2);
4. a matched small dense residual MLP control.

Run at least three seeds on a mixture of order-sensitive and order-insensitive
worlds.  The candidate must beat additive on every order-sensitive slice, stay
within one baseline standard error on order-insensitive slices, and beat the
matched dense control on median capability per resident byte.  Aggregate loss
cannot hide a failed slice.

### G3 — physical serving

Only after G2 passes: one fused kernel, actual bank/program lookup, router,
allocator ledger, HBM/L2 counters, and decode cells under the frozen resource
contract.  Final promotion requires <=2% latency/cost regression, exact byte
accounting, and a protected capability improvement outside repeatability noise.

## Collision boundary

- PEER retrieves singleton MLP experts but adds their outputs independently.
- UltraMem/UltraMemV2 retrieve value vectors or one-neuron FFN values; neither
  source inspected here specifies an ordered nonlinear program compiled through
  its cross-Gram recurrence.
- Products of linear rank-one transforms have classical compact WY/Householder
  representations.  The retained claim is the arbitrary-activation scalar
  recurrence plus sparse neural program retrieval, not invention of compact
  rank-one linear algebra.

## Pre-execution ledger/harness erratum

Independent review before a valid full-grid result found that the first ledger
omitted expert bias/scale metadata, and that the first smoke harness used a full
`L x L` coupling allocation plus batch-global error.  The numbers above include
the metadata.  G1 now uses the physically claimed packed strict triangle,
maximum per-row relative error, an explicit full-grid gate, and includes `L=4`
in the frozen `B<=8,L<=16` additive-latency check.  The earlier ten-sample FP16
smoke is diagnostic only and is not promotion evidence.

A second review found that full residual output error was diluted by the
identity path.  Numerical promotion is therefore evaluated on the maximum
per-token relative error of the nonlinear update `y-x`, with full-output and
absolute update errors reported separately.  No valid full-grid result preceded
this correction.
- Ordinary multi-hop/deep MoE reroutes and materializes a full hidden vector at
  each hop.  This candidate retrieves once and analytically avoids those vector
  passes.
