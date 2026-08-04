# Sparse Cayley programs — current prior-art boundary

Date checked: 2026-07-30  
Status: **components collided; only the combined rank/active-payload claim remains**

## Directly established components

- **Cayley-parametrized orthogonal neural weights:** scoRNN learns an
  orthogonal recurrent matrix through a skew-symmetric Cayley transform.
  <https://arxiv.org/abs/1707.09520>
- **Neumann and Cayley in a neural model:** NC-GRU uses a Neumann-series method
  with scaled Cayley orthogonal GRU weights.  Its series updates a dense inverse
  during training; it does not establish sparse feature-axis conditional
  experts.
  <https://arxiv.org/abs/2208.06496>
- **Learned sparse global switching:** Neural Shuffle-Exchange, learned
  butterfly factorizations, ButterflyFlow, and Dimension Mixer already cover
  nonlinear or learned `O(D log D)` switch/butterfly networks.  A broad
  "conditional switch network" novelty claim is closed.
  <https://arxiv.org/abs/1907.07897>
  <https://proceedings.mlr.press/v97/dao19a.html>
  <https://proceedings.mlr.press/v162/meng22a.html>
  <https://proceedings.mlr.press/v280/sapkota25a.html>
- **Conditional decision trees:** The Tree Ensemble Layer (Hazimeh et al.,
  ICML 2020, arXiv:2002.07772) already supplies differentiable sparse routing
  and true conditional execution for learned tree layers, including custom
  forward/backward algorithms.  Neither hard tree paths nor conditional tree
  execution is novel here.
  <https://arxiv.org/abs/2002.07772>
- **Hierarchical expert routing:** Hi-MoE (arXiv:2605.08292, May 2026) reports
  two-level grouped routing and a 5.6% perplexity reduction over its OLMoE-7B
  baseline at 58B training tokens.  Hierarchical balancing/specialization is
  therefore a strong current control, not a novelty claim.
  <https://arxiv.org/abs/2605.08292>
- **Alternating structured bases and diagonals:** deep diagonal-circulant
  networks and later structured-matrix work establish that products of cheap
  transforms and learned diagonal/block factors can be expressive.
  <https://arxiv.org/abs/1901.10255>
- **Strong structured dense-layer controls:** Monarch and BTT already show
  meaningful compute-quality gains, including language-model experiments.
  Any positive program-tree result must compare against them rather than only
  against dense SwiGLU.
  <https://arxiv.org/abs/2204.00595>
  <https://arxiv.org/abs/2406.06248>
- **Massive sparse vector memory:** PEER and Memory Layers at Scale establish
  cheap conditional capacity from large key/value or tiny-expert banks.  The
  latter reports beating dense models with more than twice the compute at
  contemporary scale.
  <https://arxiv.org/abs/2407.04153>
  <https://arxiv.org/abs/2412.09764>
- **Recursive expert reuse and combinatorial paths:** Mixture of Universal
  Experts explicitly develops virtual width from recursively reused experts,
  structured rotational connectivity, trajectory-aware routing, and
  depth-aware load balance.  It reports up to 1.3% matched-budget gains.  Broad
  claims that ordered expert reuse or exponential route count are new are
  closed.
  <https://arxiv.org/abs/2603.04971>
- **Hierarchical routing:** Union of Experts and the much older hierarchical
  MoE/decision-tree families cover the general idea of routing through a
  hierarchy.  A binary tree is not novelty by itself.
  <https://arxiv.org/abs/2503.02495>
- **Implicit sparse global propagation:** implicit GNNs cover fixed-point
  global responses built from graph propagation, though for graph-structured
  examples rather than feature-axis conditional FFN experts.
  <https://arxiv.org/abs/2009.06211>

## Surviving exact claim

No inspected source establishes this complete package:

1. a degree-bounded sparse skew generator whose Cayley response supplies a
   shared cheap orthogonal feature basis;
2. an active expert consisting of only gate/up/down diagonal `D`-vectors in
   that basis;
3. a generically rank-`D` local update Jacobian from those `3D` active payload
   scalars, versus rank at most `k` for `k` active additive vector experts;
4. several independently oriented bases composed along a hard binary path;
5. a full binary program tree whose total learned scalars are no greater than
   the replaced dense SwiGLU, while a selected path has `O(D log D)` ideal
   work and payload traffic.

The surviving novelty sliver is therefore the **local-rank per active-payload
separation and its same-byte program-tree realization**, not Cayley transforms,
Neumann series, structured matrices, diagonal factors, hierarchical routing,
conditional memory, or recursive expert reuse individually.

## Claims still forbidden

- Route count is not independent information.
- Full rank is not arbitrary-matrix expressivity.
- A logical operation count is not GPU speed.
- A one-seed 10M-token result is not a scaling law.
- A quality tie to dense is not enough until BTT/Monarch, PEER/memory, and
  hierarchical/additive controls are run.
- Novelty remains provisional until a broader scholarly and patent search is
  performed after a positive capability result.
