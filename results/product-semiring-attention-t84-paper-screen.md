# T84 shared-score heterogeneous attention — revised paper screen

Date: 2026-08-02  
Status: **INDEPENDENTLY AUDITED; REVISION APPLIED; RE-AUDIT REQUIRED; NO RUN**

## 0. Decision object

Modern attention applies one propagation law to nearly every hidden feature:

\[
o_{ic}=\sum_j \operatorname{softmax}_j(s_{ij})v_{jc}.
\]

The same probability distribution over source tokens is used for every value
channel in a head. That is suitable when a feature means uncertain evidence to
be averaged. It is algebraically mismatched when a feature means a best score,
cost-to-go, reachable state, hard constraint, or Viterbi path value, whose
natural update is `max`, `min`, or another idempotent reduction.

T84 asks whether one learned relation graph can carry two kinds of state under
two propagation laws at the same layer:

- ordinary softmax feature groups for signed, fuzzy, associative language
  features; and
- unnormalized max-plus feature groups for discrete optimization and dynamic
  programming features.

The deployed proposal is a heterogeneous/direct-sum attention block. It is
**not** literally a Cartesian product semiring because its soft branch keeps
ordinary signed softmax values. A true log-semiring times max-plus product is
retained below as the mathematical motivation and a reference variant, not as
the architecture claim.

The deployment layout is static. It does not add experts, tokens, recurrent
steps, retrieval, or a runtime router. The number of heads, hidden width, QKV/O
projection dimensions, parameters, KV width, training tokens, and served depth
remain matched to an all-softmax control. Kernel time is not assumed equal and
must be measured.

This is a model-level reasoning hypothesis, not a speed hypothesis. A run is
admitted only if the mechanism can plausibly reduce a complete held-out
reasoning error by at least 20% while preserving the protected language and
retrieval suite.

## 1. Clean algebra

For one query and one feature channel, let `s_j` be a learned query-key score
and let `u_j` be any finite real value potential. For the algebraic bridge,
`exp(u_j)` may be read as a positive value. Let `beta>0`, let `n` be the number
of finite legal sources after masking, require `n>=1`, and define the
unnormalized log-semiring product

\[
B_\beta(s,u)=
\frac{1}{\beta}\log\sum_j e^{\beta(s_j+u_j)},
\qquad
Z_\beta(s)=\frac{1}{\beta}\log\sum_j e^{\beta s_j}.
\tag{1}
\]

Its normalized form is

\[
A_\beta(s,u)=
B_\beta(s,u)-Z_\beta(s).
\tag{2}
\]

For general `beta`,

\[
e^{\beta A_\beta(s,u)}
=\sum_j \operatorname{softmax}_j(\beta s)e^{\beta u_j}.
\tag{3}
\]

At `beta=1`, Equation 3 is exactly positive-valued softmax attention represented
in the log semiring. It is not ordinary signed attention. The zero-temperature
limits are

\[
B_\infty(s,u)=\max_j(s_j+u_j),
\qquad
A_\infty(s,u)=B_\infty(s,u)-\max_j s_j.
\tag{4}
\]

The subtraction is a normalization: it makes the result invariant to adding a
constant to all query-key scores. It does not change the winning predecessor.

The log semiring and max-plus semiring share ordinary addition as their
multiplication and differ only in addition:

\[
a\oplus_{\log,\beta}b=\beta^{-1}\log(e^{\beta a}+e^{\beta b}),
\qquad
a\oplus_{\max}b=\max(a,b).
\]

Their Cartesian product, with componentwise operations, is itself a semiring.
Therefore an exactly log-domain feature vector paired with max-plus coordinates
can propagate over one learned score graph. The proposed production block keeps
ordinary signed softmax in the first coordinate instead, so this product result
motivates heterogeneous aggregation but does not prove a property of the whole
deployed block. The identity is not an empirical advantage.

### Proposition T84.1 — finite-temperature error

For `n` finite inputs,

\[
0\le B_\beta(s,u)-B_\infty(s,u)\le\frac{\log n}{\beta},
\qquad
\left|A_\beta(s,u)-A_\infty(s,u)\right|
\le \frac{\log n}{\beta}.
\]

**Proof.** For any vector `x`, write

\[
\beta^{-1}\log\sum_j e^{\beta x_j}=\max_jx_j+\delta_x,
\qquad 0\le\delta_x\le\frac{\log n}{\beta}.
\]

The first bound is immediate. Equation 2 minus Equation 4 is
`delta_(s+u)-delta_s`; both terms lie in the
same interval, so their difference has absolute value at most
`log(n)/beta`. QED.

This bound predicts a concrete OOD failure mode: a finite-temperature surrogate
needs inverse temperature to grow at least logarithmically with sequence length
to maintain fixed worst-case max-plus error. The exact max-plus branch does not.
The claim is conditional on finite values; ties preserve the value but make
provenance non-unique.

### Proposition T84.2 — exact Bellman feature update

Let `S=(s_ij)` be a weighted relation matrix and `U=(u_jc)` contain `C`
independent state potentials. Then

\[
Y_{ic}=\max_j\{s_{ij}+u_{jc}\}
\]

is exactly the unnormalized limit `B_infinity` and one max-plus matrix product.
Each feature channel may choose a
different predecessor `j`. With fixed `S`, stacking `L` such updates computes
the maximum score over length-`L` walks for every channel by induction on `L`.

The normalized limit `A_infinity` instead performs the same operation on the
row-normalized relation `s_ij-max_k(s_ik)`. It preserves each local winner but
changes multi-step path values. T84 therefore uses `B_infinity` inside the
tropical feature group and normalizes only when fusing the completed branch
back into the residual stream. The exact Bellman statement applies before that
ordinary neural normalization and output projection.

This is the useful primitive: one learned relation graph updates many
cost-to-go or reachability features in parallel. It does not solve grounding,
learn the correct graph automatically, recover a proof trace, or establish a
language-model gain by itself.

## 2. Why this is not merely hard softmax

A standard softmax head uses one source distribution `a_ij` for all `d_h`
value channels. In its zero-temperature limit, all channels in that head copy
from the same winning token. A max-plus head computes

\[
y_{ic}=\max_j(s_{ij}+u_{jc}),
\]

so its `d_h` channels can have `d_h` different winning predecessors while
sharing the expensive vector-valued relation score `s_ij`.

Ordinary multi-head attention can recover channel-specific winners by using
more, narrower heads. That is a mandatory control, not a refutation. To emulate
`d_h` independent winners while retaining the same rank-`d_h` relation score,
however, it must either duplicate that score across heads, reduce each score to
a lower-rank matcher, or reconstruct the max in later layers. T84 tests whether
the max-plus aggregation is a more useful allocation of the same width and
depth; it does not claim a distribution-free function-class separation from an
arbitrary Transformer plus MLP.

The direct construction makes the possible compression explicit. Suppose the
desired channel-specific logits are

\[
L_{ijc}=q_i^Tk_j+u_{jc},\qquad q_i,k_j\in\mathbb{R}^r,\quad c\in[C].
\]

A conventional hard-softmax construction needs `C` score maps to obtain `C`
independent winners. If each map directly retains the same rank-`r` relation
term, it uses `C(r+1)` query/key score coordinates after adding a constant-query
coordinate for `u_jc`. The max-plus construction evaluates the rank-`r`
relation once and combines it with `C` value coordinates, using `r+C` logical
coordinates. Sharing the relation score and then adding a channel term inside
the reduction is precisely the candidate operation; it is unavailable to a
standard head whose attention weights are shared across value channels.

This is a conditional direct-construction saving, not a lower bound over all
Transformers. A deep or differently encoded softmax model may compute the same
function, which is why the many-narrow-head and matched-depth controls remain
fatal controls rather than straw baselines.

## 3. Important false equivalence

The statement

\[
\lim_{\beta\to\infty}\operatorname{softmax}(\beta S)V
= S\otimes_{\max+}V
\]

is false. The left side selects the single value vector belonging to a maximum
entry of `S`; the right side computes `max_j(S_ij+V_jc)` separately for every
value channel. For example, with `S=[10,0]` and `V=[0,100]^T`, the left side
tends to `0` while the max-plus product is `100`.

This invalidates the central equality claimed by *The Geometry of Thought:
Disclosing the Transformer as a Tropical Polynomial Circuit* (arXiv:2601.09775,
Theorem 1) as written. T84 does not use that result. Equations 1--4 give the
correct log-semiring-to-tropical limit.

## 4. Direct prior-art boundary

- *Tropical Attention* replaces every attention block with tropical
  projections, a tropical Hilbert metric, and max-plus aggregation. Its paper
  reports large combinatorial OOD gains, but explicitly says generative
  language scaling and tropical runtime overhead remain untested, and names
  hybrid semiring architectures as future work.
- *HydraHead* establishes that different mechanisms can coexist at head
  granularity and that their output scales need explicit normalization, but it
  mixes full and linear attention for long-context efficiency rather than log
  and max-plus algebras for reasoning.
- *Semiring Activation in Neural Networks* makes trainable semiring operators
  legitimate neural components, but its larger ConvNeXt experiment
  underperforms the ordinary MLP and exposes initialization and numerical
  sensitivity.
- maxout, hard attention, sparsemax/entmax, low-temperature softmax, neural
  algorithmic reasoning, Viterbi/inside algorithms, and explicit graph solvers
  are direct controls.

The broad ingredients are occupied. The screened conjunction is narrower:
static heterogeneous feature reducers inside one language-capable attention
block, sharing a learned relation graph, with an all-softmax special case and a
full production-cost ledger. No novelty claim is made until a broader search
and independent audit finish.

Primary sources:

- https://arxiv.org/abs/2505.17190
- https://github.com/Baran-phys/Tropical-Attention
- https://arxiv.org/abs/2606.20097
- https://arxiv.org/abs/2405.18805
- https://arxiv.org/abs/2601.09775

### Why the expected magnitude is not automatically too small

Using the nine classification rows in Tropical Attention's published length-OOD
table and choosing the strongest listed softmax or recurrent baseline separately
for each row, the sum of `100 - micro-F1` errors is `372.47` for the controls and
`201.18` for the tropical models: a `45.99%` relative aggregate-error reduction.
That clears the magnitude prior for an audit, but it is **not** a T84 success:
the models are task-specific, every block is tropical, several individual gains
are small, some value/noise rows regress, no language suite is protected, no
simultaneous 20% confidence bound is supplied, and the reported tiny GPU timings
do not substitute for synchronized production-shaped measurement. T84 must
reproduce the causal gain against stronger controls inside one mixed learner.

## 5. Architecture considered

For hidden width `D`, retain the baseline QKV and output projection shapes. The
primary construction computes each head's score matrix `S=QK^T/sqrt(d_h)` once,
then a static mask partitions that head's contiguous value channels into two
branches:

\[
P_{ij}=\operatorname{softmax}_{j\in J_i}(S_{ij}),
\qquad
O^{\rm soft}_{ic}=\sum_{j\in J_i}P_{ij}V^{\rm soft}_{jc},
\]

\[
O^{\rm max}_{ic}=\max_{j\in J_i}\{S_{ij}+V^{\rm max}_{jc}\},
\]

where `J_i` is the nonempty set of legal causal sources. Masked scores are set
to negative infinity before both reductions. `V_soft` and `V_max` are disjoint
columns of the same ordinary real-valued linear value projection. The max branch
uses those finite real coordinates directly as max-plus potentials: there is no
log, exp, sign split, epsilon, clipping, inverse map, or hidden auxiliary state.
The two outputs are group-normalized, concatenated, passed through the matched
output projection, and returned to the ordinary residual stream.

The two channel groups therefore share one learned relation graph but apply
different reductions to it. Whole-head softmax/tropical allocation is a
mandatory ablation and a possible kernel-friendly fallback, but it gives up the
relation-sharing claim. The frozen first comparison applies identical
non-affine per-group RMS normalization to every candidate and all-softmax
control; a conventional all-softmax block without that normalization is also a
control. Naive unnormalized concatenation is not admitted because branch scale
alone can create bypass. An initial run uses a fixed branch ratio and exact max
subgradients from step one. Finite-`beta`, annealed, and straight-through paths
are controls, not rescue tuning. Deployment never evaluates two reducers for
the same static value coordinate.

The all-softmax assignment exactly contains only the correspondingly
group-normalized all-softmax baseline. It does not establish containment of a
conventional pretrained Transformer block, and no claim of optimization or
generalization non-regression follows from architectural containment.

## 6. Complete cost equation

At the same width and head count, parameters, resident bytes, KV width, and
nominal pairwise aggregation count remain first-order matched. Replacing an
ordinary `P@V` multiply-add reduction with `max(S+U)` substitutes add/max
operations for multiply/add operations; it does **not** make them equally cheap
on a GPU. Tensor cores favor GEMM, while a tropical kernel may be limited by
reductions, broadcasts, register pressure, or memory traffic.

For one served workload, charge

\[
C = C_{QKV/O}+C_{\rm soft}+C_{\rm trop}+C_{\rm norm}
+C_{\rm FFN}+C_{\rm KV}+C_{\rm workspace}+C_{\rm generated\ tokens}.
\]

Training additionally charges per-query/per-channel winner provenance or
recomputation, backward workspace, all search/calibration examples, compiler
and autotuning time, and failed configurations. If a later assignment supernet
is introduced, all duplicate-branch training work and optimizer state are added
then. No claim may use parameter equality as a proxy for complete cost. Once
measured, any tropical runtime premium is given to an all-softmax control as
the best ordinary extra width, depth, heads, or recurrent steps it can buy.

## 7. Cheapest decisive falsifier

No GPU run is authorized by this screen. After a separate preregistration, the
first possible run is a reject-only CPU ladder. It must not place soft and
max-plus examples in separable task families.

1. **Exact row unit.** One causal query sees earlier records. The same record
   set defines both a signed softmax-weighted evidence target and a
   channel-wise `max(score + potential)` target. The final answer is a balanced
   XOR of independently balanced thresholded subanswers, so either subanswer
   alone is at chance. This verifies equations and masks only.
2. **Learned-relation row.** Raw record attributes replace supplied scores;
   the model must learn one relation map used by both targets. Attribute names,
   record order, lengths, values, and final composition are held out. A family
   or length probe must be unable to identify which reducer matters because
   every instance needs both.
3. **Decoder-causal weighted automaton.** An unordered, permuted token list
   describes one weighted transition system before the query. Marginal/fuzzy
   evidence and a Viterbi/best-path quantity use the same transition relation,
   and the final decision requires both. Train on short unrollings and test
   longer unrollings, value shifts, renamed states, changed graph shapes, and
   unseen compositions. Ordinary causal masking and the intended positional
   scheme are mandatory; supplied DP tables, operator labels, privileged graph
   masks, and fixed topological serialization are forbidden.

Serialization permutation, task-label, surface-statistic, direct-target, and
memorization probes are fatal integrity controls. Branch lesions must show that
removing the soft group damages the fuzzy subanswer and removing the max group
damages the path subanswer; a score-unsharing lesion tests the claimed shared
relation. Exact operator solvers are ceilings, not learned baselines.

Required matched controls are:

- all-softmax at the baseline head count;
- all-softmax with identical feature partition, group normalization, and output
  fusion;
- all-softmax with many narrow heads, including head dimension one where
  feasible;
- faithful published Tropical Attention, not merely T84 with every channel set
  to max;
- T84 with every channel set to max;
- shared-score and branch-specific-score soft/max hybrids under both matched
  parameters and matched measured CPU time;
- within-head channel split and whole-head split;
- finite-`beta` log-semiring aggregation from Equation 1, including a learned
  but deployment-frozen `beta`;
- annealed-`beta` and straight-through max training as declared controls;
- low-temperature and learned-temperature softmax;
- sparsemax/entmax or hard-attention heads;
- fixed random softmax/tropical allocation;
- a score-conditioned cross-token max/maxout control;
- depth-recurrent Transformer or recurrent graph processor; and
- exact algorithmic solvers as ceilings with their full invocation cost.

Every learned control receives identical examples, tokens, parameters, active
width, depth, optimizer budget, tuning budget, and seeds. The model may not be
told which algebra a sample requires. Memorization, length-only, and
task-separable controls are mandatory.

Before looking at accuracy, the preregistration must freeze branch-use and
gradient-health kills: dead/zero-variance tropical channels, winner entropy,
the fraction of oracle-required predecessor support ever receiving a gradient,
soft/max output norms, residual contribution, per-layer winner concentration,
and inference-time branch lesions. A headline gain with a bypassed branch or a
serialization shortcut is an automatic failure.

## 8. Twenty-percent admission and kill gate

The primary capability metric is macro-averaged complete-task error on the
frozen compositional-OOD suite. Error, not accuracy, is the denominator because
it measures remaining failures; absolute accuracy change and consumed headroom
are also reported. Protected fuzzy retrieval and in-distribution language-like
accuracy have a one-percentage-point non-inferiority margin.

For every candidate-versus-control comparison, compute paired per-instance loss
differences, aggregate across preregistered seeds and domains, and form a
simultaneous one-sided 95% confidence bound over all primary domains and the
macro average. The run passes only if:

1. the **lower** simultaneous bound establishes at least 20% relative OOD error
   reduction against the strongest same-cost learned control;
2. every primary reasoning domain individually has a nonnegative lower bound;
3. protected retrieval/language does not cross its regression margin; and
4. either served p50/p95 latency and memory are non-inferior, or a separately
   frozen capability-per-complete-cost score improves by at least 20% against
   the best ordinary use of the added cost.

A 20% point estimate with a lower bound below 20% does not pass. Ten to less
than 20% is an arguable finding eligible for at most one bounded replication.
A single-digit gain, a win only over weak-head softmax, a win only on one
algebra family, or a result erased by many-narrow-head or recurrent controls
closes T84 as a main lane.

The CPU ladder is reject-only even if its lower bound clears 20%. Passing all
three CPU stages merely earns a small causal language-model integration. A
component success requires the same gate on a frozen, multi-family learned
reasoning phase with protected language/retrieval. The phrase **smarter served
model** is reserved for an open autoregressive model that preserves the gain,
language quality, and full p50/p95 cost comparison after the target-shaped GPU
kernel is measured. Before any GPU use, check whether the local GPU is free;
renting is allowed only after the CPU evidence and preregistration earn it.

## 9. Current decision

T84 survives an initial algebra screen because it identifies a precise missing
primitive: current heads force all features in a head to share probabilistic
source weights, whereas a shared-score heterogeneous block can propagate fuzzy
evidence and local best-path state over one learned relation graph. It also supplies a
correct finite-temperature bound and avoids a published false equivalence.

Both independent audits return **REVISE**. It is not yet an admitted experiment.
The likely failure modes are severe:
ordinary narrow heads or MLPs may absorb the gain; sparse max gradients may make
the branch untrainable; the synthetic suite may reveal only a supplied algebra;
and GPU add/max reductions may be substantially slower than FlashAttention
GEMMs. The revised causal task, exact value contract, branch diagnostics, and
fatal controls require a short re-audit before any code or run.

Audits:

- [independent audit A](product-semiring-attention-t84-independent-audit-a.md)
- [independent audit B](product-semiring-attention-t84-independent-audit-b.md)
