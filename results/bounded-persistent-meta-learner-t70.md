# T70 bounded persistent meta-learner — train the deployed model as a learner

Date: 2026-08-02  
Status: **OPERATOR CHANNEL VALIDATED; LOOKUP WRITER-CREDIT CLAIM FAILED; CROSS-FAMILY MANIFEST NOT YET FROZEN; ALL SCALED/RENTED RUNS HELD**

## 0. Corrected target

The research target is not a novel layer name. It is a deployed model that
becomes substantially more capable from new experience while retaining prior
competence. Containment in a generic recurrent network is therefore not fatal:
the strongest recurrent learner is a valid candidate if it shifts the
capability/cost frontier.

The candidate is a language-capable model trained over complete **lifetimes**,
with:

1. a fixed-size persistent latent state that survives episode and request
   boundaries;
2. an ephemeral reasoning workspace that is deliberately erased;
3. frozen deployment weights—online learning occurs through the state update,
   not backpropagation; and
4. loss and reward assigned to future competence after the erasure, including
   retention and change-point probes.

This is not “add memory.” The causal proposal is to make one learned update
operator responsible for converting experience into the only state from which
future performance can improve.

## 1. One-sentence edge

At fixed backbone parameters, bounded per-request context, and no deployment
gradient updates, spend a declared fixed latent-state budget per learner to
obtain at least `30%` lower cumulative regret, `2x` faster acquisition, and no
material forgetting across unseen interactive worlds; later, a smaller adapted
learner should match a substantially larger frozen model.

The persistent state and write computation are added resources. They are not
called free. The claimed edge is capability per static parameter and per
interaction under a bounded serving state, not strict dominance over a
stateless model with zero memory.

The improvement belongs to the complete frozen-backbone-plus-persistent-state
system and is normally specific to one learner/environment. Resetting that
state should remove the acquired gain. A later smaller-versus-larger comparison
must give both systems identical observations and each system its best
resource-matched context, retrieval, or bounded-state evidence path.

## 2. Typed deployed operator

Let slow model parameters `theta` be frozen at deployment. Let `m_t` be `s`
persistent latent slots of width `d_m`, stored at declared precision, and let
`c_t` be a bounded current segment containing the present observation, prior
actions and feedback, and ephemeral scratch tokens.

The processor and writer are

\[
(\pi_t,w_t)=P_\theta(c_t,m_t),
\]

The noncircular step order is: observe `o_t` and prior feedback; form bounded
scratch; compute `pi_t,w_t`; execute `a_t`; receive `o_{t+1},f_t`; then commit

\[
m_{t+1}=U_\theta(m_t,w_t,o_t,a_t,o_{t+1},f_t).
\]

Writer randomness is independently resampled unless it is explicitly included
in persistent state. At a randomized boundary, apply

\[
c_{t+1}\leftarrow \varnothing,
\qquad m_{t+1}\text{ is retained}.
\]

No hidden transcript, summary, KV cache, optimizer state, retrieved episode, or
environment identifier crosses that boundary unless it is explicitly included
in the persistent-state ledger. The interface can be implemented by recurrent
state, memory tokens, cross-attention slots, an SSM state, or another bounded
map. Those are implementation choices, not separate hypotheses.

### Concrete local implementation and causal delta

The first tested implementation is a small causal Transformer with `s`
persistent state tokens. A segment is ordered as

```text
[memory-read tokens] [typed event and scratch tokens] [memory-write tokens]
```

under the causal mask. Event tokens can attend the prior memory; suffix write
tokens can attend the prior memory and the complete current event. Only the
`s` final write-token outputs, after a declared projection/quantization, become
the next memory-read tokens. All other hidden activations and KV entries are
discarded at the boundary. A prepended token alone cannot be a writer under a
causal mask because it cannot see later event tokens.

The tested intervention is **hard-erasure lifetime credit**: losses after a
boundary train writes made before that boundary. The closest ablation has the
same architecture, parameters, state bytes, optimizer opportunity, segments,
and targets, preserves forward state, but detaches that state at the boundary
so future losses do not train earlier writes. Reset state is a separate
information-channel ablation.
A lifetime-trained GRU or SSM is an alternative implementation of T70, not a
control whose success would refute the hypothesis. Architecture variants are
compared under one tuning budget to select the best bounded realization.

The first lookup smoke invalidated the stronger claim that direct
cross-boundary gradient credit into the observation-time write is necessary: a
detached-boundary model reached 92.81% while the full-credit model reached
100%, and reset/within-batch-permuted state remained near its chance or
fixed-point expectation. The detached model still updated Transformer weights
shared by its observation and query passes; this was not a globally frozen
writer. See the
[v1 decision](bounded-persistent-meta-learner-t70-operator-smoke-v1-decision.md)
and [independent audit](bounded-persistent-meta-learner-t70-operator-smoke-v1-independent-audit.md).
The direct-credit delta is closed on lookup. It remains a mandatory control in
the cross-family screen, where only compression, transfer, active acquisition,
or revision could justify a benefit.

Training samples lifetimes rather than independent examples:

\[
J(\theta)=\mathbb E_z\left[
\sum_{t=1}^{T}\ell_t(\pi_t;z)
+\lambda R_{retain}(m_t)
\right],
\]

where later losses have a gradient or policy-credit path through earlier state
writes. Environment rules, surface names, episode order, boundary positions,
and change points are randomized. Evaluation worlds and lifetimes are held out
and may be longer than training lifetimes.

## 3. Baseline failure witness: post-training random knowledge

For integers `n>=1`, `K>=2`, and `R>=1`, let a deployment environment draw

\[
\Theta=(\Theta_1,\ldots,\Theta_n)
\sim \operatorname{Uniform}([K]^n)
\]

after model training. A query supplies index `i`; after the first answer for
that index, feedback reveals `Theta_i` before its next occurrence. Each index
is queried `R` times in a randomly interleaved stream whose indices, timing,
order, and all other current inputs are independent of `Theta`.

### T70.1 — reset versus persistent learning

A predictor whose state resets before each query and receives only `i` has
success probability at most `1/K` on every query. Its expected errors are

\[
nR(1-1/K).
\]

A persistent learner with enough state to store all revealed symbols can use
one fixed default guess, overwrite the addressed cell with revealed feedback,
and answer the remaining `R-1` occurrences exactly, for expected errors

\[
n(1-1/K).
\]

The ratio of these expected error counts is `1/R`; at `R=10`, persistence
removes `90%` of the expected errors. This is a capability separation, not a
small metric polish.

### Proof

The post-training symbol is uniform and independent of reset-model weights and
current input, so Bayes success is `1/K`. The persistent construction writes
the revealed symbol into the slot addressed by `i` and retrieves it on later
queries. QED.

The witness proves that the evaluation unit matters: no amount of static
pretraining can know independent post-training information. It does not prove
that latent memory is better than text memory, retrieval, a recurrent control,
or weight updates.

T70.1 is only a causal-channel and capacity unit test. It cannot admit the
candidate or support a cross-family intelligence claim.

## 4. A minimal abstraction witness

For a prime `p`, an environment draws an affine rule

\[
y=ax+b\pmod p,
\qquad (a,b)\in\mathbb F_p^2,
\]

after training. Two examples at distinct `x` values identify `(a,b)` exactly:

\[
a=(y_2-y_1)(x_2-x_1)^{-1},
\qquad b=y_1-ax_1.
\]

Storing the two coefficients costs at most `2 ceil(log2 p)` bits and answers
every future query, including inputs and lifetime lengths absent from training.
A reset predictor has Bayes accuracy `1/p` on a fresh rule. An optimal control
can instead retain the two defining examples and recompute the coefficients;
that also has constant memory, though a literal `(x_1,y_1,x_2,y_2)` record uses
up to twice the coefficient bits. If the two `x` values are fixed or
recoverable, storing only their two `y` values uses essentially the same
information as `(a,b)`. The test is whether the learned bounded state discovers
a sufficient code and systematic computation, not whether latent memory beats
an informed algebraic solver.

This is only an existence construction. A model must learn the update and
finite-field computation from raw interfaces; a hard-coded solver is an oracle
control. The natural-language research question is whether lifetime training
induces similarly reusable state rather than a brittle task-specific code.
The local screen holds out primes, rule surfaces, symbols, and horizons, and
includes byte/work-matched two-example memory, a symbolic affine ceiling, and
learned recurrent controls. This remains a mechanism unit test, not the
production gate.

## 5. State capacity is a real bill

### T70.2 — exact and approximate memory lower bounds

After all symbols have been revealed, let `M` be the sole cross-boundary state
and let the decoder receive only `(M,i)` plus randomness independent of
`Theta`. If `M` has at most `2^B` distinguishable states, exact recall of every
`Theta in [K]^n` requires

\[
B\ge \left\lceil n\log_2 K\right\rceil.
\]

For uniform independent symbols, let `p_i` be the minimum/MAP error for
decoding `Theta_i` from `M`, let \(p_e=n^{-1}\sum_i p_i\), and assume
`0<=p_e<=1-1/K`. With base-two logs and entropy, Fano's inequality gives

\[
B\ge n\left[
\log_2 K-h_2(p_e)-p_e\log_2(K-1)
\right].
\]

### Proof sketch

Exact recall needs an injective code for `K^n` environments. For approximate
recall, conditional subadditivity, coordinate-wise `K`-ary Fano, and concavity
give

\[
I(\Theta;M)\ge n[\log_2K-h_2(p_e)-p_e\log_2(K-1)].
\]

Since `I(Theta;M)<=H(M)<=B`, the bound follows. QED.

Real-valued state does not evade the bill. Count served precision, noise
margin, error-correcting overhead, and every state copy. Unbounded independent
facts require unbounded bits or forgetting. The candidate seeks reusable
abstractions and bounded-environment belief, not magical infinite memory.

## 6. What hard erasure establishes

### T70.3 — causal state-use test

In the T70.1 world, evaluate identical fixed future queries under factual and
ablated memory. Suppose model weights, timing, boundary position, sampler
state, caches, retrieval, environment metadata, runtime randomness, and every
other surviving variable are conditionally independent of `Theta_i` given
query `i`; the current observation/feedback must not reveal the answer again.
Replacing `m_t` by a constant independent of `Theta`, or by state from an
independently sampled environment, restores the `1/K` Bayes ceiling.

Therefore an above-chance post-erasure result plus a constant or
independent-environment state ablation establishes
`I(Theta_i;m_t|i)>0` and that the declared state channel is necessary for the
observed gain. A within-lifetime shuffle is not sufficient when states are
correlated. This does not prove the state is compositional, calibrated, or
safely editable; the affine and multi-world tests assess those separately.

Hard erasure is useful because ordinary long context can hide the entire
solution trace. It turns persistent learning from an interpretation of
attention into a measured causal path.

## 7. Why this could matter for a production model

The substantial hypothesis is not lookup recall. A lifetime-trained update
operator may amortize a **learning algorithm** into the model:

- infer a new local rule from a few outcomes;
- retain it after its source transcript disappears;
- use it under renamed surfaces and longer compositions;
- explore when the current state is uncertain;
- revise a changed rule while protecting unrelated state; and
- reuse the acquired abstraction across later tasks.

This spends offline multi-lifetime training so a smaller deployed model can
learn through cheap forward state updates. It directly targets the gap exposed
by agentic automata learning: current agents fail at query planning, evidence
integration, and hypothesis construction even when their language prior is
strong.

The result would be meaningful if one 4B--14B-class learner, without
family-specific tools, improved across persistent facts, latent rules, active
automata, causal interventions, and nonstationary worlds, then retained the
gain outside its training horizon. A win on one lookup task is not admission.

## 8. Current collisions and strongest controls

- [ORBIT](https://arxiv.org/abs/2602.04089) is strong positive evidence for the
  objective: cross-episode meta-RL makes Qwen3-14B learn online in unseen
  environments and match GPT-5.2 on the reported interactive tasks.
- [LAMER](https://arxiv.org/abs/2512.16848)
  uses cross-episode meta-RL and in-context reflection, reporting 11--19 point
  gains in Sokoban, MineSweeper, and WebShop.
- [State commitment learning](https://arxiv.org/abs/2606.05201) uses
  counterfactual erasure so downstream reasoning remains correct without its
  hidden scratch path.
- [Persistent latent memory for decoder-only LLMs](https://arxiv.org/abs/2603.22329)
  already tests several gradient-free latent write/read mechanisms in a frozen
  GPT-2 and shows a strong capacity/inductive-bias dependence.
- [In-Place TTT](https://arxiv.org/abs/2604.06169) updates fast weights at
  inference and is the direct gradient-based online-adaptation control.
- [NextLat](https://arxiv.org/abs/2511.05963) is the predictive-state auxiliary
  objective control.
- [Continual experiential latent memories](https://arxiv.org/abs/2606.17803)
  distill reasoning experience into lightweight modular latent memories and
  report cross-dataset transfer.
- [Unsupervised meta-RL](https://arxiv.org/abs/1806.04640) states the central
  task-distribution control directly: meta-RL can move algorithm design into
  meta-training task design. T70 therefore cannot claim a general learner from
  in-distribution task adaptation.
- [Replay-MAML PTW](https://proceedings.mlr.press/v330/koop26a.html) combines an
  online partition-tree mixture with meta-learning in task-agnostic,
  nonstationary streams and reports matching or beating an advantaged offline
  control with `O(log T)` memory and computation overhead. It is a direct
  continual/change-point control, though it uses gradient-based base learners.
- Generic GRU/SSM/recurrent meta-RL, external text summaries, retrieval,
  modular memory, and a growing context with the same complete byte/compute
  budget are mandatory controls.

The component labels are occupied. T70's open empirical claim is the full
bounded-state lifetime behavior across a declared shared meta-distribution and
beyond context, not the invention of persistent memory or meta-RL. All task
families must use one typed serialized interaction interface and share the
predeclared problem of inferring, retaining, and revising a compact rule or
belief. Arbitrary transfer to an unrelated action space is not claimed.

## 9. Full resource equation

Let current-segment length be `C`, persistent slot count `s`, latent width
`d_m`, state precision `q` bits, and lifetime length `T`. Per learner,

\[
B_{persistent}=s d_m q+B_{metadata}+B_{redundancy}.
\]

If one token set joins ordinary self-attention, affected layer work scales
roughly with `(C+s)^2` versus baseline `C^2`, so the score increment is
`2Cs+s^2`. The concrete read/event/write layout has sequence length `C+2s` and
must charge that complete execution even though only `s` tokens persist. A
separate cross-attention implementation would have roughly `Cs` score work.
Both additionally pay projections, MLPs, KV/state traffic, reads, writes, and
synchronization. The exact kernel, active layers, KV representation,
host/device transfers, quantization, write frequency, latency, bandwidth, peak
memory, and energy where available must be measured.

The state ledger includes not only the slot tensor but per-layer persistent
KV/state, atomic-write double buffers, routing/age/confidence metadata, error
correction, host and device copies, serialization, and replica-shared state.
Count effective served precision, not training dtype alone.

Training charges all `T` environment transitions, generated tokens, verifier
or simulator calls, RL sampling, unrolled or recomputed activations, and
gradient traffic. Truncated credit assignment is cheaper but may fail to train
long-lived writes; full backpropagation through a lifetime may be prohibitive.
That is a central measured primitive, not an implementation footnote.

Operational accounting additionally includes per-tenant state isolation,
encryption, serialization, migration, eviction, corruption recovery, deletion,
and reset. Report amortized cost and the break-even number/length of learner
lifetimes, plus p50/p95 latency, throughput and batching loss. Active probes
are environment actions: report regret per environment step and performance at
a fixed query budget, not only final accuracy.

Compare Pareto frontiers because not every resource can be exactly equal.
Controls receive the same observations, feedback, boundary markers, task IDs
(preferably none), and interaction budget, with freedom to spend their state
budget optimally. Report static parameters, persistent/current bytes, forward
and update latency, environment calls, and offline training. TTT additionally
charges optimizer state and backward work; retrieval charges stored records,
index, traffic, and retrieval compute.

## 10. Historical paper gates and local falsifier

The prior
[`bounded-persistent-meta-learner-t70-local-preregistration.md`](bounded-persistent-meta-learner-t70-local-preregistration.md)
is retained as a historical control inventory, not a currently frozen or
admitted manifest. Its cheap lookup operator smoke has been completed: causal
state transport passed and the direct cross-boundary writer-credit hypothesis
failed. [T71](developmental-intelligence-gap-t71.md) supersedes the remaining
cross-family admission. No further T70 screen is admitted until a shared
mechanism grammar, held-out split, and updated comparator matrix are frozen
under T71. The prior summary gates were:

1. **Full-context information oracle:** the same backbone, given the complete
   experience transcript inside its context, must already solve each held-out
   family. This is the exact frozen per-rotation candidate checkpoint evaluated
   with transcript access, not a separately trained model of the same class.
   Otherwise the test confounds memory writing with absent task competence. The
   oracle must pass separately on every held-out family. After erasure, the
   candidate must recover at least `80%` of that oracle gain from the declared
   bounded state.
2. **Exact state path:** constant and independent-environment memory reduce
   post-erasure performance to the theorem ceiling; no transcript or
   environment-ID leak.
3. **Abstraction:** after two affine examples, the same learned updater answers
   held-out values and longer lifetimes; an example-memory control gets the
   same bits and work, and an optimal sufficient-statistic solver is the
   ceiling.
4. **Retention/change:** interleave many rules, change one, and require the
   unchanged rules to stay within one point while the changed rule is relearned.
5. **Active discovery:** on held-out DFAs, compare query count and final model
   accuracy with exact L*, generic recurrent meta-RL, and a context learner.
6. **Cross-family test:** freeze one updater and memory interface, then evaluate
   each held-out task family in a full leave-one-family-out rotation: train on
   `F\{j}`, freeze everything, test on `j`, then rotate all `j`. Use the declared
   shared typed interface with no family ID, prompt template fingerprint,
   family-specific adapter, tool, verifier, or tuning. Add factorial held-out
   combinations of familiar primitives. Every rotation must pass; a favorable
   mean cannot hide a failed family.

The first executable screen, if admitted, is a small from-zero CPU/local-GPU
model with a kilobyte-scale state. The principal causal ablation is the same
bounded operator trained without cross-boundary future credit; a
lifetime-trained GRU/SSM is a possible T70 implementation, not an external
defeater. Also compare optimally budgeted text/retrieval memory, TTT, and
context baselines. Kill the tested recipe unless it achieves a large
post-erasure gain, passes independent-environment state shuffles, retains prior
rules, extrapolates horizon, and clears the preregistered capability/resource
frontier across seeds.
Passing the synthetic screen would authorize a natural-domain pilot, not a
rented multi-hour run.

The control matrix includes reset/independent-state ablations, byte-matched raw
and reservoir examples, byte-matched learned episodic retrieval/compression,
task-optimal sufficient statistics where available, full context as an
information ceiling, Replay-MAML/PTW-style and TTT alternatives with their
update bills, and the larger frozen model only at the later natural gate. Every
competitor receives equal observations, feedback timing, actions, boundary
information, interaction budget, and tuning-search budget.

## 11. Superseded decision boundary

The old scaling ladder and readiness statement are superseded by T71. T70's
minimal witnesses retain order-one information-persistence bounds, but the
lookup result does not admit another experiment. A future bounded-state entrant
must use T71's declared transferable object, grammar, holdouts, and complete
adaptive comparator frontier. All local decisive, rented, and scaled work
remains held.
