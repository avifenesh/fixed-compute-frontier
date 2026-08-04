# T56 lifetime learning signal — corrected diagnosis and surviving model hypothesis

Date: 2026-08-01  
Status: **INDEPENDENTLY REJECTED AS A DISTINCT ARCHITECTURE; NARROW CAPABILITY SPECIFICATION RETAINED; NO RUN**

## 0. Claim in one sentence

The broad missing object is not a larger context or a memory database. It is a
**domain-general update process whose persistent state changes are trained for
their effect on later performance across a lifetime**.

The model hypothesis was correspondingly narrow: meta-train a decoder with a
small bounded latent state that survives request boundaries, is updated from
interaction and feedback without translating the experience into prose, and is
credited only through later decisions. The state must improve on disjoint task
families and horizons longer than the training context. Otherwise this is only
another memory adapter.

Independent audit confirms the diagnosis but rejects the architecture claim.
The proposed latent state and recurrent update are generic meta-RL. The
remaining unestablished capability is **family-agnostic, horizon-extrapolating
abstraction transport**: one frozen updater must discover, reuse, and revise
causal abstractions across disjoint raw interfaces without task-family labels,
a stronger teacher, or a privileged compiler. This is currently a benchmark
and capability specification, not a new computational method.

## 1. What current evidence rules out

The following are not research claims:

- "LLMs need memory." External, latent, graph, belief, and trained memory
  systems already exist.
- "LLMs should learn at test time." ORBIT, LaMer, MemoPilot, Memory-R2, ALMA,
  Continual Harness, and deployment-feedback systems already do parts of this.
- "Prediction is better than storage." Predictive, belief-based, causal, and
  surprise-driven memory systems already occupy this framing.
- "Credit memory writes with future reward." Memory-R2, MemoPilot, and HiMPO
  already attack long-horizon memory credit.
- "Use a typed mechanism library." T55's independent audit showed that this
  hides the hard compiler, candidate-generation, routing, and verification
  problems and mostly integrates current work.

The empirical boundary is more specific:

1. **Cross-family online learning can be large.** ORBIT reports a Qwen3-14B
   model matching GPT-5.2 on unseen online environments after cross-episode
   meta-RL. This is the scale of gain we seek, but its adaptation history is in
   a 32,768-token context and the reported experiments use three episodes.
2. **Long-horizon memory can be trained.** Memory-R2 trains through 8, 16, and
   32 sessions with local rerollouts, but its task is multi-session
   conversational memory and question answering, not active acquisition of a
   new environment law.
3. **Learned memory updates can strongly improve a frozen actor.** MemoPilot
   reaches the best reported Elo in two games and transfers to stronger actors;
   its central tests use five games and a textual three-tier memory. Its
   StreamBench transfer is positive but modest.
4. **Memory design itself can be searched.** ALMA discovers executable memory
   programs, but specializes a separate design to each named domain with a
   meta-agent and code-search archive.
5. **Naive persistence is not enough.** CL-Bench finds that dedicated memory
   systems can underperform naive in-context learning and that agents overfit
   recent observations or fail to reuse latent structure across instances.
6. **Cross-task skill revision is now directly trained.** SkillRise uses one
   policy to solve tasks and rewrite a skill document using downstream task
   rewards. It reports 2.3--8.5 point gains, but receives related-task sequence
   groupings and tests short, same-family streams.
7. **Cross-benchmark self-evolution can update weights.** SERPO reports up to
   20-point in-domain gains and continued OOD evolution without a stronger
   judge, but it repeatedly optimizes weights on fixed prompt sets rather than
   learning from partial interaction with a frozen deployed updater.

The unverified intersection is therefore:

> One bounded, learned, non-prose state-update algorithm that actively learns
> hidden structure, persists beyond context, and transfers the act of learning
> to held-out task families.

This is an empirical intersection, not a claim that its ingredients are new.

## 2. Why ordinary model training does not learn this operation

Let ordinary training reset state for every example:

\[
J_{\mathrm{reset}}(\theta)
=\mathbb E_{(x,y)}\,\ell(f_\theta(x,s_0),y).
\]

Suppose an update operator emits `s_1=U_phi(s_0,x,y)`, but no later loss reads
`s_1`. Then

\[
\nabla_\phi J_{\mathrm{reset}}=0.
\]

This is not a limitation of transformers. The update is behaviorally
irrelevant under the reset objective, so any architecture trained only this
way receives no signal for becoming a better learner.

A lifetime sample instead contains related episodes:

\[
s_{i+1}=U_\phi(s_i,\tau_i,f_i),\qquad
J_{\mathrm{life}}(\theta,\phi)
=\mathbb E\sum_{i=1}^{T}\ell_i(\pi_\theta(o_i,s_i)).
\]

Now a later loss can credit an earlier update through its effect on `s_i`.
This changes what is learnable. It does not by itself specify a good state,
credit estimator, or architecture.

## 3. Exact information boundary

Let `X` be everything available on the current task without persistence, `H`
the earlier experience, and `Y` the future target. Under Bayes-optimal log
loss, the maximum possible gain from retaining history is

\[
\begin{aligned}
&\mathbb E[-\log p(Y\mid X)]
-\mathbb E[-\log p(Y\mid X,H)]\\
&\qquad=H(Y\mid X)-H(Y\mid X,H)
=I(Y;H\mid X).
\end{aligned}
\]

Consequences:

- if `I(Y;H|X)=0`, no continual learner has an expected information advantage;
- a benchmark can show large learning gains only when prior experience holds
  future-relevant information absent from the current request;
- storing history is not sufficient: the learned state must preserve the
  task-relevant portion of that conditional information;
- a static-knowledge gain and a learning gain must be separated by a reset
  control on the same ordered task stream.

For a hidden task variable `Z` uniform over `N` possibilities and a persistent
state `S` with at most `B` bits, Fano's inequality gives, for identifying `Z`,

\[
P_e\ge 1-\frac{B+1}{\log_2 N}.
\]

No learned compression can retain an arbitrary `N`-way environment identity
in fewer than logarithmic bits. The intended gain is therefore not magical
unbounded memory. It is learning a compact sufficient statistic when the
environment has reusable low-dimensional structure.

## 4. Exact persistence separation, and its limit

Consider a lifetime with a hidden bit `z`. The first episode reveals `z`; all
later current observations are identical, and action `a=z` earns one. A policy
whose visible window has already discarded episode one has expected reward at
most `1/2` per later episode. One persistent bit achieves reward one.

Thus a fixed window incurs expected regret `(T-1)/2`, while the persistent
one-bit policy has zero regret after observation.

This theorem proves only that persistence can be necessary. A one-bit file,
text token, Bayesian belief, recurrent cell, or latent slot all tie. It does
not select an architecture. The candidate earns a model result only by
learning what to retain and how to use it across held-out families better than
these matched controls.

## 5. Rejected generic model sketch: Persistent Latent Meta-Learner (PLML)

PLML adds two bounded state banks to a decoder:

\[
F_i\in\mathbb R^{k_f\times d},\qquad
L_i\in\mathbb R^{k_l\times d}.
\]

`F_i` is fast within-task belief/working state and is reset at a declared task
boundary. `L_i` is slow lifetime state and persists across requests. The actor
is

\[
(a_i,H_i)=\pi_\theta(o_i,F_i,L_i),
\]

and the update is

\[
F_{i+1}=U_f(F_i,H_i,a_i,o_{i+1}),
\]

\[
L_{j+1}=U_l(L_j,\operatorname{Pool}(H_{j,1:n}),a_{j,1:n},f_j).
\]

The state banks enter selected transformer layers through a gated
cross-attention or KV-prefix path. The update is a small shared recurrent
module. No state update is required to emit language. A diagnostic decoder may
translate state to prose for audit, but the actor never depends on that prose.

Training samples whole lifetimes. Early episodes can rationally explore;
later episodes must exploit and retain discoveries. Gradients or policy credit
reach `U_l` only through downstream lifetime return. Local rerollouts from an
identical pre-update state are a required credit-assignment control, not a
novelty claim.

The architecture is intentionally less expressive than T55:

- no mechanism DSL;
- no program compiler;
- no unbounded edit payload;
- no oracle variable or schema proposal;
- no growing expert library;
- fixed parameter, state-byte, and per-token serving budgets.

Its risk is correspondingly clear: an unrestricted recurrent state is a direct
control and may learn exactly the same function. The independent audit treats
that as fatal to a distinct architecture claim. PLML remains only a neutral
interface for stating the narrower benchmark.

## 6. What would be gained

If the hypothesis survives, a smaller model should acquire a capability that a
larger stateless model must repeatedly reconstruct:

1. improve over repeated interaction rather than start each request from the
   pretrained prior;
2. carry learned strategies beyond the context window at fixed state size;
3. infer hidden rules through exploration, not merely recall supplied prose;
4. transfer the learning procedure to new surface domains and task families;
5. preserve serving speed by reading a small fixed state instead of an
   ever-growing raw history.

The target is breakthrough-sized: at the same actor parameter count and
complete serving budget, at least 30% lower cumulative regret on held-out task
families and long horizons, with no more than 2% protected static-capability
loss. A small one-domain memory gain does not count.

## 7. Required controls and complete cost

All systems receive the same observations, actions, feedback, ordering, and
task-boundary information. Required controls are:

1. reset actor with no history;
2. raw sliding context and full history where it fits;
3. matched-bit textual summary and retrieval memory;
4. exact Bayesian or recurrent belief-state learner on synthetic families;
5. ORBIT/LaMer-style cross-episode context learner;
6. same latent adapter trained for reconstruction or next-token loss rather
   than lifetime return;
7. generic GRU/state-space memory with identical persistent bytes and FLOPs;
8. online LoRA or fast-weight update with matched write and read cost.

The ledger includes actor and adapter parameters, persistent bytes, KV/cache
bytes, generated and ingested tokens, update FLOPs, environment interactions,
rollouts/rerollouts, meta-training compute, and latency at batch 1 and serving
batch.

## 8. Pre-run falsifiers

No GPU run is admitted until an independent audit answers these questions:

- Does current work already combine latent persistence, cross-episode meta-RL,
  active exploration, long horizons, and held-out task-family transfer?
- Is PLML distinguishable from an ordinary recurrent meta-RL agent after
  matched state and compute?
- Can a finite benchmark expose the claimed intersection without supplying a
  hidden task ID, hand-built schema, or oracle boundary?
- Does the lifetime objective provide any optimizer difference beyond a fixed
  baseline subtraction or reward re-labeling?
- Can the first local test falsify state learning rather than merely verify the
  obvious one-bit persistence theorem?

If the generic recurrent control ties, the architecture claim is closed. If
only same-family or short-horizon gains appear, the intelligence claim is
closed. If the state cannot outperform a matched textual sufficient statistic,
latent representation has no demonstrated advantage.

## 9. Independent audit and run decision

The
[independent audit](cross-family-persistent-meta-learning-t56-independent-audit.md)
finds that ORBIT already supplies one-policy cross-family online learning,
SkillRise supplies learned abstraction-like revision, and Continual Harness
supplies persistent long-run scaffold revision. Their conjunction does not
create an architecture.

**No run.** The information theorem and persistence separation justify the
problem, not PLML. The next paper object must specify how the updater discovers
and revises a transferable latent factorization. Only then can a small symbolic
family-of-families falsification pilot be considered.

## 10. Primary current controls

- ORBIT: <https://arxiv.org/abs/2602.04089>
- LaMer: <https://arxiv.org/abs/2512.16848>
- ALMA: <https://arxiv.org/abs/2602.07755>
- MemoPilot: <https://arxiv.org/abs/2606.08656>
- Memory-R2: <https://arxiv.org/abs/2605.21768>
- Continual Harness: <https://arxiv.org/abs/2605.09998>
- CL-Bench: <https://arxiv.org/abs/2606.05661>
- Learning on the Job: <https://arxiv.org/abs/2607.22157>
- Trained persistent decoder memory: <https://arxiv.org/abs/2603.22329>
- Memory representation and negative transfer: <https://arxiv.org/abs/2604.27003>
- HiMPO: <https://arxiv.org/abs/2606.16285>
- SkillRise: <https://arxiv.org/abs/2607.26784>
- SERPO: <https://arxiv.org/abs/2607.26873>
- PolySkill: <https://arxiv.org/abs/2510.15863>
