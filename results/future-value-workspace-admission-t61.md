# T61 future-value workspace admission — useful training seam, not a new intelligence mechanism

Date: 2026-08-01  
Status: **WORKSPACE DIAGNOSTIC RETAINED; DISTINCT MECHANISM REJECTED; NO RUN**

## 0. Result in one sentence

The tested Claude production models appear to use a small, privileged internal
workspace for deliberate reasoning, and current work has not identified the
mechanism that decides what enters it.  Training a capacity-capped gate to
retain features with high future decision value is a valid intervention, but
it does not create a new source of information: ordinary end-to-end credit can
already train an implicit selector, current J-space work already supplies
sparse recurrent writes and learned reads, and greedy marginal value scores can
provably miss synergistic facts.  This is an instrumented training seam, not
the missing lifetime updater.

## 1. Candidate and exact scope

Let a backbone produce candidate internal features

\[
C=(C_1,\ldots,C_n),
\]

and let a workspace gate choose a subset `S`, `|S| <= B`, for recurrent reuse:

\[
S\sim q_\phi(S\mid C,X,G),\qquad
W=C_S,\qquad
Y=D_\theta(X,G,W).
\]

The proposed change was to train `q_phi` by a declared future-task distribution
and its later task loss, rather than by activation magnitude, reconstruction
error, or next-token salience.  The selector may receive only `(C,X,G)` at
deployment; future labels and outcomes are training-only.  One possible paired
counterfactual score, estimated by rerunning the same frozen future example
with and without a retained item, is

\[
u_i=L_{future}(W\setminus i)-L_{future}(W).
\]

This makes the desired causal question explicit: did retaining candidate `i`
change a later decision?  It is not identifiable from one unpaired online
trajectory.  Reruns, masks, labels, estimator variance, and all extra forward
work must be charged, and the full-coalition score is not an additive ranking
rule.

## 2. A limited workspace can help only under a task distribution

Suppose `C_1,...,C_n` are independent fair bits and the future goal `G` asks
for one uniformly selected bit.  First impose the narrow condition that `S` is
independent of both `C` and `G` and that no content-dependent routing state is
visible to the decoder.  If `E|S| <= B`, then the requested bit is absent with
probability at least `1-B/n`.  On that event no decoder can beat error `1/2`, so

\[
P_e\ge \frac{1}{2}\left(1-\frac{B}{n}\right).
\]

If selection depends on `C`, this bound is false: the chosen indices themselves
can encode omitted content.  A complete representation is `(S,C_S)`, and the
subset address alone can carry up to

\[
\log_2\left(\sum_{k=0}^{B}{n\choose k}\right)
\]

bits when order is irrelevant.  Values, order, carrier identity, timing, and
router state add further rate.  A valid workspace budget must charge the whole
channel, not just the number of values selected.

If `G` is supplied before selection, a one-slot addressable selector can retain
`C_G` and achieve zero error.  The gain is real, but it comes from
goal-conditioned allocation and declared task structure.  It is not universal
compression and does not imply that a particular neural gate will discover the
rule.

This is the workspace version of T59: what may be discarded depends on the
future query family.  Unrestricted future queries require retaining every
distinction that some query can expose.

## 3. Individual future-value scores are not sufficient

Let `C_1,C_2` be independent fair bits and let the answer be

\[
Y=C_1\mathbin{\operatorname{XOR}}C_2.
\]

Then

\[
I(Y;C_1)=I(Y;C_2)=0,
\qquad
I(Y;C_1,C_2)=1\text{ bit}.
\]

For the set utility `F(S)=I(Y;C_S)`, a feature has zero marginal value at the
empty set and one bit of marginal value after its partner is present.  An
admission rule based on univariate relevance or forward greedy selection from
the empty set can therefore reject both indispensable features.  The displayed
full-coalition leave-one-out score does *not* fail on this example: removing
either member from `{C_1,C_2}` destroys perfect prediction.  The exact result is
that task value need not be submodular, so the marginal depends on the coalition
and search path.  Greedy `(1-1/e)` subset-selection guarantees apply only after
normalized monotonicity and submodularity are established; they do not hold for
arbitrary reasoning states.

Estimating full coalitional value is itself combinatorial.  Approximations can
be useful, but their unmeasured interaction order becomes part of the claimed
capability boundary.

## 4. Why the gate does not add model expressivity

For a fixed finite unroll, fixed precision, matched inputs and randomness, and
fixed `q_phi` and `D_theta`, a sufficiently unrestricted network can represent
the composed output distribution

\[
(C,X,G)\mapsto D_\theta(X,G,C_{q_\phi(C,X,G)}).
\]

An explicit gate changes the factorization, bottleneck, optimization path, and
possibly physical execution.  Under that fixed resource model, it does not
establish an extensional function-class separation from an equal-or-more-
expressive simulator.  This says nothing about equal learnability, sample
efficiency, parameter efficiency, or physical execution.  Unbounded recurrence,
dynamic stopping, external memory, exact addressing, or additional work changes
the resource model.  Therefore a model-level gain must come from one of three
empirical claims:

1. a better inductive bias or credit path at matched training data and FLOPs;
2. a real conditional-compute saving that is measured and reinvested; or
3. a robustness/generalization property induced by the bottleneck.

None follows from workspace capacity alone.  End-to-end future loss already
backpropagates through any differentiable admission path.  Calling that signal
"future value" does not define a different updater, and a within-episode
workspace is not persistent cross-episode learning.

## 5. Collision with current work

The new mechanistic evidence is still important:

- The 16 July 2026 v1 preprint *Verbalizable Representations Form a Global
  Workspace in Language Models* reports, in the Claude models it tested, that
  J-space holds on the order of tens of concepts, supports silent multistep
  reasoning, and is mechanistically privileged.  It explicitly says that the
  mechanism causing representations to enter the workspace remains unknown
  and warns that the J-lens and flat bag-of-concepts view are incomplete.
- The same work's counterfactual-reflection training shows that supervising a
  later reflective continuation can implant causally used concepts in the
  earlier workspace.  Future text influencing workspace content is therefore
  already demonstrated.
- The 24 July 2026 v1 preprint J-CoT writes a sparse,
  vocabulary-indexed recurrent state through a
  nonnegative elastic-net extraction and learns carrier embeddings plus a read
  gate.  Its adaptive support is driven by current hidden activation and
  reconstruction, not an explicit coalitional future-value estimator, but its
  full interface is trained through downstream task examples.
- GRU-Mem trains update and exit gates with end-to-end rewards for long-context
  reasoning.  It is another direct control against a generic "learn what to
  retain" claim.
- *Reasoning as Compression* already formulates reasoning traces as a
  conditional-information bottleneck trading task reward against compressed
  intermediate computation.  It is not a workspace architecture, but it
  occupies the generic task-value-versus-rate objective.

Primary references:

- https://arxiv.org/abs/2607.15495
- https://arxiv.org/abs/2607.21981
- https://arxiv.org/abs/2602.10560
- https://arxiv.org/abs/2603.08462

## 6. What is retained

The workspace is a valuable **measurement and intervention surface**.  A later
candidate should use it to test whether a proposed learner actually formed and
used a new abstraction:

1. identify candidate workspace content before and after experience;
2. swap or ablate the putative learned concept;
3. require the predicted downstream decision change;
4. compare with equally trained dense and recurrent controls; and
5. charge recurrent cycles, extraction, gating, and training credit.

This can causally validate a mechanism discovered elsewhere.  It is not that
mechanism.

## 7. Run decision

**No run.**  The candidate fails before implementation:

- no new evidence-acquisition channel;
- no retention beyond the current reasoning episode;
- no distinct credit rule beyond downstream training loss;
- no general tractable guarantee under feature synergy or an unpriced routing
  channel; and
- no reason to expect the required 10-point or 30%-regret improvement over
  matched J-CoT, dense-latent, gated-memory, and ordinary end-to-end controls.

Reopening requires a non-additive but tractable admission/update rule with a
provable sample, horizon, or computation separation on a family that a matched
recurrent learner cannot absorb.  Until then, workspace admission is closed as
the primary intelligence architecture.

The independent audit found and corrected the selected-index side channel and
the overbroad XOR claim:

- [T61 independent audit](future-value-workspace-admission-t61-independent-audit.md)
