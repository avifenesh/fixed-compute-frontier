# T83 explicit interface curriculum — independent audit

Date: 2026-08-02  
Verdict: **NO-RUN; RETAIN ONLY AS A THEOREM TARGET**

## 1. Agreement and disagreement between the audits

Both independent reviewers agree that the architecture is an occupied
conjunction. NEO already learns discrete executable primitives and latent
programs; Reusable Modules already identifies latent skills and routing
interfaces and uses SFT plus RL for unseen compositions; recursive/depth-latent
reasoners already supply recurrent hidden computation; Curriculum II and CoDE
already supply compositional autocurricula.

The first reviewer considered a CPU witness useful as a strict kill test. The
second reviewer applied the project's complete-capability rule and rejected the
run: a large synthetic scheduler win would establish active data selection, not
a new model mechanism, and would not earn GPU integration. The stricter verdict
controls because the proposed experiment cannot answer the model-level question
even if it passes.

## 2. Exact unoccupied seam

The only defensible remainder is:

> infer coverage deficits over a model's latent operator interfaces from raw
> traces and rewards, then adaptively select tasks that close those deficits,
> without true module IDs, composition labels, or generator internals.

This is an open curriculum/identification question explicitly adjacent to the
future-work boundary in [From Reasoning Traces to Reusable
Modules](https://arxiv.org/abs/2606.18089). It is not evidence for explicit
operators by itself.

## 3. Three missing theorem layers

### A. Interface identifiability

For true interfaces `E` and a raw-trace estimator `E_hat`, the proposal needs a
declared equivalence metric and a finite certificate

\[
\Pr[d(\widehat E,E)\le\epsilon_{\rm id}]
\ge 1-\delta_{\rm id}.
\]

The certificate must survive operator fusion, splitting, label permutation,
invertible latent changes of coordinates, and absorption of routing into an
operator. Otherwise coverage can be changed arbitrarily by re-encoding the same
function.

### B. Adaptive complete-cost coverage

Let curriculum action `a` realize interface `e` with unknown probability
`P_ae`, cost `c(a)`, and model-state-dependent noise. For required witness
counts `r`, a useful result has the form

\[
C_{\rm sched}
\le \rho\,\operatorname{OPT}(P,r,c)
+R(M,|A|,\delta,\epsilon_{\rm id}),
\]

where `OPT` is the strongest oracle adaptive coverage policy. It must also give
a family on which every frozen curriculum has a materially worse lower bound.
Task search, failed/repeated episodes, rollouts, verifier calls, and interface
estimation are charged.

### C. Coverage to capability

Visit counts must imply final reasoning quality:

\[
\mathcal L_{\rm comp}(\theta_T)-\mathcal L^\star
\le F(\epsilon_{\rm id},\epsilon_1,\ldots,\epsilon_M),
\]

with a corresponding lower bound when a required interface is absent. The
final model is evaluated frozen, after the scheduler and training traces are
removed.

No current T83 construction supplies all three layers.

## 4. Controls required if the theorem gate is later met

Cross explicit discrete recurrence and generic dense recurrence with both a
strong static/oracle curriculum and the adaptive interface curriculum. Add
faithful NEO, Reusable Modules SFT-plus-RL, Curriculum II, CoDE-style task
generation, difficulty/uncertainty/hard-example selection, an oracle deficit
scheduler, and random/diversity-matched selection.

Every arm receives identical raw tasks, tokens, rollout and verifier calls,
training FLOPs, tuning budget, persistent bytes, and serving compute. The dense
model receives the candidate's exact selected examples. Operator labels,
interfaces, composition IDs, and generator internals are forbidden to the
candidate unless every control receives them.

Required lesions are shuffled interface estimates, operator-specific removal,
architecture-held-fixed curriculum comparisons, and curriculum-held-fixed
architecture comparisons.

## 5. Frozen research gate

A CPU experiment is reconsidered only after the three theorem layers and a
frozen manifest exist. It must establish a simultaneous lower confidence bound
of at least `20%` against both same-architecture ordinary RL and same-data dense
recurrence on a complete acquisition-to-held-out-decision stage. It must include
independently generated mechanisms and one natural math, code, tool, or
scientific domain, with no protected regression above one absolute point.

A synthetic CPU pass earns neither GPU work nor integration by itself. Results
below `20%` follow the global gate: `10%` to less than `20%` are arguable but not
success; single-digit results close the lane. No run is currently admitted.
