# T75 guarded causal composition — independent audit

Date: 2026-08-02  
Status: **CORE COMPOSITION WORLD PASSES WITH CORRECTIONS; MODEL-FREE CPU STAGE A CONDITIONALLY ADMITTED; NO NEURAL TRAINING OR NATURAL PILOT**

## Verdict

T75 is the first post-reset construction in this sequence that genuinely places
two previously separate operations in one held-out lifetime. Its passive
impossibility and noisy adaptive-search upper bound are correct. The exact
nonadaptive selector count is misstated: full world identification requires
`m-1`, not `m-2`, locations. This strengthens rather than weakens the active
separation.

The remaining blocker is attribution. Because binary search, Bernoulli belief
updates, and the typed conjunction are simple and recognizable, success can come
from pretrained solver synthesis or a hidden two-module router rather than an
update policy learned from T73/T74 component training. The present controls are
good but need base/single-family ablations, a true factorial objective design,
and one-shot contamination discipline.

After the exact paper corrections below are frozen, this audit admits only a
model-free CPU Stage A: generator/invariant checks, exact Bayes and nonadaptive
solvers, split/leak audit, and numeric-gate feasibility. It does not admit neural
training, GPU work, or a natural pilot.

## 1. Passive mutual information passes

For every selector and orientation, the passive `(X,Y)` law is T74's same
symmetric distribution. The selector chosen at each passive step is measurable
from a history whose law is already independent of `(t,Theta_0)`, and the next
passive outcome remains independent under every choice. The chain rule therefore
gives

\[
I((t,\Theta_0);\tau_{passive})=0
\]

for any adaptive passive selector/stopping policy, including random stopping.

State explicitly that `t` and `Theta_0` are independent of the model and surface
binding; the binding encodes no role; passive actions receive no score/change
side channel; and the stopping/sampling mechanism is world-independent. T75
must also declare `t` uniform if later Fano/Bayes controls use the uniform prior.

All `2(m-1)` worlds are behaviorally distinct for `m>=3`: `Theta_0` determines
the orientation at `u=0`, and the first opposite orientation determines `t`.

## 2. Noisy adaptive upper bound passes

At each tested selector, the midpoint classifier has conditional error at most

\[
\exp(-r g^2/2),\qquad g=1/2-\epsilon,
\]

under fresh independent intervention noise. First classify `Theta_0` at `u=0`,
then binary-search the first opposite orientation among `m-1` possible
thresholds. At most

\[
L=1+\lceil\log_2(m-1)\rceil
\]

selector tests are used. A union bound remains valid for adaptively selected
midpoints because the per-test error bound holds conditional on the prior
history. Thus

\[
r\ge (2/g^2)\log(L/\delta)
\]

and total interventions at most `Lr` are correct sufficient bounds.

Clarify that the high endpoint's orientation is structurally known to be the
opposite once `Theta_0` is correct; it need not be sampled before binary search.
The exact joint Bayesian posterior over `(t,Theta_0)` and a sequential noisy
binary-search policy remain the primary controls. The repeated midpoint rule is
only a transparent upper bound.

## 3. Nonadaptive selector count is `m-1`

The adjacent-threshold argument correctly forces every interior location
`u=1,...,m-2`. Those `m-2` locations are not sufficient to identify
`Theta_0`. The worlds

\[
(t=1,\Theta_0=0)
\quad\text{and}\quad
(t=m-1,\Theta_0=1)
\]

have the same orientation at every interior selector and differ only at the two
endpoints. Therefore an exact nonadaptive design also needs at least one of
`u=0` or `u=m-1`.

Conversely, `u=0,1,...,m-2` identifies the low orientation and the first switch.
Hence the exact noiseless selector-location complexity is

\[
q_{nonadapt}=m-1.
\]

At `m=33`, the comparison is at most `6` adaptive locations versus exactly `32`
fixed locations, not `31`. This is a location result only. Finite-noise total
sample, confidence-allocation, expected-cost, and high-quantile comparisons must
come from the exact frozen Bayes/minimax controls.

## 4. State and score claims pass with explicit assumptions

If the `2(m-1)` worlds are equiprobable and exact identity must survive erasure,
the worst-case class-index capacity is

\[
\lceil\log_2(2(m-1))\rceil
\]

bits, plus any association not supplied by stable surface identifiers. Under
error, apply Fano or rate distortion to a discrete/quantized state whose decoder
sees only the state and public protocol/binding information—not the erased
evidence transcript. A posterior generally costs more served bits than the final
class index and must be charged at frozen precision.

Define the scalar Brier convention, log-score clipping, posterior normalization,
and the exact formula for “gain recovered after erasure.” The full posterior is
the right task-optimal sufficient state; no neural representation can claim a
fundamental bit advantage over its best compressed code.

## 5. The conjunction holdout is real but not attribution-safe yet

The syntactic split is clear: T73 contains selectors/guards without causal
orientation evidence, T74 contains causal orientation evidence without a guard,
and T75's selector-plus-intervention event never occurs during updater training.
That is a legitimate factorial *input-combination* holdout.

It does not by itself prove learned recombination. The typed payload makes the
new problem recognizable, binary search and likelihood updating are classical,
and a pretrained or sufficiently capable model can synthesize their composition
at evaluation without using either component-training curriculum. Conversely,
one tensor can implement two internal modules and a conjunction-triggered router
despite having no explicit adapter.

Before freezing a neural protocol, add:

1. the untouched base checkpoint evaluated on T75;
2. T73-only, T74-only, joint T73+T74, and data/compute-matched unrelated-task
   training ablations from the same initialization;
3. a joint-training control with one primitive's evidence-to-update semantics
   counterfactually relabeled, to test whether the learned composition depends
   on the intended interfaces;
4. a matched two-solver router and an explicit learned modular composer under
   the same total parameter/state/training ledger; and
5. candidate-state/evidence interventions showing that corrupting the learned
   orientation evidence changes the corresponding threshold decisions while
   unrelated retained worlds remain stable.

For a pretrained backbone, do not claim that conjunction knowledge was absent
from pretraining. Attribute only the *incremental causal effect* of the frozen
component curriculum. A from-scratch tiny backbone makes the stronger curriculum
claim cleaner.

Use a hash-committed one-shot T75 decisive generator after all candidate
hyperparameters, prompts, serialization, and gates are frozen. A T75-trained
competence ceiling may use a separate generator only after candidate freeze and
must not feed tuning decisions back into the candidate. Vocabulary audits alone
cannot exclude algorithmic prior knowledge.

## 6. The proposed “factorial” comparison is an intervention ladder

Conditions 1–4 do not form a factorial design. Detachment changes temporal
credit, added lifetime terms change the loss, and exact-policy behavior cloning
changes target/oracle information. “Match supervised targets” also conflicts
with giving only one condition exact-policy labels.

Freeze at least a `2x2` core:

- boundary credit: full BPTT versus detached state; and
- objective: ordinary per-step losses versus added future
  regret/cost/calibration/retention terms.

Cross both factors on identical off-policy component lifetimes and target sets.
Run exact-policy supervision as a separate third factor or explicit oracle
control, with the same labels either present or absent in its matched cells.
Report actual gradient work; since detachment is cheaper, add a compute-matched
cell that may spend the saving on more steps rather than asserting identical
work.

The separate on-policy phase is correct. Freeze its reward access, intervention
budget, policy initialization, and stopping rule; once policies diverge, data
are intentionally no longer matched.

## 7. Router/modular controls are necessary and nearly sufficient

The explicit router, family heads, neural composer, exact Bayesian composer,
full-context/memory, test-time update, and T75-trained ceiling cover the main
alternatives. Strengthen their ledger as follows:

- sum both specialized models and router parameters, state, training data, and
  inference calls; do not compare the candidate with only one expert's cost;
- let the modular composer use both experts sequentially on T75 rather than
  forcing a one-family choice;
- distinguish the exact composer as an information/algorithmic ceiling from the
  learned matched controls; and
- add base and single-family curriculum ablations as above.

The candidate need not beat the exact composer. It must show a large curriculum
causal effect and beat the strongest resource-matched learned or explicitly
modular control. If the hand-composed learned modules match it, the result is
successful component learning, not evidence for an emergent unified updater.

## 8. Numeric gates require executable definitions

The directions of the gates are reasonable, but they are not freeze-ready:

1. Separate the directly T75-trained ceiling from a full-information Bayes
   ceiling; state the common intervention budget and require a nontrivial
   absolute competence floor even if the noisy ceiling is below `95%`.
2. Define recovered gain explicitly, including whether score is accuracy,
   regret, Brier, or log loss, and use simultaneous confidence intervals.
3. A `30%` regret reduction needs a positive denominator, an absolute minimum
   improvement, and a frozen strongest-control selection rule; percentages near
   zero regret are unstable.
4. Define Bayes-composer cost as expected, median, high quantile, or worst case
   at the same `delta`; `1.5x` is otherwise ambiguous.
5. Freeze Brier/log-score noninferiority margins, clipping, calibration bins or
   proper-score test, and multiplicity correction.
6. Define T73/T74 retention relative to each condition's pre-joint checkpoint or
   specialized ceiling and include uncertainty; “one point” is otherwise
   untyped.
7. Replace “frontier remains preferable” with a declared Pareto-dominance or
   scalar-selection rule over parameters, bytes, update/serve latency,
   interventions, and training cost.

Also freeze seed count, success aggregation, family-wise error or simultaneous
intervals, and a no-threshold-changing rule after the first decisive seed.

## 9. Prior-art boundary and claim language

T75 is a finite noisy change-point/threshold search with Bayesian hypothesis
testing. Direct controls include
[noisy generalized binary search](https://papers.nips.cc/paper_files/paper/2009/hash/556f391937dfd4398cbac35e050a2177-Abstract.html)
and the Bayesian noisy-search literature, including
[Ben-Or and Hassidim](https://doi.org/10.1109/FOCS.2008.58). The curriculum claim
also collides with established work on
[meta-learning for compositional generalization](https://arxiv.org/abs/2106.04252),
program/library induction such as
[DreamCoder](https://arxiv.org/abs/2006.08381), and current interactive causal
evaluation such as [CausaLab](https://arxiv.org/abs/2605.26029). ORBIT already
shows large cross-episode meta-RL gains on unseen interactive environments
([primary paper](https://arxiv.org/abs/2602.04089)).

These collisions do not close T75, because its legitimate contribution is a
tightly controlled causal comparison of curriculum/objective interventions.
They do close novelty claims around binary search, Bayesian updating,
compositional meta-learning, or interactive causal discovery.

Even a clean pass should be called a **large synthetic held-out-composition
gain** or **substantial compositional-learning improvement inside T75**. It is
not yet a substantial model-level intelligence/abstraction improvement: the
world has only `2(m-1)` hypotheses and the exact solution is a short known
program. A pass can admit a richer synthetic successor. A narrowly scoped
natural pilot requires independent replication plus transfer of the same frozen
updater to a more natural domain under T71's complete adaptive Pareto controls.

## Exact disposition

- Correct `q_nonadapt` from `m-2` to `m-1` and `31` to `32` at `m=33`.
- Keep the passive and noisy adaptive bounds with their assumptions.
- Freeze the uniform prior, state/score conventions, and one-shot split audit.
- Add base/single-family/counterfactual-curriculum controls.
- Replace the four-condition ladder with a genuine factorial core.
- Operationalize every numeric gate and router/modular ledger.
- After those paper corrections, admit only model-free CPU Stage A and protocol
  integrity checks. Their pass may admit a separately reviewed tiny CPU neural
  proposal; this audit itself admits no neural training, GPU use, or natural
  pilot.
