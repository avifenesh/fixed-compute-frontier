# T75 Stage A decision — active evidence efficiency is real; learned intelligence remains untested

Date: 2026-08-02  
Status: **QUALIFYING COMPONENT SUCCESS; MODEL-FREE STAGE A PASS; LEARNED MODEL/GPU WORK REMAINS HELD**

## Outcome

The frozen CPU run passed every hard invariant and analytic gate.

| reference | accuracy | probes |
|---|---:|---:|
| constructive adaptive | 96.68% average; 96.30% worst world, exact | 168 fixed |
| fixed nonadaptive | 97.78% average; 95.40% worst world, exact | 1,280 fixed |
| entropy-greedy exact posterior | 96.03% average, simulated 95% CI 95.59–96.45% | 45.40 mean; 80 p95 |
| universal Fano/channel floor | at 95% average identification | 36.69 expected; 37 fixed-integer minimum |

At the same analytic worst-world threshold, constructive active localization
uses `1280/168 = 7.62x` fewer probes than the frozen fixed design. On average,
the entropy-greedy reference uses `28.20x` fewer probes than that fixed design
and only `1.237x` the universal information lower bound.

The greedy worst-world estimate was only `91.41%` with a conservative
Bonferroni lower bound of `80.10%`; it does not inherit the analytic worst-world
guarantee. That row is evidence of strong average efficiency, not a
worst-world pass.

## What was validated

- the 64 guarded causal worlds are distinct;
- passive evidence carries zero world information;
- the exact posterior agrees with a rational oracle and is order invariant;
- the audited `32`-location fixed-design lower-bound witnesses hold;
- the channel capacity is `0.1475894182` bits per charged result;
- Fano requires at least 37 probes for a fixed budget at 5% average error;
- a transparent adaptive policy achieves the required reliability with 168
  probes, versus 1,280 for the transparent fixed comparator;
- exact small-instance Bellman values decrease monotonically through horizon
  six, with 1,092 cached states.

This is a genuine order-one advantage for active, calibrated belief-guided
evidence collection. It validates one epistemic operator as worth learning.

## Classification under the 20% research gate

This is a qualifying **component research success**. Evidence acquisition is a
complete learning phase with a frozen input, output, quality floor, and probe
cost. At at least 95% average and worst-world exact identification, the
constructive active reference uses `168` rather than `1,280` probes: an
`86.875%` cost reduction, or `7.62x` fewer probes. That clears the project's
20% boundary analytically rather than through a favorable sample mean.

It is not a smarter-model result. The hypothesis class, posterior update, and
query rule were supplied algorithmically; no learned system discovered or
transferred them. The next unresolved claim is whether one trained updater can
retain this advantage across unseen world structures against matched learned
and exact-solver controls.

## What was not validated

No model learned the posterior, query policy, composition rule, persistent
state, or revision operation. The world has only 64 hypotheses and the
reference solver is short. Therefore this run establishes none of:

- a more intelligent language or multimodal model;
- neural compositional generalization;
- continual learning or consolidation;
- improvement over a matched learned router/modular control;
- exact `m=33` Bayes-optimal planning; or
- a reason to spend GPU time.

The exact-posterior entropy policy is a frozen heuristic. Its proximity to the
information floor does not prove optimality.

## Decision

Do not promote T75 directly to a neural run. Current compositional-learning
results already warn that isolated atomic curricula are weaker than diverse
compound curricula, and a positive T75 result would still reproduce a short
known algorithm.

Retain T75 as a unit test for the T76 developmental system. The next admitted
research action depends on the independent T76 audit: define a composition
lattice spanning at least three non-isomorphic interactive families and prove a
scaling or localized-revision advantage before selecting an architecture.

No local GPU was used. No rental was started.

Artifacts:

- [`guarded-causal-composition-t75-stage-a-preregistration.md`](guarded-causal-composition-t75-stage-a-preregistration.md)
- [`guarded-causal-composition-t75-stage-a-integrity-manifest.json`](guarded-causal-composition-t75-stage-a-integrity-manifest.json)
- [`guarded-causal-composition-t75-stage-a.json`](guarded-causal-composition-t75-stage-a.json)
- [`guarded-causal-composition-t75-stage-a-design-audit.md`](guarded-causal-composition-t75-stage-a-design-audit.md)
- [`../experiments/guarded_causal_composition_t75_stage_a.py`](../experiments/guarded_causal_composition_t75_stage_a.py)
