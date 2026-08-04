# Independent adversarial audit: cross-family persistent meta-learning (T56)

Date: 2026-08-01  
Scope: current primary-source and claim audit; no experiments run

## Bottom line

**The broad gap is not unoccupied.** ORBIT already trains a single LLM policy as an in-context online-learning algorithm on five partially observed task families and evaluates it on two disjoint families. MemoPilot and Memory-R2 learn memory update policies; SkillRise learns one policy that solves tasks while revising an abstraction-like skill document; Continual Harness autonomously revises prompts, memories, skills, sub-agents, and executable code over a continuous run. The proposed direction cannot claim novelty from “learned online learning,” persistent memory, cross-task transfer, abstraction revision, or improvement of a frozen executor separately.

One narrower conjunction remains unestablished:

> **A single frozen, family-agnostic updater learns from raw partial interaction to induce, causally reuse, and selectively revise abstractions across genuinely disjoint unseen task families, while improving a frozen base agent over deployment horizons far beyond its training horizon and context window, without task-family metadata, a stronger teacher, privileged schemas, or per-domain retraining.**

Call this capability **family-agnostic, horizon-extrapolating abstraction transport**. No reviewed work demonstrates it. However, this is presently a falsifiable capability specification or benchmark axis, not an architecture. It becomes an architectural research claim only after specifying a new learned state representation and update rule whose advantage cannot be reproduced by composing ORBIT-style meta-RL, SkillRise-style skill curation, and an external store.

**Research question: earned. New architecture run: not yet earned.** A run before formalizing the family split, abstraction semantics, information budget, horizon extrapolation, and one concrete updater would measure benchmark engineering or integration.

## What the named work actually occupies

| Work | What it establishes | Why it does not establish the remaining conjunction |
|---|---|---|
| [ORBIT](https://arxiv.org/abs/2602.04089) | A single meta-RL-trained LLM learns online from partial interaction. It trains on RPS, Minesweeper, Hangman, Wordle, and Blackjack, then improves on unseen Maze and Mastermind. | It retains full transcripts in a 32k context and evaluates only three episodes. It explicitly leaves longer horizons and memory augmentation for future work. Abstractions are implicit and neither identified, revised, nor causally tested. This is the strongest collision with the core T56 claim. |
| [ALMA](https://arxiv.org/abs/2602.07755) | A GPT-5 Meta Agent searches executable Python memory designs, including schemas, update, and retrieval, and outperforms hand-designed memories in four domains. | It runs a separate design search for each benchmark and selects a different best design per domain. Learning uses target-domain evaluation logs, benchmark feedback, GPT tools, and human-provided update/retrieve interfaces. Final testing is mostly static memory; it is not one deployed learned update algorithm transferring across families. |
| [MemoPilot](https://arxiv.org/abs/2606.08656) | A trained memory model iteratively revises structured hypotheses and action guidance to improve a frozen player. It transfers to a stronger player and revises after an opponent switch. | Separate memory models are trained for RPS and LHE. Default training lasts three games; the longer analysis trains for five and evaluates for ten. Opponent families are constructed from explicit strategy prompts, and this is within-game/opponent adaptation rather than disjoint-family transfer. |
| [Memory-R2](https://arxiv.org/abs/2605.21768) | LoGo-GRPO improves credit assignment for learned INSERT/UPDATE/DELETE memory construction over a curriculum of 8, 16, and 32 sessions, with OOD evaluation on other memory QA datasets. | The stream is persona dialogue followed by QA, not active decision-making across task families. Training rewards come from gold QA and a fixed large answer agent; the memory representation is atomic facts under a supplied operation set. It tests memory construction transfer, not autonomous abstraction transport. |
| [Continual Harness](https://arxiv.org/abs/2605.09998) | From a minimal-ish Pokémon interface, an LLM Refiner updates prompt, sub-agents, skills, code, and memory without episode reset; persistent refinements improve later play. | Red and Emerald remain one game genre. The interface includes an ASCII map and fixed meta-tools. The frozen-model variant relies on a capable frontier LLM doing the refinement; the weight-learning variant uses a PRM and Gemini-3.1-pro teacher. It does not show a learned updater checkpoint transferring to disjoint families. |
| [MemoryExplorer](https://arxiv.org/abs/2601.10744) | A multimodal policy is RL-finetuned to actively query a long-term spatial memory during embodied exploration and transfers from simulation to a physical office. | Its 3D/multimodal memory construction and retrieval pipeline are engineered. Deployment does not continually learn or revise abstract knowledge, and evaluation stays within embodied navigation/QA rather than disjoint families. |

Two very recent primary sources narrow the remaining gap further:

- [SkillRise](https://arxiv.org/abs/2607.26784) uses a single policy to alternate between task solving and rewriting an evolving skill document. Downstream rewards supervise curation, and test performance rises from sequences of two to six related tasks. But sequences are built from **environment-provided family metadata**, within the same task family, and experiments remain short and text-only. This already occupies “one learned policy forms and revises transferable abstractions from interaction.”
- [SERPO](https://arxiv.org/abs/2607.26873) updates one actor without labels, stronger judges, or external rewards, transfers OOD, and continues from HealthBench into ResearchQA. But it repeatedly optimizes on a fixed prompt set, changes model weights, and is neither partially observed interaction nor frozen-model improvement. It occupies cross-benchmark test-time self-evolution, not the remaining conjunction.

[PolySkill](https://arxiv.org/abs/2510.15863) additionally shows generalizable, compositional skill abstraction and self-exploration across websites. It further blocks “abstraction formation” alone as novelty.

## Is this a meaningful architectural claim?

Not as currently phrased. It joins six desirable properties:

1. one learned online-learning rule;
2. raw partial interaction;
3. disjoint-family transfer;
4. persistent long-horizon improvement;
5. abstraction induction and revision;
6. improvement of a frozen base model.

Showing the conjunction would be scientifically useful, but a conjunction is a benchmark result. A system assembled from an ORBIT-trained policy, a SkillRise text document, an ALMA-generated store, and a Continual Harness refiner could satisfy the checklist without introducing a new learning principle.

The architectural version must name the proposed computational object. A defensible form is:

\[
z_{t+1},\,u_t = U_\phi(z_t,o_t,a_t,r_t),
\qquad
a_{t+1}\sim F_\theta(o_{\le t+1},R(z_{t+1})),
\]

where the executor \(F_\theta\) and updater \(U_\phi\) are both frozen at deployment, \(z_t\) is a bounded persistent state, and no per-family compiler or prompt changes. The novel claim would have to concern how \(U_\phi\) discovers the factorization and revision operators of \(z\), or why its update dynamics yield family-uniform transfer and horizon extrapolation. Merely choosing text, a database, executable skills, or CRUD operations for \(z\) is already occupied.

“Substantial real-intelligence improvement” must not appear in the hypothesis or success criterion; it is not operational. Use online regret, forward transfer, retention, causal abstraction accuracy, and complete resource cost.

## Exact missing capability

The common missing capability is not memory. It is **autonomous abstraction transport under interface and horizon shift**:

- one updater checkpoint, not a separate memory model, code search, or fine-tune per domain;
- no task/family ID, environment-provided grouping, curriculum ordering, strategy description, gold QA, teacher relabel, or privileged schema;
- abstractions induced from raw observations, actions, and outcomes, rather than supplied as facts, memory fields, skills, or predicates;
- causal reuse across a new family with a different surface vocabulary, observation structure, action interface, and optimal policy;
- selective revision when a previously useful abstraction changes, followed by correct reuse when an old regime returns;
- bounded persistent state and improvement at horizons substantially beyond both training rollouts and the model context;
- a frozen deployed executor and updater, so improvement comes only from the learned online algorithm and its state.

Every reviewed work omits at least two of these; most omit four or more. ORBIT has the strongest claim to the first four but stops at three episodes and implicit in-context state. SkillRise has explicit revision and reuse but is given same-family grouping and tests only up to six tasks. Continual Harness has the longest and richest revision process but stays within Pokémon and relies on a frontier model or teacher.

## The strongest theorem that would matter

A theorem is only meaningful after defining a distribution over **families of POMDPs** with shared latent compositional mechanisms but independently permuted surface observations and actions. Let a family instance contain a latent abstraction program of description length \(d\), at most \(s\) mechanism changes, horizon \(T\), and persistent-state budget \(B\). Training and test families must have disjoint interfaces and task graphs.

The useful theorem form is:

1. a single frozen updater has family-uniform sublinear regret on unseen families;
2. its regret and state scale with latent program complexity \(d\) and change count \(s\), not raw observation/action alphabet size;
3. the same guarantee holds under arbitrary surface-symbol permutations and for \(T\) beyond the meta-training horizon;
4. after a mechanism switch, revision cost is bounded while performance on unaffected and later-returning mechanisms is retained;
5. a matched lower bound shows that a learner unable to carry reusable latent abstractions must pay a strictly larger adaptation term.

Do not claim a particular \(\widetilde O\) expression until the separation, mixing, reward, and identifiability assumptions are fixed. A theorem that assumes the latent variables, family alignment, correct abstraction library, change point, or compiler would simply put the missing capability into an oracle. Nor would such a theorem prove that gradient meta-training finds the updater; the empirical test remains load-bearing.

## Strongest falsifying experiment

Use a preregistered leave-one-family-out **family-of-families** protocol, not a memory QA benchmark.

### Protocol

- Meta-train one updater checkpoint on at least four interactive families and freeze it. Test on at least two held-out families with different observation and action grammars.
- Generate families from controlled latent mechanisms that can recur across disjoint surfaces: hidden type constraints, reversible transformations, resource conservation, causal preconditions, and regime changes. Include matched no-shared-mechanism controls.
- Randomize all labels, entity names, action tokens, and presentation conventions per family. Provide no family ID or grouping metadata.
- Give only online partial observations and environment outcomes. No gold abstraction, demonstrations, teacher, verifier, reset oracle, or future protected set.
- Train at short streams, then test at \(10\times\) and \(100\times\) the training horizon with a fixed persistent-byte budget. Include latent-rule switches, irrelevant distractor changes, and later returns to old regimes.
- Freeze both executor and updater during the entire test lifetime.

### Controls

Use ORBIT, SkillRise, MemoPilot-style learned curation, a Continual Harness refiner, full-history/RAG, a recurrent or state-space learner, and an exact Bayesian/MDL program learner over the same generator. ALMA can serve only as a per-family search upper bound. Give every control identical observations, action/reset rights, state bytes, executor calls, and training/inference compute.

### Measurements

- cumulative online regret and slope as horizon grows;
- forward transfer to a held-out family versus cold start;
- positive transfer on shared-mechanism pairs minus transfer on no-share controls;
- revision latency after a rule switch and recovery when an old regime returns;
- backward retention on unaffected mechanisms;
- state bytes, read/write traffic, model calls, wall time, actions, and outer-loop training cost;
- causal abstraction tests: interventions on a claimed abstraction must predict held-out downstream changes, and surgical removal or substitution must selectively remove the associated transfer gain.

### Decisive falsifiers

The candidate fails if any of the following holds:

- the experience-by-shared-mechanism interaction is not positive on held-out families;
- gains vanish under surface/action permutation or without family metadata;
- performance does not continue improving beyond the training/context horizon;
- per-family prompting, retrieval, or SkillRise-style document rewriting matches it at complete cost;
- its purported abstractions do not survive causal intervention tests;
- revision damages unaffected knowledge or returning regimes are no better than cold start;
- the advantage requires a stronger helper, hidden compiler, task labels, or more state/action budget.

This is stronger than reporting mean success across unrelated benchmarks. It tests whether one updater actually transports learned structure, rather than whether a large pretrained model can write useful notes in several settings.

## Run gate

**No implementation or training run is earned from the present claim alone.** The literature already demonstrates each component and two papers from 2026-07—SkillRise and SERPO—further compress the novelty window. A new run that simply combines persistent memory, skill rewriting, and meta-RL would be integration.

A small run becomes earned only after all of the following are fixed on paper:

1. the exact updater/state architecture;
2. the family-of-families generator and disjointness rule;
3. what counts as an abstraction and how causal reuse/revision is scored;
4. train/test horizon ratio and hard state/compute budgets;
5. oracle-free information interface;
6. matched controls and a preregistered falsifier.

The first earned run should be a small symbolic meta-POMDP pilot with exhaustive latent truth, not an expensive LLM benchmark. If it cannot show abstraction-specific cross-family transfer under permutation and horizon extrapolation there, stop. If it can, only then scale to language and embodied families.

## Final classification

- **Gap wholly unoccupied:** no.
- **Narrow conjunction unestablished:** yes.
- **Novel architecture as stated:** no; currently a benchmark/desiderata conjunction.
- **Potentially meaningful architecture:** yes, if centered on a single bounded frozen updater that discovers and revises its own transferable latent factorization.
- **Exact missing capability:** family-agnostic, horizon-extrapolating abstraction transport from raw interaction.
- **Any run earned now:** no; paper specification first, then one small falsification pilot.
