# Independent audit: compositional causal mechanism recombination (T51)

Date: 2026-08-01  
Scope: theorem and architecture audit only; no experiment, implementation, or edit to T51  
Audited artifact: compositional-causal-mechanism-recombination-t51-paper.md

## One verdict

**REVISE — retain T51 as a transparent-interface control lemma, but do not admit it as an architecture candidate.**

T51.1 and T51.2 correctly show how to decode a known finite mechanism codebook when the graph, slots, local intervention interface, parent-value controls, and per-slot outputs are supplied. The noisy majority bound is a valid sufficient bound. T51.3 also gives a correct worst-case membership-query lower bound for an unrestricted Boolean function.

Those facts do not establish an architecture separation. The claimed exponential contrast changes the hypothesis class, query interface, output bandwidth, and supplied side information simultaneously. A strongest learner given the same modular promise and local oracle can perform the same code lookup without adopting the proposed neural architecture. The raw-interface problem that could distinguish an architecture is explicitly unsolved, and closely related causal-representation and compositional-world-model work already exists.

## 1. Transparent theorem: what is proved

Conditional on a known library

\[
\mathcal M=\{f_1,\ldots,f_M\},
\]

a known typed acyclic composition graph, directly addressable slots, the ability to set every slot's local parents to chosen legal inputs, and direct observation of each slot's binary output, a separating anchor set

\[
A=(u_1,\ldots,u_s)
\]

induces an injective codebook \(m\mapsto c_m\in\{0,1\}^s\). In the noiseless case, evaluating those anchors identifies every slot assignment. The counts \(rs\) scalar observations and \(s\) rounds are correct when all \(r\) slots can be probed in parallel.

This is not yet causal discovery or representation learning. It is \(M\)-ary codebook decoding independently at each supplied slot. The causal graph is not inferred, the slot decomposition is not inferred, the intervention bindings are not inferred, and the mechanisms are not learned in T51. They are all inputs to the theorem.

The paper should make the following assumptions explicit:

- Every mechanism in a shared codebook has the same declared input type, or the library is partitioned by type as \(\mathcal M_\tau\) with separate anchors \(A_\tau\). A single \(f_m:\mathcal U\to\{0,1\}\) is otherwise inconsistent with heterogeneous typed slots.
- Input ports are labeled, or behavior is defined modulo a declared port-permutation symmetry.
- Parent values can be set without uncontrolled upstream effects, and a slot's output can be read locally. These are stronger than ordinary global interventions on a composed system.
- The library contains the full structural functions, not only their anchor codes; otherwise recovered identities do not imply prediction on unqueried legal inputs.
- The composition semantics, boundary inputs, and intervention semantics are known and stable.
- “Exact prediction” means prediction of the noiseless structural response. With observation noise, individual noisy outcomes are not exactly predictable, although their distribution may be.

## 2. The bit floor is not a logarithmic upper bound

The counting argument

\[
s\ge \lceil\log_2 M\rceil
\]

is correct for a fixed binary signature that distinguishes \(M\) mechanisms. It is only a lower bound.

Define the actual separating-set complexity

\[
s^*(\mathcal M)=
\min\left\{
|A|:
m\mapsto (f_m(u))_{u\in A}
\text{ is injective}
\right\}.
\]

Then

\[
\lceil\log_2 M\rceil\le s^*(\mathcal M)\le M-1
\]

for a pairwise behaviorally distinct finite library, but \(s^*(\mathcal M)\) need not be \(O(\log M)\). For example, let \(f_0\) be zero everywhere and let each \(f_i\), \(1\le i\le M-1\), be one only on its private input \(u_i\). Identifying \(f_0\) requires querying all \(M-1\) private inputs.

An adaptive decision tree has its own complexity \(D^*(\mathcal M)\). It also has the information lower bound \(\lceil\log_2 M\rceil\), but its worst-case depth can be \(M-1\). T51 must therefore replace generic \(O(\log M)\) adaptation claims by bounds in \(s^*(\mathcal M)\) or \(D^*(\mathcal M)\). The logarithmic regime is valid only as an additional balanced-code or separating-anchor assumption; the bit floor does not prove that such anchors exist.

For typed libraries the exact accounting becomes, for example,

\[
N_{\rm obs}=\sum_\tau r_\tau s^*_\tau,
\qquad
N_{\rm round}=\max_\tau s^*_\tau
\]

when all types can be probed concurrently.

## 3. Noisy recovery bound

Let each repeated binary observation be flipped independently with probability \(\eta<1/2\), and put

\[
\gamma=\frac12-\eta.
\]

For one slot-anchor pair, majority decoding fails only if the empirical flip rate is at least \(1/2\). Hoeffding therefore gives

\[
\Pr(\widehat c_{j,\ell}\ne c_{m_j,\ell})
\le \exp(-2q\gamma^2).
\]

A union bound over \(rs\) decoded bits yields failure probability at most

\[
rs\,e^{-2q\gamma^2},
\]

so the paper's sufficient condition is correct after integer rounding:

\[
q\ge
\left\lceil
\frac{\ln(rs/\delta)}{2\gamma^2}
\right\rceil.
\]

Use odd \(q\), or specify how majority ties count. Independence across different slots or anchors is not needed for the union bound, but independence of the \(q\) repeated flips for each decoded bit is needed for Hoeffding. The same bound works for heterogeneous noise rates if \(\eta\) is a known uniform upper bound below \(1/2\).

The resulting counts

\[
N_{\rm obs}=rsq,
\qquad
N_{\rm round}=sq
\]

are correct under the common-anchor, fully parallel interface. Replace \(s\) with the actual separating complexity, not automatically \(\log M\). The result is conservative: nearest-code or maximum-likelihood decoding can exploit \(d_{\min}\) and avoid requiring every code bit to be correct. Thus T51.2 is a valid sufficient guarantee, not a minimax or coding-optimal noisy theorem.

## 4. The \(2^r\) atomic lower bound

The lower bound itself is exact. To identify an arbitrary deterministic function

\[
g:\{0,1\}^r\to\{0,1\}
\]

from scalar membership queries in the worst case, every one of the \(2^r\) inputs must be queried. If one point remains unqueried, the all-zero function and a function supported only at that point have identical transcripts.

It is not a fair architecture comparison with T51:

1. The atomic learner ranges over all \(2^{2^r}\) Boolean functions; T51 is promised a much smaller family generated by a known graph and known mechanism library.
2. The atomic learner receives one global scalar output per query; T51 reads \(r\) local outputs per parallel round.
3. The atomic learner can set only a global input; T51 can directly set every slot's local parent values, including internal variables.
4. T51 receives the factorization, types, mechanism definitions, anchor set, and composition rule as side information.

The result is therefore an exact **assumption-and-interface contrast**, not an architectural sample-complexity separation. The paper substantially acknowledges the hypothesis-class issue, but “strict black-box separation” and the downstream architecture-candidate status still overstate what follows.

A matched comparison must fix one environment family \(\mathcal F\), one data distribution, one intervention oracle \(\mathcal O\), and one resource ledger. Every control receives the same library promise, graph information, local observations, and interventions. An unstructured universal learner can implement the same optimal codebook decoder; refusing to use supplied side information is not a defensible control restriction. Any remaining claim must be a computational, storage, approximation, or learned-interface edge under those matched conditions.

## 5. Description-length claim

A fixed-length encoding of \(r\) slot identities gives the conditional upper bound

\[
L_{\rm composition}
\le r\lceil\log_2 M\rceil+L_{\rm graph}.
\]

It is not generally an equality or a complete-description comparison.

Exact fixes:

- State the conditioning. The bound is conditional on a shared mechanism library, type system, input-port semantics, composition interpreter, and decoder.
- If the graph is supplied as observable environment state, condition on it and set its per-environment description charge to zero. If it must be transmitted or inferred, define the graph code and charge it.
- Charge \(L(\mathcal M)\), the anchor/codebook description, the interface and type grammar, and the encoder/router in total cost. If they are shared across \(K\) environments, show the amortized term explicitly.
- Replace \(r\lceil\log_2 M\rceil\) by \(\sum_j\lceil\log_2 M_{\tau_j}\rceil\) for typed candidate sets, or by a code for the number of behaviorally distinct assignments. Graph automorphisms and semantically equivalent compositions can reduce this number.
- A truth-table representation uses \(2^r\) bits for the declared input order. An optimal description has a \(2^r\)-bit worst case because there are \(2^{2^r}\) Boolean functions, but individual functions may compress.
- Ensure that the unrestricted Boolean function and the composed mechanism system have the same external input and output domain. T51 currently does not formally connect the generic local domain \(\mathcal U\) and typed composition to the scalar \(r\)-bit global function.

The fair statement is: the modular promise defines a lower-entropy conditional model class than unrestricted Boolean functions. That is important, but it is a property of the prior/model class, not proof that one architecture discovers or uses it better than another.

## 6. Identifiability and no-go statements

### What holds

The interventional-collision no-go is correct. If two mechanisms agree on every legal local intervention, their names are not behaviorally identifiable and must be quotiented into one equivalence class.

The novel-composition warning is also correct in substance: without an invariance assumption linking old and new arrangements, training data cannot determine responses on an unseen composition. Mechanism drift and residual interaction terms likewise invalidate the factorized model.

### What remains informal

T51.4 is not yet a theorem section. “Multiple factorizations can induce the same observed law” and “two predictors can disagree arbitrarily” are correct intuitions, but each needs an explicit pair of observationally/interventionally equivalent models and a declared equivalence relation.

The missing-interface case must distinguish at least:

- unavoidable global relabeling of latent variables;
- graph automorphisms that preserve all mechanism labels and allowed interventions;
- componentwise reparameterizations allowed by the observation model;
- Markov or interventional equivalence of distinct graphs;
- genuine behavioral collision of mechanism functions; and
- failure caused only by insufficient intervention coverage.

The positive theorem identifies assignments from a known library; it does not establish how the library was learned or canonically aligned across environments. The title “learn laws once” therefore needs either a separate meta-training theorem and cost ledger or an explicit statement that library acquisition is outside T51.

## 7. Does this earn an architecture candidate?

No. The broad capability target is important, but the current architecture contract is an unproved diagram downstream of a privileged lookup theorem.

The novelty bar is also materially higher than T51 states:

- [Variational Causal Dynamics](https://arxiv.org/abs/2206.11131) jointly learns latent representations and causally factorized dynamics, discovers sparse structure and intervention changes, and performs modular adaptation.
- [DECAF](https://proceedings.mlr.press/v236/talon24a.html) detects reusable versus environment-specific causal factors from temporal images with intervention-target supervision.
- [General Identifiability and Achievability for Causal Representation Learning](https://proceedings.mlr.press/v238/varici24a.html) gives recovery guarantees for latent variables and causal models even when pairs of intervention environments are not matched to their target node.
- [Nonparametric Identifiability of Causal Representations from Unknown Interventions](https://proceedings.neurips.cc/paper_files/paper/2023/hash/97fe251c25b6f99a2a23b330a75b11d4-Abstract-Conference.html) recovers latent variables and causal structure, up to unavoidable ambiguities, under unknown interventions.
- [Compositional Models for Estimating Causal Effects](https://proceedings.mlr.press/v275/pruthi25a.html) already studies modular causal models, sample efficiency, and unseen component combinations.
- [WM3C](https://proceedings.iclr.cc/paper_files/paper/2025/hash/79d86433c2acd12b6fa98553435d226e-Abstract-Conference.html) learns compositional causal components for unseen-environment adaptation and includes identification claims and robotic evaluation.
- [Dreamweaver](https://proceedings.iclr.cc/paper_files/paper/2025/file/ae82a60c5ce50b5c4d18cfe3214eb684-Paper-Conference.pdf) learns compositional world models from pixels.
- The July 2026 [Mechanistic World Models](https://arxiv.org/abs/2607.12474) paper explicitly organizes world models around reusable mechanisms.

These works do not automatically solve T51's exact finite codebook problem, but they already occupy the architecture and raw-representation territory T51 claims. A supplied-interface decoding lemma is below, not beyond, that frontier. Candidate admission requires a new theorem or total-cost empirical edge against these matched controls.

## 8. Audit of the proposed next object

The proposed object is:

> hidden slot identities and parent wiring under a per-environment permutation, with intervention targets and local temporal observations still available.

As stated, it is underdefined and may be trivial or already dominated.

- If observations are exactly a permutation of otherwise directly observed slots, the permutation is a label gauge, not a hidden representation problem. Distinct behavioral signatures align mechanisms exactly as in T51.
- If intervention targets are reported in canonical cross-environment coordinates, that metadata reveals the alignment and restores a privileged interface.
- If targets are reported only in each environment's observed coordinates, they support local causal discovery but do not by themselves canonically align environments.
- If each local transition and its parents can still be read and set directly, “hidden wiring” is largely a standard interventional graph-recovery problem.
- If the observations are an unknown nonlinear mixture instead, the problem becomes causal representation learning and must improve on the identifiability results cited above.

### Repaired theorem contract

The next proof should be stated as:

> **Finite-sample recovery of a reusable mechanism-labeled dynamic DAG from a per-environment unknown sensor permutation, where interventions are addressed and reported only in local sensor coordinates, up to the joint automorphism group of the graph, mechanism library, and intervention family.**

It needs all of the following before proof work:

1. **Generative model.** Define latent state \(z_t^e\), the exact observation map \(x_t^e=P_{\pi_e}z_t^e\), the unknown graph \(G_e\), mechanism assignment \(m_{e,j}\), noise, input-port semantics, and whether graphs may genuinely rewire across environments.
2. **Intervention oracle.** State whether an intervention sets a currently observed coordinate, a latent canonical variable, a mechanism, or an entire parent tuple; whether pre/post pairs and resets exist; and which target identifier is returned.
3. **Equivalence target.** Recover only
   \[
   (\pi_e,G_e,m_e)/\operatorname{Aut}(G_e,\mathcal M,\mathcal I),
   \]
   because symmetric variables and mechanisms cannot be canonically distinguished.
4. **Positive assumptions.** Declare causal sufficiency or confounding structure, maximum indegree, intervention coverage, persistence of mechanisms, a nonzero interventional separation/faithfulness gap, controllability, and code separation including input-port symmetries.
5. **Finite-sample bound.** Give rounds and scalar observations as an explicit function of \(r,M,d,\delta\), noise, and the minimum separation gap. The logarithmic code term must use \(s^*(\mathcal M)\) or decision-tree complexity, while graph discovery carries its own intervention and estimation cost.
6. **Matched lower bound.** Use the identical latent family, observation permutation, intervention oracle, and output bandwidth. Do not compare with unrestricted scalar black-box functions.
7. **No-go theorem.** Construct two non-isomorphic labeled systems with identical allowed interventional trajectory laws when an automorphism, code collision, unprobed parent relation, or insufficient target metadata is present.
8. **Prior-art dominance check.** Show exactly which assumption or guarantee is stronger than existing unknown-intervention CRL, VCD, DECAF, and WM3C. A pure-permutation result with known targets is unlikely to suffice.

Until this contract produces a nontrivial identifiability or finite-sample result, it is a boundary-analysis problem, not authorization to implement an architecture.

## 9. Conditional note on T52

I have not audited T52 and make no claim about its proof. A finite-class hidden-wiring theorem would change this verdict only if it actually:

- removes the supplied graph and cross-environment alignment rather than re-encoding them in target metadata;
- proves recovery up to the correct automorphism class;
- uses the same intervention oracle and observation bandwidth for the candidate and lower-bound control;
- states finite-sample dependence on separation, noise, indegree, and library complexity; and
- supplies an architecture-specific computational or learned-interface edge beyond exact finite-class enumeration and existing causal discovery.

If T52 proves only that a finite candidate class can be exhaustively distinguished by sufficiently separating interventions, it strengthens the control boundary but does not turn T51 into an architecture candidate. A genuine matched hidden-interface theorem could reopen candidate admission, but only on the strength of that new T52 result, not retroactively from T51's \(2^r\) contrast.

## Exact paper fixes

1. Change the status to **transparent-interface control theorem; architecture admission pending raw-interface theorem**.
2. Replace every generic \(O(\log M)\) statement by \(s^*(\mathcal M)\) or \(D^*(\mathcal M)\); name the balanced-code assumption when specializing to \(O(\log M)\).
3. Add integer rounding, tie handling, and within-bit repeated-noise independence to T51.2.
4. Label T51.3 an unrestricted-class/interface lower bound, not an architecture separation.
5. Normalize both sides of any comparison to the same oracle, output bandwidth, side information, and total scalar observations.
6. Make the description claim conditional and charge the shared library, anchors, graph/interface grammar, encoder, router, and meta-training cost.
7. Replace informal no-go prose by explicit indistinguishable-model constructions and a quotient equivalence class.
8. State that mechanism-library acquisition is unproved.
9. Add VCD, WM3C, Dreamweaver, and modern unknown-intervention CRL to mandatory controls.
10. Formalize the hidden-wiring object using the repaired contract above before any implementation.

## Final disposition

The mathematics supports a clean and useful statement: **given a supplied modular decomposition and a separating behavioral code, mechanism assignment is a noisy decoding problem whose cost is governed by the actual separating complexity.** It also supports a no-free-lunch contrast with unrestricted black-box functions.

It does not yet support “architecture candidate.” The intelligence-relevant problem is discovering and aligning the decomposition under matched information and cost. T51 leaves that problem open, while current literature already provides substantial theory and systems in adjacent—and sometimes stronger—settings.
