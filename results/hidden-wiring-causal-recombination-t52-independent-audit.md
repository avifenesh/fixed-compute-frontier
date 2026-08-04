# Independent audit: hidden-wiring causal recombination (T52)

Date: 2026-08-01  
Scope: mathematical and architecture audit only; no experiment, implementation, or edit to T52  
Audited artifact: hidden-wiring-causal-recombination-t52-paper.md

## Verdict

**HOLD WITH CORRECTIONS as an exact finite-class control theorem; T52 does not change the rejection of T51 as an architecture candidate.**

T52 materially improves the fairness of T51's mathematical comparison. Both the structured learner and an unrestricted transition learner can now be given the same global full-state intervention and full next-state vector. The theorem correctly proves recovery of a known, bounded-arity finite mechanism class without a supplied graph.

The gain is nevertheless the gain from a known sparse hypothesis class, not from a learned architecture. Exact ERM already attains it. The per-environment permutation is a harmless relabeling in the stated observable-coordinate model, the sample bound is valid but loose in the separation parameter, and the information floor needs to be written for the full \(r\)-output environment rather than only one binary node.

## 1. Model and theorem assumptions

For each observed next-state coordinate \(j\), T52 assumes

\[
y_{t,j}=f_{m_j}(x_{t,P_j})\oplus E_{t,j},
\]

where:

- every current-state coordinate is directly observable and can be set arbitrarily;
- every next-state coordinate is observed after one step;
- the mechanism library is known and executable;
- every mechanism has known arity \(k\);
- \(P_j\) is an ordered tuple of distinct observed coordinates;
- the same local law remains valid in every composition;
- the noise is independent of the intervention and hypothesis; and
- the repeated noises \(E_{t,j}\) are independent over \(t\), with common rate \(\eta<1/2\).

The paper should index the noise by both round and node and explicitly state the independence over rounds required by concentration. Cross-node independence is not needed for the union bound, although it is used by the simple channel-capacity description.

Because the model is a transition from \(x_t\) to \(y_t\), contemporaneous cycles are not a problem: the unrolled time graph is acyclic. Calling the wiring a DAG would require care if the same-time graph is allowed to contain reciprocal edges.

The reference to a typed library is not represented in the hypothesis count. Either all mechanisms share the same binary \(k\)-input type, or the paper should define node-specific legal libraries and parent tuples.

## 2. Hypothesis counting and behavioral quotient

For a common ordered arity \(k\),

\[
\mathcal H=
\{x\mapsto f_m(x_P):m\in[M],P\in(r)_k\}
\]

is a set of functions on the full Boolean cube. Therefore duplicate parameterizations are already collapsed by set equality, and

\[
H=|\mathcal H|\le M(r)_k\le Mr^k
\]

is correct.

The exact count can be much smaller because of:

- duplicate mechanisms;
- symmetric or ignored input ports;
- parent tuples that induce the same Boolean function;
- node types and forbidden parents; and
- environment-wide graph automorphisms.

The paper should distinguish two quotients:

\[
h\equiv h'
\quad\Longleftrightarrow\quad
h(x)=h'(x)\ \text{for every legal }x,
\]

and

\[
h\equiv_Q h'
\quad\Longleftrightarrow\quad
h(x)=h'(x)\quad Q\text{-almost surely}.
\]

Data sampled only from \(Q\) can identify at most the second quotient. T52's assumption that every globally inequivalent pair has \(d_Q(h,h')\ge\alpha>0\) makes the two quotients coincide on the finite candidate class, but this should be stated rather than left implicit.

For the complete environment, the naive labeled class is a subset of \(\mathcal H^r\), with size at most \(H^r\). If there are coupled graph constraints or the target is quotiented by a simultaneous variable permutation, define the actual environment class

\[
\mathcal G\subseteq \mathcal H^r/\!\sim
\]

and count \(|\mathcal G|\). Per-node ERM remains justified only because the stated loss and candidate constraints factor across output coordinates.

## 3. ERM gap and sample bound

Let \(h_j^*\) be the true node hypothesis and let \(h\) disagree with it on \(Q\)-mass \(d\). The population losses are

\[
L(h_j^*)=\eta
\]

and

\[
L(h)
=\eta(1-d)+(1-\eta)d
=\eta+(1-2\eta)d.
\]

Thus every inequivalent candidate has excess loss at least

\[
g=(1-2\eta)\alpha.
\]

This calculation is exact.

For every node-candidate pair, Hoeffding gives

\[
\Pr\left(
|\widehat L-L|\ge g/3
\right)
\le 2e^{-2ng^2/9}.
\]

The union bound over at most \(rH\) pairs yields the stated sufficient condition

\[
n\ge
\frac{9}{2g^2}\ln\frac{2rH}{\delta}.
\]

On the good event, the true empirical loss is at most \(\eta+g/3\), while every inequivalent candidate's empirical loss is at least \(\eta+2g/3\), so every ERM is in the true behavioral class. T52.1 is therefore correct, apart from integer rounding:

\[
n\ge
\left\lceil
\frac{9}{2g^2}\ln\frac{2rH}{\delta}
\right\rceil.
\]

### The bound is not separation-optimal

The \(\alpha^{-2}\) dependence is an artifact of separately concentrating two absolute losses. Compare them directly. For a wrong \(h\), define

\[
Z_t=
\mathbf 1\{h(x_t)\ne y_{t,j}\}
-\mathbf 1\{h_j^*(x_t)\ne y_{t,j}\}.
\]

Then \(Z_t=0\) when the hypotheses agree, \(Z_t\in\{-1,+1\}\) when they disagree,

\[
\mathbb E Z_t=(1-2\eta)d,
\qquad
\mathbb E Z_t^2=d.
\]

A Bernstein or conditional Chernoff argument therefore gives a failure exponent of order

\[
n\,d(1-2\eta)^2,
\]

not \(n\,d^2(1-2\eta)^2\). A sharper finite-class ERM guarantee has the scaling

\[
n=
O\!\left(
\frac{\ln(rH/\delta)}
{\alpha(1-2\eta)^2}
\right).
\]

This is also the natural dependence because even noiseless random interventions need order \(1/\alpha\) samples to hit a disagreement set of mass \(\alpha\). The current T52 bound remains valid, but it should be labeled conservative and should not be used as a tight interaction ledger.

The algorithm does not need to know \(\eta\) to minimize empirical loss; \(\eta\) and \(\alpha\) enter only the guarantee.

## 4. Adaptive information lower bound

For one noiseless binary-output hypothesis, the depth-\(n\) decision-tree argument is correct for exact, zero-error identification:

\[
n\ge\lceil\log_2 H\rceil.
\]

Adaptivity cannot create more than two children from one binary answer. The bound is only a floor; a poorly splittable finite class can require depth \(H-1\), and some classes are not distinguishable under the legal intervention set at all.

The environment-level statement should use the actual class \(\mathcal G\). Each round returns \(r\) output bits, so a depth-\(n\) transcript tree has at most \(2^{rn}\) leaves. Consequently,

\[
n\ge
\left\lceil
\frac{\log_2|\mathcal G|}{r}
\right\rceil.
\]

When \(\mathcal G=\mathcal H^r\), this reduces to \(n\ge\lceil\log_2 H\rceil\), which is the paper's scaling. Writing the full form makes the oracle bandwidth explicit and handles coupled or quotient environment classes correctly.

With independent BSC noise and a uniform prior on \(\mathcal G\), an error-\(\delta\) Fano bound can be written as

\[
nr[1-h_2(\eta)]
\ge
\log_2|\mathcal G|
-h_2(\delta)
-\delta\log_2(|\mathcal G|-1).
\]

Feedback through adaptive interventions does not increase binary symmetric channel capacity. This formalizes the paper's qualitative statement. It still does not imply achievability: the evaluation queries available to the hypothesis class may split candidates very unevenly.

## 5. Oracle and bandwidth fairness

T52 removes the largest oracle mismatch in T51. Its structured learner uses:

\[
x_t\in\{0,1\}^r
\longmapsto
y_t\in\{0,1\}^r.
\]

An unrestricted vector-transition learner can be given exactly the same oracle. To identify an arbitrary map from \(r\) input bits to \(r\) output bits, all \(2^r\) input states are still necessary in the exact worst case. Thus the interaction contrast can now be made under matched query and response bandwidth.

What remains unmatched is the hypothesis information. T52 receives a known mechanism library, known arity, exact observability, and the promise that every node is one library function of \(k\) distinct parents. The unrestricted learner receives none of that structure.

The fair conclusion is:

> A known bounded-arity reusable-mechanism prior changes the interaction complexity of exact transition identification relative to unrestricted vector truth tables.

It is not:

> A causal mechanism-bank architecture learns more efficiently than another architecture.

Every strongest control must receive the same finite class and can run the same ERM or a better structured identification algorithm.

## 6. Per-environment permutation interpretation

No latent permutation variable appears in the likelihood. Each environment simply exposes its own local coordinates \(x_1,\ldots,x_r\), and the algorithm enumerates all local parent tuples. A simultaneous renaming of current and next-state coordinates only permutes the recovered local graph.

Therefore T52 proves **equivariance to arbitrary variable renaming**, not recovery of a canonical cross-environment variable alignment.

Behavioral mechanism identities provide limited cross-environment alignment of laws: two locally named nodes can be recognized as using the same library function. They do not align the variables themselves when mechanisms repeat, graph automorphisms exist, or multiple nodes have identical interventional behavior. The output of T52 is a mechanism-labeled transition graph in each environment's local coordinate system, modulo behavioral symmetries.

The paper should replace phrases suggesting that it “recovers the hidden permutation” with the narrower and correct claim that no shared variable names are needed.

## 7. Naive computation

For each node, evaluating \(H\) candidate losses across \(n\) samples gives naive comparison work

\[
O(nrH)
\le
O(nrM(r)_k)
\le
O(nrMr^k).
\]

This is a valid upper bound if one mechanism evaluation and parent-tuple extraction are treated as constant cost. More exactly, include the cost \(c_f(k)\) of evaluating a \(k\)-input mechanism:

\[
O(nHc_f(k)+nrH),
\]

because candidate predictions can be shared across the \(r\) output-label columns before the losses are accumulated.

The full environment class may have \(H^r\) elements, but the separable observation model avoids enumerating it. That factorization is already an important structured-algorithm advantage.

The stated search cost is not a computational lower bound. Bit-parallel evaluation, influence/group-testing methods, sparse Boolean learning, constraint solving, cached truth tables, symmetry reduction, and adaptive intervention design may all beat the naive loop. A neural amortizer must be compared with the best applicable structured inference method, not only \(O(nrMr^k)\) enumeration.

The resource ledger must also include:

- storage and access cost for the mechanism library;
- generation or optimization of interventions;
- candidate quotienting and type constraints;
- router/encoder inference;
- meta-training and amortization horizon; and
- active compute, not only nominal hypothesis evaluations.

## 8. Does T52 change the T51 architecture verdict?

No.

T52 upgrades the mathematical case from a privileged local-slot decoder to a global-oracle finite-class identification theorem. That is a genuine improvement and a useful mandatory control. It also makes the architecture claim harder: exact non-neural ERM already recovers the hidden wiring under the declared interface.

The remaining privileges are stronger than in relevant current work:

- [General Identifiability and Achievability for Causal Representation Learning](https://proceedings.mlr.press/v238/varici24a.html) studies latent causal variables under general observation transformations and uncoupled interventions.
- [Nonparametric Identifiability from Unknown Interventions](https://proceedings.neurips.cc/paper_files/paper/2023/hash/97fe251c25b6f99a2a23b330a75b11d4-Abstract-Conference.html) addresses latent variables and graph recovery without known intervention targets, up to unavoidable ambiguities.
- [Variational Causal Dynamics](https://arxiv.org/abs/2206.11131) jointly learns latent representations, sparse causal dynamics, and modular adaptation.
- [WM3C](https://proceedings.iclr.cc/paper_files/paper/2025/hash/79d86433c2acd12b6fa98553435d226e-Abstract-Conference.html) learns compositional causal components for unseen environments and provides identification claims.

T52 is cleaner and more exact in its finite Boolean world, but easier in observation and intervention access. It does not establish a new function, resource edge, or architecture selector beyond these lines of work.

Disposition:

- T52 theorem note: **retain after correction**.
- T51 architecture admission: **still rejected**.
- Next architecture experiment: **not authorized by T52**.

## 9. Conceptual audit of one proposed synthesis

### Versioned residual mechanism lattice

Proposed role: online addition, splitting, and refinement of reusable mechanisms without destructive weight merging.

### Is it distinct from the closed shared-weight-delta MoE lane?

It can be, but only under a strict functional contract.

The closed BBCM lane used one mutable shared INT8 weight base plus four routed dense ternary weight deltas under a fixed served-byte and MAC budget. Its learned route matrices remained almost identical, the conditional gain was too small, and it lost to the dense BF16 control. That lane was about quantized parameter allocation and fixed-route specialization.

A genuinely different versioned mechanism object would instead use:

- immutable, addressable executable mechanism versions;
- append-only residual child versions attached to explicit evidence and validity regions;
- versioned dispatch, so later router training cannot silently change old addressing;
- split operations that preserve the parent while creating children for disjoint regimes;
- refinement operations that fork rather than overwrite;
- structural sharing of unchanged mechanism code; and
- exact rollback, replay, and provenance.

Its purpose would be temporal knowledge revision and causal-model continuity, not compressing several routed matrices around one shared weight base. Residuals may be represented by neural modules, programs, tables, or local likelihood corrections; dense additive weight deltas must not define the object.

### “Lattice” must be earned

A version tree or DAG is not automatically a lattice. Let \(u\preceq v\) mean that \(v\) is a refinement of \(u\), inherits its protected behavior, and narrows or extends a declared validity domain. To call the structure a lattice, every compatible pair must have:

- a greatest common coarsening \(u\wedge v\); and
- a least common refinement \(u\vee v\).

Conflicting corrections may have no unique join. Unless meet and join are defined and closed, the honest object is a **versioned residual mechanism DAG or refinement poset**, not a lattice.

### Minimal mathematical contract

Each version should specify

\[
v=(\operatorname{id},\operatorname{parent},D_v,\rho_v,W_v),
\]

where \(D_v\) is its validity region, \(\rho_v\) is an executable functional correction, and \(W_v\) is the evidence or intervention witness that justified it. The child behavior must be defined in function space, for example

\[
f_v(x)=
\begin{cases}
\operatorname{Update}(f_{\operatorname{parent}(v)}(x),\rho_v(x)),
&x\in D_v,\\
f_{\operatorname{parent}(v)}(x),
&x\notin D_v.
\end{cases}
\]

For probabilistic causal mechanisms, the update must preserve normalization and causal semantics; arbitrary weight addition is insufficient.

The paper-worthy theorem target should jointly guarantee:

1. **Exact retention:** every immutable prior version remains bit-identical when addressed by its version identifier.
2. **Routing stability:** old inputs continue to select the old version unless an explicit version-policy change is itself recorded and reversible.
3. **Detection validity:** false add/split/refine probabilities are controlled under a declared change-point or piecewise-stationary mechanism model.
4. **Online performance:** regret or excess predictive loss scales with the number of genuine mechanism changes, not total stream length.
5. **Growth control:** stored versions and active chain depth are bounded in terms of real changes and approximation tolerance.
6. **Recombination:** a version learned in one composition transfers to another under an explicit invariance condition.
7. **No-go boundary:** indistinguishable drift, overlapping incompatible validity regions, or absent intervention coverage prevents safe refinement.

Exact retention alone is trivial under immutable storage. The substantive contribution must be correct version selection, bounded growth, and transfer under change.

### Is the idea already absorbed?

The broad ingredients are already crowded:

- [Dynamic Mixture of Curriculum LoRA Experts](https://proceedings.mlr.press/v267/ge25d.html) allocates LoRA experts dynamically for continual adaptation.
- [Theory on Mixture-of-Experts in Continual Learning](https://proceedings.iclr.cc/paper_files/paper/2025/hash/17a234c91f746d9625a75cf8a8731ee2-Abstract-Conference.html) analyzes specialization and router convergence in continual task arrival.
- [DIMoE-Adapters](https://arxiv.org/abs/2605.07494) evolves a sparse expert pool through expansion and pruning.
- [SAME](https://arxiv.org/abs/2602.01990) explicitly targets router drift and expert overwriting.

Therefore dynamic experts, freezing, expansion, residual adapters, and non-forgetting are not a new synthesis by themselves.

### Conceptual disposition

**It is a real different next paper object, but only as a version-semantics and online-theory object; the generic neural implementation is already absorbed.**

It is distinct from the dead shared-weight-delta MoE lane when:

- versions are immutable functional objects rather than dense route-specific weight deltas;
- the central theorem concerns evidence-backed add/split/refine operations, retention, regret, and storage;
- dispatch is versioned and reversible; and
- no fixed-budget advantage is inferred from parameter sharing.

Admit it only to paper-stage formalization. Do not authorize an architecture run until it yields a nontrivial theorem and survives direct comparison with expandable/frozen-expert continual-learning controls.

## Exact fixes to T52

1. Add integer rounding and explicit round-indexed iid noise assumptions to T52.1.
2. Label the Hoeffding guarantee conservative and add the paired-loss \(O(1/\alpha)\) refinement.
3. Define global behavioral equivalence versus \(Q\)-almost-sure equivalence.
4. Replace the one-node information floor by \(\log_2|\mathcal G|/r\) for the complete environment; give the explicit Fano expression if retaining the noisy claim.
5. State that the theorem is renaming-equivariant and does not recover canonical variable alignment.
6. Make type, self-parent, port-order, and graph constraints explicit in the class count.
7. Keep \(O(nrH)\) as a naive upper bound, not a lower bound or neural opportunity.
8. State the fair gain as structured-prior versus unrestricted-class interaction complexity under a matched global oracle.
9. Preserve the conclusion that exact ERM is the mandatory control and no architecture experiment follows.

## Final disposition

T52 proves a useful finite-class fact: a known bounded-arity mechanism library permits exact recovery of locally wired transition laws from global interventions with logarithmic dependence on the number of candidate functions, subject to behavioral separation.

After sharpening, the natural random-intervention sample dependence is \(1/\alpha\), the complete information floor is \(\log|\mathcal G|/r\), and the per-environment permutation is only a coordinate gauge. The theorem strengthens the classical control boundary but does not reopen T51 as an architecture.

The versioned residual mechanism synthesis is different from the closed BBCM weight-delta lane only when formulated as immutable functional versioning with online correctness and growth theorems. In that form it is a legitimate next paper object; otherwise it collapses into existing dynamic-expert continual learning.
