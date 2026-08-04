# Independent audit: amortized counterexample prior boundary (T50)

Date: 2026-08-01  
Scope: mathematical audit only; no experiments, code execution, or edits to T50  
Audited artifact: `amortized-counterexample-prior-boundary-t50-paper.md`

## Verdict

**The core mathematics is sound, subject to several assumptions being made explicit. T49 closes completely as an architecture.**

T50 correctly reduces proposal over a fixed legal-test catalog to an optimal guessing problem. Under the stipulated hit/miss oracle, a shared prior can reduce expected discovery rank, and its value can be amortized over deployments. That is a useful boundary theorem. It does not establish a new learned architecture: the strongest lawful control receives the same context and orders the catalog by the exact Bayes rule, attaining the same optimum. A learned proposer is only an approximation or implementation of that control unless it introduces a separately specified function or resource edge.

The paper should retain the result but qualify the theorem statements as noted below. None of the qualifications rescues T49 as an architectural branch.

## 1. Exact no-signal rank theorem

For a unique valid test $W^*$, with $W^*$ uniform on $N$ candidates and independent of the context, every nonrepeating adaptive proposer whose only feedback before success is “not this candidate” induces a permutation of the catalog. The position of a uniformly random target in that permutation is uniform on $\{1,\ldots,N\}$. Therefore

\[
\Pr(R=r)=\frac1N,
\qquad
\mathbb E[R]=\frac{N+1}{2}.
\]

This remains true for randomized proposers after conditioning on their internal randomness and then averaging.

The exact statement needs three explicit conditions:

1. The proposer does not repeat candidates.
2. It is complete—absent an earlier hit, it eventually queries every candidate. If stopping is allowed, the discovery rank can be infinite or undefined.
3. A failure reveals only that the queried candidate is not $W^*$. Structured traces, graded scores, counterexample contents, or correlations among tests create a different decision problem.

Thus this is an exact no-free-lunch result for the paper's deliberately failure-only catalog, not for active learning or counterexample-guided synthesis in general.

## 2. The $v$-valid order statistic

If the valid set is a uniformly sampled $v$-subset of the $N$ candidates, independent of context and proposer randomness, the first valid rank is the minimum of $v$ uniformly placed order statistics. Its exact mean is

\[
\mathbb E[R_{(1)}]=\frac{N+1}{v+1},
\qquad 1\le v\le N.
\]

The formula is correct. It does not automatically extend to a merely exchangeable-looking or renamed valid set: different orbits of valid subsets can have different laws. It also ceases to be the operative result when failures reveal structure or query costs differ.

## 3. Bayes ordering and unequal costs

For posterior probabilities

\[
p_w=\Pr(W^*=w\mid C=c),
\]

unit-cost expected rank is minimized by sorting candidates in nonincreasing $p_w$. The exact optimum is

\[
L^*(c)=\sum_{i=1}^{N} i\,p_{(i)}(c),
\qquad
L^*=\mathbb E_C[L^*(C)].
\]

For positive candidate costs $c_w$, the expected cost of an ordering $\sigma$ is

\[
S_p(\sigma)=\sum_{r=1}^{N}p_{\sigma(r)}
  \sum_{j=1}^{r}c_{\sigma(j)}.
\]

A two-item interchange gives the exact Bayes ordering: candidate $i$ precedes $j$ iff

\[
\frac{p_i}{c_i}\ge\frac{p_j}{c_j}.
\]

This part is correct. It assumes a single target, deterministic positive costs known before ranking, and failure-only observations. Ties may be resolved arbitrarily.

## 4. The $L_1$ excess-cost bound

Let $\sigma_q$ rank candidates by $q_w/c_w$, while $\sigma_p$ is the exact $p_w/c_w$ ordering. The paper's bound

\[
S_p(\sigma_q)-S_p(\sigma_p)
\le C_\Sigma\,\lVert p-q\rVert_1,
\qquad
C_\Sigma=\sum_w c_w,
\]

is valid, though loose.

For an inverted pair in which Bayes prefers $j$ but the $q$-ordering places $i$ first, its excess is

\[
D_{ij}=p_jc_i-p_ic_j\ge0.
\]

The $q$-inversion implies $q_jc_i\le q_ic_j$, so

\[
D_{ij}
\le c_i|p_j-q_j|+c_j|p_i-q_i|.
\]

Summing over inverted pairs charges each probability error by at most the total catalog cost, yielding the stated bound. The theorem requires $p$ and $q$ to be normalized distributions over the same unique-target catalog. It is a robustness guarantee, not evidence that learning $q$ provides an advantage over an exact or otherwise stronger posterior-ranking control.

## 5. Entropy and mutual-information rank bound

For fixed context $c$, optimal ranking is a bijective relabeling of $W^*$, so

\[
H(R\mid C=c)=H(W^*\mid C=c).
\]

The maximum-entropy distribution on positive integers at fixed mean is geometric, which implies the loose but valid bound

\[
H(R\mid C=c)\le \log_2(e\,L^*(c)),
\qquad
L^*(c)\ge \frac{2^{H(W^*\mid C=c)}}{e}.
\]

Averaging over context and applying Jensen gives

\[
L^*\ge \frac{2^{H(W^*\mid C)}}{e}.
\]

When the marginal target is uniform, $H(W^*)=\log_2 N$, hence

\[
L^*\ge \frac{N}{e\,2^{I(W^*;C)}}.
\]

The derivation is correct. It is a one-sided converse and is not generally achievable. More importantly, it applies to information already present in $C$. If an adaptive experiment produces structured observations, the relevant object is a decision tree with a transcript and acquisition costs; one cannot obtain a full active-search theorem merely by replacing $C$ with the final transcript after the fact.

## 6. Random-renaming assumptions

The renaming result is correct only under a stronger operational statement than informal “invariance.” The theorem should require all of the following:

- A uniformly sampled group element acts transitively on the full candidate orbit.
- The realized renaming is independent of the canonical target and of all other randomness used by the proposer.
- Context is conditionally independent of the realized label-to-role assignment—not merely encoded with an invariant neural representation.
- Candidate descriptions, legality, costs, grammar, and all side observations are renamed equivariantly. Any unrenamed feature can leak the target's role.
- In the multiple-valid-target case, the group action induces the claimed law on valid subsets. Transitivity on individual candidates does not imply a uniform law over all $v$-subsets.

Under those conditions, the renamed target is uniform conditional on context and the no-signal theorem follows. If relational or equivariant context reveals the alignment, that information must be counted in $C$, not treated as a free consequence of architecture.

## 7. Amortization inequality

Within the paper's scalar, equal-test-cost model,

\[
C_{\mathrm{learn}}
=C_{\mathrm{meta}}+K(c_{\mathrm{prop}}+c_{\mathrm{test}}L^*),
\qquad
C_{\mathrm{blind}}=Kc_{\mathrm{test}}B,
\qquad
B=\frac{N+1}{2},
\]

and the rearrangements

\[
Kc_{\mathrm{test}}(B-L^*)>C_{\mathrm{meta}}+Kc_{\mathrm{prop}}
\]

and, when the denominator is positive,

\[
K>
\frac{C_{\mathrm{meta}}}
{c_{\mathrm{test}}(B-L^*)-c_{\mathrm{prop}}}
\]

are algebraically correct.

The inequality is not yet an architecture comparison. It measures the value of an informative shared prior against blind search. The strongest control is the exact Bayes $p/c$ ranker supplied with the same context, legal catalog, costs, and meta-training information. That control already attains $L^*$. Against it, a learned T49 proposer has no rank advantage; if it uses an approximate $q$, its actual term is $S_p(\sigma_q)$, not $L^*$, and the excess-cost bound is a penalty.

A valid amortized resource claim would additionally need to charge context acquisition, posterior computation, persistent model state, training data and verifier/search-generation costs, deployment drift, and any online inference. Heterogeneous resources also need an explicit scalarization or a vector Pareto comparison; time, bytes, energy, and oracle calls cannot be silently added as if they shared a natural unit.

## 8. Does T49 close?

**Yes—T49 closes completely as an architecture.**

The surviving mathematical objects are already identifiable controls:

- Exact state birth from a distinguishing test is the classical table/partition-refinement or CEGAR-style update.
- Proposal over a frozen catalog is Bayesian optimal guessing or search under a shared prior.
- A learned ranking is an approximation to that Bayes control, not a new computational primitive.
- Random-renaming symmetry removes any apparent advantage unless legal information about role survives and is explicitly charged.
- Amortization explains when reusable prior information pays for itself, but it does not distinguish the learned proposer from the strongest prior-aware control.

Reopening T49 would require a separately defined function edge—such as generating a test outside the frozen catalog—or a measured resource edge in computing the Bayes action. Either would be a new branch with new controls, not a repair of the closed T49 architecture.

## 9. Exactly one next, different mathematical object

### Compositional causal mechanism recombination

The next object should be a **sample-complexity separation for learning and recombining reusable causal mechanisms across novel compositions**, not a richer counterexample-ranking heuristic.

Let each environment $e$ be assembled from $r$ mechanisms drawn from a library $\mathcal M$:

\[
P_e(x'\mid x,\operatorname{do}(a))
=\prod_{j=1}^{r}
P_{\phi_{e,j}}
(x'_j\mid \operatorname{pa}_{e,j},a_j),
\qquad
\phi_{e,j}\in\mathcal M.
\]

The research target is a theorem with four parts:

1. **Identifiability:** state explicit interventions, anchors, sparsity, or interface assumptions under which variables and reusable mechanisms are identifiable up to harmless permutations.
2. **Constructive learner:** give an algorithm that learns the mechanism library and composes it in a previously unseen graph or interface arrangement.
3. **Strict separation:** after meta-learning, prove a new-composition adaptation bound scaling with the number of active mechanisms and their identifiers—for example $O(r\log|\mathcal M|)$ under a concrete model—while an atomic-environment learner requires asymptotically more samples, bits, or interventions under the same distribution and resource ledger.
4. **No-go boundary:** prove that without the identifiability assumptions, modular reuse cannot be guaranteed; include the strongest modular causal and meta-learning controls.

This object targets abstraction, intervention, recombination, and out-of-distribution world-model transfer—capabilities much closer to substantial intelligence than optimizing which member of a fixed catalog to try next. It is also meaningfully different from T49: the unknown is a reusable factorization and its composition law, not the rank of a hidden valid test.

The novelty burden is real. Current work already studies compositional causal estimation and reusable causal representations, while the emerging “mechanistic world models” program explicitly advocates modular causal mechanisms. The contribution must therefore be the precise identifiability-plus-separation theorem and matched empirical protocol, not the broad modularity premise itself: [Compositional Models for Estimating Causal Effects](https://proceedings.mlr.press/v275/pruthi25a.html), [Towards Reusability and Compositionality of Causal Representations](https://proceedings.mlr.press/v236/talon24a.html), and [Mechanistic World Models](https://arxiv.org/abs/2607.12474).

## Final disposition

- Exact ranking theorems: **hold with explicit completeness and feedback assumptions**.
- $v$-valid order statistic: **holds for a uniform independent $v$-subset**.
- Bayes $p$ and $p/c$ orderings: **hold**.
- $L_1$ excess-cost bound: **holds, loose, and penalizes approximation**.
- Entropy/mutual-information lower bound: **holds as a one-sided guessing converse**.
- Random-renaming theorem: **holds only with full operational symmetry and no side-channel leakage**.
- Amortization inequality: **algebraically holds, but compares prior information with blindness rather than a learned architecture with the strongest control**.
- T49 architecture: **closed**.
- Next object: **compositional causal mechanism recombination with an identifiability and sample-complexity separation theorem**.
