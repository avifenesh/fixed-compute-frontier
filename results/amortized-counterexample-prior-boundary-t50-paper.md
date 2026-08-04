# T50 amortized counterexample search — the prior is the resource

Date: 2026-08-01  
Status: **EXACT BOUNDARY; LEARNED RANKING ALONE IS NOT AN ARCHITECTURE; NO EXPERIMENT**

## 0. Question inherited from T49

T49 leaves one learned component: propose a legal experiment that separates two
histories currently represented by the same abstract state. Before building
that component, ask whether meta-learning can make counterexample discovery
strictly cheaper than exact blind search.

The answer is conditional:

> Meta-learning can amortize a reusable prior over useful experiments. It
> cannot improve average search on an exchangeable family whose task context
> and failed probes contain no information about the valid experiment.

This document derives that statement exactly and charges the prior's
acquisition cost.

## 1. Minimal search world

Let

\[
\mathcal W=\{1,\ldots,N\}
\]

be a frozen finite library of legal tests. A task contains one canonical valid
distinguishing test `W*`. The test oracle returns

\[
Y(w)=\mathbf 1\{w=W^*\}.
\]

Thus every failed test says only that the queried item is not `W*`. This is a
deliberately hard but legal subfamily. Rich failed traces are considered later.

The learner may observe task context `C`, choose tests adaptively, and stop at
the first success. An admitted proposer is nonrepeating and complete: absent a
hit, it eventually queries every candidate. Repeating a failed test is weakly
dominated, so such a learner induces a possibly context-dependent ordering of
`W`. Let `R` be the rank at which `W*` is tested. Incompleteness would make the
discovery rank infinite or undefined on some tasks.

## 2. T50.1 — exact no-signal theorem

Assume `W*` is uniform on `W` and independent of `C`:

\[
P(W^*=w\mid C=c)=1/N.
\]

Then every nonrepeating adaptive proposer has

\[
P(R=r)=1/N,\qquad r=1,\ldots,N,
\]

and hence

\[
\boxed{\mathbb E[R]=(N+1)/2.}
\]

**Proof.** Before any query, the target is uniform over all `N` tests. After
`r-1` failures, the failure-only oracle merely removes those tests, leaving the
target uniform over the `N-r+1` unqueried tests. Equivalently, conditional on
any ordering selected independently of `W*`, the target occupies every rank
uniformly. Randomizing or adapting the ordering to prior failures does not
change this symmetry. QED.

This is a no-free-lunch result for the relevant subfamily. A meta-trained neural
proposer, hand-coded heuristic, larger language model, and random permutation
all tie unless something observable breaks exchangeability.

### More than one valid test

If the valid set is a uniformly sampled `v`-subset of `W`, is independent of
`C`, and a test returns only hit/miss, the first valid rank in any ordering has

\[
\boxed{\mathbb E[R_{\min}]=(N+1)/(v+1).}
\]

This is the standard mean of the first order statistic of a uniform subset.
Again, no proposer improves it without task information.

## 3. T50.2 — exact Bayes-optimal informed ranking

Now let the context be informative. For a fixed context `c`, define

\[
p_w(c)=P(W^*=w\mid C=c)
\]

and sort the posterior probabilities as

\[
p_{(1)}(c)\ge \cdots\ge p_{(N)}(c).
\]

Because a failure only deletes the queried item and leaves the relative
posterior ordering of all remaining items unchanged, the Bayes-optimal policy
queries in descending posterior order. Its exact expected search length is

\[
\boxed{
L^*=\mathbb E_C\!\left[\sum_{i=1}^{N} i\,p_{(i)}(C)\right].
}
\]

**Proof.** For any adjacent pair with probabilities `p_i < p_j` queried in
positions `r < r+1`, swapping them changes expected rank by

\[
(r p_j+(r+1)p_i)-(r p_i+(r+1)p_j)=p_i-p_j<0.
\]

Repeated swaps yield descending order. Failure-only observations do not add a
new signal beyond eliminating the queried item, so no adaptive reordering can
improve it. QED.

### Unequal execution costs

Let test `w` cost `c_w>0`. For an ordering `sigma`, the expected cumulative
execution cost through the successful test is

\[
S_p(\sigma)=\sum_{r=1}^{N}p_{\sigma(r)}
\sum_{j=1}^{r}c_{\sigma(j)}.
\]

The Bayes-optimal ordering is decreasing `p_w/c_w`. For a pair `i,j`, placing
`i` before `j` contributes `p_j c_i`, while the reverse contributes `p_i c_j`;
the former is no larger exactly when `p_i/c_i >= p_j/c_j`.

If a learned estimate `q` is ranked by `q_w/c_w`, its conditional excess cost
obeys

\[
\boxed{
0\le S_p(\sigma_q)-S_p(\sigma_p)
\le C_{\Sigma}\lVert p-q\rVert_1,
\qquad C_{\Sigma}=\sum_w c_w.
}
\]

**Proof.** The exact excess is the sum over pairwise inversions of
`p_j c_i-p_i c_j`. An inversion induced by `q` satisfies

\[
p_jc_i-p_ic_j
\le c_i|p_j-q_j|+c_j|p_i-q_i|.
\]

Summing each estimation error over all pairs in which it occurs gives a
coefficient no larger than `C_Sigma`. QED.

Therefore the strongest control is not uniform search. It is exact posterior
cost-aware ranking under the same task prior and context. A neural proposal
mechanism has an acquisition claim only if it estimates that ranking cheaply;
extracting extra context is a different learned edge and cannot be denied to
the control.

## 4. T50.3 — information required for large search compression

Assume the marginal `W*` is uniform and all logarithms in this section are base
two. For a fixed context, the posterior probabilities arranged by the optimal
ranking are also the distribution of `R`. A positive-integer distribution with
mean `mu` has entropy at most that of the matching geometric distribution:

\[
H(R\mid C=c)\le \log_2(e\,\mu_c),
\qquad \mu_c=\mathbb E[R\mid C=c].
\]

Consequently,

\[
\mu_c\ge \frac{2^{H(W^*\mid C=c)}}{e}.
\]

Jensen's inequality then gives

\[
\boxed{
L^*\ge \frac{2^{H(W^*\mid C)}}{e}
=\frac{N}{e\,2^{I(W^*;C)}}.
}
\]

The constant is loose, but the scaling is the point: compressing search by an
exponential factor requires a corresponding number of task-relevant bits in
the context. A proposer cannot manufacture those bits internally.

If failed experiments return structured traces `Z_1,Z_2,...` rather than a
binary miss, the problem becomes an adaptive decision tree rather than the
static ranking theorem above. The potential signal in a purchased trace is
measured by terms such as

\[
I(W^*;Z_t\mid C,Z_{<t}),
\]

which is real information purchased by executing the experiment. Merely
substituting the final transcript for `C` does not prove an active-search bound;
the acquisition policy and all trace costs must be analyzed explicitly.

## 5. T50.4 — random renaming removes lexical transfer

Let a permutation group act transitively on the full candidate orbit. Generate
each task by independently drawing a uniform group element, independent of the
canonical target and proposer randomness. Candidate descriptions, legality,
costs, grammar, and side observations must transform equivariantly, while the
supplied context must be conditionally independent of the realized
label-to-role alignment. Under this full operational symmetry—not merely an
invariant neural encoding—`W*` remains uniform conditional on `C`, and T50.1
applies. For multiple valid tests, transitivity on individual candidates does
not by itself imply a uniform law over valid subsets.

A learned equivariant encoder can recover role alignment only from relational
observations that co-vary with the renaming. Those observations break the
independence assumption and must be counted in `C` or in the interaction
history. Surface-token familiarity alone cannot support a renamed-task gain.

This is why T49 requires raw renamed interfaces: otherwise a proposal model can
win by memorizing test names rather than learning how to construct a
distinguishing experiment.

## 6. T50.5 — amortization ledger

Let

- `C_meta` be all meta-training interactions and compute used to learn the
  proposer;
- `c_prop` be proposal inference cost per deployment task;
- `c_test` be the cost of one real test;
- `K` be the number of deployment tasks; and
- `B=(N+1)/2` be the unique-target blind baseline.

The learned route costs

\[
C_{learn}=C_{meta}+K(c_{prop}+c_{test}L^*),
\]

while blind search costs

\[
C_{blind}=Kc_{test}B.
\]

The learned proposer has a complete-cost advantage only when

\[
\boxed{
Kc_{test}(B-L^*)>C_{meta}+Kc_{prop}.
}
\]

If `c_test(B-L*) <= c_prop`, it never breaks even. Otherwise the minimum reuse
count is

\[
K>\frac{C_{meta}}{c_{test}(B-L^*)-c_{prop}}.
\]

Meta-learning can still be extremely valuable when real experiments are
expensive and task structure repeats. The theorem says exactly what pays for
that value: reusable task information and sufficient deployment reuse.

## 7. What this means for intelligence

The positive possibility is substantial but narrower than “a smarter
proposer.” If contexts reveal a small reusable causal grammar, a model may turn
an exponential raw experiment library into a short posterior-ranked search.
That would improve how quickly it forms a correct world model after deployment.

But learned ranking by itself is already Bayesian search/meta-learning. It is
not a new model architecture, and it does not explain persistent storage,
noninterfering consolidation, or compositional generation of tests outside the
frozen library.

[Counterexample Guided Learning in the Large](https://arxiv.org/abs/2606.11521)
is current evidence that structured counterexamples can substantially improve
LLM induction. [Bayesian Optimal Active Search and Surveying](https://arxiv.org/abs/1206.6406)
is a direct classical control for prior-driven active search. T50's contribution
to this ledger is the exact admission boundary, not a novelty claim over those
fields.

## 8. Decision for T49

The “learned test proposer” does not yet earn a model experiment.

Close it as a standalone mechanism if its gain is fully explained by posterior
ranking over a frozen test catalog. Reopen T49 only if the candidate provides a
separately testable operation not contained by that control, such as generating
novel executable tests compositionally and converting verified failures into
localized, durable structural updates.

The next proof object must be chosen after the independent T49 audit. No CPU,
local GPU, or rented GPU run is admitted by T50.
