# T80 independent audit: representation-gauge transport

Date: 2026-08-02

Verdict: **REVISE THE STATEMENTS; RETAIN METHOD CLOSURE AND NO-RUN**

T80 is directionally correct. Its exact result is a substitution lemma, its
mean-error bound is valid under named norms and a protected-input distribution,
and its anchor-rank counterexample correctly shows ambient non-identifiability.
The draft nevertheless overcompresses three distinctions: mean reconstruction
error versus classification preservation, identifying a full ambient transport
versus only its task-relevant restriction, and transport-parameter cost versus
the cost of retaining the old continuation. Those points should be corrected,
but they do not reopen the candidate. The June 2026 Transport Keys report
implements the same operator and stitched evaluation and already demonstrates
order-one recovery. Under the charter, another experiment would reproduce an
occupied diagnostic rather than test a new intelligence mechanism.

## 1. Exact transport theorem

Proposition T80.1 is correct on the stated protected set. If

\[
f_0(x)=g_0(h_0(x)),\qquad h_0(x)=T(h_1(x))
\]

for every protected \(x\), then
\(g_0(T(h_1(x)))=f_0(x)\) there. This is exact preservation by
substitution; it does not prove that such a \(T\) exists in the chosen family,
can be identified from finite anchors, or generalizes beyond the protected set.
Those are the substantive learning questions.

The special linear calculation is also correct for column activations:
if \(h_1=G h_0\) and \(G\) is invertible, then
\(W_1=W_0G^{-1}\) gives \(W_1h_1=W_0h_0\). Full ambient
invertibility is stronger than necessary. It is enough that the map from
\(h_0\) to \(h_1\) be injective on the protected activation subspace and have a
left inverse there. Conversely, a bias or translation requires an affine
transport (or homogeneous coordinates); the displayed purely linear formula
does not cover it.

The theorem should therefore be labeled an **exact conditional preservation
lemma**, not evidence that coordinate drift is generally recoverable.

## 2. Approximate bound and classifier margin

Choose norms explicitly and let \(X\) follow a named protected-input
distribution. If \(g_0\) is \(L\)-Lipschitz between those norms and

\[
\mathbb E\|h_0(X)-T(h_1(X))\|^2\leq \epsilon^2,
\]

then the draft's bound

\[
\mathbb E\|f_0(X)-\tilde f(X)\|\leq L\epsilon
\]

follows from Lipschitzness and Cauchy--Schwarz. It is a distributional mean
bound. It supplies neither pointwise preservation nor a worst-case guarantee,
and by itself says nothing about accuracy on low-margin examples or under
distribution shift.

The classifier wording must define “logit perturbation.” Let
\(z_0=g_0(h_0(x))\), \(\tilde z=g_0(T(h_1(x)))\),
\(c=\arg\max_j z_{0j}\), and

\[
m(x)=z_{0c}-\max_{j\ne c}z_{0j}.
\]

A precise sufficient pointwise condition is

\[
\|\tilde z-z_0\|_\infty < \frac{m(x)}{2}.
\]

If “perturbation” instead means the maximum change in any pairwise logit gap,
the threshold is \(m(x)\), not half the margin. The draft should not slide from
the expected representation bound to a claim that every prediction is
protected. A useful distributional consequence, for any \(\tau>0\), is

\[
\Pr[\tilde c(X)\ne c(X)]
\leq \Pr[m(X)\leq 2\tau]
+\Pr[\|\tilde z(X)-z_0(X)\|_\infty\geq\tau].
\]

If \(g_0\) is \(L_\infty\)-Lipschitz into logit \(\ell_\infty\) and the
stated mean-square representation error holds, Markov's inequality bounds the
second term by \(L_\infty^2\epsilon^2/\tau^2\). This exposes the missing
margin-distribution and tail assumptions.

## 3. What the anchor-rank argument does and does not prove

Let \(S=\operatorname{span}(H_1)\). If \(\operatorname{rank}(H_1)<d\),
paired equations \(H_0=TH_1\) identify only \(T\) restricted to \(S\).
Choosing a nonzero \(A\) with \(AH_1=0\) gives the draft's valid witness:
\((T+A)H_1=TH_1\), while \(T+A\) differs from \(T\) somewhere outside
\(S\). Thus unrestricted ambient \(T\in\mathbb R^{d\times d}\) is not
identified.

Four qualifications are required:

1. **Ambient uniqueness is not the operational target.** If every protected
   activation lies in \(S\), anchors already cover the relevant linear span.
   If the protected continuation is linear, only \(W_0T\), not all of \(T\),
   affects logits; changes in directions annihilated by \(W_0\) are irrelevant.
   The correct necessity statement is that anchors or structural assumptions
   must identify the transport modulo downstream invariances on the protected
   activation support.
2. **Anchor count is not rank.** At least \(d\) anchors is necessary for full
   ambient rank but not sufficient. The activation matrix must actually have
   rank \(d\). For an affine map, the augmented activation matrix must have the
   corresponding affine rank.
3. **Full rank is not stable identification.** With noisy or approximate
   pairs, error depends on coverage and the smallest relevant singular value.
   An ill-conditioned full-rank anchor matrix can give an arbitrarily unstable
   estimate. Exact anchor fit still provides no distributional generalization
   without a transport-family and coverage assumption.
4. **Structural constraints change identifiability.** Orthogonal, diagonal,
   sparse, equivariant, low-rank-correction, or known task-subspace families may
   be identified from fewer anchors under family-specific conditions. T80's
   impossibility is for an unrestricted linear ambient map, not all transports.

The draft's phrase “anchors spanning every protected feature direction” is
defensible after these qualifications. It should not be replaced by the
stronger and false claim that every method requires \(d\) independent anchors.

## 4. Complete cost accounting

The asymptotic transport costs need sharper definitions:

- a generic dense \(d\times d\) transport requires \(O(d^2)\) stored scalars
  and \(O(d^2)\) work per application;
- a factorized rank-\(r\) map \(UV^\top\) uses \(O(dr)\) storage and work, but
  is rank deficient when \(r<d\); a full-rank near-identity low-rank drift
  model should instead be written \(I+UV^\top\);
- paired anchor activations cost \(O(nd)\) storage if retained, and producing
  them requires both checkpoints plus two prefix passes; regression/training
  cost and conditioning must be charged; and
- the proposed stitched path retains and executes the **old continuation
  \(g_0\)**. That tail can dominate the transport key. A residual plastic path
  adds its own parameters and serving work, while multiple task-specific tails
  or keys can grow with tasks or protected snapshots.

Accordingly, “dense \(O(d^2)\), low-rank \(O(dr)\)” is locally correct but is
not a complete system-cost claim. Transport Keys itself notes that access to
pre-update components makes its current construction primarily diagnostic.
Repeated composition can amplify approximation error through operator norms
and conditioning, but compounding is not inevitable: an isometric or freshly
re-estimated map can avoid it. State the risk conditionally rather than as an
unqualified consequence.

Likewise, a nonlinear stitcher does not literally “hide a second model” merely
because it is nonlinear. A sufficiently expressive stitcher can relearn the
protected task or absorb substantial downstream computation, so its capacity,
labels, anchors, optimization, and inference work must be charged and controlled.

## 5. Direct prior-art collision

The central method is already occupied at unusually high specificity:

- [Forgetting is Not Erasure: Recovering Latent Knowledge via Transport Keys](https://arxiv.org/abs/2606.02860)
  combines a post-update early network with its pre-update late network, learns
  an interface-alignment key from paired old/new anchor activations, and
  evaluates the keyed stitch. It reports Split CIFAR-100 Task-A accuracy moving
  from 0.392 post-update to 0.721 keyed versus 0.750 pre-update (92% of the lost
  accuracy recovered), as well as substantial recovery under domain shift and
  in a compact ViT. This is T80's proposed operator, evidence pattern, and
  motivating interpretation. It also explicitly limits the current method to
  short task sequences and notes the predecessor-network deployment cost.
- [Exemplar-free Continual Representation Learning via Learnable Drift Compensation](https://arxiv.org/abs/2407.08536)
  learns old-feature-to-new-feature projectors and transports stored prototypes
  through a moving backbone. Its direction and protected object differ from
  T80, so it is strong adjacent occupation rather than the exact collision.
- [Query Drift Compensation](https://proceedings.mlr.press/v330/goswami26a.html)
  projects queries from an updated embedding model into the old indexed
  document space, directly occupying backward compatibility via a learned
  transport in continual retrieval.
- [Revisiting Model Stitching to Compare Neural Representations](https://arxiv.org/abs/2106.07682)
  and the earlier Lenc--Vedaldi stitching framework already establish learned
  interfaces between frozen prefixes and continuations. T80's substitution
  lemma is the algebra behind that construction, not a new theorem.
- [Functional Alignment Can Mislead](https://openreview.net/forum?id=glLqTK9En3)
  shows that successful stitching can occur even between representations using
  different information. [Grounding Functional Similarity by Invariance-Aware
  Model Stitching](https://arxiv.org/abs/2505.20142) sharpens the forward- versus
  backward-compatibility caveat. Therefore successful recovery supports
  compatibility of the stitched computation, but not by itself the stronger
  claim that the change was only a benign coordinate gauge.

The exact collision is Transport Keys. LDC, QDC, and model stitching make the
surrounding design space mature and remove any residual novelty from merely
changing transport direction or application domain.

## 6. Intelligence threshold and closure decision

The user's at-least-20% threshold does not rescue T80. In fact, Transport Keys
already reports gains far larger than 20 percentage points on some old-task
recovery comparisons. Magnitude is therefore plausible but already demonstrated
for the occupied method. It also measures restoration of a previously learned
classifier through a retained predecessor tail, not acquisition of a new
variable, causal model, abstraction, reasoning procedure, or self-correcting
learning rule.

Under the charter, a candidate must contribute a new operator to formation,
error localization, or reliable knowledge-to-behavior control and then produce
a substantial lifetime-intelligence gain at complete cost. T80 supplies a
valuable control for the last channel but neither a novel operator nor a bounded
lifetime solution. A fresh T80 run would answer a question the cited work has
already answered while undercharging the preserved continuation.

Therefore:

1. **Retain no-run for T80 as proposed.** Do not spend CPU or GPU resources on
   another paired-anchor transport-and-stitch experiment for novelty.
2. **Retain transport as a diagnostic/control.** In a future genuinely new
   continual learner, a keyed old-tail stitch can test whether a loss is
   representational erasure or interface/use failure, with the caveats above.
3. **Narrow “method closed.”** Close the presented anchor-fitted interface map
   as a research candidate. Do not claim that all in-place online alignment,
   transport-free gauge-invariant computation, or bounded consolidation is
   closed; those would require a genuinely different operator and their own
   prior-art audit.

## Required corrections to T80

Before treating the record as final:

1. Call T80.1 a conditional preservation lemma and separate existence,
   identification, and out-of-anchor generalization.
2. Specify norms and protected distribution; replace the classifier sentence
   with the pointwise \(\ell_\infty\)-margin condition and a distributional
   margin/tail bound.
3. State anchor impossibility on the ambient unrestricted map, modulo protected
   support and downstream invariances; add conditioning and affine caveats.
4. Distinguish a rank-\(r\) map from an identity-plus-low-rank correction.
5. Charge the retained old continuation, paired-anchor generation, fitting,
   residual path, and task/snapshot growth, not only the transport matrix.
6. Cite stitching's diagnostic failure modes and avoid inferring “pure gauge
   change” from successful functional alignment alone.
7. Preserve **METHOD CLOSED AS NOVEL CANDIDATE; DIAGNOSTIC RETAINED; NO RUN**.
