# Raw-only reader learnability — pre-run theorem and decision

Status: **PAPER FAIL FOR A STANDALONE CLOZE READER; NO IMPLEMENTATION OR GPU RUN**  
Date: 2026-07-31

## Decision in plain language

Do not run the proposed masked-span or inverse-cloze reader as the next
experiment.

Removing text and asking a model to reconstruct it can teach a useful reader,
but it does not by itself explain why the learned notion of relevance matches
an unseen natural question.  That match is an additional assumption.  If the
reader writes only an opaque document vector, the strongest byte-matched
learned-memory control can implement the same function.  If it retains the
complete raw record, then any advantage belongs to the lossless storage and
conditional read path, not to a new cloze algebra.

The project has also already run the nearest empirical version of this idea.
T22a fitted the exposed raw-derived functions strongly, yet transferred only
0.9615 percentage points over the best dense control on title-disjoint natural
questions and regressed protected natural NLL by 5.5362%.  Rephrasing that
experiment as masked-span training would leave the same unidentified transfer
link and would be rescue tuning of a closed lane.

## The proposed block

Let a raw document be `X in [V]^L`.  A raw-only corruption process samples a
mask `M`, exposes

\[
U=(M,X_{\bar M}),\qquad Y=X_M,
\]

and trains a compiler/reader pair

\[
S=C_\phi(X),\qquad
p_\theta(Y\mid U,S)
\]

without questions, answers, support spans, a teacher, or a pretrained parser.
After training, `C` and the reader are frozen.  A natural question `Q` then
asks for answer `A` from the same document state `S`.

This is a well-typed self-supervised objective.  It is not yet a semantic
contract between `(U,Y)` and `(Q,A)`.

## Theorem 1: raw-only proxy transfer has no universal guarantee

Fix any learning algorithm whose complete observations are the raw corpus and
raw-derived corruption pairs.  There exist two downstream question worlds
with the same distribution over every observed training variable but opposite
correct answers for at least one question.  The trained compiler and reader
have the same distribution in both worlds, so they cannot be correct in both.

### Proof

Choose a corpus distribution and corruption process, then train the algorithm.
Because no downstream labels are observed, define world 0 to answer a chosen
question with `0` and world 1 to answer the same question with `1`, leaving the
raw corpus and every corruption pair unchanged.  The algorithm receives
identical observations and therefore emits the same predictor.  That predictor
must fail in at least one world.  QED.

This is not a claim that self-supervision is useless.  It says that the
corruption rule encodes an inductive bias, and its relationship to the future
task must be stated rather than hidden inside the phrase "raw-only."

## Theorem 2: the exact assumption under which reconstruction can transfer

There is a clean positive family.  Let a latent fact `Z in {1,...,K}` generate
two raw views `U,V` and a question/answer pair, with

\[
U\perp V\mid Z,
\qquad
p(u,v,q,a,z)=p(z)p(u\mid z)p(v\mid z)p(q\mid z)p(a\mid q,z).
\]

Let the emission matrix

\[
M_{v,z}=p(V=v\mid Z=z)
\]

have full column rank.  Then the optimal raw proxy prediction

\[
s(U)=p(V\mid U)=M\,p(Z\mid U)
\]

uniquely determines the latent posterior:

\[
p(Z\mid U)=M^+s(U),
\]

where `M^+` is any left inverse.  Given a question, Bayes' rule then gives

\[
p(z\mid Q,U)\propto p(Q\mid z)p(z\mid U),
\]

so `s(U)` together with `Q` is sufficient to form

\[
p(A\mid Q,U)=\sum_z p(A\mid Q,z)p(z\mid Q,U).
\]

### Meaning

Reconstruction transfers when the masked raw view and the future question are
different observations of the same identifiable latent fact.  Full column
rank is the separation condition: two latent facts may not induce identical
distributions over the reconstruction target.

### Failure boundary

The conclusion fails if:

- the raw views collapse two facts that future questions distinguish;
- the future answer depends on information not shared by the proxy views;
- surface or position shortcuts solve reconstruction without identifying the
  latent fact;
- the learned finite representation does not approximate the optimal
  conditional distribution; or
- the natural question language is not connected to the recovered latent
  coordinates.

The frozen titled-prose corpus provides no raw-only certificate that these
assumptions hold for its unseen comparison questions.  Testing that connection
is legitimate, but it would be the principal empirical uncertainty rather than
a proved consequence of cloze training.

## Strongest-control absorption

Suppose the candidate writes an opaque byte-bounded state `S=C(X)` and both
candidate and control receive the same raw-derived examples, reader operator,
state bytes, and training work.  A free learned-summary or conditional-memory
control can choose the same map `C` and the same reader.  Therefore the cloze
objective alone creates no function-class separation.

To escape this absorption, a candidate needs an independently specified
object that the control does not already possess, for example:

- a lossless near-entropy record with exact random access;
- a discrete graph whose edges have a raw-observable meaning and a recovery
  theorem;
- an algebraic code with a proved query/read identity; or
- a physical conditional-access primitive that changes the resident-byte to
  active-work relationship.

The first item is already the direct `uint16`-record control.  It establishes
storage, not autonomous semantic structure.  A future sidecar must add a
declared, testable relation beyond that control.

## Existing empirical counterexample in this project

T22a trained a 36,577,152-parameter model from zero on raw-derived
`document x relation-skeleton -> missing value` reads and exact equality
composition.  Targets were absent from the queries, records were causally
available, and documents had disjoint training/evaluation functions.

The candidate reached exposed-batch equality loss `0.00100`, used all sixteen
record levels, and received record-gradient norm `21.0093`; the path was
trainable.  Nevertheless:

| frozen result | value |
|---|---:|
| held-out unary NLL gain, correct vs zero | 0.6124% |
| held-out equality, candidate correct record | 58.3333% |
| held-out equality, best dense control | 61.1111% |
| natural QA gain over best dense control | +0.9615 points |
| protected natural NLL regression | 5.5362% |

That result demonstrates the exact danger exposed by Theorem 1: fitting a
raw-derived read game does not imply a reusable natural-language coordinate
system.  It closes the local anchor-window quotient and makes a nearby cloze
rerun scientifically weak.

## Prior-art boundary

This training signal is also established rather than unexplored:

- [REALM](https://arxiv.org/abs/2002.08909) trains a latent retriever from
  masked-language-model likelihood over raw text;
- [ORQA](https://arxiv.org/abs/1906.00300) initializes retrieval with the
  inverse cloze task;
- [Contriever](https://arxiv.org/abs/2112.09118) learns unsupervised dense
  retrieval through contrastive raw-text views;
- [provable self-supervised learning](https://arxiv.org/abs/2008.01064)
  obtains transfer guarantees only after declaring statistical connections
  between proxy components and downstream latent variables.

These works show that the proxy can be useful.  They do not provide the active
project's conjunction of an autonomous discrete sidecar, a ten-point natural
gain over the strongest byte/work-matched memory, and unchanged complete
serving cost.

## Breakthrough-size arithmetic

On the current 104-question slice, the best dense reference is 56.7308% and
the full explicit-record information ceiling is 89.4231%.  The available gap
is 32.6923 points.

Reaching the mandatory 80% absolute target requires recovering at least

\[
\frac{80-56.7308}{89.4231-56.7308}=0.7118
\]

or **71.18% of the entire observed oracle gap**, before any protected-quality
penalty.  Merely beating the baseline by ten points would still require 30.59%
of that gap.  The previous raw-function transfer recovered only 0.9615 points.

There is therefore no effect-size argument for spending an H100 run on the
standalone cloze reader.

## Microbench contract retained for a genuine successor

A later candidate may use reconstruction as its training signal only after its
new structural object passes these independent blocks:

1. **Identifiable synthetic family.** Exhaustively enumerate a small latent
   family, verify the emission-rank condition, and prove that the declared
   sidecar recovers the latent posterior or sufficient statistic.
2. **Shortcut destruction.** Randomize positions, surface forms, entity names,
   and document order while preserving the latent relation; the sidecar result
   must remain invariant where declared.
3. **Wrong-world counterexample.** Construct a corpus with identical proxy
   statistics but a different future task and verify the declared failure.
4. **Control absorption test.** Implement the strongest byte/work-matched
   summary, self-index, and conditional-memory operators; reject the candidate
   if any reproduces its claimed object.
5. **Frozen natural coverage.** Freeze the raw-only compiler, then reveal
   support spans only for scoring.  A two-record candidate must clear
   per-record coverage `p >= sqrt(0.8) = 0.8944` before reader and reasoning
   errors.
6. **Reader identity and margin.** Only after coverage, test the exact deployed
   bottleneck with correct, zero, random, shuffled, and wrong-record state.

No target-GPU timing or from-zero composition run is admitted until all six
blocks pass and the remaining uncertainty is only proxy-to-natural transfer.

## Final paper decision

Close **standalone raw-cloze reader training** as the next candidate.  Retain
reconstruction only as a possible learning signal inside a future architecture
whose structural object, control separation, resource currency, and
breakthrough-sized effect path are already established on paper.
