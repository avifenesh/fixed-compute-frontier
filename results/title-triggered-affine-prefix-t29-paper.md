# Title-triggered affine prefix operators T29 — paper gate

Status: **EXACT STAGE 0 PASS; ONLY A FROZEN UNQUANTIZED INFORMATION ORACLE IS ADMITTED**  
Date: 2026-07-31

## P0. One-sentence edge

At unchanged served checkpoint bytes, recurrent-state shape, per-token update
shape, prompt length, and operator count, compile each raw document into the
same affine recurrence transition that an input-driven memory layer would have
performed online, then substitute that transition when the document title is
seen so held-out facts behave like an internal prefix without serving the
document tokens.

The different currency is a training-time scan of each raw document plus a
static in-checkpoint table.  It is charged.  No teacher, parser, generated
answer, external retrieval state, extra prompt token, or stronger compiler is
used.

## Why this is a real reset

T21--T24 asked a generic Transformer to invent the meaning of an injected
document vector.  Even an optimized continuous record did not become a useful
held-out read interface.

T29 does not invent a record convention.  The record **is an executable prefix
transition of the reader's own recurrence**.  Its meaning is operational and
fixed before document-specific fitting:

> applying the compiled pair must produce the exact same memory state as
> scanning the represented document tokens through that memory layer.

This directly answers the compiler objection.  The compiler is not better at
language than the served model; it caches an associative fold that the same
model already computes.  A weaker compiler cannot silently change the state,
because transition equivalence is independently testable.

## P1. Baseline witness

Let two documents `D0` and `D1` differ in a fact, and let the served query be
identical except for an exactly routed title.  A normal recurrent model starts
from the same zero request state and receives only the short query.  Unless the
fact was absorbed into shared gradient-trained weights, the transient states
produced while reading `D0` and `D1` are gone.

The compiled candidate receives a different, raw-derived transition at the
title token.  On the declared memory layer it recreates the state that would
exist after a virtual insertion of that document.  Thus the mechanism creates
a causal distinction even for a writer-held-out document whose prose never
updated shared model weights.

This is an acquisition/organization edge, not a larger finite-precision
function class.  A matched recurrent RAM with the same table and transitions
can execute the same program.

## P2. Typed operator

Reserve an `m`-coordinate input-driven affine memory layer

\[
h_t=a(x_t)\odot h_{t-1}+b(x_t),
\]

where `a,b : token -> R^m` depend only on the current token representation
supplied to this dedicated layer, not on `h`, another recurrent state, or
document-specific latent context.  The restriction is essential: it makes a
document a fixed affine map for every incoming state.

For token sequence `X=(x_1,...,x_n)`, define

\[
\Phi(X)=(A_X,B_X)
\]

such that scanning `X` from any state `h` gives

\[
T_X(h)=A_X\odot h+B_X.
\]

The blocks are:

```text
scan(raw document body)       -> exact affine pair (A_D, B_D)
compose(title, document pair) -> one title-trigger pair (A'_D, B'_D)
quantize(pair)                -> 220 four-bit cells
route(title tokens)           -> the addressed pair or identity
recur(query token)            -> ordinary fixed-shape state update
read(final memory state)      -> ordinary model hidden computation
```

Every intermediate has a direct meaning.  There is no learned per-document
coordinate system.

## P3. Exact composition theorem

For affine pairs define

\[
(A_2,B_2)\circ(A_1,B_1)
=
(A_2\odot A_1,\ A_2\odot B_1+B_2).
\]

### Associativity

Both bracketings of three pairs yield

\[
(A_3\odot A_2\odot A_1,
 A_3\odot A_2\odot B_1+A_3\odot B_2+B_3).
\]

Therefore composition is associative, with identity `(1,0)`.  Raw document
tokens can be folded sequentially or by a parallel scan without changing the
pair.

### Prefix-substitution identity

Let `P`, `D`, and `Q` be arbitrary token sequences.  Then for every initial
state `h0`,

\[
T_Q(T_D(T_P(h_0)))=T_{P D Q}(h_0).
\]

If a query contains title tokens `t_1...t_k`, precompose the final title-token
pair with the document-body pair:

\[
(A'_D,B'_D)=\Phi(D)\circ\Phi(t_k).
\]

Processing `t_k` once with `(A'_D,B'_D)` is exactly equivalent, for this
memory layer, to processing `t_k` followed by every token of `D`.  Multiple
title triggers compose in query order and equal the correspondingly expanded
virtual sequence.

This is the central positive proof.  It is stronger than reconstruction from a
latent vector: it holds for every incoming state and every later suffix.

### Quantization error bound

Let the stored pair satisfy coordinatewise

\[
\|\widehat A-A\|_\infty\le\epsilon_A,
\qquad
\|\widehat B-B\|_\infty\le\epsilon_B,
\]

and let the incoming state obey `||h||_infinity <= H`.  Immediately after the
trigger,

\[
\|\widehat h-h\|_\infty
\le \epsilon_A H+\epsilon_B.
\]

If every later memory coefficient satisfies `||a(x)||_infinity <= alpha <= 1`,
then after `q` suffix tokens,

\[
\|\widehat h_q-h_q\|_\infty
\le \alpha^q(\epsilon_A H+\epsilon_B).
\]

A readout with Lipschitz constant `L` changes by at most `L` times this bound.
The Stage-0 numerical gate must measure the actual margins; the inequality does
not declare four-bit states sufficient.

## P4. Exact scope and obstructions

1. **Dedicated input-driven layer only.**  A full stacked Mamba or hybrid does
   not generally admit one fixed affine document operator.  Higher-layer token
   inputs depend nonlinearly on lower-layer incoming states.  T29 claims no
   exact compilation of the whole model.
2. **Diagonal state restriction.**  The compact pair stores `2m` scalars.
   General dense transition matrices require `m^2+m` values and do not fit the
   plane.  Diagonal recurrences may lack binding capacity.
3. **Finite information.**  Four-bit storage supplies at most 880 logical bits
   per document.  Arbitrary facts cannot be compressed below entropy.
4. **Title trigger.**  Exact multi-token routing and trigger placement are
   separate physical blocks.  The theorem starts after the correct pair is
   selected.
5. **Virtual insertion semantics.**  The identity equals a document inserted
   after its title mention in the dedicated memory stream.  It does not equal a
   Transformer prompt prefix or reproduce missing attention KVs.
6. **Usefulness.**  An exact state can preserve syntax, topic, or nothing useful
   for QA.  Natural sufficiency is the one empirical composition hypothesis.
7. **Staleness.**  Compiled pairs must be recomputed after the shared recurrence
   is frozen.  Updating model weights afterward invalidates them.
8. **Interference.**  Reallocating checkpoint entries and recurrent channels can
   hurt ordinary language.  Protected quality remains mandatory.

These boundaries prevent the affine identity from being mislabeled a smarter
model.

## P5. Matched control

All arms use the same input-driven recurrence, state width, title router,
training data, query losses, parameter count, state bytes, and per-token
operations.  Controls are:

1. identity/zero document transitions;
2. shuffled-document transitions;
3. fixed random four-bit transitions;
4. equal-capacity free per-document states;
5. unquantized exact affine summaries as a stronger information oracle;
6. document tokens actually scanned through the same memory layer as an exact
   online reference;
7. ordinary dense and recurrent models receiving at least the same charged
   training work.

The online scan and compiled unquantized pair must agree numerically before any
capability result is interpretable.

## P6. Fixed served-resource ledger

Choose `m=110`.  Storing `A_D` and `B_D` uses exactly `2m=220` radix-16 cells,
or 880 logical bits per document.  For 2,405 documents, reuse the already
audited prospective plane:

| write | existing BF16 entries reassigned |
|---|---:|
| title token codes | 132,800 |
| per-document address/gate keys | 76,960 |
| thresholds | 2,405 |
| up constants | 2,405 |
| affine payloads | 529,100 |
| **total** | **743,670** |

This is 64 entries below the frozen 743,734 cap and 2.033% of the 36,577,152
parameter model used by the existing small-scale harness.

The candidate and controls must already contain the same 110-coordinate memory
state.  The title-trigger pair replaces the normal token pair; the update
remains one elementwise multiply and add of the same shape.  The final stored
pair may precompose the fixed title-token transition offline, avoiding extra
served arithmetic.

The eventual physical gate must still prove that lookup, payload materializing,
and coefficient selection fit the ordinary graph and latency envelope.  A
logical operation count is not a GPU result.

## P7. Breakthrough-size path

The mechanism can create a qualitative acquisition change.  After the shared
model is frozen, a previously unseen document receives one raw forward scan and
one deterministic checkpoint write.  A dense control requires gradient updates
and may interfere with existing knowledge.  At serving, both receive the same
short question and spend the same model/state budget.

If the dedicated recurrent state is naturally sufficient, the candidate can
behave like a model that read the evidence inline while serving none of those
document tokens.  That is potentially a large knowledge and prompt-compute
gain, not a sub-percent polish.

The claim dies if any stronger upper bound fails:

- online full-precision memory state does not give at least 80% held-out QA;
- exact compiled full-precision state differs materially from online scanning;
- four-bit state loses more than the frozen noninferiority margin;
- correct state does not exceed zero and shuffle by at least 15 points;
- protected language or serving resources regress.

## Prior-art boundary

[Mamba](https://arxiv.org/abs/2312.00752) establishes input-selective recurrent
state and associative scan.  [Marconi](https://openreview.net/pdf?id=RUaMUu7vMX)
and [sparse recurrent prefix caching](https://arxiv.org/abs/2605.05219) store
recurrent prefix states in serving systems.  Attention-side work already
demonstrates composable cached representations in
[Models Take Notes](https://arxiv.org/abs/2606.17107) and
[C2KV](https://arxiv.org/abs/2607.17715).  Current work also explores external attention-compatible
[Knowledge Capsules](https://arxiv.org/abs/2604.20487), recurrent retrieval
adapters such as [MaRA](https://arxiv.org/abs/2607.19326), and compact latent
context artifacts.

T29 claims no novelty for affine scan, prefix caching, recurrent memory,
composable cache representations, or weight-resident lookup alone.  The
unverified conjunction is: from-zero
input-driven memory; raw document affine pairs; exact title-trigger
substitution; four-bit pairs written into already-paid model entries; and a
large title-disjoint natural reasoning gain with identical served resources.
An exact collision on this conjunction would narrow or close novelty before a
model run.

## Block microbench contract

| block | explainable invariant | first kill test |
|---|---|---|
| affine fold | pair equals the transition of a raw token sequence for every incoming state | exhaustive finite sequences plus independent sequential and tree-scan implementations |
| composition | pair product is associative and respects sequence order | all triples in a finite codebook; explicit noncommuting counterexample for any generalized transition |
| title substitution | one trigger equals title token followed by document body | random prefixes/suffixes, multiple titles, repeated titles, wrong-title control |
| quantizer | 220 cells survive BF16 and bound state/readout error | exhaustive levels, adversarial `H`, suffix lengths, zero/shuffle/corruption |
| capacity | `m=110` and the full write ledger fit without new state or ops | independent byte/MAC/state accounting |
| information oracle | online unquantized state is naturally sufficient | frozen correct/zero/shuffle/full-document conditions before quantized or physical work |
| learnability | one input-driven recurrence learns useful one-shot document state | from-zero held-out documents and unseen queries; free-state and dense controls |
| physical plane | addressed pair replaces normal coefficients in the same graph | target H100 p50/p95, HBM/workspace, numerical replay, protected quality |

## Paper decision

T29 passed its frozen
[Stage-0 exact algebra and finite-precision gate](title-triggered-affine-prefix-t29-stage0-decision.md).
It has a constructive identity, an explicit 880-bit/unchanged-state ledger,
and one named capability hypothesis.  Stage 0 did not test that the dedicated
memory is naturally useful: the worst four-bit state error was `0.296875`, so
semantic noninferiority cannot be inferred from the bound.  Only a separately
reasoned and frozen unquantized information oracle is admitted next; quantized,
learned-routing, physical-kernel, and production claims remain unadmitted.

The direct refinement is the
[T30 dihedral-monomial operator](dihedral-monomial-prefix-t30-paper.md).  It
retains T29's executable document-offset semantics while addressing the proved
absence of cross-coordinate routing in diagonal homogeneous transitions.
