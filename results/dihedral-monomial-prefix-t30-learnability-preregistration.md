# Dihedral-monomial prefix operators T30 — learnability-screen preregistration

Status: **WITHDRAWN BY PRE-RUN BASELINE AUDIT; NEVER EXECUTED**  
Date: 2026-07-31

> This frozen design must not be run or interpreted as evidence.  Before any
> training result existed, the matched-control audit found that the document
> arm fixes every diagonal scale to one and uses only the rotation generator.
> It therefore compares T30 with an additive identity recurrence rather than
> the full T29 diagonal-affine class, and its document task does not require
> noncommutativity.  The exact audit and replacement theorem are recorded in
> [`dihedral-monomial-prefix-t30-learnability-pre-run-audit.md`](dihedral-monomial-prefix-t30-learnability-pre-run-audit.md).

## Question

Can a width-109 dihedral-monomial head learn and use its noncommutative route,
compile held-out documents exactly, and recover ordered relation/value bindings
that a byte-matched diagonal head aliases, with a breakthrough-size rather than
incremental accuracy margin?

This is a synthetic operator-use screen.  It is not a natural-language result.

## Why this task is diagnostic

The screen contains two independently scored blocks.

### Block A: learn the two group generators

Token `R` denotes the unit rotation `rho=(+1,1)` and token `F` denotes the
reflection `tau=(-1,0)`.  Starting from a state with marker value one at
coordinate zero and marker value two at coordinate one, a word over `R,F`
has an exact target state under `D_109`.

The candidate does not receive the routes.  It has hard straight-through route
logits over four choices:

```text
identity, rho, rho^4, tau
```

for each of `R` and `F`.  Training words have random length 1 through 16.
Evaluation contains all 218 group elements in canonical words and a second
version lengthened by inserting exact neutral words `FF` and `R^109`.

#### Constructive solution

Set `route(R)=rho` and `route(F)=tau`.  With scale one and offset zero, the
candidate state is exactly the target group action at every length.  The
Stage-0 theorem guarantees closure and length-independent execution.

A homogeneous diagonal head cannot faithfully implement both generators in
one layer because all its linear actions commute while `rho` and `tau` do not.
The matched diagonal arm retains the same route logits as unused parameters.

### Block B: held-out composed-document binding

There are eight relations and eight values.  Each document assigns values to
relations by a permutation of `0..7`, so **every document contains exactly the
same multiset of value tokens**.  Documents differ only in which value occurs
at each of the eight ordered relation positions.

Each value is represented by one raw value token followed by four `R` tokens.
Identity-route filler tokens are inserted in a fixed template.  A learned
value offset writes into the width-109 state.  Two documents are scanned in
order, then a query supplies one relation index.  The target is

```text
(value_in_first_document - value_in_second_document) mod 8
```

as an eight-class answer.

#### Constructive solution

Set each value offset's first four coordinates to its four-bit code and the
remaining coordinates to zero.  Set `route(R)=rho`.  Four rotations after a
value move its code into the next four-coordinate chunk.  After eight facts,
one document occupies 32 coordinates.  Scanning the second document shifts the
first document by another 32 coordinates and writes the second into the first
32.  All requested chunks lie below coordinate 68, so width 109 is sufficient.
A two-layer query-conditioned decoder can implement the finite eight-value
subtraction table.

For the additive diagonal subcase `a=1`, every document state is the same sum
of the same eight value offsets and fixed filler offsets.  It is independent of
the value permutation.  On uniformly held-out permutations, the answer is
uniform conditional on the relation, so accuracy is at most `1/8=12.5%`.

The trained diagonal control receives the same value offsets, decoder, state
width, token sequences, optimizer, and updates.  It lacks only the permutation
action.  This exact alias is the reason for this microbenchmark; success does
not imply arbitrary prose understanding.

## Frozen data

- total documents: 4,096 distinct permutations sampled without replacement;
- training documents: first 3,072 after seed-locked permutation generation;
- evaluation documents: remaining 1,024, never presented to gradients;
- eight relation positions and eight values per document;
- all documents have byte-identical token-count and token-multiset histograms;
- training document pairs and relations are sampled deterministically per
  update;
- evaluation uses 16,384 deterministic pair/relation cases;
- group training words: batch-sampled `R/F` words, lengths 1 through 16;
- group evaluation: all 218 canonical elements plus neutral-word-lengthened
  equivalents.

Data seed: `30_001`.  Shuffle-control permutation seed: `30_019`, with a
nonzero cyclic offset over evaluation documents.

## Frozen models and arms

State width is 109 in both arms.  Both reserve one uint16 controller word, so
served state accounting is 220 bytes.  Both expose the same trainable tensors:

- two `4`-choice route-logit rows for `R,F`;
- eight width-109 value-offset rows;
- one eight-row relation embedding of width 16;
- one query decoder `125 -> 256 -> 8` with SiLU;
- identical scalar initialization and parameter count.

All recurrence scales are exactly one.  Non-value offsets are exactly zero.
Every token still executes one explicit width-109 multiply and add in both
arms.  The candidate applies the hard selected dihedral permutation before the
multiply; the diagonal arm applies identity and keeps the route logits as
matched unused parameters.

Candidate and diagonal arms start from byte-identical trainable tensors for a
given seed and train independently.  Seeds are:

```text
3109, 3203, 3301
```

## Frozen optimization

- updates: 2,500;
- document batch: 256;
- group-word batch: 256;
- optimizer: AdamW;
- learning rate: `3e-3`, constant;
- betas: `(0.9,0.999)`;
- epsilon: `1e-8`;
- weight decay: zero;
- gradient clipping: global norm 1.0;
- precision: FP32;
- loss: document cross-entropy plus the sum over state coordinates of group
  target squared error, averaged over the group batch;
- hard-route forward with temperature-1 softmax straight-through gradient;
- no warmup, scheduler, early stopping, checkpoint selection, retry, or
  hyperparameter sweep.

## Frozen evaluation conditions

For each seed report:

1. candidate group-state exact accuracy on canonical and lengthened words;
2. diagonal group-state exact accuracy;
3. candidate document accuracy using compiled full-precision operators;
4. candidate online-token accuracy and maximum state/logit difference from the
   compiled condition;
5. candidate zero/identity-document accuracy;
6. candidate shuffled-document accuracy;
7. candidate state-cache accuracy, replacing each `(g,A,B)` by `(e,1,B)`;
8. diagonal online document accuracy;
9. route choices, finite-loss/gradient checks, held-out isolation, parameter
   counts, state bytes, record cells, and operation counts.

Tie-breaking is argmax's lowest class index in every arm.  Accuracy is exact;
no calibration or threshold fitting is allowed.

## Mandatory gates

Every seed must satisfy all gates:

1. candidate group-state exact accuracy at least 95% on both canonical and
   lengthened words;
2. candidate exceeds diagonal group accuracy by at least 70 percentage points;
3. compiled and online candidate states differ by at most `1e-5`, logits by at
   most `1e-5`, and predictions are identical on all 16,384 cases;
4. candidate compiled document accuracy at least 90%;
5. candidate exceeds the diagonal arm by at least 30 absolute points;
6. candidate exceeds each of zero, shuffle, and state-cache controls by at
   least 30 absolute points;
7. candidate route choices are finite and stable under exact hard decoding;
8. training/evaluation documents are disjoint, evaluation documents receive
   zero gradient presentations, all document multisets match, and all losses
   and gradients are finite with zero retries;
9. trainable parameter counts match exactly; both reserve 220 state bytes and
   220 record cells; per-token recurrence arithmetic is 109 multiplies and 109
   additions.

## Kill boundary

Any seed failure closes this exact learnability composition.  Do not change
steps, learning rate, route choices, group words, document order, state width,
decoder, seeds, or thresholds after seeing results.

A pass admits only a separately reasoned unquantized natural-information
oracle.  It does not establish natural semantics, four-bit noninferiority,
physical GPU efficiency, architectural novelty over PD-SSM, or a production
model improvement.
