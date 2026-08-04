# Spectral successor training hypothesis

Status: pre-candidate; mathematical and prior-art boundary only  
Date: 2026-07-30

## Problem

Next-token prediction gives a final causal state direct credit for one future
token.  Multi-token prediction extends this to a short fixed horizon with one
target/head per offset.  Future Summary Prediction compresses a long suffix,
but its cheap handcrafted control is order-blind bag-of-words and its adaptive
summary uses a second reverse language model.

We need a target that is simultaneously:

- ordered and multi-horizon;
- computed exactly from raw text without a teacher;
- fixed-width and cheap for every prefix;
- training-only, leaving the deployed model unchanged.

## Construction

Give each token `x` a fixed zero-mean code `phi(x) in R^r`.  For complex points
`z_j = gamma_j exp(i omega_j)` inside the unit disk, define the suffix resolvent

```
S_t(j) = phi(x_(t+1)) + z_j S_(t+1)(j)
       = sum_(k>=1) z_j^(k-1) phi(x_(t+k)).
```

Concatenate real and imaginary parts for `j=1..m`.  One reverse linear scan
computes the target for every prefix.  A training-only linear head predicts
`S_t` from the ordinary final causal hidden state, jointly with next-token
loss.  The code table, head, and target buffers are deleted at export.

This is the truncated `z`-transform of future token features: a spectral
successor representation of the suffix.

## Algebraic properties

### Order separation

For two different finite suffixes, their target difference is a nonzero vector
polynomial `P(z)`.  A nonzero degree-`H-1` polynomial has at most `H-1` roots.
With a continuously sampled code/frequency, two distinct suffixes collide with
probability zero in exact arithmetic.  In particular, permutations that are
identical under bag-of-words are almost surely separated.

Finite width/precision affects margin and recoverability, so empirical collision
and conditioning gates are mandatory; exact-real injectivity is not a capacity
claim.

### Multi-horizon projection

For `m >= H` distinct `z_j`, the Vandermonde system recovers the first `H`
future feature vectors in exact arithmetic.  For `m < H`, the target is a
fixed spectral projection over all horizons rather than an independent head per
horizon.

### Predictive-state optimum

Under squared auxiliary loss, the Bayes-optimal causal prediction is

```
E[S_t | x_<=t].
```

Unpredictable suffix components average away; predictable long-range structure
survives.  Any future reward linear in `phi` can be evaluated from the successor
state by a dot product.  This is the useful transfer property of successor
features, imported from reinforcement learning into causal LM pretraining.

### Complexity

Target construction is `O(T m r)` additions/multiplies in a reverse scan.  The
auxiliary head costs `D * 2mr` weights/MACs per token and one MSE loss.  With
`D=384` and `2mr=64`, that is 24,576 training-only weights: 0.0651% of the
37.75M scratch model and 0.1302% of its 49,152-way unembedding matrix.  No
reverse Transformer is required.

## Hard claim boundary

The broad “predict a future summary” idea is prior art, as are successor
features, random sequence codes, and holographic/vector-symbolic encodings.
The potentially new sliver is their combination as a cheap ordered resolvent
target for from-zero causal LM training.

Relevant boundaries:

- Future Summary Prediction uses bag-of-words or a reverse LM:
  https://arxiv.org/abs/2510.14751
- Multi-token prediction uses offset-specific future heads:
  https://arxiv.org/abs/2404.19737
- Successor features encode discounted future feature occupancy:
  https://arxiv.org/abs/1901.11437
- Random permutations/holographic reduced representations already encode
  sequence order:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4405220/

No novelty or capability claim is admitted until a broader literature audit and
the fatal controls pass.

## Fatal controls before natural-language training

1. Exact recurrence equals direct summation.
2. Same-multiset permutation pairs: bag-of-words must collide and the spectral
   code must separate with a stable finite-precision margin.
3. Random suffixes: compact targets must not be described as lossless or evade
   the finite-bit capacity bound.
4. A from-zero order-sensitive latent-plan task must compare NTP, short-horizon
   MTP, equal-width bag-of-words FSP, spectral successor, and frequency-shuffled
   spectral targets.
5. Candidate compute is charged; its allowed steps are reduced to match the
   strongest control's training FLOPs.
6. The effect must be qualitative or at least 1.25x compute-equivalent at two
   untouched budgets.  Small endpoint polish closes the branch.
7. The same method must not regress an order-insensitive task or a random-label
   memory control.

A pass authorizes a two-seed 37M language screen against NTP, MTP, and FSP-BoW.
It is not a breakthrough until the full definition contract is met.
