# Assumptions and hard walls of modern from-zero LM training

Date: 2026-07-30

## 1. Full-likelihood wall

For an autoregressive model,

```
-log p_theta(x_1:T) = sum_t -log p_theta(x_t | x_<t).
```

Next-token prediction is not observing only a fragment of the document's
likelihood; by the chain rule it is the complete joint log-likelihood.  If an
auxiliary target `a=f(x_1:T)` is deterministically computed from the same
sequence, data processing gives

```
I(theta_star; x, a) = I(theta_star; x)
```

under a generative parameter `theta_star`.  The target creates no new corpus
information.  It may still create a better finite-compute bias, representation,
conditioning, or credit path.

Consequence: “more labels from the same text” is not itself an explanation.
Every candidate must identify why its encoding makes useful structure cheaper
for the fixed model to acquire.

## 2. Infinitesimal descent wall

For differentiable loss,

```
L(theta + delta) = L(theta) + gradient(L)^T delta + O(||delta||^2).
```

Under a fixed Euclidean step norm, the negative gradient maximizes immediate
first-order loss reduction.  A different update can win only by changing at
least one of:

- the metric/trust region;
- finite-step curvature handling;
- stochastic generalization rather than current-batch descent;
- the objective;
- the time horizon, optimizing future learnability rather than immediate loss.

A proposed optimizer that cannot name one of these is a disguised learning-rate
or normalization change.

## 3. Natural-gradient wall

Under an infinitesimal KL constraint on the model distribution, the locally
optimal update is proportional to

```
-F(theta)^(-1) gradient(L),
```

where `F` is the Fisher information.  Muon, SOAP, Shampoo, K-FAC, and related
methods occupy approximations to this geometry at different costs.  A new
geometry method must beat these controls or show that it optimizes a different,
non-myopic quantity.

## 4. Export-capacity wall

Training-only state can improve which solution is found, but after it is
deleted the exported model still has the original finite function class and bit
capacity.  No training method can make a `B`-bit fixed decoder exactly retain
more than `B` independent random bits.  A learning breakthrough must exploit
structure, interference, or poor allocation—not manufacture incompressible
knowledge.

## 5. Teacher-compute wall

A critic, reverse model, generator, verifier, or search procedure may provide
privileged training signal, but it is a paid training subsystem.  Its forward,
backward, state, data, and generation costs belong in the ledger.  If the helper
is a pretrained stronger model, the result is transferred compute rather than
from-zero learning.

## 6. The status-quo assumptions that remain attackable

Modern training still usually assumes:

1. one-step conditional likelihood is the best representation-shaping signal,
   even when long-horizon structure is present;
2. a scalar sum of token losses allocates gradient value appropriately across
   easy, redundant, rare, and structural events;
3. current-batch descent is a good proxy for the endpoint after trillions of
   future tokens;
4. low-order optimizer state retains enough of past gradient geometry;
5. block-local matrix geometry is sufficient and cross-layer functional
   interference can be ignored;
6. random initialization need not use any cheap global statistics of the data;
7. teacher forcing does not materially delay acquisition of latent plans;
8. the same representation should serve immediate token identity and
   long-horizon predictive state without direct pressure to do both;
9. uniformly mixed online exposure is close enough to the best learning
   trajectory;
10. feature acquisition and feature retention need no explicit separation.

## 7. Remaining sources of a large edge

After the walls, only five broad sources survive:

- **representation credit:** reshape already-observed sequence information so
  causal hidden states acquire reusable structure much earlier;
- **non-myopic update credit:** judge an update by its effect on future/disjoint
  data, not only its originating batch;
- **global functional geometry:** approximate cross-layer distribution-space
  curvature more efficiently than current matrix-local methods;
- **interference control:** use training-only state to prevent common updates
  from erasing rare or compositional features;
- **data-dependent birth:** use a small charged prefix of the stream to create a
  materially better initial feature basis.

The first experimental branch will attack representation credit.  It has a
clean fatal test and can be rejected cheaply.  Optimizer geometry will not be
the first branch because the local metric lane already has unusually strong
2026 controls.
