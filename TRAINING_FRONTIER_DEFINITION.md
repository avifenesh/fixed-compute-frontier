# From-zero training frontier definition

Status: research contract before mechanism search  
Date: 2026-07-30

## 1. The object being changed

The deployed model is fixed.  A training method is a state transition

```
(theta_(t+1), z_(t+1)) = T(theta_t, z_t, batch_t, randomness_t)
```

where:

- `theta` is the ordinary model state that will be exported;
- `z` is arbitrary training-only state: moments, factors, critics, auxiliary
  heads, memories, curricula, or statistics;
- `T` includes initialization, objective construction, feedback/gradient
  estimation, parameter update, and data selection/order.

At export, `z` is deleted.  Candidate and controls must have the same tokenizer,
inference graph, checkpoint format, parameter count, precision, context state,
and serving implementation.  Better weights are the only allowed deployed
difference.

This isolates a **learning edge**: can a different trajectory place the same
model in a materially better region of its existing function class?

## 2. “From zero”

A valid candidate:

1. starts from the exact same random model initialization as its paired
   controls;
2. starts its mechanism at step zero, or initializes training-only state from
   at most the first 0.1% of the candidate's own charged token stream;
3. uses no pretrained teacher, stronger checkpoint, post-trained policy,
   external verifier, retrieval system, hidden evaluation feedback, or
   synthetic corpus generated before the accounting window;
4. may use labels computable from the same raw training sequence, but all extra
   views, targets, samples, and forward/backward passes are charged.

Warm-starting from a built model, adding a late pretraining phase, distillation,
SFT, DPO, RLHF, and RLVR are separate research objects.  They cannot prove this
claim.

## 3. The current frontier baseline is an envelope

There is no single modern baseline.  A candidate must beat the strongest
applicable point in this envelope under matched accounting:

### Objective/feedback

- causal next-token maximum likelihood with exact reverse-mode gradients;
- multi-token/future prediction when the candidate claims improved long-range
  credit or sample efficiency;
- masked-input auxiliary prediction in repeated/data-constrained regimes;
- the strongest baseline-containing auxiliary objective whose extra training
  work fits the same budget.

Multi-token prediction is not a new direction by itself.  It has reported
large code gains at scale but mixed natural-language results:
[Gloeckle et al.](https://arxiv.org/abs/2404.19737).
Masked-input regularization has reported about `1.3x` unique-data-equivalent
gain in a data-constrained regime:
[Xu et al.](https://arxiv.org/abs/2606.06888).

Reinforcement as a pretraining objective is an important boundary, but the
published RLP method is introduced as a later pretraining phase over an already
trained base; it is not a from-zero control for this contract:
[Hatamizadeh et al.](https://openreview.net/forum?id=9Gp45bnDrJ).

### Update geometry

- tuned AdamW;
- tuned Muon/NorMuon-style matrix updates;
- SOAP or another scalable matrix preconditioner when it fits the target
  scale and memory regime;
- update-RMS-matched learning-rate controls.

Beating AdamW alone is insufficient.  Muon has reported approximately `2x`
compute efficiency over AdamW under compute-optimal training:
[Liu et al.](https://arxiv.org/abs/2502.16982).  NorMuon reports a further
21.74% training-efficiency improvement over Adam on its 1.1B setting:
[Li et al.](https://arxiv.org/abs/2510.05491).  A July 2026 multi-billion-scale
study reports both SOAP and Muon outperforming AdamW and uses update-RMS
matching as a fairness requirement:
[Khona et al.](https://arxiv.org/abs/2607.20548).

### Data exposure

- identical tokenizer and unique raw document pool;
- document-disjoint validation;
- equal raw-token availability and contamination controls;
- a tuned random-shuffle/mixing control;
- repeated-token and fresh-token regimes reported separately.

Corpus improvement, domain reweighting, or adding better data can be valuable,
but it is not the training-algorithm breakthrough sought here.

## 4. Training resource ledger

For every arm, charge from initialization through exported checkpoint:

```
C_train = forward + backward + optimizer + auxiliary + sampling
          + data transformation + communication + recomputation.
```

Report:

- mathematical FLOPs by datatype and actual issued GPU work;
- total GPU-seconds, joules, and cost;
- unique raw tokens, total token presentations, and derived target tokens;
- peak HBM, persistent optimizer bytes, host memory, and communication;
- all training-only parameters/state and the time when they are created;
- failed/retried steps and discarded candidate samples.

Training throughput is not itself model quality.  A faster kernel and a better
learning algorithm are separate claims, even when both reduce time to target.

## 5. “Better model”

At identical deployed resources, quality is a protected vector rather than one
average score:

```
Q = (held-out modeling, reasoning/composition, knowledge acquisition,
     retention/interference, robustness/calibration, context use).
```

The primary curve is quality against cumulative charged training compute, not
quality against optimizer steps or token count alone.

For a protected quality target `q`, define compute-equivalent gain

```
G_C(q) = C_frontier(q) / C_candidate(q).
```

Also report unique-data-equivalent gain and endpoint quality at fixed compute.
For a fitted excess-loss law

```
L(C) - L_infinity = A * C^(-alpha),
```

separate:

- a horizontal shift (`A` falls; fewer resources reach the same endpoint);
- an exponent change (`alpha` rises; the advantage grows with scale);
- a different irreducible floor (`L_infinity` falls in the tested regime).

Only measured interpolation is evidence.  Extrapolated crossover is a
hypothesis.

## 6. Breakthrough threshold

A result is a **breakthrough** only if it satisfies one of these forms and all
protected noninferiority gates:

1. **Large compute-equivalent shift:** at least `1.5x` less total training
   compute to the same protected quality, across at least two model scales,
   three independent seeds, and both fresh-token and repeated/data-constrained
   regimes where applicable.
2. **Scaling-law break:** a statistically supported better exponent or lower
   asymptote across at least three compute budgets and two model scales, with
   the predicted advantage confirmed at the next untouched budget.
3. **Qualitative acquisition change:** the candidate reliably learns a
   preregistered procedural or compositional capability that the frontier
   controls fail to acquire even with `2x` training compute, while general
   modeling and every protected slice remain noninferior.

The method must also be mechanistically distinct from the strongest known
control and survive a current prior-art audit.  A result already explained by
learning-rate scaling, batch size, gradient clipping, weight decay, data order,
or an established optimizer is not novel.

## 7. Admission ladder

The G7 can discover and reject candidates; it cannot by itself prove a
frontier-scale breakthrough.

1. **T0 theorem/fatal control:** identify the information or credit-assignment
   gap and prove the mechanism is not merely scalar step-size rescaling.
2. **T1 exact small systems:** show a large advantage on tasks with known latent
   structure, plus random-label and rotation controls.
3. **T2 37M-class language screen:** two seeds and at least three compute
   checkpoints.  Advance only for a growing advantage or at least `1.25x`
   compute-equivalent signal—not a sub-percent endpoint polish.
4. **T3 adjacent scale:** at least three seeds, stronger optimizer/objective
   envelope, private capability slices, and full resource ledger.
5. **T4 replication:** second model scale and untouched predicted budget.  Only
   here may the formal breakthrough threshold be adjudicated.

## 8. Explicit non-results

These do not satisfy the objective:

- lower loss because the candidate consumed more FLOPs, targets, or generated
  samples;
- a small one-seed NLL improvement;
- beating an undertuned AdamW baseline while losing to Muon/SOAP;
- learning-rate or schedule tuning presented as a new training method;
- a data-quality/corpus-selection gain;
- late-phase training of an already capable checkpoint;
- best-seed selection, public-benchmark feedback, or contamination;
- inference architecture changes, larger checkpoints, retrieval, tools, or
  longer hidden/test-time reasoning;
- a training-only mechanism whose gain disappears after controlling update RMS
  and direction.

The target is therefore precise: **a new from-initialization transition rule
that moves an unchanged deployed model along a materially better quality-vs-
charged-training-resource frontier than the strongest current training
envelope.**
