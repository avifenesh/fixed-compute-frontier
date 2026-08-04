# Qwen3.5 full-attention KV-group localization — preregistration

Status: **calibration rule frozen before the group-localization run**  
Date: 2026-07-23

## Scope

The global algebraic-replacement sweep failed its frozen gate. This follow-up
does not repair that claim and cannot admit an architecture. It asks only
whether the failure is distributed across all full-attention state or
concentrated in a few physical cache-bearing units.

The unit is `(full-attention layer, KV group)`: six layers times two KV groups,
for 12 units. Four query heads share each KV group, so query heads are not
treated as independently removable state.

## Calibration screen

Use the already-opened 2,048-token suite from the global sweep. It is now
explicitly calibration data, not held-out evidence. For each of the 12 units,
intervene on that unit alone through the 64-token continuation while every
other unit remains exact. Test four oracle families:

1. `topk64`: renormalized exact top-64;
2. `tailmean64`: exact top-64 probability contribution plus exact tail mass
   times the omitted values' unweighted mean;
3. `uniform`: zeroth-order value moment;
4. `firstorder`: first-order key/value moment.

All four retain the exact prefix cache and compute full-history scores. The
screen measures frozen-model sensitivity only; it pays none of the causal
selector, recurrent-state, write-path, storage, or kernel costs.

## Frozen localization score

For unit `u` and family `f`, define

\[
s_{u,f}=\max\left(
\frac{K_{natural}}{0.001},
\frac{K_{shuffled}}{0.002},
\frac{K_{structured}}{0.002},
\frac{P^{natural}_{95}}{0.01},
\frac{U^{natural}_{\Delta NLL}}{0.01},
\frac{1-A_{all}}{0.01},
\frac{K_{worst\ case}}{0.01}
\right).
\]

Here `K` is mean full-vocabulary KL, `P95` is token KL p95, `U` is the
reported one-sided delta-NLL bound, and `A` is top-1 agreement. The per-unit
family is the family with the smallest score; ties use the order `uniform`,
`firstorder`, `topk64`, `tailmean64`, favoring the algebra with less oracle
selection machinery.

Rank units by their best score. A resource-screened combination keeps the
three worst units exact and replaces the other nine with their frozen best
families. This is a deliberately optimistic upper bound. The assignment is
written down before that combined intervention is run.

## Decision

- If fewer than nine units individually have `s <= 1`, stop. The model is not
  locally separable enough for the 9-of-12 resource screen.
- If nine or more pass individually, run the frozen 9-of-12 assignment on this
  calibration suite. Individual effects are not assumed additive.
- Only if the simultaneous assignment passes the original global oracle gate
  may it be evaluated on new disjoint natural, code/math, retrieval, and mixed
  state/retrieval panels at 2K and 8K.
- Even a new-data oracle pass would only justify building causal bounded-state
  replacements. It would not establish memory, FLOP, or latency savings.

The exact-hook and destructive controls from the global sweep remain the
harness validity controls. Thresholds and the three-exact-unit limit will not
be relaxed after localization results are visible.
