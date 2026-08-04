# Qwen3.5 attention-algebra separability — preregistration

Status: **gate frozen before the 2K held-out run**  
Date: 2026-07-23

## Question

Can the six full-attention layers of the frozen Qwen3.5-0.8B Base hybrid be
replaced, at a 2,048-token prompt boundary, by one of three smaller algebraic
read families without materially changing the model's future distribution?

The families are:

1. renormalized top-k attention: sparse/tropical selection;
2. exact top-k contribution plus a uniform tail value: sparse selection plus
   one additive tail summary;
3. zeroth- or first-order key/value moments: bounded recurrent aggregation.

This is an oracle upper-bound diagnostic. It still computes every exact score,
uses exact top-k selection, and (for the tail variant) uses exact tail mass. It
cannot establish a latency, FLOP, or memory saving.

## Frozen evaluation

- Model: `/root/models/Qwen3.5-0.8B-Base`, BF16, eager attention, no training.
- Prompt: 2,048 tokens; teacher-forced continuation: 64 tokens.
- Natural panel: eight disjoint WikiText validation windows.
- Distribution-shift control: the same eight windows shuffled independently
  inside 64-token blocks.
- Structured stress panel: seeded non-identical code, random bindings, and
  periodic state text. A positive claim cannot rest on this panel alone.
- Seed: 17.
- Exact numerical control, recent-window control, bottom-score control, and
  zero-full-attention control are executed through the same cache path.

A 128-token/four-continuation-token handcrafted smoke run occurred before this
document solely to verify hook identity, causal masks, cache isolation, and that
the controls perturb logits. Its quality numbers are not used to choose or
change this gate.

## Interventions

- `topk{1,4,16,64,256}`: keep the highest-scoring entries and renormalize.
- `tailmean{1,4,16,64}`: retain the exact selected probability contribution;
  replace the omitted values by their unweighted mean scaled by exact tail mass.
- `uniform`: zeroth-order value moment.
- `firstorder`: first-order Taylor/moment read around isotropic attention.
- `recent64`, `bottom64`, and `zero`: sensitivity controls.

Only a result with `k <= 64` can survive the resource screen. At the prompt
boundary this is at most `64 / 2048 = 3.125%` of value reads after selection.
The selector, index, summary state, and write path remain unpaid future work.

## Noninferiority gate

The gate reuses the earlier frozen-cache no-harm scale instead of selecting a
new threshold after seeing this mechanism.

An intervention passes only if all conditions hold:

1. natural held-out mean full-vocabulary KL is at most `0.001` nat/token;
2. maximum natural-window p95 KL is at most `0.01`;
3. block-shuffled and structured panel mean KL are each at most `0.002`;
4. one-sided 95% upper bound on paired delta NLL is at most `0.01` nat/token
   on every panel;
5. overall top-1 agreement is at least `0.99`, and every panel is at least
   `0.98`;
6. no individual case has mean KL above `0.01`;
7. the exact hook control has mean and p95 KL at most `1e-7`;
8. zeroing full attention produces mean KL of at least `0.05`, and both zero
   and bottom-64 are at least ten times worse than the passing intervention.

## Decision rule

- If `topk64` passes, the full-attention reads are provisionally
  **tropical-compatible**. The next problem is a causal bounded-cost selector.
- If `topk64` fails but `tailmean64` passes, they are provisionally
  **split-compatible**. The next problem is a causal estimator of tail mass and
  a bounded recurrent tail summary.
- If either moment-only variant passes, they are provisionally
  **recurrent-compatible**.
- If only `k=256` passes, or no family passes, there is no fixed-budget
  architecture admission from this experiment.

Passing this diagnostic does **not** admit candidate 004. It only admits the
cheapest causal implementation test for the surviving algebra. Thresholds will
not be relaxed after the held-out result.
