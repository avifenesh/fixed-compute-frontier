# T75 Stage A preregistration — model-free mechanics only

Date: 2026-08-02  
Status: **FROZEN BEFORE EXECUTION; NO NEURAL TRAINING; NO GPU**

## Purpose and claim boundary

This run validates only the T75 finite world, exact posterior, information
floor, transparent adaptive/nonadaptive references, entropy-greedy reference,
and small-instance Bellman recursion. It cannot establish learned composition,
a useful architecture, or a more intelligent model.

The independent design audit is
[`guarded-causal-composition-t75-stage-a-design-audit.md`](guarded-causal-composition-t75-stage-a-design-audit.md).
Its restriction against an exact `m=33` Bayes-optimal/minimax claim is binding.

## Frozen world

- `m=33`, selectors `u=0,...,32`.
- uniform worlds `(t,s)`, `t=1,...,32`, `s in {0,1}`.
- local orientation `o(u)=s xor 1[u>=t]`.
- `epsilon=0.1`.
- one charged probe returns one binary result with
  `P(Y=1|o=0)=0.1`, `P(Y=1|o=1)=0.5`.
- no passive action, metadata, timing, score, cache, or transcript side channel.

## Frozen invariants

The script must fail unless all hold:

1. exactly 64 worlds have distinct 33-selector orientation strings;
2. likelihoods are positive and normalized;
3. a passive likelihood vector of ones leaves the prior unchanged;
4. float log-space posterior agrees with a rational oracle on a fixed history;
5. evidence order does not change the posterior;
6. selector reflection maps `(u,t,s)` to `(32-u,33-t,1-s)` exactly;
7. selectors `0,...,31` identify all worlds;
8. deleting any interior selector has the audited adjacent-threshold collision;
9. using interiors without an endpoint has the audited orientation collision;
10. simulator and filter likelihood tables are equal entry by entry.

## Information floor

Compute the capacity of the binary channel with 80-decimal arithmetic:

\[
C=\max_q h_2(0.1+0.4q)-(1-q)h_2(0.1)-q.
\]

The run must place `q*` inside `[0.462312,0.462314]` and `C` inside
`[0.1475894,0.1475895]` bits. With `M=64` and average error `0.05`, report

\[
B_{Fano}=\frac{6-h_2(0.05)-0.05\log_2 63}{C}.
\]

Gate: `36.68 < B_Fano < 36.70`; hence a fixed integer budget needs at least
37 probes. For variable stopping, report only `E[N]>=B_Fano`.

## Transparent constructive adaptive reference

Use exact equal-prior likelihood-ratio classification between `Bern(0.1)` and
`Bern(0.5)`. Choose orientation `1` iff its binomial likelihood is strictly
larger; ties choose `0`.

Freeze

\[
r_A=\min\{r:6\max(e_{0,r},e_{1,r})\le0.05\}.
\]

The implementation must derive `r_A=28`, not hard-code it, and report the two
class-conditional errors. Classify `s` at `u=0`, then use five balanced
threshold decisions. For current inclusive range `[lo,hi]`, query
`u=floor((lo+hi)/2)`; local orientation different from estimated `s` selects
`[lo,u]`, otherwise `[u+1,hi]`. Every world uses exactly `6*r_A=168` probes.

The exact per-world correctness probability is the product of the six
class-conditional correct-decision probabilities along its true search path.
Gate: analytic average and worst-world accuracy are each at least `0.95`.

## Fixed nonadaptive reference

Probe every selector `u=0,...,31` equally. Derive

\[
r_{NA}=\min\{r:32\max(e_{0,r},e_{1,r})\le0.05\}=40.
\]

Classify each local orientation with the same likelihood-ratio rule. Decode `s`
from `u=0`; decode `t` as the first later selector with the opposite classified
orientation, or `32` if none. It uses exactly `32*r_NA=1280` probes.

Compute exact per-world correctness probabilities from the independent local
classification errors. Gate: analytic average and worst-world accuracy are
each at least `0.95`. This is a frozen exact algorithm, not a globally optimal
noisy nonadaptive allocation.

## Exact-posterior entropy-greedy reference

Maintain the normalized 64-world posterior. At each step choose the selector
maximizing expected one-step entropy reduction. This choice is frozen instead
of one-step 0-1 risk because, at the uniform prior, every selector with both
orientations has the same one-step MAP risk; smallest-index tie-breaking would
spend its first probes only on `s` and is an avoidable objective degeneracy.

Use IEEE float64 and `numpy.argmax`, which selects the smallest first maximum.
Stop when maximum posterior is at least `0.95`, or after `H_max=168`. At the
cap, censored episodes are decoded by MAP and included as failures when wrong.
MAP ties use world order lexicographic in `(t,s)`.

This policy is not claimed Bayes-optimal. Report:

- 8,192 i.i.d. uniform-world episodes for Bayes-average accuracy/cost;
- 128 independently seeded episodes for each of the 64 worlds for exploratory
  worst-world accuracy/cost;
- exact action counts, censored count, mean, median, p95, and max probes; and
- a 95% Clopper-Pearson interval for average accuracy plus Bonferroni-adjusted
  95% per-world intervals.

The constructive reference, not the greedy simulation, carries the analytic
95% validity gate.

Random streams are fixed and distinct:

- average worlds: `75001`;
- average outcomes: `75002`;
- stratified outcomes: `75003`.

## Small exact dynamic-programming calibration

Only `m=3`, horizons `0,...,6`, uniform prior, action cost `lambda=0.01`, and
terminal 0-1 Bayes risk are admitted. Use exact rational posterior states and
the audited Bellman recursion with optional stopping. Report values and number
of cached belief/horizon states. Assert monotonic nonincrease in value as the
horizon grows. This checks the recursion only; it says nothing about tractable
exact planning at `m=33`.

## Decision

Stage A passes only if every invariant, numeric capacity bracket, analytic
repetition identity, analytic average/worst reference gate, and small-DP
monotonicity assertion passes. The entropy-greedy row may be weak without
failing Stage A; it is a measured heuristic.

Regardless of outcome, no neural T75 run, local GPU use, rental, or natural
pilot is admitted. T76 separately decides whether this diagnostic belongs in a
credible path toward a substantially more intelligent model.
