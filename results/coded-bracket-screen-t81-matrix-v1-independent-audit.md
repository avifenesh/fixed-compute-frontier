# T81 coded-bracket screen v1 — independent audit

Date: 2026-08-02

Verdict: **ACCEPT THE NEGATIVE DECISION; STOP DENSE, \(k=64\), AND GPU
WORK.** The saved development result closes the preregistered matrix seam.
There is one real gate-denominator bug, but it excludes the failed
\((d=4,\sigma=0.10)\) cell and therefore makes the reported result more
favorable to the star code. Correcting it strengthens the negative verdict.
No arithmetic, support-scoring, baseline, cost-ledger, or seeding bug found in
this audit can reverse the result.

This is a static audit of:

- [the frozen preregistration](coded-bracket-screen-t81-matrix-preregistration.md);
- [the executed source](../experiments/coded_bracket_screen_t81_matrix.py); and
- [the saved v1 result](coded-bracket-screen-t81-matrix-v1.json).

No experiment, resampling, or model run was performed.

## 1. Contract and data-family checks

The implementation matches the registered development family:

- \(k=32\), hence \(N=32\cdot31/2=496\) upper-triangle pairs;
- degrees \(d\in\{1,2,4\}\), noises
  \(\sigma\in\{0,0.05,0.10\}\), and 64 trials per cell;
- 31 stars of widths \(n_p=31,30,\ldots,1\);
- \(d_p=\min(d,n_p)\) uniformly sampled partners;
- four-dimensional Gaussian directions normalized onto the sphere; and
- bracket norms uniform on \([1,1.5)\), preserving the registered beta-min
  value of 1.

The JSON contains all nine cells, 48 methods per cell
(two pair controls plus 23 settings for each star variant), and 64 trials for
every method. The stored method metrics are finite and the deterministic star
costs are identical across cells for the same \(r\).

## 2. Wilson gate and phase-transition selection

The Wilson implementation is the standard two-sided 95% score interval with
\(z=1.959963984540054\). For 64 trials:

- 61/64 has lower bound 0.871003;
- 62/64 has lower bound 0.893027 and therefore fails;
- 63/64 has lower bound 0.916659 and passes; and
- 64/64 has lower bound 0.943376 and passes.

Thus 63 successes is the first integer count satisfying both the registered
point-estimate threshold of 0.95 and lower-bound threshold of 0.90. The
implementation applies both conditions. Inspection of all saved \(r=2,\ldots,24\)
points confirms that each selected star point is the smallest saved \(r\) that
passes. The \((d=4,\sigma=0.10)\) unnormalized star has no passing point:
its best late counts include 62/64 at \(r=22\), 61/64 at \(r=23\), and 62/64
at \(r=24\).

The independent sign/noise stream at each \(r\) makes the empirical curve
nonmonotone—for example, a point can pass and a larger \(r\) can fail. The
preregistration did not require nested measurement rows, and choosing the first
of 23 separately inspected points is favorable rather than adverse to the
star. There is no multiplicity correction, also as preregistered. Neither
feature explains a false negative.

## 3. Support and coefficient metrics

The support-scoring function compares the complete \(N\)-row Boolean support
arrays, so exact rate is genuinely **global exact row-support recovery per
world**, not per-star or average pair accuracy. Precision, recall, and F1 are
computed per world and then averaged, matching the requested mean pair-level
metrics. NMSE uses the total squared coefficient error divided by total truth
energy.

The \(10^{-12}\) row-norm cutoff is harmless here. SOMP, exhaustive selection,
and adaptive detection all return explicit zero rows plus selected nonzero
rows. Known \(d_p\) means SOMP selects at most the registered row degree.
There is no truth leakage into the correlation scores or least-squares
coefficients; truth is used only for the explicitly allowed sparsity oracle and
evaluation.

The program does not include F1 in its Boolean development-pass expression,
which is a preregistration omission. It has no effect on v1: every selected
unnormalized point has mean F1 at least 0.99949, comfortably above the frozen
0.98 gate, while the ninth cell has no selected point at all.

## 4. Pairwise baseline fairness

The adaptive baseline implements the strong registered control: it permutes
each star, observes one partner at a time, thresholds at beta-min/2 \(=0.5\),
and stops after \(d_p\) detections. With zero noise its expected stopping
position in a width-\(n\), degree-\(d\) star is
\(d(n+1)/(d+1)\). Summing over the 31 stars predicts approximately:

- 263.5 loops for \(d=1\), versus observed means 259.7--263.4;
- 351 loops for \(d=2\), versus 348.0--352.2; and
- 420.4 loops for \(d=4\), versus 417.2--422.9.

These fluctuations are consistent with 64 randomized permutations. Adaptive
pairwise is the selected pair control in every cell and clears the same Wilson
gate in all nine cells, including 63/64 at \((d=2,\sigma=0.10)\). Consequently,
the exhaustive method's additional global-sparsity oracle cannot be causing
the negative comparison. It only makes an already strong control stronger.

The direct control does not average repeats, but one pass already clears the
support criterion everywhere, so registered optional repeats cannot reduce
its transition cost. Comparing the mean stopping cost of the adaptive method
to the deterministic star cost is exactly the frozen protocol.

## 5. Star arithmetic and cost ledgers

For a common request \(r\), the identity-loop count is

\[
L(r)=\sum_{n=1}^{31}\min(r,n).
\]

This reproduces all selected JSON values: \(L(13)=325\),
\(L(14)=343\), \(L(15)=360\), \(L(16)=376\),
\(L(17)=391\), \(L(21)=441\), \(L(23)=460\), and
\(L(24)=468\).

For coded stars \(n>r\), the ledger charges \(1+n\) active components per
loop. For direct fallback \(n\le r\), it charges two components per pair.
Unnormalized quadratic energy is the same \(1+n\); equal-energy signs have
partner squared norm one and therefore cost two energy units including the
anchor. The pair baselines likewise cost two active components and two energy
units per loop. The common factor for the forward/inverse legs is omitted from
all methods and cancels in every ratio. The stored ledger values agree with
these formulas.

The physical result is strictly negative:

- unnormalized star active-component and energy costs are many times the
  adaptive-pair costs;
- the equal-energy star passes in only seven of nine cells; and
- in all seven passing cells its energy is higher, not 20% lower, than the
  selected pair control. Pair/star energy ratios range from 0.757 to 0.967.

The code also implements the registered fixed endpoint-noise model correctly:
unnormalized mixtures receive their signal-amplitude advantage at fixed noise,
while equal-energy mixtures divide coefficients by \(\sqrt{n_p}\) without
reducing endpoint noise. This is favorable to the unnormalized batch-only
claim and correctly exposes the equal-energy SNR penalty.

## 6. Recomputed decision gate

The eight cells with a passing unnormalized star report:

| \(d\) | \(\sigma\) | pair loops | star loops | pair/star |
|---:|---:|---:|---:|---:|
| 1 | 0.00 | 262.797 | 360 | 0.730 |
| 1 | 0.05 | 259.734 | 325 | 0.799 |
| 1 | 0.10 | 263.375 | 360 | 0.732 |
| 2 | 0.00 | 348.016 | 360 | 0.967 |
| 2 | 0.05 | 351.375 | 376 | 0.935 |
| 2 | 0.10 | 352.172 | 343 | 1.027 |
| 4 | 0.00 | 422.859 | 468 | 0.904 |
| 4 | 0.05 | 417.219 | 460 | 0.907 |
| 4 | 0.10 | 419.406 | no pass | failure |

The reported summary filters out null ratios before calculating the gate.
That is a real bug: the preregistration applies the rule separately to every
cell and says that no passing point is a failure. It must not disappear from
the denominator.

Even under the implementation's favorable exclusion:

- median pair/star ratio is 0.905271;
- zero of eight cells reaches 2x; and
- the star loses by more than 20% in three cells.

Counting the failed ninth cell as a required loss gives:

- zero of nine cells at 2x;
- three numeric losses over 20% plus one stronger no-phase failure; and
- a conservative median ratio of 0.903546 if failure is represented as zero.

The passing-cell median already means that the star uses about 10.5% **more**
identity loops than pairwise, rather than gaining 20%. The exact numerical
treatment of the failed cell therefore cannot change the close decision.

There is a second gate-implementation mismatch: the development-pass flag
applies the full retain conditions, whereas staging step 2 says dense
diagnostics should stop specifically when median identity gain is below 20%.
In another result this could stop dense work prematurely. Here the passing-cell
median itself is below 1.0, so both rules independently require stopping.

## 7. Seed independence and integrity gaps

World, exhaustive, adaptive, unnormalized-star, and equal-energy-star streams
use distinct NumPy seed entropy tuples, with method tags 11, 13, 17, and 19
and \(r\) included for star methods. Trial indices are independent within each
cell. Including \(k\) makes a hypothetical \(k=64\) stream disjoint from the
\(k=32\) development stream. No method shares its sensing/noise generator with
another method.

The world tuple includes \(\sigma\), so different noise cells use different
underlying supports and coefficients rather than the same worlds with new
noise. The preregistration is ambiguous on cross-noise pairing. This does not
affect within-cell method fairness, and with 64 worlds per cell it provides no
mechanism that could systematically reverse this result.

Three registered integrity items are not evidenced by the program or JSON:

1. there is no explicit numerical restricted-wedge/star-row equality check;
2. there is no separately recorded small full-rank noiseless identity check;
   small direct-fallback stars incidentally exercise the same code path, but
   are not an asserted integrity test; and
3. the execution did not produce a human-readable decision artifact.

Software versions, base seed, wall time, and platform are recorded. The JSON
does not embed source/preregistration hashes or the exact command, so it cannot
cryptographically prove which source revision produced it. For audit
provenance, the files inspected here currently hash to:

- source:
  f29a5b933f2cece46225a5d29635faa052b60048ee104dd656c4bb605e242722;
- preregistration:
  0b3019f7475de747eb7f094b615aa9afe5c738cf5ec1986adb8c9b5c9d25db88;
  and
- JSON:
  0f560fbf82359a2cb4d2ac6ec834f314219212f6b7d44b3df2b5317c3ab6a335.

These omissions lower provenance quality but do not identify a numerical
mechanism that favors pairwise or invalidates the saved negative result.

## 8. Stop decision and scope

The frozen staging rule says to stop after \(k=32\) when the unnormalized
star's median identity-loop gain is below 20%. It is negative here. The stronger
retain rule also fails every material clause: 0% rather than 75% of cells reach
2x, several cells lose by more than 20%, and one cell never reaches the recovery
criterion. Equal-energy recovery never saves energy.

Therefore:

- **do not run dense wedge or generic dense diagnostics;**
- **do not run the \(k=64\) confirmation;**
- **do not run local or rented GPU work;** and
- **close the registered T81 matrix-screen execution lane.**

SOMP was frozen in advance, so substituting a new decoder after seeing the
result would be a new hypothesis and preregistration, not a repair. The result
does not prove that every deterministic star code or every sparse decoder must
fail, but no inspected bug can reverse the decision for the registered
Rademacher/SOMP method. It also does not reopen T81's already rejected
grounding or broad-capability claims.
