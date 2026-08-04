# Address-factored transport attention: learned-screen decision

## Decision

The frozen learned screen fails. Do not advance this model to a tiny language
model. Close the exact `1x64 shared + 8x16 protected` allocation under the
unconstrained Q/K training parameterization, frozen seeds, 1,000-step mixture,
and gates.

Keep the AFTA operator alive for one successor screen only: a training-time
per-map Q/K weight-RMS radial constraint that is folded into ordinary dense
Q/K weights before evaluation and serving. Head counts, widths, data, steps,
controls, and thresholds remain unchanged; seeds must be fresh. Failure of that
screen closes AFTA rather than escalating to longer training or another rescue.

## Frozen outcome

The artifact is complete and independently reconstructed: all 255 expected
rows are present and all nine reported gates reproduce exactly.

Passed: integrity, hard shared gain, ordinary protection, overall gain, and no
shortcut. Failed: shared gain, independent protection, causal use, and score
guard.

Descriptively, `hybrid_afta` exceeded `split12` by:

- `+14.68` points on shared-record accuracy;
- `+11.85` points on hard shared-record accuracy;
- `+6.99` points on independent-record accuracy;
- `+38.30` points on ordinary-record accuracy;
- `+19.99` points on the three-task overall mean.

These means do not establish a robust gain. In seeds `3253`, `4969`, and
`7753`, shared accuracy was `82.30%`, `86.72%`, and `85.47%`, and ablating the
wide branch cost `43.98`, `46.19`, and `46.03` points. In seeds `1201` and
`9901`, the wide-branch drops were `-0.52` and `+0.46` points; the candidate
also lost the protected shared and independent comparisons.

The candidate wide/narrow score-RMS ratio ranged from `0.399x` to `2.157x`;
absolute-q99 ratios ranged from `0.387x` to `2.499x`. Dead branches occupy the
low-score end, while active branches can overshoot the frozen guard. The result
therefore supports learnability and causal usefulness in a successful basin,
but not reliable acquisition.

All 150 address-destruction evaluations passed; their maximum accuracy was
`6.91%`. The maximum audited class prior was `7.07%` and the maximum absolute
query/label correlation was `0.0489`. The gains are not explained by the
previous context-majority shortcut.

## One justified successor

For each executed map block `h`, train raw Q/K weights through

`W_eff[h] = W_raw[h] * rho_0[h] / sqrt(mean(W_raw[h]^2) + eps)`,

where `rho_0[h]` is the same denominator measured immediately after loading
the common per-seed initialization. Differentiate through the normalization,
use zero weight decay only for normalized raw Q/K tensors in every arm, and do
not project weights after optimizer steps. This keeps Adam moments in their
native coordinates.

Before formal evaluation, materialize `W_eff` into ordinary Q/K linear weights.
Require numerical export equivalence. The reparameterization adds training
work and 24 nonpersistent scalar references, but no learned parameter. After
folding it adds no serving parameter, byte, QK/PV MAC, cache scalar, attention
map, or token-time operation.

This tests whether a per-map radial constraint prevents the measured whole-map score collapse/runaway without adding
activation normalization at serving. It does not assume the unmeasured Q/K
Gram imbalance required to justify an SVD/Stiefel parameterization.

## Frozen artifacts

- learned source SHA-256: `3c969819e558007090bc5dfb031bf59d9c268ac75ee57425215bb3fae12cb8a0`
- preregistration SHA-256: `fb931993d65e34f4e0e493d73c4df8939683c750f16d9d5bbb7cb7f49bcf2739`
- result SHA-256: `77a3db085a0187133204543a7682807cb86e04322f2badd3c096f30a75874885`
