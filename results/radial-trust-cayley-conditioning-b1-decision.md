# Radial-trust Cayley B1 conditioning decision

Status: **PASS; AUTHORIZE B2 PAPER ONLY**  
Date: 2026-08-01  
Execution device: CPU only; CUDA hidden

## Decision

The exact radial correction, initialization, and width-384 depth-nine tree
survive B1.  On the frozen Gaussian and Rademacher RMS-one input families at
`lambda=1`, no natural path in any of five seeds crossed either the declared
amplitude-response threshold `r_amp=6.8966100058` or the directional-gradient
threshold `r_grad=sqrt(127)`.

The result does **not** establish language learnability.  Rare forced routes
and coordinate-concentrated stress inputs do enter the declared response
region, so “full-rank and bounded” remains an insufficient description.  B1
only shows that precision-anchored saturation is not typical at the exact
initialized ordinary reference scale.

## Integrity

- Result SHA-256:
  `f8874c9935698f5043f232bc81fd8cf784c579ad77dcf1f70455e5102d113dd4`
- Compressed-array SHA-256:
  `0b673dd4e0e58216f1cbc34867269998ac52b3b3aeab6dcd809c4b587808eb29`
- 570,240 edge rows and 63,360 complete path rows.
- 176 exact `D=8` Jacobian cells, 2,816 JVPs, and 2,816 VJPs.
- 28,160 single-application `D=4096` dtype-reference rows.
- Every protocol, algebra, instrumentation, deterministic-reference,
  numerical-stability, natural-path, forced-path, and resource gate passed.
- The independent reviewer reproduced every JSON summary and depth ECDF from
  the NPZ arrays and found the embedded hashes and predicates consistent.

The full receipt and arrays total 43,499,017 bytes.  Post-write wall time was
14.3589 seconds and peak RSS was 1.0131 GiB.  No GPU was visible or rented.

## Ordinary-scale result

At `lambda=1`:

| Domain | `F_amp` | `F_grad` | maximum path `r` |
|---|---:|---:|---:|
| natural Gaussian, worst of five seeds | 0 | 0 | 3.862596 |
| natural Rademacher, worst of five seeds | 0 | 0 | 0.124484 |
| forced Gaussian, worst input/seed | 6/512 = 0.01171875 | 0 | 10.188095 |
| forced Rademacher, worst input/seed | 0 | 0 | 0.137987 |

Across all 40,960 forced paths at `lambda=1`, only seven crossed `r_amp`; none
crossed `r_grad`.  The forced maximum crosses the response threshold but not
the more severe `1/128` radial/tangential gradient-ratio surface.

## Stress boundary

The result is scale- and concentration-sensitive:

- At `lambda=2`, worst natural-Gaussian `F_amp=0.03515625` and
  `F_grad=0.015625`.
- At `lambda=4`, worst natural-Gaussian `F_amp=0.359375`,
  `F_grad=0.21484375`, and maximum `r=81.713339`.
- For one-coordinate (`k=1`) sparse inputs at `lambda=1`, the worst
  `F_amp=0.25` and worst `F_grad=0.15625`; those maxima occur in different
  seeds.
- Dense (`k=384`) Rademacher-like stress remains far below both surfaces.

Thus the candidate has initialization margin on the frozen ordinary families,
not a global conditioning guarantee.  Later language work must measure actual
hidden-state radii and loss-gradient alignment; B1 intentionally did not bundle
those as a second empirical unknown.

## What is now permitted

Permitted next: write and independently audit a target-shape B2
physical-operator preregistration that freezes the incomplete width-4,096 tree,
its exact byte/operation/traffic ledger, kernel critical path, strongest
controls, target GPU stratum, workload cells, and material latency gate.

Not permitted: execute B2, rent/start a GPU, run language training, claim a
speedup, or claim improved knowledge/reasoning.
