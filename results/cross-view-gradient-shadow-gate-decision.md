# Cross-view gradient shadow gate — decision

Decision: **close the diagonal cross-view filter; do not run a candidate LM
arm.**

The run was causal, finite, protocol-valid, and used two nonoverlapping training
regions separated by 40,608 sequences and 19,751 inserted EOS document
boundaries. The candidate never modified the model update.

## Result

The cross-view reliability estimate contained real out-of-sample information:

- held-out normalized MSE improved in all seven matrix families;
- aggregate MSE fell by 0.222746% versus the causal scalar;
- the moving-block interval for cross-minus-scalar MSE was
  `[-0.00401333, -0.00007404]`;
- cross-view beat the shuffled estimator in all seven families;
- no filter fallback occurred and maximum norm-preservation error was
  `2.15e-7`.

It was too weak and not directionally dominant:

- the MSE gate required 0.5%, more than twice the observed gain;
- mean held-out pseudo-Adam cosine rose only 0.001318, below the 0.002 gate;
- cross-view beat identity in only four of seven families and its blocked
  interval crossed zero;
- it lost to the best energy-only control in every family. The aggregate
  cross-minus-control cosine was `-0.005110`, with interval
  `[-0.005967, -0.004204]`.

## What survived

The winning control multiplied the Adam direction along each fixed smaller
matrix axis by the square root of lagged raw-gradient energy, then preserved the
whole update norm. It consistently favored rows or columns with persistent
gradient magnitude. Algebraically, this partially restores group-scale
information that elementwise Adam cancels.

That observation is useful as a mandatory optimizer control, not a new frontier
candidate. Current row/column-normalized and rescaled-Adam work already studies
this matrix-geometry family, including 2026 MOGA. A full screen would therefore
measure an adjacent known optimizer, not test the proposed independent-view
mechanism or break the architecture/matmul status quo.

The exact beta-0.95, empirical-Bayes diagonal cross-view filter is closed. A
future cross-view proposal must predict something unavailable from a single
view's lagged energy and pass the same no-update shadow test first.

## Cost boundary

The 256-step shadow run took 114.24 seconds and peaked at 14,559,455,744 allocated
bytes. It used 130,560 FP32 optimizer-statistic scalars plus diagnostic gradient
snapshots. There was no candidate checkpoint or serving change.

Result SHA-256:
`2836f8dc23dd0db643af29ed1c10cd8dfe1032d25155006f95990a7b977c90da`.
