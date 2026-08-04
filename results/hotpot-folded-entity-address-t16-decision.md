# Hotpot folded entity address T16 — decision

Status: **FAIL exactness; one class-specific-key successor justified**  
Date: 2026-07-31

## Decision

Close the shared rank-64 projection protocol.  Do not sweep its dimension,
views, optimizer, seed, or steps.  Retain the causal evidence that a
raw-prose-only address objective improved natural-question routing from 81.25%
to 96.1538% with zero served operation or model update.

One successor is admitted: train the actual class-specific FFN key rows rather
than factor every key through one shared projection.  This directly targets the
observed near-neighbor errors and maps to physical rows already present in the
served model.

## Evidence

- All eight contracts passed after one pre-result implementation correction
  from inference-only tensors to no-gradient tensors.
- The compiler used 800 finite updates, 0.949 seconds, and 196,612 optimizer
  bytes; the served checkpoint was unchanged.
- The fold identity held within `2.38e-7`.
- Canonical retrieval was 2,405/2,405.
- Training raw-context retrieval was 9,564/9,620 (99.4179%).
- Held raw-context retrieval was 2,249/2,405 (93.5135%).
- Natural-question retrieval was 282/292 (96.5753%) on train surfaces and
  200/208 (96.1538%) on development surfaces.
- Projection artifact SHA-256:
  `70861caae201e145e156bcadbe2330990f7a09ab98b91f912623b9cd2d29b08c`.
- Result artifact SHA-256:
  `f3bb7cc9b2182334e5ec9faa1dbe6b779a38491e7fb06baa1ef0b51db620a917`.

## Error structure

The eight development errors included:

- `Sussex Spaniel -> Welsh Springer Spaniel`;
- `Brooklyn -> List of bus routes in Brooklyn`;
- `Cleveland State University -> Pennsylvania State University`;
- `Jacques Audiard -> Michel Audiard`;
- `Odyssey -> The Sundays`.

Most errors are close lexical or semantic neighbors with small negative
margins.  The shared projection cannot give one class a discriminating
direction without changing every other class in the same 64-dimensional
metric.

## Sole analog successor

T17 removes the shared low-rank factorization.  It trains one normalized
384-dimensional key per entity from the same raw-title/raw-prefix views.  Keys
are assigned deterministically across the final three FFNs, at most 1,024 per
layer.  Each key is the actual row later written into an existing
`gate.weight`; it is not an added projection or parameter.

The key ledger is 2,405 x 384 = 923,520 existing parameter entries, 2.525% of
the served model.  Dense FFN shapes and MACs remain unchanged.  T17 must be
exact on canonical, held raw-context, and every natural-question surface.

Failure closes analog contextual entity addressing.  It will not admit more
views, more dimensions, another metric, or another classifier rescue.
