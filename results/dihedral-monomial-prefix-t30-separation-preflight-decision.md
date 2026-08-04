# T30 arbitrary-state separation preflight — decision

Status: **PASS AS A LOCAL THEOREM; NO TRAINING ADMITTED**  
Date: 2026-07-31

## Verdict

The corrected arbitrary-incoming-state witness passed.  It establishes a
large, exact separation between the `D_109` routed operator and the full class
of diagonal-affine word maps on this declared operator task.

It does not establish a useful raw-prose statistic, learning, architectural
novelty, lower physical cost, or better language-model capability.  Because
permutation-diagonal state tracking is prior art, this pass does not justify a
GPU training run by itself.

## Authoritative result

- unit tests: 8/8 passed;
- group elements: 218;
- centered-basis inputs per group: 218;
- exhaustive state evaluations: 47,524;
- canonical-word failures: zero;
- distinct group actions: 218/218;
- routed construction maximum error: exactly zero;
- observed strongest relaxed diagonal-affine NMSE: `0.9908256880733948`;
- proved value `108/109`: `0.9908256880733946`;
- maximum per-group formula discrepancy: `2.220446049250313e-16`;
- state ledger: 220 bytes.

The control is deliberately stronger than a shared T29 recurrence for this
lower-bound calculation: every word receives its own independently optimal
diagonal and offset.  Since even this relaxation has error `108/109` on
average, a shared diagonal-affine token recurrence cannot do better.

## Integrity

- preregistration SHA-256:
  `47c97367c088800e1ad7392ae5f2f44e74d0c667dea72e24e8434e38de2d3a86`
- implementation SHA-256:
  `965aeb296143bfd77d4ab6768c98e55323140a888af934d3a3322cd07d6dda37`
- tests SHA-256:
  `4edeb017988cfed02ff03fe786e16e2b05a7aa9830c0d2b43af46337bedda2b2`

The run used local CPU NumPy only.  No H100 work, model training, retry, or
threshold change occurred.

## Admission boundary

The next required paper must identify a raw-prose statistic with all of these
properties:

1. it predicts a protected natural knowledge or reasoning capability;
2. one-pass T30 token routes can construct it without semantic labels;
3. the full matched diagonal-affine or hybrid baseline provably aliases or
   loses it at the same width and serving envelope;
4. an overpowered reader can exploit it with a breakthrough-size predicted
   margin;
5. the claim remains nonredundant with PD-SSM's existing finite-state and
   English-transition results.

Without such a witness, retain T30 as a correct compiled recurrence primitive
and close it as the active breakthrough direction.
