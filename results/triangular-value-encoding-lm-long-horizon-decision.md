# TVE 100M-token long-horizon decision

## Decision: NO-GO under the frozen protocol

The architectural endpoint was positive, but one required control-validity gate
failed. This result must not be relabeled as a formal pass.

At 99,942,400 prediction tokens per arm and seed, triangular value encoding
(TVE) beat the gauge-canonical control in all three frozen seeds. The mean
candidate-minus-control validation NLL was `-0.0055371200`; the seed-clustered
two-sided 95 percent Student-t interval was
`[-0.0083579751, -0.0027162649]`. Mean terminal NLL improved from
`5.0969107921` to `5.0913736721`, a relative improvement of `0.1086368%`.

TVE also beat the packed raw transformer in all three seeds. The mean
candidate-minus-raw NLL was `-0.0047309895`, with seed-clustered 95 percent
interval `[-0.0091363462, -0.0003256329]`, or a `0.0928354%` mean relative
improvement. This direct native comparison was reported but was not the frozen
primary gate.

The failure came from seed 7717's gauge-canonical control. Its paired terminal
control-minus-raw mean was `+0.0020261556`, and its per-batch 95 percent upper
bound was `+0.0028233393`, narrowly outside the frozen 0.05 percent
noninferiority allowance. Consequently
`canonical_control_valid_for_all_seeds=false` and `long_horizon_pass=false`.

## What the result establishes

- The 10M-token effect did not disappear by 100M tokens at this 37.8M-parameter
  scale.
- The TVE-minus-matched-control effect has the same sign in all three untouched
  seeds and clears the frozen clustered effect-size threshold.
- The complete TVE model also has the same favorable sign against the raw model
  in all three seeds, but the frozen experiment was not powered or gated around
  that comparison.
- Disabling the learned encoding at the endpoint hurts NLL in every seed; the
  clustered full-minus-disabled interval is
  `[-0.1197443296, -0.0626308879]`. This proves strong learned use of the path,
  not that the entire net gain is uniquely representational.

## Key refinement

Gauge conversion is function-preserving at initialization, but elementwise
AdamW is not invariant to an orthogonal change of parameter coordinates. A
future native comparison therefore needs candidate-versus-packed-raw as its
primary endpoint. The gauge-matched arm remains the causal encoding control,
while an optimizer-covariant diagnostic should determine how much raw-versus-
canonical drift is caused by Adam's coordinate dependence.

The next breakthrough claim remains blocked on a larger, second-domain direct
native win and a full-attention served-cost gate. No post-hoc threshold change
is permitted for this result.
