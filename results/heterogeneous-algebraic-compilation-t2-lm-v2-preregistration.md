# Heterogeneous algebraic compilation — 37M LM coexistence v2 preregistration

Frozen before the v2 full training arms: 2026-07-30

## Predecessor failure

The v1 quick implementation check was run before any language-training arm.
Its finite SiLU interpolation was exact in FP32 tests but ill-conditioned in
BF16: initial compiled parity accuracy was `85.141%` in world 731 and `98.198%`
in world 947.  V1 is rejected and no full v1 arm may run.

## V2 compiler

All model, data, optimizer, arm, checkpoint, accounting, and decision rules in
`heterogeneous-algebraic-compilation-t2-lm-preregistration.md` remain fixed,
except for this preregistered compiler replacement:

- reserve `9+k` of 384 hidden coordinates for recovered degree `k`: nine
  control coordinates plus one per-position leaf coordinate (at most 41);
- block 0 uses two SwiGLU channels per recovered position to form
  `1 + bit` in its leaf coordinate;
- block 1 uses one causal attention head to gather all leaves and a second to
  preserve the copy query;
- blocks 2 onward use a balanced exact product tree based only on
  `SiLU(z)-SiLU(-z)=z`;
- the next block copies parity or protected value into a dedicated tied-output
  coordinate, preventing output weights from corrupting input leaf values;
- each multiplication uses four channels: two for the product and two to
  subtract the overwritten residual value;
- at most 64 routing channels and 64 product channels are reserved in their
  respective 1,024-wide FFNs;
- later blocks preserve compiler coordinates by residual identity;
- all compiled arithmetic must be evaluated under the same BF16 autocast as
  training before any full arm starts.
- fixed-entry gradients are zeroed before global norm measurement/clipping and
  fixed values are restored after the optimizer step, so frozen compiler
  gradients cannot suppress language-lane updates.

The larger reservation makes the protected-language gate strictly harder.  The
unchanged v1 rule still requires compiler validation NLL to be no more than
`0.5%` above the paired baseline at steps 250, 500, and 1,000.  No tolerance is
relaxed because the robust circuit consumes more capacity.

The full v2 run is forbidden unless quick BF16 checks reach at least 99% parity
and copy accuracy in both sealed worlds and unit tests verify exact mask
restoration.
