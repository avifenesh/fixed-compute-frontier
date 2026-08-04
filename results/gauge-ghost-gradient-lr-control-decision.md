# Gauge Ghost Gradient learning-rate control decision

Decision: **the original seed-21017 capability interpretation is rejected;
fresh-seed replication remains sealed.**

The frozen formal classification is `endpoint_extension_required`, because the
largest canonical learning rate in the grid, `6e-4`, achieved the lowest NLL.
That label must not be rewritten as a completed optimum search.

The already-interior `4.242640687e-4` canonical arm is nevertheless a decisive
counterexample to the narrower claim made from the original same-LR screen:

- canonical `3e-4`: `6.5238095224` NLL;
- backward-only TVE `3e-4`: `6.5070632696` NLL;
- canonical `4.242640687e-4`: `6.3833171278` NLL;
- canonical `6e-4` endpoint: `6.2681802735` NLL.

The interior canonical arm beats the frozen ghost arm by `0.1237461418` NLL,
or `1.9017%` relative to the ghost loss.  It captures `8.38948x` the original
`0.0167462528` ghost delta.  Across the same 64 validation batches, the paired
canonical-minus-ghost mean is `-0.12374614`, with standard error `0.00100033`
and the project's normal-approximation 95% interval
`[-0.12570678, -0.12178550]`.  It is already better at steps 61, 152, and 305
from an exactly identical step-zero state.

This falsifies the inference that the observed seed-21017 improvement required
the virtual Jacobian.  It does not prove that ghost gradients cannot beat a
tuned canonical frontier, because only the canonical arm was swept.  The only
authorized continuation is a matched LR frontier for both arms, extending the
upper boundary until both optima are interior.  Replication is allowed only if
the best ghost arm then beats the best canonical arm by the frozen materiality
margin.

Integrity:

- result SHA-256: `f4838d47c6775085c562e9d6bca1da0b3247d7489926fd577411a3fddd448328`;
- source SHA-256: `f6ab0e4c366eb7deac60c324ccaf062aa08d74d910629b79438e636010ba5f88`;
- preregistration SHA-256: `18fb9beafef23b4a55bd7f67e4728df26c6b3f53f4521839905ef8d591702edf`;
- manifest SHA-256: `f0411dbd58f2f287682129c820f89159a869e15b1c1588ae57bdb32812b793de`.

An independent audit recomputed every dependency hash, arm/accounting gate,
protocol binding, and decision field and returned GO for result validity and
NO-GO for advancing the present capability claim.
