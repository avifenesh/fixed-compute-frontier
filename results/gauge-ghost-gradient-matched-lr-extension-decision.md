# Gauge Ghost Gradient matched-LR extension decision

Decision: **close Gauge Ghost Gradient as the current capability direction; do
not release the five replication seeds.**

The frozen formal label remains `endpoint_extension_required` because the best
canonical arm is the `4.8e-3` upper endpoint.  That label is correct and is not
rewritten.  It only says the ordinary model may improve further.

The ghost side is already bracketed on the tested geometric grid:

- `1.2e-3`: `6.1027457491` NLL;
- `1.697056e-3`: **`6.0891532525`** NLL;
- `2.4e-3`: `6.1066945642` NLL;
- `3.394113e-3`: `11.3999495357` NLL;
- `4.8e-3`: `7.5934029594` NLL.

The best observed ordinary arm is already `5.9182367697` NLL at `4.8e-3`.
Thus ghost's best tested loss is `0.1709164828` higher: `2.88796%` regret
relative to canonical, with paired 64-batch interval
`[0.16704129, 0.17479168]`.  Extending the canonical frontier can only preserve
or widen this observed gap; it cannot make the existing ghost frontier win.

The trajectory explains the false start.  Ghost improves the under-scaled
canonical optimizer at every LR through `8.485e-4`.  At `1.2e-3` it becomes
slightly worse; at `1.697e-3` its best loss is still worse than same-LR
canonical (`6.0371616632`); above that it narrows the stable LR region, reaching
terminal gradient norms `48,451.7` at `3.394e-3` while the canonical model stays
stable.  The virtual Jacobian acted like an optimization preconditioner whose
benefit disappeared after ordinary LR tuning and then became destabilizing.

This does not invalidate the algebra, same-LR gradient effect, or ordinary
serving identity.  It rejects the claim relevant to this project: no better
model was obtained than the tuned ordinary Transformer, and training was also
about 1.5x slower.  No scrambled-Jacobian control, fresh-seed replication,
fused backward, or 360M-token scale run is authorized.

Integrity:

- result SHA-256: `b0b05e271c4f40495ab51c64e0e0e9f07d05c600ce2bbba2d65a2f3203190631`;
- source SHA-256: `0d609b4e15551787fd649f4618461d748cbdd8b2e433d31caff563d948746c6e`;
- preregistration SHA-256: `d45c3fc15e66e03c57ccbb46e162766b4d3f5870513c431d017a925b2ee78aac`;
- manifest SHA-256: `30cf54a7bbf2cbc3ea1aa78e29c81bef03fe6d08ff710675a25155eb59479730`.

The broad lesson is mandatory for every successor: bracket the ordinary and
candidate optimization frontiers before interpreting a same-hyperparameter
architecture delta.
