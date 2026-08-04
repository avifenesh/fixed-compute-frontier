# Conditional-weight allocation frontier decision

Decision: BBCM has no unconditional budget or traffic advantage; independent low-bit experts are mandatory controls.

- K4-W4 fits the same BF16 resident budget at width 14,320 with 98,552 bytes slack, versus 33,080 for each shared-base layout.
- One K4-W4 route reads four payload bits per coordinate. `8 + 4x2` must read ten bits per coordinate before metadata and therefore has 2.5 times the ideal active payload.
- K8-W2 fits at width 14,304 and reads two payload bits per active route; K2-W8 fits at width 14,320 and reads eight.
- In the optimistic scalar high-rate model, `8 + 4x2` beats K4-W4 only below `Var(Delta)/Var(Base)=0.06640625`. For independent deltas this corresponds to pairwise route correlation above `0.93772894`; if the base is the four-route mean, the exchangeable zero-sum conversion gives `0.91697` instead. The learning gate therefore measures variance ratio directly.
- `12 + 4x1` requires correlation above `0.98437523`; `4 + 4x3` never strictly wins in that approximation.
- The allocation-neutral corner witness routes all four formats and reconstructs exactly, so later differences are not caused by an impossible 1-bit codebook.
- Result SHA-256: `db46a4fcff169e774f61e8296f5901caa31a169e02287ea1c9d2e937b561defc`.

These are allocation targets, not LLM predictions. Real ternary codebooks, joint errors, routing, and nonlinear loss require a matched learning screen. The result removes the claim that four two-bit deltas are naturally optimal.
