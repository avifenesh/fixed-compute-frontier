# Conditional-weight allocation frontier preregistration

Status: frozen before execution.

## Question

Does `INT8 base + four INT2 deltas` have an unconditional bit/traffic advantage, or is it only one point in an allocation frontier that must beat independent low-bit experts?

## Frozen exact ledger

At baseline `D=4096`, `M=14336`, compare under BF16 resident bytes and baseline matrix MACs including router MACs:

- shared `12 + 4x1`, `8 + 4x2`, and `4 + 4x3`, each with five BF16 per-output scale sets;
- independent K4-W4, K8-W2, and K2-W8, each with one BF16 scale set per expert;
- BF16 affine router weights and biases; aligned width may shrink by 16.

Record resident bytes, metadata-funded aligned width, selected-matrix plus router MACs, and active weight payload bits. Packing, decode operations, grouping, TP replication, and scratch remain outside this gate.

## Frozen analytic screen

Use the transparent optimistic approximation `distortion = variance * 2^(-2 bits)`. Compare each K4 shared allocation with independent K4-W4. For route matrices `W_e=B+Delta_e` with independent deltas, convert the maximum permitted `Var(Delta)/Var(B)` into pairwise route correlation `1/(1+ratio)`.

This is not a realistic 1-bit or ternary quantizer model. It intentionally omits codebook constants, the unused ternary code, base/delta error cross terms, routing errors, nonlinear loss, and runtime. It may only reject broad claims and set an empirical target.

An allocation-neutral implementation witness uses the four corners `(-1,-1)`, `(-1,1)`, `(1,-1)`, `(1,1)` as both inputs and router rows, base `(1,2)`, and route deltas equal to the corners. It must route uniquely, reconstruct with zero MSE in every frozen K4 format, and leave MSE four for the best single bias-free linear row. The delta values use only `{-1,+1}`, so the 1-bit arm is not asked to represent zero.
