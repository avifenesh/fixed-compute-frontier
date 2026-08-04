# Feedback-SwiGLU stage-0 preregistration

Status: frozen before execution.

Scope: algebra, strict expressivity witness, and exact resource ledger only. This gate cannot establish a language-model quality gain.

## Candidate

For a bias-free width-`M` SwiGLU with model dimension `D`,

```text
g  = G x
u  = U x
z0 = SiLU(g) * u
h  = P^T z0
z1 = SiLU(g + R tanh(h)) * u
y  = V z1
```

where `P,R in R^(M x q)` and `q << D`.

Setting `R=0` recovers the original SwiGLU exactly. The hypothesis is that one low-rank nonlinear feedback step lets hidden features condition other hidden features without another `D x M` projection.

## Frozen reference ledger

- `D=4096`, baseline `M=14336`, `q=8`, BF16 weights.
- Baseline parameters and MACs per token: `3 D M`.
- Added candidate parameters and MACs per token at equal width: `2 M q`.
- Ignore elementwise activation cost in both ledgers, but report it separately in the hardware gate.
- Equal-parameter candidate width is `floor(3 D M / (3 D + 2 q))`.

## G0 gates

All must pass:

1. With `R=0`, maximum float64 absolute output difference from SwiGLU is at most `1e-12` on a frozen random test.
2. In the square-activation analogue, ordinary gated FFNs have polynomial degree at most 3 at any width, while the frozen feedback construction has a nonzero degree-7 term.
3. A frozen SiLU feedback skeleton has a nonzero rectangular mixed difference, while its no-feedback counterpart is zero to numerical tolerance.
4. Equal-width parameter/MAC overhead is below `0.14%` at the reference shape.
5. The exact equal-parameter width reduction is below `0.20%`.

## Required later controls

The first learnability gate must compare matched total parameters and training FLOPs against:

- ordinary SwiGLU;
- a parallel preactivation low-rank mixer using the same `P,R`;
- a parallel postactivation low-rank mixer using the same `P,R`;
- token-adaptive Mixture of Activations;
- a deeper/narrower SwiGLU with equal total training FLOPs.

The first H100 gate must report same-width and equal-parameter variants separately. A PyTorch-launch failure only rejects that executor; a serving claim requires a fused implementation. Key decode cells are `B in {1,8}`. Candidate median latency must be within `1.02x` of baseline there and within `1.05x` on the full frozen grid.

## Stop rules

- Stop the branch if G0 fails.
- Do not call the polynomial witness evidence for deployed-SiLU superiority.
- Do not call the candidate a breakthrough unless it wins a matched language-model gate and a fused serving gate.

