# TVE mechanism v2 decision: shift to the backward algebra

## Valid frozen classification

`backward_signal_captures_gain_this_seed`

At seed `21017`, 9,994,240 prediction tokens, deterministic math SDPA, and
37,758,336 parameters:

| Arm | Terminal validation NLL | Relative gain vs canonical |
| --- | ---: | ---: |
| Canonical | 6.5238095 | — |
| Full TVE | 6.5092766 | 0.2228% |
| Backward-only TVE | **6.5070633** | **0.2567%** |

Backward-only captured `115.23%` of full TVE's gain and beat full TVE at every
post-training evaluation (steps 61, 152, and 305). Its forward values are
bit-exactly ordinary, it requires no encoded KV cache, and it exports the same
dense attention tensors with zero metadata bits.

The result falsifies the claim that the nonlinear TVE forward representation
was necessary for this seed's gain. The useful intervention was the virtual
Jacobian used during training.

## Key shift

Retain the algebra as a training-only ghost feature:

```text
forward:   v_ghost = v
backward:  J(v_ghost) = J(TVE(v; O))
serving:   delete the ghost operation
```

This keeps the ordinary Transformer's parameter count, hypothesis class,
attention graph, KV cache, and serving state. It changes only the finite-budget
optimization vector field. At zero reused coefficients, it adds signed
second-moment updates to selected physical O weights; once those coordinates
move, it also supplies triangular feedback to V.

## Claim boundary

This is one-seed causal evidence, not a breakthrough claim. The current exact
PyTorch surrogate took about 50% longer to train and 6.7% more peak training
memory than canonical. Five fresh initialization seeds, an efficient fused
backward, native ordinary-inference proof, and a second scale/domain are still
required.

The bound result is
`results/triangular-value-encoding-forward-mechanism-v2.json` with SHA-256
`706f38007b87400420ab916748796b0ffb75a5c431fa084c0add1a25a6c836e4`.
