# Gauge Ghost Gradient: asymmetric train/serve algebra

## The status-quo assumption being broken

Modern dense LLM training normally uses one algebra twice:

```text
forward function F(theta, x)
backward Jacobian dF/dtheta
```

The served graph and the graph that defines parameter updates are assumed to be
the same differentiable program. Gauge Ghost Gradient (GGG) deliberately
separates them. The served function remains an ordinary Transformer, while
training uses the Jacobian of a richer virtual feature map.

For a value block `v in R^B`, lower-triangular coefficients `C` reused from
physical O weights, and `phi(v) = v * |v|`, define

```text
T(v, C) = v + C phi(v)
G(v, C) = stopgrad(v) + T(v, C) - stopgrad(T(v, C)).
```

Then exactly:

```text
forward:   G(v, C) = v
backward:  J_G = J_T.
```

The ghost operation can be deleted after training. No coefficient tensor is
added: `C` is read from the same capped lower-triangular O coordinates that the
ordinary output projection already stores and serves.

## Local update equations

For incoming gradient `g` and `C_ji = O_ji / tau` for `i < j`, the virtual VJP
adds

```text
dL/dv_i = g_i + 2 |v_i| sum_{j>i} C_ji g_j
dL/dO_ji += (1/tau) sum_tokens g_j v_i |v_i|,  i < j.
```

The ordinary output-projection gradient is still present and sums into the same
physical O tensor. At canonical initialization `C=0`, the value gradient is
ordinary, but active O coordinates immediately receive signed-second-moment
updates. Once those coordinates move, the triangular value feedback turns on.
This gives a bootstrap sequence:

```text
moment signal -> physical O coordinates -> value feedback -> later layers.
```

It is generally a non-conservative update field: there need not be a scalar
loss on the ordinary forward model whose exact gradient equals GGG. Therefore
GGG does not enlarge the served hypothesis class or its theoretical best
optimum. Its possible edge is optimization: reaching a better ordinary model
under a fixed finite training budget.

## Arithmetic scaling

With head width `d` split into blocks of capped width `B`, each KV head has

```text
slots = d(B - 1) / 2
```

ghost interactions per layer. Both the triangular value VJP and active O
moment reduction are linear in `d` for fixed `B`, rather than quadratic in
model width.

For the 37.8M test (`d=64`, `B=16`, two KV heads):

- 480 active interactions per KV head;
- 960 active physical slots per layer;
- roughly 1,920 extra multiply-like interactions per token/layer for both
  virtual VJP terms together;
- about 1.57M ordinary dense projection/FFN MACs per token/layer before
  attention-score work.

The arithmetic ratio is about `0.12%`; the current 50% measured slowdown is
therefore an implementation artifact from materializing and differentiating a
duplicate PyTorch surrogate graph, not an algebraic FLOP requirement.

## Kernel endpoint

A production training kernel should not run the virtual forward:

1. Ordinary attention forward writes ordinary values.
2. Attention backward already produces the per-token value gradient `g`.
3. Its epilogue adds the block-triangular value feedback into the existing
   `dV` tensor.
4. Token tiles reduce `g_j phi(v_i)` only for capped active slots and add those
   FP32 partials into the already allocated dense `dO` gradient.
5. Export deletes the custom backward rule; inference loads ordinary dense
   Q/K/V/O weights.

No persistent model bytes, optimizer tensors, inference metadata, KV width, or
serving operations are added. Physical validation must still measure kernel
launches, partial-reduction workspace, synchronization, and full-model training
throughput; the arithmetic count alone is not a resource result.

## Evidence boundary

The valid causal screen currently establishes the effect for one seed only:
backward-only GGG improved relative NLL by 0.2567%, captured 115.2% of full
TVE's gain, and beat full TVE at all post-training checkpoints. Five fresh
seeds, an efficient backward trajectory test, and a second scale/domain remain
required before calling this a model-scaling breakthrough.
