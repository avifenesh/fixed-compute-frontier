# Orbit-activated SwiGLU — Stage-0 preregistration

Status: **frozen before execution**.

## Question

Can a SwiGLU use its existing per-neuron scale-gauge coordinate as a real
cross-feature coefficient, increasing functional capacity without adding a
serialized weight or changing any dense matrix shape?

This is a different claim from removing one coordinate and packing another
parameter into the saved slot.  The coefficient is encoded in the scale of the
ordinary up/down weights themselves.

## Algebra

Ordinary bias-free SwiGLU is

\[
g=Gx,\qquad u=Ux,\qquad z_i=\operatorname{SiLU}(g_i)u_i,
\qquad y=Vz.
\]

For every hidden unit and nonzero `s_i`, the transformation

\[
U_i\mapsto s_iU_i,\qquad V_{:i}\mapsto V_{:i}/s_i
\]

leaves the ordinary function unchanged.  Choose deterministic pivots
`p_i = i mod D`, nonzero fixed chart values `c_i`, and carrier gain `kappa`.
The candidate decodes

\[
a_i=\frac{U_{i,p_i}/c_i-1}{\kappa}.
\]

Within groups of eight, using `clip(t)=min(1,max(-1,t))`, it evaluates

\[
q_0=\operatorname{SiLU}(g_0+a_0)u_0,
\]

\[
q_i=\operatorname{SiLU}(g_i+a_i\operatorname{clip}(q_{i-1}))u_i,
\quad i=1,\ldots,7,
\qquad y=Vq.
\]

At `a=0`, this is exactly ordinary SwiGLU.  During training, use canonical
weights `U0,V0` with `U0[i,p_i]=c_i`, learn one `a_i` in place of each fixed
pivot, and export

\[
s_i=1+\kappa a_i,\quad U_i=s_iU0_i,\quad V_{:i}=V0_{:i}/s_i.
\]

The deployed kernel recovers `a_i` from the actual dense `U` pivot.  It stores
only `G,U,V`; no `a` tensor is serialized.

## Frozen Stage-0 gates

All must pass:

1. An arbitrary ordinary SwiGLU with nonzero deterministic pivots can be
   canonicalized without changing float64 output by more than `1e-11`.
2. `a=0` matches the canonical ordinary SwiGLU within `1e-11`.
3. Folding and decoding nonzero coefficients reproduces the explicit
   candidate within `1e-11` in float64.
4. Candidate movement along an ordinary SwiGLU scale orbit is nonzero, while
   the ordinary output movement along the same orbit is at most `1e-11`.
5. In the square-activation, unclipped local analogue, a two-unit candidate
   has a nonzero degree-seven term; an ordinary square-gated FFN of any width
   is homogeneous degree three.
6. Both training trainable-parameter count and serialized inference-word count
   equal exactly `3*D*M`.
7. Dense matrix MACs remain exactly `3*D*M` per token.  All added arithmetic,
   pivot reads, serial depth, and training-only storage are reported rather
   than hidden in this equality.
8. A BF16 export simulation recovers coefficients with finite error and keeps
   candidate output error below `0.5%` row-relative RMS on the frozen random
   screen.  This is only a codec sanity gate, not a serving or quality claim.

## Reference ledger

For `D=4096`, `M=14336`, BF16:

- baseline and candidate serialized weights: `3*D*M = 176,160,768` words,
  or `352,321,536` bytes;
- candidate dense MACs/token: `176,160,768`, exactly baseline;
- coefficient carrier reads: `M = 14,336` BF16 words per layer invocation,
  ideally `28,672` bytes or `1/(3D) = 0.008138%` of weight payload;
- group-local serial depth: seven predecessor steps;
- training representation: fixed `M` canonical pivots plus `M` learned
  coefficients, so trainable count is unchanged, but physical training storage
  may be larger and must be disclosed.

## Stop rules and required controls

Stop before a language-model run if any algebra, ledger, or BF16 gate fails.
Passing Stage 0 authorizes only:

1. a fused H100 FFN gate against ordinary SwiGLU at batch/token counts
   `1,8,32,128`;
2. a matched LM screen with ordinary SwiGLU, orbit-carried parallel gate bias,
   orbit-carried detached one-hop coupling, and full recursive coupling;
3. immediate ablations that zero all `a` and remove paths longer than one hop.

No capability claim is allowed without beating ordinary SwiGLU.  No fixed-cost
claim is allowed if coefficient extraction, recurrence, quantization scales,
workspace, or kernel fusion adds more than `2%` serving latency or any
persistent model-sized tensor.

