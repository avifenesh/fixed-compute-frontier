# Gauge-exposed partner-feedback SwiGLU — Stage-0 preregistration

Status: **frozen before execution**.

## Candidate

Let ordinary SwiGLU produce

\[
z_i=\operatorname{SiLU}(g_i)u_i.
\]

Choose a fixed-point-free involution `P` (adjacent pairs after a free hidden
neuron reserialization) and compute

\[
q_i=\operatorname{SiLU}(g_i+\lambda\operatorname{clip}(z_{P(i)}))u_i,
\qquad y=Vq.
\]

This is a parallel second gate pass, not a recurrence: all `z` values are
computed from the original projections, and every `q` can then be computed
simultaneously.

At `lambda=0` it is exactly SwiGLU.  For nonzero `lambda`, the ordinary gauge
transformation `U_j -> s_j U_j`, `V_:j -> V_:j/s_j` no longer cancels
everywhere: `z_j` changes the partner's gate.  Thus the activation makes every
formerly redundant neuron-scale coordinate functional without decoding a
per-neuron coefficient.

The one layer scalar is itself carried by the existing scale orbit of neuron
zero.  With deterministic chart value `c` and carrier gain `kappa=4`,

\[
\lambda=(U_{0,0}/c-1)/\kappa.
\]

Only that one semantic pivot is read by the activation kernel.  `G,U,V` remain
the only serialized tensors.

## Gates

1. Canonicalizing the carrier row to `U[0,0]=c` and inversely rescaling its
   down column preserves ordinary float64 SwiGLU within `1e-11`.
2. The decoded `lambda=0` endpoint matches ordinary SwiGLU within `1e-11`.
3. Folding and decoding a nonzero layer coefficient matches the explicit
   partner-feedback function within `1e-11`.
4. Scaling a noncarrier source row and inversely scaling its down column leaves
   ordinary SwiGLU invariant within `1e-11` but changes partner-feedback output
   by more than `1e-6`.
5. In the square-activation, locally unclipped analogue, one paired target has
   a nonzero degree-seven term while ordinary square-gated FFNs are homogeneous
   degree three.
6. Trainable and serialized parameter counts remain exactly `3*D*M`; dense
   MAC count remains exactly `3*D*M` per token.
7. No per-feature coefficient tensor, permutation metadata, serial dependency,
   reduction, or additional activation-sized workspace is allowed.

Containment is exact on the open dense chart where the carrier pivot is
nonzero.  The fixed matching can always be made adjacent by permuting `G/U`
rows and inverse-permuting `V` columns at serialization, so no route table is
charged.

Passing authorizes a fused full-FFN H100 gate.  It does not authorize an LM
screen until the kernel passes and a reviewer validates that the experiment
includes ordinary SwiGLU, self-feedback, gauge-null, and partner-feedback
controls.

