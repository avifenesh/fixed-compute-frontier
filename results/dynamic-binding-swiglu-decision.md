# Margin-damped Dynamic Binding SwiGLU — decision

## Verdict

**Closed before replication or kernel work.**

The algebra is genuinely outside ordinary SwiGLU and the learned model used
the token-specific route.  It nevertheless produced worse language modeling.

At 10M prediction tokens:

| Arm | Validation NLL | Relative to raw |
|---|---:|---:|
| raw SwiGLU | 6.5270798 | — |
| exact identity wrapper | **6.5269290** | +0.00231% |
| exact fixed-shift reparameterization | 6.5270715 | +0.00013% |
| amplitude-only control | 6.5342807 | -0.1103% |
| fixed-relation margin mixing, normalized | 6.5734540 | -0.7105% |
| dynamic binding, unnormalized | 6.5614856 | -0.5271% |
| dynamic binding, variance-normalized | 6.5715185 | **-0.6808%** |

The exact controls reproduce raw SwiGLU.  This validates the fixed-permutation
equivalence and excludes wrapper/indexing effects.  Variance normalization did
not rescue the dynamic model, and amplitude-only behavior was not the source
of a hidden gain.

## Causal but harmful

The candidate did not ignore its new path:

- median normalized route entropy: `0.9968`;
- median routing strength: `0.3775`;
- correction RMS: `0.8272x` the ordinary activation RMS;
- forcing identity after training raised NLL to `6.9406242`;
- token-misrouting raised NLL to `6.8752236`.

The weights co-adapted strongly to the route, but that routed function was
worse than the ordinary FFN.  A large ablation effect therefore cannot be used
as evidence of a capability gain.

## Retained algebra

For group gate values `g`, values `u`, winner shift `s`, and top-two margin
`m`, the map

\[
u'=u+\tanh^2(m)(P_su-u)
\]

removes the winner-switch discontinuity: the route-dependent correction
vanishes quadratically at a winner tie.  It uses no learned router or selector
state and is piecewise smooth.  Runner-up identity changes can still create
derivative kinks.  Variance-normalized interpolation is required as a control.

Do not tune group size, damping, normalization, or routing codebooks to rescue
this FFN claim.  Reuse the margin-damped sparse-choice primitive only where the
chosen alternatives already have independently useful semantics.

Evidence SHA-256:
`24dde8c5ef1ff017b268e3757de8ad1d523eb7b829de20df3303318edb0c27b2`.
