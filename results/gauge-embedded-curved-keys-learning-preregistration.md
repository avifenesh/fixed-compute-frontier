# GECK-G2 matched learned retrieval — preregistration

## Fixed question

Can both recovered RoPE-gauge directions be learned as useful nonlinear
content-addressing features, rather than appearing only in a Jacobian rank
calculation?

## Frozen task

Each example contains one of two query types and two iid Gaussian keys with
two content coordinates. The hidden pivot coordinate is identically zero, so
moving the two aliased K weights cannot create a linear shortcut. Soft labels
come from an exact GECK-G2 teacher with identity Q/K content maps,
`sin(phi)=0.8`, and `tanh(s)=0.75`. Query type 0 needs the odd-square-to-even
path; query type 1 needs the even-square-to-odd path.

Five seeds each contain 5,000 examples split into 3,500 train / 1,500 test.
All arms expose ten raw parameters and receive three L-BFGS-B restarts. Within
each teacher/world, every arm receives the exact same three Q/K starts.
Runtime is frozen to Python 3.14.4, NumPy 2.3.5, and SciPy 1.18.0.
Restarts are selected by train NLL only; test metrics never select a fit.
Every selected fit must report optimizer success and finite parameters and
metrics.

## Frozen arms

1. bilinear attention;
2. released polar pivot but linear-only attention;
3. linear-epilogue functional control (not a compute-matched control);
4. constant-only epilogue;
5. one-curvature GECK;
6. centered-square full GECK;
7. full GECK-G2.

A separate bilinear teacher is a negative control for bilinear,
one-curvature, and full GECK-G2.

## Per-world pass gates

- Full GECK test NLL is within `1e-5` of the nonlinear teacher optimum and hard
  retrieval agreement is at least 99.9%.
- Every selected optimizer result is successful and finite.
- Full GECK beats one-curvature by at least 0.04 NLL and bilinear by at least
  0.08 NLL.
- Centered-square matches full within `1e-7`.
- Released-pivot linear-only matches bilinear within `1e-5`; linear and
  constant epilogues do not beat bilinear by more than `1e-5`.
- On the bilinear teacher, full GECK reaches the teacher optimum within `1e-5`,
  does not regress versus bilinear by more than `1e-5`, and returns both polar
  coordinates within `1e-3` of zero.

All five worlds must pass. Passing admits a natural learned RoPE addressed-bag
screen with centered-square and positional-only controls. It does not admit an
H100 kernel or language-model run.
