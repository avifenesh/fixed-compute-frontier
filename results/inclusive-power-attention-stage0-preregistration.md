# Inclusive-Power versus Power-Evidence Attention — Stage 0 preregistration

Status: frozen before formal execution  
Date: 2026-07-27

## Question

Does the ordinary online-softmax partition state already support the useful
part of Power-Evidence Attention (PEA), making its extra `l2` statistic
unnecessary?

For chunk `b`, let

\[
Z_{1,b}=\sum_j e^{s_j},\qquad
\mu_b=\frac{\sum_j e^{s_j}v_j}{Z_{1,b}}.
\]

Inclusive-Power Attention (IPA) with router power `r=2` is

\[
W_{I,b}=\frac{Z_{1,b}^2}{n_b},\qquad
y_I=\frac{\sum_b W_{I,b}\mu_b}{\sum_b W_{I,b}}.
\]

Its common token-logit lift is

\[
A_{I,b}=\log(Z_{1,b}/n_b)=m_b+\log l_b-\log n_b.
\]

PEA instead uses `W_P,b=Z_2,b=sum exp(2s)`. Both select chunks more
sharply while retaining the ordinary temperature-one conditional read inside
each chunk. IPA uses only ordinary finalized state `(m,l,o,n)`; PEA adds `l2`.

## Frozen algebra gates

All calculations use float64 and tolerance `1e-12`.

1. Direct lifted-token IPA and a two-level `(W_I,mu)` construction agree for
   random unequal chunks and under row shifts `+/-500`.
2. Router power `r=1` is exactly ordinary attention for every partition.
3. IPA preserves ordinary within-chunk conditional weights, while global
   temperature two changes them by at least `1e-3`.
4. A third token changes IPA cross-chunk odds by at least `0.1`; ordinary
   pairwise IIA changes by at most `1e-12`.
5. Adding a common score constant leaves IPA probabilities invariant.
6. When every score in a chunk equals `c`, IPA and PEA router mass both equal
   `n*exp(2c)` within tolerance.
7. For positive `x_j=exp(s_j)`, verify exactly

   `W_PEA/W_IPA = mean(x^2)/mean(x)^2 = 1 + CV(x)^2`,

   with equality only for a constant block in the fixed witnesses.

## Frozen IID/cardinality characterization

Enumerate all `2^8` assignments of `x in {0.5,2}` into chunks of sizes
`(1,3,4)`.

- PEA expected raw router mass per token must equal `E[x^2]` for every size,
  and its expected normalized shares must equal `(1,3,4)/8`.
- IPA expected raw router mass per token must equal
  `E[x]^2 + Var(x)/n` for every size.
- IPA's expected normalized share must differ from cardinality share by at
  least `0.02` for one chunk. This is a required exposed failure, not a
  claimed virtue: a smaller IID chunk receives a variance premium.

## Frozen inductive-bias witnesses

### Anchor-to-payload

Use the existing sequence-128, chunk-16, eight-label exact enumeration at
anchor gaps `Delta in {0.5,1,2,3,4}`. The desired payload has score zero and
shares a chunk with the only nonzero-score anchor.

- PEA target-payload probability must strictly exceed IPA at every Delta.
- IPA target-payload probability must strictly exceed ordinary at every
  Delta.
- The measured PEA/IPA target-chunk router-mass ratio must match the frozen
  `1+CV^2` identity within `1e-12`.

### Consensus versus spike

Compare two equal 16-token chunks:

- spike chunk: one score `3`, fifteen scores `0`;
- consensus chunk: four scores `1.8`, twelve scores `0`.

IPA must assign greater router mass to consensus; PEA must assign greater
router mass to the spike. This isolates average-evidence sharpening from the
additional dispersion preference.

## Resource boundary

For a native finalized chunk with shifted online-softmax state `(m,l,o,n)`,
IPA2 can feed the outer merge using center `2m`, scalar mass `l^2/n`, and
value numerator scale `l/n`. It adds no score multiply, reduction accumulator,
exponential, QK/PV/QKV/FFN matmul, KV-cache field, or learned parameter.

This does not erase the two-level vector-lifetime constraint: if semantic
chunks do not equal native PV/split units, a current local value accumulator
may coexist with the outer accumulator. Therefore Stage 0 makes no runtime or
monolithic-prefill claim.

Passing admits IPA as a mandatory learned control. It admits neither IPA nor
PEA to a language-model or kernel claim.
