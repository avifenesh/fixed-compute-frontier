# Full-gauge bi-curved keys (GECK-G2) — Stage 0 preregistration

## Fixed question

Can an exact RoPE query/key gauge redundancy be converted into nonlinear
content-addressing features with no new learned scalars, no KV-cache width, and
no new dense matmul?

## Frozen construction

For each raw pre-RoPE key pair `(u_f, v_f)`, the model-load atlas selects a
nonzero hidden-coordinate pivot `p_f` whose radius is closest to `tau`; its
index is then frozen as a compile-time constant. Use the commuting rotation
plus reciprocal Q/K scale gauge to map
the selected K column to `(0, tau)`, where `tau=1/sqrt(D)`, then write

Let `a=K[2f,p_f]`, `b=K[2f+1,p_f]`, and `r^2=a^2+b^2`. Write

`u_f <- u_f + (a/r) * v_f^2`

`v_f <- v_f + ((r^2-tau^2)/(r^2+tau^2)) * u_f^2`.

The first coefficient is defined as zero at `r=0`. Both coefficients are
bounded in `[-1,1]`. Training uses centered polar gauge coordinates for the two
pivot scalars so ordinary weight decay has its fixed point at the nonlinear-off
slice; physical K weights alone are stored for serving.

Both right-hand sides use the original linear `u_f,v_f`.

The coefficients derive from two existing key-projection weights. No duplicate
coefficient tensor is credited. Compile-time pivot-index metadata is counted
explicitly in the ledger. RoPE is applied after the mutation.

For norm-free ordinary RoPE attention, jointly applying a commuting complex
scale/rotation to one K pair and the inverse transpose to every Q pair in its
GQA group preserves scores. Choose it to set the selected K column to
`(0,tau)`; both curved terms vanish, proving containment on the generic
nonzero-pivot chart. The model-load atlas chooses a nonzero pivot (and, at
initialization, the radius closest to `tau`) per pair; this covers every
nonzero K pair and avoids the fixed-column `1/r` conditioning tail. A zero K
pair produces zero curved features. Pivot choices are frozen compile-time
indices, not learned scalars or per-token state. Their exact bit count is
included in the ledger; no zero-metadata claim is made.

## Frozen gates

1. Random multi-pair GQA scores before/after gauge fixing agree within `1e-10`.
2. GECK on the zero-coefficient gauge slice agrees with the original baseline
   within `1e-10`.
3. Every pair gauge commutes with RoPE. Head-wide QK normalization is explicitly
   out of scope because independent per-pair scaling does not commute with it.
   On the frozen iid initialization, the closest-radius atlas must keep every
   reciprocal scale factor within `[0.5,2.0]`.
4. The executable polar chart must satisfy
   `a=tau*exp(s)*sin(phi)`, `b=tau*exp(s)*cos(phi)`, hence
   `c_rot=sin(phi)` and `c_scale=tanh(s)`, with `(phi,s)=(0,0)` both
   nonlinear-off and the centered weight-decay fixed point. Zero-column,
   nonzero-alternative, and all-zero-pair atlas cases must pass.
5. A fixed-query second finite difference in key input is nonzero for GECK and
   zero for every ordinary bilinear QK score.
6. Across five seeds of 64 frozen grouped attention log-odds (one query, two keys, and at least
   two relative RoPE angles) with 12 raw Q/K parameters, numerical functional
   rank is 10 for ordinary RoPE and the full 12 for GECK-G2 both on the exact
   baseline slice and generically.
7. The resource ledger has identical K weights, dense K MACs, and cache width.
   Token mutation adds four multiplications and two adds per RoPE pair. The
   separate coefficient ledger charges two squares, one add, one square root,
   two divisions, and two add/subtracts per pair per forward/kernel unless a
   stored sidecar is used; this Stage 0 credits no sidecar.

This stage does not claim useful language quality or H100 neutrality. A pass
admits only a matched learned addressed-retrieval screen. Learned
head-wide or coordinatewise QK norms are out of scope because they break the
independent reciprocal per-pair scale gauge.

The learning screen must include nonlinear-off polar baseline, released-pivot
linear-only, compute-matched linear epilogue, fixed positional-only square,
centered-square, and pivot/tau controls. This separates content curvature from
tied linear motion, optimizer centering, and a learned RoPE-relative bias.
