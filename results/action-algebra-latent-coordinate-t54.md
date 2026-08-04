# T54 action algebra as latent coordinates — exact affine control and its wall

Date: 2026-08-01  
Status: **AUDITED EXACT AFFINE CONTROL; ARCHITECTURE NOVELTY REJECTED; NO RUN**

## 0. Question

T53 leaves a possible circularity: a mechanism learner needs variables, but a
learned compiler that extracts the variables may contain the whole intelligence
problem. Can the algebra of legal actions define the variables directly from
raw observations?

The answer is yes under strong excitation and observation-structure
assumptions. The same calculation proves why arbitrary raw encodings restore an
exponential bill.

## 1. Finite affine world

Let latent state be

\[
z\in\mathbb F_2^r.
\]

Environment `e` exposes only

\[
o=A_e z+b_e,
\]

where `A_e` is an unknown invertible `r x r` binary matrix and `b_e` is an
unknown offset. The learner can reset to one unknown anchor `z_0`. It has `r`
action labels. Action `i` toggles one latent coordinate,

\[
T_i(z)=z+e_{\pi(i)},
\]

for an unknown permutation `pi`. Thus the actions are commuting involutions.
No latent coordinate or shared variable name is observed.

## 2. T54.1 — coordinate recovery from one action basis

Observe the anchor

\[
o_0=A_ez_0+b_e
\]

and one transition from the reset anchor under each action:

\[
o_i=A_e(z_0+e_{\pi(i)})+b_e.
\]

Define the observed differences

\[
d_i=o_i-o_0=A_e e_{\pi(i)}
\]

and `D=[d_1 ... d_r]`. Then

\[
D=A_eP_\pi
\]

is invertible. For every future observation `o`,

\[
\boxed{
\widetilde z=D^{-1}(o-o_0)
=P_\pi^{-1}(z-z_0).
}
\]

Therefore `r+1` reset observations identify the latent coordinates up to the
unavoidable permutation of action labels and choice of origin.

**Proof.** Subtraction cancels both the anchor and offset. The `r` action
differences are the columns of `A_e` in permuted order, so `D` is invertible.
Left multiplication by its inverse gives the displayed equality. QED.

The exact algebraic learner needs `r+1` distinct observations, `r` action
transitions and exact resets, `r^2+r` calibration bits for `D,o_0`, `O(r^3)`
preprocessing for binary matrix inversion, and `O(r^2)` bit operations per
encoded state with a dense inverse. Action execution and observation
acquisition are charged. Faster structured multiplication is possible only if
more structure is assumed.

## 3. T54.2 — excitation-rank impossibility

Generalize the action effects to columns of an unknown latent action matrix

\[
B=[v_1,\ldots,v_q]\in\mathbb F_2^{r\times q}.
\]

The observed action differences reveal `A_eB`. Let `W=span(B)` with
`dim(W)=d<r`. Except when `(r,d)=(1,0)`, there is a nonidentity `S in GL(r,2)`
that fixes `W` pointwise. If `W` is nonzero, choose nonzero `w in W` and a
nonzero linear functional `ell` vanishing on `W`; the transvection
`Sx=x+ell(x)w` is invertible and fixes `W`. If `W=0` and `r>=2`, an elementary
shear works. The two factorizations

\[
(A_e,z)
\quad\text{and}\quad
(A_eS^{-1},Sz)
\]

produce identical observations and identical effects for every generated
action sequence, because `SB=B`. In the degenerate case `r=1,B=0`, the affine
translation `z'=z+1` with `b'_e=b_e+A_e` gives the same non-identifiability.

Consequently the latent factorization is identifiable at most modulo

\[
\operatorname{Stab}(B)=\{S\in GL(r,2):SB=B\}
\]

plus the affine-origin gauge. Any deployment query whose answer changes under
this stabilizer but is not action-observation invariant cannot be guaranteed.
Full rank is sufficient for every linear coordinate, but is unnecessary for a
query depending only on the excited subspace or invariant under the stabilizer.

This is the exact resource: the action effects must span every latent direction
the deployment query needs. Model size cannot replace missing excitation.

## 4. T54.3 — arbitrary observation maps restore the bill

Now let

\[
o=\phi_e(z)
\]

for an arbitrary injective map on `2^r` latent states. The commuting toggle
actions still define a free transitive action of the group

\[
G=(\mathbb Z_2)^r.
\]

Given an anchor and the action word that reached an observation, its coordinate
is the parity vector of that word, up to generator permutation and origin. But
to encode an observation presented without its action path, an arbitrary
`phi_e` has no extrapolatable structure. In the worst case the learner needs
`2^r-1` mapped pairs; the final mapping is then forced by elimination.

**Counting proof.** There are `(2^r)!` bijections between latent states and an
opaque observation alphabet of the same size. After observing fewer than
`2^r-1` state-symbol pairs, at least two unseen latent states can be swapped
while preserving all evidence. A query distinguishing those states is
unanswerable. Specifying an arbitrary permutation costs
`log2((2^r)!)=Theta(r 2^r)` bits, excluding storage for the opaque symbols.
Thus a general path-independent decoder needs exponential coverage/storage;
the action group alone does not compress an arbitrary sensor.

This bound does not apply to purely on-trajectory control from the anchor when
the learner logs every action and there is no uncontrolled transition: the
action-word parity already gives the current coordinate. It applies to an
observation presented without a trusted path, after an unknown history, or to
path-independent decoding of arbitrary symbols.

## 5. What this gains

Under affine binary-vector observations and full-rank actions, the action
interface replaces
a supplied variable compiler with an exact `r+1`-observation construction. A
mechanism learner can then operate in the recovered coordinate gauge. Combined
with T52, the complete idealized pipeline is:

\[
\text{raw affine observations}
\xrightarrow{\ r+1\ \text{anchor probes}\ }
\text{latent coordinates up to gauge}
\xrightarrow{\ \text{finite-class ERM}\ }
\text{local mechanism wiring}.
\]

The gain comes from paid structure: a stable known-length binary sensor with
known finite-field arithmetic, invertible affine sensing, exact resets, known
action identity, single-coordinate full-rank toggles, noiseless paired probes,
and a stable observation map. It is not evidence that a neural architecture
discovered variables autonomously.

## 6. Noisy and approximate boundary

With bit noise, separately majority-decode every raw bit of the anchor and each
action observation, then subtract the decoded vectors. If every observed bit
flips independently with rate `eta<1/2` and the repeat count `q` is odd, a
Hoeffding/union bound over `r(r+1)` calibration bits gives the sufficient bound

\[
q\ge
\frac{2}{(1-2\eta)^2}
\ln\frac{r(r+1)}{\delta}.
\]

This costs `q(r+1)` observations and `qr` reset/action executions. Majority
voting noisy pairwise XORs instead has effective bias `(1-2eta)^2` and needs
fourth-power dependence. Every noisy future observation also needs repetition
or posterior propagation through the dense inverse; calibration does not make
later inputs noiseless.

This does not handle an ill-conditioned real-valued mixing, action drift,
state-dependent actions, partial resets, nonlinear sensors, or aliasing. Those
are not epsilon details: each changes the identifiability class and must be
proved separately.

## 7. Architecture and prior-art disposition

This construction is an exact control, not an architecture candidate.
Interventional causal representation learning already proves recovery under
general observation transforms with declared interventions; current controlled
world-model theory jointly identifies representations and dynamics under
spectral separation and action-excitation conditions; equivariant world models
already encode transformation algebra.

A neural equivariant encoder trained to satisfy

\[
E(T_i o)=\rho_i E(o)
\]

would be a numerical amortization of this idea. It earns a model claim only if
it removes materially weaker assumptions and beats exact/algebraic, causal-
representation, JEPA, PSR, and recurrent controls at complete cost. The affine
theorem itself selects none.

## 8. Decision and next shift

T54 closes “let actions define the latent variables” as a standalone novelty.
It retains two durable facts:

1. causal coordinates can be grounded by transformation algebra rather than
   surface names; and
2. the exact gain is paid for by excitation rank plus observation-map
   structure.

The next high-leverage seam is no longer initial coordinate recovery. It is
whether a learner can **maintain and revise a reusable mechanism factorization
when only a sparse subset changes**, without assuming an oracle that identifies
the changed mechanism and without globally relearning the encoder/router.

No CPU, local GPU, or rented GPU run is admitted by T54.
