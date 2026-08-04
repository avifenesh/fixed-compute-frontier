# Independent audit: action algebra as latent coordinates (T54)

Date: 2026-08-01  
Scope: mathematical and architecture audit; no experiment  
Audited artifact: action-algebra-latent-coordinate-t54.md

## Verdict

**HOLD WITH CORRECTIONS as an exact affine control construction. Architecture novelty remains rejected.**

T54.1 is correct, T54.2 needs one degenerate-case repair, and T54.3 is correct only for path-independent exact decoding of arbitrary opaque observations. The noisy rate is achievable, but only through separate denoising of anchor and action observations and with an additional cost for every noisy future state.

## 1. T54.1 gauge algebra and cost

With

\[
o_0=A_ez_0+b_e,
\qquad
o_i=A_e(z_0+e_{\pi(i)})+b_e,
\]

subtraction over \(\mathbb F_2\) gives

\[
d_i=A_ee_{\pi(i)},
\qquad
D=A_eP_\pi.
\]

Since both factors are invertible,

\[
D^{-1}(o-o_0)
=P_\pi^{-1}(z-z_0).
\]

The recovered representation is exactly the latent state relative to the reset anchor, in stable action-label coordinates. The unavoidable gauges are the unknown origin and the permutation relating action labels to any externally named latent coordinates.

The stated algebraic costs are correct in the classical dense model:

- \(r+1\) distinct observations and \(r\) action transitions;
- \(r^2+r\) stored calibration bits for \(D\) and \(o_0\);
- \(O(r^3)\) binary field operations for Gaussian-elimination inversion; and
- \(O(r^2)\) bit operations per dense matrix-vector decode.

The interaction bill must additionally count \(r\) exact resets to the same \(z_0\), action execution, and observation acquisition. “Raw affine observations” are already a privileged interface: the sensor emits a known-length binary vector with stable bit positions and known \(\mathbb F_2\) arithmetic. This is not raw pixels, text, or an unknown symbol stream.

## 2. T54.2 excitation-rank construction

Let \(W=\operatorname{span}(B)\) with \(\dim W=d<r\). Except for one degenerate case, a nonidentity \(S\in GL(r,2)\) that fixes \(W\) pointwise does exist.

- If \(W\ne\{0\}\), choose nonzero \(w\in W\) and a nonzero linear functional \(\ell\) that vanishes on \(W\). Then
  \[
  Sx=x+\ell(x)w
  \]
  is an invertible transvection, fixes \(W\), and is nontrivial.
- If \(W=\{0\}\) and \(r\ge2\), any nontrivial elementary shear works.
- If \(r=1\) and \(B=0\), \(GL(1,2)\) contains only the identity, so the stated linear construction fails.

The degenerate case is still non-identifiable because the model is affine. The translation \(z'=z+1\), together with \(b'_e=b_e+A_e\), preserves observations and zero action effects. T54 should either exclude \((r,d)=(1,0)\) from the linear-\(S\) claim or formulate the gauge in the affine group from the start.

For valid \(S\),

\[
A_eS^{-1}(Sz)=A_ez,
\qquad
SB=B,
\]

so every action word has the same observed effect. The exact conclusion is that the latent factorization is identifiable only modulo the stabilizer

\[
\operatorname{Stab}(B)=\{S\in GL(r,2):SB=B\},
\]

plus affine-origin gauge. It is slightly too broad to say that the entire complement is unidentifiable: only queries that are not invariant under this stabilizer are impossible. T54's following sentence already states this narrower, correct boundary.

Full action rank is sufficient for recovering every linear coordinate. It is not necessary for a deployment query that depends only on the excited subspace or is stabilizer-invariant.

## 3. T54.3 counting and the action-path caveat

Let \(N=2^r\). For an opaque alphabet of size \(N\), there are \(N!\) arbitrary bijections \(\phi_e\). After observing at most \(N-2\) state-symbol pairs, two unseen states can be swapped without changing the transcript, so exact path-independent decoding of every possible symbol needs at least \(N-1\) mapped pairs in the worst case. The final mapping is then determined by elimination.

Thus the exponential coverage claim is correct. The exact storage-description bill is stronger than “\(2^r\) symbols”: specifying an arbitrary permutation costs

\[
\log_2(N!)=\Theta(N\log N)=\Theta(r2^r)
\]

bits, modulo the representation cost of the opaque symbols themselves.

The action-path qualification is load-bearing. If the learner starts at the anchor, logs every action, and no exogenous transition occurs, it always knows the parity coordinate of its current state without decoding \(\phi_e\). The exponential lower bound applies when an observation is presented without a trusted path, after an unknown history, or when arbitrary previously unseen symbols must be decoded. It does not apply to purely on-trajectory control that can retain the action history.

Reset access, path logging, known group relations, stable action labels, and the absence of uncontrolled dynamics are therefore part of the oracle. If the desired query is invariant under an observation-symbol quotient, the full permutation need not be learned.

## 4. Noisy bound

The claimed

\[
q=O\!\left(
\frac{\log(r^2/\delta)}
{(1-2\eta)^2}
\right)
\]

repetitions per calibration probe is valid if the learner separately majority-decodes every bit of \(o_0\) and every \(o_i\), then forms \(d_i=\widehat o_i-\widehat o_0\). For odd \(q\), Hoeffding and a union bound over \(r(r+1)\) raw calibration bits give the sufficient condition

\[
q\ge
\frac{2}{(1-2\eta)^2}
\ln\frac{r(r+1)}{\delta}.
\]

This costs \(q(r+1)\) observation acquisitions and \(qr\) resets/action executions.

If instead the learner XORs two noisy observations and majority-votes the differences, the effective flip probability is \(2\eta(1-\eta)\), whose bias is \((1-2\eta)^2\). That procedure needs fourth-power dependence \(O((1-2\eta)^{-4})\). The paper must specify the separate-denoising procedure.

Calibration alone does not denoise future observations. Every future \(o\) must also be repeatedly observed and majority-decoded, or the decoder must propagate a posterior through \(D^{-1}\). A single sensor-bit error can affect many recovered coordinates under a dense inverse. Any complete noisy cost claim must include the number of future states decoded.

## 5. Oracle and prior-art fairness

T54 pays for its result with:

- exact reset to a fixed hidden anchor;
- full access to every basis action;
- known commuting-involution action algebra and stable action identities;
- deterministic, state-independent, full-rank effects;
- invertible affine binary sensing of known dimension;
- paired one-step observations; and
- no drift, aliasing, confounding, or uncontrolled transition.

Every architecture control must receive the identical traces and group-law information. The exact \(D^{-1}\) learner is mandatory; a neural encoder merely amortizes it unless it weakens the sensing, reset, or action assumptions at matched cost.

The novelty rejection is correct. [Score-based causal representation learning](https://www.jmlr.org/papers/v26/24-0194.html) already treats latent causal recovery under unknown general observation transformations and interventions. [Controlled world-model identifiability](https://arxiv.org/abs/2607.22430) makes representation and transition recovery depend explicitly on predictable-signal and action-excitation margins. Equivariant representation and world-model work also treats transformation algebra as supervision. T54 is a particularly transparent finite-field control, not a new architecture principle.

## 6. Architecture disposition

No architecture claim survives. T54 establishes:

1. exact coordinate recovery when action effects form a basis and sensing is affine;
2. recovery only modulo the action stabilizer when excitation is deficient; and
3. exponential worst-case decoding for arbitrary path-independent opaque sensors.

These are useful admission controls. They do not select a Transformer, GNN, equivariant encoder, causal MoE, or recurrent system. The proposed shift toward sparse mechanism revision remains paper-stage only; T54 supplies no evidence that a learned model can localize changes without an oracle or avoid global encoder/router drift.

## Exact fixes

1. Count resets and action executions in T54.1, and stop calling the binary affine vector fully raw.
2. Repair T54.2 with the stabilizer construction and the \(r=1,B=0\) affine-translation exception.
3. State identifiability modulo \(\operatorname{Stab}(B)\), not blanket loss of the whole complement.
4. In T54.3, state the exact \(2^r-1\) pair lower bound and \(\Theta(r2^r)\)-bit permutation description.
5. Make the path-independent deployment condition explicit.
6. Specify separate raw-bit majority decoding, odd \(q\), and future-observation denoising.
7. Preserve exact algebraic, causal-representation, controlled-world-model, PSR, and recurrent controls and retain the no-run decision.
