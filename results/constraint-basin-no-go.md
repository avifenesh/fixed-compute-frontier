# Constraint-basin completion — closed at algebra gate

Status: **closed before GPU, implementation training, or candidate numbering**  
Date: 2026-07-26  
Candidate number: **none**

## Outcome and claim boundary

The proposed fixed-step constraint-basin model does not expose a new capability
algebra. It is an ordinary tied recurrent hypergraph update written as energy
descent. For the specified log-sum-exp prototype energy, one unit step is a
softmax attention read; on parity factors it is message passing and, at binary
states, threshold bit-flipping.

No GPU was rented. The branch is closed because its claimed architectural edge
is unreachable, not because optimization failed.

This result closes only the claim of a **distinct best-achievable capability
algebra**. It does not establish universal resource dominance over every basin
implementation. A finite-training, stability, or node-state-versus-edge-state
decoder trade remains untested and is explicitly not claimed here.

## Frozen proposal

The considered state was `x in [-1,1]^N`, initialized from a corrupted cue
`x0`, with sparse parity factors `H` and a fixed number `S` of updates:

\[
E(x;x_0,H)=\frac12\|x-x_0\|^2
+\frac{\lambda}{2}\sum_a\left(1-\prod_{i\in N(a)}x_i\right),
\qquad
x^{s+1}=\operatorname{clip}(x^s-\eta_s\nabla E).
\]

`H` is visible input, not learned model knowledge. The intended claim was more
structured correction per persistent rule byte at fixed state and fixed typed
operations. Mandatory controls were exact MAP/direct algebraic decoding and a
generic recurrent graph update with the same messages, state, and steps.

## Fatal inclusion proof

For variable `i`, direct differentiation gives

\[
\nabla_i E=x_i-x_{0,i}
-\frac{\lambda}{2}\sum_{a\ni i}\prod_{j\in a\setminus i}x_j,
\]

and therefore

\[
x_i^{s+1}=\operatorname{clip}\left(
(1-\eta_s)x_i^s+\eta_s x_{0,i}
+\frac{\eta_s\lambda}{2}\sum_{a\ni i}m_{a\to i}^s
\right),
\quad
m_{a\to i}^s=\prod_{j\in a\setminus i}x_j.
\]

This is a factor-to-variable message followed by a tied recurrent node update.
Let `R` be the family of generic local recurrent maps with the same visible
inputs, state, arithmetic, and `S` steps. Choosing its update map `Phi` to be
the equation above reproduces every basin trajectory exactly, with no extra
state or operations. Thus the basin family is a subset of `R`:

\[
\mathcal B\subseteq\mathcal R
\quad\Longrightarrow\quad
Q^*_{\mathcal R}\ge Q^*_{\mathcal B}.
\]

A basin may be a useful inductive bias for a finite optimizer. It cannot have a
strict best-achievable capability advantage over this mandatory control. That
would be a different, training-protocol claim rather than the requested
fixed-serving architecture frontier.

## Two additional collapses

At the initial binary state `x=x0`, let `d_i` be the number of checks incident
to `i`, `u_i` their unsatisfied count, and `c=eta*lambda/2`. Then

\[
x_i^+=\operatorname{clip}\left[x_i(1+c(d_i-2u_i))\right].
\]

Its sign flips exactly when

\[
u_i>\frac{d_i+1/c}{2},
\]

which is a parallel threshold bit-flipping decoder. Interior values add analog
confidence, but not a different computational primitive.

For prototype patterns `p_m`, the standard smooth attractor energy

\[
E(x)=\frac12\|x\|^2-
\frac1\beta\log\sum_m\exp(\beta p_m^\top x)
\]

has gradient `x - sum_m softmax(beta p_m^T x) p_m`. One unit gradient step is
therefore exactly a softmax attention memory read. Replacing parity factors
with learned prototypes returns to attention rather than opening another
algebraic path.

## Direct ceiling

Under exact source-recovery loss, MAP is Bayes-optimal for the same visible
observation. A candidate cannot exceed it; it can only seek a cheaper resource
point. For the concrete Hamming(7,4) bounded-corruption panel, the syndrome
directly identifies a clean or single-flipped bit. The zero-parameter direct
decoder and exhaustive MAP both recover all `16 * 8 = 128` episodes exactly.

The sparse factor graph also does not store 16 independent memories. If
`rank_GF(2)(H)=r`, it defines `2^(N-r)` valid codewords; the cue supplies which
one is intended. Count the actual encoded graph bytes, dynamic state, messages,
factor products, arithmetic precision, and traffic—not the number of implicit
codewords—as the resource ledger.

## Executed evidence

The deterministic CPU gate checks:

- prototype-energy steps against an independently finite-differenced gradient;
- direct parity-energy steps against independently written factor messages;
- the binary threshold identity over every 7-bit state at three couplings;
- every clean and single-flip Hamming(7,4) episode against syndrome and MAP.

The frozen negative-control panel uses the 21-bit Hamming parity matrix (12
edges, 3 packed logical bytes), all 16 codewords, and clean plus each of seven
single-bit corruptions. The JSON records the arithmetic mode, deterministic
seeds, execution command, and SHA-256 hashes of the executable and focused
test. Those hashes establish reproducibility, not a physical resource claim.

No candidate/control byte or issued-operation comparison is reported because
no resource-dominance result is claimed. Promoting the retained decoder trade
would require a new preregistration with exact encoded-graph bytes, immutable
cue and ping-pong state, factor workspace, precision, traffic, issued
operations, and matched direct, bit-flipping, min-sum, and recurrent controls.

Artifacts:

- [Executable no-go gate](../experiments/constraint_basin_no_go.py)
- [Machine-readable result](constraint-basin-no-go.json)
- [Focused tests](../tests/test_constraint_basin_no_go.py)

## Decision

Do not train or rent a GPU for the closed algebra claim. Retain only the narrow
engineering question—whether analog node confidence approaches per-edge belief
propagation while using less dynamic decoder state. That is a decoder tradeoff,
not evidence for a generally smarter fixed-cost language model, so it is not
candidate 004.
