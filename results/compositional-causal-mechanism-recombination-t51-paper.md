# T51 compositional causal mechanism recombination — learn laws once, rewire them

Date: 2026-08-01  
Status: **TRANSPARENT-INTERFACE CONTROL LEMMA; ARCHITECTURE NOT ADMITTED; NO EXPERIMENT**

## 0. Intelligence claim

T45-T50 show that state, intervention, predictive signatures, structural splits,
and prior-ranked searches are each contained by strong classical controls once
their interfaces are supplied. T51 changes the object being learned.

> Do not memorize each environment as an atomic input-output map. Learn a
> reusable library of local causal mechanisms, then identify and compose those
> mechanisms when they appear in a new arrangement.

If real tasks reuse mechanisms combinatorially, this can produce a substantial
capability gain: exact prediction and intervention in combinations never seen
during training, with a short new-environment description. The hypothesis is
not that modularity is new. The research burden is to prove exactly when it
changes sample and description complexity, then determine whether a learned
model can recover the required interfaces from raw observations.

## 1. Transparent finite mechanism world

For one declared input type, let a library contain `M` deterministic
binary-output mechanisms

\[
\mathcal M=\{f_1,\ldots,f_M\},
\qquad f_m:\mathcal U\to\{0,1\}.
\]

An environment is a known typed acyclic composition graph with `r` mechanism
slots. Slot `j` contains unknown identity `m_j in {1,...,M}`. Its parent wiring,
local input type, and the rule for composing node outputs are initially
observable. Only the library identities are hidden.

The learner may set a slot's local parents to legal anchor inputs and observe
its output through independent bit-flip noise `eta<1/2`. A parallel round may
set and observe all `r` slots. This resettable, typed intervention interface is
strong and will not be hidden.

Choose anchor inputs

\[
A=(u_1,\ldots,u_s)
\]

and define each mechanism's interventional code

\[
c_m=(f_m(u_1),\ldots,f_m(u_s))\in\{0,1\}^s.
\]

Assume the codes are distinct, with minimum Hamming distance

\[
d_{min}=\min_{m\ne m'}d_H(c_m,c_{m'})\ge 1.
\]

The identities are therefore defined behaviorally, not by surface names. A
heterogeneous system must be partitioned into typed libraries
`M_tau`, each with its own anchors. Input ports are labeled unless a declared
port-permutation symmetry is included in the behavioral equivalence class. The
full mechanism functions, composition interpreter, and boundary semantics—not
only the codes—are supplied.

## 2. T51.1 — noiseless identifiability and the bit floor

In the noiseless case, executing all `s` anchors identifies every slot by code
lookup. If slots are probed in parallel, this takes `s` environment rounds and
`rs` scalar output observations.

Any binary anchor code capable of distinguishing `M` mechanisms needs

\[
\boxed{s\ge\lceil\log_2 M\rceil.}
\]

**Proof.** There are only `2^s` binary signatures of length `s`. Injectivity of
`m -> c_m` requires `2^s >= M`. QED.

Define the actual nonadaptive separating complexity

\[
s^*(\mathcal M)=\min\{|A|:m\mapsto(f_m(u))_{u\in A}
\text{ is injective}\}.
\]

For a pairwise distinct finite library,

\[
\boxed{
\lceil\log_2 M\rceil\le s^*(\mathcal M)\le M-1.
}
\]

The lower bound need not be achievable. If `f_0` is zero everywhere and each
other `f_i` is one only at a private input `u_i`, identifying `f_0` requires all
`M-1` private anchors. Logarithmic identification therefore requires an
additional balanced-code/separating-family assumption; it does not follow from
counting alone. Adaptive decision-tree complexity has the same bit floor and
can also be `M-1` in the worst case.

## 3. T51.2 — noisy recovery bound

Choose a separating set of size `s` (optimally `s=s*`), repeat each anchor an
odd number `q` of times at every slot, and recover each code bit by majority.
Repeated flips for a fixed slot-anchor pair are independent. Let

\[
\gamma=1/2-\eta>0.
\]

Hoeffding gives a per-bit error bound

\[
P(\widehat c_{j,\ell}\ne c_{m_j,\ell})
\le e^{-2q\gamma^2}.
\]

A union bound over all `rs` recovered bits shows that all `r` mechanism
identities are correct with probability at least `1-delta` whenever

\[
\boxed{
q\ge
\left\lceil\frac{\ln(rs/\delta)}{2\gamma^2}\right\rceil,
\quad q\text{ rounded upward to odd}.
}
\]

The complete adaptation cost is therefore

\[
\boxed{
N_{obs}=rsq,
\qquad
N_{round}=sq
}
\]

when all slots can be probed in parallel. A positive `d_min` also permits
nearest-code correction without recovering every bit, but the conservative
all-bits bound is a sufficient, not minimax, guarantee. Independence across
different bits is not needed by the union bound.

Once the identities are recovered, the known graph composed from the stored
mechanisms predicts every legal noiseless input and intervention, including
global configurations never observed in the new environment. Multi-step claims
require the same mechanisms to remain invariant over time and the composed
transition graph to be correctly typed.

## 4. T51.3 — unrestricted-class/interface contrast

Consider a scalar atomic environment represented only as an arbitrary
deterministic function

\[
g:\{0,1\}^r\to\{0,1\}.
\]

An exact learner with membership-query access needs `2^r` queries in the worst
case.

**Proof.** Let an adversary answer zero on every queried input. If fewer than
`2^r` inputs were queried, choose an unqueried `x*`. Both the all-zero function
and the function that is one only at `x*` are consistent with the transcript,
so exact identification is impossible. QED.

By contrast, when that global function is promised to be the known composition
of `r` directly observable slots drawn from the supplied mechanism library,
T51.1-T51.2 identify the new assignment in `s*(M)` parallel intervention rounds
and `r s*(M)` scalar observations in the noiseless regime. The special
`O(log M)` regime exists only under the balanced separating-code assumption.

The description-length contrast is equally sharp:

\[
L_{composition}\le r\lceil\log_2 M\rceil+L_{graph},
\]

whereas an unrestricted Boolean truth table contains `2^r` bits.

This is an exact assumption-and-interface contrast, not an architecture or
matched sample-complexity separation. The atomic side ranges over all Boolean
functions and receives one scalar global output; the modular side receives a
smaller promised class, `r` local outputs per round, internal parent controls,
the graph, types, full library, anchors, and interpreter. The conditional code
also omits their shared description. A complete ledger must charge
`L(M)`, the typed interface/decoder, graph code if not supplied, encoder/router,
and meta-training, amortized over declared deployments. The strongest learner
receiving the same modular promise can match the lookup exactly.

## 5. T51.4 — no-go without identifiability

The positive theorem fails under any of the following.

### Interventional collision

If two mechanisms agree on every legal intervention,

\[
f_m(u)=f_{m'}(u)\quad\forall u\in\mathcal U_{legal},
\]

no learner can distinguish their identities. They must be treated as one
behavioral equivalence class.

### Missing interface

If raw observations do not reveal which object is a slot, which variables are
its parents, or how actions bind to local inputs, multiple factorizations can
induce the same observed law. Identifiability then requires extra anchors,
intervention targets, sparsity, temporal independence, equivariance, or a
declared equivalence class. A neural encoder cannot create absent information.

### Unconstrained novel composition

If training examples show the mechanisms only in old arrangements but no
assumption requires the same local laws to persist in a new arrangement, two
global predictors can agree on all training data and disagree arbitrarily on
the new composition. Systematic recombination is an inductive assumption, not
a theorem of generic next-token training.

### Mechanism drift and interaction terms

If a local law changes with its global context or unmodeled cross-slot
interactions exist, reusing `f_m` is misspecified. Residual coupling must be
represented, tested, and charged; otherwise modular reuse trades accuracy for
an unreported approximation.

## 6. Unadmitted model contract

The architecture suggested by the positive case is a causal, compositional
mixture of mechanisms:

```text
raw stream
  -> entity/variable slots with stable identity
  -> typed parent and intervention interfaces
  -> behavior-based mechanism identifier
  -> sparse reusable mechanism bank
  -> dynamic composition graph
  -> predicted consequences / policy / targeted new intervention
```

Unlike ordinary token MoE, experts represent local transition laws and can be
rewired into a new graph. Unlike an external prose memory, each stored object is
executable under interventions. The active model cost scales with the
mechanisms in the current composition rather than all stored environments.

The transparent theorem actually supplies the slots, graph, local controls,
mechanism functions, and interpreter; it proves only the behavior-code lookup.
Therefore this diagram is not an architecture candidate. A future candidate
would need a shared learned encoder to recover the decomposition from raw,
renamed observations and a complete-cost edge over matched causal modular and
meta-learning controls.

## 7. Why this could make a model materially smarter

If successful, the result model would improve in four concrete ways.

1. **Novel recombination:** apply known laws in a graph or task arrangement
   absent from training, rather than interpolate among memorized environments.
2. **Rapid causal adaptation:** identify which known laws are present with
   `s*(M)` probes—logarithmic only for a balanced library—instead of relearning
   a global predictor.
3. **Dense knowledge:** store one mechanism once and reuse it across
   combinatorially many compositions.
4. **Continual extension:** add one genuinely new mechanism without changing
   old mechanisms; end-to-end routing interference remains an empirical risk.

These are capability claims, not merely lower perplexity or faster inference.

## 8. Strongest controls and current boundary

Mandatory controls are:

- the exact oracle typed-interface learner in T51;
- modular causal models and causal representation learning with intervention
  targets;
- object-centric and slot-based world models;
- graph neural world models with shared local transition functions;
- sparse MoE with matched experts, routing, and active compute;
- recurrent/meta-RL and full-context Transformer agents with identical
  interventions and persistent state; and
- an atomic learner whose larger hypothesis class and resulting cost are
  explicitly reported, not presented as the sole control.

[Mechanistic World Models](https://arxiv.org/abs/2607.12474) already argues for
reusable explanatory mechanisms as the center of world-model organization.
[Compositional Models for Estimating Causal Effects](https://proceedings.mlr.press/v275/pruthi25a.html)
reports sample-efficiency and unseen-composition benefits from modular causal
models, while [DECAF](https://proceedings.mlr.press/v236/talon24a.html) reuses
causal representations under known intervention targets. Therefore neither
modularity nor causal reuse is a novelty claim. Stronger controls also include
[Variational Causal Dynamics](https://arxiv.org/abs/2206.11131),
[unknown-intervention causal representation learning](https://proceedings.mlr.press/v238/varici24a.html),
[WM3C](https://proceedings.iclr.cc/paper_files/paper/2025/hash/79d86433c2acd12b6fa98553435d226e-Abstract-Conference.html),
and Dreamweaver-style compositional world models from pixels. T51 remains below
that raw-learning frontier.

## 9. Pre-run fatal tests

The first condition is already true of T51, so architecture admission is closed.
For any descendant candidate, close without implementation if:

1. the learned system needs supplied slots, parent graph, or intervention
   targets at evaluation time;
2. its behavior signature is merely supervised expert classification under
   names or features unavailable after renaming;
3. a matched graph world model or causal modular network learns the same
   compositions with no more data or active compute;
4. apparent transfer disappears when residual cross-mechanism interactions are
   introduced;
5. storage savings vanish after the encoder, graph, router, mechanism bank,
   replay buffer, and meta-training corpus are counted;
6. the candidate cannot extrapolate to more slots or unseen wiring than used in
   training; or
7. gains stay confined to binary truth-table worlds and do not survive at
   least one continuous and one partially observed family.

## 10. Disposition

Retain T51 as a transparent-interface noisy decoding control. Its library is
not learned, its interfaces are privileged, and its exponential contrast is
unmatched. T52 separately examines finite hidden-wiring recovery:

> [T52 hidden-wiring causal recombination](hidden-wiring-causal-recombination-t52-paper.md)

No CPU, local GPU, or rented GPU run is admitted by T51.
