# T83 explicit operator interfaces and active coverage — paper screen

Date: 2026-08-02  
Status: **INDEPENDENTLY AUDITED; CLOSED ON PAPER; NO RUN**

## 0. Decision before implementation

The screened system would execute a recurrent latent program

\[
h_{t+1}=F_{z_t}\!\left(G_{r_t}(h_{\le t},x)\right),
\]

where `z_t` selects a learned reasoning operator and `r_t` selects the source,
destination, and argument interface used at that step. Compound supervised
traces would first expose reusable operations. Outcome-reward RL would then
train new compositions, while a closed-loop curriculum preferentially selects
problems whose learned operator interfaces remain under-identified.

This is a coherent architecture, but it is not presently a new architectural
breakthrough. Learned discrete primitives, recurrent latent execution,
variable depth, routing, modular reasoning, and off-support composition are all
occupied blocks. The only narrow remainder is:

> estimate which *learned* operator interfaces lack evidence and actively buy
> training episodes that identify them, without using the generator's true
> operator labels.

That remainder is a curriculum and identification claim. It cannot be called a
smarter-model result unless the explicit architecture also beats a dense
recurrent model trained on the exact same selected examples.

## 1. What a real gain would buy

The target is not a smaller proxy loss. It is systematic reuse of a learned
operation in a composition absent from training, including greater reasoning
depth and reuse of older intermediate values. A successful model would require
fewer rewarded training episodes to reach fixed held-out compositional
competence and would retain that competence under operator renaming and new
composition graphs.

The desired algebraic edge is that `M` reusable operators can express up to
`M^L` length-`L` sequences without storing one independent solution per
sequence. This is a representation possibility, not yet a sample-complexity
advantage over a Transformer: a sufficiently capable dense recurrent network
can emulate the same machine. Any strict advantage must therefore come from a
declared optimization, locality, sparsity, or data-coverage condition and must
survive a same-data dense control.

## 2. Direct collision boundary

- [Learning to Theorize / NEO](https://arxiv.org/abs/2605.03413) already learns
  a discrete primitive vocabulary and latent executable programs from raw
  observation pairs. It reports order-one compositional and length-OOD gains,
  but uses a fixed codebook and short deterministic synthetic programs, costs
  roughly twice single-pass training, and can use expensive test-time sampling.
- [From Reasoning Traces to Reusable
  Modules](https://arxiv.org/abs/2606.18089) gives the most direct collision:
  compound SFT material becomes reusable atomic skills and routing mechanisms,
  while RL produces large gains on unseen compositions. It explicitly leaves
  adaptive coverage of under-explored local interfaces as future work.
- [Recursive Latent Reasoning](https://arxiv.org/abs/2510.14095) combines
  recurrence, algorithmic supervision, a discrete latent bottleneck, and error
  correction for length generalization.
- [Depth-Recurrent
  Transformers](https://arxiv.org/abs/2603.21676) decouple parameter count from
  reasoning depth with a shared recurrent latent block.
- [Curriculum II](https://arxiv.org/abs/2606.27721) already supplies a recursive
  compositional curriculum and a subpolynomial supervision result for its
  semiautomata setting.

Therefore the relevant controls are not ordinary SFT or a one-pass Transformer.
They are compound SFT plus off-support RL, NEO-like discrete execution, dense
depth recurrence, fixed MoE recurrence, and adaptive difficulty curricula.

## 3. The only clean conditional advantage

Let `E` be a finite set of reachable, reward-relevant interfaces. Suppose a
candidate problem exposes an identifiable learned interface signature for scan
cost `c_s`, an update plus verified reward costs an additional `c_r`, every
interface has pool frequency at least `pi_min`, and `n` local witnessed updates
are sufficient for each interface. These are strong assumptions: in particular,
collapsed or undiscovered operators do not have a reliable signature.

For uniform sampling, a conservative simultaneous coverage certificate follows
from a binomial lower-tail bound. With `L=ln(E/delta)`,

\[
T_{\rm uniform}\ge \frac{2n+8L}{\pi_{\min}}
\]

ensures at least `n` occurrences of every interface with probability at least
`1-delta`. A selector allowed to inspect learned signatures can spend rewarded
updates only on deficits, so it needs at most

\[
T_{\rm update,active}=nE
\]

rewarded updates after obtaining coverage. If both procedures scan the same
`S` candidates, their charged costs are

\[
C_{\rm uniform}=S(c_s+c_r),\qquad
C_{\rm active}=Sc_s+nEc_r.
\]

The relative saving is therefore

\[
1-\frac{C_{\rm active}}{C_{\rm uniform}}
=\frac{(S-nE)c_r}{S(c_s+c_r)}.
\]

This can exceed `20%` when verified updates are expensive, interfaces are
imbalanced, and learned signatures are valid. It proves no architectural edge:
the dense same-data control receives those same `nE` selected episodes. It also
does not remove the scan cost, proposal-support assumption, or circularity of
recognizing a missing interface.

## 4. Why operator birth is not added

An undiscovered operator cannot be targeted by its operator ID. If a proposer
assigns the required operation zero mass, no curriculum can recover it. More
generally, among `N` observationally indistinguishable candidates, evaluating
only `B` candidates has worst-case hit probability at most `B/N`; informative
interventions must supply and charge the missing bits.

The exact XOR-fingerprint construction considered during this screen illustrates
the boundary. For one unknown pairwise parity among `d=64` coordinates, `18`
random binary interventions plus `7` fresh validations identify the pair at a
one-percent proposal and admission error allocation. Including the triggering
failure gives `26` interactions versus `2,016` exhaustive candidates, a
`98.7%` reduction. But the construction is exactly finite-field sparse recovery
or group testing. The strongest matched control ties it, so its research gain
is zero. It is retained only as a mandatory formation control.

## 5. Frozen factorial test, if a run is later admitted

The smallest informative matrix crosses architecture and curriculum:

| Architecture | Uniform/off-support RL | Interface-targeted RL |
|---|---:|---:|
| Dense depth-recurrent Transformer | required | required, same selected data |
| NEO-like discrete shared executor | required | optional diagnostic |
| Fixed matched MoE recurrent model | required | optional diagnostic |
| Explicit operator/interface recurrence | required | required candidate |

Every cell receives identical compound traces, final rewards, parameter budget,
active compute, recurrence steps, rollout budget, and teacher calls. If step
boundaries, argument wiring, or intermediate labels are exposed, every cell
receives them. The selector may use only learned routes, predictions, and
rewards—not the data generator's true composition signature.

Three non-isomorphic CPU domains are required: non-commuting string operators,
graph/data-flow programs that reuse old intermediates, and stack or arithmetic
programs. Train compositions and operator names are separated from held-out
interfaces, deeper programs, renamed operators, and new graph shapes.

## 6. Twenty-percent admission and kill gate

The primary denominator is rewarded episodes to a frozen held-out competence
target, including all scanned examples and model compute in the accompanying
cost ledger. A run cannot earn GPU/model integration unless the simultaneous
confidence bounds establish both:

1. explicit recurrence plus targeted curriculum uses at least `20%` less
   complete acquisition cost than explicit recurrence plus uniform/off-support
   RL; and
2. it uses at least `20%` less complete acquisition cost than dense recurrence
   trained on the exact same selected examples, or reaches at least `20%` lower
   held-out compositional error at matched complete cost.

The result must hold across all three algebra families, operator renaming, and
new composition graphs without a material in-distribution regression. Passing
only the first comparison is data selection. Passing only against SFT repeats
the Reusable Modules result. Route interpretability alone is engineering.

A point estimate above `20%` whose frozen lower confidence bound does not reach
`20%` does not pass. A result from `10%` to less than `20%` is arguable but not
successful; it may earn one bounded replication only when the confidence design
can plausibly cross the gate. A single-digit result closes the lane.

## 7. Audited decision

No architecture claim is admitted. The active-interface selector has a valid
conditional cost advantage but is downstream of learned signatures and is
primarily a curriculum mechanism. A cheap CPU test would be a kill test, not a
breakthrough demonstration. The second independent audit concludes that even a
clean toy pass would not change the model-level decision beyond NEO and Reusable
Modules and therefore votes `NO-RUN`. Reopening requires raw-trace interface
identifiability, adaptive coverage against the strongest oracle/static policy,
and a coverage-to-capability theorem before code. See the
[independent audit](explicit-interface-curriculum-t83-independent-audit.md).
No local GPU or rented instance is authorized.
