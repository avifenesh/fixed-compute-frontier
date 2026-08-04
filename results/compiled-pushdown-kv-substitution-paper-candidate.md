# Compiled top-k pushdown KV substitution: paper candidate

Date: 2026-08-01  
Status: **k=1 REJECTED; k<=3 REVISED FOR RE-AUDIT; NO CPU OR GPU YET**

## Decision reset

The original proposal compiled a weighted/nondeterministic training-time stack
to one deterministic served path.  That version is rejected.

The decisive evidence is internal to the motivating result: Stack Attention's
deterministic/superposition model was worse than its Transformer control on
Penn Treebank (98.67 versus 92.84 perplexity), while only the nondeterministic
model improved it (88.54).  The supported mechanism is therefore preservation
of multiple latent stack configurations, not selection of one.  This also has
an algebraic obstruction: nondeterministic pushdown automata cannot in general
be determinized because deterministic context-free languages are a strict
subset of context-free languages.

The only surviving candidate is deliberately bounded:

> Replace one complete GQA KV group by one persistent pushdown residual that
> serves at most three prefix-consistent, future-distinguishable stack
> configurations.  Train an exactly specified 32-configuration causal
> weighted beam, then export an online top-k beam.  Close the lane if k<=3 does
> not retain the training model's capability gain.

The compiler is not claimed to create a better state than the model.  Its only
possible advantage is spending extra training compute to discover a compact
latent transition policy, while the served model preserves the few competing
states that actually matter.  If useful posterior mass is not sparse, the
compiler loses information and the proposal fails.

## 1. Served operator

Let a complete configuration be

\[
c=(q,p,\ell),
\]

where `q` is finite control state, `p` points to the top of a complete
persistent stack, and `ell` is its log weight.  A node contains a parent
pointer, a discrete symbol, and a pointer to a payload produced at its source
token.  Buried stack contents are part of configuration identity: two states
with equal current tops but different ancestors cannot be merged because a
future pop can distinguish them.

At token `t`, expand every configuration in the previous beam by the finite
transition set, score the candidates causally, merge only byte-identical
configurations, and retain the top `k<=3`:

\[
\mathcal B_t=
\operatorname{TopK}_k\left(
\bigcup_{c\in\mathcal B_{t-1}}\operatorname{Expand}(c,x_t)
\right).
\]

Weights are renormalized inside the retained beam.  The read is

\[
\hat r_t^{(k)}=
\sum_{c\in\mathcal B_t}\hat P_t(c)v(top(c)),
\qquad
y_t=W_o\hat r_t^{(k)}.
\]

For a concrete ledger, use one stack expert, payload width `d_s=64`, three
control states, three stack symbols plus a root symbol, and transitions
consisting of next-state times `{push(symbol), pop, noop}`.  A shared 15-logit
projection from `x_t` plus a `(current-state, top-symbol)` bias table scores
transitions.  The pushed payload `W_v x_t` is shared across all configurations
created at that token.  There is no Q projection, K projection, query-key
product, softmax over history, or global stack read.

The complete transition table is:

| Operation | Count per input config | New top | New control | Allocation | Root behavior |
|---|---:|---|---|---|---|
| `push(gamma,q')` | 9 | new node `(parent=p,symbol=gamma,payload=t)` | `q'` | one only if retained | valid |
| `pop(q')` | 3 | `parent[p]` | `q'` | none | masked invalid at root |
| `noop(q')` | 3 | `p` | `q'` | none | valid |

There is no implicit `replace`; it would be a pop followed by push and is not
in this candidate.  Candidate weights are computed in probability space as
`beam_weight * softmax(action_logits)`.  Candidates with identical
`(new-control, complete-new-stack)` keys are summed before top-k.  Prospective
pushes use virtual `(parent,symbol,payload-token)` keys and allocate nodes only
after selection, so rejected branches consume no persistent state.

## 2. Exact attention exchange and conservative state ledger

For model width `d`, query-head width `h`, and grouped-query ratio
`g=H_q/H_kv`, removing one complete KV group removes

\[
C_{group}=2dh(g+1)
\]

projection coefficients and dense MACs per token, before the
history-dependent attention read.  It removes

\[
B_{KV}=2h b_{kv}
\]

mutable cache bytes per token.  At `g=4`, `h=128`, and BF16 KV, the exchange is
`1280d` projection MACs/coefficients and 512 cache bytes/token.

The concrete stack's dense interfaces cost

\[
C_{stack-proj}=d(2d_s+15)=143d
\]

MACs and coefficients, plus a 180-entry transition-bias table.  This count
includes value, output, action, symbol, and controller-transition logits.  It
does not hide the top-k work: at `k=3`, the runtime also scores and selects at
most 45 scalar transition candidates, merges duplicates, normalizes three
weights, gathers three payloads, forms the weighted read, handles invalid pops,
and updates pointers.

Reserve `9d` operations/token for the bounded branch machinery, and use the
rest of the dense budget for a bias-free SwiGLU residual of width

\[
r_s=376.
\]

The served dense total is then `143d+3(376)d=1271d`, versus the removed
`1280d`.  The strongest group-removed control spends the same parameter ceiling
on a width-426 SwiGLU (`1278d`).  The stack therefore costs 50 matched FFN
neurons and has fewer, not more, parameters.  The 180-entry bias table fits the
remaining parameter slack for every admitted `d>=512`.  Exact parameter
equality is not required; every candidate resource coordinate must be less
than or equal to baseline.

The persistent state uses a preallocated flat arena, not a general allocator.
One aligned 16-byte node is exactly:

| Field | Type | Bytes |
|---|---|---:|
| parent node index | `uint32` | 4 |
| payload-token index | `uint32` | 4 |
| stack symbol | `uint8` | 1 |
| flags | `uint8` | 1 |
| reserved/generation | `uint16` | 2 |
| alignment reserve | bytes | 4 |

Finite control and probability belong to the current beam, not every historic
node.  Each of its three aligned entries is 16 bytes: top `uint32`, control and
validity bytes, padding, FP32 probability, and one reserved word.  A bump
counter and root record add bounded per-sequence state.

A token contributes one shared BF16 payload (`2d_s=128` bytes) and, in the
worst all-push case, at most `k` retained nodes:

\[
B_{stack/token}\le128+16k.
\]

At `k=3`, this is 176 bytes/token, leaving 336 bytes against the removed 512.
The arena has at most `kL+1` nodes for declared maximum context `L`; 32-bit
indices require `kL+1<2^32`.  This is a large but bounded machine, not a literal
unbounded PDA.

The exact reference branch routine for `k=3` expands 45 candidates, uses three
15-way softmaxes, compares all 990 unordered candidate pairs for identical
128-bit virtual keys, aggregates duplicates, and performs three fixed 45-entry
argmax scans.  A key uses two 64-bit words and losslessly encodes new control,
new top or prospective `(parent,symbol,payload-token)`, and operation kind under
the declared 32-bit arena bounds.  Its worst-case non-dense work is:

- 45 bias additions, 42 max comparisons, 45 subtractions, 45 exponentials,
  42 sum additions, 3 reciprocals, and 90 multiplications for action weights;
- 1,980 64-bit equality comparisons and at most 44 duplicate-weight additions;
- at most 135 selection comparisons and 45 key constructions;
- 192 multiplies and 128 additions for the three-way FP32 weighted payload
  read, plus at most three node writes and three beam-entry writes.

Counting an exponential or reciprocal as 16 scalar-operation equivalents gives
fewer than 3,700 scalar equivalents, below the reserved `9d>=4608` for
`d>=512`.  This abstract count does **not** prove latency equality across mixed
GPU operations.  The removed attention work, reported separately rather than
hidden in the projection exchange, is `2gTh` QK/AV MACs at cached length `T`,
plus `gT`-scale softmax/reduction operations, a 512-byte/token cache write, and
at least `2hTb_kv` logical K/V reads.  Added logical state writes are at most
176 bytes/token; the beam table, node headers, and at most three 128-byte
payload gathers must be measured at the exact target shapes.  RoPE work removed
with the Q/K dimensions and SwiGLU activation/multiply work on both candidate
and controls remain separate ledger rows.

Thus parameters, dense MACs, and allocated long-lived state are already below
the baseline ceiling.  Kernel traffic and latency remain unknown and cannot be
converted from this operation count.

The replacement removes the whole GQA group.  Removing one query head alone
does not save grouped K/V state.

## 3. What is and is not gained algebraically

In the abstract machine model, an unbounded deterministic stack strictly
extends bounded-precision fixed-state recurrence on deterministic nested
families, and nondeterminism extends the recognized language class further.
The proposed implementation has a finite context cap and a fixed `k` beam, so
it claims neither literal unboundedness nor general context-free recognition.

The proposed gain is narrower: natural or code prefixes may induce a sparse
posterior over complete hierarchical states even when one state is
insufficient.  Persistent nodes then store only the causally relevant
hierarchy instead of dense K/V for every token, allowing the removed attention
budget to fund both that state and nearly a full residual MLP.

This is a hypothesis, not a theorem.  A full Transformer can implement bounded
stack behavior; equal-byte local attention, recurrence/SSMs, depth memory, and
the fully reinvested MLP are mandatory controls.

## 4. Conditional top-k approximation theorem

Let the frozen 32-configuration training model at a fixed prefix define its
exact finite distribution `P_t` over complete configurations, with read

\[
r_t=\sum_cP_t(c)v(c),\qquad \|v(c)\|\le B.
\]

For a prefix-consistent retained set `B_t` whose exact mass is
`1-delta_t`, renormalize exact weights within that set.  Then

\[
\|r_t-\hat r_t^{(k)}\|\le2B\delta_t.
\]

This local statement is exact but insufficient for a recurrent model.  If

\[
h_{t+1}=F(h_t,r_t,x_t)
\]

is `L_h`-Lipschitz in hidden state and `L_r`-Lipschitz in stack read, the
rollout error obeys

\[
e_{t+1}\le L_h e_t+2L_rB\delta_t+\epsilon_t,
\]

and hence

\[
e_t\le
\sum_{s<t}L_h^{t-1-s}(2L_rB\delta_s+\epsilon_s).
\]

Thus even small local discarded mass can compound when `L_h>=1`.  Posterior
concentration under a teacher-forced soft model cannot by itself prove that the
independently rolled-out top-k model remains close, because changed hidden
states change future routing logits.  The CPU gate must measure both:

1. teacher-forced full-configuration retained mass and local read error; and
2. free recurrent-rollout sequence KL and retained task gain.

No final-Viterbi path, local action entropy, or equality of current top symbols
can substitute for those measurements.

## 5. Prior evidence and why it is not this claim

[Stack Attention](https://arxiv.org/abs/2310.01749) replaces attention with
differentiable stacks and reports a 4.3-point Penn Treebank perplexity advantage
for its nondeterministic variant under a constrained parameter budget.  Its
deterministic/superposition result is worse than the Transformer control, and
its table selects the best validation model from 20 restarts.  Its soft
algorithms also retain quadratic or cubic training work.  This motivates a
multi-configuration test; it does not establish that three configurations are
enough or that a served fixed-cost version wins.

[StackTrans](https://arxiv.org/abs/2507.15343) reports natural and formal gains,
including matched-20B-token validation-loss improvements, but adds a global
soft stack, increases inference time and memory, truncates depth, and uses a
parallel approximation that breaks temporal stack dependencies between
tokens.  It does not satisfy this substitution contract.

Older neural pushdown work already trains soft stacks and extracts discrete
automata.  Neither stack augmentation, weighted pushdown computation,
persistent stacks, nor soft-to-hard compilation is claimed as novel alone.
The only research claim left is the conjunction: a useful 32-configuration
training model whose state is sparse enough to compile to three configurations,
plus a complete fixed-cost KV-group substitution.  It is not a claim about
faithfully approximating an infinite NPDA posterior.

## 6. Breakthrough-size end gate

The candidate is not interesting if it wins only formal languages or saves a
few percent.  A from-zero language-model result must eventually satisfy all:

1. no increase in parameters, peak served state, training tokens, decode
   traffic/latency envelope, or served token count;
2. general held-out NLL noninferiority within 0.01;
3. at least 5 absolute points on a frozen aggregate of natural code,
   structured-output, compositional, and recursive-reasoning slices;
4. no material regression on knowledge, retrieval, or nonhierarchical
   reasoning; and
5. capability competitive with a dense control at least 2x its parameter count
   under the same data and evaluation contract.

Extra training compute is the allowed alternative currency and must be
reported.  These are admission thresholds, not predicted outcomes.

## 7. Strongest controls

- unchanged matched GQA;
- the group removed with no replacement;
- group removed and the full budget reinvested in width-426 SwiGLU;
- the proposed stack plus width-376 SwiGLU;
- the exact defined `S32` training model, online `k=1,2,3`, and oracle top-k
  beams;
- equal-byte local attention and recurrent/SSM replacements;
- matched deterministic stack, queue, content-addressed cache, and residual
  depth-memory controls; and
- identical training compiler loss with no served stack.

All trainable comparisons report every seed.  Best-of-N selection is not an
admissible estimate.

## 8. Admission order

No GPU run is authorized.

1. Freeze a CPU acquisition preregistration on ambiguous, not merely
   deterministic, context-free sources.
2. Compare the exact finite `S32` reference with online prefix-consistent
   `k=1,2,3`; measure complete-configuration mass, local read error, recurrent
   sequence KL, and retained next-token/task gain.
3. Close the lane if `k<=3` cannot retain the nondeterministic gain.  Increasing
   `k`, widening payloads, changing thresholds, or switching grammars after
   seeing the result creates a new lane and does not rescue this one.
4. Only an acquisition pass authorizes a target-kernel test.  Before any such
   run, sample the local GPU for 60 seconds; use it only if it is genuinely
   idle and has at least 24 GiB free, otherwise rent a new isolated H100 without
   touching existing instances.
5. Natural-model training remains last.
