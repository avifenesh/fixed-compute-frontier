# Independent audit — T61 future-value workspace admission

Date: 2026-08-01  
Verdict: **NO-GO upheld, but the argument requires two material mathematical corrections.**

## Bottom line

No breakthrough-sized mechanism survives. A capacity-capped, task-conditioned
workspace can be a useful inductive bias, conditional-compute device, and causal
measurement surface. T61 does not specify a new source of evidence, a persistent
lifetime update, a new credit signal, or a tractable interaction-aware admission
rule. Its no-run decision is therefore justified and is stronger after correcting
the proof: the current formulation also permits an uncharged information channel
through the selected indices.

## Material finding 1 — the Section 2 lower bound is false as written

The candidate defines `S ~ q(S | C, X, G)`. In the theorem's pre-goal setting,
`S` may still depend on all of `C`. If the decoder observes which candidates were
selected—as `C_S` normally implies—the mask or indices encode information. A
budget of `B` selected slots is not a budget of `B` bits.

A counterexample is `n=2, B=1`. Before seeing `G`, select `{1}` when `C_2=0` and
`{2}` when `C_2=1`. The identity of the selected slot reveals `C_2`, including
when it is nominally absent. For uniform `G`, the Bayes error is `1/8`, below the
claimed bound `1/4`. Thus “requested bit absent” does not imply conditional error
`1/2`.

The result can be repaired in either of two ways:

1. Require `S` to be independent of both `C` and `G`, with no content-dependent
   routing observable to the decoder. Then
   `P(G notin S) = 1-E|S|/n >= 1-B/n`, and the stated error lower bound follows.
2. Treat the complete representation `(S, C_S)` as the communication channel and
   charge its total rate. The subset identity alone can carry as many as
   `log2(sum_{k=0}^B binom(n,k))` bits (more if order is meaningful); use a proper
   rate-distortion or Fano-style bound instead of counting slots.

The post-goal observation—`B=1` can retain `C_G` and achieve zero error—is correct
given addressable, noiseless candidates. It establishes task-conditioned routing,
not universal compression.

## Material finding 2 — XOR does not refute the displayed ablation score

For independent fair bits and `Y=C_1 XOR C_2`, the information identities are
correct:

`I(Y;C_1)=I(Y;C_2)=0`, while `I(Y;C_1,C_2)=1 bit`.

This disproves univariate relevance scores and thresholded forward selection from
the empty set. It also shows that the natural utility
`F(S)=I(Y;C_S)` is not submodular: a bit has zero marginal value at the empty set
and one bit of marginal value when its partner is present.

It does **not** show that T61's displayed full-coalition leave-one-out score

`u_i = L_future(W without i) - L_future(W)`

misses either feature when `W={C_1,C_2}`. Removing either member destroys perfect
prediction, so both scores are positive. The paper must replace “coordinate-wise
value scores provably miss synergistic facts” with the narrower result: marginal
scores can fail depending on the coalition and search path, and arbitrary task
utility lacks the greedy `(1-1/e)` guarantee unless normalized monotonicity and
submodularity are proved. Conditional information or Shapley-style scores can
represent the XOR interaction, but exact coalitional evaluation is generally
exponential.

## Information, estimation, and expressivity

- With no external observation after `(C,X,G)`, the gate cannot add Shannon
  information about the world; it can only change which existing information is
  computationally accessible. “Future value” must not be described as an updater.
- The future loss is not observable from one online trajectory. T61 must specify
  the future-task distribution, what is known at admission time, whether future
  labels/outcomes are train-only, and how paired counterfactual losses are
  estimated without leakage. Estimator cost and variance belong in the compute
  ledger.
- The expressivity argument is correct only for a fixed finite unroll, finite
  precision, the same inputs, and (for stochastic `q`) matched randomness/output
  distributions. Under those conditions a sufficiently unrestricted network can
  simulate the composed gate and decoder.
- That extensional statement proves neither equal learnability nor equal
  parameter, sample, or compute efficiency. Conversely, unbounded recurrence,
  external memory, dynamic stopping, exact addressing, or extra FLOPs would change
  the resource model and cannot be folded into a same-budget feed-forward claim.

## 2026 literature collision check

- Anthropic's 16 July v1 preprint, [*Verbalizable Representations Form a Global
  Workspace in Language Models*](https://arxiv.org/html/2607.15495), supports the
  limited-capacity, broadcast, silent-reasoning, counterfactual-reflection, and
  unknown-entry-mechanism claims. Scope the opening sentence to the production
  Claude models actually studied; it is not evidence that all current LLMs share
  this organization. The paper also warns that the J-lens is incomplete and that
  its “bag of concepts” may miss relational binding.
- The 24 July v1 preprint [*J-CoT: Chain-of-Thought in J-Space*](https://arxiv.org/html/2607.21981)
  does implement nonnegative elastic-net extraction, adaptive sparse support,
  vocabulary-indexed recurrent state, carrier positions, and a learned read gate
  in J-CoT-Train. It therefore occupies the sparse recurrent read/write interface.
  It does not implement a coalitional future-value estimator, but downstream task
  training of an interface is already occupied; merely relabeling the credit as
  “future value” is not a distinct mechanism.
- [GRU-Mem](https://arxiv.org/abs/2602.10560) already learns text-controlled update
  and exit decisions through end-to-end RL rewards for long-context reasoning. It
  is not the same representation or granularity, but directly collides with the
  generic claim “learn what to retain and when to stop.”
- The unexplained fourth reference is [*Reasoning as Compression: Unifying Budget
  Forcing via the Conditional Information Bottleneck*](https://arxiv.org/abs/2603.08462),
  not another workspace system. It occupies task-reward-versus-compression framing
  for reasoning traces and should be named and compared explicitly or removed.

These are all recent preprints/workshop results, not settled universal laws. The
collision is sufficient to defeat novelty at the mechanism level; it does not by
itself prove that every implementation of future-conditioned selection is useless.

## Required corrections before this record is final

1. Replace the Section 2 theorem with a content-independent-selector result or a
   total-rate theorem that charges masks, indices, values, order, and router state.
2. Correct the XOR claim: marginal/forward-greedy scoring fails; the displayed
   full-coalition leave-one-out score succeeds on the two-bit example.
3. Define a set utility `F(S)`, future-task distribution, information available at
   selection and deployment, and the counterfactual estimator.
4. Explicitly prohibit or charge steganographic selection-mask and routing
   channels, plus extraction, read, recurrence, and estimator compute.
5. Scope the expressivity statement to a fixed resource model and separate
   representability from learnability and efficiency.
6. Separate within-episode working state from persistent cross-episode learning;
   nothing here is a lifetime updater or evidence-acquisition mechanism.
7. Name the conditional-information-bottleneck reference and label all 2026
   evidence by model, version, and preprint/workshop status.

## Run decision and reopening bar

**Do not run the present candidate.** A generic future-loss gate would primarily
retest already occupied end-to-end memory selection and could win through the
unpriced index channel. Reopen only with a concrete, tractable, interaction-aware
rule plus a theorem separating it from a matched recurrent learner under the same
total training and inference budget. The separation must target a project-sized
gate—sample complexity, horizon, regret, or measured compute—and the experiment
must include dense-latent, J-CoT, GRU-Mem-style, and ordinary end-to-end controls.

Until such a mechanism exists, the workspace remains a good diagnostic and
intervention surface, not the primary intelligence architecture.
