# Independent audit — T62 slow-thought automaticity consolidation

Date: 2026-08-01  
Verdict: **NO-GO upheld; useful control/component, no distinct breakthrough mechanism.**

## Bottom line

The slow-to-fast trade is real, but T62 proposes ordinary distillation with an
interpretability-derived auxiliary target. The linear example is sound after
separating a finite unroll from its infinite fixed point and after charging the
compiled map's actual serving resources. The representability statement is true
but tautological. J-space supervision can be a useful hint or diagnostic; it adds
no new evidence or learning rule, has an unresolved coordinate-alignment problem,
and may be unnecessary for the automatic computation the student is meant to
learn. The current literature already contains both generic System-2-to-System-1
compilation and more direct sleep-time consolidation mechanisms.

No experiment is warranted under the project's strict fixed-compute gate.

## 1. Linear lemma: correct limit, mismatched to the finite teacher

For

\[
h_{t+1}=Ah_t+Bx,
\]

the exact finite-time state is

\[
h_T=A^T h_0+\sum_{k=0}^{T-1}A^kBx.
\]

If `h_0=0`, a single linear map
`B_T = sum_{k=0}^{T-1} A^k B` computes `h_T`. No stability assumption is needed
for a fixed finite `T`. If `h_0` is a linear function of `x`, its contribution can
also be folded into the map; a fixed nonzero `h_0` requires a bias. A nonlinear or
history-dependent initialization cannot be silently omitted.

When `rho(A)<1`, `A^T h_0 -> 0`, `I-A` is invertible, and

\[
h_*(x)=(I-A)^{-1}Bx
\]

is indeed the unique fixed point and the limit from every finite `h_0`. Thus
`B_*=(I-A)^{-1}B` exactly computes the **infinite-iteration state in exact real
arithmetic**. T62 initially defines a finite `F_theta^(T)` teacher, then proves the
infinite-limit case; those are different claims and must be labeled separately.
If the answer applies an output readout, that readout must also be composed into
the compiled map.

Three resource qualifications are required:

- `rho(A)<1` does not imply that `(I-A)^{-1}` is well conditioned, especially for
  non-normal `A`; finite-precision equality can fail.
- The inverse is not merely “paid during compilation.” Its result must be stored
  and multiplied at serving time. `B_*` can destroy sparsity, low rank, locality,
  or other structure that made each recurrent step cheap.
- Equal input/output shape is not equal parameter, memory-bandwidth, precision,
  or FLOP budget. The positive case proves amortization only after those quantities
  are measured.

## 2. Representability boundary: true definition, not a separation theorem

If `H_fast` is defined as every function realizable by the allowed served graph and
`f_slow` is outside it, exact equality over the full domain is impossible. This is
correct by definition. It does not quantify approximation error on the deployment
distribution, nor show that any concrete slow function lies outside a matched fast
class.

The text should also distinguish:

- fixed finite recurrence, which supplies additional sequential depth and can be
  represented by an explicitly unrolled deeper graph;
- adaptive recurrence or search, which may supply input-dependent depth; and
- unbounded iteration/external state, which can change the computational model.

A fixed recurrence is not automatically input-adaptive. To make Section 3 more
than a tautology, T62 would need a size/depth/precision lower bound or an
approximation-rate separation under the exact served budget. On a finite support,
a sufficiently large student can memorize the slow mapping even if it cannot
represent the algorithm compositionally; held-out family generalization is the
relevant test.

## 3. J-space-to-weight adds a target, not a learning principle

The proposed loss is feature/representation distillation with privileged teacher
activations. It can improve optimization or sample efficiency, but the teacher's
workspace is still a deterministic product of its input and parameters. The
parameter update, replay, and task loss remain ordinary post-training; no new
world evidence, causal credit mechanism, or protection against forgetting appears.

There are also unresolved technical problems:

1. A J-lens is built from a model's averaged downstream Jacobian. After changing
   `theta` to `theta+delta`, the teacher and student J-spaces are not automatically
   the same coordinate system. Keeping the teacher lens fixed risks a stale
   readout; recomputing the student lens creates a moving, unaligned target.
   T62 must specify fixed dictionaries, normalization, layer/token alignment, and
   an explicit cross-model transport map.
2. The J-space is a sparse, non-orthogonal and incomplete concept frame. Squared
   coordinate error is not intrinsically semantic or causal, and a “settled” state
   is not defined for multiple valid slow trajectories.
3. Matching a workspace state is neither sufficient for matching behavior nor
   necessary for automaticity. Anthropic's [global-workspace study](https://arxiv.org/html/2607.15495)
   reports that flexible reasoning depends on J-space while routine automatic
   processing can proceed with it suppressed. A successful compiled circuit may
   therefore bypass the slow teacher's workspace. Forcing equality can preserve a
   slow representation rather than discover the cheapest fast route.
4. Causal swaps/ablations can test whether a learned target is used, but they do
   not turn the target into a distinct updater. Output-only, generic hidden-state,
   low-rank-trajectory, and J-space losses must be matched controls.

The only potentially distinct empirical hypothesis is that a causally selected,
cross-model-aligned J-space target yields a large sample or future-compute
advantage over those controls. T62 supplies no theorem or evidence for it, so this
is a reopening condition rather than a surviving mechanism.

## 4. Current literature collision

The five cited summaries are substantially accurate, with important scope labels:

- [System-1.5](https://arxiv.org/abs/2505.18962) is explicitly a work-in-progress
  preprint. It uses two-stage language-to-latent and full-path-to-shortcut
  distillation and reports over `20x` GSM8K inference acceleration, but its fast
  path changes the graph with routers/adapters rather than folding everything into
  an unchanged one-pass graph.
- [Fast Thinking](https://arxiv.org/abs/2509.23633) learns discrete strategy
  codebooks from concise traces and routes between codebook-guided single-pass and
  explicit slow reasoning.
- [TRSD](https://arxiv.org/abs/2603.13274) trains a same-architecture student to
  recover a frozen teacher's answer distribution from truncated reasoning and
  reports shorter later traces.
- [LoRi](https://arxiv.org/abs/2606.05315) directly aligns teacher and student
  reasoning trajectories in a shared low-rank latent subspace. This is the closest
  listed collision with “workspace rather than text” supervision.
- [Beyond Inference-Only Deployment](https://arxiv.org/html/2605.24657) reports
  the stated `80.4%` versus `36.8%` retention, but only over ten synthetic software
  conversations. It uses Claude Sonnet 4 for extraction, synthesis, conversation
  generation, and judging, trains on A100s, and explicitly does not measure broad
  general-knowledge/coding or safety retention. T62's caveat is warranted.

The collision review is incomplete without three especially direct precedents:

- [*Distilling System 2 into System 1*](https://arxiv.org/abs/2407.06023) already
  states the generic thesis almost verbatim: compile higher-quality System-2
  outputs into generations without intermediate reasoning tokens and reserve slow
  reasoning for tasks not yet handled well.
- [*Do Language Models Need Sleep? Offline Recurrence for Improved Online
  Inference*](https://arxiv.org/abs/2605.26099) performs offline recurrent passes
  over context, writes the result into persistent SSM fast weights, clears the KV
  cache, and preserves wake-time prediction latency. This is a direct
  offline-compute-to-served-weights collision, not merely an analogy.
- [*Language Models Need Sleep: Learning to Self-Modify and Consolidate
  Memories*](https://arxiv.org/abs/2606.03979) combines replay, knowledge-seeding
  distillation into longer-term parameters, and synthetic dreaming for continual
  learning. It further occupies the “sleep-time consolidation” framing.

These are recent preprints, not universal evidence. They nevertheless eliminate
mechanism-level novelty. J-space changes the instrumentation and possibly the
quality of the hint, not the underlying consolidation operation.

## 5. Required corrections

1. Split the finite-`T` closed form from the stable infinite fixed-point lemma and
   state initialization/output-readout assumptions.
2. Replace “inverse paid during compilation” with a full ledger for compiled
   weight storage, density/structure, serving FLOPs, bandwidth, precision, and
   conditioning.
3. Present the representability boundary as definitional unless a concrete
   resource lower bound or approximation separation is supplied; do not call all
   recurrence input-adaptive.
4. Define teacher/student J-space alignment under the parameter update, including
   whether the lens is frozen or recomputed and how coordinates are transported.
5. Treat workspace loss as an optional auxiliary distillation target and require
   output-only, generic-hidden-state, LoRi-style, and causal-target controls.
6. Add the canonical System-2-to-System-1 and both 2026 sleep/consolidation papers;
   label model changes, external teachers/judges, dataset scope, and preprint status.
7. Keep verification, abstention/fallback, protected-capability tests, rollback,
   teacher/verifier calls, and all offline updates in the total cost ledger.

## 6. Run decision

**Do not run the candidate as written.** The positive lemma is a useful analytic
control, while the proposed nonlinear method is an occupied distillation pipeline
with an unvalidated auxiliary representation target. The newly identified sleep
papers make the collision stronger than T62 states.

Reopen only if a concrete J-space transport/target rule predicts and then delivers
a strict advantage over matched output, dense-state, low-rank-trajectory, and
sleep-consolidation controls under the same total offline-plus-served compute. The
result must generalize across held-out task families, reduce amortized future
compute after consolidation cost, preserve protected capabilities, and retain a
slow fallback for out-of-class inputs. Until then, automaticity belongs in the
system ledger as a control and downstream component, not as the missing primary
learning mechanism.
