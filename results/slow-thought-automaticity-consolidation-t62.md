# T62 slow-thought automaticity consolidation — real currency trade, occupied mechanism

Date: 2026-08-01  
Status: **DIFFERENT-CURRENCY CONTROL RETAINED; DISTINCT RESEARCH MECHANISM REJECTED; NO RUN**

## 0. Result in one sentence

A model can spend training or sleep-time work to distill a slow recurrent or
chain-of-thought solution into a faster served path, and this can produce large
practical gains without using a stronger teacher.  The operation is genuine
automaticity, but it is already a crowded self-distillation and consolidation
family; supervising J-space rather than text is a useful measurement choice,
not a distinct learning principle, and no objective can compile a slow map that
the fixed fast architecture cannot represent.

## 1. Candidate

Let the same model solve an input slowly, by recurrence, search, or an explicit
trace:

\[
Y_{slow}=F_{\theta}^{(T)}(X),
\qquad
W_{slow}=P_J(H_{slow}),
\]

where `P_J` reads an internal workspace.  A fast copy or mergeable parameter
delta is then trained to predict the verified slow answer and, optionally, the
settled workspace in one pass:

\[
\min_{\delta}\;L_{task}(F_{\theta+\delta}^{(1)}(X),Y_{slow})
+\lambda\|P_J(H_{fast})-\operatorname{sg}(W_{slow})\|^2
+\beta L_{protect}.
\]

After training, `delta` is folded into the served weights.  Recurrent traces and
workspace targets are deleted.  The explicit exchange is offline teacher
execution, verification, examples, and update work for lower future inference
cost.

## 2. Exact positive cases, with the full resource ledger

Consider a stable linear recurrence

\[
h_{t+1}=Ah_t+Bx,
\qquad \rho(A)<1.
\]

For a fixed finite horizon, the exact state is

\[
h_T=A^T h_0+\sum_{k=0}^{T-1}A^kBx.
\]

If `h_0=0`, the one-pass matrix `B_T=sum_{k=0}^{T-1}A^kB` computes
the finite recurrent state exactly.  No stability assumption is required for
this finite case.  A linear initialization can also be folded into the map; a
fixed nonzero initialization requires a bias.

Separately, stability gives the infinite-horizon result.  Its fixed point is

\[
h_*(x)=(I-A)^{-1}Bx.
\]

A one-pass linear map with weight

\[
B_*=(I-A)^{-1}B
\]

computes the infinite-iteration state exactly in real arithmetic.  This proves a clean
automaticity case: repeated computation can be moved into weights of the same
input/output shape.

That does **not** make the compiled path free.  `B_T` or `B_*` must be stored and
multiplied at serving time; either can destroy sparsity, low rank, or locality.
`(I-A)^{-1}` may also be badly conditioned, especially for non-normal `A`, so
finite-precision equality can fail.  Any claimed amortization must charge the
compiled matrix's density, parameters, FLOPs, bandwidth, precision, and the
output readout—not merely its input/output shape.

The lemma is useful but narrow.  It assumes one stable linear operator, a
representable closed form, exact knowledge of the operator, and no changing
task that requires a different inverse.

## 3. Exact representability boundary

Let `H_fast` be the complete function class of the frozen served graph under its
precision and resource limits.  If the slow teacher computes

\[
f_{slow}\notin H_{fast},
\]

then no loss, hidden-state target, optimizer, or amount of training data can
make the fast graph equal `f_slow` on the full domain.  Distillation can only
choose an approximation inside `H_fast` or change the graph/resource budget.

This statement is definitional, not a complexity-separation theorem.  A fixed
finite recurrence supplies additional sequential depth; adaptive recurrence or
search can also supply input-dependent depth; unbounded iteration or external
state changes the computational model again.  Proving a concrete advantage
requires a lower bound or approximation-rate separation under the exact served
size, depth, and precision budget.  On a finite support, a large student can
simply memorize the mapping, so held-out family generalization is the relevant
test.  Successful automaticity requires frequently reused slow paths to be
compressible into the permitted fast class; novel hard cases retain the slow
fallback.

## 4. Why workspace supervision is not a new updater

J-space creates a convenient, causal target that may be less verbose than a
text trace.  Matching it can improve optimization or reveal whether one path
reconstructed a selected intermediate concept.  But the later task loss,
self-distillation, protected replay, and parameter update remain ordinary
consolidation.  The operation neither acquires new world evidence nor solves
continual credit and forgetting by itself.

The target is also not automatically well-defined after the student weights
change.  A teacher lens can become stale; a recomputed student lens introduces
a new, unaligned coordinate system.  A valid version must freeze or transport
dictionaries, define normalization/layer/token alignment, and compare against
output-only, generic hidden-state, and low-rank trajectory targets.  More
fundamentally, an automatic circuit may bypass the workspace used by the slow
teacher.  Matching that workspace is therefore neither necessary nor sufficient
for successful compilation and can force the student to imitate an unnecessarily
slow internal route.

If the slow result is wrong, consolidation makes the error automatic.  A legal
pipeline therefore requires external outcome evidence or a sound verifier,
confidence/abstention, held-out future tests, rollback, and a protected
capability suite.  Those sources and checks belong in the cost ledger.

## 5. Current collision

The broad operation is already demonstrated in several forms:

- System-1.5 uses two-stage self-distillation from full System-2 reasoning into
  adaptive latent shortcuts and reports over `20x` acceleration with comparable
  GSM8K performance.
- Fast Thinking learns latent strategy codebooks from concise reasoning traces
  and routes between single-pass and slow reasoning.
- Truncated-Reasoning Self-Distillation trains the same architecture to recover
  the teacher answer from partial reasoning and reduces later trace length.
- LoRi aligns teacher and student reasoning trajectories in a low-rank latent
  subspace.
- Nightly weight consolidation turns interaction histories into LoRA updates and
  reports `80.4%` retention versus `36.8%` after repeated context compaction,
  though it uses an external LLM for extraction/synthesis and does not report a
  broad protected-capability suite.
- Distilling System 2 into System 1 states the generic compilation thesis
  directly: use higher-quality slow outputs to train generations that omit the
  intermediate reasoning, retaining slow reasoning for unsolved tasks.
- Do Language Models Need Sleep? performs offline recurrent context processing,
  writes the result into persistent SSM fast weights, and clears the KV cache
  before wake-time inference.
- Language Models Need Sleep combines replay, knowledge-seeding distillation,
  and synthetic dreaming for continual consolidation.

Primary references:

- https://arxiv.org/abs/2505.18962
- https://arxiv.org/abs/2509.23633
- https://arxiv.org/abs/2603.13274
- https://arxiv.org/abs/2606.05315
- https://arxiv.org/abs/2605.24657
- https://arxiv.org/abs/2407.06023
- https://arxiv.org/abs/2605.26099
- https://arxiv.org/abs/2606.03979

These results establish that slow-to-fast consolidation is valuable.  They also
make a generic workspace-to-weight version an occupied integration, not a new
frontier.

## 6. What is retained

Automaticity is a mandatory control and a later system component:

1. new or uncertain tasks use slow reasoning and real feedback;
2. repeated verified solution families may be consolidated;
3. held-out tests must show that future served work actually falls;
4. the slow fallback must remain for out-of-class cases; and
5. all teacher calls, verifier work, training FLOPs, adapters, and forgetting
   are charged.

This can make a continual learner increasingly efficient.  It does not specify
how the learner discovers transferable variables, chooses experiments, or
updates a correct world model.

## 7. Run decision

**No run.**  The exact positive lemma supplies a control family, while the
representability boundary prevents a universal claim.  Current methods already
occupy latent shortcut distillation, fast/slow routing, truncated self-
distillation, and weight consolidation.  A J-space target alone does not justify
another experiment.

Reopening requires an online consolidation rule that demonstrates a large
cross-family learning gain or future-compute reduction over all of those
controls, while acquiring its targets from legal interaction evidence and
preserving protected capabilities.  The missing mechanism remains upstream:
forming and revising reusable knowledge from experience.
