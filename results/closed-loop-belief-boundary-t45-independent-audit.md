# T45 independent audit — corrected boundary, no candidate

Date: 2026-08-01  
Verdict: **HOLD THE RESET-VERSUS-STATE THEOREM; NO ARCHITECTURE OR EXPERIMENT**

## Main finding

T45 identifies a necessary channel: run-specific evidence must remain in some
model-readable state after it leaves the current request. It does not identify
the missing architecture. Context writeback, RNN/SSM state, fast weights,
external memory, and cross-episode meta-RL can all implement the two-world
update.

The two-world construction assumes away the harder work: discovering a
behaviorally sufficient state, identifying the factor affected by an action,
calibrating its likelihood, assigning delayed outcomes to the responsible
action, and choosing an intervention that distinguishes competing mechanisms.
Because every action in T45 provides equally strong evidence, it is not an
active-learning world.

## Corrections applied to T45

1. Separate cumulative reward `G_T` from regret `Reg_T`.
2. State the conditional-independence and known-likelihood assumptions.
3. Restrict bounded regret to fixed `Delta` and call it an upper bound.
4. Treat Fano as a bound for a discrete or capacity-limited physical channel;
   an unconstrained real scalar is not a bit ledger.
5. Treat `ceil(log2(2T+1))` as a sufficient horizon-wide signed-counter
   allocation, not a tight fixed-time lower bound.
6. Remove the invalid claim that a Hoeffding sample count licenses recurrent
   counter clipping. A stopping/freeze rule and a continually saturating
   counter are different objects.
7. Keep reward gap `Delta` distinct from learned evidence bias `gamma`:

   \[
   P(\widehat Z_n\ne Z)\le e^{-2n\gamma^2},\qquad
   \mathbb E[\operatorname{Reg}_T]
   \le\Delta+{2\Delta\over e^{2\gamma^2}-1}.
   \]

8. Replace “first missing resource” with “one necessary resource.”

## Strong current controls

- [Algorithm Distillation](https://arxiv.org/abs/2210.14215) trains a causal
  Transformer to implement an RL learning algorithm in context.
- [ORBIT](https://arxiv.org/abs/2602.04089) trains cross-episode online learning
  with meta-RL.
- [PABU](https://arxiv.org/abs/2602.09138) explicitly learns compact belief
  update and selective retention for LLM agents.
- [Unified Memory Agent](https://arxiv.org/abs/2602.18493) jointly trains memory
  CRUD, consolidation, and answering.
- [RL-squared](https://arxiv.org/abs/1611.02779) and
  [VariBAD](https://arxiv.org/abs/1910.08348) are recurrent/meta-RL controls for
  learned update and uncertainty-conditioned exploration.
- [In-Place TTT](https://arxiv.org/abs/2604.06169) and
  [Titans](https://arxiv.org/abs/2501.00663) cover fast-weight/neural-memory
  routes.

## Next justified object

The next algebraic object is an interventional predictive-state quotient:

\[
h\sim_I h'\iff
\forall\pi,\quad
P(O_f,R_f\mid h,\operatorname{do}(\pi))
=P(O_f,R_f\mid h',\operatorname{do}(\pi)).
\]

Paper work must establish a controlled family, observable intervention
separation, a closed update, delayed-credit failure, an information-seeking
policy, finite-sample bounds, and the exact containment relation to the strong
controls. No code or compute is admitted by this audit.
