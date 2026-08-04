# 003 — Bounded semantic feedback

Status: **closed — positive mechanism, rejected architecture**  
Date: 2026-07-23

## Claim tested

A single bounded memory written from the completed top representation of token
`t` might preserve more useful temporal state than the identical memory written
from the raw token input. The proposed memory had:

- an exact ring of eight records;
- a normalized recurrent summary `(S, z)` of evicted records;
- one shared commit per token, read by all three layers.

All global variants had 64,129 learned parameters, 1,752 persistent state
elements per sequence, identical reader/writer modules, identical physical read
and update paths, and the same post-token commit timing. Only the write source or
a masked read contribution changed.

## Valid causal probe

The final calibration used a four-state machine with two fixed noncommuting
mutations:

\[
A(s)=(s+1)\bmod 4, \qquad F(s)=s\mathbin{\mathrm{xor}}1.
\]

Sequences were 48 tokens. `ASK` queried the state after prior mutations; neither
mutation token nor `ASK` exposed the answer. A three-layer GRU positive control
trained on the same state-only generator reached **96.14%** after 200 steps,
versus its measured **27.11%** majority prior, proving the probe was learnable.

Configuration: batch 128, width 48, three layers, memory feature width 24,
200 train steps, 10 evaluation batches, seeds 1–3, PyTorch 2.11.0+cu130 on the
reserved H100 PCIe.

## Result

| Variant | What can be read | Mean state accuracy | Seed range |
|---|---|---:|---:|
| Input-written control | exact + summary | 26.68% | 26.41–26.93% |
| Top-written candidate | exact + summary | 46.94% | 35.30–56.55% |
| Top, exact masked | summary only | 26.88% | 26.78–27.00% |
| Top, summary masked | exact only | 50.13% | 42.06–63.85% |
| Measured majority prior | neither | 27.12% | 26.85–27.51% |

Top writing beat input writing in every seed, by **20.26 percentage points on
average**. This is a real causal signal: a completed representation can carry a
computed state forward where a bounded collection of raw event embeddings
cannot.

The proposed two-tier mechanism did not survive. Summary-only matched the
input/prior controls, while exact-only exceeded the full candidate by **3.19
points on average**. The recurrent summary was not necessary and was mildly
harmful at this training horizon.

## Structural kill gate

The exact feedback dependency is:

\[
h_{t,1}\rightarrow\cdots\rightarrow h_{t,L}
\rightarrow h_{t+1,1}.
\]

For a prompt of `n` tokens, the critical path therefore contains `nL`
nonlinear layer evaluations. Standard Transformer prefill has roughly `L`
sequential layer stages because prompt positions run together inside each
layer. Wavefront scheduling cannot cross the top-to-bottom edge. An associative
scan is exact only after restricting the state transition to a closed
scan-compatible algebra, which changes the tested mechanism. Blockwise feedback
or parallel-prefill/recurrent-decode switching are different architectures.

This is a different cost, but it violates the fixed serving-latency envelope.
The branch is closed before a language-model pretrain or binding sweep.

## Evidence boundary

- The result supports **top-written temporal feedback for a small structured
  state task**.
- It does not establish better language modeling, denser parametric knowledge,
  arbitrary long-context recall, or a serving Pareto improvement.
- High-level feedback and condensed layer caches already have close precedents
  in the [Feedback Transformer](https://arxiv.org/abs/2002.09402) and
  [LCKV](https://arxiv.org/abs/2405.10637). The narrow exact-plus-summary form did
  not earn a stronger novelty claim.

## Audit history

The original `003-screen-seed1.json` is retained but inadmissible: it contained
a trivial untouched-state prior, reversed ablation labels, an off-by-one age
boundary, a missing normalizer state, over-softened cosine lookup logits,
ambiguous answer-refresh categorization, and incomplete timing/count reporting.
The corrected mixed-task calibration is also not used for the causal verdict
because neither the candidate nor a GRU learned its mod-16 state program.

Valid raw artifacts:

- `003-gru-state-only-control-seed1.json`
- `003-state-only-top-vs-input-seed1.json`
- `003-state-only-endpoints-seed1.json`
- `003-state-only-seeds2-3.json`

The executable benchmark is
`experiments/003_bounded_semantic_feedback.py`.
