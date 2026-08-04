# Sparse Cayley program FFN — 10M-token pilot preregistration

Status: **frozen before implementation benchmark or training**  
Date: 2026-07-30

## Primary claim under test

Edge A: replace every width-1,024 dense SwiGLU in the frozen 12-layer,
width-384 scratch language model with two width-384 token-conditioned sparse
Cayley branches.  The candidate must use fewer total checkpoint parameters and
less ideal active FFN arithmetic while recovering substantially more language
quality than an equal-FFN-parameter dense SwiGLU.

This pilot admits a multi-seed/longer run; it cannot establish a breakthrough
alone.

## Frozen candidate

Each of the gate, value, and down maps in each branch is a length-two ordered
program over a bank of four sparse Cayley experts.  Each expert has three
perfect-match edge sets.  A learned `D x E` router chooses one expert at each
program step.  Training uses hard-forward/soft-backward selection; serving
executes only the selected expert.  Each Cayley response uses four Neumann
sparse multiplies with generator row-sum bound `alpha=0.25`.

At `D=384`, one program projection has:

- router: `2 * 384 * 4 = 3,072` parameters/MACs;
- expert edges: `4 * 3 * 384 / 2 = 2,304` parameters;
- input/output diagonals: `768` parameters;
- total: `6,144` parameters.

Six projections per two-branch FFN use `36,864` parameters, exactly the same
as a dense width-32 SwiGLU (`3 * 384 * 32`).  The baseline width-1,024 SwiGLU
uses `1,179,648` parameters per layer.

The ideal hard-route candidate executes at most 78,336 multiply-like scalar
operations per FFN, 6.64% of the dense baseline, excluding indices, routing
softmax, nonlinearities, gathers, launches, and synchronization.  Training
evaluates all experts for router gradients and is not claimed efficient.

## Arms

1. frozen full width-1,024 parallel SwiGLU baseline;
2. equal-parameter width-32 dense SwiGLU;
3. static-route sparse Cayley program with the same stored router bytes;
4. token-conditioned sparse Cayley program.

Use the frozen FineWeb-Edu-dedup token stream, seed 815, 305 optimizer steps,
512 tokens, global batch 64, AdamW schedule, and 10,000,000 prediction tokens
from the existing cycle-factor screen.  The new G7 software/hardware manifest
must be recorded; older H100 numbers are context, not protocol-equal evidence.

## Pilot gates

All are required to advance.

1. Candidate FFN parameters equal width-32 dense SwiGLU exactly and candidate
   total model parameters are below 70% of the full baseline.
2. All arms are finite; data hashes, examples, seed, optimizer steps, global
   batch, and evaluation examples match.
3. Dynamic Cayley recovers at least 25% of the NLL gap between width-32 and
   full SwiGLU, with paired evaluation-loss upper 95% interval below zero
   against width-32.
4. Dynamic Cayley improves NLL by at least 2% relative to width-32.  This
   intentionally rejects small polish.
5. Dynamic Cayley beats the separately trained static-route control by at
   least 1% NLL and its fixed-route terminal ablation loses at least 1%, showing
   token-dependent program choice is causal.
6. Median route perplexity is at least 2.0 of four experts, maximum expert load
   is below 80%, and changing route IDs changes outputs materially.

Failure closes this exact `E=4,L=2,B=2,alpha=.25,T=4` model.  Do not tune it
under the same result.

## Hard boundary

Even a pass would show a large parameter/active-work capability gain only at
small scale.  Promotion requires untouched seeds, longer training, a stronger
matched sparse/recurrent control, exact checkpoint bytes, a fused hard-route
kernel, end-to-end decode/prefill measurements, and direct prior-art review.
