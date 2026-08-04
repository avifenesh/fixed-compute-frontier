# Learned bit-sliced logic residual: paper audit

Date: 2026-08-01  
Status: **HOLD AS A NARROW PRIMITIVE; NOT ADMITTED TO CODE, GPU, OR MODEL TRAINING**

## Decision

A bit-sliced logic lane can implement some exact discrete circuits with far
fewer stored coefficients and arithmetic operations than a real-valued dense
layer.  That is a valid change of algebra.  It is not yet a language-model
candidate: independent learned gates do not map to one GPU `LOP3`, learned
wiring is not free, and no raw-text acquisition or breakthrough-size natural
capability bridge has been supplied.

The retained claim is therefore narrow:

> A group-shared, compiled Boolean sidecar may be an efficient representation
> for already typed, highly structured procedural features.  It is not a
> general replacement for QKV or the FFN, and it has no admitted end-model
> effect until the semantic interface and full physical ledger are proved.

## Candidate object

Let a layer contain `G` three-input Boolean lookup tables.  For gate `g`,

\[
z_g=\operatorname{LUT}_{\theta_g}(z_{i_g},z_{j_g},z_{k_g}),
\qquad \theta_g\in\{0,1\}^{8}.
\]

The proposal was to bit-slice 32 independent feature instances into a 32-bit
word, run a shallow logic circuit, decode a small real-valued residual, and
retain an ordinary analog path as a fallback.

NVIDIA's PTX `lop3.b32` computes one arbitrary three-input Boolean function on
all 32 bit positions.  Its function is selected by one *immediate constant*
from 0 to 255; it does not supply 32 independently selectable LUTs in one
instruction.  See the current
[PTX instruction definition](https://docs.nvidia.com/cuda/archive/13.1.0/hopper-tuning-guide/parallel-thread-execution/index.html#logic-and-shift-instructions-lop3).

## What is mathematically real

### 1. Exact local separation

One-bit full addition has

\[
s=a\oplus b\oplus c,\qquad
r=(a\land b)\lor(a\land c)\lor(b\land c).
\]

With PTX's truth-table ordering, the two functions have immediates `0x96` and
`0xE8`.  Two `LOP3` instructions therefore compute 32 lane-aligned full
adders.  A single affine threshold cannot represent parity, so this is a real
separation from one threshold unit.

It is not a separation from a complete MLP, a binary/ternary network, or a
small arithmetic circuit.  Those are the required controls.

### 2. Description bound

A fixed-topology network with `G` three-LUTs has at most

\[
2^{8G}
\]

gate-function configurations.  There are

\[
2^{m2^n}
\]

functions from `n` input bits to `m` output bits.  A family able to express
every such function consequently needs

\[
G\ge \frac{m2^n}{8}.
\]

LUT storage is exponentially bad for arbitrary functions.  Its opportunity is
only functions possessing small circuit descriptions.

### 3. Dependency bound

With fan-in three and depth `S`, one output's input cone contains no more than

\[
3^S
\]

leaves, before duplication and reconvergence.  Long-range or high-arity
decisions require depth, explicit reductions, or learned routing.  The fixed
expander cannot be treated as universal at shallow depth.

## The missing GPU ledger

There are three physically different proposals that must not be conflated.

### A. One function shared across 32 lanes

- persistent function state: 8 bits per 32-gate group;
- work: one `LOP3` per group;
- restriction: all 32 aligned gates execute the same truth table.

This is genuinely cheap but strongly tied.

### B. One independent function per lane

- persistent function state: 8 bits per gate;
- a dynamic eight-entry bit-sliced table needs eight truth masks;
- selecting one entry from the three input bits takes a seven-mux tree, hence
  roughly seven fixed-function `LOP3` operations plus mask loads and registers.

This is trainable as independent gates, but the one-instruction argument is
lost.

### C. Compile each distinct group into an immediate

- work can approach one `LOP3` per group;
- generated code, compilation/JIT time, instruction-cache pressure, launch
  structure, and recompilation frequency become charged state and work.

For all three cases, selecting arbitrary inputs costs wiring.  Explicitly
learned three-way wiring over `N` sources requires approximately

\[
3G\lceil\log_2 N\rceil
\]

index bits before alignment metadata.  Irregular wiring also introduces
gathers or bit permutations and can dominate the truth-table operation.

The claim that integer logic can execute "for free" beside tensor-core GEMM is
only a hypothesis.  The relevant zero-harm condition is

\[
T_{quant}+\max(T_{GEMM},T_{logic})+T_{join}
\le (1+\mu)T_{base},
\]

and it must include shared scheduling, registers, occupancy, LSU/cache
traffic, dependencies, and launch overhead.  Different nominal execution
units do not establish overlap.

## Learnability boundary

The circuit description is compact only if useful connectivity is also
compact and learnable.  Recent differentiable-logic results explicitly report
large gains from optimizing connections rather than using fixed connectivity;
that supports the importance of the missing routing term, not the current
candidate.  See
[Mommen et al.](https://arxiv.org/abs/2607.09399).  FPGA-native LUT learning is
also already an active architecture family, including
[BitLogic](https://arxiv.org/abs/2602.07400), so merely replacing arithmetic
with learned LUT nodes is not a novelty claim.

No current argument establishes all of the following from raw prose:

1. which hidden features should become bits;
2. which examples share a compiled Boolean program;
3. how the input wiring is recovered without semantic supervision;
4. why the resulting program improves held-out knowledge or reasoning rather
   than only exact synthetic logic;
5. why the improvement exceeds the capacity recovered by reinvesting the same
   bytes and latency in the strongest quantized or sparse analog control.

## Strongest controls

Any reopening must compare against:

- a matched binary/ternary MLP using the same typed inputs and byte budget;
- a small quantized dense residual funded by the same removed FFN/QKV slice;
- a table or sparse expert containing the same procedures;
- zero injection into the unchanged analog baseline;
- group-shared and independently learned LUT variants with all routing and
  compilation costs charged.

## Why no experiment runs now

A `LOP3` kernel benchmark could answer whether one frozen, aligned circuit is
fast.  It cannot answer whether raw language yields such circuits, whether a
model can route to them, or whether their best-case contribution is materially
larger than the funded analog control.  Hardware would therefore measure a
downstream implementation detail before the end-effect bridge exists.

The primitive may reopen only after one paper packet supplies:

1. a raw-observable typed interface;
2. a natural capability witness with a strict matched-control separation;
3. a complete group/wiring/code/state ledger; and
4. an optimistic end-effect lower bound large enough to cross the project's
   material-gain threshold.

Until then, the valid result is the circuit-compression lemma and its exact
boundary—not a smarter production LLM.
