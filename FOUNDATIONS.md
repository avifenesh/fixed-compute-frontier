# Foundations

For model/system `M` under workload `w`, define the logical-core projection:

`R_core(M,w) = (P, F, K, L)`

- `P`: every model-specific served static byte, learned or not, including
  weights, scales, indices, routers, lookup tables, constants, codebooks,
  compiled auxiliary data, draft models, and streamed weights.
- `F`: every active operation, including routing, drafting, verification,
  retrieval encoding, and recomputation.
- `K`: every bit of persistent model-readable request state, not merely tensors
  named KV cache.
- `L`: TTFT and TPOT p50/p95/p99 over a declared hardware, batch, context, and
  concurrency envelope.

`R_core` is not the full physical ledger. Workspace, memory traffic, throughput,
energy, CPU/network/storage work, cost, and training amortization are separately
mandatory under [`RESOURCE_CONTRACT.md`](RESOURCE_CONTRACT.md). Winning only the
projection is not a served Pareto result.

Let `Q(M,D)` be a vector over protected distributions: held-out loss, factual
knowledge, reasoning, context use, robustness, and calibration. It is not one
averaged benchmark score.

A strict served Pareto result requires, within preregistered tolerances:

`Q'_j >= Q_j - delta_j` for every protected quality slice, and
`R'_i <= R_i` for every core and full-ledger served resource, with at least one
statistically strict improvement. If an unconstrained resource rises, the result
is a trade rather than Pareto dominance.

## What cannot be escaped

- `M` effective stored bits cannot encode more than `M` arbitrary independent
  random bits exactly under a fixed decoder.
- `S` bits of model-readable runtime state distinguish at most `2^S` histories.
- Computations with a real circuit/work lower bound above `F` cannot be solved
  universally within `F`.
- Latency is bounded below by the slowest of compute throughput, memory traffic,
  dependency depth, and communication.
- No finite model dominates on every possible input-label distribution at equal
  resources.

## Matched-machine inclusion theorem

Let a finite-precision architecture `A` have the `P` static bytes defined above
(`8P` bits), `S` mutable bits, and an inference program of at most `T` charged
primitive operations per token.
A recurrent RAM controller can store the same static and mutable bits and run
the same compiled instruction sequence with the same instruction/addressing
set. For randomized `A`, couple the controller to the same random tape.
Induction over those primitive invocations gives the identical next state and
output after every token. If architecture/program code is counted for `A`, count
the same code for the controller; if it is treated as fixed system code, treat
it identically for both.

Therefore a changed algebra cannot prove strict capability dominance over a
matched unrestricted recurrent-RAM baseline at the same logical work and
resources. This does not assert equal wall-clock latency for different physical
implementations. It can beat a restricted dense recurrence, attention scan, or
accelerator kernel, but then the withheld instruction, addressing mode,
precision, randomness, approximation error, workload structure, or hardware
path is the actual added resource and must be named.

This is a control theorem, not an argument that all architectures train equally
well or run equally well on a GPU. Learnability, inductive bias, and physical
execution can still create an empirical edge; they cannot be inferred solely
from a data-structure or algebraic inclusion proof.

## Verification adds no task information

Suppose one model pass produces answer `A` and certificate `C` from input `X`,
and a verifier sees only `(X,A,C)` plus fresh independent randomness `R`. With no
retry, extra candidate, interaction, external prover, or added compute:

- returning `A` leaves accuracy unchanged;
- turning rejection into abstention raises precision only by lowering coverage,
  and cannot raise exact accuracy when abstention is wrong;
- producing a replacement answer makes generator, verifier, and replacement
  logic one composed randomized circuit, which is the matched direct control.

If `R` is conditionally independent of `Y` given `(X,A,C)` and
`hat Y=f(X,A,C,R)`, data processing gives

\[
I(Y;\hat Y\mid X)\le I(Y;A,C\mid X).
\]

Postprocessing can improve accuracy by decoding information already present in
`(A,C)`. Certificates can expose that information, detect faults, improve
calibration, or guide paid search. They do not add task information; the same
postprocessor must be included in the matched direct control.

## Where a genuine gain can still come from

Real language and current execution are not worst-case optimal. A candidate
must name at least one concrete source of slack:

1. training, data, or objective slack: the existing graph can hold better
   weights than ordinary training finds;
2. statistical compressibility: useful knowledge shares structure and is not a
   random key-value table;
3. redundant precision or representation: bits are spent where they do not
   affect protected quality;
4. poor allocation: FLOPs, bytes, and state are spent uniformly although their
   marginal value differs;
5. systems slack: real execution sits above hardware and dependency lower bounds.

If a claim cannot identify and measure one of these gaps, it is likely moving
cost or renaming combinations rather than creating an edge.

## Edge-specific impossibility controls

- **A — smaller with larger capability:** include a random key-value memory
  task; arbitrary facts above the student's effective bit budget cannot be
  compressed without external state.
- **B — faster with slower capability:** perform a roofline and critical-path
  audit; if the baseline is already at the bound, speed requires removing work,
  traffic, state, or dependencies.
- **C — light with dense knowledge:** report both knowledge per byte and absolute
  knowledge; a tiny but useless model does not win on density.
- **D — lower GPU cost with more performance:** count residency, concurrency,
  CPU/network services, and total output tokens under the same SLA.
- **E — other dominance:** randomize semantics where possible and count every
  auxiliary state or operation; hidden retrieval, drafts, caches, and longer
  reasoning traces fail the strict claim.
