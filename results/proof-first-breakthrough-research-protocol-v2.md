# Proof-first breakthrough research protocol v2

Status: **FROZEN METHODOLOGY; SUPERSEDES V1 FOR NEW RUN ADMISSION**  
Date: 2026-07-31

## One rule

An experiment may measure an unresolved fact, but it may not substitute for an
unwritten mechanism.

Before a training or target-GPU run, the candidate must be an explainable
composition of typed blocks.  Exact claims are proved, physical claims are
microbenchmarked, learnability claims are isolated, and the end-to-end run is
left with one important causal uncertainty.  If the proposed edge still needs
two independent miracles, it returns to paper.

This does not demand a theorem that natural-language training will work.  Such
a theorem is generally unavailable.  It demands that we stop calling these
different statements the same thing:

1. an operator can represent a function;
2. a particular optimization process can learn that function;
3. the learned function improves useful capability;
4. its physical implementation meets the serving ledger.

## What can be learned from real breakthroughs

The public record shows the evidence the researchers exposed; it does not show
their complete private creative process.

| Work | Obstruction made concrete | New mathematical object | Pre-scale evidence pattern |
|---|---|---|---|
| [Transformer](https://arxiv.org/abs/1706.03762) | recurrent dependency makes sequence processing serial and lengthens information paths | an all-pairs, content-addressed dependency graph | operation count, sequential depth, path length, ablations, then translation |
| [Sparse MoE](https://arxiv.org/abs/1701.06538) and [Switch](https://www.jmlr.org/papers/v23/21-0998.html) | dense capacity and per-token computation rise together | conditional parameter activation | capacity/active-work separation first; routing, balance, communication, precision, and quality treated separately |
| [Mamba](https://arxiv.org/abs/2312.00752) | an LTI state update cannot selectively retain content | input-dependent state dynamics | selective-copy witnesses, a gate interpretation, hardware-aware scan, ablations, then language scaling |
| [FlashAttention](https://arxiv.org/abs/2205.14135) | attention is constrained by memory traffic, not just FLOPs | an exact tiled online-softmax schedule | algebraic equality, I/O analysis, numerical check, kernel measurement, then end-to-end models |
| [AlphaTensor](https://www.nature.com/articles/s41586-022-05172-4) | free-form algorithm search has an enormous unverifiable space | tensor decomposition as a finite game with an exact terminal condition | rediscover known algorithms, verify exactness, preserve diverse solutions, then optimize rank or real hardware time |
| [FunSearch](https://deepmind.google/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/) and [AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) | an LLM can generate ideas but cannot certify them | small executable programs judged by automated evaluators | constrain the mutable surface, reject incorrect programs, keep diverse winners, inspect compact human-readable constructions |

The common research grammar is:

> name the obstruction; expose a minimal separating witness; change the
> algebraic object; prove the local invariant; identify the different resource
> currency; microbenchmark the new primitive; then test the complete causal
> claim against the strongest legal control.

The novelty is often the change of object, not the number of tuned components:

- sequence becomes a dependency graph;
- a dense parameter set becomes a routed population;
- a fixed recurrence becomes an input-conditioned dynamical system;
- a nominal FLOP graph becomes an I/O schedule;
- an informal idea becomes a verifiable program.

## Candidate research packet

Every candidate is written as

\[
\mathcal C=(G,\,W,\,\mathcal O,\,\Delta,\,\mathcal B,\,\Pi,\,\Phi),
\]

where:

- `G` is one promised production capability gain and its minimum effect;
- `W` is a minimal witness family on which the named baseline mechanism fails;
- `O` is the typed operator or short operator composition;
- `Delta` is the alternative currency or slack that funds the gain;
- `B` is the complete conserved serving vector;
- `Pi` is the set of proved claims and explicit no-go boundaries;
- `Phi` is the one important empirical composition hypothesis left for the
  final experiment.

If `Delta` is merely more parameters, more active arithmetic, more model bytes,
more prompt tokens, a stronger external model, or task labels unavailable to
the baseline, the candidate has not changed the scaling path.

## Paper gate

### P0. End claim, not component claim

Name the user-visible capability, comparison class, threshold, protected
slices, and unchanged served resources.  A kernel speed, exact codec, lower
training loss, or stronger information oracle is not itself a smarter model.

### P1. Minimal separating witness

Construct the smallest input family that exercises the exact alleged weakness.
Prove the failure for the named restricted mechanism.  State whether it is a
universal lower bound or only a failure of one parameterization.

The witness must not contain shortcuts.  If noncommutativity is the proposed
edge, powers of one generator are invalid.  If query-dependent memory is the
edge, a fixed-position lookup is insufficient.  If autonomous retrieval is the
edge, supplying the correct record handle is insufficient.

### P2. Typed operator graph

Write every edge with shapes and meanings:

```text
raw observation -> extracted statistic -> written state
query, written state -> selected evidence -> answer-bearing state
```

Every intermediate needs a semantic contract such as recoverability,
equivariance, conservation, calibrated approximation, or a declared causal
meaning.  “The model will learn a useful vector” is an empirical hypothesis,
not a semantic contract.

### P3. Positive construction

Provide an explicit parameter setting or algorithm witnessing the desired local
property.  An expressivity statement without a construction is insufficient.
An oracle using unavailable labels or a stronger hidden model is an upper
bound, not the construction.

### P4. Obstruction and failure boundary

Prove or derive where the construction fails: information bound, rank ceiling,
precision margin, adversarial family, dependency depth, routing ambiguity, or
physical roofline.  A mechanism without a known failure surface is not ready
to spend compute.

### P5. Strongest-control construction

Actively try to make the baseline win on paper.  Grant it every operation and
resource permitted by the end claim.  Include typed buffers, integer state,
conditional reads, compilation, or parameter reuse if the candidate uses them.

If the strongest control can implement the same operator, the claim is about
optimization or inductive bias, not function class.  The experiment must test
that narrower claim.  A dominated representation or implementation is
withdrawn before timing; this is a design correction, not rescue tuning.

### P6. Complete ledger

Count, with units:

- persistent bytes by dtype;
- active multiplies and non-matmul operations;
- bytes moved at each memory level;
- mutable state and KV entries;
- workspace and graph pools;
- critical-path depth and synchronization;
- prompt/decode tokens and auxiliary calls;
- training, compilation, and serving costs separately.

Equal parameter count is not equal model bytes.  Equal FLOPs is not equal
latency.  A non-trainable buffer is not free.

### P7. Breakthrough-size derivation

Derive how the primitive could move the end metric by the required amount.  For
a targeted mechanism with maximum conditional improvement `delta` operating on
a fraction `f` of requests, the optimistic aggregate ceiling is

\[
\Delta_{total}\le f\,\delta-\Delta_{protected}.
\]

If even the optimistic ceiling is small, do not run.  If the mechanism changes
an asymptotic regime or creates a previously absent capability, state the
finite scale at which that change becomes material.

## Explainable block card

Every block must have one card before implementation:

| Field | Required content |
|---|---|
| name | one stable identifier |
| type | exact input and output shapes/dtypes |
| meaning | what each output represents |
| local construction | formula or executable pseudocode |
| invariant | exact or bounded property |
| assumptions | all distributional, numerical, and routing assumptions |
| resource function | bytes, operations, traffic, state, and depth as functions of shape |
| reference | smallest obvious implementation |
| independent check | a second form, exhaustive enumeration, or proof assistant where useful |
| microbench domain | exhaustive size or frozen sample construction |
| adversaries | zero, shuffle, random, permutation, collision, and shortcut controls as applicable |
| kill condition | one result that closes the exact block |
| non-claim | what a pass does not establish |

Required block families for a digital knowledge plane are:

1. **extractor:** raw prose to a declared statistic;
2. **writer:** statistic to persistent state;
3. **representation:** state meaning, code distance, and alias boundary;
4. **router:** query to candidate addresses without privileged labels;
5. **reader:** addressed state to evidence-bearing code;
6. **reasoner/decoder:** code to task behavior;
7. **physical operator:** exact mapping to the served hardware graph;
8. **composition:** the one interaction not already guaranteed locally.

A raw-token packer is a representation/writer, not a semantic extractor.  An
exact supplied handle is a router contract, not autonomous retrieval.  A large
teacher reading explicit text is an information ceiling, not proof that a
small bottleneck reader can use it.

## Microbench requirements

Each microbench must report:

- theorem or empirical claim being tested;
- domain size and whether it is exhaustive;
- reference and independent implementation;
- precision and numerical tolerance;
- positive, negative, shuffled, and shortcut controls;
- asymptotic and measured resources;
- frozen kill threshold;
- what decision changes on pass and on fail.

Representative tests:

| Block | Microbench |
|---|---|
| extractor | entity renaming, paraphrase, relation reversal, held-out compositions, label-leak audit |
| writer | exact oracle comparison, order permutations, collision/capacity sweep, adversarial documents |
| representation | exhaustive small codebook, quantized round-trip, perturbation margin, entropy efficiency |
| router | unseen aliases, distractor corpus growth, absent-key behavior, no privileged handle |
| reader | exact identity, random/zero/shuffle records, unseen keys/relations, margin distribution |
| decoder | code-distance sweep, nearest wrong code, calibration, counterfactual evidence |
| physical | target shapes, numerical equivalence, actual latency/traffic/workspace, compiled graph integrity |
| composition | stronger information ceiling followed by a small multi-seed causal A/B with matched controls |

## Two ladders, one join gate

Semantic admission and physical admission are separate:

```text
semantic: witness -> information -> addressing -> bottleneck -> reasoner
physical: algebra -> reference -> kernel -> complete serving graph
                                  \            /
                                   join gate
```

The join gate opens only when both ladders still support the same end claim.
A physical primitive may be retained independently, but it is not timed merely
because hardware is available.  Fatal uncertainties are resolved in this
order:

1. goal alignment;
2. information possibility;
3. strongest-control separation;
4. bottleneck sufficiency and learnability;
5. resource possibility;
6. target-hardware realization;
7. end-to-end composition.

Within the same fatality level, choose the test with the greatest expected
decision information per unit of time and money.

## LLM-assisted discovery loop

An LLM supplies breadth and transformations; the evaluator supplies truth.

1. Freeze a minimal witness and the strongest legal control.
2. Express a candidate as small typed `extract`, `write`, `route`, `read`, and
   `resource_count` programs.
3. Mutate one critical operator at a time while holding a correct skeleton.
4. Maintain separate populations for information coding, conditional
   computation, dynamical state, graph/permutation algebra, physical schedules,
   and optimization geometry.
5. Reject candidates on type errors, failed identities, counterexamples,
   ledger overflow, dominated controls, or sub-breakthrough ceilings.
6. Score surviving programs on proof margin, adversarial generalization,
   resource margin, simplicity, and projected end effect—not one noisy LM
   metric.
7. Require rediscovery of known small constructions as an evaluator sanity
   check; failure means the search representation or evaluator is broken.
8. Preserve diverse survivors and every counterexample.
9. Translate a survivor into a human-readable research packet and manually
   audit it before any expensive run.

This adopts the verifiable-program lesson of AlphaTensor, FunSearch, and
AlphaEvolve without pretending that natural-language intelligence is itself a
fully machine-checkable objective.

## Stop rules

- A theorem about expressivity does not authorize a performance claim.
- A FLOP inequality does not authorize a latency claim.
- An information ceiling from a stronger model does not authorize a reader.
- A supplied identifier does not authorize an autonomous-retrieval claim.
- A failed stronger-information oracle closes the representation.
- A matched-control construction that absorbs the operator narrows or closes
  the claim before execution.
- A dominated implementation is withdrawn before timing.
- A physical failure closes the mapping, not necessarily the abstract block.
- A small gain does not authorize scale under the breakthrough objective.
- A new candidate restarts at P0; it cannot inherit admission from a nearby
  passed primitive.

