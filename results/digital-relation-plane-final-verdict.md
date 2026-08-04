# Norm-compensated digital relation plane — final research verdict

Status: **validated digital-storage component; original production goal open**  
Date: 2026-07-31

This status supersedes the earlier interpretation that passing the synthetic
qualitative-acquisition gate completed the project.  It did not.  The result
below proves an exact storage and readout component, but the experiment supplied
already parsed IDs and therefore bypassed the language-understanding problem
that a smarter production model must solve.

## The result in one sentence

A standard causal Transformer can reserve an exact digital relation plane in
its existing weights, store 32,768 arbitrary 16-way facts at one robust payload
scalar per fact, and retrieve every fact exactly after language co-training,
while calibrated Muon and AdamW controls remain near 7–12% after two complete
fact passes and twice the mixed-step budget—without changing deployed parameter
count, inference graph, KV state, precision, or serving FLOPs.

## Why this is not a 1% result

The primary endpoint is a capability discontinuity:

| scale | seeds | candidate | Muon 2x | AdamW 2x |
|---|---:|---:|---:|---:|
| 36,577,152 parameters | 3 | **100.000%** | 9.498% | 7.335% |
| 110,776,960 parameters | 3 | **100.000%** | 11.856% | 7.273% |

Across six sealed seeds, the candidate made zero errors on 196,608 terminal
fact queries.  Every candidate checkpoint at steps 250, 500, and 1,000 was also
exact.  Worst-case BF16 margins were 2.625 at 36.6M and 6.625 at 110.8M.

The controls were given the answer token directly, not an indirect reasoning
objective.  Each 2x control saw all 32,768 facts twice, then received 2,000
mixed steps with additional fact queries.  Their failure is a failure of
ordinary gradient acquisition, not missing labels.

## Mathematical finding

For 32,768 independently arbitrary 16-way facts, the number of possible tables
is

```text
16^32768 = 2^131072.
```

Any zero-error representation must distinguish at least that many states and
therefore carry at least 131,072 logical payload bits.  The digital plane uses
32,768 independently settable 16-level scalars, exactly four logical bits each,
for exactly 131,072 logical payload bits.

This meets the counting lower bound in robust 16-ary payload cells.  It does
not claim physical bit optimality: FP32/BF16 storage has more physical bits and
the shared router/decoder plus isolation masks add overhead.

## Algebraic construction

Each entity embedding row stores 64 odd amplitude levels
`-15,-13,...,+15`, one per relation.  Raw amplitude packing normally fails in a
pre-RMSNorm Transformer because different entity rows acquire different RMS
scales.  One compiler-written compensation coordinate makes every entity row
have identical squared norm, so all 16 levels survive RMSNorm with fixed
calibration.

A six-bit relation query and two existing attention heads copy the selected
entity plane into the query.  Sixty-four shared SwiGLU channels route one
relation coordinate.  A 48-channel shared triangular decoder converts the
selected amplitude into a four-bit result code.  The decoder width does not
grow with entity/fact count at the tested dimensions.

At export the compiler is deleted.  The result is only ordinary embedding,
attention, RMSNorm, and SwiGLU weights in the same dense model.

## Protected language and resource results

At 110.8M, mean candidate natural NLL was 6.1221 versus 6.1351 for Muon 2x;
candidate wall time was 84.0 versus 207.0 seconds and estimated energy was
16.61 versus 45.05 kJ.

At 36.6M, candidate natural NLL was 6.1983 versus 6.1599 for Muon 2x (0.62%
worse), but 6.3491 for its compute-matched Muon 1x control.  This passes the
protected noninferiority rule but does not show a universal language-NLL
advantage over twice the natural training.

Candidate training HBM was modestly higher because frozen values/masks remain
resident: 4.72 versus 4.16 GB at 110.8M and 2.65 versus 2.47 GB at 36.6M.
Those are training-only costs.  Serving resources and operations are exactly
identical across arms.

## Prior-art boundary

The following are prior and are not claimed:

- Transformer FFNs behaving as key-value memories;
- synthetic factual-recall constructions with storage linear in parameters;
- product-key or external memory layers;
- compiling human-readable programs into Transformer weights;
- model editing and closed-form fact injection broadly.

The supported narrower contribution is:

> norm-compensated, amplitude-coded arbitrary relation payloads compiled into
> existing deep causal-Transformer weights, at the logical counting bound of
> one 16-ary cell per fact, with exact BF16 retrieval, unchanged serving graph,
> real-language coexistence, twice-exposed calibrated controls, and sealed
> replication at 36.6M and 110.8M parameters.

This satisfies the project contract's **intermediate synthetic**
qualitative-acquisition gate: frontier controls fail with 2x training while the
candidate is exact and protected slices remain noninferior, replicated at two
scales and three seeds per scale.  It does not satisfy the original production
objective because extraction, language-facing addressing, and composition were
not tested.

## What this does not yet solve

The compiler receives parsed `(entity, relation, value)` IDs, and the experiment
uses token rows absent from the natural stream.  It has not learned to extract
facts from ordinary prose, merge contradictory mentions, resolve aliases,
answer paraphrases, or compose multiple relations.

The result is therefore a proven dense-storage/training advance, not yet a
drop-in recipe for a generally smarter production LLM.  The next frontier is a
from-zero extractor that converts ordinary token sequences into this plane
without a pretrained teacher, while a learned front end maps paraphrases and
compositions onto its digital address space.

## Sealed evidence

- `digital-relation-plane-t8-train.json`: 110.8M, three seeds, 42/42 gates.
- `digital-relation-plane-t9-small-train.json`: 36.6M, three seeds, 42/42 gates.
- `digital-relation-plane-t8-quick.json`: exhaustive BF16 construction check.
- `digital-relation-plane-t9-small-quick.json`: untouched second-scale BF16
  construction check.
- Both optimizer pilots, preregistrations, source hashes, data hashes, and
  predecessor hashes are embedded in the sealed artifacts.
