# Packed token evidence plane T27 — decoded-prefix oracle decision

Status: **FAIL — T27 CLOSED**  
Date: 2026-07-31

## Verdict

The first and only prompt-scoring attempt failed mandatory gate 5.  Qwen3.5-9B
produced an exact BF16 tie between the frozen `yes` and `no` token logits for
evaluation row `5adc53f75542996e6852530a` in the `full_document` condition.

The frozen protocol says an exact tie is invalid.  It also forbids changing
precision, scoring, prompts, labels, readers, thresholds, or retrying after an
observed result.  Therefore no accuracy estimate is reported and the
fixed-55-token natural payload is closed.

## Execution integrity

- Frozen preregistration, source, tests, Stage-0 artifact, data, model revision,
  Transformers version, and PyTorch version matched.
- Two earlier launches stopped during environment preparation, before any
  prompt was scored: first for a missing Accelerate dependency, then for a
  transient CUDA-library shadow introduced by its resolver.
- The isolated environment was restored to PyTorch `2.11.0+cu128`; the frozen
  source and artifacts were never edited.
- Exactly one run reached prompt scoring.  It validated 283 preceding margins,
  then aborted on the preregistered zero-margin condition.  There was no
  adaptive scoring retry.
- The H100 returned to zero allocated compute memory after the abort.

## What survives

T27 Stage 0 remains a valid component result:

- a 55-token tape is injectively represented by 220 radix-16 cells;
- all 65,536 token IDs round-trip exactly;
- the paired-SiLU identity constructs squared distance;
- the reference unique-match reader and fixed output code are exact;
- logical channel counts fit widths 880, 220, and 192.

What does **not** survive is the claim that this particular uncompressed prefix
is a sufficiently strong natural payload to justify physical integration or
from-zero LM training.  The next candidate starts again at Stage -2 and may
reuse only the proved digital primitives, not T27's admission.

## Research interpretation

This is an implementation-boundary failure of the oracle protocol, but a wall
for the T27 candidate as defined.  Relaxing the boundary would create a new
hypothesis after seeing the result.  The correct move is to retain the exact
algebra, close the payload, and search for a different explainable bridge from
raw prose to a fixed served representation.
