# Scale-referenced whole-record plane T32 — Stage-0 decision

Status: **PASS CODEC; ADMIT ONE FROZEN NATURAL-INFORMATION ORACLE**  
Date: 2026-07-31

## Verdict

The exact full-record codec, common-scale BF16 decoder, frozen-corpus domain,
order invariance, and prospective entry ledger passed. T32 therefore advances
only to the previously specified full-record/local-window information oracle.

This is not yet a model result. No Transformer reader, training step, model
weight, evaluator label, or GPU operation entered Stage 0.

## Authoritative evidence

- 2,405/2,405 frozen raw-prose records round-tripped with zero collision;
- all records use 382 allocated cells plus two zero reserved cells;
- stored lengths span 16--128 tokens; 978 documents have raw encodings longer
  than the frozen 128-token record;
- observed stored token IDs span 0--49,137, inside the 49,152-row model
  vocabulary;
- all 576 clean BF16 digit/scale cases and 2,304 independently perturbed cases
  decoded exactly;
- 181,440 registered linear boundary comparisons had zero error;
- minimum clean boundary margin was `1.0*s`; minimum perturbed margin was
  `0.685*s`;
- title-sorted and reverse-input record digests both equal
  `a8f45105d3d1c1d7122ddc2fc20bb47eaad8613485e729414b59c427ee206cf7`;
- the complete provisional write ledger is 1,680,736 entries, or 4.595043% of
  the 36,577,152-entry checkpoint;
- local and remote unit suites passed 13/13;
- the corrected run used CPU only with CUDA hidden; the H100 remained at zero
  utilization and zero allocated memory after execution.

## Attempt-1 provenance erratum

The first preserved output reported `FAIL` because the verifier expected the
loaded `GPT2Tokenizer` to populate optional `_commit_hash` metadata. It did
not. A read-only cache audit showed that all actual tokenizer artifacts came
from the exact requested snapshot. Before a corrected run, the sole verifier
change and a no-scientific-output-change requirement were frozen in
`scale-referenced-whole-record-t32-stage0-provenance-erratum.md`.

The corrected run resolved and recorded all five tokenizer artifacts under
the exact snapshot and reproduced every scientific value from attempt 1
exactly. The failed attempt remains immutable; there is no hidden retry or
changed scientific threshold.

## What the pass means

T32 has removed T27's two representation defects: it stores 128 rather than
55 tokenizer slots, and any regular token is addressable from three cells
without sequential entropy decoding. The result also shows that the scale
reference survives the exact registered BF16 arithmetic.

It does not establish semantic sufficiency. Forty-one percent of documents
are truncated, exact evidence may still omit the needed fact, and a natural
question may not expose a usable raw-token key. It also does not establish
that an ordinary ten-layer Transformer can preserve, select, and reason over
the carrier without damaging its language state.

## Next sole gate

Before any physical reader or model training, freeze one oracle comparing:

1. aliased question only;
2. correctly routed decoded 128-token records;
3. the same records under a fixed document derangement;
4. deterministic label-blind local windows selected only from raw question
   and record token IDs;
5. the same local selector on deranged records.

Use complete forced-string answer likelihood and the paper's frozen 80%,
75%, 15-point, five-point, and ten-point gates. Failure closes T32 as a useful
natural payload. Passing admits a separate reader theorem and physical
microbenchmark, not training.

## Integrity

- authoritative result SHA-256:
  `cc4882785b27ce7e19ea443783dbeadc22e859a06e286804199d0210f656e232`
- preserved attempt-1 SHA-256:
  `4cebce4831d54bcdeb61a77cb145d40308d9fbf0d642c3c5038d108a5515cf5c`
- original preregistration SHA-256:
  `0c371b91db1ef3b4318e57223448f1b2be562f28a2bcff54b05697a08969cd12`
- provenance erratum SHA-256:
  `a010172b0fc701f9a628ee6ff54d2d94333642fe6969539650cc620a79786201`
- corrected source SHA-256:
  `552dc2bd190ef50d1632b0e669aacc4c84d8256919b7acea03f5b5d86786b716`
- corrected tests SHA-256:
  `6bad6f87c358ce870111b35019f83878e98899af631db7b701b44702ae83a0c0`
