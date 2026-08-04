# T32 whole-record/local-window natural-information oracle — decision

Status: **PASS INFORMATION; ADMIT READER ALGEBRA AND PHYSICAL MICROBENCH ONLY**  
Date: 2026-07-31

## Verdict

The exact T32 raw-token payload contains a breakthrough-sized amount of causal
natural reasoning information. The frozen full-record and local-window arms
both passed every absolute, retention, and shuffled-control gate.

This is the first result in this branch that establishes more than storage or
a local operator identity. It shows what can improve: on 104 title-disjoint
comparison questions, supplying the compiled raw evidence raises a fixed
reader from near chance to about 89% while wrong records remain near chance.

It does **not** establish that the unchanged 36.6M model can execute the
reader, that language quality survives the reallocated weights, or that
serving latency is unchanged. Those are now the only relevant uncertainties.

## Frozen results

| condition | correct | accuracy |
|---|---:|---:|
| question only | 48/104 | 46.1538% |
| shuffled full 128 | 50/104 | 48.0769% |
| shuffled local 48 | 50/104 | 48.0769% |
| correct full 128 | 93/104 | **89.4231%** |
| correct local 48 | 92/104 | **88.4615%** |

Causal deltas:

- full minus question only: **+43.2692 points**;
- full minus shuffled full: **+41.3462 points**;
- local minus question only: **+42.3077 points**;
- local minus shuffled local: **+40.3846 points**;
- local minus full: **-0.9615 points**.

Paired changes were 51 gained versus six lost for full against question-only,
and 45 gained versus three lost for local against shuffled-local. A diagnostic
exact paired-binomial calculation gives two-sided values `5.68e-10` and
`1.31e-10`; these were not preregistered gates. Diagnostic Wilson 95% intervals
are 82.05--93.99% for full and 80.91--93.28% for local.

## What the selector result says

The selector made 416 document-window decisions. Of those, 278 had zero
surviving exact query-token overlap and therefore used the frozen earliest
48-token fallback. Median start was zero and mean start was 2.226 tokens.
Nevertheless local evidence lost only one correct answer relative to all 128
tokens.

For this corpus, the key natural regularity is therefore not sophisticated
semantic retrieval. Wikipedia-style lead text places the comparison-bearing
fact early. The record needs an efficient bounded scan/reader, not a learned
semantic compressor. This is valuable but distribution-specific and must not
be generalized without a new corpus gate.

## Execution integrity

- all 104 questions routed exactly two distinct raw titles;
- all 2,405 records passed exact ID pack/unpack and decoded-text equivalence;
- all complete `yes<|im_end|>\n` and `no<|im_end|>\n` likelihoods were finite
  and non-tied;
- prompts and selections were frozen before answer values were accessed;
- support fields were never accessed;
- exact cached revisions, artifact hashes, H100 identity, Transformers
  `5.14.1`, and PyTorch `2.11.0+cu128` matched;
- one model-scoring run consumed 104.777 seconds, with 22.03 kJ trapezoidal
  sampled GPU energy and maximum prompt length 334 tokens;
- the H100 returned to zero allocated memory after the run.

Two pre-score harness corrections are preserved separately: package-mode
launch/schema access, and the proof-based replacement of the invalid
`encode(decode(ids)) == ids` assertion by the necessary decoded-text invariant.
Neither reached model loading or answer access, and no scientific condition,
threshold, or score changed.

## Next admitted problem

Prove a reader placement that uses the unchanged served model budget. The
reader must explain, layer by layer, how two routed records remain accessible,
how bounded slices enter existing K/V and hidden states, how query-conditioned
evidence accumulates, and where RMSNorm/residual contamination is controlled.

Before natural training, microbenchmark each block:

1. immutable record persistence across layers;
2. two-document layer-striped slice transport through existing K/V shapes;
3. exact or margin-bounded digit comparison and token recovery;
4. query-to-anchor mapping separated from factual payload;
5. parameter, FLOP, memory-traffic, KV, and p50/p95 latency equality;
6. protected-language capacity under exactly matched weight reallocation.

Failure of the reader construction retains the information result but closes
T32 as a production architecture. Passing admits one from-zero composition
experiment, not a production claim.

## Integrity hashes

- result SHA-256:
  `2295a64c66eb1eb77cdda8eb5176d1a86499b53ecc1aaf37f85b417163a290be`
- preregistration SHA-256:
  `2b8fa4f919b06ddb7fd88bc053d64e887cd778ea7301fae8b588cf6ed1392c4c`
- scored source SHA-256:
  `58431109b75675d145c76cd9d953e5ab7a5f4e6c0d42a01093c673de5fa5286f`
- tests SHA-256:
  `96620e9d9b44304b82db1c93744c67dc7dd8b41481c3ccfcee88487643ca47cf`
- scoring log SHA-256:
  `58c6cfe27b676255315acb7289b510d26c87b5d73eb8e7f3e798d4c52e1a1073`
