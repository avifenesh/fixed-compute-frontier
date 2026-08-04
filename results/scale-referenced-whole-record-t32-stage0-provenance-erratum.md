# T32 Stage-0 tokenizer-provenance verifier erratum

Status: **FROZEN AFTER INVALID ATTEMPT 1, BEFORE CORRECTED RUN**  
Date: 2026-07-31

## Why attempt 1 is invalid rather than a scientific failure

Attempt 1 is preserved at
`results/scale-referenced-whole-record-t32-stage0-attempt1.json`, SHA-256
`4cebce4831d54bcdeb61a77cb145d40308d9fbf0d642c3c5038d108a5515cf5c`.
It reported `FAIL` solely because `tokenizer.init_kwargs["_commit_hash"]` was
`null`. All codec, corpus, numeric, determinism, resource, and isolation gates
were true.

A read-only audit then resolved `tokenizer_config.json`, `tokenizer.json`, and
`special_tokens_map.json` from the local Hugging Face cache. Every path was
inside the exact requested snapshot directory:

```text
/root/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M/
snapshots/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/
```

The loaded tokenizer's `vocab_file` and `merges_file` paths were in the same
snapshot. The requested tokenizer was therefore present; the verifier queried
an optional object field that this `GPT2Tokenizer` load path does not retain.

## Sole admitted correction

Replace the `_commit_hash` equality test with direct cache resolution of the
five tokenizer artifacts actually used:

- `tokenizer_config.json`;
- `tokenizer.json`;
- `special_tokens_map.json`;
- `vocab.json`;
- `merges.txt`.

Every resolved path must contain `/snapshots/<frozen revision>/`, exist as a
regular file, and be recorded in the result. The tokenizer name check remains.

No codec, source data, tokenizer revision, unit world, scale, perturbation,
threshold, ledger entry, or gate is changed. The original preregistration and
failed output remain immutable.

## Corrected-run gate

The corrected run passes only if:

1. all original gates pass under the corrected provenance predicate;
2. all corpus census values, record digests, numerical replay values, ledger
   values, and non-provenance gates exactly equal attempt 1; and
3. CUDA remains hidden and zero GPU operations are reported.

Any scientific-output difference, unresolved tokenizer artifact, or second
failure closes T32. There is no third run.
