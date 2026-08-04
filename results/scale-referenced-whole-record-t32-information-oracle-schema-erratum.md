# T32 information oracle — sealed-schema access-boundary erratum

Status: **FROZEN AFTER PRE-SCORE SCHEMA STOP, BEFORE MODEL LOAD**  
Date: 2026-07-31

The package-correct launch verified artifact hashes, runtime, H100 identity, and
cached revisions, then stopped at the evaluator schema predicate. The source
incorrectly required rows to contain only `id`, `question`, and `answer`.
The sealed rows also contain `supporting_sentence_ids` and
`supporting_titles`. Their names were inspected after the stop; their values
were not accessed.

The preserved traceback is
`scale-referenced-whole-record-t32-information-oracle-prescore-schema-error.log`,
SHA-256
`a8892fc943f7bfceccc591f8075fc35a8077d790ba4bfbcbd14c5033bc4e9f9d`.
The output JSON was zero bytes, the model was not loaded, no prompt was built,
no answer value was accessed, no score was produced, and GPU allocation stayed
zero.

The sole admitted source correction is to require the actual five-key schema:

```text
answer, id, question, supporting_sentence_ids, supporting_titles
```

Prompt construction may access only `id` and `question`; final scoring may
then access `id` and `answer`. The two support values remain forbidden. No
record, selector, condition, prompt, continuation, model, scoring rule, or
threshold changes. Because zero model scores exist, the one registered scoring
run remains unspent.
