# T38/T38R empirical-protocol closure

Date: 2026-08-01  
Status: **FROZEN CENSUS CLOSED; ALGEBRA RETAINED; NO FURTHER FETCH OR RUN**

## Decision

The T38 canonical orbit and covariant `COPY/LITERAL` algebra remains valid.
The frozen T38 and T38R natural-corpus census does not.  It is closed rather
than retried or silently repaired.

No corpus statistic, model result, CPU census, local-GPU result, or rented-GPU
result exists for T38.

## The exact contradiction

The preregistration freezes the CodeParrot population at 477,249 rows and also
requires every fetched block to report `partial=false`.  Live Hugging Face
dataset-server metadata on 2026-08-01 reported, for
`transformersbook/codeparrot/default`:

```text
partial = true
num_rows = 477249
estimated_num_rows = 18715658
parquet_files = 10
parquet_bytes = 1822847978
```

The pinned repository itself describes roughly 22 million Python files and
about 50 GB compressed.  Therefore the 477,249-row universe is a partial
viewer derivative, not a demonstrated complete population at the pinned Git
revision.

The two possible responses both violate the frozen experiment:

1. accepting the 477,249 rows violates `partial=false` and lacks a
   revision-addressed byte manifest;
2. sampling the complete pinned shards changes the population size `R`, the
   row ordering, and every frozen offset because each offset is reduced modulo
   `R-31`.

The first attempt materialized no corpus before HTTP 429.  T38R materialized
no corpus before rejecting `partial=true`.  Consequently there is no old
corpus hash or row-order witness through which a different transport can prove
byte-for-byte equivalence.

This is not evidence against the T38 mechanism.  It is evidence that the
declared empirical protocol cannot be completed from the preserved artifacts.

## Why a third transport attempt is not admitted

T38R explicitly forbids a third remote retry.  More importantly, the live
metadata shows that a retry cannot turn the declared partial derivative into
the required complete population.  Rate limiting, another client library, a
Parquet reader, or a larger machine does not repair that logical mismatch.

A future study may prospectively choose either:

- an immutable partial viewer snapshot as its complete, narrowly scoped
  population; or
- a revision-addressed direct-shard population with a new manifest and
  sampling law.

Either is a new experiment.  It may not inherit the T38/T38R seals or be
reported as their completion.

## Stronger scientific boundary

Even a valid opportunity census would compare the quotient chiefly with a
concrete lexical dictionary.  A strongest control carrying the same orbit
canonicalizer, binding vector, relative interpreter, table bytes, and reader
executes the identical served function.  Thus T38 has no function-class
advantage over that control under the post-T35 absorption theorem.

The surviving possible claim is narrower:

> on a declared equivariant source family, explicit orbit factorization has a
> finite-sample or parameter-sharing advantage over surface-specific lexical
> memory, while the identical-quotient control ties by construction.

That is an E3 learnability/representation claim.  It needs its own theorem,
strongest controls, physical ledger, and end-effect equation before a newly
named natural census is justified.

## Retained results

Retain only:

1. the token-renaming orbit normal form;
2. the covariant `COPY/LITERAL` interpreter;
3. the exact orbit-size sharing factor against a concrete-key dictionary;
4. the within-corpus necessary ceiling for correct `COPY` events; and
5. the old frozen protocol as an integrity lesson, not an empirical result.

## Next admissible action

Reformulate the algebra under E3 and prove its finite-sample separation on a
declared orbit family.  Then independently audit whether its optimistic
repair-minus-harm ceiling can be material.  No data fetch or hardware work is
admitted by this closure.
