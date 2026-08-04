# T38R transport-only retry supplement

Date frozen: 2026-08-01  
Status: **FROZEN AFTER HTTP 429 AND BEFORE ANY CORPUS MATERIALIZATION**

T38R is a separately sealed transport retry of the T38 CPU census.  The first
attempt failed before writing a corpus or observing a statistic, as recorded in
`equivariant-template-engram-t38-fetch-failure.md` (SHA-256
`a9b97d71b44b670a40def70730cee99a5bc9d938036b75b10c4cef767b531def`).

## Immutable scientific protocol

All data sources, repository revisions, row counts, deterministic block
offsets, document selection and split laws, tokenizer, truncation, literal
partition, heads, compiler, record caps, controls, confidence quantization,
fallback, metrics, bootstrap, and decision gates remain those in the sealed
T38 preregistration with SHA-256
`6e1167fe160ce7d4dec764ead38654f9bc472cc0af6061f8d50332462ecd86e3` and
implementation SHA-256
`3935bed4683fc280ef534e45fda5d4b6fc911e80a8c2413497c144dbf3a29261`.
The base files may not be edited for T38R.
Before redirecting any base path, the wrapper must prove that the original T38
corpus, manifest, finalization seal, and result are absent and that the original
opening seal and failure report match their frozen hashes.

## Sole change: transport

- use separately named `t38r` corpus, manifest, opening seal, finalization
  seal, and result paths;
- retain the same 96 viewer requests (32 frozen blocks per stratum);
- space viewer-row requests by at least **3.1 seconds**;
- on HTTP 429 or 5xx, honor `Retry-After` when present and otherwise use
  capped exponential backoff, at most **8 attempts per request** and at most
  **120 seconds** for one delay;
- Hub metadata requests may use the same retry policy without the viewer
  cadence;
- write no partial corpus; any second failure remains sealed and is not retried.

The T38R opening seal must bind the original preregistration, original
implementation, original tests, this supplement, and the retry wrapper.  Its
companion must bind the opening seal, completed corpus, and manifest hashes.
No prepare-only path exists.

No gate changes are permitted after this supplement.  A completed T38R result
has exactly the evidentiary meaning defined by T38: a CPU decoded-memory
opportunity census, not a model, benchmark, serving, or GPU result.
