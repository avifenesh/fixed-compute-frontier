# T38 sealed census — pre-materialization transport failure

Date: 2026-08-01  
Decision: **INCONCLUSIVE; NO CORPUS OR HELD-OUT STATISTIC MATERIALIZED**

The one-shot T38 CPU census opened its exclusive seal and then stopped while
fetching the frozen Hugging Face viewer blocks.  The server returned HTTP 429
after four transport retries.  `fetch_source` holds candidates in memory until
all three strata finish, so it wrote no corpus, manifest, finalization seal, or
result.

Preserved opening seal:

- file SHA-256: `5674a876b1a95023e886afe59f60c8a9ea7bc9eaa5fd5c292aa2d8457ceb1934`;
- implementation SHA-256: `3935bed4683fc280ef534e45fda5d4b6fc911e80a8c2413497c144dbf3a29261`;
- preregistration SHA-256: `6e1167fe160ce7d4dec764ead38654f9bc472cc0af6061f8d50332462ecd86e3`;
- tests SHA-256: `9a4b466467de8770a7c7480a46e44a46edaa98bb80dc01ca8ce756d57a19ce3e`;
- tokenizer SHA-256: `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c`.

This is not a T38 pass or fail.  The original seal and code remain untouched.
Any retry must use a separately named seal/result and a supplement frozen
before fetching.  It may change transport cadence only: source rows, offsets,
tokenizer, compiler, controls, statistics, and thresholds must remain byte-for-
byte or constant-for-constant identical.
