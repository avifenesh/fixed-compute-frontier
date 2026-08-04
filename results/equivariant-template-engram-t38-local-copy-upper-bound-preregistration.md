# T38 local oracle COPY upper bound — preregistration

Date frozen: 2026-08-01  
Status: **REJECTED PRE-RUN; ZERO CORPUS EXECUTIONS; NOT A T38 CENSUS RETRY**

An independent pre-run audit rejected the inference below before the corpus was
opened.  The implementation remains an unexecuted prototype; no result JSON,
opening seal, final seal, or corpus-derived statistic exists.

## Question

Can the fixed single-token, windows `{8,16,32}`, top-8192-literal T38 candidate
possibly satisfy its preregistered requirement that correct non-literal `COPY`
predictions cover at least 3% of all math/code token events?

This audit grants an oracle operation for every representable event.  It does
not compile a table, impose capacity, estimate purity, compare a lexical
control, or open either failed remote corpus.  Therefore its measured copyable
fraction is an optimistic upper bound on actual T38 `COPY` correctness.

## Frozen local sources

1. **prose diagnostic:** every row of
   `data/hotpot-real-prose-t12/candidate-raw-prose.jsonl`, text field only;
2. **math/reasoning optimistic slice:** every top-level `results/*.md` regular
   file, excluding this preregistration and both T38/T38R fetch-failure reports;
3. **code optimistic slice:** every regular `*.py`, `*.c`, `*.cc`, `*.cpp`,
   `*.cu`, and `*.h` file under `experiments/` and `tests/`.

Files are sorted by relative POSIX path.  Symlinks, generated JSON/logs,
checkpoints, data, and Python caches are excluded.  Each file/row is a separate
document.  The audit records path/document IDs and SHA-256 hashes.  These
project code and reasoning sources are deliberately structure-rich and
in-domain; they are not claimed representative.  This bias makes a failure
more informative and a pass only inconclusive.

## Frozen tokenization and quotient

- the exact sealed SmolLM2 tokenizer file with SHA-256
  `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c`;
- `tokenizers==0.22.2`, no inserted special token;
- first 1,024 token IDs per document; no cross-document context or EOS event;
- `L` is the 8,192 most frequent token IDs over the complete local audit
  corpus, with smaller ID breaking ties; all other IDs form `E`.

Using the complete corpus to choose `L` can only make the audit descriptive;
it does not create a held-out claim.  It exactly retains T38's frozen partition
size.

## Oracle statistic

For every token position `i >= 8`, target `y=x_i` is oracle-copyable when:

1. `y in E`; and
2. `y` occurs in `x[max(0,i-32):i]`.

The 32-token condition is the union of the three frozen heads, so it upper-
bounds any head arbitration.  Also report exact-window copyability for 8, 16,
and 32 and the unrestricted repeated-token fraction that ignores `L/E`.

Within a stratum, divide by all positions `i>=8`.  The decisive statistic is
the unweighted mean of the math/reasoning and code oracle-copyable fractions.

## Superseded decision

- The proposed rule was to close fixed single-token T38 when the equal
  math/code local statistic was below **3.0%**.
- This rule is invalid.  The statistic is an upper bound only for the fixed
  local corpus and its local `L`; it is not an upper bound for the frozen remote
  adjudication corpus and train-derived `L` in the T38 census.

The exact review and counterexample are recorded in
`equivariant-template-engram-t38-local-upper-bound-pre-run-audit.md`.  No local
CPU census execution is admitted by this document.  No GPU, rental, network
dataset request, or model training is admitted.
