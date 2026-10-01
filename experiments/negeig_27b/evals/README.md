# Stage A general eval runner

Scores one arm of the negeig 27B retrofit on the "no regression" gates of `results/negeig-27b/PLAN.md` ("What decides",
item 3), each arm against the untouched model on the same box. It runs on the GPU box (8x B300, H200 fallback), not on
the rig. The arm is served through SGLang 0.5.20 with data parallelism, the five evals hit that one server, and
`compare_general.py` pairs the arm with the base item by item.

| gate | band (points) | eval |
|---|---|---|
| BFCL v4, every category, think off | 3 | `ev_bfcl.py` (17 categories, 21 gates in all with the four below) |
| MMLU-Pro, scored by option text (cloze) | 1 | `ev_mmlupro.py` (letter scoring is reported beside it) |
| HumanEval pass@1 | 3 | `ev_humaneval.py` |
| IFEval | 2 | `ev_ifeval.py` (prompt-level strict, see "Decisions for the owner") |
| Global-MMLU he | 1 | `ev_gmmlu.py` (full he-test) |

## Files

| file | job |
|---|---|
| `prepare_box.sh` | one-time box setup: evals venv, BFCL venv and pinned gorilla checkout, pinned data, nltk data, self check |
| `serve_arm.sh` | `up`: merge, export gates, launch SGLang with DP, wait for readiness, prove engagement. `down`, `status` |
| `run_general.sh` | all five evals for the arm being served, resumable, receipts per eval |
| `compare_general.py` | arm against base: paired bootstrap interval per gate, PASS or FAIL per band, one JSON, one markdown table |
| `ev_*.py`, `evalcommon.py` | the five evals and their shared journal, server client and exit codes |
| `arm_kind.py` | wide or ctrl from the trainable file, merge receipt check, engagement check on the server log |
| `prepare_data.py`, `pins.json` | pinned datasets and IFEval harness, normalized-file hashes |
| `bfcl/` | handler for `/v1/completions`, the patch applied to the pinned checkout, requirements |
| `run_tests.sh`, `test_*.py`, `fake_*.py`, `testlib.py` | the CPU tests (no GPU, no network, no real SGLang) |

The scripts derive their tree from their own location (`.../negeig_27b/evals`), so they work from wherever
`push_data.sh` mirrors the code (`$W/code/experiments/negeig_27b/evals`, `W=/workspace/negeig`). The mirror carries
`evals/` because it syncs the whole `negeig_27b/` directory. If the exec bit is lost, call them with `bash`.

## One-time box setup

```bash
# on the rig
experiments/negeig_27b/box/push_data.sh <box> --code-only          # or the usual push; evals/ rides along
# on the box
cd /workspace/negeig/code/experiments/negeig_27b/evals
./prepare_box.sh all          # evals venv, BFCL venv + gorilla at the pinned commit, data, nltk, then check
./prepare_box.sh check        # re-verify without changing anything; lists every gap, exit 1 if any
```

`prepare_box.sh all` needs network (uv, GitHub, Hugging Face, nltk data). It writes `$W/evalvenv`, `$W/bfcl-venv`,
`$W/gorilla`, `$W/evaldata`. Every dataset is checked against the sha256 in `pins.json` and the IFEval harness files
against their own hashes. `PREP_OFFLINE=1` skips the network steps that already have their result.

## One arm

Run the commands from the evals directory on the box. `NAME` is the label that ends up in every receipt.

```bash
# 1. serve (about 54 GB merged model per adapter arm, written under $W/merged/NAME)
./serve_arm.sh up --arm base --kind base
./serve_arm.sh up --arm wide_s1 --kind wide --trainable /workspace/negeig/runs/<run>/trainable_NNNNNN.pt
./serve_arm.sh up --arm ctrl_s1 --kind ctrl --trainable /workspace/negeig/runs/<run>/trainable_NNNNNN.pt

# 2. evaluate (resumable: after any stop, run the same line again)
./run_general.sh --arm wide_s1

# 3. stop the server, check the GPUs are free, optionally drop the merged copy
./serve_arm.sh down --purge
```

`--kind` must match the file: `wide` for a trainable with the fp32 gate W, `ctrl` for LoRA only. `serve_arm.sh up`
reads the file and refuses a mismatch. `base` takes no `--trainable`.

Before touching anything, `serve_arm.sh up ... --print-cmd` validates the arguments and prints the merge, export,
launch and environment lines. Same flag on `run_general.sh`.

### What `up` proves before it exits 0

1. The merge keeps the adapter: `MERGE-RECEIPT.json` must show `delta_kept >= 0.95` (`merge-lora`, scale 2.0,
   stochastic rounding, seed 20261001).
2. `wide`: `export-gates` writes the fp32 W file. `ctrl`: `export-gates` must refuse it ("no negeig_w tensors"), so a
   ctrl file that carries gates cannot be served as ctrl.
3. The server comes up under a deadline (`--ready-s`, default 2700) and `up` stops early if the server process dies.
4. A liveness probe asks 17 times 23 through chat and the answer must contain 391.
5. Engagement, read from the server log:
   - `wide`: exactly `dp x tp` engaged ranks, each logging `negeig: active ... 48/TP gated GDN layers ...
     sha256=<hash of the exported file> ... max|W|=<finite, above 0>`. The plugin writes each line twice (a bare print and
     logger.info, which SGLang renders as `[date DPn TPn] negeig: ...`), so the rank count is the larger of the two forms.
     `negeig: installed v<N> ...` lines (one per process that loads the plugin) are allowed when their version matches the
     active lines; any other `negeig:` line fails. A zero or nan gate, a wrong layer count, a rank that loaded a
     different file, a silent rank or a rank that engaged twice fails.
   - `base` and `ctrl`: no `negeig:` line at all (the plugin must not be loaded).

On any failure the server is stopped and the reason goes to stderr. On success it writes, in `$EVAL_ROOT/NAME/`,
`serve.json`, `engagement.json`, `arm.json` (with `engagement.ok`), `trainable.kind.json`, `launch.cmd`, `probe.json`,
and `$EVAL_ROOT/serving.json` (which arm is up). `run_general.sh` refuses to run if `serving.json` names a different
arm or `arm.json` does not say `engagement.ok`.

The launch is the same for every arm: all three `--linear-attn-*` backends are triton (the patch raises on a gated
layer otherwise, and a comparison needs one kernel set), bf16, radix cache off (logprob requests need it off), context
131072, no speculative decoding. `serve.json` records the launch and `compare_general.py` refuses to compare arms whose
context, radix, TP, backends, SGLang version, model revision, GPU, dtype or `--extra` words differ. DP and port may
differ.

`serve_arm.sh status` exits 0 only when the recorded server answers `/health`. `serve_arm.sh up --replace` stops a
running server first. Without it a second `up` is refused.

## The five arms

Order: base first (twice if you want the noise floor), then each adapter.

```bash
./serve_arm.sh up --arm base --kind base && ./run_general.sh --arm base && ./serve_arm.sh down
./serve_arm.sh up --arm wide_s1 --kind wide --trainable <file> && ./run_general.sh --arm wide_s1 && ./serve_arm.sh down --purge
./serve_arm.sh up --arm ctrl_s1 --kind ctrl --trainable <file> && ./run_general.sh --arm ctrl_s1 && ./serve_arm.sh down --purge
# optional noise floor: the base again under another label, same launch
./serve_arm.sh up --arm base_repeat --kind base && ./run_general.sh --arm base_repeat && ./serve_arm.sh down
```

Then, per adapter arm:

```bash
python compare_general.py --base /workspace/negeig/eval/base --arm /workspace/negeig/eval/wide_s1 \
  --base-repeat /workspace/negeig/eval/base_repeat \
  --out-json wide_s1.vs_base.json --out-md wide_s1.vs_base.md
```

Use `$W/evalvenv/bin/python` on the box. `--base-repeat` is optional. It adds the base-against-base difference next to
the arm's, so a delta can be read against run-to-run noise (BFCL samples at temperature 0.001).

Exit codes of `compare_general.py`: 0 written, 1 the inputs cannot be compared (the evals' pins or config differ, the
launch settings in `serve.json` differ, a results dir is missing; reason on stderr), 2 bad command line.
`--allow-config-diff` writes the report anyway and lists the differences under `protocol_problems`.
`--power-table` prints the power of the base's sample sizes and exits.

### Reading the report

Each gate gets one verdict from the paired bootstrap interval (95%, 10,000 draws, seed fixed, four paired cells drawn
multinomially) against its band:

| verdict | meaning |
|---|---|
| PASS | the whole interval is inside the band |
| FAIL | the whole interval is outside it |
| PASS_UNRESOLVED | the interval straddles the band edge and the point estimate is inside |
| FAIL_UNRESOLVED | the interval straddles the edge and the point estimate is outside |

UNRESOLVED means the sample is too small to decide at that band. It does not mean the arm is fine. The report prints,
per gate, the regression the sample would have caught 80% of the time. A category is flagged `coarse` when one item
moves the score by more than the band (`live_parallel`, `live_relevance`, `live_parallel_multiple`), and the overall
line is given twice, with and without the coarse ones. The report also lists report-only rows (`mmlupro_letter`,
`ifeval_loose`, the instruction-level IFEval rows, `gmmlu_he_pop2000`, `bfcl/all`), the arm validity checks
(engagement, merge receipt, kind) and the run-health notes (see "Caveats").

If the arm or the base has a failed engagement record, or no engagement or serve record, the report is written with
`valid: false`, the reasons in `validity_notes`, and the overall line `INVALID`. Per-gate verdicts alone are never the
decision. It stays with the owner.

## What each eval does

All five run as full sets by default (`--stride-<eval> 1`). Generation is temperature 0, except BFCL's stock 0.001
(MMLU-Pro cloze scores logprobs and generates nothing). Every chat eval is think off (`enable_thinking: false`).

| eval | n | how it is asked and scored |
|---|---|---|
| mmlupro | 12,032 | cloze (the gate): one batch request per question, the option texts as continuations of `The following is a question about {category}. ... Answer:`, score is the mean per-token logprob of the option, argmax. Letter scoring from the same server is report only. Radix cache off |
| humaneval | 164 | raw completion, greedy, 512 token budget (the HF run used 384). Each program in its own process: RLIMIT_AS 2 GiB, RLIMIT_CPU 10 s, 30 s wall |
| ifeval | 541 | chat, temperature 0, max_tokens 4096, the pinned google-research harness, langdetect seed fixed to 0 |
| gmmlu | 14,042 | chat, system message "Answer with only the letter", 5 dev shots per subject, max_tokens 1, byte-identical to the Hebrew lane's prompts (checked on all 14,042 rows). `pop2000` (seed 20260905) is reported beside the full run |
| bfcl | 4,441 | the pinned gorilla checkout through `bfcl generate` and `bfcl evaluate` with the handler in `bfcl/`: `/v1/completions`, the base model's chat template for every arm, temperature 0.001, 17 categories |

Why SGLang and not HF for MMLU-Pro: the merged adapter is scored exactly as it serves (merged bf16 weights, plugin
gate), eight DP replicas give eight times the throughput without eight 54 GB HF loads, and HF is slower on Gated
DeltaNet layers. The risk that comes with the port is the meaning of SGLang's logprob fields. `ev_mmlupro.py
--selfcheck N` (run first by `run_general.sh`, 48 items) pins it on the box: batch against single-sequence scores,
`logprob_start_len`, the tail token ids against the option ids, letter scores by token id against top-k, the predicted
letter against the greedy token. Exit 5 stops the run.

## Power against the PLAN bands

Interval half-width and the smallest regression caught 80% of the time, at a 4% discordant rate q (the fraction of
items where arm and base disagree), 95% interval. Computed with `compare_general.power_stats`; `--power-table`
prints the same for other q.

| gate | n | band | half-width | catches | reading |
|---|---|---|---|---|---|
| mmlupro_cloze | 12,032 | 1 | 0.36 | 1.51 | resolves a band of 1 at q up to at least 0.10 |
| gmmlu_he | 14,042 | 1 | 0.33 | 1.47 | same |
| humaneval | 164 | 3 | 3.06 | 7.37 | cannot certify a PASS at true zero when q >= 0.02 |
| ifeval_strict | 541 | 2 | 1.69 | 4.41 | cannot certify a PASS at true zero when q >= 0.04 |

BFCL, band 3, q = 0.04:

| category | n | half-width | catches |
|---|---|---|---|
| live_irrelevance | 884 | 1.3 | 4.9 |
| live_multiple | 1,053 | 1.2 | 4.7 |
| live_simple | 258 | 2.4 | 6.5 |
| irrelevance | 240 | 2.5 | 6.6 |
| simple_python | 400 | 2.0 | 5.8 |
| multiple, parallel, parallel_multiple | 200 each | 2.8 | 7.0 |
| multi_turn_* (4 categories) | 200 each | 2.8 | 7.0 |
| simple_java | 100 | 3.9 | 8.6 |
| simple_javascript | 50 | 5.5 | 10.9 |
| live_parallel_multiple | 24 | 8.0 | 14.4 |
| live_parallel, live_relevance | 16 each | 9.8 | 17.0 |

What this means: the two big accuracy gates (MMLU-Pro cloze, Global-MMLU he) are resolved at the 1 point band, so
their PASS and FAIL are real. HumanEval, IFEval and most BFCL categories are too small to certify a PASS at true
zero, so expect PASS_UNRESOLVED on a clean arm and read FAIL and FAIL_UNRESOLVED as the signal. The three tiny live_*
categories (16 to 24 items) cannot be resolved at any band; the full set is already all there is, there is no larger
subset to run. A stride would only make this worse, so none of the five evals defaults to one. `bfcl/all` (4,441
items) is the pooled number to read beside them.

## Pinned inputs

| input | pin |
|---|---|
| model | `Qwen/Qwen3.8-27B` @ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (checked against `.negeig_revision`) |
| MMLU-Pro | `TIGER-Lab/MMLU-Pro` @ `b189ec765aa7ed75c8acfea42df31fdae71f97be` (12,032 rows) |
| HumanEval | `openai/openai_humaneval` @ `7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544` (164 rows) |
| IFEval data | `google/IFEval` @ `966cd89545d6b6acfd7638bc708b98261ca58e84` (541 rows) |
| IFEval harness | `google-research/google-research` @ `e6890f85757dd84e27ca6df2dd30651dafad28e0` (`instruction_following_eval`, files hashed in `pins.json`); nltk 3.10.3, langdetect 1.0.9, immutabledict 4.3.1, absl-py 2.5.0; nltk data `punkt`, `punkt_tab` (no revision exists, see open risks) |
| Global-MMLU | `CohereLabs/Global-MMLU` @ `0e619dbeb34206cd48705a1a0ea7fb21cae09993` (he test 14,042, dev 285) |
| BFCL | `ShishirPatil/gorilla` @ `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`, registry key `negeig/qwen3.8-27b-FC`, python 3.12 |

`prepare_data.py` writes normalized jsonl files and the sha256 of each is pinned too. Each eval checks the hash of its
data file before it asks anything and exits 2 on a mismatch (`--allow-unpinned` skips that, tests only).

## Decisions for the owner

1. **IFEval gate.** PLAN.md says "IFEval within 2 points" without saying which number. The runner gates
   prompt-level strict accuracy and reports loose and the instruction-level rows beside it. Say if the gate should be
   another one; the change is small and local to `compare_general.py` (`BANDS`, `GATE_ORDER`, `REPORT_ONLY`).
2. **BFCL context window.** Every arm is served with context 131072. The Hebrew lane's base run had 44 of 200
   `multi_turn_long_context` items overflow 32,768, and an overflow is scored wrong, so a small window would put the
   gate on the window instead of the model. One value for all arms keeps the comparison paired. The cost is KV memory
   and some throughput; if 131072 does not fit a B300 beside the Gated DeltaNet state, lower `--ctx` for every arm
   including base and rerun the base.
3. **Unresolvable categories.** Whether `live_parallel`, `live_relevance` and `live_parallel_multiple` stay as gates
   or become report-only is a call on the PLAN's "every category". The runner keeps all 17 as gates.

## Time budget

The target is 30 to 45 minutes for the five evals of one arm on 8 B300, five arms. That is not measured yet. It
needs a box.

- `mmlupro` is prefill heavy: 12,032 cloze batches up to ~2,200 tokens each plus 12,032 letter prompts. `gmmlu` is
  14,042 prompts of a 5-shot context and one output token. Both stay fast only with DP and an open request queue.
- `ifeval` is 541 requests up to 4,096 tokens, `humaneval` 164 up to 512.
- `bfcl` is the long pole: 800 multi-turn items that loop through the server, several of them at long context.
- `run_general.sh --overlap` runs `bfcl` in the background beside the other four, so the GPUs stay full while the
  small evals tail. BFCL request concurrency is `16 x dp` by default (`BFCL_WORKERS`, or
  `--extra-bfcl "--workers N"`).
- Every `receipts/<eval>.status.json` and `receipts/run_general.json` records seconds. The first base run is the
  measurement.

If the first base run is over budget, cut in this order: `--overlap`, more BFCL workers, then a stride on the two
large accuracy evals (`--stride-mmlupro 2 --stride-gmmlu 2`; half-width grows by about 1.41, so about 0.5 points,
still inside the 1 point band). The same stride must then be used for every arm. BFCL, HumanEval and IFEval are
already at their power limit and are not strided.

## Resuming, smoke runs, subsets

Each eval keeps a journal under `<arm dir>/journal/<eval>.jsonl`. After a crash, a preemption, or a server restart,
run the same `run_general.sh` line again. Finished items are not asked again and a finished eval is skipped.
`run_general.sh` retries an eval that ended with some items failed (exit 3) for `--retries` passes (default 3, `--backoff-s`
20 between). Exit 1 and 2 are not retried. Exit 4 (the server is gone) stops everything.

The journal refuses a changed selection, so smoke runs go to their own `--out`, or add `--fresh`:

```bash
./run_general.sh --arm wide_s1 --out /workspace/negeig/eval/smoke_wide_s1 --limit 20 --selfcheck 8
./run_general.sh --arm wide_s1 --evals ifeval,humaneval            # a subset of the evals
```

`--limit N` takes the first N selected items of each eval (per category for BFCL). The result records the subset, and
`compare_general.py` prints "ran a subset, not the full set" for it.

`--extra-<eval> "ARGS"` appends words to one eval's command line. Everything that changes what is asked belongs on
every arm. `serve_arm.sh --extra "ARGS"` is recorded in `serve.json` and compared, so a base served with different
launch words is refused by `compare_general.py`.

Exit codes of `run_general.sh`: 0 all five complete, 1 an eval failed for good, 2 bad command line, 3 incomplete after
the retries, 4 the server went away.

## Caveats the report handles

- **BFCL transport errors.** A result row "Error during inference" is a transport failure, not a wrong answer. It is
  regenerated up to `--error-passes` times and then scored wrong. An error that survives all passes lowers the score
  of a normal category and raises the score of an irrelevance category, where an empty answer is the right one.
  `compare_general.py` prints the count per arm in the run-health notes. A base with 0 and an arm with 12 means a
  server problem, not the adapter.
- **Context overflow.** A request longer than the window is scored wrong and counted separately (`n_context_overflow`).
- **Noise.** BFCL temperature 0.001 and batch-dependent kernels make two runs of one arm differ. `--base-repeat`
  measures that.
- **Merged adapter against the HF-side LoRA.** The arm is scored as merged (bf16, stochastic rounding, delta kept
  >= 0.95), which is what would be served. It is not the training-time LoRA forward.

## CPU tests

```bash
./run_tests.sh                       # test_unit test_evals test_run_general test_bfcl test_serve_arm
./run_tests.sh test_serve_arm        # one module
./run_tests.sh test_unit -k bootstrap
```

Capped: `nice -n 10`, `taskset -c 8-15` (when those CPUs exist), `OMP_NUM_THREADS=2`, `CUDA_VISIBLE_DEVICES=` empty, a
scratch dir that is removed on exit, a timeout. No GPU, no network, no real SGLang. `unittest` only, no pytest.
`EVAL_TEST_PY` picks the interpreter (default `~/.venvs/negeig/bin/python`), `EVAL_TEST_CPUS`, `EVAL_TEST_TIMEOUT_S`.

What the tests cover:

- Argument parsing of all five evals, `run_general.sh`, `serve_arm.sh`, `prepare_box.sh`, `compare_general.py`.
- Scoring functions on hand-made items (cloze and letter scoring, HumanEval sandbox limits, the IFEval and BFCL
  parsers, Global-MMLU prompt construction).
- Comparison math on synthetic data, including a red arm that must FAIL, the UNRESOLVED cases and the config checks.
- `fake_server.py`, a tiny OpenAI-compatible server (chat, completions, logprobs, health), with dry runs of every eval
  and of `run_general.sh` against it, including server death, failed items, resume and a changed selection.
- `serve_arm.sh` against a fake launcher and a fake negeig CLI: base, ctrl, wide, TP 2, every engagement failure
  (silent rank, wrong gate, extra line, plugin loaded on base), bad merge receipt, server crash, ready deadline,
  liveness probe failure, replace and port guards, `down` and `down --purge`.
- `bash -n` and shellcheck on every `.sh`.

The real-data tests skip unless pointed at a prepared tree:

| env var | what it enables |
|---|---|
| `EVAL_TEST_DATA` | the directory `prepare_data.py` writes (normalized jsonl files, IFEval inputs and harness): the Global-MMLU prompt byte-identity check, the HumanEval canonical-solution check, the IFEval runs |
| `EVAL_TEST_IFEVAL_PY`, `EVAL_TEST_NLTK_DATA` | a python with the IFEval deps and the nltk data, for the real harness |
| `EVAL_TEST_BFCL` | a directory holding `bfcl-venv/`, `gorilla/` (a git checkout) and `bfcl-runs/`, for the BFCL parser tests on a real run copy |
| `EVAL_TEST_TOKENIZER` | the Qwen3.8-27B tokenizer directory, for the tests that tokenize (default `/data/ai-ml/hf-models/qwen3.8-27b-tokenizer`) |

## What only the box can prove

The CPU tests use fakes. The following has not been run against a real SGLang 0.5.20 and needs the first box run.

- That SGLang accepts the launch line (`--served-model-name`, `--dtype bfloat16`, the three triton linear-attn
  flags, `--mamba-full-memory-ratio 3`) and that context 131072 fits a B300.
- The engagement log shape under DP: one `negeig: active` line per rank (`dp x tp` ranks, 48/TP layers each), bare and
  through logging, plus `installed` lines. This is read from the plugin source and SGLang's `configure_logger`, not from
  a real 0.5.20 log. If the real log differs the engagement check fails loudly (it never passes on a shape it does not
  recognise), and the first wide launch is where that shows.
- Logprob semantics: `ev_mmlupro.py --selfcheck` pins them on the first run, tolerance 0.25.
- Throughput, and with it the 30 to 45 minute target.
- The network steps of `prepare_box.sh` (uv, git clone of gorilla, a shallow fetch by commit SHA, Hugging Face,
  nltk data).
- The radix cache is off for every arm, so its behavior with Gated DeltaNet state was never exercised here.
- That the merged adapter scores like the HF-side LoRA on the same prompts.
