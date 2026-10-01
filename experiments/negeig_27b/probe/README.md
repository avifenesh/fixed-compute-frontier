# Generative probe: on-call sessions and S1, thinking off and on, tokens per task

Stage A showed wide = ctrl on teacher-forced S1 (both about 93.5% from an untouched 69.8%). This probe asks the
product question instead: what does each served arm do as an ON-CALL AGENT over long sessions (hundreds to thousands
of events, acting with tools mid-session), and how many tokens does it spend per task? With thinking on, the gain may
be "less thinking for the same answer", so every generation's reasoning is counted with the real tokenizer.

Arms: `untouched` (the base), `ctrl_s0` and `wide_s0` (train27.py `trainable_000700.pt` of the 1e-4 sweep, seed 0).

## Files

| file | job |
|---|---|
| `run_oncall.py` | drives each agentic_env item (`data/oncall.py` sessions, or `data/s2.py` items) turn by turn with its tools; scores each turn with agentic_env's `call_key` and `turn_score` |
| `run_s1gen.py` | answers S1 eval sessions turn by turn with the model's own answers in history; exact match against gold |
| `report.py` | per arm x task x mode tables, paired comparisons with bootstrap intervals; one JSON, one markdown file |
| `run_all.sh` | the box driver: per arm serve (evals/serve_arm.sh), check the parsers, run all jobs, take it down; then the report |
| `probe_common.py` | modes, token accounting, the per-generation ledger, the settings guard, `check-server` |
| `fake_probe_server.py`, `test_probe.py` | the CPU tests: an oracle fake with scripted defects and SGLang's context refusals, and 42 tests including red arms |

Reused, not copied: selfdistill's `ToolEpisode` (agentic_env's state machine at message level), `Client`, `JsonlSink`,
`run_pool`, `FailureGate`, manifests; agentic_env's `call_key`, `turn_score`, errors and step limit; `data/common.py`
(S1 rendering). `sd_common.Client.chat` gained two keyword arguments, both off by default so selfdistill is
unchanged (its 53 tests pass): `chat_template_kwargs` (it hard-coded think-off) and `extra_body` (for SGLang's
`routed_dp_rank`).

## Run it on the box

```bash
# on the rig: push code and data (probe/ rides along with negeig_27b/), the on-call items, the two trainables
experiments/negeig_27b/box/push_data.sh <box> --code-only
#   on-call eval items: data/oncall.py --split eval --n 40 --out <dir>, then copy oncall.eval.jsonl to
#   /workspace/negeig/data/oncall/ (or set ONCALL_ITEMS)
#   trainables: /data/ai-ml/models/_runs/negeig-27b/box/{a,c}/runs/sweep-lr1e-4/{wide,ctrl}_s0/trainable_000700.pt
#   to /workspace/negeig/runs/sweep-lr1e-4/{wide,ctrl}_s0/ (or set WIDE_TRAINABLE / CTRL_TRAINABLE)

# on the box
cd /workspace/negeig/code/experiments/negeig_27b/probe
./run_all.sh --print-cmd      # validate, print every command, change nothing
./run_all.sh                  # all three arms; rerun the same line after any stop or preemption
```

`run_all.sh` refuses a trainable whose sha256 is not the step-700 file on the rig (`WIDE_SHA256`, `CTRL_SHA256`;
`1122621b...` and `6017470c...`), and refuses to serve when the longest session plus the token cap plus a reply
allowance does not fit the context (CTX 131072). Measured with the real template: S1 subset 192 sessions, 14,686
turns a mode, longest 33.7k gold tokens, 58k needed thinking on; on-call 2048-event sessions up to 287 turns,
about 41k gold tokens, 75k needed thinking on.

Per arm it writes, under `$PROBE_ROOT` (`/workspace/negeig/probe`): `runs/<arm>/<task>_<mode>/{ledger,gens}.jsonl`,
`receipts/` (plan, trainable hash, serve record with engagement, server check, one receipt per job with exit code
and completeness), `logs/`, and at the end `report.json` and `report.md`.

Defaults (env): S1 `--lengths 256,1024 --per-cell 16` (16 sessions per domain and length, six domains, 4 Hebrew in each
of the four Hebrew domains); on-call `--per-length 16` (all four lengths of the file: 64, 256, 1024, 2048 events);
caps 16384 tokens per generation thinking on, 1024 off; `PARALLEL=1` (all four jobs of an arm at once);
`JOB_TIMEOUT_S` 8 h; `PURGE=1` (drop the 54 GB merged copy after an adapter arm). `GREEDY_MODES=off` adds a greedy
think-off measurement run. The header of `run_all.sh` lists every knob.

## What a run does

**Serving.** `evals/serve_arm.sh up` unchanged: merge-lora with its `delta_kept >= 0.95` gate, export-gates (ctrl
must be refused), the plugin only for wide, DP 8, the liveness probe, the engagement check from the server log.
serve_arm did not pass the parsers; `run_all.sh` adds `--tool-call-parser qwen3_coder --reasoning-parser qwen3`
through its `--extra`, turns the radix cache on (every turn re-sends the session), and keeps CTX 131072. Serve arm
names are `probe_<arm>` under `$PROBE_ROOT/serve`, so the general-eval merges and arm records are never touched.
Then `probe_common.py check-server` must pass: a tool call parsed in both modes, reasoning split out thinking on and
absent thinking off, a history message carrying `reasoning_content` accepted, and `routed_dp_rank` reaching rank 7.

**Data-parallel pinning.** SGLang 0.5.20 routes DP requests round robin, so consecutive turns of one session would land
on different ranks and re-prefill up to 75k tokens each time. Each session is pinned to one rank with the request's
`routed_dp_rank` (the DP controller honors it), round robin over the sessions in longest-first order, so its history
stays in that rank's radix cache.

**Modes** (checked against the Qwen3.8 `chat_template.jinja` by rendering it in the tests):

| | thinking off | thinking on |
|---|---|---|
| chat_template_kwargs | `enable_thinking: false` | `enable_thinking: true, preserve_thinking: false` |
| sampling | vendor non-thinking arm: 0.7 / 0.8 / 20, presence 1.5 | vendor thinking arm from generation_config.json: 1.0 / 0.95 / 20, presence 0 |
| history | every assistant turn with an empty think block (the training layout) | reasoning dropped from earlier turns, kept inside the current turn's tool loop |

The template's default is `preserve_thinking: true`: a client that sends reasoning back without the switch keeps every
earlier turn's reasoning, which is not Qwen3's old drop-earlier convention. `--history-reasoning keep|strip` gives
the other two layouts. Thinking on, the template adds its default "Reasoning effort is set to xhigh" instruction to
the system turn; `--reasoning-effort medium|low` changes it (recorded in the manifest). `--greedy` is temperature 0, a
measurement instrument only.

**Scoring.** On-call: agentic_env's `turn_score` per user turn (matched / max(expected, made) over canonical calls; a
no-call turn scores 1 when no tool is called). History carries the model's own messages, its tool calls and the canned
results those get (a wrong call gets `ERROR_NO_MATCH`), so errors compound. S1: `ok_norm` is the score (NFKC, whitespace,
wrapping `**` / backticks / quotes, one trailing period, case for English custody, toggles, ops and orders); order
and codetrace/fsys case are never normalized; `ok_strict` is reported beside it.

**Limits.** agentic_env ends an episode at a truncated generation or at 4 tool steps in one turn. Over a 287-turn
session that would let one runaway think erase every later turn, so the default `--on-limit continue` scores the
turn (0 when truncated; the calls it made when out of steps) and goes on. `--on-limit end` is agentic_env exactly.
Either way each such turn carries status `truncated` or `turn_budget` and the report counts them. S1 truncations score
0, are counted, and the session continues.

**Context limit.** SGLang 0.5.20 refuses (HTTP 400, never truncates) a request whose prompt plus `max_tokens` exceeds
CTX. When the prompt itself fits with at least 256 tokens to spare, the runner sends the request once more with the cap
clamped to the room left (the 16,384 cap is the harness's; a truncation at the clamped cap is counted like any other,
and `gens.jsonl` records the cap each generation got). When the prompt no longer fits, the session ends with status
`context_overflow`: a final row (the same history is refused on every rerun, so retrying it would block the job
forever), the refused turn and the unreached rest scored 0, and counted per cell in the report (`ctx overflow items`,
turns lost). Any other HTTP 400 stays `http_400`: not final, rerun on resume, kept out of the report.

**Token accounting** per generation, summed per turn and per task: `completion_tokens` (usage), `reasoning_tokens`
(the reasoning_content text through the real tokenizer: the primary measure), `server_reasoning_tokens` (SGLang
0.5.20 does report `usage.reasoning_tokens`: generated ids through `</think>`, one or two above the text count, the
whole completion when `</think>` never comes), `content_tokens` (completion minus reasoning: answer, tool-call markup,
delimiters), `visible_tokens` (the answer text), prompt tokens and latency.

**Resume.** Every generation goes to `gens.jsonl` as it arrives; a rerun replays an unfinished session through a fresh
episode (its state is a pure function of the generations) and continues live from there, so a preemption costs at most
the generations in flight. Final rows (including `context_overflow`) are never asked again; `http_400` rows are. A ledger written under other shaping settings (mode, sampling, cap, selection, policy,
seed, model) refuses new rows; use a fresh `--out-dir`.

## Reading the report

`report.md` has, per task and mode, one row per arm: accuracy over primary turns (on-call: the action turns; S1: every
answer) and by session length, tokens per turn (reasoning and content, median / mean; total mean), tokens per task,
accuracy per 1k reasoning tokens, truncation rates, unreached and step-limited turns, context-overflow sessions;
accuracy and reasoning per turn by request kind (on-call; the JSON also splits kind/variant, since `ack_open`'s two
phrasings read "open" differently) or by domain (S1); accuracy by position (events delivered before the turn); on-call
call-turn accuracy (all, and split at <= 4 vs >= 5 expected calls: agentic_env's 4-step limit caps a one-call-per-response
model at 4/n), no-call and events-turn rates, and the delivery turns' reasoning on their own; S1 strict and the answers
after event 256 and 512. Then the paired table: wide vs ctrl, wide vs untouched, ctrl vs untouched on the items both finished, A
minus B with a 95% bootstrap interval over items (turns of a session resampled together): accuracy, reasoning and total
tokens per turn, reasoning per turn on the turns both scored 1.0 (the "same answer, less thinking" reading), tokens per
task, accuracy per 1k reasoning tokens.

## Time per arm (estimate; the first arm measures it)

The think-on critical path is the longest session's chain of generations, not GPU throughput: the 2048-event on-call
sessions take about 330 sequential generations (287 turns plus the call turns' second step), the 1024-event S1
sessions about 137. At an assumed 30 to 40 decode tokens/s per stream under this load:

| piece | estimate |
|---|---|
| bring-up and down | base 10 min; ctrl and wide 20 to 25 min (merge-lora, export, load, graphs) |
| think off, all four jobs' worth | 10 to 15 min (330 generations at about 1 to 2 s each with pinned radix hits) |
| think on, R reasoning tokens per generation | R = 300: about 50 min; R = 1,000: 2.5 to 3 h; R = 2,000: 5 to 6 h |
| one arm | about 1.3 h (R = 300) to 3.5 h (R = 1,000) |
| three arms | about 4 to 10 h of the 8x H200 box (about $80 to $200 at $19.6 per preemptible box-hour) |

Aggregate tokens do not bind (about 23M generated tokens per arm at R = 1,000 against about 10k tokens/s at these
concurrencies, under an hour). R is the unknown: the untouched arm's first think-on minutes give it in the logs and
`gens.jsonl`. If it comes out high, the critical path shrinks by dropping the 2048-event on-call sessions from the
think-on run (`ONCALL_ARGS="--per-length 16 --lengths 64,256,1024"`), or by serving the three arms side by side on
GPU subsets (serve_arm.sh serves one arm per box today; that would be its extension).

KV capacity is the other unknown, and it can make the estimate optimistic. The earlier untouched server on this box
type (H200, DP 8, mem fraction 0.85, `--mamba-full-memory-ratio 0.9`) logged 579,825 KV tokens and a 31 GB mamba
state pool per rank (`box/a/logs/sglang_untouched.log`). serve_arm.sh defaults to `MAMBA_RATIO=3`, which by the same
split leaves about 256k KV tokens per rank (about 70 running requests). With `PARALLEL=1` each rank holds the pinned
sessions of all four jobs, about 1.2M tokens of live context near the end of the sessions (8 on-call sessions per
job, 24 S1 sessions per job, the S1 1024-event ones at about 30k each), so the radix cache cannot keep
most histories between turns: turns re-prefill and queue behind each other. Watch the server log's token usage and
queue length in the first minutes. Knobs, all the same for every arm: `MAMBA_RATIO=0.9` (or lower) for more KV and
fewer running requests (serve_arm.sh reads it from the environment), `PARALLEL=0` (one job at a time), lower
`CONC_ONCALL` / `CONC_S1`.

## CPU tests

```bash
cd experiments/negeig_27b/probe
nice -n 10 taskset -c 8-15 env OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= ~/.venvs/negeig/bin/python -m unittest -v test_probe
```

About 2.5 minutes, no GPU. The token tests use the real tokenizer at `/data/ai-ml/hf-models/qwen3.8-27b-tokenizer`
(`PROBE_TEST_TOKENIZER`), the on-call tests the Hebrew lane's agentic_env (`SELFDISTILL_LANE_DIR`). Red arms covered:
a wrong, omitted, extra or spurious call scores below 1; a truncated turn is counted as truncated and scores 0 (and
under `--on-limit end` leaves the rest unreached); a step-limited turn is counted (and under `--on-limit end` the
next turn is unreached, not a phantom reached turn); malformed arguments end the episode; a session that outgrows the
context gets its cap clamped while the prompt fits, then a final `context_overflow` row that a rerun never asks again
and that the report counts with its lost turns at 0 (SGLang's two refusal messages are parsed verbatim; any other 400
is rerun); reasoning tokens are counted exactly against the tokenizer and reach the report; S1 history holds the model's
own wrong answer and the fake flags gold in history; a server missing either parser, or with fewer DP ranks than the
runner pins to, fails `check-server`; `run_all.sh` stops on a failed check, a wrong trainable hash, or a context that
does not fit, and a rerun after a full run serves nothing. The `run_all.sh` tests drive the real script with a stub
`SERVE_ARM` that starts the fake.

## Open points

- The SGLang gate parity (G0 to G3) was measured with the radix cache off. This probe runs radix on for every arm
  (same launch for all three), which is the serving path; the parity of the gated kernel through radix hits is not
  measured. `RADIX=off` reruns any arm without it.
- `data/oncall.py` is another lane's work in progress; the tests build their own items in its contract and run the
  real generator's items only when it builds and verifies (it did at the last run).
