# Long on-call sessions (oncall): build note

Built 2026-10-01 from `experiments/negeig_27b/data/oncall.py` (tests: `test_oncall.py`). Purpose: test the wide and
ctrl adapters where the product needs state tracking, an on-call agent with tools over long sessions, and give the
runner what it needs to measure accuracy and token use per task. English only, eval entity pools (`split_pool`).

## What an item is

One item is one on-call shift in the Hebrew lane's `agentic_env` multi-turn format (same `call_key` and turn
structure as `s2.py`; a turn is `{user, expect: {calls, parallel: false} | {no_call: true}, results, meta}`).

- Turn 0: `Initial state:` (services and versions, primary and secondary on-call, 2 to 5 incidents). No call.
- Delivery turns: `Events:` with 1 to 16 lines from `ops.Sim` (opens, acks, mitigations, resolutions, reopens,
  deploys, rollbacks, handoffs, swaps, and no-change alerts, notes, pages, canaries, maintenance, failed deploys).
  Expected: a short acknowledgement, no call.
- Action turns: a delivery plus a request, one position drawn in every 32-event stratum, marked with
  `meta.after_events`. The correct calls depend on the state at that moment.
- The agent's actions take effect. The next delivery starts with lines such as `The on-call agent acknowledges
  INC-4821.`, and the simulator state changes with them. The world is canned, like agentic_env's results: a missed
  call does not leave the rest of the session out of step with the text.

Tools (all parameters required): `acknowledge_incident(incident_id)`, `resolve_incident(incident_id)`,
`page_oncall(person, incident_id)`, `rollback_service(service, version)` (version = the one it runs after the
rollback). No read-only status tool: any status, version or rota lookup would hand the agent the state it must track.

| request kind | what it needs | no call when |
|---|---|---|
| `ack_open` | every incident whose status is open | none open |
| `resolve_mitigated` | every mitigated incident | none mitigated |
| `page_role` | current primary or secondary (swap and handoff chain) plus the incident's status | incident resolved (about 25%) |
| `page_sev` | role holder plus status and severity of every incident | no unresolved incident at that severity |
| `rollback` | the version before the newest on the service's stack (depth 2 to 9) | never |
| `rollback_if` | current version (condition) plus the previous one | condition false; `meta.decoy` says what the asked version is: the previous one (185), older in the stack (54), or other, not in the stack (79: never deployed, or deployed and since rolled back) |

The event lines are the S1 `ops` templates the arms trained on (train pools there, eval pools here). What is new:
tool calls, the request shapes, acting mid-session, and sessions to 2,048 events (S1 eval stops at 1,024).

## Fixed eval set

`/data/ai-ml/models/_runs/negeig-27b/data/oncall/oncall.eval.jsonl`, sha256
`8e3e2450ec44858842d3ab78ca00437b21a4568e21396a35f273961b12c60ea9` (deterministic in the seeds, rebuilt twice to the
same hash). Beside it: `oncall.eval.census.json` (per session token counts, prompt tokens at every action turn),
`oncall.eval.report.json`, `build.log`.

It replaces the first build, `ac3215da...`. User text, expectations, results and the census are identical. Only 94
`rollback_if` decoy labels changed. The first build labelled the decoy by the branch that drew it, and a drawn older
tag or proposed version can be anything: 13 "proposed" or "older" decoys were the previous version, 2 "proposed" were
older in the stack, and the other 79 "proposed" are now "other" (41 of them had run on the service before, so the
first note's "never applied" was wrong). Nothing had read the first file.

| events | sessions | turns (mean) | action turns | call turns | no-call action turns | calls | most calls in a turn |
|---|---|---|---|---|---|---|---|
| 64 | 40 | 10.8 | 80 | 53 | 27 | 70 | 3 |
| 256 | 40 | 36.2 | 320 | 209 | 111 | 290 | 6 |
| 1,024 | 40 | 142.1 | 1,280 | 813 | 467 | 1,120 | 8 |
| 2,048 | 40 | 279.7 | 2,560 | 1,563 | 997 | 2,269 | 8 |
| all | 160 | | 4,240 | 2,638 | 1,602 | 3,749 | |

By kind (all lengths, no-call in brackets): ack_open 776 (405), page_role 1,016 (373), page_sev 593 (288),
resolve_mitigated 593 (218), rollback 617 (0), rollback_if 645 (318). Action turns by position: events 1 to 64: 320,
65 to 256: 720, 257 to 512: 640, 513 to 1,024: 1,280, 1,025 to 2,048: 1,280. Calls per call turn: 1 in 2,097
turns, 2 to 8 in 541. State sizes: up to 161 incidents tracked, unresolved at a request 3 on average (max 12).
Ages at the request: role last changed 1, 3, 6 events ago (quartiles; swaps come every 8 events or so, so the role
is a long chain of transpositions, not an old fact to recall); the named incident or service last changed 5, 14, 39
events ago (quartiles), max 1,731.

## Verification

- `verify()` passes on all 160 sessions. `replay()` reads only the rendered user turns: its own regexes for every
  ops line, agent line and request (no simulator, no templates), the state machine with legal-transition checks,
  and the expected calls per turn. It also checks that the agent lines repeat the previous turn's expected calls,
  that `after_events` counts the simulator lines, and that `results` holds exactly the expected calls' answers.
  Blanking every expectation, result and meta field leaves the replay's answers unchanged (test).
- Every turn's meta must equal the replay's labels, key for key: request kind and phrasing (`action`, `variant`),
  `role`, `sev`, `entity`, `decoy`, stack `depth`, state sizes (`n_incidents`, `n_open`, `n_mitigated`,
  `n_unresolved`, `n_calls`), ages since the last change (`role_age`, `entity_age`), and line counts
  (`batch_events`, `agent_lines`). A meta key the replay cannot check fails the item.
- Red test (in the build, which fails on any miss): 2,594 corrupted copies, 2,594 caught. Expectations: drop a call
  104/104, call turn to no call 153/153, extra call 153/153, wrong argument 153/153, wrong tool 124/124, page the
  other role's holder 136/136, roll back to the current version 122/122, the tempting call on a no-call action turn
  137/137, a call on a delivery turn 160/160, `after_events` off by one 160/160, result dropped 153/153, result
  wrong 153/153, agent line dropped 153/153, agent line naming another incident or version 153/153. Each of these
  keeps results and counts consistent, so only the semantics can fail it. Labels: wrong kind 160/160, age off by one
  155/155, wrong decoy 111/111, a count off by one 154/154.
- Generator bugs injected by monkeypatching, caught on the full set: actions reported but not applied to the
  simulator, 121 of 160 sessions fail; `page_role` paging the other role, 125 of 160 fail. In the tests also: ages
  off by one, and a decoy label that disagrees with the stack. With the label comparison removed from `verify()`,
  those tests and the label arms of the red test fail (mutation check).
- A separate parser written for this review (plain string splitting, none of `replay()`'s regexes) gives the same
  expected calls on all 18,753 turns, and the same ages and state sizes on all 4,240 action turns.
- Through the existing episode loop (`selfdistill/tool_gen.ToolEpisode` with the lane's `agentic_env.turn_score` and
  `call_key`, generation cap raised): an oracle scores 1.0 on every turn of the 8 sessions tested; dropping one of n
  calls scores (n-1)/n on that turn only; a policy that never calls scores 0 on call turns and 1 elsewhere.
- `test_oncall.py`: 20 CPU tests, 13 s, `nice -n 10 taskset -c 8-15`, `OMP_NUM_THREADS=2`, no GPU.

## Token census (Qwen3.8 tokenizer and chat template, tools rendered as SGLang renders them)

Ideal transcript: expected calls, canned results, replies "Noted." / "Done." / one sentence for nothing to do.
History carries no reasoning. Think-on adds the reasoning-effort line to the system prompt (38 tokens). System
prompt plus tools: 951 tokens think-off, 989 think-on.

| events | final context mean / max (think-on) | room left in 131,072 | generations per session (mean / max) | reasoning that fits per generation if kept in history |
|---|---|---|---|---|
| 64 | 2,369 / 2,806 | 128,266 | 12.1 / 16 | 8,032 |
| 256 | 5,936 / 6,888 | 124,184 | 41.5 / 47 | 2,659 |
| 1,024 | 20,191 / 22,404 | 108,668 | 162.4 / 180 | 609 |
| 2,048 | 39,120 / 42,727 | 88,345 | 318.8 / 339 | 265 |

About 19 tokens per event. 2,048-event sessions fit with at least 88k tokens of room per generation. No cap needed.

How the census was checked. The SGLang v0.5.20 source (`serving_chat.py`, `protocol.py` at the v0.5.20 tag) renders
the request the way `office_env.server_messages` and `template_tools` model it: the `Tool` model is unchanged since
0.5.19 (`strict: false`, tool-level `defer_loading: null`), history tool-call arguments are parsed to objects, and
`reasoning_content` passes through to the template. A separate render of the largest session per length with
`apply_chat_template` gives the same counts token for token (2,806 / 6,888 / 22,404 / 42,727 think-on).

The census renders history with `preserve_thinking` unset, so every past assistant turn carries an empty think
block. The runner's default (`run_oncall.py --history-reasoning drop`: reasoning sent back, `preserve_thinking=false`)
drops the block on turns before the last user message, about 4 tokens less per past assistant message (41,419 instead
of 42,727 in the largest 2,048-event session). The census is an upper bound for that mode.

The room assumes the ideal replies ("Noted."). Two things eat it in a real run: the model's own replies, which stay
in history, and, in drop mode, the reasoning of the current turn's tool loop. At the runner's think-on cap of 16,384
per generation, a 4-step loop could hold 3 x 16,384 of kept reasoning plus the 16,384 being generated: 42.7k + 65.5k
= 108k of 131k, which leaves about 23k, or about 80 tokens per turn of replies beyond "Noted.", in the largest
2,048-event session. If the model writes long acknowledgements, a late turn of a 2,048-event session can get an
HTTP 400, and `run_oncall.py` ends the episode there. Watch `prompt_tokens_last` per turn and the `http_400` count.

## For the runner

1. **Score action turns, not the mean turn reward.** Turns without a request are 77 to 81% of all turns, and 86 to
   88% of turns expect no call: a policy that never calls gets mean reward 0.86 to 0.88 and action-turn accuracy
   0.34 to 0.39. Report call turns and no-call action turns separately, by `after_events` bucket and by kind. The
   no-call share drifts from 34% (64 events) to 39% (2,048), so compare within kind and class, not raw pooled
   accuracy across lengths.
2. **Budgets.** agentic_env's `MAX_GENERATIONS` is 12 and both episode loops end the whole episode at a turn budget
   or a truncated generation. Sessions here need up to 339 generations without retries. Give each turn its own step
   budget, close a failed turn and continue; the agent lines keep the text consistent after a miss.
3. **Do not keep past reasoning.** The template keeps past `reasoning_content` by default (`preserve_thinking`
   unset means true). Kept, a 2,048-event session leaves 265 reasoning tokens per generation. `run_oncall.py`'s
   default `--history-reasoning drop` is right: it sends reasoning back with `preserve_thinking=false`, so the
   template keeps it only inside the current turn's tool loop. `--history-reasoning keep` must not be used on these
   sessions. `max_tokens` must stay at most 131,072 minus the prompt (prompt per action turn is in the census).
4. **Turn the prefix cache on.** `serve_arm.sh` defaults to `--radix off`. Without reuse the 160 sessions prefill
   291M prompt tokens per arm and think mode; with reuse about 2.7M. Nothing here needs logprobs. Check the server
   log's cached-token counts on the first sessions (hybrid GDN prefix matching).
5. **Token use per task.** Per generation keep completion tokens and reasoning length. Sum them per action turn
   (all steps of the turn) and report delivery turns separately: the "less thinking" reading is reasoning tokens on
   action turns at matched accuracy, per kind and position.
6. **Cost shape with thinking on.** 21,391 generations per arm and mode (ideal). The 2,048-event sessions are chains
   of about 320 dependent generations, so their wall time is 320 x reasoning tokens per generation / per-sequence
   decode rate. Measure that rate on the first sessions. `--max-batch 32` would roughly halve the generations, but
   it is a different set: rebuild and re-verify it, do not mix it with this one. `run_all.sh` defaults to
   `--per-length 16` (the first 16 sessions per length by id, 64 of 160), so its counts are 40% of the ones above.
7. **The 4-step limit caps turns with many calls.** agentic_env's `MAX_STEPS_PER_TURN` is 4, and `run_oncall.py`
   keeps it. 156 of 2,638 call turns need 4 to 8 calls, 70 need 5 or more. A model that makes one call per response
   scores at most 4/n on those 70, and on the 86 turns needing exactly 4 its turn closes with status `turn_budget`
   even when every call is right. The system prompt asks for independent actions together, but a model that issues
   calls one at a time loses score there for its calling style, not for state. Report call-turn accuracy split at
   n_expected <= 4 and >= 5, and the count of turns closed at `turn_budget`, per arm.
8. **Reading the results.**
   - The agent lines report the correct calls whatever the model did. After an action turn the incidents and
     services it touched are back in their true state, so errors on them do not compound through actions, and when
     the model got the turn wrong, the next delivery contradicts its own history. The arms see the same text, so the
     comparison is paired, but accuracy here is state tracking with corrected action outcomes.
   - `ack_open` variant 1 says "whose status is open"; variant 2 says "currently open", which in ops usage can mean
     unresolved. The two readings differ on 564 of 776 `ack_open` turns. The system prompt and the tool description
     define open, but compare `ack_open` per `meta.variant` before reading a kind-level difference.
