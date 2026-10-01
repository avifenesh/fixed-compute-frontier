#!/usr/bin/env python3
"""Generative on-call probe: drive each agentic_env item (data/oncall.py sessions, or any item in that format such as
data/s2.py) through an OpenAI-compatible SGLang server turn by turn with its tools, and score every turn with
agentic_env's own call_key and turn_score.

The episode is selfdistill's ToolEpisode (agentic_env.AgenticEpisode.observe at message level), imported. History
carries the model's own assistant messages, its tool calls and the canned results those calls get (a wrong call gets
agentic_env's ERROR_NO_MATCH), so errors compound as they would in use. A turn whose expectation is no_call scores 1
when no tool is called (turn_score). ProbeEpisode below adds, without changing the scoring:

  * per-turn records: calls expected and made, score, status, and the token accounting of every generation in it
    (probe_common.gen_record: prompt, completion, reasoning, content, visible tokens, finish, latency)
  * the reasoning of each generation kept on its assistant message when the mode sends reasoning back (the chat
    template decides what is rendered, see probe_common)
  * a limit policy. agentic_env ENDS the episode when a generation is truncated (finish "length") or a turn reaches
    MAX_STEPS_PER_TURN tool steps; over a 240-turn session one runaway think would erase every later turn.
      --on-limit continue (default)  the turn scores 0 when truncated (its partial tool calls are not executed; the
                                     returned content stays in history), or keeps the turn_score of the calls it made
                                     when it ran out of steps; the session goes on with the next user turn
      --on-limit end                 agentic_env exactly: the episode stops, later turns are unreached (score 0)
    Either way every limit is counted per turn (status truncated / turn_budget) and reported, never folded in silently.
    Malformed tool-call arguments still end the episode under both policies (SGLang refuses the next request with
    them in history), status bad_arguments.
  * the generation budget is the item's n_turns x MAX_STEPS_PER_TURN (agentic_env's 12 is for 1-3 turn items); it
    cannot bind before the per-turn step limit does.

Resumable: final rows go to ledger.jsonl (one per item x sample); every generation also goes to gens.jsonl as it
happens, and a resume replays an unfinished session's generations through a fresh episode (the episode state is a
pure function of them) and continues with live requests from there. Context limit (probe_common.chat_fit): a request
whose prompt fits but prompt + cap does not is retried once with the cap clamped to the room left; a prompt that no
longer fits ends the session with status context_overflow, a FINAL row (scored 0 from that turn on, counted), since
the same history is refused on every rerun. Any other HTTP 400 is recorded as http_400 and rerun on resume, as
selfdistill does.

  run_oncall.py --items DATA/oncall/oncall.eval.jsonl --thinking on --arm wide_s0 --out-dir OUT [--per-length 16]
  run_oncall.py ... --plan          selection, turn counts, the longest gold-transcript context (no server)
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
import time
import types
from pathlib import Path

import probe_common as pc
from probe_common import ApiError, FailureGate, JsonlSink, log, read_jsonl, seed_for

import tool_gen  # selfdistill, through probe_common's sys.path  # noqa: E402
from s2 import call_key as s2_call_key  # data/, agentic_env's call_key copied verbatim (s2 --selftest)  # noqa: E402

LIMIT_POLICIES = ('continue', 'end')
NAME = 'oncall'


def load_lane():
    """(office_env, agentic_env) from the Hebrew lane, imported and never modified."""
    sys.dont_write_bytecode = True
    lane = str(pc.sd_common.LANE_DIR)
    if lane not in sys.path:
        sys.path.append(lane)
    import agentic_env
    import office_env
    return office_env, agentic_env


def env_shim(office_env, max_tokens):
    """office_env as ToolEpisode reads it, with this run's cap per generation."""
    return types.SimpleNamespace(MAX_TOKENS=int(max_tokens), TRUNCATED=office_env.TRUNCATED,
                                 TURN_BUDGET=office_env.TURN_BUDGET, HTTP_400=office_env.HTTP_400)


def ag_shim(agentic_env, item):
    """agentic_env as ToolEpisode reads it: its own call_key, turn_score, errors and step limit; a generation budget
    sized to the item."""
    budget = (item.get('meta') or {}).get('max_generations') or len(item['turns']) * agentic_env.MAX_STEPS_PER_TURN
    return types.SimpleNamespace(call_key=agentic_env.call_key, turn_score=agentic_env.turn_score,
                                 ERROR_UNKNOWN=agentic_env.ERROR_UNKNOWN, ERROR_NO_MATCH=agentic_env.ERROR_NO_MATCH,
                                 MAX_STEPS_PER_TURN=agentic_env.MAX_STEPS_PER_TURN,
                                 MAX_GENERATIONS=max(budget, agentic_env.MAX_GENERATIONS))


def keys_of(gen, ag):
    """Canonical keys of one response's tool calls, computed as ToolEpisode does (bad JSON gets a marker)."""
    out = []
    for c in gen.tool_calls or []:
        name = c['function']['name']
        try:
            a = json.loads(c['function']['arguments'])
        except (ValueError, TypeError):
            out.append(name + ' <bad-json>')
            continue
        out.append(ag.call_key(name, a) if isinstance(a, dict) else name + ' <non-object>')
    return out


class ProbeEpisode(tool_gen.ToolEpisode):
    """ToolEpisode plus per-turn records, reasoning kept on the assistant message, and the limit policy."""

    def __init__(self, item, env, ag, mode, policy):
        super().__init__(item, env, ag)
        assert policy in LIMIT_POLICIES, policy
        self.mode, self.policy = mode, policy
        self.turn_recs = []
        self.cur = {'gens': [], 'made': []}
        self.limits = collections.Counter()
        self.gen_budget_hit = False

    # -- turn bookkeeping
    def _finish_turn(self, t, score, status):
        turn = self.item['turns'][t]
        expect = turn['expect']
        meta = turn.get('meta') or {}
        exp = [] if expect.get('no_call') else [self.ag.call_key(c['name'], c['arguments']) for c in expect['calls']]
        rec = {'t': t, 'type': meta.get('type'), 'action': meta.get('action'), 'variant': meta.get('variant'),
               'decoy': meta.get('decoy'), 'after_events': meta.get('after_events'),
               'expect': 'no_call' if expect.get('no_call') else 'call', 'parallel': bool(expect.get('parallel')),
               'n_expected': len(exp), 'expected': exp, 'made': self.cur['made'],
               'n_made': sum(len(s) for s in self.cur['made']), 'score': float(score), 'status': status,
               'reached': True, **pc.sum_gens(self.cur['gens'])}
        self.turn_recs.append(rec)
        self.cur = {'gens': [], 'made': []}
        if status in ('truncated', 'turn_budget', 'gen_budget', 'context_overflow'):
            self.limits[status] += 1

    def next_request(self):
        was_done = self.done
        out = super().next_request()
        if out is None and not was_done:  # ToolEpisode ended the episode here: the generation budget ran out
            self.gen_budget_hit = True
        return out

    def _next_turn_or_end(self):
        if self.turn_index == len(self.item['turns']):
            self._end()
        else:
            self.messages.append({'role': 'user', 'content': self.item['turns'][self.turn_index]['user']})

    # -- the episode interface
    def observe(self, gen, rec=None):
        t, n_msgs, n_scores = self.turn_index, len(self.messages), len(self.turn_scores)
        self.cur['gens'].append(rec or {})
        truncated = gen.finish_reason not in pc.STOP_FINISH
        if not truncated and gen.tool_calls:
            self.cur['made'].append(keys_of(gen, self.ag))
        if truncated and self.policy == 'continue':
            # ToolEpisode.observe's own bookkeeping, then the turn fails and the session goes on
            self.generations += 1
            self.gens.append({'finish': gen.finish_reason,
                              'completion_tokens': int(gen.usage.get('completion_tokens') or 0),
                              'prompt_tokens': int(gen.usage.get('prompt_tokens') or 0)})
            self.reasoning_chars += len(gen.reasoning or '')
            self.messages.append(pc.assistant_message(gen, self.mode, tool_calls=False))
            self.turn_scores.append(0.0)
            self.turn_index += 1
            self.steps = []
            self._finish_turn(t, 0.0, 'truncated')
            self._next_turn_or_end()
            return
        super().observe(gen)
        if (self.mode.attach_reasoning and gen.reasoning and len(self.messages) > n_msgs
                and self.messages[n_msgs].get('role') == 'assistant'):
            self.messages[n_msgs]['reasoning_content'] = gen.reasoning
        if len(self.turn_scores) > n_scores:  # the turn closed
            if self.done and self.errors and self.errors[-1] == self.env.TURN_BUDGET:
                self._finish_turn(t, self.turn_scores[-1], 'turn_budget')
                if self.policy == 'continue':  # ToolEpisode ended the episode at the step limit; resume the session
                    self.done = False
                    self.errors.pop()
                    self._next_turn_or_end()
            else:
                self._finish_turn(t, self.turn_scores[-1], 'scored')
        elif self.done:  # stopped inside the turn: truncated under --on-limit end, or malformed arguments
            e = self.errors[-1] if self.errors else ''
            self._finish_turn(t, 0.0, 'truncated' if e == self.env.TRUNCATED else 'bad_arguments')

    def http_error(self, detail='', overflow=False):
        """The server refused the request. overflow: SGLang's context-length refusal (probe_common.context_refusal),
        a final outcome of the session; any other refusal stays http_400 (rerun on resume)."""
        self.http_detail = detail
        self.overflow = bool(overflow)
        t = self.turn_index
        super().http_error()
        if t < len(self.item['turns']) and len(self.turn_recs) == t:
            self._finish_turn(t, 0.0, 'context_overflow' if overflow else 'http_400')

    def finalize(self):
        """Close the turn the episode stopped in (generation budget) and mark the rest unreached. A turn already closed
        by the step limit under --on-limit end is not closed twice: the next turn was never asked, it is unreached."""
        t = len(self.turn_recs)
        if t < len(self.item['turns']) and (self.cur['gens'] or (self.gen_budget_hit and t == self.turn_index)):
            e = self.errors[-1] if self.errors else ''
            status = 'gen_budget' if e == self.env.TURN_BUDGET else 'http_400' if e == self.env.HTTP_400 else 'ended'
            self._finish_turn(t, 0.0, status)
        for t in range(len(self.turn_recs), len(self.item['turns'])):
            meta = self.item['turns'][t].get('meta') or {}
            exp = self.item['turns'][t]['expect']
            self.turn_recs.append({'t': t, 'type': meta.get('type'), 'action': meta.get('action'),
                                   'variant': meta.get('variant'), 'decoy': meta.get('decoy'),
                                   'after_events': meta.get('after_events'),
                                   'expect': 'no_call' if exp.get('no_call') else 'call',
                                   'n_expected': 0 if exp.get('no_call') else len(exp['calls']),
                                   'score': 0.0, 'status': 'unreached', 'reached': False, 'n_gens': 0})


def item_key(item, sample):
    return f"{item['id']}#{sample}"


def replay_episode(item, sample, max_tokens, on_limit, mode, lane, count, store):
    """A ProbeEpisode with the session's stored generations observed again in order; (episode, replayed)."""
    office_env, agentic_env = lane
    ep = ProbeEpisode(item, env_shim(office_env, max_tokens), ag_shim(agentic_env, item), mode, on_limit)
    replayed = 0
    for row in store.get(item_key(item, sample), []):
        if ep.done or ep.next_request() is None:
            break
        gen = pc.gen_from_row(row)
        cap = row.get('max_tokens') or max_tokens  # the cap that generation got (clamped near the context)
        ep.observe(gen, pc.gen_record(gen, row.get('latency_s') or 0.0, count, cap))
        replayed += 1
    return ep, replayed


def run_episode(client, item, sample, args, mode, lane, count, gens_sink, store, dp_rank=None):
    ep, replayed = replay_episode(item, sample, args.max_tokens, args.on_limit, mode, lane, count, store)
    key = item_key(item, sample)
    while True:
        max_tokens = ep.next_request()
        if max_tokens is None:
            break
        idx, turn = ep.generations, ep.turn_index
        try:
            gen, lat, used = pc.chat_fit(client, ep.messages, tools=item['tools'], max_tokens=max_tokens, mode=mode,
                                         seed=seed_for(args.seed, NAME, item['id'], sample, idx), dp_rank=dp_rank)
        except ApiError as error:
            if error.retryable:
                raise  # the server never answered: no final row, a resume replays and continues
            ep.http_error(str(error)[:300], overflow=pc.context_refusal(error) is not None)
            break
        gens_sink.write(pc.gen_row(key, idx, turn, gen, lat, used))
        ep.observe(gen, pc.gen_record(gen, lat, count, used))
    ep.finalize()
    return episode_row(ep, item, sample, args.arm, mode, replayed, dp_rank)


def episode_row(ep, item, sample, arm, mode, replayed, dp_rank):
    """The ledger row of a finalized episode."""
    res = ep.result()
    if getattr(ep, 'overflow', False):
        res['failure'] = 'context_overflow'  # final, unlike http_400
    key = item_key(item, sample)
    meta = item.get('meta') or {}
    reached = [r for r in ep.turn_recs if r['reached']]
    calls = [r for r in ep.turn_recs if r['expect'] == 'call']
    return {'key': key, 'item_id': item['id'], 'sample': sample, 'arm': arm, 'mode': mode.label,
            'kind': item.get('kind'), 'domain': item.get('domain'), 'lang': item.get('lang'),
            'split': item.get('split'), 'n_events': meta.get('n_events'), 'n_turns': len(item['turns']),
            'reward': res['reward'], 'reward_reached': (sum(r['score'] for r in reached) / len(reached)) if reached
            else 0.0, 'call_turn_acc': (sum(r['score'] for r in calls) / len(calls)) if calls else None,
            'exact': float(all(r['score'] == 1.0 for r in ep.turn_recs)), 'failure': res['failure'],
            'errors': res['errors'], 'closed': res['closed'], 'limits': dict(ep.limits),
            'unreached': sum(not r['reached'] for r in ep.turn_recs), 'generations': ep.generations,
            'replayed_generations': replayed, 'http_detail': getattr(ep, 'http_detail', None), 'dp_rank': dp_rank,
            'totals': {k: sum((r.get(k) or 0) for r in reached) for k in pc.TOKEN_FIELDS + ('latency_s',)},
            'turns': ep.turn_recs}


# ------------------------------------------------------------------ selection and plan
def load_items(patterns):
    items = []
    for pat in patterns:
        paths = sorted(glob.glob(pat))
        if not paths:
            raise SystemExit(f'--items {pat}: no such file')
        for p in paths:
            items += read_jsonl(p)
    ids = [i['id'] for i in items]
    if len(set(ids)) != len(ids):
        raise SystemExit('duplicate item ids across --items files')
    for it in items:
        assert it.get('turns') and it.get('tools') is not None and it.get('system') is not None, it.get('id')
    return items


def select_items(items, args):
    lengths = {int(x) for x in args.lengths.split(',')} if args.lengths else None
    kinds = set(args.kinds.split(',')) if args.kinds else None
    per = collections.Counter()
    out = []
    for it in sorted(items, key=lambda i: ((i.get('meta') or {}).get('n_events') or 0, i['id'])):
        L = (it.get('meta') or {}).get('n_events')
        if args.split and it.get('split') != args.split:
            continue
        if lengths is not None and L not in lengths:
            continue
        if kinds and it.get('kind') not in kinds:
            continue
        if args.per_length and per[L] >= args.per_length:
            continue
        per[L] += 1
        out.append(it)
    if args.limit:
        out = out[:args.limit]
    return out


def gold_transcript(item):
    """The ideal conversation (expected calls, canned results, a short reply) as wire messages."""
    msgs = [{'role': 'system', 'content': item['system']}]
    for t in item['turns']:
        msgs.append({'role': 'user', 'content': t['user']})
        calls = t['expect'].get('calls') or []
        if calls:
            msgs.append({'role': 'assistant', 'content': '', 'tool_calls': [
                {'id': f'call_{j}', 'type': 'function', 'function': {'name': c['name'], 'arguments': c['arguments']}}
                for j, c in enumerate(calls)]})
            for j, c in enumerate(calls):
                result = t.get('results', {}).get(s2_call_key(c['name'], c['arguments']), {'ok': True})
                msgs.append({'role': 'tool', 'tool_call_id': f'call_{j}',
                             'content': json.dumps(result, ensure_ascii=False)})
        msgs.append({'role': 'assistant', 'content': 'Done.'})
    return msgs


def plan(items, args, mode, tok):
    by_len = collections.defaultdict(list)
    for it in items:
        by_len[(it.get('meta') or {}).get('n_events')].append(it)
    out = {'items': len(items), 'max_tokens': args.max_tokens, 'samples': args.samples, 'by_length': {}}
    longest = 0
    for L, its in sorted(by_len.items(), key=lambda kv: kv[0] or 0):
        turns = [len(i['turns']) for i in its]
        d = {'items': len(its), 'turns_mean': sum(turns) / len(turns), 'turns_max': max(turns),
             'call_turns': sum(1 for i in its for t in i['turns'] if t['expect'].get('calls'))}
        if tok is not None:
            ctx = []
            for it in its:
                text = tok.apply_chat_template(gold_transcript(it)[:-1], tools=it['tools'], tokenize=False,
                                               add_generation_prompt=True, **mode.ctk)
                ctx.append(len(tok(text, add_special_tokens=False).input_ids))
            d['gold_context_max'] = max(ctx)
            longest = max(longest, max(ctx))
        out['by_length'][str(L)] = d
    out['gold_context_max'] = longest if tok is not None else None
    # the model's own replies run longer than the gold ones (and a thinking tool loop keeps its reasoning in the
    # turn): --reply-allowance tokens per turn of the longest session on top. SGLang 0.5.20 refuses (HTTP 400) a
    # request whose prompt plus max_tokens exceeds the context; it does not truncate unless --allow-auto-truncate.
    out['reply_allowance'] = args.reply_allowance * max(len(i['turns']) for i in items)
    out['context_needed'] = (longest + args.max_tokens + out['reply_allowance']) if tok is not None else None
    out['turns_total'] = sum(len(i['turns']) for i in items) * args.samples
    return out


# ------------------------------------------------------------------ run
def add_args(ap):
    pc.add_common_args(ap)
    g = ap.add_argument_group('on-call selection')
    g.add_argument('--items', nargs='+', required=True, help='agentic_env-format JSONL files (globs allowed)')
    g.add_argument('--split', default='eval', help='keep items of this split ("" keeps all)')
    g.add_argument('--lengths', default='', help='comma list of meta.n_events to keep (default all)')
    g.add_argument('--per-length', type=int, default=0, help='first N items per length, by id (default all)')
    g.add_argument('--kinds', default='')
    g.add_argument('--samples', type=int, default=1, help='episodes per item')
    g.add_argument('--on-limit', choices=LIMIT_POLICIES, default='continue')
    return ap


def run(args, client=None, count=None):
    started = time.time()
    mode = pc.mode_from_args(args)
    items = select_items(load_items(args.items), args)
    if not items:
        raise SystemExit('no items selected')
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = None
    if count is None:
        count, tok = pc.load_counter(args.tokenizer)
    if args.plan:
        tok = tok or pc.sd_common.load_tokenizer(args.tokenizer)  # the plan renders through the chat template
        p = plan(items, args, mode, tok)
        (out / 'plan.json').write_text(json.dumps(p, indent=1) + '\n')
        print(json.dumps(p, indent=1))
        return {'plan': p}
    lane = load_lane()
    client = client or pc.make_client(args)
    if client.model == 'auto':
        client.model = client.discover_model()
    selection = {'items': [i['id'] for i in items], 'samples': args.samples}
    settings = {'task': NAME, 'arm': args.arm, 'model': client.model, 'thinking': mode.thinking,
                'greedy': mode.greedy, 'mode': mode.label, 'sampling': mode.sampling,
                'chat_template_kwargs': mode.ctk, 'attach_reasoning': mode.attach_reasoning,
                'history_reasoning': mode.history, 'max_tokens': args.max_tokens, 'limit_policy': args.on_limit,
                'seed': args.seed, 'selection': selection, 'items_files': args.items,
                'max_steps_per_turn': lane[1].MAX_STEPS_PER_TURN, 'tokenizer': args.tokenizer,
                'lane': pc.sd_common.lane_hashes(), 'argv': sys.argv}
    pc.guard_settings(out, NAME, settings, args.allow_settings_change)
    ledger, gens_path = out / 'ledger.jsonl', out / 'gens.jsonl'
    done = {r['key'] for r in read_jsonl(ledger) if r.get('failure') != 'http_400'}
    todo = [(it, s) for it in items for s in range(args.samples) if item_key(it, s) not in done]
    todo.sort(key=lambda t: -len(t[0]['turns']))  # longest sessions first: they are the critical path
    ranks = pc.assign_dp_ranks([item_key(it, s) for it in sorted(items, key=lambda i: (-len(i['turns']), i['id']))
                                for s in range(args.samples)], args.dp_ranks)
    store = pc.load_gen_store(gens_path)
    resumable = sum(1 for it, s in todo if store.get(item_key(it, s)))
    log(f'oncall {mode.label} [{args.arm}]: {len(items)} items x {args.samples}, {len(done)} done, {len(todo)} to go '
        f'({resumable} resume mid-session)')
    gate, stats = FailureGate(args.abort_after), collections.Counter()
    summary = {'task': NAME, 'arm': args.arm, 'mode': mode.label, 'model': client.model, 'items': len(items),
               'planned': len(items) * args.samples, 'resumed_done': len(done)}
    with JsonlSink(ledger) as sink, JsonlSink(gens_path) as gsink:
        def on_result(task, row):
            sink.write(row)
            stats['written'] += 1
            if row['failure'] == 'http_400':
                stats['refused'] += 1
                gate.fail(ApiError(400, f"{row['key']} refused: {row.get('http_detail')}"))
            else:
                if row['failure'] == 'context_overflow':
                    stats['context_overflow'] += 1
                    log(f"{row['key']}: context overflow at turn {len(row['turns']) - row['unreached'] - 1}, "
                        f"{row['unreached']} turns unreached (final, scored 0)")
                gate.ok()

        def on_error(task, error):
            if isinstance(error, ApiError) and not error.retryable:
                raise error
            stats['failed'] += 1
            gate.fail(error)

        try:
            pc.run_pool(todo, lambda t: run_episode(client, t[0], t[1], args, mode, lane, count, gsink, store,
                                                    ranks.get(item_key(t[0], t[1]))),
                        concurrency=args.concurrency, on_result=on_result, on_error=on_error,
                        label=f'oncall {mode.label}', total=len(todo), progress_every=args.progress_every)
        except pc.Aborted as error:
            log(f'ABORTED: {error}. Finished rows are in the ledger; rerun the same command to resume.')
            summary['aborted'] = str(error)
    rows = {r['key']: r for r in read_jsonl(ledger)}
    final = [r for r in rows.values() if r.get('failure') != 'http_400']
    summary.update(written=stats['written'], refused=stats['refused'], failed_unwritten=stats['failed'],
                   context_overflow_this_run=stats['context_overflow'],
                   context_overflow_rows=sum(r.get('failure') == 'context_overflow' for r in final),
                   complete=len(final) == len(items) * args.samples and not summary.get('aborted'),
                   rows_final=len(final), client=dict(client.stats), seconds=round(time.time() - started, 1))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    args = ap.parse_args(argv)
    summary = run(args)
    if 'plan' in summary:
        return 0  # run() printed the plan
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0 if summary.get('complete') else (2 if summary.get('aborted') or summary.get('failed_unwritten') else 3)


if __name__ == '__main__':
    sys.exit(main())
