"""Tool-use replay from the untouched Qwen3.8-27B.

Runs the Hebrew lane's agentic pool (agentic_pool.build() + build_hard(), imported and never modified) through the
served model with the items' own tools, N samples per item. Canned results are fed exactly as agentic_env feeds
them, and the score is agentic_env's own call_key and turn_score. A transcript is kept only when every turn scores
1.0 and all turns close, so the replay shows the model doing the right thing on its own terms.

ToolEpisode below is agentic_env.AgenticEpisode.observe at the message level: the lane's class is driven by token
ids, this job talks to a server that returns parsed messages. test_selfdistill.py drives both in lockstep over all
2,537 items and requires identical scores, errors, calls and messages.

Reads   the pool (train split by default; the monitor split is held out for evaluation)
Writes  <out>/tool_raw.jsonl   one row per (item, sample): wire-form messages, score, errors, failure class
        <out>/tool_sft.jsonl   kept transcripts in chat-message form with OpenAI tool_calls (dict arguments) and
                               tool messages, plus `tools` (env.template_tools, the form the model was served)
        <out>/filter_report.json  per-kind solved and kept rates (see sd_filters.py)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import sd_filters
from sd_common import (Aborted, ApiError, FailureGate, JsonlSink, add_client_args, add_run_args, lane_hashes,
                       load_pool, log, make_client, read_jsonl, run_pool, sampling_profile, seed_for,
                       write_manifest)


class ToolEpisode:
    """One item driven by parsed server messages. Mirrors agentic_env.AgenticEpisode.observe line for line."""

    def __init__(self, item, env, ag):
        self.item, self.env, self.ag = item, env, ag
        self.tool_names = {t['function']['name'] for t in item['tools']}
        self.messages = [{'role': 'system', 'content': item['system']},
                         {'role': 'user', 'content': item['turns'][0]['user']}]
        self.turn_index, self.steps, self.turn_scores = 0, [], []
        self.calls, self.errors = [], []
        self.generations, self.done = 0, False
        self.reasoning_chars = 0
        self.gens = []

    def _end(self, error=None):
        if error:
            self.errors.append(error)
        self.done = True

    def next_request(self):
        """max_tokens for the next generation, or None when the episode is over."""
        if self.done:
            return None
        if self.generations == self.ag.MAX_GENERATIONS:
            self._end(self.env.TURN_BUDGET)
            return None
        return self.env.MAX_TOKENS

    def http_error(self):
        """The server refused the request (context too long is a 400 on the eval server too)."""
        self._end(self.env.HTTP_400)

    def _close_turn(self):
        self.turn_scores.append(self.ag.turn_score(self.item['turns'][self.turn_index]['expect'], self.steps))
        self.turn_index += 1
        self.steps = []

    def observe(self, gen):
        """gen: sd_common.Generation for one response."""
        assert not self.done
        env, ag = self.env, self.ag
        self.generations += 1
        self.gens.append({'finish': gen.finish_reason,
                          'completion_tokens': int(gen.usage.get('completion_tokens') or 0),
                          'prompt_tokens': int(gen.usage.get('prompt_tokens') or 0)})
        self.reasoning_chars += len(gen.reasoning or '')
        if gen.finish_reason not in ('stop', 'tool_calls'):
            self._end(env.TRUNCATED)
            return
        message = {'role': 'assistant', 'content': gen.content or ''}
        if gen.tool_calls:
            message['tool_calls'] = gen.tool_calls
        self.messages.append(message)
        turn = self.item['turns'][self.turn_index]
        if not message.get('tool_calls'):
            self._close_turn()
            if self.turn_index == len(self.item['turns']):
                self._end()
                return
            self.messages.append({'role': 'user', 'content': self.item['turns'][self.turn_index]['user']})
            return
        keys = []
        for c in message['tool_calls']:
            try:
                arguments = json.loads(c['function']['arguments'])
                json.dumps(arguments, ensure_ascii=False).encode()
            except (json.JSONDecodeError, UnicodeEncodeError) as error:
                self._end(type(error).__name__ + ':' + str(error)[:200])
                return
            name = c['function']['name']
            key = ag.call_key(name, arguments) if isinstance(arguments, dict) else name + ' <non-object>'
            keys.append(key)
            self.calls.append({'name': name, 'arguments': arguments})
            if name not in self.tool_names:
                result = ag.ERROR_UNKNOWN
            else:
                result = turn.get('results', {}).get(key, ag.ERROR_NO_MATCH)
            self.messages.append({'role': 'tool', 'tool_call_id': c['id'],
                                  'content': json.dumps(result, ensure_ascii=False)})
        self.steps.append(keys)
        if len(self.steps) >= ag.MAX_STEPS_PER_TURN:
            self._close_turn()
            self._end(env.TURN_BUDGET)
            return

    def result(self):
        assert self.done
        closed = len(self.turn_scores) == len(self.item['turns'])
        scores = self.turn_scores + [0.0] * (len(self.item['turns']) - len(self.turn_scores))
        reward = sum(scores) / len(scores)
        return {'reward': reward, 'turn_scores': scores, 'errors': list(self.errors), 'closed': closed,
                'failure': self._failure(reward, closed)}

    def _failure(self, reward, closed):
        env = self.env
        if self.errors:
            e = self.errors[0]
            return ('truncated' if e == env.TRUNCATED else 'turn_budget' if e == env.TURN_BUDGET
                    else 'http_400' if e == env.HTTP_400 else 'bad_arguments')
        if not closed:
            return 'unclosed'
        return 'wrong_calls' if reward < 1.0 else None


def run_episode(client, item, sample, args, sampling, env, ag):
    ep = ToolEpisode(item, env, ag)
    while True:
        max_tokens = ep.next_request()
        if max_tokens is None:
            break
        try:
            gen = client.chat(ep.messages, tools=item['tools'], max_tokens=max_tokens, sampling=sampling,
                              seed=seed_for(args.seed, 'tool', item['id'], sample, ep.generations))
        except ApiError as error:
            if error.retryable:
                raise  # the server never answered: no ledger row, a resume tries the episode again
            ep.http_error()
            break
        ep.observe(gen)
    res = ep.result()
    return {'item_id': item['id'], 'sample': sample, 'kind': item['kind'], 'lang': item['lang'],
            'split': item['split'], 'seed': seed_for(args.seed, 'tool', item['id'], sample), **res,
            'generations': ep.generations, 'reasoning_chars': ep.reasoning_chars, 'gens': ep.gens,
            'usage': {'completion_tokens': sum(g['completion_tokens'] for g in ep.gens),
                      'prompt_tokens': sum(g['prompt_tokens'] for g in ep.gens)},
            'messages': ep.messages}


def select_items(items, args):
    kinds = set(args.kinds.split(',')) if args.kinds else None
    chosen, per_kind = [], Counter()
    for it in sorted(items, key=lambda i: i['id']):
        if not args.include_monitor and it['split'] != 'train':
            continue
        if kinds and it['kind'] not in kinds:
            continue
        if args.item_limit_per_kind and per_kind[it['kind']] >= args.item_limit_per_kind:
            continue
        per_kind[it['kind']] += 1
        chosen.append(it)
    return chosen, dict(per_kind)


def add_args(ap, client_args=True, run_args=True):
    if client_args:
        add_client_args(ap)
    if run_args:
        add_run_args(ap)
    ap.add_argument('--n-samples', type=int, default=4, help='episodes per item')
    ap.add_argument('--keep-per-item', type=int, default=2,
                    help='most solved, distinct transcripts kept per item; 0 keeps every one')
    ap.add_argument('--kinds', default='', help='comma list of kinds (default: all nine)')
    ap.add_argument('--item-limit-per-kind', type=int, default=0, help='first N items per kind (smoke runs)')
    ap.add_argument('--include-monitor', action='store_true',
                    help='also run the monitor split; it is held out for evaluation, so off by default')
    return ap


def run(args, client=None):
    started = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    env, ag, items = load_pool()
    chosen, per_kind = select_items(items, args)
    assert chosen, 'no items selected'
    client = client or make_client(args)
    if client.model == 'auto':
        client.model = client.discover_model()
    sampling = sampling_profile(args.sampling, args.generation_config)
    write_manifest(out, 'tools', {'model': client.model, 'sampling': sampling, 'think': 'off',
                                  'n_samples': args.n_samples, 'keep_per_item': args.keep_per_item,
                                  'items': len(chosen), 'items_per_kind': per_kind, 'seed': args.seed,
                                  'include_monitor': args.include_monitor, 'max_generations': ag.MAX_GENERATIONS,
                                  'max_steps_per_turn': ag.MAX_STEPS_PER_TURN, 'max_tokens': env.MAX_TOKENS,
                                  'argv': sys.argv, 'lane': lane_hashes()})
    path = out / 'tool_raw.jsonl'
    # An episode the server refused (HTTP 400) is recorded but not final: a resume runs it again, so a server that
    # was misconfigured for one run cannot leave a ledger of zero-score rows that looks finished.
    done = {(r['item_id'], int(r['sample'])) for r in read_jsonl(path) if r.get('failure') != 'http_400'}
    todo = [(it, s) for it in chosen for s in range(args.n_samples) if (it['id'], s) not in done]
    log(f'tools: {len(chosen)} items x {args.n_samples} samples, {len(done)} already in the ledger, {len(todo)} to go')
    gate, stats = FailureGate(args.abort_after), Counter()
    summary = {'model': client.model, 'items': len(chosen), 'episodes_planned': len(chosen) * args.n_samples,
               'resumed': len(done)}
    with JsonlSink(path) as sink:
        if sink.repaired:
            log(f'tool_raw.jsonl: cut a torn {sink.repaired}-byte last line')

        def on_result(task, row):
            sink.write(row)
            stats['done'] += 1
            if row['failure'] == 'http_400':
                stats['refused'] += 1
                gate.fail(ApiError(400, f"episode {row['item_id']}#{row['sample']} refused after "
                                        f"{row['generations']} generations"))  # a streak means the server is wrong
            else:
                gate.ok()

        def on_error(task, error):
            if isinstance(error, ApiError) and not error.retryable:
                raise error  # run_episode turns non-retryable 400s into an episode outcome, so this is a bug
            stats['failed'] += 1
            gate.fail(error)

        try:
            run_pool(todo, lambda t: run_episode(client, t[0], t[1], args, sampling, env, ag),
                     concurrency=args.concurrency, on_result=on_result, on_error=on_error, label='tools',
                     total=len(todo), progress_every=args.progress_every)
        except Aborted as error:
            log(f'ABORTED: {error}. Finished rows are in the ledger; rerun the same command to resume.')
            summary['aborted'] = str(error)
    summary.update(written=stats['done'], refused=stats['refused'], failed_unwritten=stats['failed'],
                   client=dict(client.stats), seconds=round(time.time() - started, 1))
    if not args.no_filter:
        sd_filters.finalize(out, items_by_id={i['id']: i for i in items}, template_tools=env.template_tools,
                            keep_per_item=args.keep_per_item,
                            meta={'model': client.model, 'sampling': args.sampling}, seed=args.seed)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    args = ap.parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 2 if summary.get('aborted') or summary.get('failed_unwritten') else 0


if __name__ == '__main__':
    sys.exit(main())
