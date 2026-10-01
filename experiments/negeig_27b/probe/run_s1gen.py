#!/usr/bin/env python3
"""Generative S1 probe: answer each S1 eval session turn by turn through an OpenAI-compatible SGLang server, with the
model's OWN previous answers in history, and score every answer by exact match against the session's gold answer.

eval27.py scores S1 teacher-forced (gold history, argmax per answer token, think-off only). This runner asks what a
served model does: it generates every answer, sampled with the vendor arm of the mode (or greedy as a measurement),
keeps its own answers in the conversation so an early mistake stays in front of it, and runs both thinking modes
with the same token accounting as run_oncall.py (probe_common.gen_record).

Scoring per answer (the gold format is fixed by each domain's system prompt, so only presentation is normalized):
  ok_strict  the content, stripped of surrounding whitespace, equals the gold answer
  ok_norm    (the score) after NFKC, whitespace runs collapsed to one space, a wrapping pair of ** or backticks removed,
             a wrapping pair of quotes removed (not in codetrace, where 'pine' is the repr), one trailing period
             removed, and case folded for English custody, toggles, ops and orders (names, places, statuses, flags
             and cities carry no case information there; codetrace (Python repr: True, 'Pine') and fsys (paths) keep
             case). Order is never normalized: the formats say sorted, so an unsorted list is wrong.
  A truncated answer (finish "length") scores 0 whatever it holds and is counted as truncated; its content stays in
  history (what the client would have kept) and the session continues.

Selection: the S1 eval files (<s1>/<domain>.eval.jsonl), per domain and length the first --per-cell sessions by seed
(the build interleaves Hebrew at every fourth seed, so 16 keeps four Hebrew sessions in the four domains that have
Hebrew). --domains ops gives the ops-only run.

Resumable as run_oncall.py: ledger.jsonl final rows per session, gens.jsonl every generation, a resume replays a
session's generations and continues. The context limit is handled as there: cap clamped once when only the cap does
not fit, a final context_overflow row when the prompt itself does not, any other HTTP 400 rerun on resume.

  run_s1gen.py --s1 DATA/s1 --thinking off --arm untouched --out-dir OUT [--lengths 256,1024 --per-cell 16]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import probe_common as pc
from probe_common import ApiError, FailureGate, JsonlSink, log, read_jsonl, seed_for

from common import Turn, to_messages, user_text  # data/  # noqa: E402

NAME = 's1gen'
DOMAINS = ['custody', 'toggles', 'fsys', 'codetrace', 'ops', 'orders']
CASE_FREE = {'custody', 'toggles', 'ops', 'orders'}
_WS = re.compile(r'\s+')


def normalize(text, domain, lang):
    s = _WS.sub(' ', unicodedata.normalize('NFKC', text or '')).strip()
    for _ in range(3):
        before = s
        if len(s) >= 4 and s.startswith('**') and s.endswith('**'):
            s = s[2:-2].strip()
        if len(s) >= 2 and s[0] == '`' and s[-1] == '`':
            s = s.strip('`').strip()
        if domain != 'codetrace' and len(s) >= 2 and s[0] == s[-1] and s[0] in '"\'':
            s = s[1:-1].strip()
        if s.endswith('.'):
            s = s[:-1].rstrip()
        if s == before:
            break
    if lang == 'en' and domain in CASE_FREE:
        s = s.casefold()
    return s


def score_answer(content, gold, domain, lang, truncated):
    if truncated:
        return False, False
    return (content or '').strip() == gold, normalize(content, domain, lang) == normalize(gold, domain, lang)


def session_key(s):
    return f"{s['domain']}:{s['seed']}"


class S1Episode:
    """One S1 session answered by the model, one generation per question."""

    def __init__(self, sess, mode):
        self.s, self.mode = sess, mode
        self.turns = [Turn(**t) for t in sess['turns']]
        self.messages = [{'role': 'system', 'content': sess['system']}]
        self.t, self.recs, self.done, self.generations = 0, [], False, 0
        self.http_detail = None
        self._push_user()

    def _push_user(self):
        turn = self.turns[self.t]
        self.messages.append({'role': 'user', 'content': user_text(self.s['lang'], turn,
                                                                   self.s['initial'] if self.t == 0 else None)})

    def next_request(self):
        return not self.done

    def _rec(self, t, status, score=0.0, extra=None, gens=()):
        turn = self.turns[t]
        return {'t': t, 'after_events': turn.meta.get('after_events'), 'kind': turn.meta.get('kind'),
                'gold': turn.answer, 'score': float(score), 'status': status, 'reached': status != 'unreached',
                **(extra or {}), **pc.sum_gens(list(gens))}

    def observe(self, gen, rec):
        self.generations += 1
        turn = self.turns[self.t]
        content = gen.content or ''
        truncated = gen.finish_reason not in pc.STOP_FINISH
        ok_s, ok_n = score_answer(content, turn.answer, self.s['domain'], self.s['lang'], truncated)
        self.recs.append(self._rec(self.t, 'truncated' if truncated else 'scored', float(ok_n),
                                   {'answer': content[:300], 'ok_strict': ok_s, 'ok_norm': ok_n}, [rec]))
        self.messages.append(pc.assistant_message(gen, self.mode, tool_calls=False))
        self.t += 1
        if self.t == len(self.turns):
            self.done = True
        else:
            self._push_user()

    def http_error(self, detail, overflow=False):
        """overflow: SGLang's context-length refusal, a final outcome (probe_common); else http_400, rerun on resume."""
        self.http_detail = detail
        self.overflow = bool(overflow)
        self.recs.append(self._rec(self.t, 'context_overflow' if overflow else 'http_400'))
        self.done = True

    def finalize(self):
        for t in range(len(self.recs), len(self.turns)):
            self.recs.append(self._rec(t, 'unreached'))


def run_session(client, sess, args, mode, count, gens_sink, store, dp_rank=None):
    ep = S1Episode(sess, mode)
    key = session_key(sess)
    replayed = 0
    for row in store.get(key, []):
        if ep.done:
            break
        gen = pc.gen_from_row(row)
        cap = row.get('max_tokens') or args.max_tokens  # the cap that generation got (clamped near the context)
        ep.observe(gen, pc.gen_record(gen, row.get('latency_s') or 0.0, count, cap))
        replayed += 1
    while ep.next_request():
        idx = ep.generations
        try:
            gen, lat, used = pc.chat_fit(client, ep.messages, tools=None, max_tokens=args.max_tokens, mode=mode,
                                         seed=seed_for(args.seed, NAME, key, idx), dp_rank=dp_rank)
        except ApiError as error:
            if error.retryable:
                raise
            ep.http_error(str(error)[:300], overflow=pc.context_refusal(error) is not None)
            break
        gens_sink.write(pc.gen_row(key, idx, ep.t, gen, lat, used))
        ep.observe(gen, pc.gen_record(gen, lat, count, used))
    ep.finalize()
    reached = [r for r in ep.recs if r['reached']]
    n = len(ep.recs)
    failure = None
    if ep.http_detail:
        failure = 'context_overflow' if getattr(ep, 'overflow', False) else 'http_400'
    return {'key': key, 'arm': args.arm, 'mode': mode.label, 'domain': sess['domain'], 'lang': sess['lang'],
            'seed': sess['seed'], 'n_events': sess['n_events'], 'n_turns': n,
            'acc': sum(r['score'] for r in ep.recs) / n,
            'acc_strict': sum(bool(r.get('ok_strict')) for r in ep.recs) / n,
            'failure': failure, 'http_detail': ep.http_detail,
            'limits': {'truncated': sum(r['status'] == 'truncated' for r in ep.recs),
                       'context_overflow': sum(r['status'] == 'context_overflow' for r in ep.recs)},
            'unreached': sum(not r['reached'] for r in ep.recs), 'generations': ep.generations,
            'replayed_generations': replayed, 'dp_rank': dp_rank,
            'totals': {k: sum((r.get(k) or 0) for r in reached) for k in pc.TOKEN_FIELDS + ('latency_s',)},
            'turns': ep.recs}


# ------------------------------------------------------------------ selection and plan
def select_sessions(args):
    domains = args.domains.split(',') if args.domains else DOMAINS
    lengths = [int(x) for x in args.lengths.split(',')]
    out = []
    for d in domains:
        paths = sorted(glob.glob(os.path.join(args.s1, f'{d}.eval.jsonl')))
        if not paths:
            raise SystemExit(f'no {d}.eval.jsonl under {args.s1}')
        rows = read_jsonl(paths[0])
        for L in lengths:
            cell = sorted((r for r in rows if r['n_events'] == L), key=lambda r: r['seed'])
            if len(cell) < args.per_cell:
                raise SystemExit(f'{d} at {L} events has {len(cell)} sessions, fewer than --per-cell {args.per_cell}')
            out += cell[:args.per_cell]
    if args.limit:
        out = out[:args.limit]
    return out


def plan(sessions, args, mode, tok):
    cells = collections.defaultdict(list)
    for s in sessions:
        cells[(s['domain'], s['n_events'])].append(s)
    out = {'sessions': len(sessions), 'max_tokens': args.max_tokens, 'turns_total': sum(len(s['turns'])
                                                                                       for s in sessions),
           'cells': {}}
    longest = 0
    for (d, L), ss in sorted(cells.items()):
        c = {'sessions': len(ss), 'turns_mean': sum(len(s['turns']) for s in ss) / len(ss),
             'turns_max': max(len(s['turns']) for s in ss), 'he': sum(s['lang'] == 'he' for s in ss)}
        if tok is not None:
            ctx = []
            for s in ss:
                from common import Session
                sess = Session(**{**{k: v for k, v in s.items() if k != 'messages'},
                                  'turns': [Turn(**t) for t in s['turns']]})
                text = tok.apply_chat_template(to_messages(sess)[:-1], tokenize=False, add_generation_prompt=True,
                                               **mode.ctk)
                ctx.append(len(tok(text, add_special_tokens=False).input_ids))
            c['gold_context_max'] = max(ctx)
            longest = max(longest, max(ctx))
        out['cells'][f'{d}:{L}'] = c
    out['gold_context_max'] = longest if tok is not None else None
    # the model's own replies run longer than the gold ones (and a thinking tool loop keeps its reasoning in the
    # turn): --reply-allowance tokens per turn of the longest session on top. SGLang 0.5.20 refuses (HTTP 400) a
    # request whose prompt plus max_tokens exceeds the context; it does not truncate unless --allow-auto-truncate.
    out['reply_allowance'] = args.reply_allowance * max(len(s['turns']) for s in sessions)
    out['context_needed'] = (longest + args.max_tokens + out['reply_allowance']) if tok is not None else None
    return out


def add_args(ap):
    pc.add_common_args(ap)
    g = ap.add_argument_group('S1 selection')
    g.add_argument('--s1', required=True, help='directory with <domain>.eval.jsonl')
    g.add_argument('--domains', default='', help='comma list (default all six); "ops" is the ops-only run')
    g.add_argument('--lengths', default='256,1024')
    g.add_argument('--per-cell', type=int, default=16, help='sessions per domain and length, first by seed')
    return ap


def run(args, client=None, count=None):
    started = time.time()
    mode = pc.mode_from_args(args)
    sessions = select_sessions(args)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = None
    if count is None:
        count, tok = pc.load_counter(args.tokenizer)
    if args.plan:
        tok = tok or pc.sd_common.load_tokenizer(args.tokenizer)  # the plan renders through the chat template
        p = plan(sessions, args, mode, tok)
        (out / 'plan.json').write_text(json.dumps(p, indent=1) + '\n')
        print(json.dumps(p, indent=1))
        return {'plan': p}
    client = client or pc.make_client(args)
    if client.model == 'auto':
        client.model = client.discover_model()
    selection = {'sessions': [session_key(s) for s in sessions]}
    settings = {'task': NAME, 'arm': args.arm, 'model': client.model, 'thinking': mode.thinking,
                'greedy': mode.greedy, 'mode': mode.label, 'sampling': mode.sampling,
                'chat_template_kwargs': mode.ctk, 'attach_reasoning': mode.attach_reasoning,
                'history_reasoning': mode.history, 'max_tokens': args.max_tokens, 'limit_policy': 'continue',
                'seed': args.seed, 'selection': selection, 's1': args.s1, 'tokenizer': args.tokenizer,
                'argv': sys.argv}
    pc.guard_settings(out, NAME, settings, args.allow_settings_change)
    ledger, gens_path = out / 'ledger.jsonl', out / 'gens.jsonl'
    done = {r['key'] for r in read_jsonl(ledger) if r.get('failure') != 'http_400'}
    todo = [s for s in sessions if session_key(s) not in done]
    store = pc.load_gen_store(gens_path)
    log(f's1gen {mode.label} [{args.arm}]: {len(sessions)} sessions, {len(done)} done, {len(todo)} to go '
        f'({sum(1 for s in todo if store.get(session_key(s)))} resume mid-session)')
    gate, stats = FailureGate(args.abort_after), collections.Counter()
    summary = {'task': NAME, 'arm': args.arm, 'mode': mode.label, 'model': client.model, 'sessions': len(sessions),
               'resumed_done': len(done)}
    # longest sessions first: they are the critical path
    todo.sort(key=lambda s: -len(s['turns']))
    ranks = pc.assign_dp_ranks([session_key(s) for s in sorted(sessions, key=lambda s: (-len(s['turns']),
                                                                                       session_key(s)))],
                               args.dp_ranks)
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
                    log(f"{row['key']}: context overflow, {row['unreached']} answers unreached (final, scored 0)")
                gate.ok()

        def on_error(task, error):
            if isinstance(error, ApiError) and not error.retryable:
                raise error
            stats['failed'] += 1
            gate.fail(error)

        try:
            pc.run_pool(todo, lambda s: run_session(client, s, args, mode, count, gsink, store,
                                                    ranks.get(session_key(s))),
                        concurrency=args.concurrency, on_result=on_result, on_error=on_error,
                        label=f's1gen {mode.label}', total=len(todo), progress_every=args.progress_every)
        except pc.Aborted as error:
            log(f'ABORTED: {error}. Finished rows are in the ledger; rerun the same command to resume.')
            summary['aborted'] = str(error)
    rows = {r['key']: r for r in read_jsonl(ledger)}
    final = [r for r in rows.values() if r.get('failure') != 'http_400']
    summary.update(written=stats['written'], refused=stats['refused'], failed_unwritten=stats['failed'],
                   context_overflow_this_run=stats['context_overflow'],
                   context_overflow_rows=sum(r.get('failure') == 'context_overflow' for r in final),
                   complete=len(final) == len(sessions) and not summary.get('aborted'), rows_final=len(final),
                   client=dict(client.stats), seconds=round(time.time() - started, 1))
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
