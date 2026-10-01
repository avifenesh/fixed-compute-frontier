#!/usr/bin/env python3
"""Self-distilled replay for the Qwen3.8-27B retrofit: driver.

Every label this job writes is the untouched Qwen3.8-27B's own output, served think-off. No other model's text is
ever a label. The job talks to an SGLang 0.5.20 OpenAI-compatible server started with

    --tool-call-parser qwen3_coder --reasoning-parser qwen3

Order on the box:
    python -m unittest test_selfdistill   # CPU, 44 tests: parity with the lane's episode on all 2,537 items, filters, client
    python run.py dry-run      # CPU, no server needed: the whole pipeline against a fake server. Run it first.
    python run.py all          # translate 1,500 prompts, answer 7.5k prompts, then the tool pool, then filter
    python run.py filter       # refilter the ledgers (no server), e.g. after reading the spot sample
Then read 50 rows per source in <out>/spot_read.jsonl (chat_en, chat_he, translation, tool) before the rows go
into the Stage A mix.

    chat     chat_gen.py    translations + answers        -> chat_sft.jsonl
    tools    tool_gen.py    agentic pool, N samples/item  -> tool_sft.jsonl
    filter   sd_filters.py  ledgers -> training files, spot_read.jsonl, filter_report.json
    all      chat, then tools, then one filter
    dry-run  run.py + sd_fake_server.py, see dry_run() below

Sampling. The default is the model card's non-thinking arm (temperature 0.7, top_p 0.8, top_k 20, presence_penalty
1.5), because every request is think-off. generation_config.json is the THINKING arm (1.0 / 0.95 / 20); use
`--sampling generation_config` to send it instead. Greedy is never used: it loops, and a looped answer is not a
replay target.

Resume: every command is rerunnable. Finished rows are skipped, a torn last line is cut, and a request that failed
every retry is left out of its ledger so the rerun tries it again.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import chat_gen
import sd_filters
import tool_gen
from sd_common import (DEFAULT_OUT, Client, add_client_args, add_run_args, hebrew_share, lane_hashes,
                       load_encode_chat, load_pool, load_tokenizer, log, read_jsonl, write_jsonl_atomic)

# ------------------------------------------------------------------ real runs


def cmd_filter(argv):
    ap = argparse.ArgumentParser(prog='run.py filter', description='refilter the ledgers, no server needed')
    ap.add_argument('--out-dir', default=str(DEFAULT_OUT))
    ap.add_argument('--keep-per-item', type=int, default=None, help='default: the value the tool run recorded')
    ap.add_argument('--spot-n', type=int, default=50)
    ap.add_argument('--seed', type=int, default=20261001)
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    items_by_id = template_tools = None
    if (out / 'tool_raw.jsonl').exists():
        office_env, _, items = load_pool()
        items_by_id, template_tools = {i['id']: i for i in items}, office_env.template_tools
    report = sd_filters.finalize(out, items_by_id=items_by_id, template_tools=template_tools,
                                 keep_per_item=args.keep_per_item, spot_n=args.spot_n, seed=args.seed)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_all(argv):
    ap = argparse.ArgumentParser(prog='run.py all', description='chat replay, then tool replay, then one filter')
    add_client_args(ap)
    add_run_args(ap)
    chat_gen.add_args(ap, client_args=False, run_args=False)
    tool_gen.add_args(ap, client_args=False, run_args=False)
    args = ap.parse_args(argv)
    chat_args = copy.copy(args)
    chat_args.no_filter = True  # one filter at the end, over every ledger
    chat_summary = chat_gen.run(chat_args)
    print(json.dumps({'chat': chat_summary}, ensure_ascii=False, indent=2))
    if chat_summary.get('aborted'):
        log('chat aborted, the tool phase is not started. Fix the server and rerun the same command.')
        if not args.no_filter:
            chat_gen.finalize_out(Path(args.out_dir), chat_summary['model'], args)
        return 2
    tool_summary = tool_gen.run(args)
    print(json.dumps({'tools': tool_summary}, ensure_ascii=False, indent=2))
    unfinished = sum(v.get('failed_unwritten', 0) for v in chat_summary.values() if isinstance(v, dict))
    return 2 if tool_summary.get('aborted') or tool_summary.get('failed_unwritten') or unfinished else 0


# ------------------------------------------------------------------ dry run
# A compact prompt file with one prompt per defect the filters must catch. The marker words are read by the fake
# server (sd_fake_server.chat_answer and fake_translate); the real model ignores them as noise.
_TOPICS = ['river deltas', 'sourdough starters', 'binary search', 'tax forms', 'sleep cycles', 'Roman roads',
           'password managers', 'solar panels', 'chess openings', 'tide tables', 'index funds', 'bee colonies']
_TEMPLATES = ['Explain how {t} work in simple terms, in about three sentences.',
              'Give me two practical tips about {t} and one common mistake to avoid.',
              'What separates {t} from {u}? Keep the answer short.',
              'Write a short note for a beginner who wants to learn about {t}.']
_CATEGORIES = ['open_qa', 'closed_qa', 'brainstorming', 'general_qa', 'classification', 'summarization']

CHAT_MARKERS = {  # marker -> the chat filter reason it must produce
    'MODE_EMPTY': 'empty', 'MODE_LENGTH': 'truncated', 'MODE_THINK': 'reasoning_leak',
    'MODE_MARKUP': 'markup_leak', 'MODE_TOOLCALL': 'markup_leak', 'MODE_REFUSE': 'refusal',
    'MODE_LOOP': 'degenerate_repeat', 'MODE_400': 'api_error',
}
TRANSLATION_MARKERS = {
    'MODE_TR_EMPTY': 'empty', 'MODE_TR_LENGTH': 'truncated', 'MODE_TR_ENGLISH': 'not_hebrew',
    'MODE_TR_SHORT': 'length', 'MODE_TR_DROPNUM': 'numbers_changed', 'MODE_TR_REFUSE': 'refusal',
    'MODE_TR_PREAMBLE': 'preamble', 'MODE_TR_MARKUP': 'markup', 'MODE_TR_CJK': 'cjk',
    'MODE_TR_CODE': 'code_changed', 'MODE_TR_400': 'api_error',
}
_TR_BODY = {
    'MODE_TR_DROPNUM': 'MODE_TR_DROPNUM Convert the 12 crates, 48 boxes and 7 pallets from the list into a single '
                       'total for 2025.',
    'MODE_TR_CODE': 'MODE_TR_CODE Why does `sorted(xs, key=len)` keep the order of equal items when the list has '
                    '9 words?',
}


def dry_run_prompts():
    rows = []

    def add(prompt, lang='en', source='dolly-15k'):
        i = len(rows)
        rows.append({'id': f'chat-d{i:03d}', 'prompt': prompt, 'lang': lang, 'source': source,
                     'license': 'dry-run', 'category': _CATEGORIES[i % len(_CATEGORIES)]})

    for k in range(30):
        t, u = _TOPICS[k % len(_TOPICS)], _TOPICS[(k * 5 + 3) % len(_TOPICS)]
        add(_TEMPLATES[k % len(_TEMPLATES)].format(t=t, u=u) + (f' Use at most {k + 3} lines.' if k % 3 == 0 else ''),
            source='oasst2' if k % 4 == 0 else 'dolly-15k')
    add('What does `len(items)` return when items holds 3 lists with 2 values each, for 10 users?')
    add('Fix this code so it prints 1 to 10:\n```python\nfor i in range(10):\n    print(i)\n```\nKeep it short.')
    add('Suggest a name for a small bakery in a coastal town.')
    add('Suggest a name for a small bakery in a coastal town.')  # an exact duplicate prompt
    for marker in CHAT_MARKERS:
        add(f'{marker} Name three everyday uses of a paperclip and say why each one works.')
    for marker in TRANSLATION_MARKERS:
        add(_TR_BODY.get(marker, f'{marker} Summarize the following paragraph about rivers in two sentences please.'))
    add('מתי נוסדה מדינת ישראל ומי הכריז על כך?', lang='he', source='oasst2')
    add('ספר לי בקצרה על מחזורי שינה ועל הסיבה שהם חשובים.', lang='he', source='oasst2')
    add('כתוב לי רשימה קצרה של שלושה רעיונות לשם לחנות ספרים MODE_EN_REPLY', lang='he', source='oasst2')
    return rows


def _sha(path):
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else None


def _parse(module, argv):
    ap = argparse.ArgumentParser()
    module.add_args(ap)
    return ap.parse_args(argv)


def _client(server, stats_sink, retries=4):
    client = Client(server.url, model='auto', max_retries=retries, timeout=60.0, sleep=lambda s: None)
    stats_sink.append(client)
    return client


def _tear(path):
    """What a SIGKILL mid-write leaves behind: a last line with no newline."""
    with open(path, 'ab') as f:
        f.write(b'{"id": "torn-')


def _merge_stats(clients):
    total = Counter()
    for c in clients:
        total.update(c.stats)
    return dict(total)


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, name, ok, detail=''):
        self.rows.append((name, bool(ok), str(detail)))

    @property
    def failed(self):
        return [r for r in self.rows if not r[1]]

    def as_dict(self):
        return {name: ('ok' if ok else f'FAILED {detail}'.strip()) for name, ok, detail in self.rows}


def _refused(row):
    """A row the server refused (HTTP 400): recorded, never final, sent again by a resume."""
    return bool(row.get('error')) or row.get('failure') == 'http_400'


def _strict_lines(path):
    """Every line of a ledger parses, none twice: what resume must leave behind."""
    rows = []
    for n, line in enumerate(Path(path).read_text(encoding='utf-8').split('\n')[:-1], 1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise AssertionError(f'{path}: line {n} is not JSON ({error})') from error
    return rows


def render_check(out, items_by_id, template_tools, office_env, n=25):
    """Train/serve equality on kept rows: the text the training template renders for a kept tool row equals the text
    the serving template renders for the wire transcript, and train27.encode_chat finds labeled assistant tokens."""
    tok = load_tokenizer()
    if tok is None:
        return {'status': 'skipped: no tokenizer on this machine'}
    encode_chat = load_encode_chat()
    res = {'status': 'ok', 'chat_rows': 0, 'tool_rows': 0, 'tool_render_equal': 0, 'labeled_tokens_min': None,
           'tool_call_markup_rows': 0}
    problems = []
    chat = read_jsonl(Path(out) / 'chat_sft.jsonl')[:n]
    raw = {(r['item_id'], int(r['sample'])): r for r in read_jsonl(Path(out) / 'tool_raw.jsonl')}
    tools = read_jsonl(Path(out) / 'tool_sft.jsonl')
    by_kind = {}
    for row in tools:  # a spread across kinds, not the first n by id
        by_kind.setdefault(row['kind'], []).append(row)
    picked = [rows[i] for i in range(3) for rows in by_kind.values() if i < len(rows)][:n * 2]
    counts = []
    for row in chat:
        ids, mask = encode_chat(tok, row['messages'])
        counts.append(sum(mask))
        answer = row['messages'][1]['content']
        labeled = tok.decode([t for t, m in zip(ids, mask) if m])
        if not answer[:40] in labeled or sum(mask) == 0:
            problems.append(f"chat {row['id']}: the answer is not in the labeled span")
        res['chat_rows'] += 1
    for row in picked:
        item_id, sample = row['id'].rsplit('#s', 1)
        wire = raw[(item_id, int(sample))]['messages']
        serve = office_env.render_text(tok, wire, template_tools(items_by_id[item_id]['tools']),
                                       add_generation_prompt=False)
        train = tok.apply_chat_template(row['messages'], tools=row['tools'], tokenize=False, enable_thinking=False)
        res['tool_rows'] += 1
        if serve == train:
            res['tool_render_equal'] += 1
        else:
            problems.append(f"tool {row['id']}: serving render differs from the training render")
        ids, mask = encode_chat(tok, row['messages'], row['tools'])
        counts.append(sum(mask))
        labeled = tok.decode([t for t, m in zip(ids, mask) if m])
        if any(m.get('tool_calls') for m in row['messages'] if m['role'] == 'assistant'):
            if '<tool_call>' in labeled and '<function=' in labeled:
                res['tool_call_markup_rows'] += 1
            else:
                problems.append(f"tool {row['id']}: tool-call markup is not inside the labeled span")
        if not sum(mask):
            problems.append(f"tool {row['id']}: no labeled tokens")
    res['labeled_tokens_min'] = min(counts) if counts else None
    if problems:
        res['status'] = 'FAILED'
        res['problems'] = problems[:10]
    return res


def dry_run(argv):
    """The whole pipeline on CPU against sd_fake_server, three ways:

    clean     one uninterrupted run with a healthy transport
    hostile   the same inputs through 503s, 429s, dropped connections and garbled bodies, with the server killed
              mid-translation, mid-answers and mid-tool-run, a torn last line after every kill, and a resume
    and requires the hostile run to end byte-identical to the clean one, every filter reason to fire, every request
    to have the right shape (think-off, vendor sampling arm, wire-form tool calls), and kept rows to render."""
    from sd_fake_server import Faults, FakeServer

    ap = argparse.ArgumentParser(prog='run.py dry-run', description=dry_run.__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work-dir', default=None, help='scratch directory (default: a fresh temp dir, deleted at the end)')
    ap.add_argument('--keep', action='store_true', help='keep the scratch directory')
    ap.add_argument('--items-per-kind', type=int, default=6)
    ap.add_argument('--n-samples', type=int, default=4)
    ap.add_argument('--concurrency', type=int, default=16)
    ap.add_argument('--no-render', action='store_true', help='skip the tokenizer render check')
    args = ap.parse_args(argv)
    started = time.time()
    work = Path(args.work_dir or tempfile.mkdtemp(prefix='selfdistill-dry-'))
    work.mkdir(parents=True, exist_ok=True)
    checks, summary, clients = Checks(), {'work_dir': str(work)}, []
    try:
        lane_before = lane_hashes()
        office_env, ag, items = load_pool()
        items_by_id = {i['id']: i for i in items}
        prompts = dry_run_prompts()
        prompts_path = work / 'chat_prompts.jsonl'
        write_jsonl_atomic(prompts_path, prompts)
        eligible = chat_gen.translation_candidates(prompts, len(prompts), 0)
        n_translate = len(eligible)  # every English prompt is tried, so every translation reason is reachable

        common = ['--concurrency', str(args.concurrency), '--abort-after', '5', '--progress-every', '0']
        chat_argv = lambda out: common + ['--out-dir', str(out), '--prompts', str(prompts_path),  # noqa: E731
                                          '--n-translate', str(n_translate), '--max-tokens', '512']
        tool_argv = lambda out: common + ['--out-dir', str(out), '--n-samples', str(args.n_samples),  # noqa: E731
                                          '--item-limit-per-kind', str(args.items_per_kind), '--keep-per-item', '2']
        chosen, per_kind = tool_gen.select_items(items, _parse(tool_gen, tool_argv(work / 'x')))
        by_kind = {}
        for it in chosen:
            by_kind.setdefault(it['kind'], []).append(it['id'])
        stall = [by_kind['single'][0], by_kind['chain'][0]]
        http400 = [by_kind['long_result'][0], by_kind['multi_turn'][0]]
        drift = [next(it['id'] for it in chosen if it['kind'] == 'parallel' and (
            hebrew_share(it['turns'][0]['user']) or 0) >= 0.6)]  # a Hebrew request, answered in English
        summary['inputs'] = {'prompts': len(prompts), 'translation_candidates': n_translate,
                             'tool_items': len(chosen), 'tool_items_per_kind': per_kind,
                             'tool_episodes': len(chosen) * args.n_samples, 'stalled_items': stall,
                             'http400_items': http400, 'english_reply_items': drift}

        def server(faults):
            return FakeServer(items, faults=faults, http400_item_ids=http400, stall_item_ids=stall,
                              drift_item_ids=drift)

        # ---- clean
        clean = work / 'clean'
        with server(Faults()) as srv:
            c = chat_gen.run(_parse(chat_gen, chat_argv(clean)), client=_client(srv, clients))
            t = tool_gen.run(_parse(tool_gen, tool_argv(clean)), client=_client(srv, clients))
            clean_stats = srv.stats()
        for name, s in (('chat', c), ('tools', t)):
            checks.add(f'clean {name} finished', not s.get('aborted') and not any(
                v.get('failed_unwritten') for v in s.values() if isinstance(v, dict)) and not s.get('failed_unwritten'))
        report = json.loads((clean / 'filter_report.json').read_text())
        summary['clean'] = {'chat': c, 'tools': {k: t[k] for k in t if k != 'client'}, 'server': clean_stats}

        # ---- hostile
        hostile = work / 'hostile'
        faults = Faults(every_503=37, every_429=53, every_drop=29, every_garbled=71)
        runs = {}
        with server(faults) as srv:
            srv.kill_after(10)  # dies during translation
            r1 = chat_gen.run(_parse(chat_gen, chat_argv(hostile)), client=_client(srv, clients, retries=2))
            checks.add('kill 1 aborts the chat run', bool(r1.get('aborted')), r1)
            n_tr = len(read_jsonl(hostile / 'translations.jsonl'))
            checks.add('kill 1 landed in translation', n_tr > 0 and not (hostile / 'chat_raw.jsonl').exists(),
                       f'{n_tr} translation rows, chat_raw exists={(hostile / "chat_raw.jsonl").exists()}')
            _tear(hostile / 'translations.jsonl')
            srv.revive()
            srv.kill_after(n_translate + 15)  # dies during the answers
            r2 = chat_gen.run(_parse(chat_gen, chat_argv(hostile)), client=_client(srv, clients, retries=2))
            n_ans = len(read_jsonl(hostile / 'chat_raw.jsonl'))
            checks.add('kill 2 aborts the chat run', bool(r2.get('aborted')), r2)
            checks.add('kill 2 landed in the answers', n_ans > 0, f'{n_ans} answer rows')
            _tear(hostile / 'chat_raw.jsonl')
            n_ans = len({r['id'] for r in read_jsonl(hostile / 'chat_raw.jsonl') if not _refused(r)})
            srv.revive()
            r3 = chat_gen.run(_parse(chat_gen, chat_argv(hostile)), client=_client(srv, clients))
            checks.add('chat resume finishes', not r3.get('aborted') and not r3['answers']['failed_unwritten'], r3)
            checks.add('chat resume skipped the finished rows', r3['answers']['resumed'] == n_ans,
                       f"resumed {r3['answers']['resumed']}, ledger had {n_ans}")
            srv.kill_after(150)  # dies during the tool run, after the first 16 episodes (2 to 6 requests each) closed
            r4 = tool_gen.run(_parse(tool_gen, tool_argv(hostile)), client=_client(srv, clients, retries=2))
            n_eps = len(read_jsonl(hostile / 'tool_raw.jsonl'))
            checks.add('kill 3 aborts the tool run', bool(r4.get('aborted')) and n_eps > 0, f'{n_eps} episode rows')
            _tear(hostile / 'tool_raw.jsonl')
            n_eps = len({(r['item_id'], r['sample']) for r in read_jsonl(hostile / 'tool_raw.jsonl')
                         if not _refused(r)})
            srv.revive()
            r5 = tool_gen.run(_parse(tool_gen, tool_argv(hostile)), client=_client(srv, clients))
            checks.add('tool resume finishes', not r5.get('aborted') and not r5.get('failed_unwritten'), r5)
            checks.add('tool resume skipped the finished episodes', r5['resumed'] == n_eps,
                       f"resumed {r5['resumed']}, ledger had {n_eps}")
            hostile_stats = srv.stats()
        runs.update(kills=[bool(r1.get('aborted')), bool(r2.get('aborted')), bool(r4.get('aborted'))])
        for name in ('translations', 'chat_raw', 'tool_raw'):
            rows = _strict_lines(hostile / f'{name}.jsonl')
            key = (lambda r: (r['item_id'], r['sample'])) if name == 'tool_raw' else (lambda r: r['id'])  # noqa: E731
            by_key = {}
            for r in rows:
                by_key.setdefault(key(r), []).append(r)
            # a refused prompt or episode is tried again on resume (so it may appear twice); a finished one never
            resent_final = [k for k, v in by_key.items() if any(not _refused(x) for x in v[:-1])]
            checks.add(f'hostile {name} ledger is clean', not resent_final,
                       f'{len(rows)} lines, {len(by_key)} distinct, {len(rows) - len(by_key)} refused rows retried, '
                       f'finished rows sent again: {resent_final[:3]}')
        for name in ('chat_sft.jsonl', 'tool_sft.jsonl'):
            checks.add(f'hostile {name} equals the clean run', _sha(hostile / name) == _sha(clean / name) is not None,
                       f'{_sha(hostile / name)} vs {_sha(clean / name)}')
        h_report = json.loads((hostile / 'filter_report.json').read_text())
        for part in ('translation', 'chat', 'tools'):
            checks.add(f'hostile report {part} equals the clean run', h_report[part] == report[part])
        summary['hostile'] = {'server': hostile_stats, 'client': _merge_stats(clients), **runs}

        # ---- server-side shape checks, every request of every pass
        for label, st in (('clean', clean_stats), ('hostile', hostile_stats)):
            checks.add(f'{label}: every request had the right shape', st['n_violations'] == 0, st['violations'][:3])
            checks.add(f'{label}: fake server had no internal error', not st['internal_errors'], st['internal_errors'])
        hc = hostile_stats['counts']
        for kind in ('fault_503', 'fault_429', 'fault_drop', 'fault_garbled', 'fault_dead'):
            checks.add(f'hostile injected {kind}', hc.get(kind, 0) > 0, hc.get(kind, 0))
        cs = _merge_stats(clients)
        for key in ('retries', 'reconnects', 'exhausted', 'http_error_400'):
            checks.add(f'client exercised {key}', cs.get(key, 0) > 0, cs)

        # ---- coverage: every filter reason fires, every failure class shows up
        for section, reasons in (('translation', sd_filters.TRANSLATION_REASONS), ('chat', sd_filters.CHAT_REASONS),
                                 ('tools', sd_filters.TOOL_REASONS)):
            fired = set(report[section]['dropped'])
            checks.add(f'every {section} filter reason fires', fired == set(reasons),
                       f'missing {sorted(set(reasons) - fired)}')
        failures = Counter()
        for r in read_jsonl(clean / 'tool_raw.jsonl'):
            failures[r.get('failure') or 'solved'] += 1
        for f in ('solved', 'wrong_calls', 'turn_budget', 'truncated', 'http_400', 'bad_arguments'):
            checks.add(f'tool episodes include {f}', failures[f] > 0, dict(failures))
        kinds = report['tools']['by_kind']
        checks.add('per-kind kept rates cover all nine kinds', set(kinds) == set(ag.KINDS), sorted(kinds))
        checks.add('every kind has at least one kept transcript', all(k['kept'] > 0 for k in kinds.values()),
                   {n: k['kept'] for n, k in kinds.items()})
        tool_sft = read_jsonl(clean / 'tool_sft.jsonl')
        per_item = Counter(r['item_id'] for r in tool_sft)
        checks.add('at most keep-per-item transcripts per item', max(per_item.values()) <= 2, max(per_item.values()))
        checks.add('the lane files (agentic_env, agentic_pool, office_env) are byte-identical after the run',
                   lane_hashes() == lane_before, 'a lane file changed')
        checks.add('no monitor-split item was run', all(items_by_id[r['item_id']]['split'] == 'train' for r in tool_sft))
        spot = Counter(r['spot_source'] for r in read_jsonl(clean / 'spot_read.jsonl'))
        checks.add('spot sample covers every source', set(spot) == {'translation', 'chat_en', 'chat_he', 'tool'}, dict(spot))
        chat_sft = read_jsonl(clean / 'chat_sft.jsonl')
        checks.add('chat rows are user+assistant pairs', all(
            [m['role'] for m in r['messages']] == ['user', 'assistant'] for r in chat_sft))
        checks.add('hebrew answers exist for translated prompts', any(r.get('meta', {}).get('translated_from') for r in chat_sft))
        raw_chat = read_jsonl(clean / 'chat_raw.jsonl')
        ok_tr, _ = sd_filters.filter_translations(read_jsonl(clean / 'translations.jsonl'))
        checks.add('every prompt and every good translation was answered once',
                   len(raw_chat) == len(prompts) + len(ok_tr), f'{len(raw_chat)} vs {len(prompts)}+{len(ok_tr)}')
        manifest = json.loads((clean / 'manifest_tools.json').read_text())
        checks.add('manifest pins the lane files by sha256', manifest['lane'] == lane_before)
        checks.add('manifest records think-off and the vendor arm', manifest['think'] == 'off'
                   and manifest['sampling']['presence_penalty'] == 1.5 and manifest['sampling']['temperature'] == 0.7)

        # ---- render
        if args.no_render:
            summary['render'] = {'status': 'skipped: --no-render'}
        else:
            summary['render'] = render_check(clean, items_by_id, office_env.template_tools, office_env)
            checks.add('kept rows render like the server and carry labels', summary['render']['status'] in (
                'ok', 'skipped: no tokenizer on this machine'), summary['render'])

        # ---- the summary
        summary['chat_report'] = {k: report['chat'][k] for k in ('rows', 'kept', 'kept_rate', 'dropped', 'by_lang')}
        summary['translation_report'] = report['translation']
        summary['tool_report'] = {'episodes': report['tools']['episodes'], 'kept': report['tools']['kept'],
                                  'kept_rate': report['tools']['kept_rate'], 'dropped': report['tools']['dropped'],
                                  'by_kind': {n: {k: v[k] for k in ('items', 'episodes', 'solved_rate', 'pass_at_n',
                                                                      'kept', 'kept_rate', 'not_solved_by')}
                                              for n, v in kinds.items()}}
        summary['tool_failure_classes'] = dict(failures)
    finally:
        summary['seconds'] = round(time.time() - started, 1)
        summary['checks'] = checks.as_dict()
        summary['passed'] = not checks.failed and len(checks.rows) > 0
        if args.keep or args.work_dir:
            summary['kept_work_dir'] = str(work)
        else:
            shutil.rmtree(work, ignore_errors=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if checks.failed:
        for name, _, detail in checks.failed:
            log('CHECK FAILED:', name, detail[:300])
        return 1
    log(f'dry run passed: {len(checks.rows)} checks in {summary["seconds"]}s')
    return 0


COMMANDS = {'chat': chat_gen.main, 'tools': tool_gen.main, 'filter': cmd_filter, 'all': cmd_all, 'dry-run': dry_run}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        print('usage: run.py {' + ','.join(COMMANDS) + '} [options]   (run.py <command> --help)', file=sys.stderr)
        return 0 if argv and argv[0] in ('-h', '--help') else 2
    return COMMANDS[argv[0]](argv[1:])


if __name__ == '__main__':
    sys.exit(main())
