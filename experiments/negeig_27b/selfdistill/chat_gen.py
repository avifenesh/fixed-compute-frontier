"""Chat replay from the untouched Qwen3.8-27B.

  (a) translate a sample of 1,500 English prompts into natural Hebrew with the model itself;
  (b) answer every English prompt and every Hebrew prompt once, think-off, with the vendor sampling arm.

Reads   s4/chat_prompts.jsonl (6,002 human-written prompts: dolly-15k and oasst2)
Writes  <out>/translations.jsonl   raw translation ledger (one row per attempt, ok or not)
        <out>/chat_raw.jsonl       raw answer ledger, resumable
        <out>/chat_sft.jsonl       filtered rows in chat-message format: {id, source, lang, messages, meta}
        <out>/filter_report.json, spot_read.jsonl (see sd_filters.py)

Every label is this model's own output. Rerun the same command to resume: finished ids are skipped, a torn last
line is cut, and a request that failed every retry is left out of the ledger so the rerun tries it again.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import sd_filters
from sd_common import (CJK, TRANSLATE_SYSTEM, Aborted, ApiError, FailureGate, JsonlSink, add_client_args,
                       add_run_args, file_sha256, hebrew_share, lane_hashes, log, make_client, prose, read_jsonl,
                       run_pool, sample_stratified, sampling_profile, seed_for, write_manifest)

MAX_SRC_CHARS = 2500
POOL_FACTOR = 2.0


def load_prompts(path):
    rows = read_jsonl(path)
    assert rows, f'no prompts in {path}'
    seen = set()
    for r in rows:
        assert r['id'] not in seen, f"duplicate prompt id {r['id']}"
        seen.add(r['id'])
        assert r['lang'] in ('en', 'he') and isinstance(r['prompt'], str), r['id']
    return rows


def translation_candidates(rows, n, seed):
    """A deterministic order over English prompts worth translating: a category-stratified sample of 2n prompts,
    shuffled. Any prefix is about stratified (not exact), and the tail is the spare pool for translations that fail
    validation. 2x, not less: a validation pass rate under 1/POOL_FACTOR would leave the target unreachable."""
    eligible = [r for r in rows if r['lang'] == 'en' and len(r['prompt']) <= MAX_SRC_CHARS
                and hebrew_share(r['prompt']) is not None and not CJK.search(prose(r['prompt']))]
    eligible.sort(key=lambda r: r['id'])
    pool = int(math.ceil(n * POOL_FACTOR))
    return sample_stratified(eligible, pool, key=lambda r: r['category'], rng=random.Random(seed_for('translate', seed)))


def _unwrap(text):
    t = text.strip()
    if t.startswith('<message>') and t.endswith('</message>'):
        t = t[len('<message>'):-len('</message>')].strip()
    return t


def translate_one(client, args, sampling, row):
    messages = [{'role': 'system', 'content': TRANSLATE_SYSTEM},
                {'role': 'user', 'content': f"<message>\n{row['prompt']}\n</message>"}]
    seed = seed_for(args.seed, 'tr', row['id'])
    gen = client.chat(messages, max_tokens=args.translate_max_tokens, sampling=sampling, seed=seed)
    return {'id': row['id'], 'src': row['prompt'], 'out': _unwrap(gen.content), 'finish': gen.finish_reason,
            'usage': gen.usage, 'category': row['category'], 'source': row['source'], 'seed': seed}


def translate_phase(client, args, rows, target):
    """Translate until `target` translations pass validation, in waves sized from the failure rate so far."""
    path = Path(args.out_dir) / 'translations.jsonl'
    sampling = sampling_profile(args.translate_sampling or args.sampling, args.generation_config)
    order = translation_candidates(rows, target, args.seed)
    # An error row (the server refused the prompt) is a record, not an attempt: a resume sends that prompt again, so
    # a server that was misconfigured for one run cannot leave a ledger that looks finished.
    ledger = {r['id']: r for r in read_jsonl(path)}
    ledger = {i: r for i, r in ledger.items() if not r.get('error')}
    good = {i for i, r in ledger.items() if sd_filters.translation_reason(r) is None}
    tried = set(ledger)
    stats = Counter()
    gate = FailureGate(args.abort_after)
    with JsonlSink(path) as sink:
        if sink.repaired:
            log(f'translations.jsonl: cut a torn {sink.repaired}-byte last line')
        wave_no = 0
        while len(good) < target:
            pending = [r for r in order if r['id'] not in tried]
            if not pending:
                break
            attempts = len(ledger)
            pass_rate = len(good) / attempts if attempts >= 20 else 0.85  # validation pass rate so far
            need = target - len(good)
            size = min(len(pending), int(math.ceil(need / max(pass_rate, 0.3) * 1.05)) + 1)
            wave = pending[:size]
            wave_no += 1
            log(f'translate wave {wave_no}: {len(good)}/{target} ok, sending {len(wave)}')

            def on_result(task, result):
                nonlocal good
                result['reason'] = sd_filters.translation_reason(result)
                result['ok'] = result['reason'] is None
                sink.write(result)
                ledger[result['id']] = result
                stats['done'] += 1
                gate.ok()
                if result['ok']:
                    good.add(result['id'])

            def on_error(task, error):
                tried.add(task['id'])
                if isinstance(error, ApiError) and not error.retryable:
                    sink.write({'id': task['id'], 'src': task['prompt'], 'error': {'status': error.status,
                                                                                  'body': str(error.body)[:300]}})
                    stats['api_error'] += 1
                    gate.fail(error)  # a streak of refusals is a server or request bug, not bad prompts
                    return
                stats['failed'] += 1
                gate.fail(error)

            for r in wave:
                tried.add(r['id'])
            run_pool(wave, lambda task: translate_one(client, args, sampling, task), concurrency=args.concurrency,
                     on_result=on_result, on_error=on_error, label='translate', total=len(wave),
                     progress_every=args.progress_every)
    shortfall = max(0, target - len(good))
    if shortfall:
        log(f'translation shortfall: {len(good)} of {target} passed validation, candidates exhausted or failed')
    return {'target': target, 'ok': min(len(good), target), 'surplus': max(0, len(good) - target),
            'shortfall': shortfall, 'attempted': len(ledger), 'refused': stats['api_error'],
            'failed_unwritten': stats['failed']}


def chosen_translations(out, prompts, args):
    """Exactly n_translate valid translations: the first ones in candidate order. A wave can overshoot the target,
    and the extra rows stay in the ledger unanswered, so the Hebrew share of the answers is what was asked for."""
    if not args.n_translate:
        return []
    ok, _ = sd_filters.filter_translations(read_jsonl(Path(out) / 'translations.jsonl'))
    usable = {t['id']: t for t in ok}
    order = translation_candidates(prompts, args.n_translate, args.seed)
    return [usable[r['id']] for r in order if r['id'] in usable][:args.n_translate]


def build_answer_tasks(prompts, translations):
    tasks = [{'id': r['id'], 'lang': r['lang'], 'source': r['source'], 'category': r['category'],
              'prompt': r['prompt']} for r in prompts]
    by_id = {r['id']: r for r in prompts}
    for t in translations:
        src = by_id[t['id']]
        tasks.append({'id': f"{t['id']}-he", 'lang': 'he', 'source': src['source'], 'category': src['category'],
                      'prompt': t['out'], 'translated_from': t['id']})
    tasks.sort(key=lambda t: t['id'])
    return tasks


def answer_one(client, args, sampling, task):
    seed = seed_for(args.seed, 'chat', task['id'])
    gen = client.chat([{'role': 'user', 'content': task['prompt']}], max_tokens=args.max_tokens, sampling=sampling,
                      seed=seed)
    row = {'id': task['id'], 'lang': task['lang'], 'source': task['source'], 'category': task['category'],
           'prompt': task['prompt'], 'content': gen.content, 'reasoning': gen.reasoning,
           'finish': gen.finish_reason, 'usage': gen.usage, 'seed': seed}
    if gen.tool_calls:
        row['content'] = (row['content'] + '\n<tool_call>').strip()  # no tools were offered: a leak, filtered
    if task.get('translated_from'):
        row['translated_from'] = task['translated_from']
    return row


def answer_phase(client, args, tasks):
    path = Path(args.out_dir) / 'chat_raw.jsonl'
    sampling = sampling_profile(args.sampling, args.generation_config)
    done = {r['id'] for r in read_jsonl(path) if not r.get('error')}  # a refused prompt is tried again on resume
    todo = [t for t in tasks if t['id'] not in done]
    log(f'answers: {len(tasks)} prompts, {len(done)} already in the ledger, {len(todo)} to go')
    gate, stats = FailureGate(args.abort_after), Counter()
    with JsonlSink(path) as sink:
        if sink.repaired:
            log(f'chat_raw.jsonl: cut a torn {sink.repaired}-byte last line')

        def on_result(task, row):
            sink.write(row)
            stats['done'] += 1
            gate.ok()

        def on_error(task, error):
            if isinstance(error, ApiError) and not error.retryable:
                sink.write({'id': task['id'], 'lang': task['lang'], 'source': task['source'],
                            'category': task['category'], 'prompt': task['prompt'],
                            'error': {'status': error.status, 'body': str(error.body)[:300]}})
                stats['api_error'] += 1
                gate.fail(error)  # a streak of refusals is a server or request bug, not bad prompts
                return
            stats['failed'] += 1
            gate.fail(error)

        run_pool(todo, lambda t: answer_one(client, args, sampling, t), concurrency=args.concurrency,
                 on_result=on_result, on_error=on_error, label='answers', total=len(todo),
                 progress_every=args.progress_every)
    return {'prompts': len(tasks), 'resumed': len(done), 'written': stats['done'], 'api_error': stats['api_error'],
            'failed_unwritten': stats['failed']}


def add_args(ap, client_args=True, run_args=True):
    if client_args:
        add_client_args(ap)
    if run_args:
        add_run_args(ap)
    ap.add_argument('--prompts', default=None, help='chat_prompts.jsonl (default: the S4 file)')
    ap.add_argument('--n-translate', type=int, default=1500)
    ap.add_argument('--translate-sampling', default=None, choices=[None, 'vendor_nonthinking', 'generation_config'],
                    help='sampling arm for the translation step (default: same as --sampling)')
    ap.add_argument('--translate-max-tokens', type=int, default=3072)
    ap.add_argument('--max-tokens', type=int, default=3072, help='answer budget; a length finish is dropped')
    ap.add_argument('--limit', type=int, default=0, help='only the first N prompts by id (smoke runs)')
    return ap


def run(args, client=None):
    from sd_common import DEFAULT_PROMPTS
    started = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prompts_path = Path(args.prompts or DEFAULT_PROMPTS)
    prompts = sorted(load_prompts(prompts_path), key=lambda r: r['id'])
    if args.limit:
        prompts = prompts[:args.limit]
    client = client or make_client(args)
    if client.model == 'auto':
        client.model = client.discover_model()
    sampling = sampling_profile(args.sampling, args.generation_config)
    write_manifest(out, 'chat', {'model': client.model, 'sampling': sampling, 'think': 'off',
                                 'prompts_file': str(prompts_path), 'prompts_sha256': file_sha256(prompts_path),
                                 'prompts': len(prompts), 'n_translate': args.n_translate, 'seed': args.seed,
                                 'max_tokens': args.max_tokens, 'argv': sys.argv, 'lane': lane_hashes()})
    summary = {'model': client.model}
    try:
        if args.n_translate:
            summary['translate'] = translate_phase(client, args, prompts, args.n_translate)
        tasks = build_answer_tasks(prompts, chosen_translations(out, prompts, args))
        summary['answers'] = answer_phase(client, args, tasks)
    except Aborted as error:
        log(f'ABORTED: {error}. Finished rows are in the ledgers; rerun the same command to resume.')
        summary['aborted'] = str(error)
    summary['client'] = dict(client.stats)
    summary['seconds'] = round(time.time() - started, 1)
    if not args.no_filter:
        finalize_out(out, client.model, args)
    return summary


def finalize_out(out, model, args):
    """Filter the ledgers. The tool half needs the pool when a tool ledger sits beside the chat ledgers."""
    items_by_id = template_tools = None
    if (Path(out) / 'tool_raw.jsonl').exists():
        from sd_common import load_pool
        office_env, _, items = load_pool()
        items_by_id, template_tools = {i['id']: i for i in items}, office_env.template_tools
    return sd_filters.finalize(out, items_by_id=items_by_id, template_tools=template_tools,
                               meta={'model': model, 'sampling': args.sampling}, seed=args.seed)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    args = ap.parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    unfinished = sum(v.get('failed_unwritten', 0) for v in summary.values() if isinstance(v, dict))
    return 2 if summary.get('aborted') or unfinished else 0


if __name__ == '__main__':
    sys.exit(main())
