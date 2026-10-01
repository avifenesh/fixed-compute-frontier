#!/usr/bin/env python3
"""Report of the generative probe: per arm x task x mode, accuracy by session length and by position, tokens per turn
and per task (reasoning, content, total; medians and means), accuracy per 1k reasoning tokens, truncations; then paired
comparisons by item (wide vs ctrl, each vs the untouched model) with bootstrap intervals. One JSON, one markdown file.

Inputs: --root ROOT reads every ROOT/<arm>/<task>_<mode>/ledger.jsonl that run_all.sh writes (arm, task and mode come
from each run's manifest), or --run DIR (repeatable) names run directories directly.

Primary turns: on-call (run_oncall.py) the ACTION turns, the turns that end with a request (turn meta type "action";
an item without turn types falls back to the turns that expect calls); S1 (run_s1gen.py) every answer. Accuracy is the
micro mean of turn scores over primary turns, unreached and truncated turns included as 0 (truncations are also
reported on their own). Token columns are per reached primary turn, summed over the generations of the turn (a tool
loop has several), and per task (all reached turns of one item). reasoning = the tokenizer count of reasoning_content
(probe_common); content = completion - reasoning; total = completion tokens.

A context_overflow row (the session's prompt outgrew the server context; probe_common) is final and counts, its refused
turn and the unreached rest at 0, and is reported per cell; an http_400 row is not final and stays out.

On-call splits (data/oncall.py's note): accuracy by request kind and by kind/variant (ack_open's two phrasings read
"open" differently), call turns needing at most 4 calls vs 5 or more (agentic_env's 4-step limit), and the tokens of
the delivery turns on their own. S1: accuracy by domain.

Partial runs (--partial): an arm too slow to finish its sessions is read from DIR/partial_ledger.jsonl as well
(score_partial.py replays the in-flight sessions from gens.jsonl). Its closed turns are exact; the turn it is inside and
every later one are in flight and count nowhere: not in accuracy, tokens or truncations, and a pair compares only the
turns both arms reached. Exact sessions count finished sessions only; tokens per task of a partial session cover its
closed turns, so per-task numbers are comparable across arms only with --align, which cuts every arm's session to the
turns all arms of the cell reached (sessions some arm never started are dropped from the cell).

Paired comparisons use the items both arms finished or reached (same item id and sample). Intervals are 95% percentile bootstrap
over items (turn metrics resample whole items, so the turns of one session stay together), --draws resamples, seed
fixed. "reasoning per turn, both correct" compares reasoning tokens on the primary turns that BOTH arms scored 1.0: the
"same answer, less thinking" reading.

  report.py --root /workspace/negeig/probe/runs --out-json report.json --out-md report.md
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import sys
from pathlib import Path

POS_EDGES = [0, 64, 128, 256, 512, 1024, 2048]
TOKS = ('reasoning_tokens', 'content_tokens', 'completion_tokens', 'visible_tokens', 'server_reasoning_tokens')
SHORT = {'reasoning_tokens': 'reasoning', 'content_tokens': 'content', 'completion_tokens': 'total',
         'visible_tokens': 'visible', 'server_reasoning_tokens': 'server_reasoning'}


def read_jsonl(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def pos_bucket(after):
    if after is None:
        return 'n/a'
    if after == 0:
        return '0'
    for lo, hi in zip(POS_EDGES, POS_EDGES[1:]):
        if lo < after <= hi:
            return f'{lo + 1}-{hi}'
    return f'>{POS_EDGES[-1]}'


def bucket_order(b):
    if b == '0':
        return -1
    if b == 'n/a':
        return 10 ** 9
    if b.startswith('>'):
        return 10 ** 8
    return int(b.split('-')[0])


def stats(vals):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return {'n': 0, 'mean': None, 'median': None}
    n = len(vals)
    med = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
    return {'n': n, 'mean': sum(vals) / n, 'median': med}


def is_primary(task, rec, item_has_types):
    if task == 's1gen':
        return True
    if item_has_types:
        return rec.get('type') == 'action'
    return rec.get('expect') == 'call'


# ------------------------------------------------------------------ loading
def load_run(d, partial=False):
    d = Path(d)
    man = {}
    for p in sorted(d.glob('manifest_*.json')):
        try:
            man = json.loads(p.read_text(encoding='utf-8'))
        except ValueError:
            pass
    ledger = d / 'ledger.jsonl'
    if not ledger.exists():
        return None
    rows = {}
    for r in read_jsonl(ledger):  # the last row of a key wins, but a refused row never replaces a finished one
        prev = rows.get(r['key'])
        if prev is not None and prev.get('failure') != 'http_400' and r.get('failure') == 'http_400':
            continue
        rows[r['key']] = r
    n_partial = 0
    if partial and (d / 'partial_ledger.jsonl').exists():
        for r in read_jsonl(d / 'partial_ledger.jsonl'):
            prev = rows.get(r['key'])
            if prev is None or prev.get('failure') == 'http_400':  # a final ledger row always wins
                rows[r['key']] = r
                n_partial += bool(r.get('partial'))
    task = man.get('task') or ('oncall' if d.name.startswith('oncall') else 's1gen')
    arm = man.get('arm') or d.parent.name
    mode = man.get('mode') or d.name.split('_', 1)[-1]
    summary = {}
    if (d / 'summary.json').exists():
        try:
            summary = json.loads((d / 'summary.json').read_text())
        except ValueError:
            pass
    return {'dir': str(d), 'task': task, 'arm': arm, 'mode': mode, 'rows': rows, 'manifest': man,
            'summary': summary, 'partial_rows': n_partial}


def live_turns(r):
    """The turns of a row that are decided: all of them, less the in-flight tail of a partial session."""
    return [t for t in r['turns'] if not t.get('in_flight')]


def align_runs(runs):
    """Cut each session, in every arm of a task x mode cell, to the turns all arms reached (in-flight turns of any arm
    become in flight in all), and drop the sessions some arm of the cell has no row for."""
    cells = collections.defaultdict(list)
    for run in runs:
        cells[(run['task'], run['mode'])].append(run)
    out = []
    for cell in cells.values():
        keys = set.intersection(*[{k for k, r in run['rows'].items() if r.get('failure') != 'http_400'}
                                  for run in cell])
        cut = {}
        for k in keys:
            n = min(len(run['rows'][k]['turns']) for run in cell)
            for run in cell:
                ts = run['rows'][k]['turns']
                live = next((i for i, t in enumerate(ts) if t.get('in_flight')), len(ts))
                n = min(n, live)
            cut[k] = n
        for run in cell:
            rows = {}
            for k in keys:
                r = run['rows'][k]
                if cut[k] < len(r['turns']):
                    r = dict(r, partial=True, aligned_cut=cut[k],
                             turns=[t if i < cut[k] else dict(t, in_flight=True) for i, t in enumerate(r['turns'])])
                rows[k] = r
            out.append(dict(run, rows=rows, aligned=True,
                            aligned_dropped=sum(1 for k, r in run['rows'].items()
                                                if k not in keys and r.get('failure') != 'http_400')))
    return out


def discover(root):
    return sorted(str(p.parent) for p in Path(root).glob('*/*/ledger.jsonl'))


# ------------------------------------------------------------------ per cell
def cell_metrics(run):
    """Final rows only: an http_400 row is not final (rerun on resume) and stays out; a context_overflow row is final
    (the session could not go on) and counts, its refused turn and the unreached rest scoring 0."""
    task, rows = run['task'], [r for r in run['rows'].values() if r.get('failure') != 'http_400']
    refused = sum(1 for r in run['rows'].values() if r.get('failure') == 'http_400')
    prim, allt, per_item = [], [], []
    by_len, by_pos = collections.defaultdict(list), collections.defaultdict(list)
    by_kind, by_variant = collections.defaultdict(list), collections.defaultdict(list)
    extra = collections.defaultdict(list)
    gens = trunc_gens = 0
    for r in rows:
        has_types = any(t.get('type') for t in r['turns'])
        item_tot = {k: 0 for k in TOKS}
        for t in live_turns(r):
            allt.append(t)
            gens += t.get('n_gens') or 0
            trunc_gens += t.get('truncated_gens') or 0
            if t.get('reached'):
                for k in TOKS:
                    item_tot[k] += t.get(k) or 0
            if is_primary(task, t, has_types):
                prim.append(t)
                by_len[r.get('n_events')].append(t['score'])
                by_pos[pos_bucket(t.get('after_events'))].append(t['score'])
                kind = t.get('action') if task == 'oncall' else r.get('domain')
                if kind is not None:
                    by_kind[kind].append(t)
                    if task == 'oncall' and t.get('variant') is not None:
                        by_variant[f"{kind}/v{t['variant']}"].append(t)
            if task == 'oncall':
                if t.get('type') == 'events':
                    extra['events_turn_acc'].append(t['score'])
                if t.get('type') == 'action' and t.get('expect') == 'no_call':
                    extra['action_no_call_acc'].append(t['score'])
                if t.get('expect') == 'call':
                    extra['call_turn_acc'].append(t['score'])
                    # agentic_env's 4-step limit caps a one-call-per-response policy at 4/n on turns needing 5+
                    extra['call_turn_acc_le4' if (t.get('n_expected') or 0) <= 4 else 'call_turn_acc_ge5'].append(
                        t['score'])
            else:
                extra['acc_strict'].append(float(bool(t.get('ok_strict'))))
                a = t.get('after_events')
                if a is not None:
                    extra['le256' if a <= 256 else 'gt256'].append(t['score'])
                    if a > 512:
                        extra['gt512'].append(t['score'])
        per_item.append(item_tot)
    reached_prim = [t for t in prim if t.get('reached')]

    def group(ts):
        rt = [t for t in ts if t.get('reached')]
        return {'acc': mean([t['score'] for t in ts]), 'n': len(ts),
                'reasoning_mean': mean([t.get('reasoning_tokens') for t in rt]),
                'total_mean': mean([t.get('completion_tokens') for t in rt])}

    m = {'items': len(rows), 'refused_items': refused, 'primary_turns': len(prim), 'turns': len(allt),
         'context_overflow_items': sum(r.get('failure') == 'context_overflow' for r in rows),
         'context_overflow_turns_lost': sum(r.get('unreached') or 0 for r in rows
                                            if r.get('failure') == 'context_overflow'),
         'by_kind': {k: group(v) for k, v in sorted(by_kind.items())},
         'by_kind_variant': {k: group(v) for k, v in sorted(by_variant.items())},
         'acc': mean([t['score'] for t in prim]), 'acc_all_turns': mean([t['score'] for t in allt]),
         'exact_items': mean([float(all(t['score'] == 1.0 for t in r['turns'])) for r in rows
                              if not r.get('partial')]),
         'partial_items': sum(bool(r.get('partial')) for r in rows),
         'in_flight_turns': sum(len(r['turns']) - len(live_turns(r)) for r in rows),
         'by_length': {str(k): {'acc': mean(v), 'n': len(v)} for k, v in sorted(by_len.items(),
                                                                                  key=lambda kv: kv[0] or 0)},
         'by_position': {k: {'acc': mean(v), 'n': len(v)} for k, v in sorted(by_pos.items(),
                                                                              key=lambda kv: bucket_order(kv[0]))},
         'per_turn': {SHORT[k]: stats([t.get(k) for t in reached_prim]) for k in TOKS},
         'per_turn_all': {SHORT[k]: stats([t.get(k) for t in allt if t.get('reached')]) for k in TOKS},
         'per_turn_events': ({SHORT[k]: stats([t.get(k) for t in allt
                                               if t.get('reached') and t.get('type') == 'events'])
                              for k in TOKS} if task == 'oncall' else None),
         'per_task': {SHORT[k]: stats([it[k] for it in per_item]) for k in TOKS},
         'latency_per_turn_s': stats([t.get('latency_s') for t in reached_prim]),
         'generations': gens, 'truncated_generations': trunc_gens,
         'truncation_rate_generations': (trunc_gens / gens) if gens else None,
         'truncated_turns': sum(t.get('status') == 'truncated' for t in allt),
         'truncation_rate_turns': mean([float(t.get('status') == 'truncated') for t in allt if t.get('reached')]),
         'items_with_truncation': sum(any(t.get('status') == 'truncated' for t in r['turns']) for r in rows),
         'turn_budget_turns': sum(t.get('status') == 'turn_budget' for t in allt),
         'unreached_turns': sum(not t.get('reached') for t in allt),
         'bad_arguments_items': sum(any(t.get('status') == 'bad_arguments' for t in r['turns']) for r in rows)}
    m.update({k: mean(v) for k, v in extra.items()})
    rs = sum(t.get('reasoning_tokens') or 0 for t in reached_prim)
    m['acc_per_1k_reasoning'] = (sum(t['score'] for t in prim) / (rs / 1000)) if rs else None
    srv = [(t.get('server_reasoning_tokens'), t.get('reasoning_tokens')) for t in reached_prim
           if t.get('server_reasoning_tokens') is not None]
    m['server_minus_tokenizer_reasoning_per_turn'] = mean([a - b for a, b in srv]) if srv else None
    return m


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


# ------------------------------------------------------------------ paired comparisons
def item_vectors(run_a, run_b, task):
    keys = sorted(k for k in run_a['rows'] if k in run_b['rows'] and run_a['rows'][k].get('failure') != 'http_400'
                  and run_b['rows'][k].get('failure') != 'http_400')
    vec = collections.defaultdict(list)
    for k in keys:
        ra, rb = run_a['rows'][k], run_b['rows'][k]
        ta, tb = ra['turns'], rb['turns']
        if len(ta) != len(tb):
            raise SystemExit(f'{k}: the two arms have different turn lists ({len(ta)} vs {len(tb)}); same data?')
        has_types = any(t.get('type') for t in ta)
        n = sa = sb = 0
        rA = rB = cA = cB = nra = nrb = 0
        bc_n = bc_ra = bc_rb = 0
        kept = [(x, y) for x, y in zip(ta, tb) if not x.get('in_flight') and not y.get('in_flight')]
        for x, y in kept:
            if not is_primary(task, x, has_types):
                continue
            n += 1
            sa += x['score']
            sb += y['score']
            if x.get('reached'):
                nra += 1
                rA += x.get('reasoning_tokens') or 0
                cA += x.get('completion_tokens') or 0
            if y.get('reached'):
                nrb += 1
                rB += y.get('reasoning_tokens') or 0
                cB += y.get('completion_tokens') or 0
            if x['score'] == 1.0 and y['score'] == 1.0:
                bc_n += 1
                bc_ra += x.get('reasoning_tokens') or 0
                bc_rb += y.get('reasoning_tokens') or 0
        tot = lambda j, f: sum((p[j].get(f) or 0) for p in kept if p[j].get('reached'))  # noqa: E731
        for name, val in (('n', n), ('sa', sa), ('sb', sb), ('nra', nra), ('nrb', nrb), ('ra', rA), ('rb', rB),
                          ('ca', cA), ('cb', cB), ('bc_n', bc_n), ('bc_ra', bc_ra), ('bc_rb', bc_rb),
                          ('task_ra', tot(0, 'reasoning_tokens')), ('task_rb', tot(1, 'reasoning_tokens')),
                          ('task_ca', tot(0, 'completion_tokens')), ('task_cb', tot(1, 'completion_tokens'))):
            vec[name].append(float(val))
    return keys, vec


def _ratio(num, den):
    """num / den, NaN where den is 0; works on floats and on numpy arrays of bootstrap sums."""
    if hasattr(num, 'shape') or hasattr(den, 'shape'):
        import numpy as np
        num, den = np.broadcast_arrays(np.asarray(num, dtype=float), np.asarray(den, dtype=float))
        return np.divide(num, den, out=np.full(num.shape, np.nan), where=den != 0)
    return num / den if den else float('nan')


METRICS = {
    # name: (function of summed vectors -> value for A minus B, unit)
    'accuracy (primary turns)': (lambda s: _ratio(s['sa'], s['n']) - _ratio(s['sb'], s['n']), 'points'),
    'reasoning tokens per turn': (lambda s: _ratio(s['ra'], s['nra']) - _ratio(s['rb'], s['nrb']), 'tokens'),
    'total tokens per turn': (lambda s: _ratio(s['ca'], s['nra']) - _ratio(s['cb'], s['nrb']), 'tokens'),
    'reasoning per turn, both correct': (lambda s: _ratio(s['bc_ra'], s['bc_n']) - _ratio(s['bc_rb'], s['bc_n']),
                                         'tokens'),
    'reasoning tokens per task': (lambda s: (s['task_ra'] - s['task_rb']) / s['_items'], 'tokens'),
    'total tokens per task': (lambda s: (s['task_ca'] - s['task_cb']) / s['_items'], 'tokens'),
    'accuracy per 1k reasoning': (lambda s: _ratio(s['sa'], s['ra'] / 1000) - _ratio(s['sb'], s['rb'] / 1000),
                                  'per 1k'),
}


def bootstrap(vec, draws, seed):
    """{metric: (point, lo, hi)} with items resampled; numpy when present, else pure Python."""
    names = list(vec)
    N = len(vec['n'])
    if N == 0:
        return {}
    point_s = {k: sum(vec[k]) for k in names}
    point_s['_items'] = N
    out = {}
    try:
        import numpy as np
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, N, size=(draws, N))
        arr = {k: np.asarray(vec[k], dtype=float)[idx].sum(axis=1) for k in names}
        arr['_items'] = np.full(draws, N, dtype=float)
        for m, (fn, unit) in METRICS.items():
            with np.errstate(divide='ignore', invalid='ignore'):
                samples = fn(arr)
            samples = samples[np.isfinite(samples)]
            pt = fn(point_s)
            lo, hi = (np.percentile(samples, [2.5, 97.5]).tolist() if samples.size else (float('nan'),) * 2)
            out[m] = _fmt_ci(pt, lo, hi, unit)
    except ImportError:
        rng = random.Random(seed)
        draws = min(draws, 2000)
        samp = collections.defaultdict(list)
        for _ in range(draws):
            pick = [rng.randrange(N) for _ in range(N)]
            s = {k: sum(vec[k][i] for i in pick) for k in names}
            s['_items'] = N
            for m, (fn, _unit) in METRICS.items():
                v = fn(s)
                if not math.isnan(v):
                    samp[m].append(v)
        for m, (fn, unit) in METRICS.items():
            xs = sorted(samp[m])
            lo = xs[int(0.025 * (len(xs) - 1))] if xs else float('nan')
            hi = xs[int(0.975 * (len(xs) - 1))] if xs else float('nan')
            out[m] = _fmt_ci(fn(point_s), lo, hi, unit)
    return out


def _fmt_ci(pt, lo, hi, unit):
    scale = 100.0 if unit == 'points' else 1.0

    def f(v):
        return None if v is None or (isinstance(v, float) and math.isnan(v)) else v * scale
    return {'diff': f(pt), 'lo': f(lo), 'hi': f(hi), 'unit': unit}


def default_pairs(arms):
    wide = [a for a in arms if a.startswith('wide')]
    ctrl = [a for a in arms if a.startswith('ctrl')]
    base = [a for a in arms if a in ('untouched', 'base')]
    pairs = []
    for w in wide:
        suf = w[len('wide'):]
        for c in ctrl:
            if c[len('ctrl'):] == suf or len(ctrl) == 1:
                pairs.append((w, c))
    for b in base:
        pairs += [(a, b) for a in wide + ctrl]
    return pairs


def arm_order(a):
    return (0 if a in ('untouched', 'base') else 1 if a.startswith('ctrl') else 2 if a.startswith('wide') else 3, a)


# ------------------------------------------------------------------ output
def fmt(v, nd=1, pct=False):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return 'n/a'
    return f'{v * 100:.{nd}f}' if pct else (f'{v:.{nd}f}' if isinstance(v, float) else str(v))


def markdown(doc):
    out = ['# Generative probe report', '']
    out.append('Accuracy over primary turns (on-call: action turns; S1: every answer), unreached and truncated turns '
               'count 0. Tokens per reached primary turn (median / mean); per task = all reached turns of one item. '
               'reasoning = tokenizer count of reasoning_content; total = completion tokens.')
    for (task, mode), cells in sorted(doc['cells_by_task_mode'].items()):
        out += ['', f'## {task}, {mode}', '']
        lens = sorted({L for c in cells.values() for L in c['by_length']},
                      key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else 0, x))
        hdr = (['arm', 'items', 'acc %'] + [f'acc@{L}' for L in lens] +
               ['reasoning/turn med / mean', 'content/turn med / mean', 'total/turn mean', 'reasoning/task mean',
                'total/task mean', 'acc per 1k reasoning', 'trunc gens %', 'trunc turns %', 'unreached', 'budget',
                'ctx overflow items', 'in-flight items / turns'])
        out.append('| ' + ' | '.join(hdr) + ' |')
        out.append('|' + '---|' * len(hdr))
        for arm in sorted(cells, key=arm_order):
            c = cells[arm]
            pt = c['per_turn']
            row = [arm, str(c['items']), fmt(c['acc'], 1, True)]
            row += [fmt((c['by_length'].get(L) or {}).get('acc'), 1, True) for L in lens]
            row += [f"{fmt(pt['reasoning']['median'], 0)} / {fmt(pt['reasoning']['mean'], 0)}",
                    f"{fmt(pt['content']['median'], 0)} / {fmt(pt['content']['mean'], 0)}",
                    fmt(pt['total']['mean'], 0), fmt(c['per_task']['reasoning']['mean'], 0),
                    fmt(c['per_task']['total']['mean'], 0), fmt(c['acc_per_1k_reasoning'], 2),
                    fmt(c['truncation_rate_generations'], 2, True), fmt(c['truncation_rate_turns'], 2, True),
                    str(c['unreached_turns']), str(c['turn_budget_turns']), str(c.get('context_overflow_items', 0)),
                    f"{c.get('partial_items', 0)} / {c.get('in_flight_turns', 0)}"]
            out.append('| ' + ' | '.join(row) + ' |')
        kinds = sorted({k for c in cells.values() for k in c.get('by_kind', {})})
        if kinds:
            what = 'request kind' if task == 'oncall' else 'domain'
            out += ['', f'Accuracy % by {what} (reasoning tokens per reached turn, mean):', '']
            out.append('| arm | ' + ' | '.join(kinds) + ' |')
            out.append('|' + '---|' * (len(kinds) + 1))
            for arm in sorted(cells, key=arm_order):
                bk = cells[arm].get('by_kind', {})
                out.append('| ' + arm + ' | ' + ' | '.join(
                    f"{fmt((bk.get(k) or {}).get('acc'), 1, True)} ({fmt((bk.get(k) or {}).get('reasoning_mean'), 0)},"
                    f" n {(bk.get(k) or {}).get('n', 0)})" for k in kinds) + ' |')
        buckets = sorted({b for c in cells.values() for b in c['by_position']}, key=bucket_order)
        out += ['', 'Accuracy % by position (events delivered before the turn):', '']
        out.append('| arm | ' + ' | '.join(buckets) + ' |')
        out.append('|' + '---|' * (len(buckets) + 1))
        for arm in sorted(cells, key=arm_order):
            bp = cells[arm]['by_position']
            out.append('| ' + arm + ' | ' + ' | '.join(
                f"{fmt((bp.get(b) or {}).get('acc'), 1, True)} (n {(bp.get(b) or {}).get('n', 0)})" for b in buckets)
                + ' |')
        if task == 'oncall':
            out += ['', '| arm | call-turn acc % | calls <= 4 % | calls >= 5 % | action no-call acc % | '
                    'events-turn no-call % | all turns % | exact sessions % | events-turn reasoning mean |',
                    '|---|---|---|---|---|---|---|---|---|']
            for arm in sorted(cells, key=arm_order):
                c = cells[arm]
                ev = (c.get('per_turn_events') or {}).get('reasoning') or {}
                out.append(f"| {arm} | {fmt(c.get('call_turn_acc'), 1, True)} | "
                           f"{fmt(c.get('call_turn_acc_le4'), 1, True)} | {fmt(c.get('call_turn_acc_ge5'), 1, True)} | "
                           f"{fmt(c.get('action_no_call_acc'), 1, True)} | {fmt(c.get('events_turn_acc'), 1, True)}"
                           f" | {fmt(c.get('acc_all_turns'), 1, True)} | {fmt(c.get('exact_items'), 1, True)} | "
                           f"{fmt(ev.get('mean'), 0)} |")
        else:
            out += ['', '| arm | strict % | answers after event <= 256 % | > 256 % | > 512 % |',
                    '|---|---|---|---|---|']
            for arm in sorted(cells, key=arm_order):
                c = cells[arm]
                out.append(f"| {arm} | {fmt(c.get('acc_strict'), 1, True)} | {fmt(c.get('le256'), 1, True)} | "
                           f"{fmt(c.get('gt256'), 1, True)} | {fmt(c.get('gt512'), 1, True)} |")
    if doc['paired']:
        out += ['', '## Paired comparisons (A minus B, 95% bootstrap interval over items)', '',
                '| task | mode | A vs B | items | metric | diff | interval |', '|---|---|---|---|---|---|---|']
        for p in doc['paired']:
            for m, v in p['metrics'].items():
                out.append(f"| {p['task']} | {p['mode']} | {p['a']} vs {p['b']} | {p['items']} | {m} | "
                           f"{fmt(v['diff'], 2)} {v['unit']} | [{fmt(v['lo'], 2)}, {fmt(v['hi'], 2)}] |")
    if doc['notes']:
        out += ['', '## Notes', ''] + [f'- {n}' for n in doc['notes']]
    return '\n'.join(out) + '\n'


def build(runs, pairs=None, draws=10000, seed=20261002, align=False):
    if align:
        runs = align_runs(runs)
    cells = collections.defaultdict(dict)
    notes = []
    by = {}
    for run in runs:
        k = (run['task'], run['mode'])
        if run['arm'] in cells[k]:
            raise SystemExit(f"two runs for {run['arm']} {run['task']} {run['mode']}: {run['dir']}")
        cells[k][run['arm']] = cell_metrics(run)
        by[(run['task'], run['mode'], run['arm'])] = run
        if run.get('partial_rows'):
            notes.append(f"{run['arm']} {run['task']} {run['mode']}: {run['partial_rows']} sessions in flight, scored on "
                         f"their closed turns ({cells[k][run['arm']]['in_flight_turns']} turns in flight)")
        if run.get('aligned'):
            notes.append(f"{run['arm']} {run['task']} {run['mode']}: aligned, {cells[k][run['arm']]['partial_items']} "
                         f"sessions cut to the turns every arm reached, {run['aligned_dropped']} sessions dropped")
        if run['summary'] and not run['summary'].get('complete'):
            notes.append(f"{run['arm']} {run['task']} {run['mode']}: the run is not complete "
                         f"({run['summary'].get('rows_final')} final rows); its numbers cover what finished")
    paired = []
    for (task, mode), arms in sorted(cells.items()):
        for a, b in (pairs or default_pairs(sorted(arms, key=arm_order))):
            if a not in arms or b not in arms:
                continue
            keys, vec = item_vectors(by[(task, mode, a)], by[(task, mode, b)], task)
            if not keys:
                notes.append(f'{task} {mode}: {a} and {b} share no finished item')
                continue
            metrics = bootstrap(vec, draws, seed)
            if mode.startswith('think_off'):
                metrics.pop('accuracy per 1k reasoning', None)
                metrics.pop('reasoning per turn, both correct', None)
            paired.append({'task': task, 'mode': mode, 'a': a, 'b': b, 'items': len(keys), 'metrics': metrics})
    doc = {'cells': {f'{t}|{m}|{a}': c for (t, m), arms in cells.items() for a, c in arms.items()},
           'cells_by_task_mode': {k: v for k, v in cells.items()}, 'paired': paired, 'notes': notes,
           'runs': [{'dir': r['dir'], 'task': r['task'], 'mode': r['mode'], 'arm': r['arm'],
                     'complete': (r['summary'] or {}).get('complete'), 'partial_rows': r.get('partial_rows', 0)}
                    for r in runs], 'draws': draws, 'seed': seed, 'aligned': align}
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='', help='ROOT/<arm>/<task>_<mode>/ledger.jsonl')
    ap.add_argument('--run', action='append', default=[], help='a run directory (repeatable)')
    ap.add_argument('--pairs', default='', help='comma list A:B (default: wide vs ctrl, each vs untouched)')
    ap.add_argument('--partial', action='store_true', help='also read partial_ledger.jsonl (score_partial.py)')
    ap.add_argument('--align', action='store_true', help='cut every arm to the turns all arms of a cell reached')
    ap.add_argument('--draws', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=20261002)
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--out-md', required=True)
    args = ap.parse_args(argv)
    dirs = list(args.run) + (discover(args.root) if args.root else [])
    runs = [r for r in (load_run(d, args.partial) for d in dirs) if r is not None]
    if not runs:
        print('report: no run directories with a ledger.jsonl', file=sys.stderr)
        return 1
    pairs = [tuple(p.split(':')) for p in args.pairs.split(',')] if args.pairs else None
    doc = build(runs, pairs, args.draws, args.seed, args.align)
    js = dict(doc)
    js['cells_by_task_mode'] = {f'{t}|{m}': v for (t, m), v in doc['cells_by_task_mode'].items()}
    Path(args.out_json).write_text(json.dumps(js, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    Path(args.out_md).write_text(markdown(doc), encoding='utf-8')
    print(f'report: {len(runs)} runs, {len(doc["paired"])} paired comparisons -> {args.out_json}, {args.out_md}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
