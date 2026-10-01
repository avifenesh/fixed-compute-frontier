#!/usr/bin/env python3
"""Score the on-call sessions of a run that are still in flight, from its gens.jsonl, without a server.

A thinking-on session of 2,048 events runs about 280 turns, and an arm whose late turns think for minutes cannot finish
in a useful time. Its closed turns are already decided: the episode state is a pure function of the stored generations
(run_oncall's resume rests on that), so replaying them through a fresh ProbeEpisode gives the exact records the live
run will write for those turns. The turn the session is inside is not decided: its generations so far are dropped and
it is marked in flight, with every later turn. report.py --partial leaves in-flight turns out of the cells and pairs two
arms only on the turns both reached, so a partial session never scores as if the model had failed its tail.

For each run directory: the manifest (manifest_oncall.json) gives the selection, the mode, the cap and the limit
policy; every selected session that has generations and no final ledger row is replayed. A session whose replay ends
the episode (it finished between the last ledger write and now) is written as a final row, partial false. Output:
DIR/partial_ledger.jsonl (rewritten each time), one ledger-shaped row per session, plus partial: true, closed_turns,
in_flight_generations; its in-flight turns carry in_flight: true. The live run never reads this file.

  score_partial.py --run RUN_DIR [--run ...] [--items ITEMS.jsonl] [--tokenizer DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import probe_common as pc
import run_oncall as oc
from probe_common import read_jsonl

OUT_NAME = 'partial_ledger.jsonl'


def mode_of(man, tokenizer):
    """The run's request profile, rebuilt from its manifest; refuses one that renders differently."""
    thinking = bool(man['thinking'])
    history = man.get('history_reasoning') if thinking else 'drop'
    ctk = man.get('chat_template_kwargs') or {}
    gc = Path(tokenizer) / 'generation_config.json'
    mode = pc.make_mode(thinking, bool(man.get('greedy')), history, ctk.get('reasoning_effort', ''),
                        str(gc) if gc.exists() else None)
    if mode.ctk != ctk or mode.attach_reasoning != bool(man.get('attach_reasoning')) or mode.label != man['mode']:
        raise SystemExit(f"mode rebuilt as {mode.label} {mode.ctk} attach={mode.attach_reasoning}, the manifest says "
                         f"{man['mode']} {ctk} attach={man.get('attach_reasoning')}")
    return mode


def score_run(d, items_override=None, tokenizer=None, count=None, allow_lane_change=False):
    d = Path(d)
    man_path = d / f'manifest_{oc.NAME}.json'
    if not man_path.exists():
        raise SystemExit(f'{d}: no {man_path.name}')
    man = json.loads(man_path.read_text(encoding='utf-8'))
    lane_now = pc.sd_common.lane_hashes()
    if man.get('lane') and man['lane'] != lane_now and not allow_lane_change:
        raise SystemExit(f'{d}: the lane code differs from the run ({man["lane"]} vs {lane_now}); the replay would '
                         f'score with other code (--allow-lane-change to accept)')
    tokenizer = tokenizer or man['tokenizer']
    mode = mode_of(man, tokenizer)
    if count is None:
        count, _ = pc.load_counter(tokenizer)
    sel = man['selection']
    by_id = {it['id']: it for it in oc.load_items(items_override or man['items_files'])}
    missing = [i for i in sel['items'] if i not in by_id]
    if missing:
        raise SystemExit(f'{d}: {len(missing)} selected items are not in the items files (first {missing[0]})')
    final = {r['key'] for r in read_jsonl(d / 'ledger.jsonl') if r.get('failure') != 'http_400'} \
        if (d / 'ledger.jsonl').exists() else set()
    store = pc.load_gen_store(d / 'gens.jsonl')
    lane = oc.load_lane()
    rows, counts = [], {'final_in_ledger': 0, 'partial': 0, 'finished_from_gens': 0, 'not_started': 0,
                        'closed_turns': 0, 'in_flight_turns': 0}
    for iid in sel['items']:
        item = by_id[iid]
        for s in range(sel['samples']):
            key = oc.item_key(item, s)
            if key in final:
                counts['final_in_ledger'] += 1
                continue
            if not store.get(key):
                counts['not_started'] += 1
                continue
            ep, replayed = oc.replay_episode(item, s, man['max_tokens'], man['limit_policy'], mode, lane, count,
                                             store)
            finished = ep.done or ep.next_request() is None
            dropped = 0
            if not finished:
                dropped = len(ep.cur['gens'])
                ep.cur = {'gens': [], 'made': []}  # the open turn is undecided: in flight, not ended
                ep.gen_budget_hit = False
                ep.done = True
            closed = len(ep.turn_recs)
            ep.finalize()
            row = oc.episode_row(ep, item, s, man['arm'], mode, replayed, None)
            row.update(partial=not finished, closed_turns=closed, in_flight_generations=dropped)
            if not finished:
                row['failure'] = 'in_flight'
                for t in row['turns'][closed:]:
                    t['in_flight'] = True
                counts['partial'] += 1
                counts['in_flight_turns'] += len(row['turns']) - closed
            else:
                counts['finished_from_gens'] += 1
            counts['closed_turns'] += closed
            rows.append(row)
    tmp = d / (OUT_NAME + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    os.replace(tmp, d / OUT_NAME)
    return {'dir': str(d), 'arm': man['arm'], 'mode': man['mode'], **counts}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', action='append', required=True, help='an on-call run directory (repeatable)')
    ap.add_argument('--items', nargs='+', default=None, help="items files (default: the manifest's items_files)")
    ap.add_argument('--tokenizer', default='', help="tokenizer dir (default: the manifest's)")
    ap.add_argument('--allow-lane-change', action='store_true')
    args = ap.parse_args(argv)
    count = None
    for d in args.run:
        res = score_run(d, args.items, args.tokenizer or None, count, args.allow_lane_change)
        print(json.dumps(res))
    return 0


if __name__ == '__main__':
    sys.exit(main())
