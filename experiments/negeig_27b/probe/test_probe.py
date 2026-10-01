"""CPU tests for the generative probe (run_oncall.py, run_s1gen.py, report.py, run_all.sh). No GPU, no SGLang, no
network: a fake OpenAI server (fake_probe_server.py, an oracle with scripted defects) stands in for the model.

    cd experiments/negeig_27b/probe
    nice -n 10 taskset -c 8-15 env OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= ~/.venvs/negeig/bin/python -m unittest -v test_probe

The Qwen3.8 tokenizer (PROBE_TEST_TOKENIZER, default /data/ai-ml/hf-models/qwen3.8-27b-tokenizer) is used when it is
there: token counts are then checked exactly against it and the chat-template claims are rendered for real. Without
it those tests skip and a whitespace counter stands in for the rest.

Red arms (each must fail the way it should): a wrong call scores below 1; an omitted, an extra and a spurious call
too; a truncated turn is counted as truncated and scores 0; a step-limited turn is counted; reasoning tokens are
counted and reach the report; S1 history carries the model's own (wrong) answer, not the gold; a server without the
reasoning or tool parser fails the server check and run_all stops; a step-limited turn under --on-limit end leaves no
phantom reached turn after it; a context refusal (SGLang's messages verbatim) clamps the cap, then ends the session in
a final context_overflow row that a rerun never asks again and the report counts.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import fake_probe_server as F  # noqa: E402
import probe_common as pc  # noqa: E402
import report  # noqa: E402
import run_oncall  # noqa: E402
import run_s1gen  # noqa: E402
import score_partial  # noqa: E402

# sd_common.Client keeps one keep-alive connection per worker thread and never closes it at the end of a run (the
# process exits); the sockets of finished test runs are collected later and would print a warning each.
warnings.filterwarnings('ignore', category=ResourceWarning, message='unclosed <socket')

TOKENIZER = os.environ.get('PROBE_TEST_TOKENIZER', '/data/ai-ml/hf-models/qwen3.8-27b-tokenizer')
HAVE_TOK = (Path(TOKENIZER) / 'tokenizer.json').is_file()
LANE = pc.sd_common.LANE_DIR
HAVE_LANE = (LANE / 'agentic_env.py').is_file()
GEN_CFG = str(Path(TOKENIZER) / 'generation_config.json') if HAVE_TOK else None

_STATE = {}


def counter():
    if 'count' not in _STATE:
        if HAVE_TOK:
            _STATE['count'], _STATE['tok'] = pc.load_counter(TOKENIZER)
        else:
            _STATE['count'], _STATE['tok'] = (lambda t: len(t.split()) if t else 0), None
    return _STATE['count']


def tmpdir():
    d = Path(tempfile.mkdtemp(prefix='probe_t_'))
    _STATE.setdefault('dirs', []).append(d)
    return d


def setUpModule():
    # unittest's runner installs its own warning filters around the run; this one has to come after them
    warnings.filterwarnings('ignore', category=ResourceWarning, message='unclosed <socket')


def tearDownModule():
    for d in _STATE.get('dirs', []):
        shutil.rmtree(d, ignore_errors=True)


def mini_oncall_item(idx, n_events, seed):
    """An on-call session in data/oncall.py's contract (agentic_env item; turn meta type initial/events/action and
    after_events; item meta n_events), built here from ops.Sim so these tests do not depend on the generator's own
    state of work. Action turns: acknowledge every open incident (a set, possibly empty: then no call) or page the
    primary about an unresolved incident (one call). The agent's actions change the simulator state."""
    import random as _r
    import ops
    from s2 import GENERIC, call_key, fn
    from s2 import s as sdesc
    rng = _r.Random(seed)
    sim = ops.Sim(rng, 'en', 'eval')
    tools = [fn('acknowledge_incident', 'Acknowledge an open incident.', {'incident_id': sdesc('Incident ID.')}),
             fn('page_oncall', 'Page a person about an incident.',
                {'person': sdesc('Name of the person to page.'), 'incident_id': sdesc('Incident ID.')})]

    def turn(text, calls, meta):
        expect = {'calls': calls, 'parallel': False} if calls else {'no_call': True}
        return {'user': text, 'expect': expect, 'meta': meta,
                'results': {call_key(c['name'], c['arguments']): {'ok': True} for c in calls}}

    turns = [turn('Initial state:\n' + '\n'.join(sim.initial()), [], {'type': 'initial', 'after_events': 0})]
    done, k = 0, 0
    while done < n_events:
        n = min(n_events - done, rng.randint(1, 8))
        evs = [sim.event() for _ in range(n)]
        done += n
        k += 1
        meta = {'after_events': done, 'batch_events': n}
        block = 'Events:\n' + '\n'.join(evs)
        if k % 3:
            turns.append(turn(block, [], {'type': 'events', **meta}))
            continue
        unres = sorted(i for i, st in sim.status.items() if st != 'res')
        if k % 2 or not unres:
            calls = [{'name': 'acknowledge_incident', 'arguments': {'incident_id': i}}
                     for i in sorted(sim.status) if sim.status[i] == 'open']
            text, action = 'Acknowledge every incident whose status is open right now.', 'ack_open'
            for c in calls:
                sim.status[c['arguments']['incident_id']] = 'ack'
        else:
            inc = rng.choice(unres)
            calls = [{'name': 'page_oncall', 'arguments': {'person': sim.primary, 'incident_id': inc}}]
            text, action = f'Page whoever is primary on-call about {inc}.', 'page_role'
        turns.append(turn(block + '\n\n' + text, calls, {'type': 'action', 'action': action, 'n_calls': len(calls),
                                                          **meta}))
    return {'id': f'mini-oncall-L{n_events:04d}-{idx:03d}', 'env': 'agentic', 'kind': 'oncall_long',
            'task_type': f'oncall-L{n_events}', 'lang': 'en', 'domain': 'oncall',
            'system': GENERIC + '\n\nYou are the on-call operations agent.', 'tools': tools, 'turns': turns,
            'split': 'eval', 'meta': {'n_events': n_events, 'seed': seed, 'n_turns': len(turns)}}


def data_dir():
    """On-call items (two 64-event and one 256-event session, mini_oncall_item) and an S1 eval dir (data/common.py
    sessions of 32 and 64 events, every domain, Hebrew where the domain has it), built once and verified."""
    if 'data' in _STATE:
        return _STATE['data']
    import common
    import importlib
    d = tmpdir()
    items = [mini_oncall_item(k, L, 90_000_000 + 1_000_003 * li + k)
             for li, (L, k) in enumerate([(64, 0), (64, 1), (256, 0)])]
    assert sum(bool(t['expect'].get('calls')) for it in items for t in it['turns']) >= 4
    with open(d / 'oncall.eval.jsonl', 'w') as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + '\n')
    s1 = d / 's1'
    s1.mkdir()
    for di, name in enumerate(run_s1gen.DOMAINS):
        mod = importlib.import_module(name)
        langs = ['en', 'he'] if 'he' in mod.Sim.SYSTEM else ['en']
        with open(s1 / f'{name}.eval.jsonl', 'w') as f:
            for L in (32, 64):
                for i in range(3):
                    seed = 50_000_000 + 1_000_003 * di + L * 10 + i
                    lang = 'he' if len(langs) > 1 and i == 2 else 'en'
                    s = common.make_session(mod.Sim, name, seed, L, lang, 'eval')
                    common.verify(s, mod.replay)
                    row = json.loads(s.to_json())
                    row['messages'] = common.to_messages(s)
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
    _STATE['data'] = (d, items, s1)
    return _STATE['data']


def call_turns(items):
    return [(it['id'], t) for it in items for t, tt in enumerate(it['turns']) if tt['expect'].get('calls')]


def events_turns(items):
    return [(it['id'], t) for it in items for t, tt in enumerate(it['turns']) if tt['meta']['type'] == 'events']


def oncall_args(out, url, thinking='off', extra=()):
    ap = argparse.ArgumentParser()
    run_oncall.add_args(ap)
    d, _, _ = data_dir()
    return ap.parse_args(['--items', str(d / 'oncall.eval.jsonl'), '--thinking', thinking, '--out-dir', str(out),
                          '--base-url', url, '--arm', 'test', '--concurrency', '3', '--model', F.MODEL,
                          '--tokenizer', TOKENIZER, *(['--generation-config', GEN_CFG] if GEN_CFG else []),
                          *extra])


def s1_args(out, url, thinking='off', extra=()):
    ap = argparse.ArgumentParser()
    run_s1gen.add_args(ap)
    _, _, s1 = data_dir()
    return ap.parse_args(['--s1', str(s1), '--lengths', '32,64', '--per-cell', '2', '--thinking', thinking,
                          '--out-dir', str(out), '--base-url', url, '--arm', 'test', '--concurrency', '4',
                          '--model', F.MODEL, '--tokenizer', TOKENIZER,
                          *(['--generation-config', GEN_CFG] if GEN_CFG else []), *extra])


def fast_client(url):
    return pc.Client(url, model=F.MODEL, max_retries=1, backoff=0.01, timeout=30)


def ledger(out):
    return {r['key']: r for r in pc.read_jsonl(Path(out) / 'ledger.jsonl')}


@contextlib.contextmanager
def serving(**kw):
    srv = F.ProbeFakeServer(**kw)
    srv.start()
    try:
        yield srv
    finally:
        srv.stop()


def run_oc(srv, out, thinking='off', extra=(), client=None):
    return run_oncall.run(oncall_args(out, srv.url, thinking, extra), client=client or fast_client(srv.url),
                          count=counter())


def run_s1(srv, out, thinking='off', extra=(), client=None):
    return run_s1gen.run(s1_args(out, srv.url, thinking, extra), client=client or fast_client(srv.url),
                         count=counter())


def fake_kw(thinking=False, **kw):
    d, items, s1 = data_dir()
    sessions = run_s1gen.select_sessions(s1_args(tmpdir(), 'http://x'))
    base = dict(items=items, sessions=sessions, thinking=thinking, counter=counter(), generation_config=GEN_CFG)
    base.update(kw)
    return base


# ====================================================================== modes and the template
class ModeTests(unittest.TestCase):
    @unittest.skipUnless(HAVE_TOK, 'needs the Qwen3.8 generation_config.json')
    def test_profiles(self):
        off = pc.make_mode(False, generation_config=GEN_CFG)
        self.assertEqual(off.ctk, {'enable_thinking': False})
        self.assertEqual(off.sampling, pc.SAMPLING['vendor_nonthinking'])
        self.assertFalse(off.attach_reasoning)
        on = pc.make_mode(True, generation_config=GEN_CFG)
        self.assertEqual(on.ctk, {'enable_thinking': True, 'preserve_thinking': False})
        self.assertEqual((on.sampling['temperature'], on.sampling['top_p'], on.sampling['top_k'],
                          on.sampling['presence_penalty']), (1.0, 0.95, 20, 0.0))  # the vendor thinking arm
        self.assertTrue(on.attach_reasoning)
        self.assertEqual(pc.make_mode(True, history='keep', generation_config=GEN_CFG).ctk['preserve_thinking'], True)
        strip = pc.make_mode(True, history='strip', generation_config=GEN_CFG)
        self.assertEqual(strip.ctk, {'enable_thinking': True})
        self.assertFalse(strip.attach_reasoning)
        g = pc.make_mode(True, greedy=True, reasoning_effort='low', generation_config=GEN_CFG)
        self.assertEqual(g.sampling['temperature'], 0.0)
        self.assertEqual(g.ctk['reasoning_effort'], 'low')
        self.assertEqual(g.label, 'think_on_greedy')

    def test_meaningless_combinations_raise(self):
        with self.assertRaises(ValueError):
            pc.make_mode(False, reasoning_effort='low')
        with self.assertRaises(ValueError):
            pc.make_mode(True, reasoning_effort='high')  # the template accepts xhigh, medium, low only
        with self.assertRaises(ValueError):
            pc.make_mode(True, history='sometimes')

    @unittest.skipUnless(HAVE_TOK, 'needs the Qwen3.8 tokenizer')
    def test_template_renders_history_reasoning_as_documented(self):
        counter()
        tok = _STATE['tok']
        tools = [{'type': 'function', 'function': {'name': 'f', 'description': 'd', 'parameters': {
            'type': 'object', 'properties': {'x': {'type': 'string'}}, 'required': ['x']}}}]
        msgs = [{'role': 'system', 'content': 'SYS'}, {'role': 'user', 'content': 'U1'},
                {'role': 'assistant', 'content': 'A1', 'reasoning_content': 'REASON_ONE'},
                {'role': 'user', 'content': 'U2'},
                {'role': 'assistant', 'content': '', 'reasoning_content': 'REASON_TWO', 'tool_calls': [
                    {'id': 'c1', 'type': 'function', 'function': {'name': 'f', 'arguments': {'x': '1'}}}]},
                {'role': 'tool', 'tool_call_id': 'c1', 'content': '{"ok": true}'}]

        def render(mode, m=msgs):
            return tok.apply_chat_template(m, tools=tools, tokenize=False, add_generation_prompt=True, **mode.ctk)

        drop = render(pc.make_mode(True, generation_config=GEN_CFG))
        self.assertNotIn('REASON_ONE', drop)   # earlier turn: dropped, think block and all
        self.assertIn('<|im_start|>assistant\nA1<|im_end|>', drop)
        self.assertIn('REASON_TWO', drop)      # the current turn's tool loop keeps it
        self.assertIn('Reasoning effort is set to xhigh', drop)
        keep = render(pc.make_mode(True, history='keep', generation_config=GEN_CFG))
        self.assertIn('REASON_ONE', keep)
        bare = [{k: v for k, v in m.items() if k != 'reasoning_content'} for m in msgs]
        off = render(pc.make_mode(False, generation_config=GEN_CFG), bare)
        self.assertIn('<|im_start|>assistant\n<think>\n\n</think>\n\nA1<|im_end|>', off)  # the training layout
        self.assertTrue(off.endswith('<think>\n\n</think>\n\n'))
        self.assertNotIn('Reasoning effort', off)


# ====================================================================== S1 scoring and token accounting
class NormalizeTests(unittest.TestCase):
    def test_presentation_is_normalized_and_content_is_not(self):
        n = run_s1gen.normalize
        self.assertEqual(n(' Layla. ', 'custody', 'en'), n('Layla', 'custody', 'en'))
        self.assertEqual(n('**the Storage Room**', 'custody', 'en'), 'the storage room')
        self.assertEqual(n('`/srv/a.txt`', 'fsys', 'en'), '/srv/a.txt')
        self.assertNotEqual(n('/SRV/a.txt', 'fsys', 'en'), n('/srv/a.txt', 'fsys', 'en'))  # paths keep case
        self.assertNotEqual(n('FALSE', 'codetrace', 'en'), n('False', 'codetrace', 'en'))  # Python repr keeps case
        self.assertNotEqual(n('pine', 'codetrace', 'en'), n("'pine'", 'codetrace', 'en'))  # quotes are the repr
        self.assertEqual(n('"open"', 'ops', 'en'), 'open')
        self.assertNotEqual(n('INC-2, INC-1', 'ops', 'en'), n('INC-1, INC-2', 'ops', 'en'))  # order is the format
        self.assertEqual(n('פתוח.', 'ops', 'he'), 'פתוח')

    def test_truncated_answer_scores_zero_even_when_it_matches(self):
        self.assertEqual(run_s1gen.score_answer('open', 'open', 'ops', 'en', True), (False, False))
        self.assertEqual(run_s1gen.score_answer('Open.', 'open', 'ops', 'en', False), (False, True))


class GenRecordTests(unittest.TestCase):
    def test_fields(self):
        count = counter()
        g = pc.Generation(content='INC-1', reasoning='first I list the open incidents', tool_calls=[],
                          finish_reason='stop', usage={'prompt_tokens': 50, 'completion_tokens': 40,
                                                       'reasoning_tokens': 31})
        r = pc.gen_record(g, 0.5, count, 1024)
        self.assertEqual(r['reasoning_tokens'], count('first I list the open incidents'))
        self.assertEqual(r['server_reasoning_tokens'], 31)
        self.assertEqual(r['content_tokens'], 40 - r['reasoning_tokens'])
        self.assertEqual(r['visible_tokens'], count('INC-1'))
        self.assertFalse(r['truncated'])

    def test_openai_form_missing_usage_and_truncation(self):
        self.assertEqual(pc.server_reasoning({'completion_tokens_details': {'reasoning_tokens': 7}}), 7)
        self.assertIsNone(pc.server_reasoning({'completion_tokens': 3}))
        g = pc.Generation(content='', reasoning='x y', tool_calls=[], finish_reason='length', usage={})
        r = pc.gen_record(g, 0.1, counter(), 16)
        self.assertTrue(r['truncated'])
        self.assertIsNone(r['completion_tokens'])
        self.assertIsNone(r['content_tokens'])
        self.assertGreater(r['reasoning_tokens'], 0)  # counted from the text even when the server reports nothing

    def test_context_refusal_parses_sglang_messages_only(self):
        """The two refusals of SGLang 0.5.20 TokenizerManager._validate_one_request, as its ErrorResponse body."""
        total = json.dumps({'object': 'error', 'type': 'BadRequest', 'param': None, 'code': 400, 'message': (
            "Requested token count exceeds the model's maximum context length of 131072 tokens. You requested a total "
            "of 131500 tokens: 115116 tokens from the input messages and 16384 tokens for the completion. Please "
            "reduce the number of tokens in the input messages or the completion to fit within the limit.")}).encode()
        inp = json.dumps({'object': 'error', 'message': "The input (131080 tokens) is longer than the model's context "
                          "length (131072 tokens).", 'type': 'BadRequest', 'param': None, 'code': 400}).encode()
        self.assertEqual(pc.context_refusal(pc.ApiError(400, total[:500])),
                         {'kind': 'total', 'input_tokens': 115116, 'context': 131072})
        self.assertEqual(pc.context_refusal(pc.ApiError(400, inp[:500])),
                         {'kind': 'input', 'input_tokens': 131080, 'context': 131072})
        self.assertIsNone(pc.context_refusal(pc.ApiError(400, b'{"message": "Input is too long"}')))  # other 400
        self.assertIsNone(pc.context_refusal(pc.ApiError(503, total, True)))
        self.assertIsNone(pc.context_refusal(ValueError('x')))

    def test_gen_store_keeps_first_copy_and_stops_at_a_gap(self):
        d = tmpdir()
        rows = [{'key': 'a', 'idx': 0, 'content': 'x'}, {'key': 'a', 'idx': 1, 'content': 'y'},
                {'key': 'a', 'idx': 1, 'content': 'dup'}, {'key': 'a', 'idx': 3, 'content': 'gap'},
                {'key': 'b', 'idx': 1, 'content': 'no zero'}]
        with open(d / 'g.jsonl', 'w') as f:
            for r in rows:
                f.write(json.dumps(r) + '\n')
        st = pc.load_gen_store(d / 'g.jsonl')
        self.assertEqual([r['content'] for r in st['a']], ['x', 'y'])
        self.assertEqual(st['b'], [])


# ====================================================================== on-call
@unittest.skipUnless(HAVE_LANE, 'needs the Hebrew lane (agentic_env.py)')
class OncallTests(unittest.TestCase):
    def test_clean_oracle_scores_every_turn_one(self):
        out = tmpdir()
        with serving(**fake_kw()) as srv:
            s = run_oc(srv, out)
            self.assertEqual(srv.stats()['n_violations'], 0, srv.stats()['violations'])
            self.assertEqual(srv.stats()['internal_errors'], [])
        self.assertTrue(s['complete'])
        rows = ledger(out)
        self.assertEqual(len(rows), 3)
        for r in rows.values():
            self.assertEqual(r['reward'], 1.0, r['item_id'])
            self.assertEqual({t['status'] for t in r['turns']}, {'scored'})
            self.assertEqual(r['unreached'], 0)
            self.assertEqual(r['n_turns'], len(r['turns']))
            calls = [t for t in r['turns'] if t['expect'] == 'call']
            self.assertTrue(all(t['n_made'] == t['n_expected'] for t in calls))

    def test_each_defect_scores_below_one_and_the_session_goes_on(self):
        _, items, _ = data_dir()
        ct = call_turns(items)
        ev = events_turns(items)
        script = {ct[0]: 'wrong_arg', ct[1]: 'omit', ct[2]: 'extra', ev[3]: 'spurious', ev[5]: 'truncated',
                  ct[3]: 'loop'}
        out = tmpdir()
        with serving(**fake_kw(script=script)) as srv:
            run_oc(srv, out)
            self.assertEqual(srv.stats()['n_violations'], 0, srv.stats()['violations'])
            bodies = list(srv.bodies)
        rows = ledger(out)
        rec = {(r['item_id'], t['t']): t for r in rows.values() for t in r['turns']}
        self.assertLess(rec[ct[0]]['score'], 1.0)
        self.assertLess(rec[ct[1]]['score'], 1.0)
        self.assertLess(rec[ct[2]]['score'], 1.0)
        self.assertGreater(rec[ct[2]]['score'], 0.0)                 # extra: matched / max(expected, made)
        self.assertEqual(rec[ev[3]]['score'], 0.0)                   # a call on a turn that needs none
        self.assertEqual((rec[ev[5]]['status'], rec[ev[5]]['score'], rec[ev[5]]['truncated_gens']),
                         ('truncated', 0.0, 1))
        self.assertEqual(rec[ct[3]]['status'], 'turn_budget')
        self.assertLess(rec[ct[3]]['score'], 1.0)
        self.assertEqual(rec[ct[3]]['n_gens'], 4)                     # agentic_env.MAX_STEPS_PER_TURN
        others = [t for k, t in rec.items() if k not in script]
        self.assertTrue(others and all(t['score'] == 1.0 and t['reached'] for t in others))
        limits = {}
        for r in rows.values():
            for k, v in r['limits'].items():
                limits[k] = limits.get(k, 0) + v
        self.assertEqual(limits, {'truncated': 1, 'turn_budget': 1})
        # the wrong call is in history with the canned error it got, so the model sees its own mistake
        wrong_item = ct[0][0]
        later = [b for b in bodies if b['messages'][1]['content'] == items[[i['id'] for i in items].index(
            wrong_item)]['turns'][0]['user'] and len([m for m in b['messages'] if m['role'] == 'user']) > ct[0][1] + 1]
        self.assertTrue(later)
        errs = [m['content'] for m in later[-1]['messages'] if m['role'] == 'tool']
        self.assertIn(json.dumps({'error': 'No record matches these arguments.'}), errs)

    def test_end_policy_is_agentic_env_exactly(self):
        _, items, _ = data_dir()
        ev = events_turns(items)
        out = tmpdir()
        with serving(**fake_kw(script={ev[1]: 'truncated'})) as srv:
            run_oc(srv, out, extra=['--on-limit', 'end'])
        r = next(r for r in ledger(out).values() if r['item_id'] == ev[1][0])
        self.assertEqual(r['turns'][ev[1][1]]['status'], 'truncated')
        self.assertEqual(r['failure'], 'truncated')
        self.assertEqual(r['unreached'], len(r['turns']) - ev[1][1] - 1)
        self.assertTrue(all(not t['reached'] and t['score'] == 0.0 for t in r['turns'][ev[1][1] + 1:]))

    def test_end_policy_step_limit_leaves_the_next_turn_unreached(self):
        """Red arm for finalize(): under --on-limit end a turn closed by the step limit must not be followed by a
        phantom 'gen_budget' turn that counts as reached with no generation."""
        _, items, _ = data_dir()
        ct = call_turns(items)
        out = tmpdir()
        with serving(**fake_kw(script={ct[0]: 'loop'})) as srv:
            run_oc(srv, out, extra=['--on-limit', 'end'])
        r = next(r for r in ledger(out).values() if r['item_id'] == ct[0][0])
        t = ct[0][1]
        self.assertEqual((r['turns'][t]['status'], r['turns'][t]['n_gens']), ('turn_budget', 4))
        self.assertTrue(all(x['status'] == 'unreached' and not x['reached'] for x in r['turns'][t + 1:]))
        self.assertEqual(r['limits'], {'turn_budget': 1})
        self.assertEqual(r['unreached'], len(r['turns']) - t - 1)

    def test_context_overflow_clamps_then_ends_final_and_is_reported(self):
        """A session that outgrows the context: the cap is clamped while the prompt fits, then the row ends with
        status context_overflow, is final (a rerun asks nothing), and counts in the report with its lost turns at 0."""
        _, items, _ = data_dir()
        clean = tmpdir()
        with serving(**fake_kw()) as srv:
            run_oc(srv, clean)
        peak = {r['item_id']: max(t.get('prompt_tokens_last') or 0 for t in r['turns'])
                for r in ledger(clean).values()}
        long_id = items[2]['id']
        short_peak = max(v for k, v in peak.items() if k != long_id)
        ctx = short_peak + 1024 + 16          # the short sessions always fit with the full think-off cap
        self.assertGreater(peak[long_id], ctx + 200, peak)  # the long one outgrows it
        out = tmpdir()
        with serving(**fake_kw(ctx_limit=ctx)) as srv:
            s = run_oc(srv, out)
            self.assertEqual(srv.stats()['n_violations'], 0, srv.stats()['violations'])
            self.assertGreaterEqual(srv.ctx_refusals['total'], 2)  # at least one clamp and the final refusal
            asked = dict(srv.by_key)
        self.assertTrue(s['complete'], s)
        self.assertEqual((s['context_overflow_rows'], s['refused']), (1, 0))
        rows = ledger(out)
        r = rows[f'{long_id}#0']
        self.assertEqual(r['failure'], 'context_overflow')
        k = next(i for i, t in enumerate(r['turns']) if t['status'] == 'context_overflow')
        self.assertGreater(k, 0)
        self.assertTrue(all(t['status'] == 'unreached' and t['score'] == 0.0 for t in r['turns'][k + 1:]))
        self.assertEqual(r['limits'].get('context_overflow'), 1)
        clamped = [g for g in pc.read_jsonl(out / 'gens.jsonl')
                   if g['key'] == f'{long_id}#0' and g['max_tokens'] < 1024]
        self.assertTrue(clamped)
        self.assertTrue(all(g['max_tokens'] >= pc.MIN_CLAMPED_TOKENS for g in clamped))
        scored = [t for t in r['turns'][:k] if t['reached']]
        self.assertTrue(scored and all(t['score'] == 1.0 for t in scored))  # the clamped turns were still answered
        for key, row in rows.items():
            if key != f'{long_id}#0':
                self.assertEqual((row['failure'], row['reward']), (None, 1.0))
        with serving(**fake_kw(ctx_limit=ctx)) as srv:  # final: a rerun asks the server nothing
            s2 = run_oc(srv, out)
            self.assertEqual(dict(srv.by_key), {})
            self.assertEqual(sum(srv.ctx_refusals.values()), 0)
        self.assertTrue(s2['complete'])
        self.assertEqual(s2['resumed_done'], 3)
        self.assertEqual(asked[long_id], r['generations'])
        run = report.load_run(out)
        run.update(arm='test', mode='think_off', task='oncall')
        cm = report.cell_metrics(run)
        self.assertEqual((cm['items'], cm['context_overflow_items'], cm['refused_items']), (3, 1, 0))
        self.assertEqual(cm['context_overflow_turns_lost'], r['unreached'])
        lost_action = [t for t in r['turns'][k:] if t.get('type') == 'action']
        if lost_action:
            self.assertLess(cm['acc'], 1.0)  # the lost action turns count 0, the session is not dropped

    def test_malformed_arguments_end_the_episode(self):
        _, items, _ = data_dir()
        ct = call_turns(items)
        out = tmpdir()
        with serving(**fake_kw(script={ct[0]: 'bad_json'})) as srv:
            run_oc(srv, out)
        r = next(r for r in ledger(out).values() if r['item_id'] == ct[0][0])
        self.assertEqual(r['turns'][ct[0][1]]['status'], 'bad_arguments')
        self.assertEqual(r['failure'], 'bad_arguments')
        self.assertGreater(r['unreached'], 0)

    @unittest.skipUnless(HAVE_TOK, 'exact token counts need the Qwen3.8 tokenizer')
    def test_thinking_tokens_are_counted_exactly_and_history_carries_reasoning(self):
        _, items, _ = data_dir()
        ev = events_turns(items)
        out = tmpdir()
        count = counter()
        with serving(**fake_kw(thinking=True, script={ev[2]: 'truncated'})) as srv:
            run_oc(srv, out, thinking='on', extra=['--max-tokens', '300'])
            self.assertEqual(srv.stats()['n_violations'], 0, srv.stats()['violations'])  # reasoning sent back
            self.assertTrue(all(b['chat_template_kwargs'] == {'enable_thinking': True, 'preserve_thinking': False}
                                for b in srv.bodies))
        gens = pc.read_jsonl(out / 'gens.jsonl')
        rows = ledger(out)
        total_r = 0
        for g in gens:
            r = count(g['reasoning'])
            total_r += r
            self.assertGreater(r, 0)
            if g['finish_reason'] != 'length':
                self.assertEqual(g['usage']['reasoning_tokens'], r + 1)  # the fake's </think>
        self.assertEqual(sum(r['totals']['reasoning_tokens'] for r in rows.values()), total_r)
        tr = rows[f'{ev[2][0]}#0']['turns'][ev[2][1]]
        self.assertEqual(tr['status'], 'truncated')
        self.assertEqual(tr['completion_tokens'], 300)
        self.assertGreaterEqual(tr['reasoning_tokens'], 290)  # the runaway think is counted, not dropped
        for r in rows.values():
            for t in r['turns']:
                if t['reached'] and t['status'] == 'scored':
                    self.assertEqual(t['content_tokens'], t['completion_tokens'] - t['reasoning_tokens'])

    def test_strip_mode_sends_no_reasoning_back(self):
        out = tmpdir()
        with serving(**fake_kw(thinking=True, history='strip')) as srv:
            run_oc(srv, out, thinking='on', extra=['--history-reasoning', 'strip', '--max-tokens', '300'])
            self.assertEqual(srv.stats()['n_violations'], 0, srv.stats()['violations'])
            self.assertTrue(all('reasoning_content' not in m for b in srv.bodies for m in b['messages']))
            self.assertTrue(all(b['chat_template_kwargs'] == {'enable_thinking': True} for b in srv.bodies))

    def test_the_fake_catches_a_runner_that_sends_reasoning_in_strip_mode(self):
        """Red arm for the history checks: a request built the wrong way must register a violation."""
        _, items, _ = data_dir()
        with serving(**fake_kw(thinking=True, history='strip')) as srv:
            it = items[0]
            mode = pc.make_mode(True, history='strip', generation_config=GEN_CFG)
            msgs = [{'role': 'system', 'content': it['system']}, {'role': 'user', 'content': it['turns'][0]['user']},
                    {'role': 'assistant', 'content': 'Noted.', 'reasoning_content': 'leaked'},
                    {'role': 'user', 'content': it['turns'][1]['user']}]
            fast_client(srv.url).chat(msgs, tools=it['tools'], max_tokens=64, sampling=mode.sampling, seed=1,
                                      chat_template_kwargs=mode.ctk)
            self.assertTrue(any('carries reasoning_content' in v for v in srv.stats()['violations']))

    def test_resume_replays_generations_and_never_asks_twice(self):
        _, items, _ = data_dir()
        ct = call_turns(items)
        script = {ct[0]: 'wrong_arg', ct[2]: 'omit'}
        clean = tmpdir()
        with serving(**fake_kw(script=script)) as srv:
            run_oc(srv, clean)
        out = tmpdir()
        with serving(**fake_kw(script=script)) as srv:
            srv.kill_after(25)
            s1 = run_oc(srv, out, extra=['--abort-after', '2'])
            self.assertFalse(s1['complete'])
            first = dict(srv.by_key)
            partial = pc.load_gen_store(out / 'gens.jsonl')
            self.assertTrue(any(partial.values()))
            srv.revive()
            s2 = run_oc(srv, out)
            self.assertTrue(s2['complete'], s2)
            self.assertEqual(srv.stats()['n_violations'], 0, srv.stats()['violations'])
            asked = dict(srv.by_key)
        rows, ref = ledger(out), ledger(clean)
        self.assertGreater(sum(r['replayed_generations'] for r in rows.values()), 0)
        for k, r in rows.items():
            self.assertEqual(asked[r['item_id']], r['generations'], k)  # every generation was asked exactly once
            self.assertEqual([t['score'] for t in r['turns']], [t['score'] for t in ref[k]['turns']])
        self.assertLess(sum(first.values()), sum(asked.values()))

    def test_refused_items_are_recorded_not_final_and_rerun(self):
        _, items, _ = data_dir()
        out = tmpdir()
        with serving(**fake_kw(http400_keys={items[0]['id']})) as srv:
            s = run_oc(srv, out)
        self.assertFalse(s['complete'])
        self.assertEqual(ledger(out)[f"{items[0]['id']}#0"]['failure'], 'http_400')
        self.assertEqual(ledger(out)[f"{items[0]['id']}#0"]['turns'][0]['status'], 'http_400')
        with serving(**fake_kw()) as srv:
            s = run_oc(srv, out)
        self.assertTrue(s['complete'])
        self.assertIsNone(ledger(out)[f"{items[0]['id']}#0"]['failure'])

    def test_a_changed_setting_cannot_append_to_the_same_ledger(self):
        out = tmpdir()
        with serving(**fake_kw()) as srv:
            run_oc(srv, out, extra=['--limit', '1'])
        with serving(**fake_kw(thinking=True)) as srv:
            with self.assertRaises(SystemExit):
                run_oc(srv, out, thinking='on', extra=['--limit', '1'])

    def test_items_from_the_real_generator_run_clean(self):
        """data/oncall.py's own sessions through the runner (skipped while the generator does not build or verify)."""
        try:
            import oncall
            items = [oncall.make_item('eval', 64, k, 90_000_000 + k) for k in range(2)]
            for it in items:
                oncall.verify(it)
        except Exception as error:  # noqa: BLE001, the generator is another lane's work in progress
            self.skipTest(f'data/oncall.py does not build and verify right now: {error!r}')
        d = tmpdir()
        with open(d / 'real.jsonl', 'w') as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + '\n')
        out = tmpdir()
        with serving(**fake_kw(items=items)) as srv:
            a = oncall_args(out, srv.url)
            a.items = [str(d / 'real.jsonl')]
            s = run_oncall.run(a, client=fast_client(srv.url), count=counter())
            self.assertEqual(srv.stats()['n_violations'], 0, srv.stats()['violations'])
        self.assertTrue(s['complete'])
        rows = ledger(out)
        self.assertTrue(all(r['reward'] == 1.0 for r in rows.values()))
        self.assertTrue(all(t['type'] in ('initial', 'events', 'action') for r in rows.values() for t in r['turns']))

    @unittest.skipUnless(HAVE_TOK, 'the plan renders through the Qwen3.8 template')
    def test_plan_measures_the_gold_context(self):
        out = tmpdir()
        p = run_oncall.run(oncall_args(out, 'http://127.0.0.1:9', extra=['--plan']), count=counter())['plan']
        self.assertEqual(p['items'], 3)
        self.assertGreater(p['gold_context_max'], 1000)
        self.assertEqual(p['reply_allowance'], 64 * 56)  # the longest session has 56 turns
        self.assertEqual(p['context_needed'], p['gold_context_max'] + 1024 + p['reply_allowance'])


# ====================================================================== S1
class S1Tests(unittest.TestCase):
    def test_clean_and_defects(self):
        sessions = run_s1gen.select_sessions(s1_args(tmpdir(), 'http://x'))
        self.assertEqual(len(sessions), 6 * 2 * 2)
        by = {run_s1gen.session_key(s): s for s in sessions}
        cust = next(k for k, s in by.items() if s['domain'] == 'custody' and s['lang'] == 'en')
        # a codetrace answer with letters (True, 'pine'), where upper case changes the value
        code, tc = next((k, t) for k, s in by.items() if k.startswith('codetrace')
                        for t in range(1, len(s['turns'])) if s['turns'][t]['answer'].upper() != s['turns'][t]['answer'])
        tt, tv = [t for t in range(1, len(by[code]['turns'])) if t != tc][:2]
        script = {(cust, 1): 'wrong', (cust, 2): 'format', (cust, 3): 'case', (code, tc): 'case',
                  (code, tt): 'truncated', (code, tv): 'verbose'}
        out = tmpdir()
        with serving(**fake_kw(script=script)) as srv:
            s = run_s1(srv, out)
            self.assertEqual(srv.stats()['n_violations'], 0, srv.stats()['violations'])
            bodies = list(srv.bodies)
        self.assertTrue(s['complete'])
        rows = ledger(out)
        c, k = rows[cust]['turns'], rows[code]['turns']
        self.assertEqual((c[1]['ok_strict'], c[1]['ok_norm']), (False, False))
        self.assertEqual((c[2]['ok_strict'], c[2]['ok_norm']), (False, True))   # trailing period
        self.assertEqual((c[3]['ok_strict'], c[3]['ok_norm']), (False, True))   # case carries nothing in custody
        self.assertEqual(k[tc]['ok_norm'], False)                                # but it does in codetrace
        self.assertEqual((k[tt]['status'], k[tt]['score']), ('truncated', 0.0))
        self.assertEqual(k[tv]['ok_norm'], False)
        for key, r in rows.items():
            if key not in (cust, code):
                self.assertEqual(r['acc'], 1.0, key)
        # history holds the model's own wrong answer, not the gold
        gold1 = by[cust]['turns'][1]['answer']
        later = [b for b in bodies if b['messages'][0]['content'] == by[cust]['system']
                 and len([m for m in b['messages'] if m['role'] == 'user']) == 3
                 and b['messages'][1]['content'] == F.pc_user_text(by[cust], 0)]
        self.assertTrue(later)
        hist = [m['content'] for m in later[0]['messages'] if m['role'] == 'assistant']
        self.assertEqual(hist[1], gold1 + 'Z')

    def test_the_fake_catches_gold_in_history(self):
        sessions = run_s1gen.select_sessions(s1_args(tmpdir(), 'http://x'))
        s = sessions[0]
        key = run_s1gen.session_key(s)
        with serving(**fake_kw(script={(key, 0): 'wrong'})) as srv:
            mode = pc.make_mode(False, generation_config=GEN_CFG)
            cl = fast_client(srv.url)
            m = [{'role': 'system', 'content': s['system']}, {'role': 'user', 'content': F.pc_user_text(s, 0)}]
            g = cl.chat(m, max_tokens=64, sampling=mode.sampling, seed=1, chat_template_kwargs=mode.ctk)
            self.assertTrue(g.content.endswith('Z'))
            m += [{'role': 'assistant', 'content': s['turns'][0]['answer']},  # gold, not what the model said
                  {'role': 'user', 'content': F.pc_user_text(s, 1)}]
            cl.chat(m, max_tokens=64, sampling=mode.sampling, seed=2, chat_template_kwargs=mode.ctk)
            self.assertTrue(any('not the answer this server gave' in v for v in srv.stats()['violations']))

    @unittest.skipUnless(HAVE_TOK, 'exact token counts need the Qwen3.8 tokenizer')
    def test_thinking_mode_tokens(self):
        out = tmpdir()
        with serving(**fake_kw(thinking=True)) as srv:
            run_s1(srv, out, thinking='on', extra=['--max-tokens', '256'])
            self.assertEqual(srv.stats()['n_violations'], 0, srv.stats()['violations'])
        rows = ledger(out)
        self.assertTrue(all(r['acc'] == 1.0 for r in rows.values()))
        self.assertTrue(all(t['reasoning_tokens'] > 0 and t['server_reasoning_tokens'] == t['reasoning_tokens'] + 1
                            for r in rows.values() for t in r['turns']))

    def test_a_prompt_that_does_not_fit_is_a_final_overflow(self):
        """SGLang's 'input longer than the context' refusal on the first question: final row, every answer 0, the
        run complete, no rerun; a later run with the same ledger asks nothing."""
        out = tmpdir()
        with serving(**fake_kw(ctx_limit=20)) as srv:
            s = run_s1(srv, out, extra=['--limit', '3'])
            self.assertEqual(srv.ctx_refusals['input'], 3)
        self.assertTrue(s['complete'], s)
        self.assertEqual(s['context_overflow_rows'], 3)
        for r in ledger(out).values():
            self.assertEqual((r['failure'], r['acc'], r['turns'][0]['status']), ('context_overflow', 0.0,
                                                                                 'context_overflow'))
            self.assertTrue(all(t['status'] == 'unreached' for t in r['turns'][1:]))
        with serving(**fake_kw(ctx_limit=20)) as srv:
            s = run_s1(srv, out, extra=['--limit', '3'])
            self.assertEqual(sum(srv.ctx_refusals.values()), 0)
        self.assertEqual(s['resumed_done'], 3)

    def test_resume_mid_session(self):
        out = tmpdir()
        with serving(**fake_kw()) as srv:
            srv.kill_after(30)
            s = run_s1(srv, out, extra=['--abort-after', '2'])
            self.assertFalse(s['complete'])
            srv.revive()
            s = run_s1(srv, out)
            self.assertTrue(s['complete'])
            asked = dict(srv.by_key)
            self.assertEqual(srv.stats()['n_violations'], 0, srv.stats()['violations'])
        rows = ledger(out)
        self.assertGreater(sum(r['replayed_generations'] for r in rows.values()), 0)
        for k, r in rows.items():
            self.assertEqual(asked[k], r['generations'])
            self.assertEqual(r['acc'], 1.0)


# ====================================================================== data-parallel pinning
@unittest.skipUnless(HAVE_LANE, 'needs the Hebrew lane (agentic_env.py)')
class DpPinningTests(unittest.TestCase):
    def test_every_session_stays_on_one_rank_and_the_ranks_are_shared_out(self):
        for task in ('oncall', 's1gen'):
            out = tmpdir()
            with serving(**fake_kw(dp=4)) as srv:
                runner = run_oc if task == 'oncall' else run_s1
                s = runner(srv, out, extra=['--dp-ranks', '4'])
                self.assertTrue(s['complete'], (task, s))
                ranks = dict(srv.ranks)
            self.assertTrue(all(len(v) == 1 for v in ranks.values()), (task, ranks))
            used = {next(iter(v)) for v in ranks.values()}
            self.assertGreaterEqual(len(used), 3 if task == 'oncall' else 4, (task, used))
            self.assertTrue(used <= {0, 1, 2, 3})
            for r in ledger(out).values():
                key = r['item_id'] if task == 'oncall' else r['key']
                self.assertEqual({r['dp_rank']}, ranks[key])

    def test_no_pinning_by_default(self):
        out = tmpdir()
        with serving(**fake_kw()) as srv:
            run_s1(srv, out, extra=['--limit', '2'])
            self.assertTrue(all('routed_dp_rank' not in b for b in srv.bodies))

    def test_more_ranks_than_the_server_has_are_refused_loudly(self):
        out = tmpdir()
        with serving(**fake_kw(dp=2)) as srv:
            s = run_s1(srv, out, extra=['--dp-ranks', '4', '--abort-after', '3'])
        self.assertFalse(s['complete'])
        self.assertGreater(s['refused'], 0)
        with serving(auto=True, generation_config=GEN_CFG, counter=counter(), dp=2) as srv:
            ok, checks = pc.check_server(fast_client(srv.url), counter(), dp_ranks=4)
        self.assertFalse(ok)
        self.assertFalse({c['check']: c['ok'] for c in checks}['routed_dp_rank_last'])

    def test_extra_body_cannot_overwrite_a_request_field(self):
        with serving(**fake_kw()) as srv:
            with self.assertRaises(AssertionError):
                fast_client(srv.url).chat([{'role': 'user', 'content': 'x'}], max_tokens=8, sampling={},
                                          extra_body={'max_tokens': 99})


# ====================================================================== report
@unittest.skipUnless(HAVE_LANE, 'needs the Hebrew lane (agentic_env.py)')
class ReportTests(unittest.TestCase):
    def test_three_arms_both_tasks_and_paired_signs(self):
        _, items, _ = data_dir()
        ct, ev = call_turns(items), events_turns(items)
        sessions = run_s1gen.select_sessions(s1_args(tmpdir(), 'http://x'))
        skey = [run_s1gen.session_key(s) for s in sessions]
        root = tmpdir()
        arms = {  # untouched errs; ctrl is clean; wide is clean and thinks half as much
            'untouched': dict(script={ct[0]: 'wrong_arg', ct[1]: 'omit', ev[2]: 'truncated', (skey[0], 1): 'wrong',
                                      (skey[3], 2): 'wrong'}),
            'ctrl_s0': dict(script={}), 'wide_s0': dict(script={}, think_scale=0.5)}
        for arm, kw in arms.items():
            for thinking in (False, True):
                lab = 'think_on' if thinking else 'think_off'
                with serving(**fake_kw(thinking=thinking, **kw)) as srv:
                    a = oncall_args(root / arm / f'oncall_{lab}', srv.url, 'on' if thinking else 'off',
                                    ['--max-tokens', '200'])
                    a.arm = arm
                    run_oncall.run(a, client=fast_client(srv.url), count=counter())
                    b = s1_args(root / arm / f's1gen_{lab}', srv.url, 'on' if thinking else 'off',
                                ['--max-tokens', '200'])
                    b.arm = arm
                    run_s1gen.run(b, client=fast_client(srv.url), count=counter())
        oj, om = root / 'r.json', root / 'r.md'
        self.assertEqual(report.main(['--root', str(root), '--out-json', str(oj), '--out-md', str(om),
                                      '--draws', '2000']), 0)
        doc = json.loads(oj.read_text())
        cells = doc['cells']
        on_w, on_c = cells['oncall|think_on|wide_s0'], cells['oncall|think_on|ctrl_s0']
        self.assertGreater(on_c['per_turn']['reasoning']['mean'], 0)
        self.assertLess(on_w['per_turn']['reasoning']['mean'], on_c['per_turn']['reasoning']['mean'])
        self.assertEqual(cells['oncall|think_off|ctrl_s0']['per_turn']['reasoning']['mean'], 0)
        un = cells['oncall|think_on|untouched']
        self.assertEqual(un['truncated_turns'], 1)                    # the red arm: counted, not folded in
        self.assertGreater(un['truncation_rate_turns'], 0)
        self.assertLess(un['acc'], on_c['acc'])
        self.assertEqual(on_c['acc'], 1.0)
        self.assertIn('256', on_c['by_length'])
        self.assertTrue(any(k.endswith('-256') or k.endswith('-64') for k in on_c['by_position']))
        s1c = cells['s1gen|think_off|untouched']
        self.assertLess(s1c['acc'], 1.0)
        self.assertIn('le256', s1c)
        # the splits data/oncall.py's note asks for: by request kind, calls <= 4 vs >= 5, delivery-turn tokens; S1
        # by domain
        self.assertEqual(set(on_c['by_kind']), {'ack_open', 'page_role'})
        self.assertEqual(sum(v['n'] for v in on_c['by_kind'].values()), on_c['primary_turns'])
        wrong_kind = items[[i['id'] for i in items].index(ct[0][0])]['turns'][ct[0][1]]['meta']['action']
        self.assertLess(un['by_kind'][wrong_kind]['acc'], 1.0)          # the untouched arm's wrong call lands there
        self.assertEqual(on_c['by_kind'][wrong_kind]['acc'], 1.0)
        self.assertEqual(on_c['call_turn_acc_le4'], 1.0)
        self.assertGreater(on_c['per_turn_events']['reasoning']['mean'], 0)
        self.assertEqual(set(s1c['by_kind']), set(run_s1gen.DOMAINS))
        self.assertEqual(cells['oncall|think_on|untouched']['context_overflow_items'], 0)
        pairs = {(p['task'], p['mode'], p['a'], p['b']): p for p in doc['paired']}
        self.assertEqual(len(pairs), 2 * 2 * 3)  # 2 tasks x 2 modes x (wide/ctrl, wide/untouched, ctrl/untouched)
        wc = pairs[('oncall', 'think_on', 'wide_s0', 'ctrl_s0')]['metrics']
        self.assertLess(wc['reasoning tokens per turn']['diff'], 0)
        self.assertLess(wc['reasoning tokens per turn']['hi'], 0)
        self.assertLess(wc['reasoning per turn, both correct']['hi'], 0)
        self.assertAlmostEqual(wc['accuracy (primary turns)']['diff'], 0.0)
        cu = pairs[('oncall', 'think_on', 'ctrl_s0', 'untouched')]['metrics']
        self.assertGreater(cu['accuracy (primary turns)']['diff'], 0)
        self.assertNotIn('reasoning per turn, both correct', pairs[('oncall', 'think_off', 'wide_s0',
                                                                     'ctrl_s0')]['metrics'])
        md = om.read_text()
        self.assertIn('Accuracy % by request kind', md)
        self.assertIn('Accuracy % by domain', md)
        self.assertIn('## oncall, think_on', md)
        self.assertIn('## Paired comparisons', md)
        self.assertNotIn(chr(0x2014), md)  # no em dash in the report

    def test_pure_python_bootstrap_matches_the_shape(self):
        vec = {k: [1.0, 2.0, 3.0, 4.0] for k in ('n', 'sa', 'sb', 'nra', 'nrb', 'ra', 'rb', 'ca', 'cb', 'bc_n',
                                                  'bc_ra', 'bc_rb', 'task_ra', 'task_rb', 'task_ca', 'task_cb')}
        vec['rb'] = [2.0, 4.0, 6.0, 8.0]
        real = report.bootstrap(vec, 500, 1)
        saved = sys.modules.get('numpy')
        sys.modules['numpy'] = None  # import numpy now raises ImportError
        try:
            pure = report.bootstrap(vec, 500, 1)
        finally:
            if saved is None:
                del sys.modules['numpy']
            else:
                sys.modules['numpy'] = saved
        self.assertEqual(set(real), set(pure))
        self.assertAlmostEqual(real['reasoning tokens per turn']['diff'], pure['reasoning tokens per turn']['diff'])
        self.assertLess(pure['reasoning tokens per turn']['diff'], 0)


@unittest.skipUnless(HAVE_LANE, 'needs the Hebrew lane (agentic_env.py)')
class PartialTests(unittest.TestCase):
    """score_partial.py and report.py --partial / --align: an in-flight session is scored on its closed turns only."""

    def killed_run(self, script=None, kill=25):
        out = tmpdir()
        with serving(**fake_kw(thinking=True, script=script or {})) as srv:
            srv.kill_after(kill)
            s = run_oc(srv, out, 'on', ['--abort-after', '2', '--max-tokens', '200'])
        self.assertFalse(s['complete'])
        return out

    def test_closed_turns_are_exactly_what_the_finished_run_writes(self):
        _, items, _ = data_dir()
        ct = call_turns(items)
        script = {ct[0]: 'wrong_arg', ct[2]: 'omit'}
        out = self.killed_run(script)
        res = score_partial.score_run(out, count=counter())
        self.assertGreaterEqual(res['partial'], 1, res)
        self.assertGreater(res['closed_turns'], 0, res)
        part = {r['key']: r for r in pc.read_jsonl(out / score_partial.OUT_NAME)}
        self.assertEqual(set(part) & set(ledger(out)), set())     # a final ledger row is never rescored
        with serving(**fake_kw(thinking=True, script=script)) as srv:
            self.assertTrue(run_oc(srv, out, 'on', ['--max-tokens', '200'])['complete'])
        final = ledger(out)
        for k, r in part.items():
            c = r['closed_turns']
            if r['partial']:
                self.assertEqual(r['failure'], 'in_flight')
                self.assertEqual(r['turns'][:c], final[k]['turns'][:c], k)  # the resume replays the same records
                self.assertTrue(all(t['in_flight'] and not t['reached'] and t['status'] == 'unreached'
                                    for t in r['turns'][c:]), k)
                self.assertNotIn('in_flight', json.dumps(r['turns'][:c]))
            else:
                self.assertEqual(r['turns'], final[k]['turns'], k)
        self.assertTrue(any(r['partial'] and r['in_flight_generations'] >= 0 for r in part.values()))

    def test_report_partial_counts_closed_turns_only_and_the_red_arm(self):
        out = self.killed_run()  # a clean model: every decided turn scores 1
        score_partial.score_run(out, count=counter())
        n_final = len(ledger(out))
        plain = report.cell_metrics(report.load_run(out))
        self.assertEqual(plain['items'], n_final)                   # without --partial the file is not read
        run = report.load_run(out, partial=True)
        self.assertGreater(run['partial_rows'], 0)
        m = report.cell_metrics(run)
        self.assertEqual(m['items'], n_final + run['partial_rows'])
        self.assertGreater(m['in_flight_turns'], 0)
        self.assertEqual(m['unreached_turns'], 0)
        self.assertEqual(m['acc'], 1.0)
        self.assertEqual(m['exact_items'], 1.0 if n_final else None)  # only finished sessions count as exact
        # red arm: the same rows without the in-flight marks score their tail as failures
        for r in run['rows'].values():
            for t in r['turns']:
                t.pop('in_flight', None)
        self.assertLess(report.cell_metrics(run)['acc'], 1.0)

    def test_pairs_and_align_use_the_turns_both_arms_reached(self):
        root = tmpdir()
        full = root / 'ctrl_s0' / 'oncall_think_on'
        with serving(**fake_kw(thinking=True)) as srv:
            a = oncall_args(full, srv.url, 'on', ['--max-tokens', '200'])
            a.arm = 'ctrl_s0'
            self.assertTrue(run_oncall.run(a, client=fast_client(srv.url), count=counter())['complete'])
        part = root / 'wide_s0' / 'oncall_think_on'
        with serving(**fake_kw(thinking=True)) as srv:  # the same clean model, killed mid-run
            srv.kill_after(25)
            a = oncall_args(part, srv.url, 'on', ['--max-tokens', '200', '--abort-after', '2'])
            a.arm = 'wide_s0'
            self.assertFalse(run_oncall.run(a, client=fast_client(srv.url), count=counter())['complete'])
        score_partial.score_run(part, count=counter())
        runs = [report.load_run(d, partial=True) for d in (full, part)]
        doc = report.build(runs, draws=500)
        p = doc['paired'][0]
        self.assertEqual((p['a'], p['b']), ('wide_s0', 'ctrl_s0'))
        self.assertEqual(p['items'], len(runs[0]['rows']))
        for m in ('accuracy (primary turns)', 'reasoning tokens per turn', 'reasoning tokens per task',
                  'total tokens per task'):
            self.assertAlmostEqual(p['metrics'][m]['diff'], 0.0, msg=m)  # same model, same turns
        self.assertTrue(any('in flight' in n for n in doc['notes']))
        al = report.build([report.load_run(d, partial=True) for d in (full, part)], draws=500, align=True)
        cw, cc = al['cells']['oncall|think_on|wide_s0'], al['cells']['oncall|think_on|ctrl_s0']
        self.assertEqual(cw['per_task'], cc['per_task'])
        self.assertEqual(cw['primary_turns'], cc['primary_turns'])
        self.assertGreater(cc['in_flight_turns'], 0)                 # the finished arm is cut to the same turns
        self.assertEqual(cc['partial_items'], cw['partial_items'])
        # red arm: unaligned, the finished arm's tokens per task cover more turns than the partial arm's
        un = doc['cells']['oncall|think_on|ctrl_s0']['per_task']['reasoning']['mean']
        self.assertGreater(un, doc['cells']['oncall|think_on|wide_s0']['per_task']['reasoning']['mean'])
        self.assertTrue(al['aligned'])


# ====================================================================== server check and run_all.sh
def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


class ServerCheckTests(unittest.TestCase):
    def check(self, **kw):
        with serving(auto=True, generation_config=GEN_CFG, counter=counter(), **kw) as srv:
            ok, checks = pc.check_server(fast_client(srv.url), counter())
            return ok, {c['check']: c['ok'] for c in checks}, srv.stats()['violations']

    def test_engaged_parsers_pass(self):
        ok, checks, viol = self.check()
        self.assertTrue(ok, checks)
        self.assertEqual(len(checks), 5)
        self.assertEqual(viol, [])

    def test_missing_reasoning_parser_fails(self):
        ok, checks, _ = self.check(reasoning_parser=False)
        self.assertFalse(ok)
        self.assertFalse(checks['reasoning_parser_think_on'])

    def test_missing_tool_parser_fails(self):
        ok, checks, _ = self.check(tool_parser=False)
        self.assertFalse(ok)
        self.assertFalse(checks['tool_parser_think_off'])


STUB_SERVE_ARM = r'''#!/usr/bin/env bash
# stub of evals/serve_arm.sh for the run_all.sh tests: starts the probe fake instead of SGLang
set -u
mode=$1; shift
arm=""; kind=""; port=30000; extra=""; radix=off; ctx=0; trainable=""; dp=1
while [ $# -gt 0 ]; do
  case "$1" in
    --arm) arm=$2; shift 2 ;; --kind) kind=$2; shift 2 ;; --port) port=$2; shift 2 ;; --extra) extra=$2; shift 2 ;;
    --radix) radix=$2; shift 2 ;; --ctx) ctx=$2; shift 2 ;; --trainable) trainable=$2; shift 2 ;;
    --dp) dp=$2; shift 2 ;; --replace|--purge) shift ;; *) echo "stub: unknown $1" >&2; exit 2 ;;
  esac
done
echo "$mode $arm $kind $trainable" >>"$STUB_LOG"
if [ "$mode" = up ]; then
  mkdir -p "$EVAL_ROOT/$arm"
  flags=""; [ "${STUB_NO_REASONING:-0}" = 1 ] && flags="--no-reasoning-parser"
  setsid "$STUB_PY" "$STUB_FAKE" --port "$port" --dp "$dp" --items "$STUB_ITEMS" --s1 "$STUB_S1" $flags \
    ${STUB_GENCFG:+--generation-config "$STUB_GENCFG"} >"$EVAL_ROOT/$arm/server.log" 2>&1 </dev/null &
  pid=$!
  for _ in $(seq 1 100); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/health")" = 200 ] && break; sleep 0.2
  done
  printf '{"arm":"%s","pid":%s,"port":%s}\n' "$arm" "$pid" "$port" >"$EVAL_ROOT/serving.json"
  printf '{"extra_args":"%s","radix":"%s","context_length":%s}\n' "$extra" "$radix" "$ctx" >"$EVAL_ROOT/$arm/serve.json"
  printf '{"arm":"%s","engagement":{"ok":true},"serve":{"extra_args":"%s"}}\n' "$arm" "$extra" >"$EVAL_ROOT/$arm/arm.json"
  exit 0
fi
if [ "$mode" = down ]; then
  pid=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["pid"])' "$EVAL_ROOT/serving.json" 2>/dev/null)
  [ -n "$pid" ] && kill -TERM -- "-$pid" 2>/dev/null; [ -n "$pid" ] && kill "$pid" 2>/dev/null
  rm -f "$EVAL_ROOT/serving.json"
  exit 0
fi
exit 2
'''


@unittest.skipUnless(HAVE_LANE and HAVE_TOK and shutil.which('curl') and shutil.which('setsid'),
                     'needs the lane, the tokenizer, curl and setsid')
class RunAllTests(unittest.TestCase):
    def setup_box(self, **env_over):
        d, items, s1 = data_dir()
        w = tmpdir()
        stub = w / 'serve_arm_stub.sh'
        stub.write_text(STUB_SERVE_ARM)
        stub.chmod(0o755)
        for arm in ('ctrl_s0', 'wide_s0'):
            (w / f'{arm}.pt').write_bytes(arm.encode())
        env = dict(os.environ)
        env.update({'NEGEIG_W': str(w / 'W'), 'PROBE_ROOT': str(w / 'probe'), 'PROBE_PY': sys.executable,
                    'PROBE_TOKENIZER': TOKENIZER, 'ONCALL_ITEMS': str(d / 'oncall.eval.jsonl'), 'S1_DIR': str(s1),
                    'ONCALL_ARGS': '--per-length 2', 'S1_ARGS': '--lengths 32 --per-cell 1 --domains ops,custody',
                    'WIDE_TRAINABLE': str(w / 'wide_s0.pt'), 'CTRL_TRAINABLE': str(w / 'ctrl_s0.pt'),
                    'WIDE_SHA256': '', 'CTRL_SHA256': '', 'SERVE_ARM': str(stub), 'PORT': str(free_port()),
                    'SELFDISTILL_LANE_DIR': str(LANE), 'POLL_S': '1', 'HEALTH_FAILS': '3', 'CONC_ONCALL': '4',
                    'CONC_S1': '4', 'MAX_TOKENS_ON': '200', 'STUB_LOG': str(w / 'stub.log'),
                    'STUB_PY': sys.executable, 'STUB_FAKE': str(HERE / 'fake_probe_server.py'),
                    'STUB_ITEMS': str(d / 'oncall.eval.jsonl'), 'STUB_S1': str(s1), 'STUB_GENCFG': GEN_CFG or '',
                    'PYTHONDONTWRITEBYTECODE': '1'})
        env.update(env_over)
        return w, env

    def run_all(self, env, *args, timeout=600):
        return subprocess.run(['bash', str(HERE / 'run_all.sh'), *args], env=env, capture_output=True, text=True,
                              timeout=timeout)

    def test_full_run_receipts_report_and_a_resume_that_serves_nothing(self):
        w, env = self.setup_box()
        p = self.run_all(env)
        self.assertEqual(p.returncode, 0, p.stderr[-3000:])
        rec = w / 'probe' / 'receipts'
        for arm in ('untouched', 'ctrl_s0', 'wide_s0'):
            chk = json.loads((rec / f'{arm}.server_check.json').read_text())
            self.assertTrue(chk['ok'])
            self.assertIn('routed_dp_rank_last', [c['check'] for c in chk['checks']])
            for job in ('oncall_think_off', 'oncall_think_on', 's1gen_think_off', 's1gen_think_on'):
                r = json.loads((rec / f'{arm}.{job}.json').read_text())
                self.assertTrue(r['complete'], (arm, job, r))
                self.assertEqual(r['exit'], 0)
        self.assertTrue((rec / 'plan.oncall_think_on.json').exists())
        doc = json.loads((w / 'probe' / 'report.json').read_text())
        self.assertEqual(len(doc['paired']), 2 * 2 * 3)
        stub = (w / 'stub.log').read_text().split('\n')
        self.assertEqual(sum(line.startswith('up ') for line in stub), 3)
        self.assertIn(f"up probe_wide_s0 wide {w / 'wide_s0.pt'}", stub)
        serve = json.loads((w / 'probe' / 'serve' / 'probe_untouched' / 'serve.json').read_text())
        self.assertEqual(serve['extra_args'], '--tool-call-parser qwen3_coder --reasoning-parser qwen3')
        self.assertEqual(serve['radix'], 'on')
        p = self.run_all(env)
        self.assertEqual(p.returncode, 0, p.stderr[-3000:])
        self.assertEqual(sum(line.startswith('up ') for line in (w / 'stub.log').read_text().split('\n')), 3)
        self.assertIn('every job complete, not serving it', p.stderr)

    def test_a_server_without_the_reasoning_parser_stops_the_run(self):
        w, env = self.setup_box(STUB_NO_REASONING='1')
        p = self.run_all(env, '--arms', 'untouched', '--modes', 'off', '--tasks', 's1gen')
        self.assertEqual(p.returncode, 1)
        self.assertIn('the server check failed', p.stderr)
        self.assertFalse(json.loads((w / 'probe' / 'receipts' / 'untouched.server_check.json').read_text())['ok'])
        self.assertTrue(any(line.startswith('down') for line in (w / 'stub.log').read_text().split('\n')))
        self.assertFalse((w / 'probe' / 'serve' / 'serving.json').exists())  # the server was taken down

    def test_a_wrong_trainable_hash_is_refused(self):
        w, env = self.setup_box(WIDE_SHA256='0' * 64)
        p = self.run_all(env, '--arms', 'wide_s0', '--modes', 'off', '--tasks', 's1gen')
        self.assertEqual(p.returncode, 1)
        self.assertIn('expected 0000', p.stderr)

    def test_a_context_that_does_not_fit_is_refused_before_serving(self):
        w, env = self.setup_box(CTX='2000')
        p = self.run_all(env, '--arms', 'untouched', '--modes', 'off')
        self.assertEqual(p.returncode, 1)
        self.assertIn('needs a context of', p.stderr)
        self.assertFalse((w / 'stub.log').exists())

    def test_jobs_runs_only_the_listed_task_modes(self):
        w, env = self.setup_box()
        p = self.run_all(env, '--arms', 'untouched', '--jobs', 'oncall:on s1gen:off')
        self.assertEqual(p.returncode, 0, p.stderr[-3000:])
        rec = w / 'probe' / 'receipts'
        self.assertTrue(json.loads((rec / 'untouched.oncall_think_on.json').read_text())['complete'])
        self.assertTrue(json.loads((rec / 'untouched.s1gen_think_off.json').read_text())['complete'])
        self.assertFalse((rec / 'untouched.oncall_think_off.json').exists())
        self.assertFalse((rec / 'untouched.s1gen_think_on.json').exists())
        p = self.run_all(env, '--arms', 'untouched', '--jobs', 'oncall:maybe')
        self.assertEqual(p.returncode, 1)
        self.assertIn("the mode is on or off", p.stderr)

    def test_print_cmd(self):
        w, env = self.setup_box()
        p = self.run_all(env, '--print-cmd')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('--extra "--tool-call-parser qwen3_coder --reasoning-parser qwen3"', p.stdout)
        self.assertIn('--radix on', p.stdout)
        self.assertIn('--thinking on', p.stdout)
        self.assertIn('--history-reasoning drop', p.stdout)
        self.assertIn('--dp-ranks 8', p.stdout)
        self.assertIn('down --purge', p.stdout)


if __name__ == '__main__':
    unittest.main()
