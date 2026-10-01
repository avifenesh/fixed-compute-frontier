"""Tests for the self-distillation job. CPU only, stdlib unittest, no GPU, no real server.

    OMP_NUM_THREADS=2 nice -n 10 taskset -c 8-15 ~/.venvs/negeig/bin/python -m unittest -v test_selfdistill

What is covered
  parity     ToolEpisode (message level) against the lane's real agentic_env.AgenticEpisode (token level), driven in
             lockstep over all 2,537 pool items with hostile policies. Scores, errors, calls, steps and messages must
             agree at every generation. Only the parse and render primitives are stubbed (they need sglang and the
             tokenizer); the state machine, call_key and turn_score are the lane's own code. A deliberately wrong
             episode must make the same check fail.
  oracle     with every defect switched off, the fake server's oracle solves every pool item (the pool's own
             expectations are reachable through this job's episode loop)
  filters    every drop reason of the translation, chat and tool filters fires on a hand-built row, and a clean row
             passes; the unsupported near-miss exemption; dedupe; quota; finalize is idempotent
  jsonl      torn-tail repair, resume append, unreadable line skipping
  client     retry, reconnect, exhaustion, non-retryable 400, request shape (think-off, vendor arm) checked server side
  phases     multi-wave translation with resume, end-to-end chat run, episode failure classes
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import tempfile
import types
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import chat_gen  # noqa: E402
import sd_fake_server  # noqa: E402
import sd_filters  # noqa: E402
import tool_gen  # noqa: E402
from sd_common import (DEFAULT_TOKENIZER, SAMPLING, THINK_OFF, Aborted, ApiError, Client, Exhausted,  # noqa: E402
                       Generation, JsonlSink, add_client_args, load_pool, read_jsonl, repair_jsonl,
                       sampling_profile, seed_for, write_jsonl_atomic, write_manifest)
from sd_fake_server import MODEL, FakeServer, Faults  # noqa: E402

import sd_common  # noqa: E402

HAVE_LANE = (sd_common.LANE_DIR / 'agentic_env.py').is_file()
_POOL = []


def pool():
    if not _POOL:
        _POOL.append(load_pool())
    return _POOL[0]


# ------------------------------------------------------------------ parity harness
class FakeTok:
    """Bytes for ids. Only what AgenticEpisode touches: encode and decode."""

    def encode(self, text, add_special_tokens=False):
        return list(text.encode('utf-8'))

    def decode(self, ids):
        return ''


def make_env(office_env, current):
    """agentic_env's `env` (office_env) with the sglang and tokenizer primitives replaced by stubs. The constants and
    server_messages (the strict arguments check behind the HTTP 400) are the lane's real ones."""

    def render_text(tok, messages, tools, add_generation_prompt=True):
        parts = []
        for m in office_env.server_messages(messages):  # raises RenderError like the served template path
            body = m.get('content', '')
            for c in m.get('tool_calls') or []:
                body += '<tool_call>' + json.dumps(c['function'], ensure_ascii=False, sort_keys=True) + '</tool_call>'
            parts.append(f"<|im_start|>{m['role']}\n{body}<|im_end|>\n")
        if add_generation_prompt:
            parts.append('<|im_start|>assistant\n')
        return ''.join(parts)

    def render_prompt(tok, messages, tools):
        text = render_text(tok, messages, tools)
        return text, tok.encode(text)

    def parse_turn(tok, prompt_ids, token_ids, stop_reason, tools):
        gen = current['gen']
        message = {'role': 'assistant', 'content': gen.content or ''}
        if gen.tool_calls:
            message['tool_calls'] = gen.tool_calls
        return message, gen.finish_reason, {'reasoning_chars': len(gen.reasoning or ''), 'eos_appended': False,
                                            'eos': office_env.IM_END}

    return types.SimpleNamespace(
        MAX_TOKENS=office_env.MAX_TOKENS, TURN_BUDGET=office_env.TURN_BUDGET, HTTP_400=office_env.HTTP_400,
        TRUNCATED=office_env.TRUNCATED, EXTRA_KEYS=office_env.EXTRA_KEYS, TASKS=office_env.TASKS,
        IM_END=office_env.IM_END, ENDOFTEXT=office_env.ENDOFTEXT, RenderError=office_env.RenderError,
        check_tokenizer=lambda tok: None, template_tools=lambda tools: list(tools), parser_tools=lambda tools: list(tools),
        context_error=lambda n: False, render_text=render_text, render_prompt=render_prompt, parse_turn=parse_turn)


def chaotic_plan(item, messages, rnd, rates, stall, drift, chaos):
    """The fake server's oracle, then shapes a parser can produce that the oracle never does."""
    plan = copy.deepcopy(sd_fake_server.plan_tool_response(item, messages, rnd, rates=rates, stall=stall, drift=drift))
    u = rnd.random()
    calls = plan['tool_calls']
    if calls and u < chaos:  # arguments that are valid JSON but not an object: the served template answers 400
        calls[0]['function']['arguments'] = rnd.choice(['[1, 2]', 'null', '"text"', '7'])
    elif calls and u < 2 * chaos:  # the same call twice in one response
        twin = copy.deepcopy(calls[0])
        twin['id'] += 'b'
        calls.append(twin)
    elif calls and u < 3 * chaos:  # cut off mid-call
        plan['finish'] = 'length'
    elif not calls and u < chaos:  # an empty stop
        plan.update(content='', finish='stop')
    return plan


def to_generation(plan):
    return Generation(content=plan['content'] or '', reasoning=plan['reasoning'], tool_calls=plan['tool_calls'],
                      finish_reason=plan['finish'], usage={'completion_tokens': 3, 'prompt_tokens': 5})


ZERO = {k: 0.0 for k in sd_fake_server.RATES}
HOSTILE = {k: min(0.3, v * 4) for k, v in sd_fake_server.RATES.items()}


class Lockstep:
    def __init__(self, test, episode_cls=None):
        self.test = test
        self.office_env, self.ag, self.items = pool()
        self.current = {}
        self.ns = make_env(self.office_env, self.current)
        self.episode_cls = episode_cls or tool_gen.ToolEpisode
        self.seen_errors, self.seen_scores = Counter(), set()

    def server_rejects(self, messages):
        try:
            self.office_env.server_messages(messages)
        except self.office_env.RenderError:
            return True
        return False

    def same(self, real, mine, where):
        t = self.test
        for name, a, b in (('done', real.done, mine.done), ('turn_index', real.turn_index, mine.turn_index),
                           ('steps', real.steps, mine.steps), ('turn_scores', real.turn_scores, mine.turn_scores),
                           ('errors', real.errors, mine.errors), ('calls', real.calls, mine.calls),
                           ('generations', real.turns, mine.generations), ('messages', real.messages, mine.messages)):
            t.assertEqual(a, b, f'{where}: {name} differs between AgenticEpisode and ToolEpisode')

    def run(self, item, rates, chaos, stall=False, drift=False, seed=0):
        ag = self.ag
        real = ag.AgenticEpisode(item, FakeTok(), 10 ** 9)
        mine = self.episode_cls(item, self.ns, ag)
        rnd = random.Random(seed_for('parity', item['id'], seed))
        for n in range(64):
            r = real.next_request()
            m = mine.next_request()
            if m is not None and self.server_rejects(mine.messages):
                mine.http_error()  # the served endpoint answers 400 to this request
                m = None
            self.test.assertEqual(r is None, m is None, f"{item['id']} step {n}: one episode ended, the other did not")
            if r is None:
                break
            plan = chaotic_plan(item, mine.messages, rnd, rates, stall, drift, chaos)
            gen = to_generation(plan)
            self.current['gen'] = gen
            real.observe([1], None, 'length' if gen.finish_reason == 'length' else 'stop')
            mine.observe(gen)
            self.same(real, mine, f"{item['id']} step {n}")
        else:
            self.test.fail(f"{item['id']}: no end after 64 generations")
        self.same(real, mine, f"{item['id']} final")
        want = real.result()['score']
        got = mine.result()
        self.test.assertEqual(want['turn_scores'], got['turn_scores'], item['id'])
        self.test.assertEqual(want['errors'], got['errors'], item['id'])
        self.test.assertEqual(real.result()['reward'], got['reward'], item['id'])
        for e in got['errors']:
            self.seen_errors[e.split(':')[0] if e.startswith(('JSON', 'Unicode')) else e] += 1
        self.seen_scores.update(got['turn_scores'])
        return got


@unittest.skipUnless(HAVE_LANE, 'needs the Hebrew lane (SELFDISTILL_LANE_DIR with agentic_env.py)')
class ParityTests(unittest.TestCase):
    def setUp(self):
        self.office_env, self.ag, self.items = pool()
        patcher = mock.patch.object(self.ag, 'env', None)  # replaced per Lockstep below
        patcher.start()
        self.addCleanup(patcher.stop)

    def lockstep(self, episode_cls=None):
        ls = Lockstep(self, episode_cls)
        self.ag.env = ls.ns
        return ls

    def test_hostile_policies_match_the_lane_episode_on_every_item(self):
        ls = self.lockstep()
        with mock.patch.object(sd_fake_server, 'SPREAD_RATE', 0.4):
            for k, item in enumerate(self.items):
                ls.run(item, HOSTILE, chaos=0.05, stall=k % 17 == 0, drift=k % 13 == 0, seed=1)
        ends = ls.seen_errors
        self.assertGreater(ends[self.office_env.TURN_BUDGET], 0, dict(ends))
        self.assertGreater(ends[self.office_env.TRUNCATED], 0, dict(ends))
        self.assertGreater(ends[self.office_env.HTTP_400], 0, dict(ends))
        self.assertGreater(ends['JSONDecodeError'], 0, dict(ends))
        self.assertTrue({0.0, 0.5, 1.0} <= ls.seen_scores, ls.seen_scores)

    def test_clean_oracle_solves_every_item(self):
        ls = self.lockstep()
        bad = []
        with mock.patch.object(sd_fake_server, 'SPREAD_RATE', 0.0):
            for item in self.items:
                res = ls.run(item, ZERO, chaos=0.0)
                if res['reward'] != 1.0 or res['errors']:
                    bad.append((item['id'], item['kind'], res['turn_scores'], res['errors']))
        self.assertEqual(bad[:5], [], f'{len(bad)} items the oracle cannot solve')

    def test_a_lenient_episode_fails_the_lockstep_check(self):
        """The red arm: an episode that scores every closed turn 1.0 must be caught."""
        class Lenient(tool_gen.ToolEpisode):
            def _close_turn(self):
                self.turn_scores.append(1.0)
                self.turn_index += 1
                self.steps = []

        ls = self.lockstep(Lenient)
        with self.assertRaises(AssertionError):
            for item in self.items[:200]:
                ls.run(item, HOSTILE, chaos=0.05)

    def test_a_wrong_step_limit_fails_the_lockstep_check(self):
        class Patient(tool_gen.ToolEpisode):
            def observe(self, gen):
                with mock.patch.object(self.ag, 'MAX_STEPS_PER_TURN', 99):
                    return super().observe(gen)

        ls = self.lockstep(Patient)
        with self.assertRaises(AssertionError):
            for k, item in enumerate(self.items[:400]):
                ls.run(item, HOSTILE, chaos=0.0, stall=k % 3 == 0)


# ------------------------------------------------------------------ filters
EN_PROMPT = 'Explain how rainbows form, in simple words for a ten year old.'
EN_ANSWER = ('Sunlight enters a raindrop, bends, bounces off the back and bends again on the way out. Each color '
             'bends by a slightly different angle, so the colors spread into a band that you see as a rainbow.')
HE_PROMPT = 'מתי נוסדה מדינת ישראל ומי הכריז על כך?'
HE_ANSWER = 'מדינת ישראל הוקמה בחמישה במאי אלף תשע מאות ארבעים ושמונה, ודוד בן גוריון הכריז על כך בתל אביב.'


def chat_row(prompt=EN_PROMPT, content=EN_ANSWER, **kw):
    row = {'id': 'p1', 'lang': 'en', 'source': 'dolly', 'category': 'open_qa', 'prompt': prompt, 'content': content,
           'reasoning': '', 'finish': 'stop', 'usage': {'completion_tokens': 40}}
    row.update(kw)
    return row


class ChatFilterTests(unittest.TestCase):
    def test_clean_row_passes(self):
        self.assertIsNone(sd_filters.chat_reason(chat_row()))
        self.assertIsNone(sd_filters.chat_reason(chat_row(HE_PROMPT, HE_ANSWER, lang='he')))

    def test_every_reason(self):
        cases = {
            'api_error': chat_row(error={'status': 400}),
            'empty': chat_row(content='   \n'),
            'truncated': chat_row(finish='length'),
            'reasoning_leak': chat_row(reasoning='let me think'),
            'markup_leak': chat_row(content=EN_ANSWER + '\n<tool_call>{"name": "x"}</tool_call>'),
            'refusal': chat_row(content="I'm sorry, but I can't help with that request."),
            'language_mismatch': chat_row(HE_PROMPT, EN_ANSWER, lang='he'),
            'degenerate_repeat': chat_row(content='The cat sat on the mat and looked out. ' * 40),
        }
        for want, row in cases.items():
            self.assertEqual(sd_filters.chat_reason(row), want, want)
        self.assertEqual(sd_filters.chat_reason(chat_row(content='<think>plan</think> ' + EN_ANSWER)), 'reasoning_leak')
        self.assertEqual(sd_filters.chat_reason(chat_row(finish='tool_calls')), 'markup_leak')

    def test_refusal_heuristic_keeps_honest_caveats_and_questions(self):
        self.assertEqual(sd_filters.chat_reason(chat_row(content='אני מצטער, אבל אני לא יכול לעזור בזה.')), 'refusal')
        ask = chat_row(content="I can't tell which city you mean. Which one should I use?")
        self.assertIsNone(sd_filters.chat_reason(ask))
        long_caveat = ("I'm sorry, but I cannot verify this. Light from the sun is a mix of every color. A raindrop "
                       "acts like a tiny prism and splits that mix. The red light bends the least and the violet "
                       "light bends the most. You only see the band when the sun is behind you and the rain is in "
                       "front of you. The higher the sun, the lower the bow sits on the horizon. Two bows appear "
                       "when the light bounces twice inside each drop, and the second one has reversed colors. "
                       "Mist from a waterfall or a garden hose can make a small bow too. Try it on a sunny day. A pilot "
                       "flying above the clouds can even see the whole circle, because nothing cuts it off at the "
                       "ground. Photographers often use a polarizing filter to make the colors stand out more.")
        self.assertGreater(len(long_caveat), 600)
        self.assertIsNone(sd_filters.chat_reason(chat_row(content=long_caveat)))

    def test_hebrew_refusal_heuristic_matches_the_english_one(self):
        """The Hebrew list once fired on any 'I cannot ...' (even 'it cannot be') and on 'I cannot know for sure',
        while the English list keeps the same hedges. Refusals need the can't-do phrase plus a help verb."""
        for text in ('אני מצטער שאני לא יכול לעזור בזה.', 'מצטער, אני לא יכול לעזור בזה.',
                     'אני לא יכולה לסייע בבקשה הזו.', 'לא אוכל לעזור', 'כמודל שפה, אני לא יכול לגלוש באינטרנט.',
                     'לצערי לא אוכל לספק מידע כזה.', 'אינני יכול לעזור בכך'):
            self.assertTrue(sd_filters.is_refusal(text), text)
        for text in ('לא יכול להיות, כי 2+2=4.', 'אני לא יכול לדעת בוודאות, אבל כנראה שכן.',
                     'אני לא יודע בדיוק, אבל נראה שזה 1990.', 'מצטער, אני לא יודע את התשובה.',
                     'סליחה, אני לא מבין את השאלה. אפשר לנסח מחדש?', 'לא, זה לא נכון. הבירה של צרפת היא פריז.'):
            self.assertFalse(sd_filters.is_refusal(text), text)
        hedge = chat_row(prompt=HE_PROMPT, content='אני לא יכול לדעת בוודאות, אבל מדינת ישראל נוסדה בשנת 1948.')
        self.assertIsNone(sd_filters.chat_reason(hedge))

    def test_language_rules(self):
        mismatch = sd_filters.language_mismatch
        self.assertTrue(mismatch(HE_PROMPT, EN_ANSWER))
        self.assertTrue(mismatch(HE_PROMPT, 'Done.'))  # too few letters for a share, still English
        self.assertFalse(mismatch(HE_PROMPT, HE_ANSWER))
        self.assertFalse(mismatch(HE_PROMPT, 'בוצע.'))
        self.assertTrue(mismatch(EN_PROMPT, HE_ANSWER))
        self.assertTrue(mismatch(EN_PROMPT, EN_ANSWER + ' 翻訳 です'))
        self.assertFalse(mismatch('12 34', 'ok'))  # no letters in the request: nothing to compare
        code_prompt = 'כתוב פונקציה ב-Python: ```python\nprint("hello")\n```'
        self.assertFalse(mismatch(code_prompt, HE_ANSWER + ' ```python\nprint("hi")\n```'))

    def test_duplicate_and_ledger_semantics(self):
        rows = [chat_row(id='a'), chat_row(id='b'), chat_row(id='c', prompt='Name two primary colors.',
                                                               content='Red and blue are two primary colors.')]
        kept, report = sd_filters.filter_chat(rows, meta={'m': 1})
        self.assertEqual([k['id'] for k in kept], ['a', 'c'])
        self.assertEqual(report['dropped'], {'duplicate': 1})
        self.assertEqual(kept[0]['messages'], [{'role': 'user', 'content': EN_PROMPT},
                                               {'role': 'assistant', 'content': EN_ANSWER}])
        again, _ = sd_filters.filter_chat(rows + [chat_row(id='a', content='A different answer to the same id.')])
        self.assertEqual([k['id'] for k in again], ['a', 'b', 'c'])
        self.assertEqual(again[0]['messages'][1]['content'], 'A different answer to the same id.',
                         'a ledger holding an id twice keeps the last row, never both')


TR_SRC = 'Explain how to reverse a list in Python using `list.reverse()` and give 3 examples.'
TR_OUT = 'הסבר כיצד להפוך רשימה בפייתון באמצעות `list.reverse()` ותן 3 דוגמאות.'


def tr_row(out=TR_OUT, src=TR_SRC, **kw):
    row = {'id': 't1', 'src': src, 'out': out, 'finish': 'stop', 'usage': {'completion_tokens': 30}}
    row.update(kw)
    return row


class TranslationFilterTests(unittest.TestCase):
    def test_clean_row_passes(self):
        self.assertIsNone(sd_filters.translation_reason(tr_row()))

    def test_every_reason(self):
        cases = {
            'api_error': tr_row(error={'status': 400}),
            'truncated': tr_row(finish='length'),
            'empty': tr_row(out=' '),
            'markup': tr_row(out=TR_OUT + ' </message>'),
            'refusal': tr_row(out="I'm sorry, but I can't translate that."),
            'preamble': tr_row(out='הנה התרגום: ' + TR_OUT),
            'cjk': tr_row(out=TR_OUT + ' 翻訳'),
            'not_hebrew': tr_row(out='Explain how to reverse a list in Python using `list.reverse()`, 3 examples.'),
            'code_changed': tr_row(out=TR_OUT.replace('`list.reverse()`', 'list.reverse')),
            'numbers_changed': tr_row(out=TR_OUT.replace('3', '9')),
            'length': tr_row(src='Name a fruit.', out='שם של פרי שאפשר לאכול בקיץ כשחם מאוד בחוץ ואוהבים משהו מרענן וקר'),
        }
        for want, row in cases.items():
            self.assertEqual(sd_filters.translation_reason(row), want, want)
        self.assertEqual(set(cases), set(sd_filters.TRANSLATION_REASONS))

    def test_a_preamble_the_source_already_has_is_not_a_defect(self):
        src = 'Sure, here is the thing you asked me to translate for the team, with 2 items.'
        out = 'בטח, הנה הדבר שביקשת שאתרגם עבור הצוות, עם 2 פריטים.'
        self.assertIsNone(sd_filters.translation_reason(tr_row(out=out, src=src)))


def solved_tool_row(kind='single', user='What is the weather in Paris?', final='Done, it is sunny there today.',
                    **kw):
    call = {'id': 'call_ab', 'type': 'function', 'function': {'name': 'get_weather', 'arguments': '{"city": "Paris"}'}}
    msgs = [{'role': 'system', 'content': 'sys'}, {'role': 'user', 'content': user},
            {'role': 'assistant', 'content': '', 'tool_calls': [call]},
            {'role': 'tool', 'tool_call_id': 'call_ab', 'content': '{"temp": 21}'},
            {'role': 'assistant', 'content': final}]
    row = {'item_id': 'i1', 'sample': 0, 'kind': kind, 'lang': 'en', 'split': 'train', 'reward': 1.0, 'closed': True,
           'errors': [], 'turn_scores': [1.0], 'reasoning_chars': 0, 'generations': 2, 'messages': msgs,
           'usage': {'completion_tokens': 30}}
    row.update(kw)
    return row


class ToolFilterTests(unittest.TestCase):
    def test_clean_row_passes(self):
        self.assertIsNone(sd_filters.tool_reason(solved_tool_row()))

    def test_every_reason(self):
        he = solved_tool_row(user=HE_PROMPT, final='It is sunny in Paris today and the temperature is pleasant.')
        no_final = solved_tool_row()
        no_final['messages'] = no_final['messages'][:-1]
        cases = {
            'not_solved': solved_tool_row(reward=0.5),
            'empty': solved_tool_row(final=' '),
            'reasoning_leak': solved_tool_row(reasoning_chars=12),
            'markup_leak': solved_tool_row(final='Done <tool_call>{"name": "x"}</tool_call>'),
            'refusal': solved_tool_row(final="I'm sorry, but I can't do that for you."),
            'language_mismatch': he,
        }
        for want, row in cases.items():
            self.assertEqual(sd_filters.tool_reason(row), want, want)
        self.assertEqual(sd_filters.tool_reason(no_final), 'empty')
        for bad in ({'closed': False}, {'errors': ['x']}, {'error': True}):
            self.assertEqual(sd_filters.tool_reason(solved_tool_row(**bad)), 'not_solved', bad)

    def two_turn(self, kind, second):
        row = solved_tool_row(kind=kind)
        row['messages'] += [{'role': 'user', 'content': 'Can you also book me a taxi?'},
                            {'role': 'assistant', 'content': second}]
        row['turn_scores'] = [1.0, 1.0]
        return row

    def test_unsupported_near_miss_turn_may_say_no_tool_does_that(self):
        second = "I'm sorry, but I can't book taxis with the tools I have. Would you like a written checklist?"
        second = second.replace('?', '.')  # a flat statement, which the refusal heuristic would otherwise drop
        self.assertIsNone(sd_filters.tool_reason(self.two_turn('unsupported', second)))
        self.assertEqual(sd_filters.tool_reason(self.two_turn('single', second)), 'refusal')
        flat = solved_tool_row(kind='unsupported', final="I'm sorry, but I can't do that for you.")
        self.assertEqual(sd_filters.tool_reason(flat), 'refusal', 'a refusal after a real tool call is still one')

    def test_transcript_key_ignores_ids_only(self):
        a = solved_tool_row()
        b = copy.deepcopy(a)
        for m in b['messages']:
            if m.get('tool_calls'):
                m['tool_calls'][0]['id'] = 'call_zz'
            if m['role'] == 'tool':
                m['tool_call_id'] = 'call_zz'
        self.assertEqual(sd_filters.transcript_key(a['messages']), sd_filters.transcript_key(b['messages']))
        c = solved_tool_row(final='Done, and the answer is sunny and twenty one degrees.')
        self.assertNotEqual(sd_filters.transcript_key(a['messages']), sd_filters.transcript_key(c['messages']))

    def test_sft_messages_are_template_ready(self):
        out = sd_filters.sft_tool_messages(solved_tool_row()['messages'])
        call = out[2]['tool_calls'][0]
        self.assertEqual(call['function']['arguments'], {'city': 'Paris'}, 'the HF template renders a dict')
        self.assertEqual((call['id'], out[3]['tool_call_id']), ('call_0', 'call_0'))
        self.assertEqual(out[2]['content'], '')
        self.assertTrue(all(isinstance(m['content'], str) for m in out))

    def test_dedupe_quota_and_report(self):
        items = {'i1': {'tools': []}, 'i2': {'tools': []}}
        rows = [solved_tool_row(sample=0), solved_tool_row(sample=1),  # identical transcript: duplicate
                solved_tool_row(sample=2, final='It is sunny and 21 degrees in Paris right now, enjoy the walk.'),
                solved_tool_row(sample=3, final='Paris is at 21 degrees and sunny, so a light jacket is enough.'),
                solved_tool_row(sample=4, reward=0.0, errors=['TURN'], failure='turn_budget', closed=True),
                solved_tool_row(item_id='i2', sample=0, kind='chain')]
        kept, report = sd_filters.filter_tools(rows, items, lambda tools: list(tools), keep_per_item=2)
        self.assertEqual([k['id'] for k in kept], ['i1#s0', 'i1#s2', 'i2#s0'])
        self.assertEqual(report['dropped'], {'not_solved': 1, 'duplicate': 1, 'over_quota': 1})
        single = report['by_kind']['single']
        self.assertEqual((single['episodes'], single['solved'], single['kept'], single['items']), (5, 4, 2, 1))
        self.assertEqual(single['not_solved_by'], {'turn_budget': 1})
        self.assertEqual(single['solved_rate'], 0.8)
        self.assertEqual(report['by_kind']['chain']['kept_rate'], 1.0)
        every, _ = sd_filters.filter_tools(rows, items, lambda tools: list(tools), keep_per_item=0)
        self.assertEqual(len(every), 4, 'keep_per_item 0 keeps every distinct solved transcript')
        self.assertEqual(kept[0]['source'], 'agentic_pool')
        self.assertIn('tools', kept[0])

    def test_finalize_is_a_rerunnable_pure_function_of_the_ledgers(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_jsonl_atomic(out / 'translations.jsonl', [tr_row(id='t1'), tr_row(id='t2', finish='length')])
            write_jsonl_atomic(out / 'chat_raw.jsonl', [chat_row(id='a'), chat_row(id='b', finish='length'),
                                                         chat_row(id='c', prompt=HE_PROMPT, content=HE_ANSWER, lang='he')])
            write_jsonl_atomic(out / 'tool_raw.jsonl', [solved_tool_row(sample=0), solved_tool_row(sample=1)])
            args = dict(items_by_id={'i1': {'tools': []}}, template_tools=lambda t: list(t), keep_per_item=2)
            sd_filters.finalize(out, **args)
            names = ('chat_sft.jsonl', 'tool_sft.jsonl', 'spot_read.jsonl', 'filter_report.json')
            first = {n: (out / n).read_bytes() for n in names}
            sd_filters.finalize(out, **args)
            self.assertEqual(first, {n: (out / n).read_bytes() for n in names})
            report = json.loads(first['filter_report.json'])
            self.assertEqual(report['chat']['dropped'], {'truncated': 1})
            self.assertEqual(report['translation']['dropped'], {'truncated': 1})
            self.assertEqual(report['tools']['kept'], 1)
            self.assertEqual({r['spot_source'] for r in read_jsonl(out / 'spot_read.jsonl')},
                             {'translation', 'chat_en', 'chat_he', 'tool'})


# ------------------------------------------------------------------ jsonl
class JsonlTests(unittest.TestCase):
    def test_torn_tail_is_cut_and_appending_continues_on_a_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'l.jsonl'
            with JsonlSink(path) as sink:
                sink.write({'id': 1})
                sink.write({'id': 2, 'text': 'שלום'})
            with open(path, 'ab') as f:
                f.write(b'{"id": 3, "te')
            self.assertEqual(len(read_jsonl(path)), 2)
            with JsonlSink(path) as sink:
                self.assertEqual(sink.repaired, len(b'{"id": 3, "te'))
                sink.write({'id': 4})
            self.assertEqual([r['id'] for r in read_jsonl(path)], [1, 2, 4])
            self.assertEqual(repair_jsonl(path), 0)
            self.assertEqual(repair_jsonl(Path(tmp) / 'missing.jsonl'), 0)

    def test_unreadable_middle_lines_are_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'l.jsonl'
            path.write_text('{"id": 1}\nnot json\n\n{"id": 2}\n', encoding='utf-8')
            self.assertEqual([r['id'] for r in read_jsonl(path)], [1, 2])

    def test_atomic_write_leaves_no_temp_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'o.jsonl'
            write_jsonl_atomic(path, [{'a': 'ב'}])
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ['o.jsonl'])
            self.assertEqual(path.read_text(encoding='utf-8'), '{"a":"ב"}\n')


# ------------------------------------------------------------------ client against the fake server
def quick_client(server, retries=3):
    return Client(server.url, max_retries=retries, timeout=30.0, sleep=lambda s: None)


class ClientTests(unittest.TestCase):
    sampling = SAMPLING['vendor_nonthinking']

    def chat(self, client, prompt='Name a color.'):
        return client.chat([{'role': 'user', 'content': prompt}], max_tokens=64, sampling=self.sampling, seed=7)

    def test_request_shape_is_think_off_with_the_vendor_arm(self):
        with FakeServer() as server:
            client = quick_client(server)
            self.assertEqual(client.discover_model(), MODEL)
            gen = self.chat(client)
            self.assertTrue(gen.content)
            self.assertEqual(server.stats()['n_violations'], 0, server.stats()['violations'])

    def test_the_server_side_shape_check_can_fail(self):
        """The red arm for every dry-run 'zero violations' claim: a wrong arm and a missing think-off are seen."""
        with FakeServer() as server:
            client = quick_client(server)
            client.chat([{'role': 'user', 'content': 'hi there'}], max_tokens=8, sampling={'temperature': 1.0}, seed=1)
            with mock.patch('sd_common.THINK_OFF', {'enable_thinking': True}):
                self.chat(client)
            violations = ' '.join(server.stats()['violations'])
            self.assertIn('sampling temperature=1.0', violations)
            self.assertIn('chat_template_kwargs', violations)

    def test_503_then_success_is_retried(self):
        with FakeServer(faults=Faults(every_503=2)) as server:
            client = quick_client(server)
            self.chat(client)
            self.chat(client)
            self.assertEqual((client.stats['ok'], client.stats['retries']), (2, 1))

    def test_garbled_body_is_retried(self):
        with FakeServer(faults=Faults(every_garbled=2)) as server:
            client = quick_client(server)
            self.chat(client)
            self.chat(client)
            self.assertEqual((client.stats['ok'], client.stats['retries']), (2, 1))

    def test_dropped_keepalive_connection_reconnects_without_a_retry(self):
        with FakeServer(faults=Faults(every_drop=2)) as server:
            client = quick_client(server)
            self.chat(client)
            self.chat(client)
            self.assertEqual(client.stats['reconnects'], 1)
            self.assertEqual(client.stats['retries'], 0)

    def test_exhaustion_is_retryable_and_counted(self):
        with FakeServer(faults=Faults(every_429=1)) as server:
            client = quick_client(server, retries=2)
            with self.assertRaises(Exhausted) as ctx:
                self.chat(client)
            self.assertTrue(ctx.exception.retryable)
            self.assertEqual((client.stats['retries'], client.stats['exhausted'], client.stats['requests']), (2, 1, 3))

    def test_a_400_is_not_retried(self):
        with FakeServer() as server:
            client = quick_client(server)
            with self.assertRaises(ApiError) as ctx:
                self.chat(client, 'MODE_400 please')
            self.assertFalse(ctx.exception.retryable)
            self.assertEqual((ctx.exception.status, client.stats['requests'], client.stats['http_error_400']), (400, 1, 1))

    def test_a_body_that_is_not_utf8_is_retried_not_a_crash(self):
        """json.loads(bytes) raises UnicodeDecodeError, a ValueError but not a JSONDecodeError: it once escaped."""
        with FakeServer(faults=Faults(every_bad_utf8=2)) as server:
            client = quick_client(server)
            self.chat(client)
            self.chat(client)
            self.assertEqual((client.stats['ok'], client.stats['retries']), (2, 1))
            self.assertEqual(server.stats()['counts']['fault_bad_utf8'], 1)

    def test_a_lone_surrogate_is_retried_because_no_ledger_can_store_it(self):
        with FakeServer(faults=Faults(every_surrogate=2)) as server:
            client = quick_client(server)
            self.chat(client)
            gen = self.chat(client)
            self.assertEqual((client.stats['ok'], client.stats['retries']), (2, 1))
            gen.content.encode('utf-8')
            self.assertEqual(server.stats()['counts']['fault_surrogate'], 1)

    def test_model_discovery_survives_a_non_utf8_body(self):
        client = Client('http://127.0.0.1:9', max_retries=2, sleep=lambda s: None)
        good = json.dumps({'data': [{'id': 'm1'}]}).encode()
        with mock.patch.object(client, '_request', side_effect=[(200, b'{"data": [{"id": "\xff"}]}'), (200, good)]):
            self.assertEqual(client.discover_model(), 'm1')
        self.assertEqual(client.stats['retries'], 1)

    def test_a_dead_server_exhausts_instead_of_hanging(self):
        with FakeServer() as server:
            server.set_dead(True)
            client = quick_client(server, retries=1)
            with self.assertRaises(Exhausted):
                self.chat(client)

    def test_sampling_profiles(self):
        self.assertEqual(sampling_profile('vendor_nonthinking'), {
            'temperature': 0.7, 'top_p': 0.8, 'top_k': 20, 'min_p': 0.0, 'presence_penalty': 1.5,
            'repetition_penalty': 1.0})
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / 'generation_config.json'
            cfg.write_text(json.dumps({'temperature': 1.0, 'top_p': 0.95, 'top_k': 20}))
            arm = sampling_profile('generation_config', cfg)
            self.assertEqual((arm['temperature'], arm['top_p'], arm['presence_penalty']), (1.0, 0.95, 0.0))
        ap = argparse.ArgumentParser()
        add_client_args(ap)
        self.assertEqual(ap.parse_args([]).sampling, 'vendor_nonthinking')
        self.assertEqual(THINK_OFF, {'enable_thinking': False})
        self.assertNotIn('temperature', {k for k, v in SAMPLING['vendor_nonthinking'].items() if v == 0}, 'never greedy')


# ------------------------------------------------------------------ phases
class ScriptedClient:
    """Client.chat without HTTP: fn(messages) -> (content, finish)."""
    model = 'scripted'

    def __init__(self, fn):
        self.fn, self.calls, self.stats = fn, [], Counter()

    def chat(self, messages, *, max_tokens, sampling, tools=None, seed=None):
        self.calls.append(messages)
        content, finish = self.fn(messages)
        return Generation(content, '', [], finish, {'completion_tokens': 5, 'prompt_tokens': 5})


def parse(module, argv):
    ap = argparse.ArgumentParser()
    module.add_args(ap)
    return ap.parse_args(argv)


def letters(n):
    return ''.join(chr(97 + (n // 26 ** k) % 26) for k in range(4))


class TranslatePhaseTests(unittest.TestCase):
    def prompts(self, n):
        return [{'id': f'p{i:04d}', 'prompt': f'Please explain the topic called {letters(i)} in simple terms, '
                                              f'and add one example.', 'lang': 'en', 'source': 'dolly',
                 'license': 'x', 'category': ('open_qa', 'brainstorming', 'summarization')[i % 3]} for i in range(n)]

    @staticmethod
    def translator(messages):
        src = messages[1]['content'][len('<message>\n'):-len('\n</message>')]
        if seed_for('flaky', src) % 10 < 3:
            return src, 'stop'  # the model echoed the English: not_hebrew
        return 'הסבר קצר על הנושא שביקשת, בשפה פשוטה וברורה מאוד, ועוד דוגמה אחת.', 'stop'

    def test_waves_until_the_target_then_resume_without_new_calls(self):
        waves = []
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(chat_gen, 'log', lambda *p: waves.append(p)):
            args = parse(chat_gen, ['--out-dir', tmp, '--n-translate', '40', '--concurrency', '4',
                                    '--progress-every', '0', '--abort-after', '50'])
            client = ScriptedClient(self.translator)
            prompts = self.prompts(300)
            res = chat_gen.translate_phase(client, args, prompts, 40)
            self.assertEqual((res['ok'], res['shortfall']), (40, 0), res)
            self.assertGreaterEqual(res['surplus'], 0)
            self.assertEqual(len(chat_gen.chosen_translations(tmp, prompts, args)), 40, 'exactly the target is answered')
            self.assertGreaterEqual(len([w for w in waves if str(w[0]).startswith('translate wave')]), 2, waves)
            ids = [r['id'] for r in read_jsonl(Path(tmp) / 'translations.jsonl')]
            self.assertEqual(len(ids), len(set(ids)), 'no prompt is sent twice')
            sent = len(client.calls)
            again = chat_gen.translate_phase(client, args, prompts, 40)
            self.assertEqual((again['ok'], len(client.calls)), (40, sent), 'a finished ledger needs no new calls')
            # the candidates are English, deterministic, and stratified over the categories
            order = chat_gen.translation_candidates(prompts, 40, args.seed)
            self.assertEqual([r['id'] for r in order], [r['id'] for r in chat_gen.translation_candidates(prompts, 40, args.seed)])
            self.assertEqual(len(order), 80, 'twice the target: a validation pass rate under 50% is survivable')
            self.assertEqual(len({r['category'] for r in order[:40]}), 3)

    def test_shortfall_is_reported_when_candidates_run_out(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(chat_gen, 'log', lambda *p: None):
            args = parse(chat_gen, ['--out-dir', tmp, '--concurrency', '2', '--progress-every', '0'])
            res = chat_gen.translate_phase(ScriptedClient(lambda m: ('English only', 'stop')), args, self.prompts(30), 40)
            self.assertEqual(res['ok'], 0)
            self.assertEqual(res['shortfall'], 40)
            self.assertEqual(res['attempted'], 30)

    def test_long_and_hebrew_prompts_are_not_translation_candidates(self):
        rows = self.prompts(10) + [{'id': 'long', 'prompt': 'word ' * 600, 'lang': 'en', 'source': 'd', 'license': 'x',
                                    'category': 'open_qa'},
                                   {'id': 'he', 'prompt': HE_PROMPT, 'lang': 'he', 'source': 'o', 'license': 'x',
                                    'category': 'open_qa'}]
        got = {r['id'] for r in chat_gen.translation_candidates(rows, 100, 1)}
        self.assertFalse({'long', 'he'} & got)
        self.assertEqual(len(got), 10)


class Refuses:
    """A client the server refuses with a non-retryable 400 when refuse(messages) says so."""
    model = 'scripted'

    def __init__(self, refuse, inner):
        self.refuse, self.inner, self.calls, self.stats = refuse, inner, [], Counter()

    def chat(self, messages, **kw):
        self.calls.append(messages)
        if self.refuse(messages):
            raise ApiError(400, 'Input is too long', False)
        return self.inner.chat(messages, **kw)


def last_rows(path):
    return {r['id']: r for r in read_jsonl(path)}


class RefusedRowsAreNotFinalTests(unittest.TestCase):
    """A 400 used to be written as a final row and reset the failure streak, so a server that refused every request
    (wrong model name, a parameter it rejects) produced a complete-looking ledger and a clean exit, and a resume
    skipped all of it."""

    def tasks(self, n):
        return [{'id': f't{i:03d}', 'lang': 'en', 'source': 'dolly', 'category': 'open_qa',
                 'prompt': f'Give me a short tip number {letters(i)} about writing clearly.'} for i in range(n)]

    def healthy(self):
        return ScriptedClient(lambda m: ('Keep your sentences short and your words plain.', 'stop'))

    def test_a_server_that_refuses_everything_aborts_the_answer_phase_and_a_resume_retries_all(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(chat_gen, 'log', lambda *p: None):
            args = parse(chat_gen, ['--out-dir', tmp, '--concurrency', '2', '--progress-every', '0',
                                    '--abort-after', '5'])
            with self.assertRaises(Aborted):
                chat_gen.answer_phase(Refuses(lambda m: True, self.healthy()), args, self.tasks(30))
            refused = read_jsonl(Path(tmp) / 'chat_raw.jsonl')
            self.assertGreaterEqual(len(refused), 5)
            self.assertTrue(all(r['error']['status'] == 400 for r in refused))
            res = chat_gen.answer_phase(self.healthy(), args, self.tasks(30))
            self.assertEqual((res['resumed'], res['written']), (0, 30), 'refused rows are not "done"')
            rows = last_rows(Path(tmp) / 'chat_raw.jsonl')
            self.assertEqual(len(rows), 30)
            self.assertFalse([r for r in rows.values() if r.get('error')])

    def test_scattered_refusals_are_recorded_without_aborting_and_retried_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(chat_gen, 'log', lambda *p: None):
            args = parse(chat_gen, ['--out-dir', tmp, '--concurrency', '2', '--progress-every', '0',
                                    '--abort-after', '5'])
            tasks = self.tasks(30)
            refused = {tasks[i]['prompt'] for i in (3, 11, 20)}
            too_long = lambda m: m[0]['content'] in refused  # noqa: E731
            res = chat_gen.answer_phase(Refuses(too_long, self.healthy()), args, tasks)
            self.assertEqual((res['written'], res['api_error']), (27, 3))
            again = self.healthy()
            res = chat_gen.answer_phase(again, args, tasks)
            self.assertEqual((res['resumed'], res['written'], len(again.calls)), (27, 3, 3))
            self.assertFalse([r for r in last_rows(Path(tmp) / 'chat_raw.jsonl').values() if r.get('error')])

    def test_a_server_that_refuses_every_translation_aborts_and_a_resume_retries_them(self):
        prompts = TranslatePhaseTests().prompts(40)
        good = ScriptedClient(TranslatePhaseTests.translator)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(chat_gen, 'log', lambda *p: None):
            args = parse(chat_gen, ['--out-dir', tmp, '--concurrency', '2', '--progress-every', '0',
                                    '--abort-after', '4'])
            with self.assertRaises(Aborted):
                chat_gen.translate_phase(Refuses(lambda m: True, good), args, prompts, 10)
            self.assertGreaterEqual(len(read_jsonl(Path(tmp) / 'translations.jsonl')), 4)
            res = chat_gen.translate_phase(good, args, prompts, 10)
            self.assertEqual((res['ok'], res['shortfall']), (10, 0), res)
            self.assertGreater(len(good.calls), 0)

    @unittest.skipUnless(HAVE_LANE, 'needs the Hebrew lane (SELFDISTILL_LANE_DIR with agentic_env.py)')
    def test_a_server_that_refuses_every_episode_aborts_the_tool_phase_and_a_resume_runs_them_again(self):
        items = pool()[2]
        argv = ['--out-dir', '', '--n-samples', '2', '--item-limit-per-kind', '1', '--concurrency', '2',
                '--progress-every', '0', '--abort-after', '5']
        with tempfile.TemporaryDirectory() as tmp, FakeServer(items) as server, \
                mock.patch('sd_filters.log', lambda *p: None), mock.patch.object(tool_gen, 'log', lambda *p: None):
            argv[1] = tmp
            refuse_all = quick_client(server)
            refuse_all.chat = mock.Mock(side_effect=ApiError(400, 'model not found', False))
            summary = tool_gen.run(parse(tool_gen, argv), client=refuse_all)
            self.assertIn('consecutive failures', summary['aborted'])
            path = Path(tmp) / 'tool_raw.jsonl'
            refused = read_jsonl(path)
            self.assertGreaterEqual(len(refused), 5)
            self.assertTrue(all(r['failure'] == 'http_400' and r['generations'] == 0 for r in refused))
            summary = tool_gen.run(parse(tool_gen, argv), client=quick_client(server))
            self.assertEqual((summary['resumed'], summary['written'], summary['refused']), (0, 18, 0), summary)
            latest = {}
            for r in read_jsonl(path):
                latest[(r['item_id'], r['sample'])] = r
            self.assertEqual(len(latest), 18)
            self.assertFalse([r for r in latest.values() if r['failure'] == 'http_400'])
            self.assertTrue(read_jsonl(Path(tmp) / 'tool_sft.jsonl'), 'the filter reads the last row of each episode')


class ManifestTests(unittest.TestCase):
    def test_a_resume_keeps_the_earlier_manifests_and_warns_about_changed_settings(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch('sd_common.log') as log:
            first = {'model': 'm', 'sampling': {'temperature': 0.7}, 'seed': 1, 'argv': ['a']}
            write_manifest(tmp, 'chat', first)
            self.assertNotIn('history', json.loads((Path(tmp) / 'manifest_chat.json').read_text()))
            write_manifest(tmp, 'chat', {**first, 'argv': ['b']})
            log.assert_not_called()  # only argv changed
            write_manifest(tmp, 'chat', {**first, 'sampling': {'temperature': 1.0}})
            self.assertEqual(log.call_count, 1)
            self.assertIn("'sampling'", log.call_args[0][0])
            got = json.loads((Path(tmp) / 'manifest_chat.json').read_text())
            self.assertEqual([h['argv'] for h in got['history']], [['a'], ['b']])
            self.assertEqual(got['history'][0]['sampling'], {'temperature': 0.7})
            self.assertNotIn('history', got['history'][1])


@unittest.skipUnless(HAVE_LANE, 'needs the Hebrew lane (SELFDISTILL_LANE_DIR with agentic_env.py)')
class ChatEndToEndTests(unittest.TestCase):
    def test_small_run_writes_filtered_chat_rows(self):
        prompts = [{'id': f'p{i}', 'prompt': f'Give me a short tip number {letters(i)} about writing clearly.',
                    'lang': 'en', 'source': 'dolly', 'license': 'x', 'category': 'open_qa'} for i in range(12)]
        prompts.append({'id': 'he1', 'prompt': HE_PROMPT, 'lang': 'he', 'source': 'oasst2', 'license': 'x',
                        'category': 'open_qa'})
        prompts.append({'id': 'empty1', 'prompt': 'MODE_EMPTY tell me a fact about rivers please.', 'lang': 'en',
                        'source': 'dolly', 'license': 'x', 'category': 'open_qa'})
        with tempfile.TemporaryDirectory() as tmp, FakeServer() as server, \
                mock.patch.object(chat_gen, 'log', lambda *p: None), mock.patch('sd_filters.log', lambda *p: None):
            path = Path(tmp) / 'prompts.jsonl'
            write_jsonl_atomic(path, prompts)
            out = Path(tmp) / 'out'
            args = parse(chat_gen, ['--base-url', server.url, '--out-dir', str(out), '--prompts', str(path),
                                    '--n-translate', '5', '--concurrency', '4', '--progress-every', '0'])
            summary = chat_gen.run(args, client=quick_client(server))
            self.assertEqual(summary['translate']['ok'], 5)
            self.assertEqual(summary['answers']['prompts'], 14 + 5)
            report = json.loads((out / 'filter_report.json').read_text())
            self.assertEqual(report['chat']['dropped'], {'empty': 1})
            sft = read_jsonl(out / 'chat_sft.jsonl')
            self.assertEqual(len(sft), 18)
            self.assertEqual(sum(1 for r in sft if r['lang'] == 'he'), 6)
            self.assertEqual(server.stats()['n_violations'], 0, server.stats()['violations'])
            manifest = json.loads((out / 'manifest_chat.json').read_text())
            self.assertEqual((manifest['think'], manifest['sampling']['presence_penalty']), ('off', 1.5))


# ------------------------------------------------------------------ tool episodes, scripted
def gen(content='', calls=None, finish=None, reasoning=''):
    return Generation(content=content, reasoning=reasoning, tool_calls=calls or [],
                      finish_reason=finish or ('tool_calls' if calls else 'stop'),
                      usage={'completion_tokens': 4, 'prompt_tokens': 9})


def call(name, arguments, n=0):
    return {'id': f'call_{n}', 'type': 'function',
            'function': {'name': name, 'arguments': arguments if isinstance(arguments, str) else json.dumps(arguments)}}


@unittest.skipUnless(HAVE_LANE, 'needs the Hebrew lane (SELFDISTILL_LANE_DIR with agentic_env.py)')
class ToolEpisodeTests(unittest.TestCase):
    def setUp(self):
        self.env, self.ag, self.items = pool()
        self.item = next(i for i in self.items if i['kind'] == 'single' and i['split'] == 'train')
        self.want = self.item['turns'][0]['expect']['calls'][0]

    def episode(self, *gens):
        ep = tool_gen.ToolEpisode(self.item, self.env, self.ag)
        for g in gens:
            self.assertIsNotNone(ep.next_request())
            ep.observe(g)
            if ep.done:
                break
        if not ep.done:
            ep.next_request()
        return ep

    def right(self):
        return call(self.want['name'], self.want['arguments'])

    def test_solved(self):
        ep = self.episode(gen(calls=[self.right()]), gen('Done, the request went through.'))
        res = ep.result()
        self.assertEqual((res['reward'], res['failure'], res['closed'], res['errors']), (1.0, None, True, []))
        self.assertEqual([m['role'] for m in ep.messages][2:], ['assistant', 'tool', 'assistant'])
        self.assertNotIn('error', ep.messages[3]['content'])

    def test_wrong_arguments_get_the_canned_error_and_score_zero(self):
        ep = self.episode(gen(calls=[call(self.want['name'], {'wrong': 'value'})]), gen('Done.'))
        self.assertEqual(json.loads(ep.messages[3]['content']), self.ag.ERROR_NO_MATCH)
        self.assertEqual((ep.result()['reward'], ep.result()['failure']), (0.0, 'wrong_calls'))

    def test_unknown_tool_and_non_object_arguments(self):
        ep = self.episode(gen(calls=[call('nonexistent_tool', {})]), gen('Done.'))
        self.assertEqual(json.loads(ep.messages[3]['content']), self.ag.ERROR_UNKNOWN)
        ep = self.episode(gen(calls=[call(self.want['name'], '[1, 2]')]), gen('Done.'))
        self.assertEqual(ep.calls[0]['arguments'], [1, 2])
        self.assertEqual(json.loads(ep.messages[3]['content']), self.ag.ERROR_NO_MATCH)

    def test_failure_classes(self):
        env = self.env
        truncated = self.episode(gen(finish='length'))
        self.assertEqual((truncated.result()['failure'], truncated.result()['closed']), ('truncated', False))
        self.assertEqual(truncated.messages[-1]['role'], 'user', 'a truncated message is never appended')
        broken = self.episode(gen(calls=[call(self.want['name'], '{"city": ')]))
        self.assertEqual(broken.result()['failure'], 'bad_arguments')
        budget = self.episode(*[gen(calls=[self.right()]) for _ in range(self.ag.MAX_STEPS_PER_TURN)])
        self.assertEqual(budget.errors, [env.TURN_BUDGET])
        self.assertEqual(budget.result()['failure'], 'turn_budget')
        refused = tool_gen.ToolEpisode(self.item, env, self.ag)
        refused.next_request()
        refused.http_error()
        self.assertEqual(refused.result()['failure'], 'http_400')
        self.assertIsNone(refused.next_request())

    def test_generation_limit(self):
        item = next(i for i in self.items if len(i['turns']) >= 3)
        ep = tool_gen.ToolEpisode(item, self.env, self.ag)
        for _ in range(self.ag.MAX_GENERATIONS):
            self.assertIsNotNone(ep.next_request())
            ep.done = False
            ep.generations += 1
        self.assertIsNone(ep.next_request())
        self.assertEqual(ep.errors, [self.env.TURN_BUDGET])


@unittest.skipUnless(HAVE_LANE, 'needs the Hebrew lane (SELFDISTILL_LANE_DIR with agentic_env.py)')
class RunEpisodeTests(unittest.TestCase):
    def setUp(self):
        self.env, self.ag, self.items = pool()
        self.item = next(i for i in self.items if i['kind'] == 'single' and i['split'] == 'train')
        self.args = types.SimpleNamespace(seed=3)

    def test_a_retryable_failure_writes_no_row_and_a_400_is_an_outcome(self):
        class Dead:
            def chat(self, *a, **k):
                raise Exhausted(503, 'gone', True)

        class TooLong:
            def chat(self, *a, **k):
                raise ApiError(400, 'context too long', False)

        with self.assertRaises(Exhausted):
            tool_gen.run_episode(Dead(), self.item, 0, self.args, {}, self.env, self.ag)
        row = tool_gen.run_episode(TooLong(), self.item, 0, self.args, {}, self.env, self.ag)
        self.assertEqual((row['failure'], row['generations']), ('http_400', 0))

    def test_select_items_holds_out_the_monitor_split(self):
        args = parse(tool_gen, ['--item-limit-per-kind', '3'])
        chosen, per_kind = tool_gen.select_items(self.items, args)
        self.assertTrue(all(i['split'] == 'train' for i in chosen))
        self.assertEqual(set(per_kind), set(self.ag.KINDS))
        self.assertTrue(all(n <= 3 for n in per_kind.values()))
        with_monitor, _ = tool_gen.select_items(self.items, parse(tool_gen, ['--include-monitor']))
        self.assertEqual(len(with_monitor), len(self.items))
        only, _ = tool_gen.select_items(self.items, parse(tool_gen, ['--kinds', 'chain,single']))
        self.assertEqual({i['kind'] for i in only}, {'chain', 'single'})


@unittest.skipUnless(HAVE_LANE, 'needs the Hebrew lane (SELFDISTILL_LANE_DIR with agentic_env.py)')
class ToolRunEndToEndTests(unittest.TestCase):
    def test_small_run_against_the_fake_server_keeps_only_solved_transcripts(self):
        items = pool()[2]
        args_list = ['--out-dir', '', '--n-samples', '3', '--item-limit-per-kind', '2', '--concurrency', '4',
                     '--progress-every', '0', '--keep-per-item', '2']
        with tempfile.TemporaryDirectory() as tmp, FakeServer(items) as server, \
                mock.patch('sd_filters.log', lambda *p: None), mock.patch.object(tool_gen, 'log', lambda *p: None):
            out = Path(tmp)
            args_list[1] = str(out)
            args = parse(tool_gen, args_list)
            summary = tool_gen.run(args, client=quick_client(server))
            self.assertEqual(summary['episodes_planned'], 18 * 3)
            self.assertEqual(summary['written'], 18 * 3)
            raw = read_jsonl(out / 'tool_raw.jsonl')
            self.assertEqual(len(raw), 54)
            sft = read_jsonl(out / 'tool_sft.jsonl')
            solved = {(r['item_id'], r['sample']) for r in raw if r['reward'] == 1.0 and not r['errors']}
            for row in sft:
                item_id, sample = row['id'].rsplit('#s', 1)
                self.assertIn((item_id, int(sample)), solved)
                self.assertIn(row['messages'][-1]['role'], ('assistant',))
                self.assertTrue(row['messages'][-1]['content'].strip())
                for m in row['messages']:
                    for c in m.get('tool_calls') or []:
                        self.assertIsInstance(c['function']['arguments'], dict)
            self.assertEqual(server.stats()['n_violations'], 0, server.stats()['violations'])
            self.assertEqual(server.stats()['internal_errors'], [])
            # rerunning the same command finds every episode in the ledger and sends nothing
            before = server.stats()['requests']
            again = tool_gen.run(parse(tool_gen, args_list), client=quick_client(server))
            self.assertEqual((again['written'], server.stats()['requests']), (0, before))


@unittest.skipUnless(HAVE_LANE, 'needs the Hebrew lane (SELFDISTILL_LANE_DIR with agentic_env.py)')
@unittest.skipUnless(DEFAULT_TOKENIZER.exists(), 'the Qwen3.8-27B tokenizer is not on this machine')
class RenderTests(unittest.TestCase):
    def test_kept_rows_render_like_the_server_and_carry_labels(self):
        import run
        env, _, items = pool()
        with tempfile.TemporaryDirectory() as tmp, FakeServer(items) as server, \
                mock.patch('sd_filters.log', lambda *p: None), mock.patch.object(tool_gen, 'log', lambda *p: None), \
                mock.patch.object(chat_gen, 'log', lambda *p: None):
            out = Path(tmp)
            prompts = [{'id': f'p{i}', 'prompt': f'Give me a short tip number {letters(i)} about writing clearly.',
                        'lang': 'en', 'source': 'dolly', 'license': 'x', 'category': 'open_qa'} for i in range(6)]
            write_jsonl_atomic(out / 'prompts.jsonl', prompts)
            chat_gen.run(parse(chat_gen, ['--out-dir', str(out), '--prompts', str(out / 'prompts.jsonl'),
                                          '--n-translate', '2', '--concurrency', '4', '--progress-every', '0',
                                          '--no-filter']), client=quick_client(server))
            tool_gen.run(parse(tool_gen, ['--out-dir', str(out), '--n-samples', '2', '--item-limit-per-kind', '2',
                                          '--concurrency', '4', '--progress-every', '0']), client=quick_client(server))
            sd_filters.finalize(out, items_by_id={i['id']: i for i in items}, template_tools=env.template_tools,
                                keep_per_item=2)
            res = run.render_check(out, {i['id']: i for i in items}, env.template_tools, env)
            self.assertEqual(res['status'], 'ok', res)
            self.assertGreater(res['tool_rows'], 0)
            self.assertEqual(res['tool_rows'], res['tool_render_equal'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
