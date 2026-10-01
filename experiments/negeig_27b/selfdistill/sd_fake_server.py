"""A tiny fake OpenAI-compatible server for the self-distillation dry run (CPU only, stdlib only).

It stands in for the SGLang 0.5.20 server that serves the untouched Qwen3.8-27B. It is an ORACLE, not a model:
on a tool request it reads the pool item the request belongs to and answers what a correct model would, then
injects the failure shapes the real model shows (wrong arguments, spread parallel calls, a spurious call, an
empty or English or refusing final answer, a leaked think block, leaked tool-call markup, a truncated reply,
a stalled loop). On a chat request it answers from the prompt text, and a marker word in the prompt (MODE_EMPTY,
MODE_REFUSE, ...) forces one specific defect. On a translation request it swaps every prose word for a Hebrew
pseudo-word and keeps code, URLs and numbers, with MODE_TR_* markers forcing each translation defect.

It also breaks the transport on purpose (503, 429, a dropped connection, a garbled body, a dead server) so the
client's retry, reconnect and resume paths run in the dry run, and it records every request whose shape differs
from what the real server must receive: the think-off switch, the vendor sampling arm, wire-form tool calls.

Nothing here decides what the job produces. test_selfdistill.py and run.py assert on what comes out the far end.
"""
from __future__ import annotations

import argparse
import http.server
import json
import random
import re
import socketserver
import threading
from collections import Counter

from sd_common import SAMPLING, TRANSLATE_SYSTEM, hebrew_share, seed_for

MODEL = 'fake-qwen3.8-27b'

# Per-generation probabilities of each injected failure shape (tool episodes). Episodes are 2 to 8 generations
# long, so these compound to a solved rate of roughly 45 to 65 percent, the band the real model is expected in.
RATES = {
    'wrong_arg': 0.040, 'unknown_tool': 0.012, 'bad_json': 0.012, 'truncated': 0.025, 'reasoning': 0.025,
    'spurious': 0.030, 'empty': 0.030, 'markup': 0.025, 'english': 0.035, 'refuse': 0.030,
}
SPREAD_RATE = 0.12       # a parallel step that should carry several calls sends only one

EN_DONE = [
    'Done. The tool results match your request, so nothing else is needed.',
    'All set. I made the calls you asked for and the results came back clean.',
    'That is taken care of. The results above show the request went through.',
    'Finished. I used the results to complete what you asked and nothing failed.',
    'Complete. Everything you requested is in the tool results above.',
    'Done, and the results confirm it. Tell me if you want anything changed.',
    'It went through. The returned values are exactly the ones you needed.',
    'Your request is handled, and the results look consistent with each other.',
]
HE_DONE = [
    'בוצע. תוצאות הכלים תואמות לבקשה שלך ואין צורך בדבר נוסף.',
    'הכול מוכן. ביצעתי את הקריאות שביקשת והתוצאות חזרו תקינות.',
    'טופל. התוצאות שלמעלה מראות שהבקשה עברה בהצלחה.',
    'סיימתי. השתמשתי בתוצאות כדי להשלים את מה שביקשת ושום דבר לא נכשל.',
    'הושלם. כל מה שביקשת נמצא בתוצאות הכלים שלמעלה.',
    'בוצע, והתוצאות מאשרות זאת. תגיד לי אם תרצה לשנות משהו.',
    'זה עבר. הערכים שחזרו הם בדיוק אלה שהיית צריך.',
    'הבקשה שלך טופלה, והתוצאות נראות עקביות זו עם זו.',
]
EN_ASK = ['Which value should I use? Please send it and I will run the request.',
          'One value is missing from your message. What should I use for it?']
HE_ASK = ['איזה ערך להשתמש? אפשר לשלוח אותו ואבצע את הבקשה.',
          'חסר ערך אחד בהודעה שלך. מה להשתמש בו?']
EN_DIRECT = ['None of my tools covers that, so I am answering directly. A short answer follows, in two sentences.',
             'That request is outside the tools I was given, so I answer it without one, briefly and plainly.']
HE_DIRECT = ['אף אחד מהכלים שלי לא מכסה את זה, ולכן אני עונה ישירות. תשובה קצרה בשני משפטים.',
             'הבקשה הזאת מחוץ לכלים שקיבלתי, ולכן אני עונה עליה בלי כלי, בקצרה ובפשטות.']
EN_REFUSE = "I'm sorry, but I can't help with that request."
HE_REFUSE = 'מצטער, אבל אני לא יכול לעזור בבקשה הזאת.'

CHAT_EN = ['The main point fits in one sentence, and the rest of this answer builds on it.',
           'A concrete example shows it better than a rule, so here is one that you can test yourself.',
           'Check one thing before you rely on it, because the edge case is where most mistakes happen.',
           'If the details differ from what you described, adjust the second step and keep the first.',
           'Two alternatives exist, and the simpler one is usually enough for this kind of request.',
           'Keep the wording short, the structure flat, and the examples close to the claim they support.']
CHAT_HE = ['הנקודה המרכזית נכנסת במשפט אחד, ושאר התשובה נשענת עליה.',
           'דוגמה קונקרטית מסבירה את זה טוב יותר מכלל, ולכן הנה אחת שאפשר לנסות בעצמך.',
           'כדאי לבדוק דבר אחד לפני שסומכים על זה, כי במקרה הקצה קורות רוב הטעויות.',
           'אם הפרטים שונים ממה שתיארת, מתאימים את השלב השני ומשאירים את הראשון.',
           'קיימות שתי חלופות, והפשוטה מביניהן בדרך כלל מספיקה לבקשה מהסוג הזה.',
           'שומרים על ניסוח קצר, על מבנה שטוח ועל דוגמאות צמודות לטענה שהן תומכות בה.']

HE_LETTERS = 'אבגדהוזחטיכלמנסעפצקרשת'
_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")
_SPLIT = re.compile(r'(```.*?```|`[^`\n]*`|https?://\S+)', re.S)


def _is_he(text, fallback=False):
    share = hebrew_share(text)
    return fallback if share is None else share >= 0.5


def _usage(prompt_chars, content):
    return {'prompt_tokens': max(1, prompt_chars // 3), 'completion_tokens': max(1, len(content) // 3),
            'total_tokens': max(2, prompt_chars // 3 + len(content) // 3)}


# ------------------------------------------------------------------ translation
def pseudo_hebrew(word):
    n = max(2, len(word))
    h = seed_for('he', word.lower(), bits=60)
    out = []
    for _ in range(n):
        out.append(HE_LETTERS[h % len(HE_LETTERS)])
        h //= len(HE_LETTERS)
    return ''.join(out)


def fake_translate(src):
    """(text, finish_reason): Hebrew pseudo-translation that keeps code, URLs and numbers, or one forced defect."""
    if 'MODE_TR_EMPTY' in src:
        return '', 'stop'
    if 'MODE_TR_LENGTH' in src:
        return 'תרגום שנקטע באמצע', 'length'
    parts = _SPLIT.split(src)
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 1:  # a code span, a fence or a URL
            out.append('' if 'MODE_TR_CODE' in src and not part.startswith('http') else part)
        else:
            out.append(_WORD.sub(lambda m: pseudo_hebrew(m.group(0)), part))
    text = ''.join(out)
    if 'MODE_TR_ENGLISH' in src:
        text = src
    elif 'MODE_TR_SHORT' in src:
        text = 'שלום עולם'
    elif 'MODE_TR_DROPNUM' in src:
        text = re.sub(r'\d+', '', text)
    elif 'MODE_TR_REFUSE' in src:
        text = "I'm sorry, but I can't translate that."
    elif 'MODE_TR_PREAMBLE' in src:
        text = 'הנה התרגום:\n' + text
    elif 'MODE_TR_MARKUP' in src:
        text = text + ' <|im_end|>'
    elif 'MODE_TR_CJK' in src:
        text = text + ' 翻訳'
    return text, 'stop'


# ------------------------------------------------------------------ chat
def chat_answer(prompt):
    """(content, reasoning, finish_reason, tool_call_leak). A pure function of the prompt text, so two requests
    with the same prompt get the same answer (what makes the exact-duplicate filter testable)."""
    if 'MODE_EMPTY' in prompt:
        return '', '', 'stop', False
    he = _is_he(prompt) and 'MODE_EN_REPLY' not in prompt
    if 'MODE_REFUSE' in prompt:
        return (HE_REFUSE if he else EN_REFUSE), '', 'stop', False
    table = CHAT_HE if he else CHAT_EN
    h = seed_for('chat', prompt, bits=40)
    n = 3 + h % 4
    sentences = [table[(h >> (3 * k)) % len(table)] for k in range(n)]
    sentences = list(dict.fromkeys(sentences))
    topic = ' '.join(re.sub(r'MODE_\w+', '', prompt).split()[:5])
    intro = f'בקשר ל: {topic}.' if he else f'About: {topic}.'
    body = ' '.join([intro] + sentences)
    if 'MODE_LENGTH' in prompt:
        return body[:len(body) // 2], '', 'length', False
    if 'MODE_LOOP' in prompt:
        return ' '.join([intro] + [table[0]] * 40), '', 'stop', False
    if 'MODE_MARKUP' in prompt:
        return body + '\n<tool_call>{"name": "x"}</tool_call>', '', 'stop', False
    if 'MODE_THINK' in prompt:
        return body, 'The user asks something, let me plan the answer first.', 'stop', False
    if 'MODE_TOOLCALL' in prompt:
        return body, '', 'tool_calls', True
    return body, '', 'stop', False


# ------------------------------------------------------------------ tool episodes
class ItemIndex:
    """Finds the pool item a tool request belongs to by system text, tool list and first user message."""

    def __init__(self, items):
        self.by_key = {}
        for it in items:
            self.by_key.setdefault(self.key(it['system'], it['tools'], it['turns'][0]['user']), []).append(it)

    @staticmethod
    def key(system, tools, first_user):
        return (system, json.dumps(tools, sort_keys=True, ensure_ascii=False), first_user)

    def find(self, messages, tools):
        system = messages[0]['content'] if messages and messages[0]['role'] == 'system' else ''
        users = [m['content'] for m in messages if m['role'] == 'user']
        if not users:
            return None
        for it in self.by_key.get(self.key(system, tools, users[0]), []):
            if all(i < len(it['turns']) and it['turns'][i]['user'] == u for i, u in enumerate(users)):
                return it
        return None


def _final_text(item, turn, user, rnd, mode):
    he = _is_he(user, item['lang'] == 'he')
    if mode == 'english':
        he = False
    if mode == 'refuse':
        return HE_REFUSE if he else EN_REFUSE
    if turn['expect'].get('no_call'):
        ask = item['kind'] == 'missing_param'
        table = (HE_ASK if he else EN_ASK) if ask else (HE_DIRECT if he else EN_DIRECT)
    else:
        table = HE_DONE if he else EN_DONE
    return rnd.choice(table)


def _call(name, arguments, rnd):
    return {'id': 'call_' + f'{rnd.getrandbits(64):016x}', 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(arguments, ensure_ascii=False)}}


def _perturb_args(arguments):
    out = dict(arguments)
    for k, v in out.items():
        if isinstance(v, str):
            out[k] = v + 'x'
            return out
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = v + 7
            return out
    out['extra'] = 1
    return out


def plan_tool_response(item, messages, rnd, rates=RATES, stall=False, drift=False):
    """What the oracle model says next. Returns {'content','reasoning','tool_calls','finish','mode'}.

    Progress is read from the messages alone: the turn is the count of user messages and the calls made so far
    in it are the tool calls of the assistant messages after the last user message. A sequential turn issues one
    call per response (a later call needs an earlier result), a parallel turn all of its remaining calls."""
    users = [m for m in messages if m['role'] == 'user']
    turn_idx = len(users) - 1
    turn = item['turns'][turn_idx]
    expect = turn['expect']
    last_user = max(i for i, m in enumerate(messages) if m['role'] == 'user')
    after = [m for m in messages[last_user + 1:] if m['role'] == 'assistant']
    made = sum(len(m.get('tool_calls') or []) for m in after)
    calls = expect.get('calls') or []
    u = rnd.random()

    def pick(table):
        acc = 0.0
        for mode, p in table:
            acc += p
            if u < acc:
                return mode
        return None

    plan = {'content': '', 'reasoning': '', 'tool_calls': [], 'finish': 'stop', 'mode': None}
    stalled = bool(stall and turn_idx == 0 and calls)  # a sticky item: the same first call, every step
    if stalled or made < len(calls):
        if stalled:
            batch = [calls[0]]
        elif expect.get('parallel'):
            batch = calls[made:]
        else:
            batch = [calls[made]]
        mode = None if stalled else pick([(m, rates[m]) for m in
                                          ('wrong_arg', 'unknown_tool', 'bad_json', 'truncated', 'reasoning')])
        if mode is None and expect.get('parallel') and len(batch) > 1 and rnd.random() < SPREAD_RATE:
            batch, mode = batch[:1], 'spread'
        if mode == 'truncated':
            return {**plan, 'content': '', 'finish': 'length', 'mode': mode}
        out = []
        for k, c in enumerate(batch):
            name, args = c['name'], c['arguments']
            if mode == 'wrong_arg' and k == 0:
                args = _perturb_args(args)
            if mode == 'unknown_tool' and k == 0:
                name = 'nonexistent_tool'
            call = _call(name, args, rnd)
            if mode == 'bad_json' and k == 0:
                call['function']['arguments'] = '{"broken": '
            out.append(call)
        plan.update(tool_calls=out, finish='tool_calls', mode=mode)
        if mode == 'reasoning':
            plan['reasoning'] = 'I should look at the tools first and then decide which one to call.'
        return plan
    mode = pick([(m, rates[m]) for m in ('spurious', 'empty', 'markup', 'english', 'refuse', 'truncated',
                                         'reasoning')])
    if drift:  # a sticky item: every final reply is in English whatever language the request was in
        mode = 'english'
    user = users[-1]['content']
    if mode == 'spurious':
        first = item['tools'][0]['function']
        args = {k: 'x' for k in (first['parameters'].get('properties') or {})}
        plan.update(tool_calls=[_call(first['name'], args, rnd)], finish='tool_calls', mode=mode)
        return plan
    if mode == 'truncated':
        return {**plan, 'content': _final_text(item, turn, user, rnd, None)[:12], 'finish': 'length', 'mode': mode}
    if mode == 'empty':
        return {**plan, 'mode': mode}
    text = _final_text(item, turn, user, rnd, mode)
    if mode == 'markup':
        text += ' <tool_call>{"name": "x"}</tool_call>'
    plan.update(content=text, mode=mode)
    if mode == 'reasoning':
        plan['reasoning'] = 'The results are in, so I can answer now.'
    return plan


# ------------------------------------------------------------------ the server
class Faults:
    """Transport faults, by global request count. 0 disables a kind."""

    def __init__(self, every_503=0, every_429=0, every_drop=0, every_garbled=0, every_bad_utf8=0, every_surrogate=0):
        self.every_503, self.every_429, self.every_drop, self.every_garbled = (
            every_503, every_429, every_drop, every_garbled)
        self.every_bad_utf8, self.every_surrogate = every_bad_utf8, every_surrogate


class _Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 256


class FakeServer:
    def __init__(self, items=None, model=MODEL, expected_sampling=None, faults=None, http400_item_ids=(),
                 stall_item_ids=(), drift_item_ids=(), port=0):
        self.model, self.port = model, port
        self.expected = dict(expected_sampling or SAMPLING['vendor_nonthinking'])
        self.faults = faults or Faults()
        self.index = ItemIndex(items or [])
        self.http400 = set(http400_item_ids)
        self.stall = set(stall_item_ids)
        self.drift = set(drift_item_ids)
        self.counts = Counter()
        self.violations = []
        self.internal_errors = []
        self.dead = False
        self._die_after = None
        self._lock = threading.Lock()
        self._n = 0
        self._chat_n = 0
        self._httpd = None
        self._thread = None

    # -- control
    def set_dead(self, dead=True):
        self.dead = dead

    def kill_after(self, chat_requests):
        """Go dead once this many chat requests were handled (counted from now on)."""
        with self._lock:
            self._die_after = self._chat_n + chat_requests

    def revive(self):
        with self._lock:
            self.dead, self._die_after = False, None

    def start(self):
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *a):
                pass

            def _send(self, status, obj=None, raw=None):
                body = raw if raw is not None else json.dumps(obj, ensure_ascii=False).encode('utf-8')
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.rstrip('/') == '/health':
                    return self._send(200, {'status': 'ok'})
                if self.path.rstrip('/') == '/v1/models':
                    if outer.dead:
                        self.close_connection = True
                        return
                    return self._send(200, {'object': 'list', 'data': [{'id': outer.model, 'object': 'model'}]})
                if self.path.rstrip('/') == '/stats':
                    return self._send(200, outer.stats())
                self._send(404, {'error': 'not found'})

            def do_POST(self):
                n = int(self.headers.get('Content-Length') or 0)
                raw = self.rfile.read(n)
                if self.path.rstrip('/') != '/v1/chat/completions':
                    return self._send(404, {'error': 'not found'})
                outer._chat(self, raw)

        self._httpd = _Server(('127.0.0.1', self.port), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, kwargs={'poll_interval': 0.05},
                                        daemon=True)
        self._thread.start()
        return self

    @property
    def url(self):
        return f'http://127.0.0.1:{self._httpd.server_address[1]}'

    def stop(self):
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    def stats(self):
        with self._lock:
            return {'requests': self._n, 'counts': dict(self.counts), 'violations': list(self.violations[:50]),
                    'n_violations': len(self.violations), 'internal_errors': list(self.internal_errors[:10])}

    # -- request handling
    def _violate(self, what):
        with self._lock:
            self.violations.append(what)

    def _transport_fault(self, handler):
        """True when the request was consumed by an injected transport fault."""
        with self._lock:
            self._n += 1
            n = self._n
            if self._die_after is not None and self._chat_n >= self._die_after:
                self.dead = True
        f = self.faults
        if self.dead:
            self.counts['fault_dead'] += 1
            handler.close_connection = True
            return True
        if f.every_drop and n % f.every_drop == 0:
            self.counts['fault_drop'] += 1
            handler.close_connection = True
            return True
        if f.every_503 and n % f.every_503 == 0:
            self.counts['fault_503'] += 1
            handler._send(503, {'error': 'overloaded'})
            return True
        if f.every_429 and n % f.every_429 == 0:
            self.counts['fault_429'] += 1
            handler._send(429, {'error': 'rate limited'})
            return True
        if f.every_garbled and n % f.every_garbled == 0:
            self.counts['fault_garbled'] += 1
            handler._send(200, raw=b'{"choices": [')
            return True
        if f.every_bad_utf8 and n % f.every_bad_utf8 == 0:
            self.counts['fault_bad_utf8'] += 1  # a 200 whose body is not utf-8 (json.loads raises UnicodeDecodeError)
            handler._send(200, raw=b'{"choices": [{"message": {"content": "\xff\xfe"}, "finish_reason": "stop"}]}')
            return True
        if f.every_surrogate and n % f.every_surrogate == 0:
            self.counts['fault_surrogate'] += 1  # valid JSON holding a lone surrogate, which no utf-8 ledger can store
            handler._send(200, raw=b'{"choices": [{"message": {"content": "caf\\ud800"}, "finish_reason": "stop"}]}')
            return True
        return False

    def _chat(self, handler, raw):
        if self._transport_fault(handler):
            return
        try:
            body = json.loads(raw)
            status, obj = self._answer(body)
        except json.JSONDecodeError:
            self._violate('request body is not JSON')
            return handler._send(400, {'error': 'bad json'})
        except Exception as error:  # noqa: BLE001, a bug in the fake must be loud in the dry run
            with self._lock:
                self.internal_errors.append(repr(error))
            return handler._send(500, {'error': repr(error)})
        with self._lock:
            self._chat_n += 1
        handler._send(status, obj)

    def _check_shape(self, body, kind):
        v = []
        if body.get('model') != self.model:
            v.append(f"model {body.get('model')!r} is not the served name")
        if body.get('chat_template_kwargs') != {'enable_thinking': False}:
            v.append(f"chat_template_kwargs {body.get('chat_template_kwargs')!r}")
        if body.get('stream') is not False:
            v.append('stream is not false')
        for key, want in self.expected.items():
            if body.get(key) != want:
                v.append(f'sampling {key}={body.get(key)!r}, want {want!r}')
        if not isinstance(body.get('max_tokens'), int) or body['max_tokens'] <= 0:
            v.append(f"max_tokens {body.get('max_tokens')!r}")
        if not isinstance(body.get('seed'), int):
            v.append(f"seed {body.get('seed')!r}")
        msgs = body.get('messages') or []
        if kind == 'chat' and not (len(msgs) == 1 and msgs[0]['role'] == 'user' and 'tools' not in body):
            v.append('a chat request must be one user message and no tools')
        if kind == 'translate' and not (
                len(msgs) == 2 and msgs[0]['content'] == TRANSLATE_SYSTEM and 'tools' not in body
                and msgs[1]['content'].startswith('<message>\n') and msgs[1]['content'].endswith('\n</message>')):
            v.append('a translation request must be the translator system text plus a <message> block')
        if kind == 'tool':
            if not body.get('tools'):
                v.append('a tool request must carry tools')
            ids = set()
            for m in msgs:
                if m['role'] == 'assistant':
                    for c in m.get('tool_calls') or []:
                        if not isinstance(c.get('id'), str) or not c['id']:
                            v.append('assistant tool call without an id')
                        if not isinstance(c['function'].get('arguments'), str):
                            v.append('assistant tool call arguments must be a JSON string on the wire')
                        ids.add(c.get('id'))
                    if m.get('content') is None:
                        v.append('assistant content must be a string')
                elif m['role'] == 'tool':
                    if m.get('tool_call_id') not in ids:
                        v.append('tool message answers no earlier tool call')
                    if not isinstance(m.get('content'), str):
                        v.append('tool content must be a string')
        for what in v:
            self._violate(f'{kind}: {what}')

    def _answer(self, body):
        msgs = body.get('messages') or []
        if body.get('tools'):
            self.counts['tool'] += 1
            self._check_shape(body, 'tool')
            item = self.index.find(msgs, body['tools'])
            if item is None:
                self._violate('tool: request matches no pool item')
                return 400, {'error': 'unknown item'}
            for m in msgs:  # the served template is strict: tool-call arguments must be a JSON object, else HTTP 400
                for c in (m.get('tool_calls') or []) if m['role'] == 'assistant' else []:
                    try:
                        ok = isinstance(json.loads(c['function']['arguments']), dict)
                    except (ValueError, TypeError):
                        ok = False
                    if not ok:
                        self.counts['http400_arguments'] += 1
                        return 400, {'error': {'message': 'arguments must be a JSON object', 'code': 400}}
            if item['id'] in self.http400:
                self.counts['http400'] += 1
                return 400, {'error': {'message': 'Input is too long for the context window', 'code': 400}}
            rnd = random.Random(body['seed'])
            plan = plan_tool_response(item, msgs, rnd, stall=item['id'] in self.stall,
                                      drift=item['id'] in self.drift)
            self.counts['mode_' + str(plan['mode'])] += 1
            prompt_chars = sum(len(json.dumps(m, ensure_ascii=False)) for m in msgs)
            return 200, self._response(plan['content'], plan['reasoning'], plan['tool_calls'], plan['finish'],
                                       prompt_chars)
        if msgs and msgs[0]['role'] == 'system':
            self.counts['translate'] += 1
            self._check_shape(body, 'translate')
            src = msgs[1]['content'][len('<message>\n'):-len('\n</message>')]
            if 'MODE_TR_400' in src:
                self.counts['http400'] += 1
                return 400, {'error': 'Input is too long'}
            text, finish = fake_translate(src)
            return 200, self._response(text, '', [], finish, len(src))
        self.counts['chat'] += 1
        self._check_shape(body, 'chat')
        prompt = msgs[0]['content']
        if 'MODE_400' in prompt:
            self.counts['http400'] += 1
            return 400, {'error': 'Input is too long'}
        content, reasoning, finish, leak = chat_answer(prompt)
        calls = [_call('get_weather', {'city': 'x'}, random.Random(body['seed']))] if leak else []
        return 200, self._response(content, reasoning, calls, finish, len(prompt))

    def _response(self, content, reasoning, tool_calls, finish, prompt_chars):
        message = {'role': 'assistant', 'content': content}  # SGLang's non-streaming path returns '' next to tool calls
        if reasoning:
            message['reasoning_content'] = reasoning
        if tool_calls:
            message['tool_calls'] = tool_calls
        return {'id': 'chatcmpl-fake', 'object': 'chat.completion', 'model': self.model,
                'choices': [{'index': 0, 'message': message, 'finish_reason': finish}],
                'usage': _usage(prompt_chars, content + json.dumps(tool_calls))}


def main():
    ap = argparse.ArgumentParser(description='fake OpenAI-compatible server for the selfdistill dry run')
    ap.add_argument('--port', type=int, default=0)
    ap.add_argument('--with-pool', action='store_true', help='index the agentic pool so tool requests are answered')
    args = ap.parse_args()
    items = None
    if args.with_pool:
        from sd_common import load_pool
        items = load_pool()[2]
    srv = FakeServer(items, port=args.port)
    srv.start()
    print(srv.url, flush=True)
    threading.Event().wait()


if __name__ == '__main__':
    main()
