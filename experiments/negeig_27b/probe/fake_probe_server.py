"""Fake OpenAI-compatible server for the probe's CPU tests (no GPU, no SGLang, stdlib plus probe_common).

It extends selfdistill's FakeServer (the HTTP server, the transport faults: 503, 429, dropped connections, garbled
bodies, a dead server, kill_after/revive) and replaces what it answers. It is an ORACLE: on a tool request it finds the
agentic_env item the conversation belongs to and makes the expected calls (the first call alone, then the rest in one
response, as a model that reads before it acts; all at once on a parallel turn), on a plain request it finds the S1
session and answers the gold value. A script {(item id or S1 session key, turn index): defect} injects the failure
shapes the scoring must see:

  on-call  wrong_arg (first call perturbed), omit (last expected call never made), extra (a duplicate call),
           spurious (a call on a no-call turn), loop (the same first call every step: the step limit), bad_json,
           truncated (finish "length": in thinking mode a long reasoning and no content)
  S1       wrong, format (gold + "."), case (upper case), verbose ("The answer is ..."), empty, truncated

Thinking on, every response carries reasoning_content: think_words(key, turn, step) words, scaled by think_scale (so
two fakes can play a model that thinks less). usage carries prompt_tokens, completion_tokens and SGLang 0.5.20's
reasoning_tokens (the reasoning plus its </think>, or the whole completion when truncated inside the think block),
counted with the counter it is given (the real tokenizer in the token tests). usage_reasoning=False drops the
reasoning_tokens field, no_usage drops usage entirely. ctx_limit=C refuses like SGLang 0.5.20 (HTTP 400, its two
messages verbatim) a request whose prompt (the counter over the message contents) reaches C, or whose prompt plus
max_tokens exceeds it.

It also checks every request against the mode it was started for and records violations: the served name, stream,
the sampling arm of the mode, chat_template_kwargs, max_tokens, seed, wire-form tool calls, reasoning_content on
history messages exactly when the mode sends reasoning back, S1 history holding the answers THIS server gave (not the
gold), and on-call history holding only tool calls this server issued.
"""
from __future__ import annotations

import itertools
import json
import threading
from collections import Counter

import probe_common as pc
import sd_fake_server as base

MODEL = 'fake-qwen3.8-27b'


def default_think_words(key, turn, step):
    return 12 + (turn % 5) * 6 + 3 * step


class ProbeFakeServer(base.FakeServer):
    def __init__(self, items=(), sessions=(), thinking=False, greedy=False, history='drop', reasoning_effort='',
                 generation_config=None, script=None, think_words=default_think_words, think_scale=1.0,
                 counter=None, usage_reasoning=True, no_usage=False, http400_keys=(), model=MODEL, faults=None,
                 port=0, auto=False, reasoning_parser=True, tool_parser=True, dp=None, ctx_limit=None):
        self.mode = pc.make_mode(thinking, greedy, history, reasoning_effort, generation_config)
        self.generation_config = generation_config
        self.auto = auto  # derive the mode of every request from the request itself (one fake for every job)
        self.reasoning_parser, self.tool_parser = reasoning_parser, tool_parser  # False: play a server without one
        super().__init__(items=list(items), model=model, expected_sampling=self.mode.sampling, faults=faults,
                         port=port)
        self._local = threading.local()
        self.dp = dp  # with a size, routed_dp_rank outside 0..dp-1 is refused as SGLang's DP controller refuses it
        self.ranks = {}  # key -> set of routed_dp_rank values seen
        self.s1 = {}
        for s in sessions:
            first = pc_user_text(s, 0)
            self.s1[(s['system'], first)] = s
        self.script = dict(script or {})
        self.think_words, self.think_scale = think_words, think_scale
        self.count = counter or (lambda text: len(text.split()) if text else 0)
        self.usage_reasoning, self.no_usage = usage_reasoning, no_usage
        self.http400_keys = set(http400_keys)
        self.ctx_limit = ctx_limit  # refuse as SGLang 0.5.20 does when the prompt (+ max_tokens) exceeds it
        self.ctx_refusals = Counter()
        self.s1_given = {}
        self.issued = set()
        self.by_key = Counter()
        self.bodies = []
        self._ids = itertools.count()
        self._olock = threading.Lock()

    # ---------------------------------------------------------------- request checks
    def _mode_for(self, body):
        """The mode this request must be in: the server's own, or (auto) the one its chat_template_kwargs and
        temperature name, which must still be a mode the runner can produce."""
        if not self.auto:
            return self.mode
        ctk = body.get('chat_template_kwargs') or {}
        thinking = bool(ctk.get('enable_thinking'))
        history = {False: 'drop', True: 'keep'}.get(ctk.get('preserve_thinking'), 'strip') if thinking else 'drop'
        try:
            return pc.make_mode(thinking, body.get('temperature') == 0.0, history, ctk.get('reasoning_effort', ''),
                                self.generation_config)
        except ValueError as error:
            self._violate(f'request mode: {error}')
            return self.mode

    @property
    def rmode(self):
        return getattr(self._local, 'mode', None) or self.mode

    def _check_shape(self, body, kind):
        mode = self._local.mode = self._mode_for(body)
        expected = mode.sampling
        v = []
        if body.get('model') != self.model:
            v.append(f"model {body.get('model')!r}")
        if body.get('stream') is not False:
            v.append('stream is not false')
        if body.get('chat_template_kwargs') != mode.ctk:
            v.append(f"chat_template_kwargs {body.get('chat_template_kwargs')!r}, want {mode.ctk!r}")
        for key, want in expected.items():
            if body.get(key) != want:
                v.append(f'sampling {key}={body.get(key)!r}, want {want!r}')
        if not isinstance(body.get('max_tokens'), int) or body['max_tokens'] <= 0:
            v.append(f"max_tokens {body.get('max_tokens')!r}")
        if not isinstance(body.get('seed'), int):
            v.append(f"seed {body.get('seed')!r}")
        msgs = body.get('messages') or []
        ids = set()
        for m in msgs:
            if m['role'] == 'assistant':
                has = isinstance(m.get('reasoning_content'), str) and m['reasoning_content'] != ''
                if mode.attach_reasoning and mode.thinking and not has and not self.canned:
                    v.append('a history assistant message lacks its reasoning_content')
                if not mode.attach_reasoning and 'reasoning_content' in m:
                    v.append('a history assistant message carries reasoning_content in a mode that sends none back')
                if not isinstance(m.get('content'), str):
                    v.append('assistant content must be a string')
                for c in m.get('tool_calls') or []:
                    if not isinstance(c.get('id'), str) or not c['id']:
                        v.append('assistant tool call without an id')
                    elif c['id'] not in self.issued and not self.canned:
                        v.append('history carries a tool call this server never made')
                    if not isinstance(c['function'].get('arguments'), str):
                        v.append('tool call arguments must be a JSON string on the wire')
                    ids.add(c.get('id'))
            elif m['role'] == 'tool':
                if m.get('tool_call_id') not in ids:
                    v.append('tool message answers no earlier tool call')
        if kind == 'tool' and not body.get('tools'):
            v.append('a tool request must carry tools')
        if kind == 's1' and body.get('tools') and not self.canned:
            v.append('an S1 request must carry no tools')
        for what in v:
            self._violate(f'{kind}: {what}')

    # ---------------------------------------------------------------- answers
    @property
    def canned(self):
        return getattr(self._local, 'canned', False)

    def _canned_plan(self, body, msgs):
        """probe_common.check_server's requests (a weather tool call, 17 times 23, the tool result read back)."""
        last = msgs[-1]
        names = [t['function']['name'] for t in body.get('tools') or []]
        reasoning = 'The user wants a fact; answer it.' if self.rmode.thinking else ''
        if last['role'] == 'tool':
            return {'mode': 'canned', 'content': 'It is 18 C and clear in Paris.', 'reasoning': reasoning,
                    'tool_calls': []}
        if 'get_weather' in names:
            return {'mode': 'canned', 'content': '', 'reasoning': reasoning,
                    'tool_calls': [self._call('get_weather', {'city': 'Paris'})]}
        if '17 times 23' in (last.get('content') or ''):
            return {'mode': 'canned', 'content': '391', 'reasoning': reasoning, 'tool_calls': []}
        return None

    def _answer(self, body):
        msgs = body.get('messages') or []
        with self._olock:
            self.bodies.append(body)
        rank = body.get('routed_dp_rank')
        if rank is not None and self.dp is not None and not (0 <= rank < self.dp):
            return 400, {'error': {'message': f'DP rank {rank} is not active.', 'code': 400}}
        first_user = next((m.get('content') or '' for m in msgs if m['role'] == 'user'), '')
        self._local.canned = ('17 times 23' in first_user or 'weather in Paris' in first_user)
        if self.canned:
            self.counts['canned'] += 1
            self._check_shape(body, 'canned')
            plan = self._canned_plan(body, msgs)
            if plan is None:
                return 400, {'error': 'unknown canned request'}
            return 200, self._response(plan, msgs, body)
        if body.get('tools'):
            self.counts['tool'] += 1
            self._check_shape(body, 'tool')
            item = self.index.find(msgs, body['tools'])
            if item is None:
                self._violate('tool: request matches no item')
                return 400, {'error': 'unknown item'}
            key = item['id']
            if key in self.http400_keys:
                self.counts['http400'] += 1
                return 400, {'error': {'message': 'Input is too long for the context window', 'code': 400}}
            refused = self._context_refusal(msgs, body)
            if refused:
                return refused
            for m in msgs:  # the served template refuses tool-call arguments that are not a JSON object
                for c in (m.get('tool_calls') or []) if m['role'] == 'assistant' else []:
                    try:
                        ok = isinstance(json.loads(c['function']['arguments']), dict)
                    except (ValueError, TypeError):
                        ok = False
                    if not ok:
                        self.counts['http400_arguments'] += 1
                        return 400, {'error': {'message': 'arguments must be a JSON object', 'code': 400}}
            plan = self._plan_tool(item, msgs, body)
        else:
            self.counts['s1'] += 1
            self._check_shape(body, 's1')
            sess = self._find_s1(msgs)
            if sess is None:
                self._violate('s1: request matches no session')
                return 400, {'error': 'unknown session'}
            key = f"{sess['domain']}:{sess['seed']}"
            if key in self.http400_keys:
                self.counts['http400'] += 1
                return 400, {'error': {'message': 'Input is too long for the context window', 'code': 400}}
            refused = self._context_refusal(msgs, body)
            if refused:
                return refused
            plan = self._plan_s1(sess, key, msgs, body)
        with self._olock:
            self.by_key[key] += 1
            self.counts['mode_' + str(plan.get('mode'))] += 1
            self.ranks.setdefault(key, set()).add(body.get('routed_dp_rank'))
        return 200, self._response(plan, msgs, body)

    def _prompt_tokens(self, msgs):
        return self.count(' '.join((m.get('content') or '') for m in msgs))

    def _context_refusal(self, msgs, body):
        """SGLang 0.5.20 TokenizerManager._validate_one_request, messages verbatim, in its ErrorResponse body."""
        if self.ctx_limit is None:
            return None
        n, m, c = self._prompt_tokens(msgs), int(body['max_tokens']), int(self.ctx_limit)
        if n >= c:
            msg = f"The input ({n} tokens) is longer than the model's context length ({c} tokens)."
            kind = 'input'
        elif n + m > c:
            msg = (f"Requested token count exceeds the model's maximum context length of {c} tokens. You requested a "
                   f"total of {n + m} tokens: {n} tokens from the input messages and {m} tokens for the completion. "
                   f"Please reduce the number of tokens in the input messages or the completion to fit within the "
                   f"limit.")
            kind = 'total'
        else:
            return None
        with self._olock:
            self.ctx_refusals[kind] += 1
        return 400, {'object': 'error', 'message': msg, 'type': 'BadRequest', 'param': None, 'code': 400}

    def _find_s1(self, msgs):
        system = msgs[0]['content'] if msgs and msgs[0]['role'] == 'system' else ''
        users = [m['content'] for m in msgs if m['role'] == 'user']
        return self.s1.get((system, users[0])) if users else None

    def _reasoning(self, key, turn, step):
        n = max(1, int(round(self.think_words(key, turn, step) * self.think_scale)))
        return ' '.join(['reason'] * n) if self.rmode.thinking else ''

    def _runaway(self, body):
        """A generation that hits max_tokens: thinking on, all of it is reasoning (no </think>, no content), as
        SGLang's qwen3 parser returns it; thinking off, all of it is a rambling answer."""
        n = max(1, int(body['max_tokens']) - 1)
        if self.rmode.thinking:
            return {'content': '', 'reasoning': ' '.join(['reason'] * n)}
        return {'content': ' '.join(['reason'] * n), 'reasoning': ''}

    def _call(self, name, arguments, raw=None):
        cid = f'call_{next(self._ids):06d}'
        with self._olock:
            self.issued.add(cid)
        return {'id': cid, 'type': 'function',
                'function': {'name': name, 'arguments': raw if raw is not None else json.dumps(arguments,
                                                                                               ensure_ascii=False)}}

    def _plan_tool(self, item, msgs, body):
        users = [i for i, m in enumerate(msgs) if m['role'] == 'user']
        t = len(users) - 1
        turn = item['turns'][t]
        expect = turn['expect']
        after = [m for m in msgs[users[-1] + 1:] if m['role'] == 'assistant']
        made = sum(len(m.get('tool_calls') or []) for m in after)
        step = len(after)
        mode = self.script.get((item['id'], t))
        calls = list(expect.get('calls') or [])
        target = calls[:-1] if mode == 'omit' else calls
        reasoning = self._reasoning(item['id'], t, step)
        if mode == 'truncated' and step == 0:
            return {'mode': mode, 'truncated': True, **self._runaway(body), 'tool_calls': []}
        if mode == 'loop' and calls:
            return {'mode': mode, 'content': '', 'reasoning': reasoning,
                    'tool_calls': [self._call(calls[0]['name'], calls[0]['arguments'])]}
        if made < len(target):
            batch = target[made:] if (expect.get('parallel') or made > 0) else target[:1]
            out = []
            for k, c in enumerate(batch):
                args = c['arguments']
                if mode == 'wrong_arg' and made == 0 and k == 0:
                    args = base._perturb_args(args)
                raw = '{"broken": ' if mode == 'bad_json' and made == 0 and k == 0 else None
                out.append(self._call(c['name'], args, raw))
            if mode == 'extra' and made == 0:
                out.append(self._call(batch[0]['name'], batch[0]['arguments']))
            return {'mode': mode, 'content': '', 'reasoning': reasoning, 'tool_calls': out}
        if mode == 'spurious' and made == len(target):
            first = item['tools'][0]['function']
            args = {k: 'x' for k in (first['parameters'].get('properties') or {})}
            return {'mode': mode, 'content': '', 'reasoning': reasoning,
                    'tool_calls': [self._call(first['name'], args)]}
        return {'mode': mode, 'content': 'Done.' if calls else 'Noted.', 'reasoning': reasoning, 'tool_calls': []}

    def _plan_s1(self, sess, key, msgs, body):
        users = [m for m in msgs if m['role'] == 'user']
        t = len(users) - 1
        hist = [m['content'] for m in msgs if m['role'] == 'assistant']
        for i, h in enumerate(hist):
            given = self.s1_given.get((key, i))
            if given is not None and h != given:
                self._violate(f's1: history at turn {i} is not the answer this server gave')
        gold = sess['turns'][t]['answer']
        mode = self.script.get((key, t))
        reasoning = self._reasoning(key, t, 0)
        truncated = False
        if mode == 'wrong':
            content = gold + 'Z'
        elif mode == 'format':
            content = gold + '.'
        elif mode == 'case':
            content = gold.upper()
        elif mode == 'verbose':
            content = 'The answer is ' + gold
        elif mode == 'empty':
            content = ''
        elif mode == 'truncated':
            truncated = True
            r = self._runaway(body)
            content, reasoning = r['content'], r['reasoning']
        else:
            content = gold
        with self._olock:
            self.s1_given[(key, t)] = content
        return {'mode': mode, 'content': content, 'reasoning': reasoning, 'tool_calls': [], 'truncated': truncated}

    def _response(self, plan, msgs, body):
        content, reasoning, calls = plan['content'], plan['reasoning'], plan['tool_calls']
        truncated = plan.get('truncated', False)
        thinking = self.rmode.thinking
        if thinking and not self.reasoning_parser:  # no --reasoning-parser: the think block stays in the content
            content, reasoning = reasoning + '\n</think>\n\n' + content, ''
        if calls and not self.tool_parser:  # no --tool-call-parser: the markup stays in the content
            content += ''.join(f"<tool_call>\n<function={c['function']['name']}>\n</function>\n</tool_call>"
                               for c in calls)
            calls = []
        finish = 'length' if truncated else ('tool_calls' if calls else 'stop')
        message = {'role': 'assistant', 'content': content,
                   'reasoning_content': reasoning if thinking else None}
        if calls:
            message['tool_calls'] = calls
        obj = {'id': 'chatcmpl-probe-fake', 'object': 'chat.completion', 'model': self.model,
               'choices': [{'index': 0, 'message': message, 'finish_reason': finish}]}
        if not self.no_usage:
            markup = ''.join(f"<tool_call>\n<function={c['function']['name']}>\n{c['function']['arguments']}\n"
                             f"</function>\n</tool_call>" for c in calls)
            if truncated:
                comp = int(body['max_tokens'])
                srt = comp if thinking else 0
            else:
                srt = (self.count(reasoning) + 1) if thinking and self.reasoning_parser else 0
                comp = srt + self.count(content) + self.count(markup) + 1
            prompt = self._prompt_tokens(msgs)
            usage = {'prompt_tokens': prompt, 'completion_tokens': comp, 'total_tokens': prompt + comp}
            if self.usage_reasoning:
                usage['reasoning_tokens'] = srt
            obj['usage'] = usage
        return obj


def pc_user_text(sess, t):
    """The user text of turn t of an S1 session row, rendered as the runner renders it."""
    from common import Turn, user_text
    turn = Turn(**sess['turns'][t])
    return user_text(sess['lang'], turn, sess['initial'] if t == 0 else None)


def main(argv=None):
    """Serve the fake on a port until killed (run_all.sh tests: SERVE_ARM stub starts it, one fake for every job)."""
    import argparse
    import sys
    ap = argparse.ArgumentParser(description='probe fake server (CPU tests only)')
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--items', nargs='*', default=[])
    ap.add_argument('--s1', default='')
    ap.add_argument('--generation-config', default=None)
    ap.add_argument('--no-reasoning-parser', action='store_true')
    ap.add_argument('--no-tool-parser', action='store_true')
    ap.add_argument('--model', default=MODEL)
    ap.add_argument('--dp', type=int, default=None)
    args = ap.parse_args(argv)
    items, sessions = [], []
    for p in args.items:
        items += pc.read_jsonl(p)
    if args.s1:
        import glob
        import os
        for p in sorted(glob.glob(os.path.join(args.s1, '*.eval.jsonl'))):
            sessions += pc.read_jsonl(p)
    srv = ProbeFakeServer(items, sessions, auto=True, generation_config=args.generation_config, model=args.model,
                          reasoning_parser=not args.no_reasoning_parser, tool_parser=not args.no_tool_parser,
                          port=args.port, dp=args.dp)
    srv.start()
    print(srv.url, flush=True)
    sys.stdout.flush()
    threading.Event().wait()


if __name__ == '__main__':
    main()
