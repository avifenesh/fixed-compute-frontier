"""Shared pieces of the generative probe runners (run_oncall.py, run_s1gen.py): the mode profile (sampling arm,
chat_template_kwargs, what history carries), per-generation token accounting with the real tokenizer, the resumable
per-generation ledger (gens.jsonl) that lets a killed run continue mid-session, the settings guard, and a server
check that proves the two parsers are engaged.

Client, retries, the JSONL sink, the worker pool, the failure gate and manifests are selfdistill's (sd_common.py),
imported, never copied.

Modes (Qwen3.8 chat template, checked against /data/ai-ml/hf-models/qwen3.8-27b-tokenizer/chat_template.jinja):
  --thinking off  chat_template_kwargs {"enable_thinking": false}: the generation prompt ends with an empty think block
                  and every assistant turn in history renders with an empty think block too (preserve_thinking is
                  left unset, which the template reads as true; no reasoning exists to keep). This is the layout the
                  adapters were trained on. Sampling: the vendor non-thinking arm (sd_common.SAMPLING).
  --thinking on   {"enable_thinking": true}. The template then adds its default reasoning-effort instruction (xhigh)
                  to the system turn unless --reasoning-effort is given. Sampling: the vendor thinking arm, read from
                  the checkpoint's generation_config.json (temperature 1.0, top_p 0.95, top_k 20, presence 0).
  --greedy        temperature 0 (top_k 1) instead of the vendor arm: a measurement instrument, never the product.

History reasoning, thinking on only (--history-reasoning):
  drop  (default) every assistant message carries its reasoning_content and the request sets preserve_thinking=false:
        the template renders reasoning only for assistant turns after the last user message (the current turn's tool
        loop) and drops it, think block and all, from earlier turns. Qwen's multi-turn convention, done by the
        template's own switch.
  keep  preserve_thinking=true and reasoning sent back: every earlier turn keeps its reasoning (context grows by the
        whole session's reasoning; short sessions only).
  strip no reasoning_content is sent and preserve_thinking is left unset: every earlier assistant turn, the current
        tool loop included, renders with an empty think block (what a client that discards reasoning gets).
NOTE: the Qwen3.8 template default is preserve_thinking=true. A client that sends reasoning back without the switch
gets "keep", not Qwen3's old drop-earlier behavior.

Token accounting per generation (SGLang 0.5.20, launched with --reasoning-parser qwen3):
  completion_tokens         usage.completion_tokens: every generated token
  reasoning_tokens          the reasoning_content text counted with the real tokenizer (no special tokens). The primary
                            reasoning measure: the same method for every arm and both modes.
  server_reasoning_tokens   usage.reasoning_tokens. SGLang 0.5.20 counts generated ids up to and including </think> when
                            the request is in reasoning mode (Req.update_reasoning_tokens), 0 in think-off. It runs one
                            or two tokens above reasoning_tokens (the </think> and its newline) and runs to the full
                            completion when </think> never comes (truncated, or a <tool_call> opened inside the think
                            block). None when the server sends no such field.
  content_tokens            completion_tokens - reasoning_tokens: everything that is not reasoning text (the visible
                            answer, tool-call markup, think delimiters, the end token). None without usage.
  visible_tokens            the content text (the answer the user sees) counted with the tokenizer

Context limit (chat_fit, context_refusal). SGLang 0.5.20 refuses with HTTP 400, never truncates, a request whose prompt
plus max_tokens exceeds the context. When the prompt itself fits with at least MIN_CLAMPED_TOKENS to spare, the request
is sent once more with max_tokens clamped to the room left (the cap is the harness's; a truncation there is counted
like any other). When the prompt alone does not fit, the session ends there with status context_overflow: a FINAL
outcome (the same history is refused again on any rerun), scored 0 for that turn and the unreached rest, and counted.
Any other HTTP 400 stays status http_400: not final, rerun on resume (selfdistill's rule), and kept out of the report.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
E27 = HERE.parent
for _p in (E27 / 'selfdistill', E27 / 'data'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import sd_common  # noqa: E402
from sd_common import (SAMPLING, Aborted, ApiError, Client, Exhausted, FailureGate, Generation,  # noqa: E402,F401
                       JsonlSink, log, read_jsonl, run_pool, sampling_profile, seed_for, write_manifest)

DEFAULT_TOKENIZER = os.environ.get('PROBE_TOKENIZER', str(sd_common.DEFAULT_TOKENIZER))
GREEDY = {'temperature': 0.0, 'top_p': 1.0, 'top_k': 1, 'min_p': 0.0, 'presence_penalty': 0.0,
          'repetition_penalty': 1.0}
HISTORY_MODES = ('drop', 'keep', 'strip')
REASONING_EFFORTS = ('xhigh', 'medium', 'low')   # what the Qwen3.8 template accepts
STOP_FINISH = ('stop', 'tool_calls')
# Settings that shape the rows: a resume that changes one of them would mix two experiments in one ledger.
SHAPING = ('task', 'model', 'thinking', 'greedy', 'sampling', 'chat_template_kwargs', 'attach_reasoning',
           'max_tokens', 'limit_policy', 'seed', 'selection')


# ------------------------------------------------------------------ mode profile
@dataclass
class Mode:
    thinking: bool
    greedy: bool
    history: str            # drop | keep | strip (thinking on); 'n/a' with thinking off
    reasoning_effort: str   # '' = the template default (xhigh)
    sampling: dict = field(default_factory=dict)
    ctk: dict = field(default_factory=dict)
    attach_reasoning: bool = False

    @property
    def label(self):
        return ('think_on' if self.thinking else 'think_off') + ('_greedy' if self.greedy else '')


def make_mode(thinking, greedy=False, history='drop', reasoning_effort='', generation_config=None):
    """The request profile of one mode. Raises on a combination that has no meaning."""
    if history not in HISTORY_MODES:
        raise ValueError(f'--history-reasoning must be one of {HISTORY_MODES}')
    if reasoning_effort and reasoning_effort not in REASONING_EFFORTS:
        raise ValueError(f'--reasoning-effort must be one of {REASONING_EFFORTS} (the Qwen3.8 template refuses others)')
    if not thinking and reasoning_effort:
        raise ValueError('--reasoning-effort only applies with --thinking on')
    if greedy:
        sampling = dict(GREEDY)
    elif thinking:
        sampling = sampling_profile('generation_config', generation_config or Path(DEFAULT_TOKENIZER) /
                                    'generation_config.json')
    else:
        sampling = dict(SAMPLING['vendor_nonthinking'])
    if thinking:
        ctk = {'enable_thinking': True}
        if history == 'drop':
            ctk['preserve_thinking'] = False
        elif history == 'keep':
            ctk['preserve_thinking'] = True
        if reasoning_effort:
            ctk['reasoning_effort'] = reasoning_effort
        attach = history in ('drop', 'keep')
    else:
        ctk, attach, history = {'enable_thinking': False}, False, 'n/a'
    return Mode(thinking=thinking, greedy=greedy, history=history, reasoning_effort=reasoning_effort,
                sampling=sampling, ctk=ctk, attach_reasoning=attach)


def assistant_message(gen, mode, tool_calls=True):
    """The assistant message history carries for one generation: the model's own content (and tool calls), plus its
    reasoning when the mode sends reasoning back."""
    msg = {'role': 'assistant', 'content': gen.content or ''}
    if tool_calls and gen.tool_calls:
        msg['tool_calls'] = gen.tool_calls
    if mode.attach_reasoning and gen.reasoning:
        msg['reasoning_content'] = gen.reasoning
    return msg


# ------------------------------------------------------------------ token accounting
class TokenCounter:
    """Counts text with the checkpoint's tokenizer. A lock keeps the calls serial (cheap next to an HTTP round trip)."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self._lock = threading.Lock()

    def __call__(self, text):
        if not text:
            return 0
        with self._lock:
            return len(self.tok.encode(text, add_special_tokens=False))


def load_counter(path):
    tok = sd_common.load_tokenizer(path)
    if tok is None:
        raise SystemExit(f'no tokenizer at {path}: reasoning tokens are counted with the real tokenizer '
                         f'(--tokenizer, or PROBE_TOKENIZER)')
    return TokenCounter(tok), tok


def server_reasoning(usage):
    """usage.reasoning_tokens (SGLang) or usage.completion_tokens_details.reasoning_tokens (OpenAI form); None when
    the server reports neither."""
    if not isinstance(usage, dict):
        return None
    if isinstance(usage.get('reasoning_tokens'), int):
        return usage['reasoning_tokens']
    det = usage.get('completion_tokens_details')
    if isinstance(det, dict) and isinstance(det.get('reasoning_tokens'), int):
        return det['reasoning_tokens']
    return None


def gen_record(gen, latency_s, count, max_tokens):
    """Per-generation numbers (see the module docstring for each field)."""
    usage = gen.usage if isinstance(gen.usage, dict) else {}
    comp = usage.get('completion_tokens')
    comp = comp if isinstance(comp, int) and not isinstance(comp, bool) else None
    prompt = usage.get('prompt_tokens')
    prompt = prompt if isinstance(prompt, int) and not isinstance(prompt, bool) else None
    rt = count(gen.reasoning)
    return {'finish': gen.finish_reason, 'truncated': gen.finish_reason not in STOP_FINISH,
            'max_tokens': int(max_tokens), 'prompt_tokens': prompt, 'completion_tokens': comp,
            'reasoning_tokens': rt, 'server_reasoning_tokens': server_reasoning(usage),
            'content_tokens': None if comp is None else max(0, comp - rt), 'visible_tokens': count(gen.content),
            'n_calls': len(gen.tool_calls or []), 'latency_s': round(latency_s, 4)}


TOKEN_FIELDS = ('reasoning_tokens', 'server_reasoning_tokens', 'completion_tokens', 'content_tokens', 'visible_tokens')


def sum_gens(gens):
    """Token and latency totals over the generations of one turn (None stays None only when every value is None)."""
    out = {}
    for k in TOKEN_FIELDS:
        vals = [g.get(k) for g in gens if g.get(k) is not None]
        out[k] = sum(vals) if vals else (0 if not gens else None)
    out['latency_s'] = round(sum(g.get('latency_s') or 0.0 for g in gens), 4)
    out['n_gens'] = len(gens)
    out['truncated_gens'] = sum(bool(g.get('truncated')) for g in gens)
    out['prompt_tokens_first'] = gens[0].get('prompt_tokens') if gens else None
    out['prompt_tokens_last'] = gens[-1].get('prompt_tokens') if gens else None
    out['finish'] = [g.get('finish') for g in gens]
    return out


def timed_chat(client, messages, *, tools, max_tokens, mode, seed, dp_rank=None):
    """One request, timed (the latency includes client retries). dp_rank pins the request to one SGLang data-parallel
    rank (routed_dp_rank, honored by SGLang 0.5.20's DP controller), so every turn of a session finds its history in
    that rank's radix cache instead of re-prefilling it on whichever rank round robin picks."""
    t0 = time.monotonic()
    gen = client.chat(messages, tools=tools, max_tokens=max_tokens, sampling=mode.sampling, seed=seed,
                      chat_template_kwargs=mode.ctk,
                      extra_body=None if dp_rank is None else {'routed_dp_rank': int(dp_rank)})
    return gen, time.monotonic() - t0


# SGLang 0.5.20 TokenizerManager._validate_one_request, the two refusals of a request that does not fit the context
# (HTTP 400, message verbatim in the error body). Neither is transient: the same history is refused again on a rerun.
_CTX_TOTAL = re.compile(r'maximum context length of (\d+) tokens\. You requested a total of \d+ tokens: (\d+) tokens '
                        r'from the input messages')
_CTX_INPUT = re.compile(r'The input \((\d+) tokens\) is longer than the model.s context length \((\d+) tokens\)')
MIN_CLAMPED_TOKENS = 256


def context_refusal(error):
    """{'kind': 'total'|'input', 'input_tokens', 'context'} when an ApiError is SGLang's context-length refusal, else
    None. 'total': the prompt fits but prompt + max_tokens does not; 'input': the prompt alone does not fit."""
    if not isinstance(error, ApiError) or error.status != 400:
        return None
    body = error.body
    text = body.decode('utf-8', 'replace') if isinstance(body, (bytes, bytearray)) else str(body)
    m = _CTX_TOTAL.search(text)
    if m:
        return {'kind': 'total', 'input_tokens': int(m.group(2)), 'context': int(m.group(1))}
    m = _CTX_INPUT.search(text)
    if m:
        return {'kind': 'input', 'input_tokens': int(m.group(1)), 'context': int(m.group(2))}
    return None


def chat_fit(client, messages, *, tools, max_tokens, mode, seed, dp_rank=None, min_tokens=MIN_CLAMPED_TOKENS):
    """timed_chat, and when SGLang refuses because prompt + max_tokens exceeds the context while the prompt itself fits
    with at least min_tokens to spare, the same request once more with max_tokens clamped to the room left (what a
    client does near the end of a long session; the cap is the harness's, not the model's). Returns (gen, latency,
    max_tokens used). Any other refusal propagates: context_refusal() tells the caller whether it is a context
    overflow (a final outcome of the session) or something else."""
    t0 = time.monotonic()
    try:
        gen, lat = timed_chat(client, messages, tools=tools, max_tokens=max_tokens, mode=mode, seed=seed,
                              dp_rank=dp_rank)
        return gen, lat, int(max_tokens)
    except ApiError as error:
        cr = None if error.retryable else context_refusal(error)
        if not cr or cr['kind'] != 'total':
            raise
        room = cr['context'] - cr['input_tokens']
        if room < min_tokens or room >= max_tokens:
            raise
    gen, _ = timed_chat(client, messages, tools=tools, max_tokens=room, mode=mode, seed=seed, dp_rank=dp_rank)
    return gen, time.monotonic() - t0, int(room)


def assign_dp_ranks(keys_longest_first, dp):
    """key -> DP rank, round robin over the sessions in longest-first order (each rank gets a similar share of the
    long ones). Computed from the whole selection, so a resume pins every session to the rank it had."""
    if not dp or dp <= 1:
        return {}
    return {k: i % dp for i, k in enumerate(keys_longest_first)}


# ------------------------------------------------------------------ per-generation ledger (resume mid-session)
def gen_row(key, idx, turn, gen, latency_s, max_tokens=None):
    return {'key': key, 'idx': idx, 'turn': turn, 'content': gen.content, 'reasoning': gen.reasoning,
            'tool_calls': gen.tool_calls, 'finish_reason': gen.finish_reason, 'usage': gen.usage,
            'latency_s': round(latency_s, 4), 'max_tokens': max_tokens}


def gen_from_row(row):
    return Generation(content=row.get('content') or '', reasoning=row.get('reasoning') or '',
                      tool_calls=row.get('tool_calls') or [], finish_reason=row.get('finish_reason') or '',
                      usage=row.get('usage') or {})


def load_gen_store(path):
    """key -> generation rows in order. A row repeated by a resume (same key and idx) keeps the first copy, and the
    sequence stops at the first gap so a replay never skips a generation."""
    by = {}
    for r in read_jsonl(path):
        by.setdefault(r['key'], {}).setdefault(int(r['idx']), r)
    out = {}
    for k, rows in by.items():
        seq = []
        while len(seq) in rows:
            seq.append(rows[len(seq)])
        out[k] = seq
    return out


# ------------------------------------------------------------------ settings guard and run plumbing
def guard_settings(out_dir, name, settings, allow_change=False):
    """Refuse to append to a ledger written under different shaping settings, then write the manifest."""
    path = Path(out_dir) / f'manifest_{name}.json'
    if path.exists():
        try:
            prev = json.loads(path.read_text(encoding='utf-8'))
        except ValueError:
            prev = {}
        changed = sorted(k for k in SHAPING if k in prev and prev.get(k) != settings.get(k))
        if changed and not allow_change:
            raise SystemExit(f'{path}: this run differs from the ledger in {changed}; use a fresh --out-dir '
                             f'(or --allow-settings-change to mix them on purpose)')
    return write_manifest(out_dir, name, settings)


def add_common_args(ap):
    g = ap.add_argument_group('server and mode')
    g.add_argument('--base-url', default=os.environ.get('PROBE_BASE_URL', 'http://127.0.0.1:30000'))
    g.add_argument('--model', default='auto', help='served model name; auto reads /v1/models')
    g.add_argument('--api-key', default=os.environ.get('PROBE_API_KEY', 'EMPTY'))
    g.add_argument('--arm', default='', help='label recorded in every row (untouched, ctrl_s0, wide_s0, ...)')
    g.add_argument('--thinking', choices=['on', 'off'], required=True)
    g.add_argument('--greedy', action='store_true', help='temperature 0: a measurement run, not the product')
    g.add_argument('--history-reasoning', choices=HISTORY_MODES, default='drop',
                   help='thinking on: what earlier assistant turns carry (module docstring)')
    g.add_argument('--reasoning-effort', choices=REASONING_EFFORTS, default='',
                   help='thinking on: the template reasoning effort (default: unset = the template default, xhigh)')
    g.add_argument('--max-tokens', type=int, default=0,
                   help='hard cap per generation (default: 16384 thinking on, 1024 thinking off)')
    g.add_argument('--tokenizer', default=DEFAULT_TOKENIZER)
    g.add_argument('--generation-config', default='',
                   help='the thinking arm file (default: <tokenizer>/generation_config.json)')
    g.add_argument('--concurrency', type=int, default=128, help='sessions in flight')
    g.add_argument('--dp-ranks', type=int, default=0,
                   help="the server's data-parallel size: pin each session to one rank (routed_dp_rank) so its "
                        'history stays in that rank\'s radix cache; 0 or 1 lets the server route')
    g.add_argument('--max-retries', type=int, default=5)
    g.add_argument('--timeout', type=float, default=1800.0, help='seconds per request')
    g.add_argument('--seed', type=int, default=20261002)
    g.add_argument('--out-dir', required=True)
    g.add_argument('--abort-after', type=int, default=50,
                   help='stop after this many consecutive failed sessions (the server is gone); rerun to resume')
    g.add_argument('--progress-every', type=int, default=20)
    g.add_argument('--limit', type=int, default=0, help='first N selected sessions (smoke runs)')
    g.add_argument('--plan', action='store_true',
                   help='print the selection, turn counts and the longest context it needs, write plan.json, exit')
    g.add_argument('--reply-allowance', type=int, default=64,
                   help='plan: tokens per turn added for replies longer than the gold ones (default 64)')
    g.add_argument('--allow-settings-change', action='store_true')
    return ap


def mode_from_args(args):
    gc = args.generation_config or str(Path(args.tokenizer) / 'generation_config.json')
    mode = make_mode(args.thinking == 'on', args.greedy, args.history_reasoning, args.reasoning_effort, gc)
    if not args.max_tokens:
        args.max_tokens = 16384 if mode.thinking else 1024
    return mode


def make_client(args):
    return Client(args.base_url, model=args.model, api_key=args.api_key, timeout=args.timeout,
                  max_retries=args.max_retries)


def summarize_values(vals):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return {'n': 0, 'mean': None, 'median': None}
    n = len(vals)
    med = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
    return {'n': n, 'mean': sum(vals) / n, 'median': med}


# ------------------------------------------------------------------ server check: are both parsers engaged?
WEATHER_TOOL = [{'type': 'function', 'function': {
    'name': 'get_weather', 'description': 'Current weather for a city.',
    'parameters': {'type': 'object', 'properties': {'city': {'type': 'string', 'description': 'City name.'}},
                   'required': ['city'], 'additionalProperties': False}}}]


def check_server(client, count=None, max_tokens_think=8192, dp_ranks=0):
    """Four requests that fail when --tool-call-parser qwen3_coder or --reasoning-parser qwen3 is missing, plus one
    that proves a history message carrying reasoning_content with preserve_thinking=false is accepted. Greedy, so a
    rerun asks the same thing. Returns (ok, checks)."""
    checks = []

    def add(name, ok, **info):
        checks.append({'check': name, 'ok': bool(ok), **info})

    tool_user = [{'role': 'user', 'content': 'What is the weather in Paris right now? Use the tool.'}]
    off = {'enable_thinking': False}
    on = {'enable_thinking': True, 'preserve_thinking': False}

    def ask(name, messages, ctk, tools=None, max_tokens=512, dp_rank=None):
        try:
            gen, lat = timed_chat(client, messages, tools=tools, max_tokens=max_tokens,
                                  mode=Mode(False, True, 'n/a', '', dict(GREEDY), ctk), seed=1, dp_rank=dp_rank)
            return gen, lat
        except ApiError as error:
            add(name, False, error=str(error)[:300])
            return None, 0.0

    def calls_ok(gen):
        if not gen or not gen.tool_calls:
            return False
        c = gen.tool_calls[0]['function']
        try:
            a = json.loads(c['arguments'])
        except ValueError:
            return False
        return c['name'] == 'get_weather' and isinstance(a, dict) and 'paris' in str(a.get('city', '')).casefold()

    gen, lat = ask('tool_parser_think_off', tool_user, off, WEATHER_TOOL)
    if gen is not None:
        add('tool_parser_think_off', calls_ok(gen) and '<tool_call>' not in (gen.content or ''),
            finish=gen.finish_reason, tool_calls=gen.tool_calls, content=(gen.content or '')[:200],
            usage=gen.usage, latency_s=round(lat, 3))
    q = [{'role': 'user', 'content': 'What is 17 times 23? Answer with the number only.'}]
    gen, lat = ask('reasoning_parser_think_on', q, on, None, max_tokens_think)
    if gen is not None:
        content = gen.content or ''
        add('reasoning_parser_think_on', bool(gen.reasoning) and '391' in content and '</think>' not in content
            and '<think>' not in content, finish=gen.finish_reason, content=content[:200],
            reasoning_chars=len(gen.reasoning or ''), reasoning_tokens=count(gen.reasoning) if count else None,
            server_reasoning_tokens=server_reasoning(gen.usage), usage=gen.usage, latency_s=round(lat, 3))
    gen, lat = ask('think_off_has_no_reasoning', q, off, None, 64)
    if gen is not None:
        add('think_off_has_no_reasoning', not gen.reasoning and '391' in (gen.content or ''),
            content=(gen.content or '')[:200], server_reasoning_tokens=server_reasoning(gen.usage))
    gen1, lat = ask('tool_parser_think_on', tool_user, on, WEATHER_TOOL, max_tokens_think)
    if gen1 is not None:
        add('tool_parser_think_on', calls_ok(gen1) and bool(gen1.reasoning), finish=gen1.finish_reason,
            tool_calls=gen1.tool_calls, reasoning_chars=len(gen1.reasoning or ''), latency_s=round(lat, 3))
        if calls_ok(gen1):
            hist = tool_user + [assistant_message(gen1, Mode(True, True, 'drop', '', {}, on, True)),
                                {'role': 'tool', 'tool_call_id': gen1.tool_calls[0]['id'],
                                 'content': json.dumps({'city': 'Paris', 'temp_c': 18, 'sky': 'clear'})}]
            gen2, lat = ask('history_with_reasoning_accepted', hist, on, WEATHER_TOOL, max_tokens_think)
            if gen2 is not None:
                add('history_with_reasoning_accepted', gen2.finish_reason in STOP_FINISH and
                    ('18' in (gen2.content or '') or 'clear' in (gen2.content or '').casefold()),
                    finish=gen2.finish_reason, content=(gen2.content or '')[:200])
    if dp_ranks and dp_ranks > 1:  # the runners pin sessions to ranks 0..dp-1: the last one must exist
        gen, lat = ask('routed_dp_rank_last', q, off, None, 64, dp_rank=dp_ranks - 1)
        if gen is not None:
            add('routed_dp_rank_last', '391' in (gen.content or ''), dp_rank=dp_ranks - 1,
                content=(gen.content or '')[:200])
    ok = bool(checks) and all(c['ok'] for c in checks)
    return ok, checks


def main(argv=None):
    ap = argparse.ArgumentParser(description='probe helpers')
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('check-server', help='prove the tool-call and reasoning parsers are engaged; writes a receipt')
    c.add_argument('--base-url', default=os.environ.get('PROBE_BASE_URL', 'http://127.0.0.1:30000'))
    c.add_argument('--model', default='auto')
    c.add_argument('--tokenizer', default=DEFAULT_TOKENIZER)
    c.add_argument('--out', required=True)
    c.add_argument('--timeout', type=float, default=600.0)
    c.add_argument('--dp-ranks', type=int, default=0, help='also prove routed_dp_rank reaches the last DP rank')
    args = ap.parse_args(argv)
    client = Client(args.base_url, model=args.model, timeout=args.timeout, max_retries=3)
    count = None
    if Path(args.tokenizer).exists():
        count, _ = load_counter(args.tokenizer)
    ok, checks = check_server(client, count, dp_ranks=args.dp_ranks)
    doc = {'ok': ok, 'base_url': args.base_url, 'model': client.model, 'checks': checks,
           'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    for ch in checks:
        print(f"{'PASS' if ch['ok'] else 'FAIL'} {ch['check']}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
