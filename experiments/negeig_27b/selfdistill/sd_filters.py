"""Quality filters for the self-distilled replay, and the filter report.

Generation writes raw ledgers only (translations.jsonl, chat_raw.jsonl, tool_raw.jsonl). This module turns the
ledgers into the training files (chat_sft.jsonl, tool_sft.jsonl) and writes filter_report.json. It is a pure
function of the ledgers, so it can be rerun with stricter rules without touching the server again.

A row is dropped for exactly one reason, the first that applies, in the order of the REASONS lists.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from sd_common import (CJK, HEBREW, LATIN, dumps, hebrew_share, log, norm_text, prose, read_jsonl, seed_for,
                       write_jsonl_atomic)

CHAT_REASONS = ('api_error', 'empty', 'truncated', 'reasoning_leak', 'markup_leak', 'refusal',
                'language_mismatch', 'degenerate_repeat', 'duplicate')
TRANSLATION_REASONS = ('api_error', 'truncated', 'empty', 'markup', 'refusal', 'preamble', 'not_hebrew', 'cjk',
                       'code_changed', 'numbers_changed', 'length')
TOOL_REASONS = ('not_solved', 'empty', 'reasoning_leak', 'markup_leak', 'refusal',
                'language_mismatch', 'duplicate', 'over_quota')

MARKUP = re.compile(r'<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>|</?tool_call>|</?tool_response>|<function=|'
                    r'<parameter=|</?message>')
THINK = re.compile(r'</?think>', re.I)

# Hebrew "cannot" is a capability or permission word. "I do not know" and "it cannot be" are answers, not refusals
# (the English list below has the same shape: a cannot-word, then a help-word).
_HE_CANT = (r"(?:(?:לא|אינני|איני) (?:יכול|יכולה|מסוגל|מסוגלת|רשאי|רשאית)|לא אוכל|לא אהיה מסוגל|לא אהיה מסוגלת)")
_HE_HELP = (r" ל(?:עזור|סייע|ספק|ענות|השיב|יצור|ייצר|כתוב|תת|עשות|מלא|פעול|המשיך|עסוק|דון|שתף|תמוך|הגיב|מסור|הציע|"
            r"המליץ|היענות)")

_REFUSAL = [re.compile(p) for p in (
    r"^(?:i'?m|i am) (?:very |so |really |truly )?sorry,? (?:but )?(?:i )?(?:can(?:no|')t|cannot|am (?:not able|unable) to|won't|will not)",
    r"^(?:sorry|apologies),? (?:but )?i (?:can(?:no|')t|cannot|am unable|won't)",
    r"^i (?:can(?:no|')t|cannot|am unable to|won't|will not) (?:help|assist|provide|comply|fulfil|fulfill|do that|answer|create|write|generate|support)",
    r"^i'?m (?:not able|unable) to (?:help|assist|provide|comply)",
    r"^as an ai(?: language model| assistant)?,? i (?:can(?:no|')t|cannot|don't|do not|am unable)",
    r"^(?:אני )?(?:מצטער|מצטערת|סליחה|אני מתנצל|אני מתנצלת)[,.!]? (?:אבל )?(?:ש?אני )?" + _HE_CANT,
    r"^(?:אני )?" + _HE_CANT + _HE_HELP,
    r"^(?:לצערי|למרבה הצער),? (?:אני )?" + _HE_CANT,
    r"^כמודל שפה(?: [^,.]{0,25})?,? (?:אני )?(?:לא|אינני|איני) ",
)]
_PREAMBLE = [re.compile(p) for p in (
    r"^(?:sure|certainly|of course|okay|ok)[,!.:]", r"^here(?:'s| is| are)\b", r"^translation\s*:",
    r"^(?:הנה|בוודאי|בטח|כמובן|אין בעיה)[,!.: ]", r"^תרגום\s*:",
)]
_NUMBER = re.compile(r'\d+')
_FENCE_BLOCK = re.compile(r'```.*?```', re.S)
_INLINE_CODE = re.compile(r'`[^`\n]+`')


def _opening(text):
    t = norm_text(text).lower().replace('’', "'").replace('‘', "'")
    return t.lstrip('*_#>"\'“” ')


def is_refusal(text, max_chars=600):
    """An apology or a flat cannot-help opener on a short answer that asks nothing back.

    A long answer that opens with a caveat is kept, and so is a short answer that asks for a missing value
    (the right behavior on a missing-parameter turn)."""
    t = _opening(text)
    if not t or len(t) > max_chars or '?' in t:
        return False
    return any(p.search(t[:220]) for p in _REFUSAL)


def is_preamble(text):
    t = _opening(text)
    return any(p.search(t[:80]) for p in _PREAMBLE)


def degenerate(text, min_words=80, floor=0.35):
    """A loop: fewer than `floor` of the 6-word shingles are distinct."""
    words = text.split()
    if len(words) < min_words:
        return False
    grams = [tuple(words[i:i + 6]) for i in range(len(words) - 5)]
    return len(set(grams)) / len(grams) < floor


def language_mismatch(user_text, answer):
    """True when a clearly Hebrew request got a clearly English answer (or the reverse), or when the answer
    drifts into CJK script without the request having any."""
    if len(CJK.findall(prose(answer))) >= 2 and not CJK.search(user_text):
        return True
    u, a = hebrew_share(user_text), hebrew_share(answer)
    if u is None:
        return False
    if a is None:  # a few letters only: "Done." to a Hebrew request is still English
        p = prose(answer)
        return u >= 0.6 and len(LATIN.findall(p)) >= 3 and not HEBREW.search(p)
    if u >= 0.6 and a < 0.25:
        return True
    return u <= 0.1 and a > 0.6


def pair_key(prompt, answer):
    return hashlib.blake2b((norm_text(prompt) + '\x1f' + norm_text(answer)).encode(), digest_size=16).hexdigest()


def _tokens(row):
    return int((row.get('usage') or {}).get('completion_tokens') or 0)


def _sorted_ids(rows, *keys):
    return sorted(rows, key=lambda r: tuple(str(r.get(k, '')) for k in keys))


# ------------------------------------------------------------------ translations
def translation_reason(row):
    """None when the Hebrew translation of row['src'] is usable as a replay prompt."""
    if row.get('error'):
        return 'api_error'
    src, out = row.get('src') or '', row.get('out') or ''
    if row.get('finish') not in ('stop', None, ''):
        return 'truncated'
    if not out.strip():
        return 'empty'
    if MARKUP.search(out) or THINK.search(out):
        return 'markup'
    if is_refusal(out):
        return 'refusal'
    if is_preamble(out) and not is_preamble(src):
        return 'preamble'
    if len(CJK.findall(prose(out))) >= 2 and not CJK.search(src):
        return 'cjk'
    share = hebrew_share(out)
    if share is None or share < 0.5:
        return 'not_hebrew'
    for block in _FENCE_BLOCK.findall(src) + _INLINE_CODE.findall(src):
        if block not in out:
            return 'code_changed'
    want, got = Counter(_NUMBER.findall(src)), Counter(_NUMBER.findall(out))
    total = sum(want.values())
    if total:
        missing, extra = sum((want - got).values()), sum((got - want).values())
        if missing > 0.1 * total or extra > 0.2 * total + 1:
            return 'numbers_changed'
    ratio = len(out.strip()) / max(len(src.strip()), 1)
    if ratio < 0.3 or ratio > 3.0:
        return 'length'
    return None


def filter_translations(rows):
    by_id = {}
    for r in rows:  # a resumed run may hold one id twice; the last row wins
        by_id[r['id']] = r
    ok, dropped, tokens = [], Counter(), 0
    for r in _sorted_ids(by_id.values(), 'id'):
        tokens += _tokens(r)
        reason = translation_reason(r)
        if reason:
            dropped[reason] += 1
        else:
            ok.append(r)
    report = {'attempted': len(by_id), 'ok': len(ok), 'dropped': {k: dropped[k] for k in TRANSLATION_REASONS if dropped[k]},
              'generated_tokens': tokens}
    return ok, report


# ------------------------------------------------------------------ chat answers
def chat_reason(row):
    if row.get('error'):
        return 'api_error'
    content = row.get('content') or ''
    if not content.strip():
        return 'empty'
    if row.get('finish') not in ('stop', 'tool_calls'):
        return 'truncated'
    if row.get('reasoning') or THINK.search(content):
        return 'reasoning_leak'
    if row.get('finish') == 'tool_calls' or MARKUP.search(content):  # no tools were offered: a parser leak
        return 'markup_leak'
    if is_refusal(content):
        return 'refusal'
    if language_mismatch(row['prompt'], content):
        return 'language_mismatch'
    if degenerate(content):
        return 'degenerate_repeat'
    return None


def chat_sft_row(row, meta):
    return {'id': row['id'], 'source': row['source'], 'lang': row['lang'], 'category': row.get('category'),
            'messages': [{'role': 'user', 'content': row['prompt']},
                         {'role': 'assistant', 'content': row['content'].strip()}],
            'meta': {**(meta or {}), 'finish': row['finish'], 'completion_tokens': _tokens(row),
                     **({'translated_from': row['translated_from']} if row.get('translated_from') else {})}}


def filter_chat(rows, meta=None):
    by_id = {}
    for r in rows:
        by_id[r['id']] = r
    kept, seen = [], set()
    dropped = Counter()
    per_lang = defaultdict(lambda: {'rows': 0, 'kept': 0, 'dropped': Counter()})
    tok_all = tok_kept = 0
    for r in _sorted_ids(by_id.values(), 'id'):
        lang = r.get('lang', '?')
        per_lang[lang]['rows'] += 1
        tok_all += _tokens(r)
        reason = chat_reason(r)
        if reason is None:
            key = pair_key(r['prompt'], r['content'])
            if key in seen:
                reason = 'duplicate'
            else:
                seen.add(key)
        if reason:
            dropped[reason] += 1
            per_lang[lang]['dropped'][reason] += 1
        else:
            kept.append(chat_sft_row(r, meta))
            per_lang[lang]['kept'] += 1
            tok_kept += _tokens(r)
    lens = sorted(len(k['messages'][1]['content']) for k in kept)
    report = {
        'rows': len(by_id), 'kept': len(kept), 'kept_rate': round(len(kept) / max(len(by_id), 1), 4),
        'dropped': {k: dropped[k] for k in CHAT_REASONS if dropped[k]},
        'by_lang': {lang: {'rows': v['rows'], 'kept': v['kept'],
                           'dropped': {k: v['dropped'][k] for k in CHAT_REASONS if v['dropped'][k]}}
                    for lang, v in sorted(per_lang.items())},
        'generated_tokens': tok_all, 'kept_tokens': tok_kept,
        'kept_answer_chars': {'median': lens[len(lens) // 2] if lens else 0, 'p95': lens[int(len(lens) * .95)] if lens else 0},
    }
    return kept, report


# ------------------------------------------------------------------ tool-use transcripts
def split_turns(messages):
    """Messages grouped by user turn: [[user, assistant, tool, ...], ...] (the system message is dropped)."""
    turns = []
    for m in messages:
        if m['role'] == 'user':
            turns.append([m])
        elif turns:
            turns[-1].append(m)
    return turns


def _texts(turn):
    """(user text, final assistant text or None) of one turn. The final assistant message is the last one."""
    user = turn[0]['content']
    last = next((m for m in reversed(turn) if m['role'] == 'assistant'), None)
    return user, last


def transcript_key(messages):
    """Identity of a transcript ignoring tool-call ids, which the server draws at random."""
    parts = []
    for m in messages:
        if m['role'] == 'assistant':
            calls = [(c['function']['name'], norm_text(c['function']['arguments'])
                      if isinstance(c['function']['arguments'], str) else dumps(c['function']['arguments']))
                     for c in m.get('tool_calls') or []]
            parts.append(('a', norm_text(m.get('content') or ''), calls))
        else:
            parts.append((m['role'], norm_text(m.get('content') or '')))
    return hashlib.blake2b(dumps(parts).encode(), digest_size=16).hexdigest()


def solved(row):
    return row.get('reward') == 1.0 and row.get('closed') and not row.get('errors') and not row.get('error')


def tool_reason(row):
    """First applicable reason a transcript is unusable, None when it is clean (before dedupe and quota)."""
    if not solved(row):
        return 'not_solved'
    msgs = row['messages']
    turns = split_turns(msgs)
    for turn in turns:
        _, last = _texts(turn)
        if last is None or last.get('tool_calls') or not (last.get('content') or '').strip():
            return 'empty'
    if row.get('reasoning_chars'):
        return 'reasoning_leak'
    for m in msgs:
        if m['role'] == 'assistant' and m.get('content') and (MARKUP.search(m['content']) or THINK.search(m['content'])):
            return 'markup_leak'
    for turn in turns:
        if row.get('kind') == 'unsupported' and not any(m.get('tool_calls') for m in turn if m['role'] == 'assistant'):
            continue  # the near-miss turn: "no tool does that" is the target answer, so a cannot-opener is not a refusal
        if is_refusal(_texts(turn)[1]['content']):
            return 'refusal'
    for turn in turns:
        user, last = _texts(turn)
        if language_mismatch(user, last['content']):
            return 'language_mismatch'
    return None


def sft_tool_messages(raw_messages):
    """Wire messages -> training messages: dict arguments (the HF template renders a dict and raises on a
    string), content never None, positional tool-call ids (the template ignores them, dedupe and diffs do not)."""
    out, ids = [], {}
    for m in raw_messages:
        if m['role'] == 'assistant':
            msg = {'role': 'assistant', 'content': m.get('content') or ''}
            if m.get('tool_calls'):
                calls = []
                for c in m['tool_calls']:
                    new_id = ids.setdefault(c['id'], f'call_{len(ids)}')
                    args = c['function']['arguments']
                    calls.append({'id': new_id, 'type': 'function',
                                  'function': {'name': c['function']['name'],
                                               'arguments': json.loads(args) if isinstance(args, str) else args}})
                msg['tool_calls'] = calls
            out.append(msg)
        elif m['role'] == 'tool':
            out.append({'role': 'tool', 'tool_call_id': ids.get(m['tool_call_id'], m['tool_call_id']),
                        'content': m['content']})
        else:
            out.append({'role': m['role'], 'content': m['content']})
    return out


def filter_tools(rows, items_by_id, template_tools, keep_per_item=2, meta=None):
    """Keep transcripts that scored 1.0 and pass the quality filters. Per item at most `keep_per_item` are
    kept (0 keeps all); the surplus counts as over_quota, which is a surplus and not a defect."""
    by_key = {}
    for r in rows:
        by_key[(r['item_id'], r['sample'])] = r
    ordered = sorted(by_key.values(), key=lambda r: (r['item_id'], int(r['sample'])))
    kept, seen = [], set()
    kinds = defaultdict(lambda: {'items': set(), 'episodes': 0, 'solved': 0, 'solved_items': set(), 'kept': 0,
                                 'kept_items': set(), 'dropped': Counter(), 'not_solved_by': Counter(), 'tokens': 0})
    per_item = Counter()
    dropped = Counter()
    for r in ordered:
        k = kinds[r['kind']]
        k['items'].add(r['item_id'])
        k['episodes'] += 1
        k['tokens'] += _tokens(r)
        if solved(r):
            k['solved'] += 1
            k['solved_items'].add(r['item_id'])
        reason = tool_reason(r)
        if reason == 'not_solved':
            k['not_solved_by'][r.get('failure') or ('unclosed' if not r.get('closed') else 'wrong_calls')] += 1
        if reason is None:
            key = (r['item_id'], transcript_key(r['messages']))
            if key in seen:
                reason = 'duplicate'
            else:
                seen.add(key)
        if reason is None and keep_per_item and per_item[r['item_id']] >= keep_per_item:
            reason = 'over_quota'
        if reason:
            dropped[reason] += 1
            k['dropped'][reason] += 1
            continue
        per_item[r['item_id']] += 1
        item = items_by_id[r['item_id']]
        kept.append({'id': f"{r['item_id']}#s{r['sample']}", 'item_id': r['item_id'], 'kind': r['kind'],
                     'lang': r['lang'], 'source': 'agentic_pool',
                     'messages': sft_tool_messages(r['messages']), 'tools': template_tools(item['tools']),
                     'meta': {**(meta or {}), 'turns': len(r['turn_scores']), 'generations': r.get('generations'),
                              'completion_tokens': _tokens(r)}})
        k['kept'] += 1
        k['kept_items'].add(r['item_id'])
    by_kind = {}
    for kind, k in sorted(kinds.items()):
        n_items, n_eps = len(k['items']), k['episodes']
        by_kind[kind] = {
            'items': n_items, 'episodes': n_eps, 'solved': k['solved'],
            'solved_rate': round(k['solved'] / max(n_eps, 1), 4),
            'items_with_a_solve': len(k['solved_items']),
            'pass_at_n': round(len(k['solved_items']) / max(n_items, 1), 4),
            'kept': k['kept'], 'kept_rate': round(k['kept'] / max(n_eps, 1), 4),
            'items_kept': len(k['kept_items']), 'items_kept_rate': round(len(k['kept_items']) / max(n_items, 1), 4),
            'dropped': {r: k['dropped'][r] for r in TOOL_REASONS if k['dropped'][r]},
            'not_solved_by': dict(sorted(k['not_solved_by'].items())), 'generated_tokens': k['tokens']}
    n_eps = sum(v['episodes'] for v in by_kind.values())
    report = {'episodes': n_eps, 'kept': len(kept), 'kept_rate': round(len(kept) / max(n_eps, 1), 4),
              'keep_per_item': keep_per_item,
              'dropped': {r: dropped[r] for r in TOOL_REASONS if dropped[r]}, 'by_kind': by_kind,
              'generated_tokens': sum(v['generated_tokens'] for v in by_kind.values())}
    return kept, report


# ------------------------------------------------------------------ spot read and report
def spot_rows(sources, n, seed):
    """Up to n random rows per source, tagged, for the human spot-read (50 answers per source)."""
    out = []
    for name in sorted(sources):
        rows = list(sources[name])
        random.Random(seed_for('spot', name, seed)).shuffle(rows)
        out += [{'spot_source': name, **r} for r in rows[:n]]
    return out


def finalize(out_dir, *, items_by_id=None, template_tools=None, keep_per_item=None, meta=None, spot_n=50, seed=0):
    """Filter every ledger that exists, write the training files, the spot-read sample and filter_report.json."""
    out_dir = Path(out_dir)
    if keep_per_item is None:  # the tool run recorded its quota; reruns of the filter must not change it
        manifest = out_dir / 'manifest_tools.json'
        keep_per_item = json.loads(manifest.read_text()).get('keep_per_item', 2) if manifest.exists() else 2
    report, spot = {'meta': meta or {}}, {}
    tr = read_jsonl(out_dir / 'translations.jsonl')
    if tr:
        ok, report['translation'] = filter_translations(tr)
        spot['translation'] = [{'id': r['id'], 'src': r['src'], 'out': r['out']} for r in ok]
    raw = read_jsonl(out_dir / 'chat_raw.jsonl')
    if raw:
        kept, report['chat'] = filter_chat(raw, meta)
        write_jsonl_atomic(out_dir / 'chat_sft.jsonl', kept)
        for lang in sorted({k['lang'] for k in kept}):
            spot[f'chat_{lang}'] = [k for k in kept if k['lang'] == lang]
    traw = read_jsonl(out_dir / 'tool_raw.jsonl')
    if traw:
        assert items_by_id is not None and template_tools is not None, 'tool filtering needs the pool'
        tkept, report['tools'] = filter_tools(traw, items_by_id, template_tools, keep_per_item, meta)
        write_jsonl_atomic(out_dir / 'tool_sft.jsonl', tkept)
        spot['tool'] = tkept
    if spot:
        write_jsonl_atomic(out_dir / 'spot_read.jsonl', spot_rows(spot, spot_n, seed))
    tmp = out_dir / 'filter_report.json.tmp'
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=False) + '\n', encoding='utf-8')
    tmp.replace(out_dir / 'filter_report.json')
    log('filter report:', dumps({k: (v.get('kept', v.get('ok')) if isinstance(v, dict) else None)
                                 for k, v in report.items() if k != 'meta'}))
    return report
