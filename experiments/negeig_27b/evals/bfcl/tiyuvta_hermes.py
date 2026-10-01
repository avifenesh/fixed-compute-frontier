"""Prompting-mode handler that renders EACH model's own chat_template.jinja.

The stock QwenFCHandler hard-codes the Qwen3 template. DictaLM-3.0, Ornith-1.5 and
Spark-X2.5 all speak hermes-style <tool_call> JSON but each ships its own template
(DictaLM forces <think> in the generation prompt, Spark has enable_thinking, Ornith has
its own tools preamble). Rendering through the tokenizer loaded from
REMOTE_OPENAI_TOKENIZER_PATH measures the model with the template it was trained on.

Parsing (tool_call extraction, <think> stripping, multi-turn history) is inherited from
QwenFCHandler unchanged. The served alias sent to the endpoint comes from
BFCL_SERVED_MODEL so a memra alias never has to equal a HF repo id.
"""
import ast
import json
import os
import re

from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.model_handler.utils import convert_to_tool
from overrides import override


class TiyuvtaHermesHandler(QwenFCHandler):
    @override
    def _format_prompt(self, messages, function):
        msgs = []
        hint_first_user = self.registry_name == "tiyuvta/DictaLM-3.0-24B-NVFP4-parallelhint-FC"
        for m in messages:
            m = dict(m)
            if hint_first_user and m.get("role") == "user":
                m["content"] += "\n\nemit all required calls in this turn"
                hint_first_user = False
            if m.get("role") == "assistant" and m.get("tool_calls"):
                m["tool_calls"] = [
                    tc
                    if "function" in tc
                    else {
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc.get("arguments", {})},
                    }
                    for tc in m["tool_calls"]
                ]
            msgs.append(m)
        # Tools go in as the OpenAI shape an API user sends ({"type":"function","function":{...}}
        # with OpenAPI types), the shape every one of these templates is written for. Spark's
        # template raises on the raw BFCL function dicts the stock Qwen handler dumps.
        tools = convert_to_tool(function, GORILLA_TO_OPENAPI, ModelStyle.OPENAI_COMPLETIONS)
        # BFCL_TEMPLATE_KWARGS (JSON) reaches the template, e.g. {"enable_thinking": false} to
        # measure a Qwen3.x arm think-off, the mode its office eval runs in.
        extra = json.loads(os.getenv("BFCL_TEMPLATE_KWARGS") or "{}")
        return self.tokenizer.apply_chat_template(
            msgs, tools=tools, tokenize=False, add_generation_prompt=True, **extra
        )

    @override
    def _query_prompting(self, inference_data: dict):
        served = os.getenv("BFCL_SERVED_MODEL")
        if served:
            self.model_path_or_id = served
        # memra answers a transient capacity shortfall with 429; the stock client retries it
        # twice with a sub-second backoff, which is not a wait. Ten retries reach ~40 s.
        if getattr(self.client, "max_retries", 2) < 10:
            self.client = self.client.with_options(max_retries=10)
        api_response, latency = super()._query_prompting(inference_data)
        # Canonicalise the XML wire styles into hermes JSON here, where the tool schema is in
        # hand, exactly as a serving-side parser (memra, vLLM qwen3_coder / glm4) types the raw
        # text values: string stays string, integer/number/boolean cast, containers json-loaded.
        # Hermes JSON output is left untouched: its typing is the model's own. The stored
        # model_responses is therefore what a tools-API client would have received.
        try:
            text = api_response.choices[0].text
            canon = self._canonicalise(text, inference_data.get("function") or [])
            if canon != text:
                api_response.choices[0].text = canon
        except Exception:
            pass
        return api_response, latency

    @staticmethod
    def _cast(v, typ, items_type=None):
        v = v.strip()
        try:
            if typ == "string":
                return v
            if typ == "integer":
                return int(float(v)) if re.fullmatch(r"-?\d+(\.0+)?", v) else TiyuvtaHermesHandler._type_value(v)
            if typ in ("float", "number"):
                return float(v)
            if typ == "boolean":
                low = v.lower()
                return True if low == "true" else False if low == "false" else TiyuvtaHermesHandler._type_value(v)
        except Exception:
            pass
        return TiyuvtaHermesHandler._type_value(v)

    @classmethod
    def _canonicalise(cls, text, functions):
        schema = {}
        for f in functions:
            props = ((f.get("parameters") or {}).get("properties") or {})
            for nm in (f.get("name"), f.get("name", "").replace(".", "_")):
                schema[nm] = {k: (p or {}).get("type") for k, p in props.items()}
        changed = False

        def repl(m):
            nonlocal changed
            body = m.group(1).strip()
            if body.startswith("{"):
                return m.group(0)
            fn = cls._QWEN_FN_RE.search(body)
            if fn:
                name = fn.group(1).strip()
                pairs = cls._QWEN_PARAM_RE.findall(fn.group(2))
            else:
                name = body.split("<arg_key>", 1)[0].strip()
                if not name or "<" in name:
                    return m.group(0)
                pairs = cls._GLM_ARG_RE.findall(body)
            types = schema.get(name, {})
            args = {k.strip(): cls._cast(v, types.get(k.strip())) for k, v in pairs}
            changed = True
            return "<tool_call>\n" + json.dumps({"name": name, "arguments": args}, ensure_ascii=False) + "\n</tool_call>"

        out = cls._TC_RE.sub(repl, text)
        return out if changed else text

    # Two wire styles share the <tool_call> tag. Hermes (DictaLM, Ornith) carries JSON;
    # GLM-style (Spark-X2.5) carries NAME<arg_key>K</arg_key><arg_value>V</arg_value>. GLM
    # renders string values raw and everything else as JSON, and evaluation has no schema in
    # hand, so a value is json.loads'ed with a string fallback (a string that looks like a
    # number is the format's own ambiguity and is charged to the model).
    _TC_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    _GLM_ARG_RE = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL)
    # Qwen3.5 / Ornith-1.5 style: <function=NAME>\n<parameter=K>\nV\n</parameter>...</function>
    _QWEN_FN_RE = re.compile(r"<function=([^>\n]+)>(.*?)</function>", re.DOTALL)
    _QWEN_PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.DOTALL)

    @staticmethod
    def _type_value(v):
        v = v.strip()
        try:
            return json.loads(v)
        except Exception:
            pass
        try:
            return ast.literal_eval(v)
        except Exception:
            return v

    @staticmethod
    @override
    def _extract_tool_calls(input_string):
        result = []
        for m in TiyuvtaHermesHandler._TC_RE.finditer(input_string):
            body = m.group(1).strip()
            if body.startswith("{"):
                try:
                    d = json.loads(body)
                    if isinstance(d, dict) and "name" in d:
                        d.setdefault("arguments", {})
                        result.append(d)
                    continue
                except Exception:
                    pass
            fn = TiyuvtaHermesHandler._QWEN_FN_RE.search(body)
            if fn:
                args = {k.strip(): TiyuvtaHermesHandler._type_value(v)
                        for k, v in TiyuvtaHermesHandler._QWEN_PARAM_RE.findall(fn.group(2))}
                result.append({"name": fn.group(1).strip(), "arguments": args})
                continue
            name = body.split("<arg_key>", 1)[0].strip()
            if not name or "<" in name:
                continue
            args = {k.strip(): TiyuvtaHermesHandler._type_value(v)
                    for k, v in TiyuvtaHermesHandler._GLM_ARG_RE.findall(body)}
            result.append({"name": name, "arguments": args})
        return result
