"""A stand-in for `transformers` so the CPU tests can run train27's own encode_chat (and sd_lengths.py) without torch:
AutoTokenizer.from_pretrained returns a regex-word tokenizer whose chat template has the shape of the model's (ChatML,
an empty think block after each assistant header when thinking is off, tool calls as <tool_call>JSON</tool_call>)."""
import json
import re


class _Tok:
    name_or_path = "fake-tokenizer"
    chat_template = "fake"

    def apply_chat_template(self, messages, tools=None, tokenize=False, enable_thinking=True):
        out = []
        if tools:
            out.append("<|im_start|>system\n# Tools\n" + json.dumps(tools) + "<|im_end|>\n")
        for m in messages:
            content = m.get("content") or ""
            if not isinstance(content, str):
                content = json.dumps(content)
            if m["role"] == "assistant":
                body = ("" if enable_thinking else "<think>\n\n</think>\n\n") + content
                for tc in m.get("tool_calls") or []:
                    body += "<tool_call>" + json.dumps(tc) + "</tool_call>"
                out.append("<|im_start|>assistant\n" + body + "<|im_end|>\n")
            else:
                out.append("<|im_start|>" + m["role"] + "\n" + content + "<|im_end|>\n")
        return "".join(out)

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        spans = [(m.start(), m.end()) for m in re.finditer(r"<\|im_start\|>|<\|im_end\|>|\w+|[^\w\s]", text)]
        return {"input_ids": list(range(len(spans))), "offset_mapping": spans}


class AutoTokenizer:
    @staticmethod
    def from_pretrained(path, **kw):
        return _Tok()
