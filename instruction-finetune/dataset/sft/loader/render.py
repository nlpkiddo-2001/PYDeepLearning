"""Chat-template rendering with assistant-only loss masks.

The mask is derived from the template itself (not re-implemented): for every assistant message i,
    prompt_i = render(messages[:i], add_generation_prompt=True)
    upto_i   = render(messages[:i+1])
and the trained span is the text `upto_i[len(prompt_i):]`. Rendering must be prefix-consistent
(the full render starts with every `upto_i`); otherwise the sample is rejected rather than mis-masked.
Token membership comes from fast-tokenizer offsets of ONE tokenization of the full text, so the ids are
exactly what the model sees at inference. Tool schemas, tool results, system and user turns are context only.
"""
from __future__ import annotations

import hashlib
import os
import json
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

RENDER_VERSION = 2   # 2: turn-by-turn assembly with fallbacks (Gemma 4); unchanged output for Qwen-style templates
OK, INCONSISTENT, ERROR, NO_TRAIN = 0, 1, 2, 3
STATUS_NAMES = {OK: "ok", INCONSISTENT: "render_inconsistent", ERROR: "render_error", NO_TRAIN: "no_trainable_tokens"}

SAMPLE = {
    "messages": [
  {
    "role": "user",
    "content": "Hey, what's the weather in Tokyo right now?"
  },
  {
    "role": "assistant",
    "tool_calls": [
      {
        "function": {
          "name": "get_current_weather",
          "arguments": {
            "location": "Tokyo, JP"
          }
        }
      }
    ]
  },
        {
            "role":"tool",
            "content": """{
        "name": "get_current_weather",
        "response": {
          "temperature": 15,
          "weather": "sunny"
        }"""
      }
]
}

class RenderError(ValueError):
    pass


@dataclass
class Rendered:
    status: int
    input_ids: np.ndarray           # int32 [n]
    loss_mask: np.ndarray           # uint8 [n]; 1 = trained
    n_misaligned: int = 0           # span boundaries that fall inside a token (those tokens are not trained)
    detail: str = ""


def _empty(status: int, detail: str = "") -> Rendered:
    return Rendered(status, np.zeros(0, np.int32), np.zeros(0, np.uint8), 0, detail)


def tokenizer_fingerprint(tok: Any, template: Optional[str]) -> str:
    """Identity of (vocabulary + merges + special tokens + chat template + render logic)."""
    h = hashlib.sha256()
    backend = getattr(tok, "backend_tokenizer", None)
    h.update((backend.to_str() if backend is not None else repr(getattr(tok, "vocab_size", ""))).encode())
    h.update((template if template is not None else (getattr(tok, "chat_template", None) or "")).encode())
    h.update(str(RENDER_VERSION).encode())
    return h.hexdigest()


def load_tokenizer(name: str, template_path: Optional[str] = None):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    if not getattr(tok, "is_fast", False):
        raise RenderError("a fast tokenizer (offset mapping) is required for loss masks")
    tok.model_max_length = 10 ** 9
    template = None
    if template_path:
        with open(template_path, encoding="utf-8") as f:
            template = f.read()
        tok.chat_template = template
    if not getattr(tok, "chat_template", None):
        raise RenderError(f"tokenizer {name!r} has no chat template; pass `chat_template:` in the config")
    return tok, template



class Renderer:

    def __init__(self, tokenizer: Any, tokenizer_config: TokenConfig=None, strip_trailing_whitespace: bool = True):
        self.assistant_token = None
        self.instruction_end = None
        self.instruction_start = None

        self.tok = tokenizer
        self.strip_trailing_whitespace = strip_trailing_whitespace

        if tokenizer_config is None:
            self._prepare()
        else:
            tokens = [tokenizer_config.instruct_start, tokenizer_config.instruct_end]
            if tokenizer_config.assistant_token:
                tokens.append(tokenizer_config.assistant_token)

            token_ids = self.tok(tokens)["input_ids"]
            token_ids = [j for i in token_ids for j in i]
            if len(token_ids) == 2:
                self.instruction_start, self.instruction_end = token_ids
            else:
                self.instruction_start, self.instruction_end, self.assistant_token = token_ids
            if not (self.instruction_end in self.tok.all_special_ids and self.instruction_start in self.tok.all_special_ids):
                raise RenderError()

    def _prepare(self):
        prefix = self._text(SAMPLE["messages"][:1], [], False)
        upto = self._text(SAMPLE["messages"][:1], [], True).strip()
        only_assistant = upto[len(prefix):]
        assistant_prefix = self.tok(only_assistant.split("\n", 1)[0], add_special_tokens=False)["input_ids"]
        end_token = self.tok(upto[:-len(only_assistant)].strip())["input_ids"][-1]

        assert len(assistant_prefix) == 2
        start_token, assistant_token = assistant_prefix
        if start_token not in self.tok.all_special_ids or end_token not in self.tok.all_special_ids:
            raise RenderError()
        self.instruction_start = start_token
        self.instruction_end  = end_token
        self.assistant_token = assistant_token

        print(self.instruction_start, self.instruction_end)


    @classmethod
    def prepare_message(cls, messages: list[dict]):
        out = []
        for msg in messages:
            d = {"role": msg["role"], "content": msg.get("content") or ""}
            if msg.get("tool_calls"):
                assert msg["role"] == "assistant"
                calls = []
                for c in msg["tool_calls"]:
                    fn = c["function"]
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args) if args.strip() else {}
                        except BaseException as e:
                            raise RenderError(f"tool call arguments are not JSON: {e}") from None
                    calls.append({"id": c.get("id"), "type": "function", "function": {"name": fn["name"], "arguments": args}})
                d["tool_calls"] = calls

            if msg["role"] == "tool":
                if msg.get("tool_call_id"):
                    d["tool_call_id"] = msg["tool_call_id"]
                if msg.get("name"):
                    d["name"] = msg["name"]
            out.append(d)
        return out

    def _text(self, messages, tools , add_generation_prompt: bool = False):
        return self.tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=False)

    def render(self, sample: dict):
        normalized_sample = self.prepare_message(sample["messages"])
        tools = sample.get("tools") or None
        full = self._text(normalized_sample, tools, False)
        encode = self.tok(full, add_special_tokens=False)
        ids = np.array(encode["input_ids"], dtype=np.int32)
        mask = np.zeros(len(ids), np.uint8)

        idx = 0
        while idx < len(ids) - 1:
            if ids[idx] == self.instruction_start and ids[idx + 1] == self.assistant_token:
                idx += 2
                while idx < len(ids) and ids[idx] != self.instruction_end:
                    mask[idx] = 1
                    idx += 1
                if idx < len(ids):
                    mask[idx] = 1
            idx += 1
        if not mask.any():
            return _empty(NO_TRAIN)
        return Rendered(OK, ids, mask)

    def decode_with_mask(self, ids: np.ndarray, mask: np.ndarray, on: str = "\033[32m", off: str = "\033[0m") -> str:
        """Human-readable view: trained tokens colored (used by `preview`)."""
        out, cur, buf = [], 0, []
        for t, m in zip(ids.tolist(), mask.tolist()):
            if m != cur and buf:
                out.append((on if cur else "") + self.tok.decode(buf) + (off if cur else ""))
                buf = []
            cur = m
            buf.append(t)
        if buf:
            out.append((on if cur else "") + self.tok.decode(buf) + (off if cur else ""))
        return "".join(out)

