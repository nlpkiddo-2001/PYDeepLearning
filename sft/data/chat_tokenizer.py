"""
This module is for tokenization given all the data records has the same canonical structure
One sample record:

{
  "schema_version": "1.0",
  "id": "swe-000123",
  "metadata": {
    "source": "swe-smith",
    "task_type": "coding"
  },
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "run_shell",
        "description": "Run a shell command in the repo and return stdout/stderr.",
        "parameters": {
          "type": "object",
          "properties": {
            "command": { "type": "string", "description": "Command to run" }
          },
          "required": ["command"]
        }
      }
    }
  ],
  "messages": [
    { "role": "system", "content": "You are a coding agent." },
    { "role": "user", "content": "Tests in utils/ are failing. Fix them." },
    {
      "role": "assistant",
      "reasoning_content": "I should run the tests first to see the failure.",
      "content": "",
      "tool_calls": [
        {
          "id": "call_1",
          "type": "function",
          "function": { "name": "run_shell", "arguments": { "command": "pytest utils/ -x" } }
        }
      ],
      "train": true
    },
    {
      "role": "tool",
      "tool_call_id": "call_1",
      "name": "run_shell",
      "content": "FAILED utils/test_math.py::test_div - ZeroDivisionError"
    },
    {
      "role": "assistant",
      "reasoning_content": "The division doesn't guard against zero.",
      "content": "The failure is a missing zero check in `div()`. I've added a guard that raises `ValueError`.",
      "train": true
    }
  ]
}

Referenced files:
https://github.com/huggingface/blog/blob/main/qwen-3-chat-template-deep-dive.md -> (Qwen 3 keeps think tag only after the last user message)

"""

from __future__ import annotations

import sys

import copy
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from jinja2 import TemplateError
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100 # pytorch CE default ignore index for padded tokens is -100

Mode = Literal["auto", "single", "per_turn"] # single mode becaomes 1 samples with loss on both turns, while per_turn is one sample per assistant 
#------ 
# (PER TURN)
# Sample 1:  [user Q] [A: tool call]
# labels:     -100     ✅ loss

# Sample 2:  [user Q] [A: tool call] [tool result] [B: answer]
# labels:     -100     -100           -100          ✅ loss


# SINGLE
# Sample 1:  [user Q] [A: tool call] [tool result] [B: answer]
# labels:     -100     ✅ loss        -100          ✅ loss
#------

class RejectReason(str, Enum):
    NO_TRAIN_TURNS = "no_train_turns"
    TEMPLATE_ERROR = "template_error"
    TOKENIZER_ERROR = "tokenize_error"
    PREFIX_MISMATCH = "prefix_mismatch"
    OVERLONG = "overlong"
    NO_SUPERVISED_TOKENS = "no_supervised_tokens"

@dataclass
class TokenizerConfig:
    max_seq_len: int = 32768

    mode: Mode = "auto"
    overlong: Literal["drop", "truncate"] = "drop" #dropping the long sequence is better than truncating because model would learn from unfinished response / HALF response
    check_roundtrip: bool = False
    raise_errors: bool = False

@dataclass
class TokenizedSample:
    input_ids: list[int]
    labels: list[int]
    num_supervised : int
    meta: dict[str, Any]

@dataclass
class Rejection:
    record_id: str
    reason: RejectReason
    detail: str = ""
    turn_idx: int | None = None

@dataclass
class TokenizeResult:
    samples: list[TokenizedSample] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)
    mode: str | None = None 
    fallback_reason: str | None = None


class ModelAdapter:
    """
    default adapter for templates that follow the HF tool-use conventions

    reasoning:
     native: (model supports reasoning_content) pass it to the model as it is
     inline: prepend it to 'content' key instead of adding it as separate key
     drop: discard
    """

    name = "default"
    probe_expectations: tuple[str, ...] = ()

    def __init__(self, reasoning: Literal["native", "drop", "inline"] = "inline"):
        self.reasoning = reasoning

    def prepare(self, messages: list[dict], tools: list[dict] | None) -> tuple[list[dict], list[dict] | None]:
        out = []
        for m in copy.deepcopy(messages):
            m.pop("train", None)
            if m["role"] == "assistant" and "reasoning_content" in m:
                reasoning = m.pop('reasoning_content')
                if self.reasoning == 'native':
                    m['reasoning_content'] = reasoning
                elif self.reasoning == "inline" and reasoning:
                    m['content'] = f"<think>\n{reasoning}\n</think>\n\n{m.get('content') or ''}"
            out.append(m)
        return out, (copy.deepcopy(tools) or None) # [] or None prefers None , so sending None in tools is better rather sending empty list.

    def template_kwargs(self, messages: list[dict]) -> dict[str, Any]:
            return {}


class Qwen3ModelAdapter(ModelAdapter):
    """
    Qwen3 Hybdrid thunking template

    Template reads 'reasoning_content' natively and keepts it only for assistant turns after the last real user query. 
    'enable_thinking' only changes the generation prompt: False appends an empty <think></think> block, so records without reasoning are trained exactly as the model is prompted in no-think mode.
    """
    name = 'qwen3'
    probe_expectations = ("<tool_call>", "<tool_response>", "<think>", "</tools>")

    def __init__(self):
        super().__init__(reasoning="native")

    def template_kwargs(self, messages: list[dict]) -> dict[str, Any]:
        has_reasoning = any(
            m["role"] == "assistant" and m.get("reasoning_content") for m in messages
        )
        return {"enable_thinking": has_reasoning}


def get_adapter(tokenizer: PreTrainedTokenizerBase) -> ModelAdapter:
    template = tokenizer.chat_template or ""
    if "last_query_index" in template and "enable_thinking" in template:
        return Qwen3ModelAdapter()
    return ModelAdapter(reasoning="native" if "reasoning_content" in template else "inline")


class ChatTokenizer:
    """

    """
    def __init__(
            self,
            tokenizer: PreTrainedTokenizerBase,
            cfg: TokenizerConfig | None = None,
            adapter: ModelAdapter | None = None
    ):
        if not tokenizer.is_fast:
            raise ValueError("ChatTokenizer needs a fast tokenizer for offset mapping")
        if not tokenizer.chat_template:
            raise ValueError("Tokenizer must have chat template.")
        self.tok = tokenizer
        self.cfg = cfg or TokenizerConfig()
        self.adapter = adapter or get_adapter(tokenizer)


    def __call__(self, record: dict) -> TokenizeResult:
        """
        only public method: one record in, one TokenizeResult out
        """
        rid = str(record.get("id", "<no-id>")) 
        try:
            raw = record["messages"]
            train_turns = [i for i, m in enumerate(raw) if m["role"] == "assistant" and m.get("train", True)]
            logger.debug("[%s] %d messages, train_turns=%s", rid, len(raw), train_turns)

            if not train_turns:
                return _reject(rid, RejectReason.NO_TRAIN_TURNS)

            metadata = record.get("metadata") or {}
            meta = {"id": rid, "source": metadata.get("source"), "task_type": metadata.get("task_type")}

            messages, tools = self.adapter.prepare(raw, record.get("tools"))
            kwargs = self.adapter.template_kwargs(messages)
            logger.debug("[%s] adapter=%s template_kwargs=%s", rid, self.adapter.name, kwargs)

            fallback_reason = None
            if self.cfg.mode != "per_turn":
                result, fallback_reason = self._single(messages, tools, kwargs, train_turns, meta)
                if result is not None:
                    return result
                logger.debug("[%s] single mode failed (%s)", rid, fallback_reason)
                if self.cfg.mode == "single":
                    reason = (
                        RejectReason.OVERLONG
                        if fallback_reason == "overlong"
                        else RejectReason.PREFIX_MISMATCH
                    )
                    return _reject(rid, reason, fallback_reason)

            result = self._per_turn(messages, tools, kwargs, train_turns, meta)
            result.fallback_reason = fallback_reason
            return result

        except TemplateError as exc:
            if self.cfg.raise_errors:
                raise
            return _reject(rid, RejectReason.TEMPLATE_ERROR, repr(exc))

        except Exception as exc:
            if self.cfg.raise_errors:
                raise
            return _reject(rid, RejectReason.TOKENIZER_ERROR, repr(exc))
        
    def _render(self, messages, tools, kwargs, add_generation_prompt=False) -> str:
        """
        Returns the chat template for the given message. 
        """
        return self.tok.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **kwargs
        )

    def _turn_span(self, messages, tools, kwargs, i) -> tuple[str, str]:
        """
        Rendering 2 times to exactly find the prompt and assistant response. 
        assistant response basically is asst_resp = upto[len(prompt):] 
        so we can train on this. 
        """
        prompt = self._render(messages[:i], tools, kwargs, add_generation_prompt=True)
        upto = self._render(messages[:i+1], tools, kwargs, add_generation_prompt=False)
        return prompt , upto

    def _single(self, messages, tools, kwargs, train_turns, meta):
        full = self._render(messages, tools, kwargs)
        logger.debug("[%s] single: full rendering, %d chars:\n%s", meta["id"], len(full), full)
        spans = []
        for i in train_turns:
            prompt, upto = self._turn_span(messages, tools, kwargs, i)
            if not (full.startswith(prompt) and full.startswith(upto)):
                bad = prompt if not full.startswith(prompt) else upto
                logger.debug(
                    "[%s] single: turn %d prefix cache failed\n%s",
                    meta['id'], i, _describe_divergence(full, bad)
                )
                return None, f"turn {i} not prefix stable"
            spans.append((len(prompt), _rstrip_end(full, len(prompt), len(upto))))
            logger.debug(
                "[%s] single: turn %d span chars [%d, %d) supervised text:\n%r",
                meta["id"], i, *spans[-1], full[slice(*spans[-1])],
            )

        sample, err = self._encode(full, spans, {**meta, "mode": "single", "turn_idx": None})
        if err == RejectReason.OVERLONG:
            return None, "overlong"
        if err is not None:
            return _reject(meta["id"], err), None
        
        return TokenizeResult(samples=[sample], mode="single"),None
          

    def _per_turn(self, messages, tools, kwargs, train_turns, meta):
        result = TokenizeResult(mode="per_turn")
        for i in train_turns:
            prompt, upto = self._turn_span(messages, tools, kwargs, i)
            if not upto.startswith(prompt):
                result.rejections.append(
                    Rejection(meta["id"], reason=RejectReason.PREFIX_MISMATCH, detail="prompt not a prefix", turn_idx=i)
                )
                continue
            end = _rstrip_end(upto, len(prompt), len(upto))
            logger.debug(
                            "[%s] per_turn: turn %d context %d chars, supervised text:\n%r",
                            meta["id"], i, len(prompt), upto[len(prompt):end],
                        )
            sample, err = self._encode(
                upto[:end], [(len(prompt), end)], {**meta, "mode": "per_turn", "turn_idx": i}
            )
            if err is not None: # didn't return None, Reason like in _single func, because single has backup per_turn, but per_turn is last resort. so nothing left to fall back to, so it always returns a result. 
                result.rejections.append(Rejection(meta["id"], err, turn_idx=i))
            else:
                result.samples.append(sample)
        return result


    def _encode(self, text, spans, meta): 
        """
        This function takes a complete rendered text, figures out which characters belong to asst response.
        """
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        input_ids: list[int] = enc["input_ids"]
        labels = [IGNORE_INDEX] * len(input_ids)
        if logger.isEnabledFor(logging.DEBUG):
            head = [(self.tok.decode([t]), off) for t, off in zip(input_ids[:8], enc["offset_mapping"][:8])]
            logger.debug("[%s] encode: %d tokens; first (token, char offsets): %s", meta["id"], len(input_ids), head)

        straddling = 0
        j = 0
        for k, (s, e) in enumerate(enc['offset_mapping']):
            while j < len(spans) and s >= spans[j][1]:
                j += 1

            if j == len(spans):
                break # because there aren't any spans left to check, j reached end of span, so breaking the loop here

            span_start, span_end = spans[j]

            if span_start <= s < span_end: # check 1 , token starts inside response, so assigning it label
                labels[k] = input_ids[k]
            elif s < span_start < e: # check 2, token crosses into the response from the prompt, so count it, do assign a label
                straddling += 1

        if self.cfg.check_roundtrip: # debug purpose
            decoded = self.tok.decode(input_ids, skip_special_tokens=False)
            assert decoded == text, f"round trip mismatch for {meta['id']}"

        if len(input_ids) > self.cfg.max_seq_len: # check 1, either drop or truncate will happen here
            logger.debug(
                            "[%s] encode: %d tokens > max_seq_len=%d -> %s",
                            meta["id"], len(input_ids), self.cfg.max_seq_len, self.cfg.overlong,
                        )
            if self.cfg.overlong == "drop":
                return None, RejectReason.OVERLONG
            input_ids = input_ids[:self.cfg.max_seq_len]
            labels = labels[:self.cfg.max_seq_len]

        num_supervised = sum(l != IGNORE_INDEX for l in labels)
        logger.debug(
                    "[%s] encode: %d/%d tokens supervised, %d straddling",
                    meta["id"], num_supervised, len(input_ids), straddling,
                )
        if num_supervised == 0: # check 2 , no valid tokens were here for assistant, so skipping it
            return None, RejectReason.NO_SUPERVISED_TOKENS

        meta = {**meta, "num_tokens": len(input_ids), "straddling_tokens": straddling}
        return TokenizedSample(input_ids, labels, num_supervised, meta), None

                         
def _rstrip_end(text: str, start: int, end: int) -> int:
    """
    Remove the last \n whitespace because the model will eventually generate only till <im_end> token.
    """
    while end > start and text[end-1].isspace():
        end -= 1
    return end


def _describe_divergence(full: str, prefix: str, context: int = 60) -> str:
    """Show where `prefix` stops being a prefix of `full`."""
    k = next((i for i, (a, b) in enumerate(zip(full, prefix)) if a != b), min(len(full), len(prefix)))
    return (
        f"  diverges at char {k}\n"
        f"  full:   {full[max(0, k - context):k + context]!r}\n"
        f"  prefix: {prefix[max(0, k - context):k + context]!r}"
    )

def _reject(rid: str, reason: RejectReason, detail: str = ""):
    return TokenizeResult(rejections=[Rejection(rid, reason, detail)])
        

# --------------------------------------------------------------------------- #
# Debugging helpers (CLAUDE)
# --------------------------------------------------------------------------- #


def render_labels(sample: TokenizedSample, tokenizer: PreTrainedTokenizerBase) -> str:
    """Decode a sample with supervised regions wrapped in [[ ]] for eyeballing."""
    parts, run, run_sup = [], [], None
    for tid, lab in zip(sample.input_ids, sample.labels):
        sup = lab != IGNORE_INDEX
        if run and sup != run_sup:  # supervised/unsupervised switches: decode the finished run
            text = tokenizer.decode(run, skip_special_tokens=False)
            parts.append(f"[[{text}]]" if run_sup else text)
            run = []
        run.append(tid)
        run_sup = sup
    if run:
        text = tokenizer.decode(run, skip_special_tokens=False)
        parts.append(f"[[{text}]]" if run_sup else text)
    return "".join(parts)


# Fixed conversation that exercises everything the canonical format allows:
# system, tools, reasoning, parallel tool calls, tool results and a final answer.
PROBE_RECORD = {
    "id": "probe",
    "metadata": {"source": "probe", "task_type": "tool_calling"},
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ],
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Weather in Paris and Rome?"},
        {
            "role": "assistant",
            "reasoning_content": "I need to call the tool for both cities.",
            "content": "",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}},
                {"id": "c2", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Rome"}}},
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "get_weather", "content": "18C, sunny"},
        {"role": "tool", "tool_call_id": "c2", "name": "get_weather", "content": "24C, cloudy"},
        {
            "role": "assistant",
            "reasoning_content": "Both results are in.",
            "content": "Paris is 18C and sunny; Rome is 24C and cloudy.",
        },
    ],
}


def probe_template(chat_tokenizer: ChatTokenizer) -> list[str]:
    """Startup smoke test: render PROBE_RECORD through the template. Returns problems (empty = ok)."""
    problems = []
    adapter = chat_tokenizer.adapter
    try:
        messages, tools = adapter.prepare(PROBE_RECORD["messages"], PROBE_RECORD["tools"])
        text = chat_tokenizer._render(messages, tools, adapter.template_kwargs(messages))
    except Exception as e:  # e.g. Llama-3.2 rejects parallel tool calls
        return [f"template failed to render probe: {e!r}"]
    # Generic strings any working template must keep, plus this model's own syntax.
    for s in ("Paris", "Rome", "18C, sunny", "get_weather", *adapter.probe_expectations):
        if s not in text:
            problems.append(f"expected {s!r} in rendered probe")
    # Catches bugs in our own code path too (e.g. a wrong template kwarg rejects every record).
    result = chat_tokenizer(PROBE_RECORD)
    if not result.samples:
        problems.append(f"probe produced no samples: {result.rejections}")
    return problems
