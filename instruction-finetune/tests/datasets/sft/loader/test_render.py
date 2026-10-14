import json
import unittest
from pathlib import Path

import numpy as np

from datasets.sft.loader import render as R
from datasets.sft.loader.helpers import INCONSISTENT, Workspace

SAMPLE = {
    "messages": [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "find x"},
        {"role": "assistant", "content": "Looking.", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": json.dumps({"q": "x"})}}]},
        {"role": "tool", "content": "RESULT-TEXT", "tool_call_id": "c1", "name": "lookup"},
        {"role": "assistant", "content": "Final answer."},
    ],
    "tools": [{"type": "function", "function": {"name": "lookup", "description": "TOOLDESC", "parameters": {}}}],
}


def trained_text(r, ids, mask):
    return r.tok.decode(ids[mask == 1])


class RenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ws = Workspace()
        cls.tok, _ = R.load_tokenizer(cls.ws.tokenizer)
        cls.r = R.Renderer(cls.tok)

    @classmethod
    def tearDownClass(cls):
        cls.ws.close()

    def test_only_assistant_turns_and_tool_calls_are_trained(self):
        out = self.r.render(SAMPLE)
        self.assertEqual(out.status, R.OK)
        text = trained_text(self.r, out.input_ids, out.loss_mask)
        self.assertIn("Looking.", text)
        self.assertIn('"name": "lookup"', text)            # the tool call itself is a training target
        self.assertIn("Final answer.", text)
        for ctx in ("SYS", "find x", "RESULT-TEXT", "TOOLDESC"):
            self.assertNotIn(ctx, text)

    def test_trained_spans_end_with_end_token_not_trailing_newline(self):
        out = self.r.render(SAMPLE)
        ids, mask = out.input_ids, out.loss_mask
        end_id = self.tok.convert_tokens_to_ids("<|im_end|>")
        ends = np.flatnonzero((mask[:-1] == 1) & (mask[1:] == 0))
        self.assertTrue(len(ends) >= 2)
        self.assertTrue(all(ids[e] == end_id for e in ends))

    def test_trained_text_equals_assistant_segments_exactly(self):
        out = self.r.render(SAMPLE)
        spans, cur = [], []
        for t, m in zip(out.input_ids, out.loss_mask):
            if m:
                cur.append(int(t))
            elif cur:
                spans.append(self.tok.decode(cur)); cur = []
        self.assertEqual(len(spans), 2)                    # two assistant messages
        self.assertEqual(spans[1], "Final answer.<|im_end|>")
        self.assertTrue(spans[0].startswith("Looking.\n<tool_call>"))

    def test_ids_equal_one_tokenization_of_the_rendered_chat(self):
        out = self.r.render(SAMPLE)
        msgs = R.Renderer.prepare_messages(SAMPLE["messages"])
        text = self.tok.apply_chat_template(msgs, tools=SAMPLE["tools"], tokenize=False)
        self.assertEqual(out.input_ids.tolist(), self.tok(text, add_special_tokens=False)["input_ids"])
        self.assertEqual(out.n_misaligned, 0)

    def test_arguments_json_string_is_decoded_for_the_template(self):
        msgs = R.Renderer.prepare_messages(SAMPLE["messages"])
        self.assertEqual(msgs[2]["tool_calls"][0]["function"]["arguments"], {"q": "x"})
        text = self.tok.apply_chat_template(msgs, tokenize=False)
        self.assertIn('"arguments": {"q": "x"}', text)      # not a double-encoded string

    def test_failure_statuses_do_not_raise(self):
        bad_args = json.loads(json.dumps(SAMPLE))
        bad_args["messages"][2]["tool_calls"][0]["function"]["arguments"] = "{not json"
        self.assertEqual(self.r.render(bad_args).status, R.ERROR)
        no_assistant = {"messages": [{"role": "user", "content": "hi"}]}
        self.assertEqual(self.r.render(no_assistant).status, R.NO_TRAIN)
        self.assertEqual(self.r.render({"messages": [{"role": "user"}]}).status, R.NO_TRAIN)
        self.assertEqual(self.r.render({"nothing": 1}).status, R.ERROR)

    def test_prefix_inconsistent_template_is_rejected_not_mismasked(self):
        with Workspace(INCONSISTENT) as ws:
            tok, _ = R.load_tokenizer(ws.tokenizer)
            out = R.Renderer(tok).render(SAMPLE)
        self.assertEqual(out.status, R.INCONSISTENT)
        self.assertEqual(len(out.input_ids), 0)

    def test_fingerprint_changes_with_template(self):
        with Workspace(INCONSISTENT) as ws:
            other, _ = R.load_tokenizer(ws.tokenizer)
        self.assertNotEqual(R.tokenizer_fingerprint(self.tok, None), R.tokenizer_fingerprint(other, None))
        self.assertEqual(R.tokenizer_fingerprint(self.tok, None), R.tokenizer_fingerprint(self.tok, None))
        self.assertNotEqual(R.tokenizer_fingerprint(self.tok, None), R.tokenizer_fingerprint(self.tok, "{{ x }}"))

    def test_missing_chat_template_is_an_error(self):
        from transformers import PreTrainedTokenizerFast
        import tempfile
        with Workspace() as ws, tempfile.TemporaryDirectory() as d:
            t = PreTrainedTokenizerFast.from_pretrained(ws.tokenizer)
            t.chat_template = None
            t.save_pretrained(d)
            Path(d, "chat_template.jinja").unlink(missing_ok=True)
            with self.assertRaisesRegex(R.RenderError, "no chat template"):
                R.load_tokenizer(d)


# Gemma-4-like behavior in a tiny template: the generation prompt adds an empty thought block that history turns
# do not contain, consecutive assistant messages and tool results share ONE model turn, that turn is only closed
# before a user/system message or at the end, and neither a tool call nor a
# trailing tool result closes it (the model stops to wait for results, then continues the same turn).
GEMMA_LIKE = (
    "{%- for m in messages -%}"
    "{%- set nxt = messages[loop.index0 + 1].role if not loop.last else none -%}"
    "{%- set prev = messages[loop.index0 - 1].role if not loop.first else none -%}"
    "{%- if m.role == 'assistant' -%}"
    "{%- if prev not in ['assistant', 'tool'] -%}{{ '<|im_start|>assistant\\n' }}{%- endif -%}"
    "{{ m.content }}{% for c in (m.tool_calls or []) %}{{ 'CALL:' + c.function.name + ';' }}{% endfor %}"
    "{%- if nxt not in ['assistant', 'tool'] and not m.tool_calls -%}{{ '<|im_end|>\\n' }}{%- endif -%}"
    "{%- elif m.role == 'tool' -%}{{ 'RESP:' + m.content + ';' }}"
    "{%- if nxt in ['user', 'system'] -%}{{ '<|im_end|>\\n' }}{%- endif -%}"
    "{%- else -%}{{ '<|im_start|>' + m.role + '\\n' + m.content + '<|im_end|>\\n' }}{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt and (not messages or messages[-1].role != 'tool') -%}"
    "{{ '<|im_start|>assistant\\n<think></think>' }}{%- endif -%}"
)


class GenerationPromptDiffersTests(unittest.TestCase):
    """Templates whose generation prompt is not a prefix of the history render (Gemma 4)."""

    @classmethod
    def setUpClass(cls):
        cls.ws = Workspace(GEMMA_LIKE)
        cls.tok, _ = R.load_tokenizer(cls.ws.tokenizer)
        cls.r = R.Renderer(cls.tok)

    @classmethod
    def tearDownClass(cls):
        cls.ws.close()

    def segments(self, out):
        """[(trained?, text)] runs of the rendered sequence."""
        runs, cur, buf = [], None, []
        for t, m in zip(out.input_ids.tolist(), out.loss_mask.tolist()):
            if m != cur and buf:
                runs.append((cur, self.tok.decode(buf)))
                buf = []
            cur = m
            buf.append(t)
        return runs + [(cur, self.tok.decode(buf))]

    def test_generation_prompt_is_context_and_answer_is_target(self):
        out = self.r.render({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"},
                                          {"role": "user", "content": "more"}, {"role": "assistant", "content": "ok"}]})
        self.assertEqual(out.status, R.OK)
        runs = self.segments(out)
        self.assertEqual([t for m, t in runs if m], ["yo<|im_end|>", "ok<|im_end|>"])
        context = "".join(t for m, t in runs if not m)
        self.assertEqual(context.count("<think></think>"), 2)        # what the model sees at inference

    def test_tool_turn_trains_call_and_answer_but_not_result(self):
        out = self.r.render(SAMPLE)
        self.assertEqual(out.status, R.OK)
        trained = [t for m, t in self.segments(out) if m]
        self.assertEqual(trained, ["Looking.CALL:lookup;", "Final answer.<|im_end|>"])

    def test_consecutive_assistant_messages_are_one_trained_turn(self):
        s = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "Let me check."},
                          {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
                              "name": "lookup", "arguments": "{}"}}]},
                          {"role": "tool", "content": "R"}, {"role": "assistant", "content": "Done."}]}
        out = self.r.render(s)
        self.assertEqual(out.status, R.OK)
        self.assertEqual([t for m, t in self.segments(out) if m], ["Let me check.CALL:lookup;", "Done.<|im_end|>"])

    def test_text_with_call_is_split_so_reasoning_stays_before_the_call(self):
        msgs = R.Renderer.prepare_messages(SAMPLE["messages"])
        split = R.Renderer.split_text_and_calls(msgs)
        self.assertEqual([(m["role"], m["content"], bool(m.get("tool_calls"))) for m in split[2:4]],
                         [("assistant", "Looking.", False), ("assistant", "", True)])


@unittest.skipUnless(
    any(Path("~/.cache/huggingface/hub/models--google--gemma-4-31B-it").expanduser().glob("snapshots/*/tokenizer.json")),
    "Gemma 4 tokenizer not cached")
class Gemma4TemplateTests(unittest.TestCase):
    def test_real_gemma4_template(self):
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        tok, _ = R.load_tokenizer("google/gemma-4-31B-it")
        r = R.Renderer(tok)
        out = r.render(SAMPLE)
        self.assertEqual(out.status, R.OK)
        text = tok.decode(out.input_ids[out.loss_mask == 1])
        self.assertIn("Looking.<|tool_call>call:lookup", text)       # reasoning stays before the call
        self.assertIn("Final answer.<turn|>", text)
        self.assertNotIn("<|channel>thought", text)                   # empty thought block = context
        self.assertIn("<|channel>thought\n<channel|>", tok.decode(out.input_ids))
        for ctx in ("RESULT-TEXT", "TOOLDESC", "find x"):
            self.assertNotIn(ctx, text)

    def test_call_only_turn_keeps_its_special_token(self):
        # "<|channel>" (generation prompt) and "<|tool_call>" (turn) share the characters "<|": the header must
        # be cut at a token boundary or the call token is lost from the target
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        tok, _ = R.load_tokenizer("google/gemma-4-31B-it")
        s = {"messages": [{"role": "user", "content": "find x"}, {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
            {"role": "tool", "content": "RESULT-TEXT"}, {"role": "assistant", "content": "Done."}], "tools": SAMPLE["tools"]}
        out = R.Renderer(tok).render(s)
        self.assertEqual(out.status, R.OK)
        call_id = tok.convert_tokens_to_ids("<|tool_call>")
        self.assertEqual(int(((out.input_ids == call_id) & (out.loss_mask == 1)).sum()), 1)
        self.assertTrue(tok.decode(out.input_ids[out.loss_mask == 1]).startswith("<|tool_call>call:lookup"))


@unittest.skipUnless(
    any(Path("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct").expanduser().glob("snapshots/*/tokenizer.json")),
    "Qwen tokenizer not cached")
class QwenTemplateTests(unittest.TestCase):
    def test_real_qwen_template_masks_tool_calls_but_not_results(self):
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        tok, _ = R.load_tokenizer("Qwen/Qwen2.5-7B-Instruct")
        r = R.Renderer(tok)
        out = r.render(SAMPLE)
        self.assertEqual(out.status, R.OK)
        self.assertEqual(out.n_misaligned, 0)
        text = tok.decode(out.input_ids[out.loss_mask == 1])
        self.assertIn("<tool_call>", text)
        self.assertIn("Final answer.<|im_end|>", text)
        for ctx in ("RESULT-TEXT", "TOOLDESC", "find x"):
            self.assertNotIn(ctx, text)


if __name__ == "__main__":
    unittest.main()
