import unittest

import numpy as np

from dataset.sft.loader.catalog import build_catalog
from dataset.sft.loader.config import MixConfig
from dataset.sft.loader.metrics import Metrics
from dataset.sft.loader.packer import IGNORE, Packer, truncate_at_turn
from dataset.sft.loader.sampler import MixtureSampler
from datasets.sft.loader.helpers import Workspace, write_source


def build(ws, **over):
    cfg = MixConfig.from_dict(ws.cfg_dict(**over))
    spec, tok, _ = ws.compile(cfg)
    cat = build_catalog(cfg, spec)
    m = Metrics([v.name for v in cat.views], [v.domain for v in cat.views])
    packer = Packer(MixtureSampler(cfg, cat.views), cat.views, cfg.seq_len, cfg.pack_window, cfg.over_length,
                    cfg.min_truncated_trainable, 0, m)
    return cfg, cat, packer, m


class TruncateTests(unittest.TestCase):
    MASK = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1], dtype=np.uint8)

    def test_cuts_at_the_last_complete_assistant_span_that_fits(self):
        self.assertEqual(truncate_at_turn(self.MASK, 10, 1), 9)     # span [6, 9) ends at 9
        self.assertEqual(truncate_at_turn(self.MASK, 8, 1), 4)      # span [6, 9) does not fit -> previous one
        self.assertEqual(truncate_at_turn(self.MASK, 13, 1), 13)
        self.assertEqual(truncate_at_turn(self.MASK[:], 3, 1), None)  # first span does not fit
        self.assertEqual(truncate_at_turn(np.zeros(9, np.uint8), 5, 1), None)

    def test_min_trainable_threshold(self):
        self.assertEqual(truncate_at_turn(self.MASK, 10, 5), 9)
        self.assertIsNone(truncate_at_turn(self.MASK, 4, 5))
        self.assertIsNone(truncate_at_turn(self.MASK, 10, 6))

    def test_span_ending_exactly_at_the_limit_and_next_token_masked(self):
        m = np.array([0, 1, 1, 1, 1, 1], dtype=np.uint8)
        self.assertIsNone(truncate_at_turn(m, 5, 1))                # span continues past the limit -> cannot cut inside it
        self.assertEqual(truncate_at_turn(m, 6, 1), 6)


class PackerTests(unittest.TestCase):
    def setUp(self):
        self.ws = Workspace().populate(n_short=150, n_tool=100, n_long=30)
        self.addCleanup(self.ws.close)

    def packs(self, packer, n):
        out = []
        for _ in range(n):
            p = packer.next_pack()
            if p is None:
                break
            out.append(p)
        return out

    def test_invariants_hold_for_every_pack(self):
        cfg, cat, packer, m = build(self.ws, max_epochs=5)
        L = cfg.seq_len
        for p in self.packs(packer, 80):
            self.assertEqual(len(p.input_ids), L)
            real = p.doc_ids >= 0
            self.assertEqual(int(real.sum()), p.n_real)
            self.assertTrue(real[:p.n_real].all() and not real[p.n_real:].any())    # padding only at the tail
            self.assertTrue((p.labels[~real] == IGNORE).all())
            self.assertTrue(((p.labels == IGNORE) | (p.labels == p.input_ids)).all())
            o = 0
            for d, (src, idx, n) in enumerate(zip(p.doc_src, p.doc_idx, p.doc_len)):
                ids, mask = cat.views[src].store.get(idx)
                np.testing.assert_array_equal(p.input_ids[o:o + n], ids[:n])        # documents are intact, in one piece
                np.testing.assert_array_equal(p.position_ids[o:o + n], np.arange(n))
                self.assertTrue((p.doc_ids[o:o + n] == d).all())
                self.assertEqual(p.labels[o], IGNORE)                               # nothing predicts a doc's first token
                np.testing.assert_array_equal(p.labels[o:o + n] != IGNORE, np.asarray(mask[:n]) == 1)
                o += n
            self.assertEqual(p.n_trainable, int((p.labels != IGNORE).sum()))

    def test_fill_rate_is_high_with_a_window_and_low_without_one(self):
        fills = {}
        for window in (1, 16, 256):
            _, _, packer, m = build(self.ws, max_epochs=10, pack_window=window,
                                    domains={"c": {"weight": 1, "sources": {"short": {}}}})
            self.packs(packer, 40)
            fills[window] = m.snapshot()["fill_rate"]
        self.assertGreater(fills[256], 0.94)
        self.assertLess(fills[1], 0.7)                         # no window = first-come packing wastes a third of the tokens
        self.assertLess(fills[1], fills[16])
        self.assertLess(fills[16], fills[256])

    def test_long_documents_are_cut_at_assistant_turn_boundaries(self):
        cfg, cat, packer, m = build(self.ws, max_epochs=3, seq_len=200)
        end_id = 2   # <|im_end|> in the tiny tokenizer's vocab
        seen_cut = 0
        for p in self.packs(packer, 120):
            o = 0
            for src, idx, n in zip(p.doc_src, p.doc_idx, p.doc_len):
                full = cat.views[src].store.n_tokens[idx]
                if n < full:
                    seen_cut += 1
                    self.assertNotEqual(p.labels[o + n - 1], IGNORE)                # last kept token is a trained one ...
                    self.assertEqual(p.input_ids[o + n - 1], end_id)                # ... the assistant turn's <|im_end|>
                o += n
        self.assertGreater(seen_cut, 0)
        self.assertEqual(sum(m.truncated), seen_cut)

    def test_document_without_a_fitting_assistant_turn_is_rejected_by_the_catalog_not_wasted_by_the_sampler(self):
        ws = Workspace()
        self.addCleanup(ws.close)

        def huge_prompt(s):
            for m in s["messages"]:
                if m["role"] == "user":
                    m["content"] = "word1 " * 400
        write_source(ws.data, "g", "hopeless", 12, mutate=huge_prompt)
        write_source(ws.data, "g", "fine", 40)
        cfg = MixConfig.from_dict(ws.cfg_dict(domains={"c": {"weight": 1, "sources": ["hopeless", "fine"]}}, seq_len=128))
        spec, *_ = ws.compile(cfg)
        cat = build_catalog(cfg, spec)
        self.assertEqual([v.name for v in cat.views], ["fine"])
        self.assertIn("no_assistant_turn_fits_seq_len", cat.skipped["hopeless"])

    def test_sampler_accounting_equals_emitted_tokens_exactly(self):
        cfg, cat, packer, m = build(self.ws, max_epochs=1, seq_len=200)
        packs = self.packs(packer, 10 ** 4)
        emitted = sum(p.n_real for p in packs)
        planned = sum(v.train_tokens for v in cat.views)
        self.assertEqual(emitted, planned)                         # nothing silently dropped or double counted
        self.assertEqual(int(m.dropped_overlength.sum()), 0)

    def test_drop_policy_never_emits_over_long_documents(self):
        cfg, cat, packer, m = build(self.ws, over_length="drop", seq_len=150, max_epochs=3)
        for p in self.packs(packer, 40):
            for src, idx, n in zip(p.doc_src, p.doc_idx, p.doc_len):
                self.assertEqual(n, cat.views[src].store.n_tokens[idx])
        self.assertEqual(int(m.truncated.sum()), 0)

    def test_state_roundtrip_gives_identical_packs(self):
        cfg, cat, packer, m = build(self.ws, max_epochs=4)
        self.packs(packer, 7)
        st, sst = packer.state_dict(), packer.sampler.state_dict()
        a = self.packs(packer, 6)
        _, _, p2, _ = build(self.ws, max_epochs=4)
        p2.sampler.load_state_dict(sst)
        p2.load_state_dict(st)
        for x, y in zip(a, self.packs(p2, 6)):
            np.testing.assert_array_equal(x.input_ids, y.input_ids)
            np.testing.assert_array_equal(x.labels, y.labels)

    def test_stream_ends_with_a_padded_final_pack(self):
        cfg, cat, packer, m = build(self.ws, max_epochs=1)
        packs = self.packs(packer, 10 ** 4)
        self.assertGreater(len(packs), 3)
        self.assertLess(packs[-1].n_real, cfg.seq_len)
        self.assertIsNone(packer.next_pack())
        real_docs = sum(len(p.doc_len) for p in packs) + int(m.dropped_overlength.sum())
        self.assertEqual(real_docs, sum(len(v.train_idx) for v in cat.views))      # every training doc used exactly once


if __name__ == "__main__":
    unittest.main()
