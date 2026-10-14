import unittest
from collections import Counter

import numpy as np

from dataset.sft.loader.catalog import SourceView
from dataset.sft.loader.config import MixConfig, SourceCfg
from dataset.sft.loader.sampler import MixtureSampler, SequentialSampler, DocRef, simulate


def make_views(spec: dict[str, tuple[str, int, int]], cfg: MixConfig):
    """spec: name -> (domain, n_docs, mean_len)"""
    rng = np.random.default_rng(1)
    views = []
    for name, (dom, n, mean) in spec.items():
        eff = np.maximum(1, rng.normal(mean, mean * 0.3, n)).astype(np.int64)
        views.append(SourceView(name, dom, SourceCfg(name), None, np.arange(n, dtype=np.int64),
                                np.zeros(0, np.int64), eff))
    return views


def cfg_for(domains, **over):
    return MixConfig.from_dict({"domains": domains, "seq_len": 10 ** 6, **over})


SPEC = {"a": ("A", 4000, 50), "b": ("B", 3000, 500), "c": ("C", 400, 3000)}
DOMAINS = {"A": {"weight": 0.5, "sources": ["a"]}, "B": {"weight": 0.3, "sources": ["b"]}, "C": {"weight": 0.2, "sources": ["c"]}}


def drain(s, n):
    out = []
    for _ in range(n):
        r = s.next()
        if r is None:
            break
        out.append(r)
    return out


class SamplerTests(unittest.TestCase):
    def test_realized_token_shares_track_targets_despite_different_doc_lengths(self):
        cfg = cfg_for(DOMAINS, max_epochs=100)
        views = make_views(SPEC, cfg)
        refs = drain(MixtureSampler(cfg, views), 20000)
        tok = Counter()
        for r in refs:
            tok[views[r.src].name] += r.length
        total = sum(tok.values())
        for name, target in (("a", 0.5), ("b", 0.3), ("c", 0.2)):
            self.assertAlmostEqual(tok[name] / total, target, delta=0.01, msg=name)
        # sample counts are NOT the mix: the long-doc source supplies far fewer documents
        docs = Counter(views[r.src].name for r in refs)
        self.assertGreater(docs["a"], 10 * docs["c"])

    def test_deviation_from_target_is_bounded_by_document_length_at_every_prefix(self):
        cfg = cfg_for(DOMAINS, max_epochs=100)
        views = make_views(SPEC, cfg)
        longest = max(int(v.eff_len.max()) for v in views)
        s = MixtureSampler(cfg, views)
        target = np.array([0.5, 0.3, 0.2])
        tok, total, worst = np.zeros(3), 0, 0.0
        for _ in range(60000):
            r = s.next()
            tok[r.src] += r.length
            total += r.length
            worst = max(worst, float(np.abs(tok - target * total).max()))
        self.assertLessEqual(worst, longest)                     # absolute token error never exceeds one long document
        self.assertLess(float(np.abs(tok / total - target).max()), 0.002)

    def test_deterministic_and_seed_changes_order_not_shares(self):
        cfg = cfg_for(DOMAINS, max_epochs=100)
        views = make_views(SPEC, cfg)
        a, b = drain(MixtureSampler(cfg, views), 500), drain(MixtureSampler(cfg, views), 500)
        self.assertEqual(a, b)
        c = drain(MixtureSampler(cfg, views, seed=99), 500)
        self.assertNotEqual([r.idx for r in a], [r.idx for r in c])
        def shares(refs):
            t = Counter()
            for r in refs:
                t[r.src] += r.length
            return np.array([t[i] for i in range(3)]) / sum(t.values())
        far = drain(MixtureSampler(cfg, views, seed=99), 20000)
        np.testing.assert_allclose(shares(far), [0.5, 0.3, 0.2], atol=0.01)   # seed changes order, not the mix

    def test_resume_is_exact_across_an_exhaustion_event(self):
        cfg = cfg_for(DOMAINS, max_epochs=100)
        cfg.domains[2].sources[0].max_epochs = 0.5
        views = make_views(SPEC, cfg)
        ref = MixtureSampler(cfg, views)
        refs = drain(ref, 60000)
        c_positions = [i for i, r in enumerate(refs) if r.src == 2]
        self.assertEqual(len(c_positions), 200)
        cut = c_positions[-1]                                   # index of the last c document
        for stop in (cut - 1, cut, cut + 1, cut + 2):
            s = MixtureSampler(cfg, views)
            drain(s, stop)
            t = MixtureSampler(cfg, views)
            t.load_state_dict(s.state_dict())
            self.assertEqual(drain(t, 400), refs[stop:stop + 400], stop)

    def test_resume_is_exact(self):
        cfg = cfg_for(DOMAINS, max_epochs=100)
        views = make_views(SPEC, cfg)
        s = MixtureSampler(cfg, views)
        drain(s, 777)
        st = s.state_dict()
        rest = drain(s, 600)
        t = MixtureSampler(cfg, views)
        t.load_state_dict(st)
        self.assertEqual(drain(t, 600), rest)
        with self.assertRaises(ValueError):
            t.load_state_dict({**st, "served": [0]})

    def test_each_epoch_visits_every_document_once_then_reshuffles(self):
        cfg = cfg_for({"A": {"weight": 1, "sources": ["a"]}}, max_epochs=2)
        views = make_views({"a": ("A", 50, 10)}, cfg)
        refs = drain(MixtureSampler(cfg, views), 1000)
        self.assertEqual(len(refs), 100)
        e1, e2 = [r.idx for r in refs[:50]], [r.idx for r in refs[50:]]
        self.assertEqual(sorted(e1), list(range(50)))
        self.assertEqual(sorted(e2), list(range(50)))
        self.assertNotEqual(e1, e2)

    def test_epoch_cap_exhausts_a_source_and_renormalizes_the_rest(self):
        cfg = cfg_for(DOMAINS, max_epochs=100)
        cfg.domains[2].sources[0].max_epochs = 1.5
        views = make_views(SPEC, cfg)
        s = MixtureSampler(cfg, views)
        refs = drain(s, 150000)
        served_c = sum(r.src == 2 for r in refs)
        self.assertEqual(served_c, 600)                                   # 1.5 epochs x 400 docs
        last_c = max(i for i, r in enumerate(refs) if r.src == 2)
        tail = Counter(r.src for r in refs[last_c + 1:last_c + 3000])
        self.assertGreater(len(refs) - last_c, 3000)
        self.assertNotIn(2, tail)
        self.assertGreater(tail[0], 0)
        self.assertGreater(tail[1], 0)
        after = refs[last_c + 1:last_c + 3000]
        tok = np.array([sum(r.length for r in after if r.src == i) for i in (0, 1)], dtype=float)
        np.testing.assert_allclose(tok / tok.sum(), [0.625, 0.375], atol=0.04)   # renormalized immediately, no catch-up burst

    def test_stream_ends_when_everything_is_capped(self):
        cfg = cfg_for(DOMAINS, max_epochs=1)
        views = make_views(SPEC, cfg)
        s = MixtureSampler(cfg, views)
        refs = drain(s, 10 ** 6)
        self.assertEqual(len(refs), 4000 + 3000 + 400)
        self.assertIsNone(s.next())
        self.assertTrue(s.exhausted)

    def test_ranks_get_disjoint_complete_partitions_each_epoch(self):
        cfg = cfg_for({"A": {"weight": 1, "sources": ["a", "b"]}}, max_epochs=1)
        views = make_views({"a": ("A", 101, 10), "b": ("A", 55, 20)}, cfg)
        seen = {}
        for rank in range(3):
            for r in drain(MixtureSampler(cfg, views, rank, 3), 10 ** 5):
                key = (r.src, r.idx)
                self.assertNotIn(key, seen, "a document was handed to two ranks")
                seen[key] = rank
        self.assertEqual(len(seen), 101 + 55)
        per_rank = Counter(seen.values())
        self.assertLessEqual(max(per_rank.values()) - min(per_rank.values()), 2)

    def test_stage_switch_resets_deficits_so_new_mix_applies_immediately(self):
        cfg = cfg_for({"A": {"weight": 1, "sources": ["a"]}, "B": {"weight": 1, "sources": ["b"]}}, max_epochs=100,
                      stages=[{"name": "s0", "until_tokens": 20000, "domain_weights": {"A": 1, "B": 0}},
                              {"name": "s1", "domain_weights": {"A": 1, "B": 1}}])
        views = make_views({"a": ("A", 5000, 20), "b": ("B", 5000, 20)}, cfg)
        s = MixtureSampler(cfg, views)
        refs = drain(s, 3000)
        first = [r for r in refs if r.stage == 0]
        self.assertTrue(first and all(r.src == 0 for r in first))
        self.assertGreaterEqual(sum(r.length for r in first), 20000)
        second = [r for r in refs if r.stage == 1]
        window = Counter(r.src for r in second[:40])
        self.assertAlmostEqual(window[0] / 40, 0.5, delta=0.1)           # no catch-up burst of B
        self.assertEqual(s.stage, 1)

    def test_stage_boundary_is_global_tokens_split_across_ranks(self):
        cfg = cfg_for({"A": {"weight": 1, "sources": ["a"]}, "B": {"weight": 1, "sources": ["b"]}}, max_epochs=100,
                      stages=[{"until_tokens": 40000, "domain_weights": {"A": 1, "B": 0}}, {"domain_weights": {"A": 0, "B": 1}}])
        views = make_views({"a": ("A", 5000, 20), "b": ("B", 5000, 20)}, cfg)
        for world in (1, 4):
            refs = drain(MixtureSampler(cfg, views, 0, world), 4000)
            tokens_stage0 = sum(r.length for r in refs if r.stage == 0)
            self.assertAlmostEqual(tokens_stage0 * world, 40000, delta=40000 * 0.05)

    def test_exhausted_stage_falls_back_to_remaining_sources(self):
        cfg = cfg_for({"A": {"weight": 1, "sources": ["a"]}, "B": {"weight": 1, "sources": ["b"]}}, max_epochs=100,
                      stages=[{"until_tokens": 10, "domain_weights": {"A": 1, "B": 0}}, {"domain_weights": {"A": 0, "B": 1}}])
        cfg.domains[1].sources[0].max_epochs = 1
        views = make_views({"a": ("A", 400, 20), "b": ("B", 30, 20)}, cfg)
        refs = drain(MixtureSampler(cfg, views), 600)
        self.assertEqual(sum(r.src == 1 for r in refs), 30)
        self.assertEqual(refs[-1].src, 0)                                  # kept training instead of stopping

    def test_simulate_reports_epochs_and_unreachable_budgets(self):
        cfg = cfg_for(DOMAINS, max_epochs=1)
        views = make_views(SPEC, cfg)
        total_tokens = sum(v.train_tokens for v in views)
        ok = simulate(cfg, views, total_tokens * 0.05)
        self.assertTrue(ok["reached"])
        self.assertLess(max(ok["epochs"].values()), 1.0)
        over = simulate(cfg, views, total_tokens * 3)
        self.assertFalse(over["reached"])
        self.assertLess(over["achieved_tokens"], total_tokens * 3)
        shares = ok["tokens_per_rank"] / ok["tokens_per_rank"].sum()
        np.testing.assert_allclose(shares, [0.5, 0.3, 0.2], atol=0.02)

    def test_sequential_sampler(self):
        refs = [DocRef(0, i, 5) for i in range(4)]
        s = SequentialSampler(refs)
        self.assertEqual([s.next().idx for _ in range(2)], [0, 1])
        st = s.state_dict()
        t = SequentialSampler(refs)
        t.load_state_dict(st)
        self.assertEqual([t.next().idx, t.next().idx, t.next()], [2, 3, None])


if __name__ == "__main__":
    unittest.main()
