import json
import unittest

import numpy as np

from dataset.sft.loader.catalog import build_catalog, discover, split_scores
from dataset.sft.loader.compile import CompileJob, make_cache_spec
from dataset.sft.loader.config import ConfigError, MixConfig
from dataset.sft.loader.store import CacheError, SourceStore
from datasets.sft.loader.helpers import INCONSISTENT, Workspace, write_source


class CompileTests(unittest.TestCase):
    def setUp(self):
        self.ws = Workspace().populate()
        self.cfg = MixConfig.from_dict(self.ws.cfg_dict())
        self.addCleanup(self.ws.close)

    def test_cache_rows_align_with_jsonl_rows_across_chunks_and_shards(self):
        spec, *_ = self.ws.compile(self.cfg, chunk_rows=4)
        store = SourceStore(spec.source_dir("short"))
        self.assertEqual(store.n, len(self.ws.short))
        self.assertEqual(store.meta.id.tolist(), [s["id"] for s in self.ws.short])
        shard = lambda stem: [json.loads(l) for l in open(self.ws.data / "g1/short" / f"{stem}.jsonl")]
        for i in (0, 3, 4, 7, 20, store.n - 1):
            stem, row = store.locate(i)
            self.assertEqual(shard(stem)[row]["id"], self.ws.short[i]["id"])
            ids, mask = store.get(i)
            self.assertEqual(len(ids), store.meta.n_tokens.iat[i])
            self.assertEqual(int(mask.sum()), store.meta.n_trainable.iat[i])

    def test_parallel_spawn_compile_matches_serial(self):
        a, *_ = self.ws.compile(self.cfg, workers=1, chunk_rows=10)
        serial = {n: SourceStore(a.source_dir(n)) for n in ("short", "toolish")}
        snap = {n: [s.get(i) for i in range(min(s.n, 25))] for n, s in serial.items()}
        b, *_ = self.ws.compile(self.cfg, workers=2, chunk_rows=10, force=True)
        for n in snap:
            st = SourceStore(b.source_dir(n))
            for i, (ids, mask) in enumerate(snap[n]):
                self.assertTrue(np.array_equal(ids, st.get(i)[0]) and np.array_equal(mask, st.get(i)[1]))

    def test_resume_recompiles_only_missing_parts_and_prunes_stale_layouts(self):
        spec, *_ = self.ws.compile(self.cfg, chunk_rows=5)
        d = spec.source_dir("short")
        n_parts = len(list(d.glob("part-*.done.json")))
        victim = sorted(d.glob("part-*.done.json"))[1]
        victim.unlink()
        from dataset.sft.loader.compile import plan_tasks
        found = discover(self.cfg.data_root)
        job = CompileJob("short", found["short"].path, None, found["short"].shard_rows)
        tasks, expected = plan_tasks(job, spec, 5, False)
        self.assertEqual(len(tasks), 1)
        self.compile_again = self.ws.compile(self.cfg, chunk_rows=5)
        self.assertEqual(len(list(d.glob("part-*.done.json"))), n_parts)
        self.ws.compile(self.cfg, chunk_rows=13)       # different layout: old parts must not linger as duplicates
        self.assertEqual(SourceStore(d).n, len(self.ws.short))

    def test_cache_is_keyed_by_tokenizer_and_template(self):
        a, *_ = self.ws.compile(self.cfg)
        with Workspace(INCONSISTENT) as other:
            other.populate()
            b, *_ = other.compile(MixConfig.from_dict(other.cfg_dict()))
        self.assertNotEqual(a.fingerprint, b.fingerprint)
        self.assertNotEqual(a.root, b.root)

    def test_inconsistent_template_rejects_exactly_the_multi_assistant_samples(self):
        with Workspace(INCONSISTENT) as w:
            w.populate(n_short=40, n_tool=5, n_long=5)
            cfg = MixConfig.from_dict(w.cfg_dict())
            spec, *_ = w.compile(cfg)
            status = SourceStore(spec.source_dir("short")).meta.status.to_numpy()
            multi = np.array([sum(m["role"] == "assistant" for m in s["messages"]) >= 2 for s in w.short])
            self.assertTrue(multi.any() and (~multi).any())
            self.assertTrue((status[multi] == 1).all())        # render_inconsistent: never silently mis-masked
            self.assertTrue((status[~multi] == 0).all())
            cat = build_catalog(cfg, spec)
            v = cat.by_name()["short"]
            self.assertTrue((v.store.meta.status.to_numpy()[v.train_idx] == 0).all())
            self.assertEqual(v.n_rejected["render_failed"], int(multi.sum()))

    def test_stale_cache_detected_when_a_shard_changes(self):
        spec, *_ = self.ws.compile(self.cfg)
        build_catalog(self.cfg, spec)
        shard = self.ws.data / "g1/short/part-00000.jsonl"
        shard.write_text(shard.read_text() + "")        # same bytes: still valid
        build_catalog(self.cfg, spec)
        lines = shard.read_text().splitlines()
        shard.write_text("\n".join(lines[:-1]) + "\n")
        with self.assertRaisesRegex(CacheError, "stale or incomplete"):
            build_catalog(self.cfg, spec)

    def test_uncompiled_source_is_a_clear_error(self):
        spec, *_ = make_cache_spec(self.cfg.tokenizer, None, self.cfg.cache_dir)
        with self.assertRaisesRegex(CacheError, "not compiled"):
            build_catalog(self.cfg, spec)


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.ws = Workspace().populate()
        self.addCleanup(self.ws.close)

    def catalog(self, **over):
        cfg = MixConfig.from_dict(self.ws.cfg_dict(**over))
        spec, *_ = self.ws.compile(cfg)
        return cfg, build_catalog(cfg, spec)

    def test_split_is_deterministic_disjoint_and_independent_of_mix(self):
        _, a = self.catalog()
        cfg2 = MixConfig.from_dict(self.ws.cfg_dict(seed=7, alpha=0.9, domains={
            "only": {"weight": 1, "sources": {"short": {}}}}))
        spec, *_ = self.ws.compile(cfg2)
        b = build_catalog(cfg2, spec)
        va, vb = a.by_name()["short"], b.by_name()["short"]
        self.assertTrue(np.array_equal(va.val_idx, vb.val_idx))
        self.assertEqual(len(set(va.train_idx) & set(va.val_idx)), 0)
        self.assertEqual(len(va.train_idx) + len(va.val_idx), len(self.ws.short))
        self.assertGreater(len(va.val_idx), 0)
        s1 = split_scores(["ab" * 8, "cd" * 8], ["a", "b"], 0)
        self.assertTrue(np.array_equal(s1, split_scores(["ab" * 8, "cd" * 8], ["a", "b"], 0)))
        self.assertFalse(np.array_equal(s1, split_scores(["ab" * 8, "cd" * 8], ["a", "b"], 1)))

    def test_val_cap_keeps_lowest_scores_and_changing_seed_changes_split(self):
        _, a = self.catalog(val_max_samples_per_source=3, val_fraction=0.5)
        self.assertEqual(len(a.by_name()["short"].val_idx), 3)
        _, b = self.catalog(val_max_samples_per_source=3, val_fraction=0.5, split_seed=5)
        self.assertFalse(np.array_equal(a.by_name()["short"].val_idx, b.by_name()["short"].val_idx))

    def test_filters(self):
        cfg, cat = self.catalog(over_length="drop", seq_len=64)
        self.assertTrue(all(w.store.n_tokens[w.train_idx].max() <= 64 for w in cat.views))
        self.assertIn("longdocs", cat.skipped)                       # every long doc exceeds 64 tokens
        short = cat.by_name()["short"]
        self.assertGreater(short.n_rejected["over_seq_len"], 0)
        spec, *_ = self.ws.compile(MixConfig.from_dict(self.ws.cfg_dict()))
        med = int(np.median(SourceStore(spec.source_dir("short")).n_tokens))
        _, cat2 = self.catalog(domains={"c": {"weight": 1, "sources": {
            "short": {"max_tokens": med}, "toolish": {"min_trainable_tokens": 10 ** 6}}}})
        self.assertEqual([v.name for v in cat2.views], ["short"])
        self.assertIn("toolish", cat2.skipped)
        v = cat2.views[0]
        self.assertTrue((v.store.n_tokens[v.train_idx] <= med).all())
        self.assertGreater(v.n_rejected["over_max_tokens"], 0)

    def test_tool_invalid_and_lang_filters(self):
        def flag(s):
            s["meta"]["tool_valid"] = s["id"].endswith(("0", "1", "2"))
            s["meta"]["lang"] = "en" if s["id"].endswith(("0", "5")) else "fr"
        ws = Workspace()
        self.addCleanup(ws.close)
        write_source(ws.data, "g", "flagged", 60, mutate=flag)
        base = {"domains": {"d": {"weight": 1, "sources": {"flagged": {}}}}}
        spec, *_ = ws.compile(MixConfig.from_dict(ws.cfg_dict(**base)))
        both = lambda **kw: build_catalog(MixConfig.from_dict(ws.cfg_dict(
            val_fraction=0, domains={"d": {"weight": 1, "sources": {"flagged": kw}}})), spec).views[0]
        ids = lambda v: {int(v.store.meta.id.iat[i].split(":")[1]) % 10 for i in v.train_idx}
        self.assertTrue(ids(both()) <= {0, 1, 2})
        self.assertTrue(ids(both(require_tool_valid=False)) > {0, 1, 2})
        self.assertEqual(ids(both(langs=["en"], require_tool_valid=False)), {0, 5})

    def test_license_deny_skips_source_and_missing_source_errors(self):
        ws = Workspace()
        self.addCleanup(ws.close)
        write_source(ws.data, "g", "ok", 20)
        write_source(ws.data, "g", "nc", 20, license_name="CC-BY-NC-4.0 (verify!)")
        raw = ws.cfg_dict(license_deny=["cc-by-nc"], domains={"d": {"weight": 1, "sources": ["ok", "nc"]}})
        cfg = MixConfig.from_dict(raw)
        spec, *_ = ws.compile(cfg)
        cat = build_catalog(cfg, spec)
        self.assertEqual([v.name for v in cat.views], ["ok"])
        self.assertIn("license", cat.skipped["nc"])
        with self.assertRaisesRegex(ConfigError, "not found"):
            build_catalog(MixConfig.from_dict(ws.cfg_dict(domains={"d": {"weight": 1, "sources": ["ghost"]}})), spec)

    def test_benchmark_overlap_flags_are_dropped_and_staleness_is_detected(self):
        import hashlib
        import pyarrow as pa
        import pyarrow.parquet as pq
        ws = Workspace()
        self.addCleanup(ws.close)
        write_source(ws.data, "g", "src", 30)
        src = ws.data / "g" / "src"
        base = dict(val_fraction=0, domains={"d": {"weight": 1, "sources": ["src"]}})
        spec, *_ = ws.compile(MixConfig.from_dict(ws.cfg_dict(**base)))
        cat = lambda **kw: build_catalog(MixConfig.from_dict(ws.cfg_dict(**base, **kw)), spec)

        self.assertEqual(len(cat().views[0].train_idx), 30)        # if_present: no file -> unchecked
        with self.assertRaisesRegex(ConfigError, "decontaminate"):
            cat(decontam="required")

        def flag(ids, cov, manifest_sha=None):
            sha = manifest_sha or hashlib.sha256((src / "_manifest.json").read_bytes()).hexdigest()
            pq.write_table(pa.table({"id": ids, "max_item_coverage": cov}).replace_schema_metadata(
                {"manifest_sha256": sha}), src / "_contamination.parquet")

        flag(["src:3", "src:7"], [1.0, 0.1])
        v = cat(decontam="required").views[0]
        kept = set(v.store.meta.id.iloc[v.train_idx])
        self.assertTrue({"src:3", "src:7"}.isdisjoint(kept))
        self.assertEqual(v.n_rejected["benchmark_overlap"], 2)
        v = cat(decontam_min_coverage=0.5).views[0]
        self.assertEqual(v.n_rejected["benchmark_overlap"], 1)
        self.assertEqual(len(cat(decontam="off").views[0].train_idx), 30)
        flag(["src:3"], [1.0], manifest_sha="0" * 64)
        with self.assertRaisesRegex(ConfigError, "predates"):
            cat()

    def test_kept_length_is_exact_and_ends_on_an_assistant_turn(self):
        _, cat = self.catalog(seq_len=100)
        cut_any = 0
        for v in cat.views:
            self.assertTrue((v.eff_len <= 100).all())
            self.assertTrue(np.array_equal(v.eff_len, v.cut[v.train_idx]))
            for i in v.train_idx[:60]:
                n, c = int(v.store.n_tokens[i]), int(v.cut[i])
                if n <= 100:
                    self.assertEqual(c, n)
                else:
                    cut_any += 1
                    mask = np.asarray(v.store.get(int(i))[1])
                    self.assertEqual(mask[c - 1], 1)                      # kept text ends with a trained token ...
                    self.assertEqual(mask[c], 0)                          # ... of a complete assistant turn
        self.assertGreater(cut_any, 0)


if __name__ == "__main__":
    unittest.main()
