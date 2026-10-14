import unittest

from dataset.sft.loader.config import ConfigError, MixConfig

BASE = {"domains": {"a": {"weight": 3, "sources": {"x": {}, "y": {}}}, "b": {"weight": 1, "sources": ["z"]}}}
AVAIL = {"x": 900.0, "y": 100.0, "z": 50.0}


class ConfigTests(unittest.TestCase):
    def test_alpha_smoothing_within_domain_and_domain_weights(self):
        p = MixConfig.from_dict({**BASE, "alpha": 0.5}).stage_weights(0, AVAIL)
        self.assertAlmostEqual(sum(p.values()), 1.0)
        self.assertAlmostEqual(p["x"] + p["y"], 0.75)
        self.assertAlmostEqual(p["z"], 0.25)
        self.assertAlmostEqual(p["x"] / p["y"], 3.0)            # sqrt(900/100)
        flat = MixConfig.from_dict({**BASE, "alpha": 1.0}).stage_weights(0, AVAIL)
        self.assertAlmostEqual(flat["x"] / flat["y"], 9.0)     # proportional to tokens
        uniform = MixConfig.from_dict({**BASE, "alpha": 0.0}).stage_weights(0, AVAIL)
        self.assertAlmostEqual(uniform["x"], uniform["y"])

    def test_explicit_weights_cap_a_big_source(self):
        raw = {"domains": {"a": {"weight": 1, "sources": {"x": {"weight": 1}, "y": {"weight": 1}}}}}
        p = MixConfig.from_dict(raw).stage_weights(0, AVAIL)
        self.assertAlmostEqual(p["x"], 0.5)

    def test_all_or_none_explicit_weights(self):
        raw = {"domains": {"a": {"weight": 1, "sources": {"x": {"weight": 1}, "y": {}}}}}
        with self.assertRaisesRegex(ConfigError, "all of its sources or on none"):
            MixConfig.from_dict(raw)

    def test_unavailable_sources_are_dropped_and_weights_renormalized(self):
        p = MixConfig.from_dict(BASE).stage_weights(0, {"x": 10.0, "y": 0.0, "z": 5.0})
        self.assertNotIn("y", p)
        self.assertAlmostEqual(sum(p.values()), 1.0)
        self.assertAlmostEqual(MixConfig.from_dict(BASE).stage_weights(0, {"z": 5.0})["z"], 1.0)

    def test_stage_overrides(self):
        cfg = MixConfig.from_dict({**BASE, "stages": [
            {"name": "broad", "until_tokens": 1000},
            {"name": "anneal", "domain_weights": {"a": 1, "b": 3}}]})
        late = cfg.stage_weights(1, AVAIL)
        self.assertAlmostEqual(late["z"], 0.75)
        self.assertAlmostEqual(cfg.stage_weights(0, AVAIL)["z"], 0.25)

    def test_validation_errors(self):
        bad = [
            ({"domains": {}}, "at least one domain"),
            ({"domains": {"a": {"sources": ["x"]}}}, "`weight` is required"),
            ({"domains": {"a": {"weight": 1, "sources": ["x"]}, "b": {"weight": 1, "sources": ["x"]}}}, "listed in domains"),
            ({**BASE, "over_length": "wrap"}, "over_length"),
            ({**BASE, "seq_len": 4}, "seq_len"),
            ({**BASE, "typo_key": 1}, "unknown config keys"),
            ({**BASE, "stages": [{"until_tokens": 5}, {"until_tokens": 5}]}, "strictly increasing"),
            ({**BASE, "stages": [{"name": "s"}, {"until_tokens": 5}]}, "only the last stage"),
            ({**BASE, "stages": [{"until_tokens": 5, "source_weights": {"nope": 1}}]}, "unknown source"),
            ({**BASE, "stages": [{"until_tokens": 5, "domain_weights": {"nope": 1}}]}, "unknown domain"),
            ({"domains": {"a": {"weight": 0, "sources": ["x"]}}}, "all domain weights are 0"),
            ({"domains": {"a": {"weight": 1, "sources": {"x": {"bogus": 1}}}}}, "unknown keys"),
        ]
        for raw, msg in bad:
            with self.assertRaisesRegex(ConfigError, msg):
                MixConfig.from_dict(raw)

    def test_data_hash_ignores_paths_but_tracks_stream_changing_fields(self):
        a = MixConfig.from_dict({**BASE, "data_root": "a", "cache_dir": "c1", "micro_batch_size": 1})
        b = MixConfig.from_dict({**BASE, "data_root": "b", "cache_dir": "c2", "micro_batch_size": 4})
        self.assertEqual(a.data_hash(), b.data_hash())
        for change in ({"seed": 1}, {"seq_len": 512}, {"alpha": 0.7}, {"max_epochs": 2.0}):
            self.assertNotEqual(a.data_hash(), MixConfig.from_dict({**BASE, **change}).data_hash(), change)
        self.assertNotEqual(a.data_hash("tmplA"), a.data_hash("tmplB"))

    def test_baseline_yaml_loads(self):
        from pathlib import Path
        cfg = MixConfig.from_yaml(Path(__file__).parents[1] / "mixes" / "baseline.yaml")
        self.assertAlmostEqual(sum(d.weight for d in cfg.domains), 1.0)


if __name__ == "__main__":
    unittest.main()
