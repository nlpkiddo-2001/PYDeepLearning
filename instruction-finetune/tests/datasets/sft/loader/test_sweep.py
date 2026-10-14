import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from datasets.sft.loader import sweep
from dataset.sft.loader.config import MixConfig
    
BASE = {"data_root": "out", "tokenizer": "x", "domains": {
    "chat": {"weight": 0.4, "sources": ["a"]}, "coding": {"weight": 0.3, "sources": ["b"]},
    "tool": {"weight": 0.2, "sources": ["c"]}, "role": {"weight": 0.1, "sources": ["d"]}}}
NAMES = list(BASE["domains"])
# synthetic ground truth: interior optimum at TRUE; metrics are smooth functions of the mixture
TRUE = np.array([0.35, 0.30, 0.25, 0.10])


def truth(w):
    return {"val_loss/chat": 2.0 - 1.2 * w[0] + 4 * float(np.sum((w - TRUE) ** 2)),
            "bfcl": 0.2 + 0.9 * w[2] - 2.0 * (w[2] - 0.25) ** 2 - float(np.sum((w - TRUE) ** 2)),
            "noise": 0.5}


def run_cli(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = sweep.main(list(argv))
    return code, out.getvalue()


TRAIN = ("import json,sys,yaml;"
         "c=yaml.safe_load(open(sys.argv[1]));w=[c['domains'][d]['weight'] for d in c['domains']];"
         "import os;sys.path.insert(0,os.environ['REPO']);from datasets.sft.loader.tests.test_sweep import truth;"
         "m=truth([x/sum(w) for x in w]);m['gpu']=os.environ.get('CUDA_VISIBLE_DEVICES','none');"
         "json.dump(m,open(sys.argv[2]+'/metrics.json','w'))")


class SweepTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)
        self.base = self.d / "base.yaml"
        self.base.write_text(yaml.safe_dump(BASE, sort_keys=False))

    def generate(self, n=30, **kw):
        extra = sum(([f"--{k.replace('_', '-')}", str(v)] for k, v in kw.items()), [])
        code, _ = run_cli("generate", "--base", str(self.base), "--out", str(self.d / "sw"), "--n", str(n), *extra)
        self.assertEqual(code, 0)
        return sweep.load_design(self.d / "sw")

    def test_generate_valid_reproducible_floor_and_baseline_first(self):
        a = self.generate(25, min_domain=0.05, seed=3)
        self.assertEqual(len(a), 25)
        self.assertTrue(a[0]["id"].endswith("_base"))
        self.assertAlmostEqual(a[0]["weights"]["chat"], 0.4)
        for it in a:
            w = np.array(list(it["weights"].values()))
            self.assertAlmostEqual(w.sum(), 1.0, places=6)
            self.assertGreaterEqual(w.min(), 0.05 - 1e-9 if it is not a[0] else 0)
            cfg = MixConfig.from_yaml(self.d / "sw" / "mixes" / f"{it['id']}.yaml")        # every file is a valid MixConfig
            self.assertEqual([x.name for x in cfg.domains], NAMES)
        b = self.generate(25, min_domain=0.05, seed=3)
        self.assertEqual(a, b)
        self.assertNotEqual(a[1], self.generate(25, min_domain=0.05, seed=4)[1])
        spread = np.array([list(it["weights"].values()) for it in a])
        self.assertGreater(spread.std(axis=0).min(), 0.02)                                  # explores, not clones the base

    def test_designs_uniform_and_grid(self):
        self.assertEqual(len(self.generate(10, design="uniform")), 10)
        g = self.generate(12, design="grid", min_domain=0.1)
        self.assertGreaterEqual(len(g), 2)
        self.assertTrue(all(abs(sum(it["weights"].values()) - 1) < 1e-6 for it in g))

    def test_run_parallel_with_gpu_slots_resume_and_failures(self):
        self.generate(6)
        import os
        os.environ["REPO"] = str(Path(__file__).parents[2])
        cmd = f"{sys.executable} -c \"{TRAIN}\" {{mix}} {{out}}"
        code, out = run_cli("run", "--dir", str(self.d / "sw"), "--cmd", cmd, "--gpus", "0,1")
        self.assertEqual(code, 0, out)
        gpus = {json.loads((self.d / "sw/runs" / it["id"] / "metrics.json").read_text())["gpu"] for it in sweep.load_design(self.d / "sw")}
        self.assertTrue(gpus <= {"0", "1"} and len(gpus) == 2)          # jobs were spread over the GPU slots
        code, out = run_cli("run", "--dir", str(self.d / "sw"), "--cmd", cmd)
        self.assertIn("0/6 runs to do", out)                            # resumable: finished runs are skipped
        (self.d / "sw/runs/mix_002/metrics.json").unlink()
        self.assertIn("1/6 runs to do", run_cli("run", "--dir", str(self.d / "sw"), "--cmd", cmd, "--gpus", "0;1")[1])
        bad = run_cli("run", "--dir", str(self.d / "sw"), "--cmd", f"{sys.executable} -c \"raise SystemExit(3)\" {{mix}}", "--retry-failed")
        self.assertEqual(bad[0], 1)
        self.assertIn("FAIL", bad[1])

    def test_fit_recovers_the_known_optimum_and_flags_unpredictable_metrics(self):
        design = self.generate(40, seed=1, concentration=10)
        runs = self.d / "sw" / "runs"
        rng = np.random.default_rng(0)
        for it in design:
            (runs / it["id"]).mkdir(parents=True)
            m = truth(np.array(list(it["weights"].values())))
            m["noise"] = float(rng.normal())                              # pure noise: must be flagged
            (runs / it["id"] / "metrics.json").write_text(json.dumps(m))
        obj = {"metrics": {"val_loss/chat": {"direction": "min", "weight": 1}, "bfcl": {"direction": "max", "weight": 1}}}
        (self.d / "obj.yaml").write_text(yaml.safe_dump(obj))
        code, out = run_cli("fit", "--dir", str(self.d / "sw"), "--objective", str(self.d / "obj.yaml"), "--top", "3", "--trust-radius", "0.5")
        self.assertEqual(code, 0, out)
        fit = json.loads((self.d / "sw/fit.json").read_text())
        self.assertGreater(fit["loo_r2"]["val_loss/chat"], 0.9)
        # judge by regret under the TRUE objective (the optimum is not TRUE: the loss has a linear term too)
        X = np.array([list(it["weights"].values()) for it in design])
        obs = {m: np.array([truth(x)[m] for x in X]) for m in ("val_loss/chat", "bfcl")}
        mu, sd = {m: v.mean() for m, v in obs.items()}, {m: v.std() for m, v in obs.items()}
        true_score = lambda w: -(truth(w)["val_loss/chat"] - mu["val_loss/chat"]) / sd["val_loss/chat"] + (truth(w)["bfcl"] - mu["bfcl"]) / sd["bfcl"]
        grid = np.random.default_rng(5).dirichlet(np.ones(4), 60000)
        best_possible = max(true_score(w) for w in grid[:6000])
        best = np.array(list(fit["recommendations"][0]["weights"].values()))
        self.assertGreater(true_score(best), best_possible - 0.15)          # recommended mix is (near-)optimal for real
        self.assertGreater(true_score(best), max(true_score(x) for x in X) - 0.15)   # and not worse than the best run
        self.assertTrue((self.d / "sw/recommended/rec_0.yaml").exists())
        MixConfig.from_yaml(self.d / "sw/recommended/rec_0.yaml")           # recommendation is a valid, runnable config
        self.assertGreater(fit["recommendations"][0]["predicted_score"], fit["recommendations"][2]["predicted_score"] - 1e-9)
        obj["metrics"]["noise"] = {"direction": "min"}
        (self.d / "obj.yaml").write_text(yaml.safe_dump(obj))
        code, out = run_cli("fit", "--dir", str(self.d / "sw"), "--objective", str(self.d / "obj.yaml"), "--trust-radius", "0.5")
        self.assertIn("NOT PREDICTABLE", out)
        self.assertIn("Warning", out)

    def test_fit_respects_constraints_and_trust_region(self):
        design = self.generate(30, seed=2)
        for it in design:
            (self.d / "sw/runs" / it["id"]).mkdir(parents=True)
            (self.d / "sw/runs" / it["id"] / "metrics.json").write_text(json.dumps(truth(np.array(list(it["weights"].values())))))
        obj = {"metrics": {"val_loss/chat": {"direction": "min"}},
               "constraints": {"max_domain": {"chat": 0.3}, "min_domain_each": {"role": 0.15}, "min_domain": 0.05}}
        (self.d / "obj.yaml").write_text(yaml.safe_dump(obj))
        code, _ = run_cli("fit", "--dir", str(self.d / "sw"), "--objective", str(self.d / "obj.yaml"), "--trust-radius", "0.35", "--top", "4")
        self.assertEqual(code, 0)
        for r in json.loads((self.d / "sw/fit.json").read_text())["recommendations"]:
            w = r["weights"]
            self.assertLessEqual(w["chat"], 0.3 + 1e-6)
            self.assertGreaterEqual(w["role"], 0.15 - 1e-6)
            self.assertGreaterEqual(min(w.values()), 0.05 - 1e-6)
            self.assertLessEqual(r["distance_to_nearest_run"], 0.35)

    def test_fit_errors_are_actionable(self):
        design = self.generate(3)
        for it in design:
            (self.d / "sw/runs" / it["id"]).mkdir(parents=True)
            (self.d / "sw/runs" / it["id"] / "metrics.json").write_text(json.dumps({"x": 1.0}))
        (self.d / "obj.yaml").write_text(yaml.safe_dump({"metrics": {"nope": {}}}))
        code, out = run_cli("fit", "--dir", str(self.d / "sw"), "--objective", str(self.d / "obj.yaml"))
        self.assertEqual(code, 2)
        self.assertIn("available", out)
        (self.d / "obj.yaml").write_text(yaml.safe_dump({"metrics": {"x": {}}}))
        code, out = run_cli("fit", "--dir", str(self.d / "sw"), "--objective", str(self.d / "obj.yaml"))
        self.assertEqual(code, 2)
        self.assertIn("more runs", out)


if __name__ == "__main__":
    unittest.main()
