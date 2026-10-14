import contextlib
import copy
import io
import json
import unittest

import numpy as np
import yaml

from datasets.sft.loader import cli
from dataset.sft.loader.config import MixConfig
from datasets.sft.loader import PackedDataset, PackedStream, StateMismatchError
from datasets.sft.loader.helpers import Workspace

KEYS = ("input_ids", "labels", "position_ids", "doc_ids")


def same(a, b):
    return all(np.array_equal(a[k], b[k]) for k in KEYS)


def take(stream, n):
    out = []
    for _ in range(n):
        b = stream.next_batch()
        if b is None:
            break
        out.append(b)
    return out


class LoaderBase(unittest.TestCase):
    N = (200, 120, 40)

    @classmethod
    def setUpClass(cls):
        cls.ws = Workspace().populate(*cls.N)
        cls.cfg_raw = cls.ws.cfg_dict(max_epochs=6, micro_batch_size=2)
        cfg = MixConfig.from_dict(cls.cfg_raw)
        cls.ws.compile(cfg, chunk_rows=25)

    @classmethod
    def tearDownClass(cls):
        cls.ws.close()

    def stream(self, rank=0, world=1, **over):
        return PackedStream.from_config(MixConfig.from_dict({**self.cfg_raw, **over}), rank, world)


class BatchTests(LoaderBase):
    def test_batch_layout_and_derived_fields(self):
        s = self.stream()
        b = s.next_batch()
        L = s.cfg.seq_len
        self.assertEqual(b["input_ids"].shape, (2, L))
        for k in KEYS:
            self.assertEqual(b[k].shape, (2, L))
        self.assertEqual(b["n_trainable"], int((b["labels"] != -100).sum()))
        self.assertEqual(b["n_real"], int((b["doc_ids"] >= 0).sum()))
        cu = b["cu_seqlens"]
        self.assertEqual((cu[0], cu[-1]), (0, 2 * L))
        self.assertTrue((np.diff(cu) > 0).all())
        flat = b["doc_ids"].reshape(-1)
        for a, z in zip(cu[:-1], cu[1:]):                    # each cu_seqlens segment is exactly one document / padding run
            self.assertEqual(len(set(flat[a:z].tolist())), 1)
        self.assertEqual(b["max_seqlen"], int(np.diff(cu).max()))
        self.assertEqual(sorted(set(cu.tolist()) & {L}), [L])  # rows never share a segment

    def test_drop_last_and_padded_tail(self):
        s = self.stream(max_epochs=1, micro_batch_size=3)
        n = len(take(s, 10 ** 4))
        s2 = PackedStream.from_config(MixConfig.from_dict({**self.cfg_raw, "max_epochs": 1, "micro_batch_size": 3}), drop_last=False)
        batches = take(s2, 10 ** 4)
        self.assertEqual(len(batches), n + 1)                # the incomplete tail is kept, padded with empty packs
        self.assertGreater(batches[-1]["n_real"], 0)
        self.assertTrue((batches[-1]["labels"][-1] == -100).all())

    def test_realized_domain_mix_converges_to_target_tokens(self):
        ws = Workspace().populate(700, 500, 160)
        self.addCleanup(ws.close)
        cfg = MixConfig.from_dict(ws.cfg_dict(max_epochs=500, pack_window=16, micro_batch_size=1))
        ws.compile(cfg, chunk_rows=60)
        s = PackedStream.from_config(cfg)
        err = []
        for horizon in (150, 1500):                                # error is bounded by the window's contents, so it shrinks
            while s.metrics.packs < horizon:
                s.next_batch()
            snap = s.metrics.snapshot()
            err.append(max(abs(snap["domain_token_share"][d] - t) for d, t in (("chat", .5), ("tools", .3), ("long", .2))))
        self.assertLess(err[0], 0.05)
        self.assertLess(err[1], 0.01)
        self.assertGreater(snap["fill_rate"], 0.85)
        self.assertAlmostEqual(sum(snap["source_token_share"].values()), 1.0)
        self.assertEqual(snap["dropped_overlength"], {})


class ResumeTests(LoaderBase):
    def test_resume_gives_bit_identical_batches_through_json(self):
        a = self.stream()
        take(a, 9)
        state = json.loads(json.dumps(a.state_dict()))       # must survive serialization into a checkpoint
        want = take(a, 8)
        b = self.stream()
        b.load_state_dict(state)
        got = take(b, 8)
        self.assertEqual(len(got), len(want))
        for x, y in zip(want, got):
            self.assertTrue(same(x, y))
        self.assertEqual(a.metrics.snapshot(), b.metrics.snapshot())

    def test_resume_after_epoch_wraparound_and_mid_stream_exhaustion(self):
        a = self.stream(max_epochs=2)
        total = len(take(self.stream(max_epochs=2), 10 ** 5))
        take(a, total - 5)
        state = a.state_dict()
        want = take(a, 10)
        b = self.stream(max_epochs=2)
        b.load_state_dict(state)
        got = take(b, 10)
        self.assertEqual(len(want), len(got))
        self.assertTrue(all(same(x, y) for x, y in zip(want, got)))

    def test_mismatched_topology_or_config_is_refused(self):
        st = self.stream().state_dict()
        for bad in (self.stream(0, 2), self.stream(seed=1), self.stream(seq_len=128)):
            with self.assertRaisesRegex(StateMismatchError, "cannot resume"):
                bad.load_state_dict(st)
        with self.assertRaisesRegex(StateMismatchError, "version"):
            self.stream().load_state_dict({**st, "version": 99})
        self.stream(micro_batch_size=5).load_state_dict(st)  # batch grouping does not change the packed stream

    def test_different_seeds_give_different_streams_same_seed_identical(self):
        x, y, z = take(self.stream(), 3), take(self.stream(), 3), take(self.stream(seed=4), 3)
        self.assertTrue(all(same(a, b) for a, b in zip(x, y)))
        self.assertFalse(all(same(a, b) for a, b in zip(x, z)))


class DistributedTests(LoaderBase):
    def test_ranks_partition_the_corpus_exactly_once_per_epoch(self):
        world = 3
        owner = {}
        for rank in range(world):
            s = self.stream(rank, world, max_epochs=1)
            while (p := s.packer.next_pack()) is not None:
                for src, idx in zip(p.doc_src, p.doc_idx):
                    self.assertNotIn((src, idx), owner, "document consumed by two ranks")
                    owner[(src, idx)] = rank
        everything = {(si, int(i)) for si, v in enumerate(self.stream().views) for i in v.train_idx}
        self.assertEqual(set(owner), everything)                   # complete: nothing lost, nothing duplicated
        self.assertEqual(set(owner.values()), {0, 1, 2})

    def test_every_rank_sees_the_configured_mix(self):
        for rank in range(2):
            s = self.stream(rank, 2, max_epochs=80, pack_window=8, micro_batch_size=1)
            take(s, 500)
            shares = s.metrics.snapshot()["domain_token_share"]
            for name, target in (("chat", 0.5), ("tools", 0.3), ("long", 0.2)):
                self.assertAlmostEqual(shares[name], target, delta=0.03, msg=f"rank {rank} {name}")


class DatasetTests(LoaderBase):
    def test_prefetch_yields_same_batches_and_exact_state_after_early_stop(self):
        ref = take(self.stream(), 12)
        ds = PackedDataset(self.stream(), prefetch=4, as_torch=False)
        it = iter(ds)
        got = [next(it) for _ in range(5)]
        state = copy.deepcopy(ds.state_dict())
        it.close()                                            # abandon early: thread had already built more batches
        self.assertTrue(all(same(a, b) for a, b in zip(ref, got)))
        resumed = self.stream()
        resumed.load_state_dict(state)
        self.assertTrue(all(same(a, b) for a, b in zip(ref[5:], take(resumed, 7))))
        again = list(zip(range(7), iter(ds)))                  # the SAME dataset continues exactly where we stopped
        self.assertTrue(all(same(ref[5 + i], b) for i, b in again))

    def test_state_matches_the_last_consumed_batch_not_the_last_built_one(self):
        ds = PackedDataset(self.stream(), prefetch=6, as_torch=False)
        it = iter(ds)
        for _ in range(3):
            next(it)
        plain = self.stream()
        take(plain, 3)
        self.assertEqual(ds.state_dict()["sampler"], plain.state_dict()["sampler"])
        self.assertEqual(ds.state_dict()["packer"], plain.state_dict()["packer"])
        it.close()

    def test_worker_error_surfaces_in_consumer(self):
        s = self.stream()
        s.packer.next_pack = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        with self.assertRaisesRegex(RuntimeError, "boom"):
            list(PackedDataset(s, prefetch=2, as_torch=False))

    def test_torch_conversion_and_dataloader_if_available(self):
        try:
            import torch
            from torch.utils.data import DataLoader
        except ImportError:
            self.skipTest("torch not installed")
        ds = PackedDataset(self.stream(), prefetch=2)
        b = next(iter(DataLoader(ds, batch_size=None, num_workers=0)))
        self.assertEqual(b["input_ids"].dtype, torch.int64)
        self.assertEqual(b["labels"].dtype, torch.int64)
        self.assertEqual(b["position_ids"].dtype, torch.int32)
        self.assertEqual(tuple(b["input_ids"].shape), (2, 256))
        with self.assertRaisesRegex(RuntimeError, "num_workers=0"):
            next(iter(DataLoader(PackedDataset(self.stream()), batch_size=None, num_workers=1)))


class EvalTests(LoaderBase):
    def test_eval_batches_are_deterministic_complete_and_rank_balanced(self):
        s = self.stream()
        v = s.catalog.by_name()["short"]
        a, b = s.eval_batches("short"), s.eval_batches("short")
        self.assertTrue(all(same(x, y) for x, y in zip(a, b)))
        expect = sum(int(np.asarray(v.store.get(int(i))[1])[:int(v.cut[i])].sum()) for i in v.val_idx)
        self.assertEqual(sum(x["n_trainable"] for x in a), expect)       # every val doc exactly once
        world = 3
        per = [self.stream(r, world).eval_batches("short") for r in range(world)]
        self.assertEqual(len({len(p) for p in per}), 1)                   # same batch count per rank: collective-safe
        self.assertEqual(sum(x["n_trainable"] for p in per for x in p), expect)
        self.assertGreater(len(self.stream().eval_batches()), len(a))     # all sources


class CliTests(LoaderBase):
    def run_cli(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertLogs("sft_data", level="DEBUG") if False else contextlib.nullcontext():
            code = cli.main(list(argv))
        return code, out.getvalue()

    def setUp(self):
        self.path = self.ws.root / "mix.yaml"
        self.path.write_text(yaml.safe_dump(self.cfg_raw))

    def test_stats_reports_budget_epochs_and_probe(self):
        code, out = self.run_cli("stats", "--config", str(self.path), "--budget-tokens", "30000", "--probe-tokens", "3000")
        self.assertEqual(code, 0)
        for needle in ("sources (after filters)", "survives", "simulated budget", "probe:", "fill_rate", "chat", "tools"):
            self.assertIn(needle, out)

    def test_stats_flags_unreachable_budget(self):
        _, out = self.run_cli("stats", "--config", str(self.path), "--budget-tokens", "1e12", "--probe-tokens", "0")
        self.assertIn("NOT reachable", out)

    def test_validate_passes_on_a_good_cache_and_fails_on_a_tampered_one(self):
        code, out = self.run_cli("validate", "--config", str(self.path), "--samples", "30")
        self.assertEqual(code, 0, out)
        self.assertEqual(out.count("PASS"), 4)
        cfg = MixConfig.from_dict(self.cfg_raw)
        spec, *_ = self.ws.compile(cfg)
        part = sorted(spec.source_dir("short").glob("part-*.ids.npy"))[0]
        ids = np.load(part)
        ids[5] = (ids[5] + 1) % 400
        np.save(part, ids)
        self.addCleanup(lambda: self.ws.compile(cfg, force=True, chunk_rows=25))
        code, out = self.run_cli("validate", "--config", str(self.path), "--samples", "200")
        self.assertEqual(code, 1)
        self.assertIn("FAIL", out)

    def test_preview_prints_documents_with_sources(self):
        code, out = self.run_cli("preview", "--config", str(self.path), "--n", "1", "--chars", "300")
        self.assertEqual(code, 0)
        self.assertIn("=== pack:", out)
        self.assertIn("<|im_start|>", out)

    def test_compile_command_is_idempotent(self):
        code, out = self.run_cli("compile", "--config", str(self.path), "--workers", "1", "--sources", "short")
        self.assertEqual(code, 0)
        self.assertIn("compiled 0 tokens", out)                           # nothing left to do
        code, out = self.run_cli("compile", "--config", str(self.path), "--sources", "ghost")
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
