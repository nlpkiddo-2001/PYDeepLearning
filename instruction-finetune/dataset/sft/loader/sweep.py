"""Proxy-model mixture sweep: generate candidate mixes -> run YOUR train/eval command on each -> fit -> recommend.

  python -m sft_data.sweep generate --base sft_data/mixes/baseline.yaml --out sweep/r1 --n 24
  python -m sft_data.sweep run      --dir sweep/r1 --cmd "python train_proxy.py --mix {mix} --out {out}" --gpus 0,1
  python -m sft_data.sweep fit      --dir sweep/r1 --objective sweep/objective.yaml

The train command receives a mixture YAML ({mix}), a run directory ({out}) and the id ({id}); it must write
`{out}/metrics.json`: a flat {name: number}, e.g. {"val_loss/chat": 1.91, "bfcl": 0.41, "ifeval": 0.33}.
Only DOMAIN weights are varied (within-domain structure comes from the base config).

The fit is RegMix-style: one small ridge regression per metric (linear or quadratic features, ridge strength chosen
by leave-one-out CV), then a large random search over the simplex for mixtures maximizing the weighted z-score
objective. Candidates are restricted to a trust region around mixes that were actually run, because regressions
extrapolate badly. Treat recommendations as candidates to CONFIRM at a larger model size, not as final answers.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

from dataset.sft.loader.config import MixConfig

log = logging.getLogger("sft_data")


def read_yaml(path) -> Any:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------------------------------- generate
def _with_domain_weights(base: dict[str, Any], names: list[str], w: np.ndarray) -> dict[str, Any]:
    d = copy.deepcopy(base)
    d.pop("stages", None)
    for name, x in zip(names, w):
        d["domains"][name]["weight"] = float(round(x, 6))
    return d


def design_points(k: int, n: int, base_w: np.ndarray, kind: str, concentration: float, floor: float, seed: int) -> np.ndarray:
    """n x k weights on the simplex, every coordinate >= floor. Row 0 is always the base mixture."""
    if floor * k >= 1:
        raise ValueError("floor too large for the number of domains")
    rng = np.random.default_rng(seed)
    pts = [base_w]
    if kind == "dirichlet":
        while len(pts) < n:
            w = rng.dirichlet(np.maximum(base_w * concentration, 1e-3))
            pts.append(floor + (1 - k * floor) * w)
    elif kind == "uniform":
        while len(pts) < n:
            pts.append(floor + (1 - k * floor) * rng.dirichlet(np.ones(k)))
    elif kind == "grid":
        step = max(1, int(round(1 / max(floor, 0.05))) // 2)
        lat = [np.array(c) / (step * 2) for c in itertools.product(range(step * 2 + 1), repeat=k) if sum(c) == step * 2]
        lat = [w for w in lat if (w >= floor - 1e-9).all()]
        order = rng.permutation(len(lat))
        pts += [lat[i] for i in order[:max(n - 1, 0)]]
    else:
        raise ValueError(f"unknown design {kind!r}")
    return np.array(pts[:n])


def cmd_generate(args) -> int:
    base = read_yaml(args.base)
    MixConfig.from_dict(base)  # validate the base first
    names = list(base["domains"])
    base_w = np.array([base["domains"][d]["weight"] for d in names], dtype=float)
    base_w /= base_w.sum()
    pts = design_points(len(names), args.n, base_w, args.design, args.concentration, args.min_domain, args.seed)
    out = Path(args.out)
    (out / "mixes").mkdir(parents=True, exist_ok=True)
    design = []
    for i, w in enumerate(pts):
        mid = f"mix_{i:03d}" + ("_base" if i == 0 else "")
        cfg = _with_domain_weights(base, names, w)
        MixConfig.from_dict(cfg)
        (out / "mixes" / f"{mid}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        design.append({"id": mid, "weights": dict(zip(names, map(float, w / w.sum())))})
    (out / "design.jsonl").write_text("".join(json.dumps(d) + "\n" for d in design), encoding="utf-8")
    (out / "base.yaml").write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    print(f"wrote {len(design)} mixtures over domains {names} to {out}/mixes")
    return 0


# ------------------------------------------------------------------------------------------------ run
def load_design(d: Path) -> list[dict]:
    return [json.loads(l) for l in (d / "design.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def run_one(d: Path, item: dict, cmd: str, gpus: Optional[str], timeout: Optional[float]) -> dict:
    run = d / "runs" / item["id"]
    run.mkdir(parents=True, exist_ok=True)
    argv = shlex.split(cmd.format(mix=str(d / "mixes" / f"{item['id']}.yaml"), out=str(run), id=item["id"]))
    env = dict(os.environ)
    if gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpus
    t0 = time.time()
    with open(run / "log.txt", "w") as lf:
        try:
            rc = subprocess.run(argv, stdout=lf, stderr=subprocess.STDOUT, env=env, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = -9
        except OSError as e:
            lf.write(f"could not start: {e}\n")
            rc = 127
    ok = rc == 0 and (run / "metrics.json").exists()
    status = {"ok": ok, "returncode": rc, "seconds": round(time.time() - t0, 1), "gpus": gpus}
    (run / "status.json").write_text(json.dumps(status))
    return status


def cmd_run(args) -> int:
    d = Path(args.dir)
    design = load_design(d)
    todo = [it for it in design if args.retry_failed or not (d / "runs" / it["id"] / "metrics.json").exists()]
    slots = [g for g in (args.gpus.split(";") if args.gpus else [None])]   # "0,1;2,3" = two jobs, two GPUs each
    if args.gpus and ";" not in args.gpus:
        slots = args.gpus.split(",")                                      # "0,1" = two jobs, one GPU each
    print(f"{len(todo)}/{len(design)} runs to do, {len(slots)} parallel slot(s)")
    q: queue.Queue = queue.Queue()
    for it in todo:
        q.put(it)
    free: queue.Queue = queue.Queue()
    for s in slots:
        free.put(s)
    results: dict[str, dict] = {}
    lock = threading.Lock()

    def worker():
        while True:
            try:
                it = q.get_nowait()
            except queue.Empty:
                return
            slot = free.get()
            try:
                st = run_one(d, it, args.cmd, slot, args.timeout)
            finally:
                free.put(slot)
            with lock:
                results[it["id"]] = st
                print(f"{'ok  ' if st['ok'] else 'FAIL'} {it['id']}  {st['seconds']}s  rc={st['returncode']}", flush=True)

    threads = [threading.Thread(target=worker) for _ in range(len(slots))]
    [t.start() for t in threads]
    [t.join() for t in threads]
    failed = [k for k, v in results.items() if not v["ok"]]
    if failed:
        print(f"{len(failed)} failed (see runs/<id>/log.txt): {failed}")
    return 1 if failed else 0


# ------------------------------------------------------------------------------------------------- fit
def load_results(d: Path) -> tuple[list[str], np.ndarray, dict[str, np.ndarray], list[str]]:
    design = load_design(d)
    names = list(design[0]["weights"])
    ids, X, rows = [], [], []
    for it in design:
        f = d / "runs" / it["id"] / "metrics.json"
        if f.exists():
            m = json.loads(f.read_text())
            ids.append(it["id"])
            X.append([it["weights"][n] for n in names])
            rows.append({k: float(v) for k, v in m.items() if isinstance(v, (int, float))})
    keys = sorted(set.intersection(*(set(r) for r in rows))) if rows else []
    return names, np.array(X), {k: np.array([r[k] for r in rows]) for k in keys}, ids


def features(X: np.ndarray, degree: int) -> np.ndarray:
    cols = [X[:, :-1]]            # simplex: drop one coordinate to avoid collinearity with the intercept
    if degree == 2:
        k = X.shape[1] - 1
        cols += [X[:, [i]] * X[:, [j]] for i in range(k) for j in range(i, k)]
    return np.hstack([np.ones((len(X), 1))] + cols)


def ridge_loo(F: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
    """Ridge fit and its exact leave-one-out RMSE (hat-matrix shortcut; intercept unpenalized)."""
    P = np.eye(F.shape[1]) * lam
    P[0, 0] = 0
    A = np.linalg.solve(F.T @ F + P, F.T)
    beta = A @ y
    H = F @ A
    resid = y - F @ beta
    loo = resid / np.maximum(1 - np.diag(H), 1e-6)
    return beta, float(np.sqrt(np.mean(loo ** 2)))


def fit_metric(X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    best = None
    for degree in (1, 2):
        F = features(X, degree)
        if F.shape[1] >= len(y):
            continue
        for lam in (1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0):
            beta, rmse = ridge_loo(F, y, lam)
            if best is None or rmse < best["loo_rmse"]:
                best = {"degree": degree, "lam": lam, "beta": beta, "loo_rmse": rmse}
    if best is None:
        raise ValueError(f"need more runs than model terms (have {len(y)}); run more mixtures")
    var = float(np.var(y))
    best["loo_r2"] = 1 - best["loo_rmse"] ** 2 / var if var > 0 else 0.0
    return best


def predict(model: dict[str, Any], X: np.ndarray) -> np.ndarray:
    return features(X, model["degree"]) @ model["beta"]


def sample_candidates(names, X_obs, n, rng, floor, max_domain, min_each) -> np.ndarray:
    k = len(names)
    parts = [rng.dirichlet(np.ones(k), n // 2)]
    best = X_obs[rng.integers(0, len(X_obs), n - n // 2)]
    parts.append(np.array([rng.dirichlet(np.maximum(b * 30, 0.05)) for b in best]))
    C = np.vstack(parts)
    lo = np.array([max(floor, min_each.get(nm, 0.0)) for nm in names])
    hi = np.array([max_domain.get(nm, 1.0) for nm in names])
    keep = ((C >= lo - 1e-9) & (C <= hi + 1e-9)).all(axis=1)
    return C[keep]


def cmd_fit(args) -> int:
    d = Path(args.dir)
    obj = read_yaml(args.objective)
    names, X, Y, ids = load_results(d)
    spec = obj["metrics"]
    missing = [m for m in spec if m not in Y]
    if missing:
        print(f"metrics not found in every finished run: {missing}; available: {sorted(Y)}")
        return 2
    cons = obj.get("constraints", {})
    floor = float(cons.get("min_domain", 0.0))
    print(f"{len(ids)} finished runs, {len(names)} domains, objective metrics {list(spec)}")
    try:
        models = {m: fit_metric(X, Y[m]) for m in spec}
    except ValueError as e:
        print(str(e))
        return 2
    mean = {m: float(Y[m].mean()) for m in spec}
    std = {m: float(Y[m].std()) or 1.0 for m in spec}
    sign = {m: -1.0 if spec[m].get("direction", "max") == "min" else 1.0 for m in spec}
    wt = {m: float(spec[m].get("weight", 1.0)) for m in spec}

    def score(Pm: dict[str, np.ndarray]) -> np.ndarray:
        return sum(wt[m] * sign[m] * (Pm[m] - mean[m]) / std[m] for m in spec)

    observed = score({m: Y[m] for m in spec})
    rng = np.random.default_rng(args.seed)
    C = sample_candidates(names, X, args.candidates, rng, floor, cons.get("max_domain", {}), cons.get("min_domain_each", {}))
    if len(C) == 0:
        print("no candidate satisfies the constraints")
        return 2
    dist = np.array([np.abs(c - X).sum(axis=1).min() / 2 for c in C[:50000]])      # total-variation distance to nearest run
    C, dist = C[:len(dist)], dist
    keep = dist <= args.trust_radius
    C, dist = C[keep], dist[keep]
    preds = {m: predict(models[m], C) for m in spec}
    sc = score(preds)
    order = np.argsort(-sc)
    chosen: list[int] = []
    for i in order:
        if all(np.abs(C[i] - C[j]).sum() / 2 >= args.min_separation for j in chosen):
            chosen.append(int(i))
        if len(chosen) >= args.top:
            break
    base = read_yaml(d / "base.yaml")
    (d / "recommended").mkdir(exist_ok=True)
    lines = ["# Mixture fit report\n", f"{len(ids)} runs; objective: " + ", ".join(
        f"{'min' if sign[m] < 0 else 'max'} {m} (w={wt[m]})" for m in spec) + "\n",
        "## Model quality (leave-one-out)\n", "| metric | terms | LOO R² | LOO RMSE | trust |", "|---|---|---|---|---|"]
    for m, mod in models.items():
        trust = "ok" if mod["loo_r2"] >= 0.5 else ("weak" if mod["loo_r2"] >= 0.2 else "NOT PREDICTABLE")
        lines.append(f"| {m} | {'quadratic' if mod['degree'] == 2 else 'linear'} | {mod['loo_r2']:.2f} | {mod['loo_rmse']:.4g} | {trust} |")
    weak = [m for m, mod in models.items() if mod["loo_r2"] < 0.2]
    if weak:
        lines.append(f"\n**Warning:** {weak} are not predictable from the mixture with this many runs; rankings that depend on them are unreliable.\n")
    lines += ["\n## Best observed runs\n", "| run | score | " + " | ".join(names) + " | " + " | ".join(spec) + " |",
              "|---|---|" + "---|" * (len(names) + len(spec))]
    for i in np.argsort(-observed)[:5]:
        lines.append(f"| {ids[i]} | {observed[i]:.2f} | " + " | ".join(f"{x:.3f}" for x in X[i]) + " | " + " | ".join(f"{Y[m][i]:.4g}" for m in spec) + " |")
    lines += ["\n## Recommended candidates (predicted; confirm with real runs)\n",
              "| # | pred. score | " + " | ".join(names) + " | " + " | ".join(f"pred {m}" for m in spec) + " | dist to nearest run |",
              "|---|---|" + "---|" * (len(names) + len(spec) + 1)]
    recs = []
    for r, i in enumerate(chosen):
        w = C[i] / C[i].sum()
        (d / "recommended" / f"rec_{r}.yaml").write_text(yaml.safe_dump(_with_domain_weights(base, names, w), sort_keys=False), encoding="utf-8")
        lines.append(f"| {r} | {sc[i]:.2f} | " + " | ".join(f"{x:.3f}" for x in w) + " | " + " | ".join(f"{preds[m][i]:.4g}" for m in spec) + f" | {dist[i]:.3f} |")
        recs.append({"rank": r, "weights": dict(zip(names, map(float, w))), "predicted_score": float(sc[i]),
                     "predicted": {m: float(preds[m][i]) for m in spec}, "distance_to_nearest_run": float(dist[i])})
    (d / "fit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (d / "fit.json").write_text(json.dumps({"recommendations": recs, "loo_r2": {m: mod["loo_r2"] for m, mod in models.items()}}, indent=1))
    print("\n".join(lines))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sft_data.sweep")
    sub = ap.add_subparsers(dest="command", required=True)  # NOT "cmd": `run --cmd` would overwrite it
    g = sub.add_parser("generate")
    g.add_argument("--base", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--n", type=int, default=24)
    g.add_argument("--design", choices=["dirichlet", "uniform", "grid"], default="dirichlet")
    g.add_argument("--concentration", type=float, default=15.0, help="dirichlet: higher = closer to the base mix")
    g.add_argument("--min-domain", type=float, default=0.03)
    g.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("run")
    r.add_argument("--dir", required=True)
    r.add_argument("--cmd", required=True, help="command template with {mix} {out} {id}")
    r.add_argument("--gpus", default=None, help='"0,1" = 2 parallel jobs on 1 GPU each; "0,1;2,3" = 2 jobs on 2 GPUs each')
    r.add_argument("--timeout", type=float, default=None)
    r.add_argument("--retry-failed", action="store_true")
    f = sub.add_parser("fit")
    f.add_argument("--dir", required=True)
    f.add_argument("--objective", required=True)
    f.add_argument("--top", type=int, default=5)
    f.add_argument("--candidates", type=int, default=200000)
    f.add_argument("--trust-radius", type=float, default=0.15, help="max total-variation distance to a mix that was run")
    f.add_argument("--min-separation", type=float, default=0.03)
    f.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return {"generate": cmd_generate, "run": cmd_run, "fit": cmd_fit}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
