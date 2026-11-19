"""Job-level robustness of the four-pair suppression ratio, and the ratio of each pair.

For the translow campaign (32 ns versus 624 ns) and the transport campaign (32 ns versus 8032 ns) this
script reuses the estimator of `analysis/block_bootstrap.py` unchanged and reports
  - the pooled ratio R = H(long)/H(32 ns) with every job left out in turn,
  - a whole-job cluster bootstrap 95th percentile of R (NBOOT 4000),
  - the ratio of each pair, H_s(long)/H_s(32 ns), with its whole-block bootstrap interval.
Descriptive; the decision of the manuscript (block bootstrap U95 < 0.50) is unchanged.

    python analysis/transport_job_robustness.py

Writes `out/transport_job_robustness.json`.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import block_bootstrap as BB  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.environ.get("FFCERT_OUT") or os.path.join(_ROOT, "out"), "transport_job_robustness.json")
NBOOT = 4000


def analyse(manifest, cache_name, low_gap, seed):
    man = json.load(open(os.path.join(BB.RES, manifest), encoding="utf-8"))
    D = BB.load(cache_name)
    arms, cp, pairs = man["arms"], man["census_patches"], man["frozen_pairs"]
    c_w = float(man["c_weight"])
    cache = {}
    for p in pairs:
        q, tp, _ = p["key"].split("_")
        tgt = int(tp[1:])
        k = [cp[tgt][i] for i in (1, 3, 5, 7)].index(int(q[1:]))
        for g in (0, low_gap):
            cache[(p["key"], g)] = (BB.series(D, arms, f"{p['key']}_full_g{g}", k),
                                    BB.series(D, arms, f"p{tgt}_alone_g{g}", k))

    def H_parts(sel=None):
        H = {}
        for p in pairs:
            for g in (0, low_gap):
                f, a = cache[(p["key"], g)]
                us = sorted(set(f) & set(a)) if sel is None else [u for u in sel if u in f and u in a]
                if not us:
                    H[(p["key"], g)] = float("nan")
                    continue
                tb = float(np.mean([f[u][0] - a[u][0] for u in us]))
                ta = float(np.mean([f[u][1] - a[u][1] for u in us]))
                H[(p["key"], g)] = tb * tb + c_w * ta * ta
        return H

    def pooled(H):
        return sum(H[(p["key"], low_gap)] for p in pairs) / sum(H[(p["key"], 0)] for p in pairs)

    all_units = sorted({u for k in cache for i in (0, 1) for u in cache[k][i]})
    jobs = sorted({u[0] for u in all_units})
    Hpt = H_parts()
    R = pooled(Hpt)
    loo = {int(j): pooled(H_parts([u for u in all_units if u[0] != j])) for j in jobs}

    rng = np.random.default_rng(seed)
    by_job = {j: [u for u in all_units if u[0] == j] for j in jobs}
    jb = []
    for _ in range(NBOOT):
        s = [u for t in rng.integers(0, len(jobs), len(jobs)) for u in by_job[jobs[t]]]
        jb.append(pooled(H_parts(s)))
    b_of = BB.block_of(D)
    blocks = sorted({b_of[u] for u in all_units})
    by_block = {b: [u for u in all_units if b_of[u] == b] for b in blocks}
    rng = np.random.default_rng(seed + 1)
    per_pair_boot = {p["key"]: [] for p in pairs}
    for _ in range(NBOOT):
        s = [u for t in rng.integers(0, len(blocks), len(blocks)) for u in by_block[blocks[t]]]
        h = H_parts(s)
        for p in pairs:
            per_pair_boot[p["key"]].append(h[(p["key"], low_gap)] / h[(p["key"], 0)])
    per_pair = {p["key"]: {"ratio": Hpt[(p["key"], low_gap)] / Hpt[(p["key"], 0)],
                           "ci_block": [float(np.percentile(per_pair_boot[p["key"]], 2.5)),
                                        float(np.percentile(per_pair_boot[p["key"]], 97.5))]}
                for p in pairs}
    return {"R_point": R, "jobs": len(jobs), "loo_R": loo, "loo_max": max(loo.values()),
            "loo_min": min(loo.values()), "job_boot_U95": float(np.percentile(jb, 95)),
            "per_pair": per_pair}


def main():
    res = {"note": "descriptive job-level robustness; decisions of the manuscript unchanged", "NBOOT": NBOOT}
    for tag, man, cache, g, seed in (("translow_624ns", "translow_manifest.json", "translow_raw.npz", 296, 20261119),
                                     ("transport_8032ns", "transport_manifest.json", "transport_raw.npz", 4000, 20261120)):
        r = analyse(man, cache, g, seed)
        res[tag] = r
        print(f"{tag}: R {r['R_point']:.4f}  LOO [{r['loo_min']:.4f}, {r['loo_max']:.4f}]  job-boot U95 {r['job_boot_U95']:.3f}  jobs {r['jobs']}")
        for k, v in r["per_pair"].items():
            print(f"   {k:14s} ratio {v['ratio']:.4f}  block CI [{v['ci_block'][0]:.4f}, {v['ci_block'][1]:.4f}]")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
