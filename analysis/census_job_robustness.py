"""Job-level robustness of the census effects, and the effects resolved by background.

Two questions about the census, answered with the recorded data:

  1. Are contrasts from blocks of the same job dependent enough to make the block bootstrap
     anti-conservative? For every one of the 48 directed effects this script computes
       - the per-job point estimate (the census estimator restricted to one job's rows),
       - the leave-one-job-out estimate (all jobs but one),
       - a job-level jackknife standard error from the eight leave-one-out estimates,
       - a whole-job cluster bootstrap standard error (jobs drawn with replacement, multiplicities
         kept, NBOOT 4000),
     and repeats the Holm test of the manuscript with the jackknife SE and a t reference with
     7 degrees of freedom (eight jobs), the more conservative of the two job-level choices.
  2. Does averaging over the states of the two other patches hide effects of opposite sign? For the
     six Holm survivors and the four test pairs it prints gamma(B) for each background B.

The point estimator is the manuscript's: per background, the logit of the pooled occupancy with and
without the aggressor, averaged over the four backgrounds (counts pooled over blocks; a Jeffreys-free
floor of 0.5 ones as in `analysis/census_bootstrap.py`). The block-bootstrap SEs the
manuscript reports are read from `data/census_boot_verification.json` for comparison.

    python analysis/census_job_robustness.py

Writes `out/census_job_robustness.json`.
"""
from __future__ import annotations

import itertools
import json
import math
import os

import numpy as np
from scipy import stats

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("FFCERT_DATA") or os.path.join(_ROOT, "data")
OUT = os.path.join(os.environ.get("FFCERT_OUT") or os.path.join(_ROOT, "out"), "census_job_robustness.json")
NP = 4
MARGIN = 0.01
NBOOT = 4000
SEED = 31
TEST_PAIRS = [(63, 2), (21, 3), (1, 0), (147, 1)]      # (ancilla, aggressor patch)


def holm(p):
    m = len(p)
    o = sorted(range(m), key=lambda t: p[t])
    adj, run = [0.0] * m, 0.0
    for t, idx in enumerate(o):
        run = max(run, (m - t) * p[idx])
        adj[idx] = min(run, 1.0)
    return adj


def main():
    man = json.load(open(os.path.join(DATA, "census4_manifest.json"), encoding="utf-8"))
    patches = [list(p) for p in man["patches"]]
    cfgs = [tuple(c) for c in man["configurations"]]
    idx_of = {c: k for k, c in enumerate(cfgs)}
    D = np.load(os.path.join(DATA, "cache", "census4_raw.npz"))["rows"]
    job, ci, pi, ai = D[:, 0], D[:, 3], D[:, 4], D[:, 5]
    n_tot, n_one = D[:, 6].astype(float), D[:, 7].astype(float)
    jobs = sorted(set(job.tolist()))
    J = len(jobs)

    # count tables: T[(cfg, patch, anc)] -> per-job (ones, total)
    T = {}
    for r in range(len(D)):
        k = (int(ci[r]), int(pi[r]), int(ai[r]))
        if k not in T:
            T[k] = np.zeros((J, 2))
        T[k][jobs.index(int(job[r]))] += (n_one[r], n_tot[r])

    def lg(v):
        p = max(v[0], 0.5) / v[1]
        return math.log(p / (1 - p))

    others = lambda i, j: [x for x in range(NP) if x not in (i, j)]       # noqa: E731
    keys, terms_of = [], {}
    for i in range(NP):
        for j in range(NP):
            if i == j:
                continue
            for k in range(4):
                terms = []
                for B in itertools.chain.from_iterable(itertools.combinations(others(i, j), r) for r in range(3)):
                    w = idx_of.get(tuple(sorted(set(B) | {i, j})))
                    n = idx_of.get(tuple(sorted(set(B) | {i})))
                    if w is None or n is None or (w, i, k) not in T or (n, i, k) not in T:
                        continue
                    terms.append((tuple(sorted(B)), T[(w, i, k)], T[(n, i, k)]))
                key = (patches[i][1 + 2 * k], i, j)
                keys.append(key)
                terms_of[key] = terms

    def est(terms, wvec):
        return float(np.mean([lg(wvec @ a) - lg(wvec @ b) for _, a, b in terms]))

    ones = np.ones(J)
    rng = np.random.default_rng(SEED)
    draws = [np.bincount(rng.integers(0, J, J), minlength=J).astype(float) for _ in range(NBOOT)]
    boot_ref = {(e["qubit"], e["patch"], e["aggressor"]): e for e in
                json.load(open(os.path.join(DATA, "census_boot_verification.json"), encoding="utf-8"))["effects"]}

    rows = []
    for key in keys:
        terms = terms_of[key]
        g = est(terms, ones)
        per_job = [est(terms, np.eye(J)[t]) for t in range(J)]
        loo = [est(terms, ones - np.eye(J)[t]) for t in range(J)]
        jk_se = math.sqrt((J - 1) / J * sum((x - np.mean(loo)) ** 2 for x in loo))
        bs = [est(terms, w) for w in draws]
        rows.append({"qubit": key[0], "patch": key[1], "aggressor": key[2], "gamma": g,
                     "effect_pct": 100 * (math.exp(g) - 1),
                     "se_block": boot_ref[key]["se_fixed"], "holm_block": boot_ref[key]["holm_fixed"],
                     "survives_block": boot_ref[key]["survives_fixed"],
                     "se_job_jackknife": jk_se, "se_job_bootstrap": float(np.std(bs, ddof=1)),
                     "per_job_pct": [100 * (math.exp(x) - 1) for x in per_job],
                     "per_job_same_sign": int(sum(1 for x in per_job if np.sign(x) == np.sign(g))),
                     "loo_pct": [100 * (math.exp(x) - 1) for x in loo],
                     "by_background_pct": {",".join(map(str, B)) or "none":
                                           100 * (math.exp(lg(a.sum(0)) - lg(b.sum(0))) - 1)
                                           for B, a, b in terms}})

    # Holm with job-level SE: t_7 reference on the jackknife SE (conservative choice)
    p_t = [2 * stats.t.sf(abs(r["gamma"]) / r["se_job_jackknife"], J - 1) for r in rows]
    p_z = [math.erfc(abs(r["gamma"]) / r["se_job_bootstrap"] / math.sqrt(2)) for r in rows]
    for r, a, b in zip(rows, holm(p_t), holm(p_z)):
        r["holm_job_jackknife_t7"] = a
        r["holm_job_bootstrap_z"] = b
        big = abs(r["effect_pct"]) > 100 * MARGIN
        r["survives_job_jackknife_t7"] = bool(a < 0.05 and big)
        r["survives_job_bootstrap_z"] = bool(b < 0.05 and big)
        r["loo_all_beyond_margin_same_sign"] = bool(all(abs(x) > 100 * MARGIN and np.sign(x) == np.sign(r["gamma"])
                                                        for x in r["loo_pct"]))

    surv = [r for r in rows if r["survives_block"]]
    print(f"{'q':>4}{'p':>3}{'a':>3}{'eff%':>8}{'se_blk':>8}{'se_jk':>8}{'se_jb':>8}"
          f"{'Holm_jk_t7':>12}{'jobs same sign':>16}{'LOO range %':>18}")
    for r in surv:
        print(f"{r['qubit']:>4}{r['patch']:>3}{r['aggressor']:>3}{r['effect_pct']:>+8.2f}{r['se_block']:>8.4f}"
              f"{r['se_job_jackknife']:>8.4f}{r['se_job_bootstrap']:>8.4f}{r['holm_job_jackknife_t7']:>12.2e}"
              f"{r['per_job_same_sign']:>13}/{J}   [{min(r['loo_pct']):+.2f}, {max(r['loo_pct']):+.2f}]")
    s_jk = sorted((r["qubit"], r["patch"], r["aggressor"]) for r in rows if r["survives_job_jackknife_t7"])
    s_jb = sorted((r["qubit"], r["patch"], r["aggressor"]) for r in rows if r["survives_job_bootstrap_z"])
    s_bl = sorted((r["qubit"], r["patch"], r["aggressor"]) for r in surv)
    print("survivors block   :", s_bl)
    print("survivors jk t7   :", s_jk)
    print("survivors job boot:", s_jb)
    ratio = [r["se_job_jackknife"] / r["se_block"] for r in rows]
    print(f"SE ratio jackknife/block over 48: median {np.median(ratio):.2f}, range [{min(ratio):.2f}, {max(ratio):.2f}]")

    print("\nBy background (effect %), backgrounds = other patches running")
    sel = {(r["qubit"], r["aggressor"]): r for r in rows}
    show = [(r["qubit"], r["aggressor"]) for r in surv] + [p for p in TEST_PAIRS if p not in [(r["qubit"], r["aggressor"]) for r in surv]]
    for q, a in show:
        r = sel[(q, a)]
        print(f"  q{q:<4} aggr {a}  avg {r['effect_pct']:+6.2f}   " +
              "  ".join(f"[{b}] {v:+6.2f}" for b, v in r["by_background_pct"].items()))

    out = {"note": "job-level robustness of the census; exploratory sensitivity analysis, no verdict of the "
                   "manuscript depends on it",
           "jobs": J, "NBOOT_job": NBOOT, "seed": SEED,
           "survivors_block": s_bl, "survivors_job_jackknife_t7": s_jk, "survivors_job_bootstrap_z": s_jb,
           "se_ratio_jackknife_over_block": {"median": float(np.median(ratio)), "min": float(min(ratio)),
                                             "max": float(max(ratio))},
           "effects": rows}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
