"""Standard errors for the four-patch census, by block bootstrap.

The census estimates 48 directed pair effects: for an ordered pair of patches (i, j) and an ancilla
in patch i, the effect of j on that ancilla. It is NOT a contrast against the isolated reference.
For each background B formed from the patches other than i and j, the estimator takes the logit
contrast between the configuration `B + i + j` and the configuration `B + i` -- the same background,
with and without the aggressor -- and averages those contrasts over every background that was
measured in both configurations. Counts are pooled across blocks within each configuration. The
point estimates do not depend on anything in this file; only their standard errors do.

THE RESAMPLING UNIT. Shots arrive in jobs, and each job runs a fixed number of syndrome cycles. The
unit that is exchangeable under the design is therefore the block `(job, cycle)`, not the shot: shots
within a block share a calibration, a job submission and a position in the queue. The bootstrap draws
blocks with replacement and keeps their MULTIPLICITIES -- a block drawn twice contributes its rows
twice, so a resampled statistic is a weighted mean with the block counts as weights.

Both draws are computed and printed side by side:

  set_as_frozen   blocks drawn with replacement and then collapsed to a set, which keeps about 81 of
                  128 distinct blocks and discards the multiplicities. This is the draw the frozen
                  analysis used. It is a subsample of roughly 63 % of the data rather than a
                  bootstrap, and its standard errors are anti-conservative for that reason.
  multiplicity    the same draw with multiplicities kept. This is the correct cluster bootstrap and
                  the one the manuscript reports.

The point estimates are identical under both, because they never involve the draw. The standard
errors move, and with them the Holm-adjusted p-values and the two summary quantities Q1 and Q3.

    python analysis/census_bootstrap.py

Reads `data/cache/census4_raw.npz` and `data/census4_manifest.json`.
Writes `out/census_boot_verification.json`.
"""
from __future__ import annotations

import itertools
import json
import math
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
RES = os.environ.get("FFCERT_DATA") or os.path.join(_ROOT, "data")
OUT = os.environ.get("FFCERT_OUT") or os.path.join(_ROOT, "out")
os.makedirs(OUT, exist_ok=True)

CACHE = os.path.join(RES, "cache", "census4_raw.npz")
MAN = os.path.join(RES, "census4_manifest.json")
FROZEN = os.path.join(RES, "census4_result.json")
NP = 4
MARGIN = 0.01
NBOOT_FROZEN = 300
NBOOT_FIXED = 4000


def holm(p):
    m = len(p)
    o = sorted(range(m), key=lambda t: p[t])
    adj = [0.0] * m
    run = 0.0
    for t, idx in enumerate(o):
        run = max(run, (m - t) * p[idx])
        adj[idx] = min(run, 1.0)
    return adj


def pair_effects(D, cfgs, patches, mode, nboot, seed=7):
    """`mode` is 'set_as_frozen' or 'multiplicity'. Point estimates identical in both."""
    ci, pi, ai = D[:, 3], D[:, 4], D[:, 5]
    n_tot, n_one = D[:, 6].astype(float), D[:, 7].astype(float)
    blk = D[:, 0] * 100 + D[:, 2]
    key = {}
    for r in range(len(D)):
        key.setdefault((int(ci[r]), int(pi[r]), int(ai[r])), []).append(r)

    def logit_of(rows, w=None):
        if w is None:
            a, b = n_one[rows].sum(), n_tot[rows].sum()
        else:
            a = float(np.dot(n_one[rows], w))
            b = float(np.dot(n_tot[rows], w))
        p = max(a, 0.5) / b
        return math.log(p / (1 - p))

    others = lambda i, j: [x for x in range(NP) if x not in (i, j)]                  # noqa: E731
    idx_of = {c: k for k, c in enumerate(cfgs)}
    out, boot = {}, {}
    blocks = sorted(set(blk.tolist()))
    nblk = len(blocks)
    bpos = {b: t for t, b in enumerate(blocks)}
    rng = np.random.default_rng(seed)

    for i in range(NP):
        for j in range(NP):
            if i == j:
                continue
            for k in range(4):
                terms = []
                for B in itertools.chain.from_iterable(
                        itertools.combinations(others(i, j), r) for r in range(3)):
                    with_j = tuple(sorted(set(B) | {i, j}))
                    no_j = tuple(sorted(set(B) | {i}))
                    if with_j not in idx_of or no_j not in idx_of:
                        continue
                    rw = key.get((idx_of[with_j], i, k), [])
                    rn = key.get((idx_of[no_j], i, k), [])
                    if rw and rn:
                        terms.append((rw, rn))
                if not terms:
                    continue
                g = np.mean([logit_of(rw) - logit_of(rn) for rw, rn in terms])
                bs = []
                for _ in range(nboot):
                    draw = rng.integers(0, nblk, nblk)
                    if mode == "set_as_frozen":
                        sel = {blocks[t] for t in draw}
                        v = []
                        for rw, rn in terms:
                            rw2 = [r for r in rw if blk[r] in sel]
                            rn2 = [r for r in rn if blk[r] in sel]
                            if rw2 and rn2:
                                v.append(logit_of(rw2) - logit_of(rn2))
                    else:
                        cnt = np.bincount(draw, minlength=nblk).astype(float)
                        v = []
                        for rw, rn in terms:
                            ww = cnt[[bpos[int(blk[r])] for r in rw]]
                            wn = cnt[[bpos[int(blk[r])] for r in rn]]
                            if ww.sum() > 0 and wn.sum() > 0:
                                v.append(logit_of(rw, ww) - logit_of(rn, wn))
                    if v:
                        bs.append(np.mean(v))
                q = patches[i][1 + 2 * k]
                out[(q, i, j, k)] = float(g)
                boot[(q, i, j, k)] = float(np.std(bs, ddof=1)) if len(bs) > 2 else float("nan")
    return out, boot


def summarise(g, se):
    ks = sorted(g, key=lambda t: (t[1], t[3], t[2]))
    vals = np.array([g[k] for k in ks])
    ses = np.array([se[k] for k in ks])
    # The preregistered test is two-sided against zero. The margin is applied separately, below, as
    # a filter on the point estimate.
    praw = [math.erfc(abs(vals[i] / max(ses[i], 1e-12)) / math.sqrt(2)) for i in range(len(ks))]
    hadj = holm(praw)
    # A stricter test that puts the margin inside the null, reported as a robustness column only.
    lm = math.log(1.0 + MARGIN)
    pmar = [0.5 * math.erfc(max((abs(vals[i]) - lm) / max(ses[i], 1e-12), -8.0) / math.sqrt(2))
            for i in range(len(ks))]
    hmar = holm(pmar)
    surv = [n for n in range(len(ks))
            if hadj[n] < 0.05 and abs(math.exp(vals[n]) - 1) > MARGIN]
    v_between = float(vals.var(ddof=1))
    v_within = float(np.mean(ses ** 2))
    q3 = 0
    for i in range(NP):
        for j in range(NP):
            if i == j:
                continue
            sub = [n for n, k in enumerate(ks) if k[1] == i and k[2] == j]
            if len(sub) < 2:
                continue
            if float(np.var(vals[sub], ddof=1)) > float(np.mean(ses[sub] ** 2)):
                q3 += 1
    return {"keys": ks, "vals": vals, "ses": ses, "holm": hadj, "holm_margin": hmar, "surv": surv,
            "var_between": v_between, "var_within": v_within,
            "ratio": v_between / max(v_within, 1e-12), "q3_of_12": q3,
            "n_pos": sum(1 for n in surv if vals[n] > 0),
            "n_neg": sum(1 for n in surv if vals[n] < 0)}


def main():
    man = json.load(open(MAN, encoding="utf-8"))
    patches = [list(p) for p in man["patches"]]
    D = np.load(CACHE)["rows"]
    cfgs = [tuple(c) for c in man["configurations"]]

    print("=" * 84)
    print("  CENSUS BOOTSTRAP VERIFICATION -- set() as frozen vs multiplicities kept")
    print("  Point estimates are identical by construction; only the SEs move.")
    print("=" * 84)

    res = {}
    for mode, nb in (("set_as_frozen", NBOOT_FROZEN), ("multiplicity", NBOOT_FIXED)):
        g, se = pair_effects(D, cfgs, patches, mode, nb)
        s = summarise(g, se)
        res[mode] = s
        print(f"\n{mode}  (NBOOT {nb})")
        print(f"  survivors beyond the 1 % margin at Holm 0.05 : {len(s['surv'])}"
              f"   ({s['n_pos']} positive, {s['n_neg']} negative)")
        print(f"  Var(gamma) {s['var_between']:.6f}   pooled within {s['var_within']:.6f}"
              f"   ratio {s['ratio']:.2f}")
        print(f"  Q3 localisation                              : {s['q3_of_12']} of 12")

    a, b = res["set_as_frozen"], res["multiplicity"]
    print("\nSURVIVORS, side by side")
    print(f"  {'qubit':>6}{'patch':>6}{'aggr':>5}{'effect %':>11}"
          f"{'se frozen':>11}{'se fixed':>10}{'Holm frozen':>13}{'Holm fixed':>12}")
    same = sorted(set(a["surv"]) | set(b["surv"]))
    for n in same:
        q, i, j, _ = a["keys"][n]
        print(f"  {q:>6}{i:>6}{j:>5}{100*(math.exp(a['vals'][n])-1):>+11.3f}"
              f"{a['ses'][n]:>11.5f}{b['ses'][n]:>10.5f}"
              f"{a['holm'][n]:>13.2e}{b['holm'][n]:>12.2e}")

    print("\nDECISIONS")
    print(f"  same survivor set : {set(a['surv']) == set(b['surv'])}")
    print(f"  Q1 heterogeneity  : frozen {a['ratio']:.2f} -> fixed {b['ratio']:.2f}"
          f"   (both > 1: {a['ratio'] > 1 and b['ratio'] > 1})")
    print(f"  Q2 signed         : frozen {a['n_pos']}+/{a['n_neg']}- -> "
          f"fixed {b['n_pos']}+/{b['n_neg']}-")
    print(f"  Q3                : frozen {a['q3_of_12']}/12 -> fixed {b['q3_of_12']}/12")

    out = {"note": "verification only; census4_result.json untouched",
           "NBOOT": {"set_as_frozen": NBOOT_FROZEN, "multiplicity": NBOOT_FIXED},
           "survivor_sets_identical": bool(set(a["surv"]) == set(b["surv"])),
           # The whole family is written, not only the survivors. A reader cannot check that the
           # same six survive without seeing the corrected standard error and adjusted p of all 48.
           "effects": [{"qubit": int(a["keys"][n][0]), "patch": int(a["keys"][n][1]),
                        "aggressor": int(a["keys"][n][2]),
                        "effect_pct": 100 * (math.exp(a["vals"][n]) - 1),
                        "se_frozen": float(a["ses"][n]), "se_fixed": float(b["ses"][n]),
                        "holm_frozen": float(a["holm"][n]), "holm_fixed": float(b["holm"][n]),
                        "holm_margin": float(b["holm_margin"][n]),
                        "survives_frozen": bool(n in set(a["surv"])),
                        "survives_fixed": bool(n in set(b["surv"]))}
                       for n in range(len(a["keys"]))],
           "survivors": {"frozen": sorted(int(x) for x in a["surv"]),
                         "fixed": sorted(int(x) for x in b["surv"])},
           "Q1": {"ratio_frozen": a["ratio"], "ratio_fixed": b["ratio"],
                  "var_between": a["var_between"],
                  "var_within_frozen": a["var_within"], "var_within_fixed": b["var_within"]},
           "Q2": {"frozen": [a["n_pos"], a["n_neg"]], "fixed": [b["n_pos"], b["n_neg"]]},
           "Q3": {"frozen_of_12": a["q3_of_12"], "fixed_of_12": b["q3_of_12"]}}
    json.dump(out, open(os.path.join(OUT, "census_boot_verification.json"), "w",
                        encoding="utf-8"), indent=2)
    print(f"\nwrote {os.path.join(OUT, 'census_boot_verification.json')}")


if __name__ == "__main__":
    main()
