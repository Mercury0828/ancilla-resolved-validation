"""Patch-level aggregates for the patch that carries the headline cancellation.

Opposite per-ancilla effects can cancel in a patch aggregate, and different aggregates can differ
when the ancillas have different baseline occupancies. This script computes five aggregates on the
census cache, with the same background-averaged estimator and the same corrected block bootstrap that the per-ancilla
effects use.

Five summaries of the same patch and aggressor:

  pooled odds      counts summed over the patch's four ancillas first, then one logit contrast.
                   This is the pooled patch value of the manuscript's Fig. 2.
  mean log-odds    the mean of the four per-ancilla logit contrasts, reported as a change in odds.
  mean of percents the mean of the four per-ancilla percentage changes.
  rate difference  the mean over ancillas of the difference in syndrome-entry rate, on the rate
                   scale rather than the odds scale.
  sign-insensitive the root mean square and the maximum of the four contrasts, which do not cancel.
                   Two RMS values are printed: of the percentage effects (`rms_pct`, the one the manuscript
                   quotes) and of the log-odds contrasts transformed back (`rms_logodds_pct`).

The point of printing all five together is that the first four are near zero and the last is not.

    python analysis/patch_aggregate.py

Writes `out/patch_aggregate.json`.
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
OUTDIR = os.environ.get("FFCERT_OUT") or os.path.join(_ROOT, "out")
CACHE = os.path.join(RES, "cache", "census4_raw.npz")
NP = 4
NBOOT = 4000
SEED = 11
# The patch and aggressor the main text and Fig. 2 report.
TARGET, AGGRESSOR = 1, 0


def logit(a, b):
    p = max(a, 0.5) / b
    return math.log(p / (1 - p))


def main():
    D = np.load(CACHE, allow_pickle=True)["rows"]
    cfgs = [tuple(sorted(c)) for c in
            json.load(open(os.path.join(RES, "census4_build.json"), encoding="utf-8"))["configurations"]]
    patches = json.load(open(os.path.join(RES, "layout2.json"), encoding="utf-8"))["patches"]

    ci, pi, ai = D[:, 3], D[:, 4], D[:, 5]
    n_tot, n_one = D[:, 6].astype(float), D[:, 7].astype(float)
    blk = D[:, 0] * 100 + D[:, 2]
    rows_of = {}
    for r in range(len(D)):
        rows_of.setdefault((int(ci[r]), int(pi[r]), int(ai[r])), []).append(r)

    idx_of = {c: k for k, c in enumerate(cfgs)}
    others = [x for x in range(NP) if x not in (TARGET, AGGRESSOR)]

    # Every background subset of the two uninvolved patches, exactly as the per-ancilla estimator
    # averages over them.
    terms = []
    for B in itertools.chain.from_iterable(
            itertools.combinations(others, r) for r in range(len(others) + 1)):
        with_j = tuple(sorted(set(B) | {TARGET, AGGRESSOR}))
        no_j = tuple(sorted(set(B) | {TARGET}))
        if with_j in idx_of and no_j in idx_of:
            terms.append((idx_of[with_j], idx_of[no_j]))

    blocks = sorted(set(blk.tolist()))
    nblk = len(blocks)
    bpos = {b: t for t, b in enumerate(blocks)}
    w_all = np.ones(len(D))

    def stats(w):
        """All five summaries under one set of block weights."""
        per_anc, rate_diff = [], []
        pooled = []
        for cw, cn in terms:
            aw = bw = an = bn = 0.0
            gs = []
            for k in range(4):
                rw, rn = rows_of.get((cw, TARGET, k), []), rows_of.get((cn, TARGET, k), [])
                if not rw or not rn:
                    continue
                sw = np.dot(n_one[rw], w[rw]), np.dot(n_tot[rw], w[rw])
                sn = np.dot(n_one[rn], w[rn]), np.dot(n_tot[rn], w[rn])
                if sw[1] <= 0 or sn[1] <= 0:
                    continue
                aw += sw[0]; bw += sw[1]; an += sn[0]; bn += sn[1]
                gs.append(logit(*sw) - logit(*sn))
                rate_diff.append(sw[0] / sw[1] - sn[0] / sn[1])
            if bw > 0 and bn > 0:
                pooled.append(logit(aw, bw) - logit(an, bn))
            if gs:
                per_anc.append(gs)
        if not pooled or not per_anc:
            return None
        g = np.mean(np.array(per_anc), axis=0)          # one mean contrast per ancilla
        return {
            "pooled_odds_pct": 100 * (math.exp(float(np.mean(pooled))) - 1),
            "mean_logodds_pct": 100 * (math.exp(float(g.mean())) - 1),
            "mean_of_percents": float(np.mean(100 * (np.exp(g) - 1))),
            "rate_diff_pct_points": 100 * float(np.mean(rate_diff)),
            # `rms_logodds_pct` is the RMS of the log-odds contrasts, transformed back to a percentage.
            # `rms_pct` is the RMS of the four percentage effects, the value the manuscript quotes.
            "rms_logodds_pct": 100 * (math.exp(float(np.sqrt((g ** 2).mean()))) - 1),
            "rms_pct": float(np.sqrt(np.mean((100 * (np.exp(g) - 1)) ** 2))),
            "max_abs_pct": float(np.max(np.abs(100 * (np.exp(g) - 1)))),
        }

    point = stats(w_all)
    rng = np.random.default_rng(SEED)
    draws = {k: [] for k in point}
    for _ in range(NBOOT):
        cnt = np.bincount(rng.integers(0, nblk, nblk), minlength=nblk).astype(float)
        w = cnt[[bpos[int(b)] for b in blk]]
        s = stats(w)
        if s:
            for k, v in s.items():
                draws[k].append(v)

    out = {"note": "patch-level aggregates for the patch reported in Fig. 2",
           "patch": TARGET, "aggressor": AGGRESSOR, "NBOOT": NBOOT,
           "aggregates": {}}
    label = {"pooled_odds_pct": "pooled patch odds",
             "mean_logodds_pct": "mean of log-odds effects",
             "mean_of_percents": "mean of percentage changes",
             "rate_diff_pct_points": "mean rate difference (percentage points)",
             "rms_logodds_pct": "RMS of the log-odds effects, as % odds",
             "max_abs_pct": "largest single effect",
             "rms_pct": "RMS of the four percentage effects"}
    print(f"{'aggregate':<44}{'value':>10}{'95% interval':>26}")
    for k, v in point.items():
        d = np.array(draws[k])
        lo, hi = (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))) if len(d) > 10 \
            else (float("nan"), float("nan"))
        out["aggregates"][k] = {"value": v, "lo": lo, "hi": hi}
        print(f"{label[k]:<44}{v:>+9.3f}%{f'[{lo:+.3f}, {hi:+.3f}]':>26}")

    os.makedirs(OUTDIR, exist_ok=True)
    json.dump(out, open(os.path.join(OUTDIR, "patch_aggregate.json"), "w", encoding="utf-8"), indent=2)
    print(f"\nwrote {os.path.join(OUTDIR, 'patch_aggregate.json')}")


if __name__ == "__main__":
    main()
