"""The fine timing sweep: the displacement coordinates, recomputed from the raw counts.

This is the campaign the manuscript's timing result and its figure report. It recomputes the two
displacement coordinates at every swept separation directly from `data/cache/gapfine_raw.npz`, so
that the headline timing numbers are derived here rather than read back from a frozen artefact.

THE CACHE. `gapfine_raw.npz` holds one row per (job, circuit position), with eight integer columns:

    0  job index
    1  position within the job
    2  syndrome cycle count
    3  arm index, into `arms` in `data/gapfine_manifest.json`
    4  n0x   transitions observed starting from syndrome 0
    5  n01   of those, transitions into 1
    6  n1x   transitions observed starting from syndrome 1
    7  n11   of those, transitions into 1

Columns 4-7 are the sufficient statistics of the one-step law. They are counts of transitions, not
of shots, and no per-shot information beyond them enters the estimator.

THE ESTIMATOR. For each arm the entry rate `b = n01 / n0x` and the persistence `a = n11 / n1x` are
formed within each block `(job, cycle)`. The displacement coordinates are then differences of block
means between the co-active arm and the isolated arm at the same separation:

    theta_b = mean_blocks log b_full  -  mean_blocks log b_alone
    theta_a = mean_blocks logit a_full - mean_blocks logit a_alone

Both are within-separation contrasts, so the pairing removes drift common to the two arms. A block
with zero observed entries is floored at half a count rather than dropped, which is why `log b` is
taken against `max(b, 0.5 / n0x)`.

The percentile intervals use a block bootstrap at the same seed and replicate count as the frozen
run, so the intervals printed here are the intervals the manuscript quotes.

    python analysis/timing_sweep.py

Reads `data/cache/gapfine_raw.npz` and `data/gapfine_manifest.json`.
Writes `out/timing_sweep.json`, and prints each coordinate beside the frozen value in
`data/gapfine_result.json`.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
RES = os.environ.get("FFCERT_DATA") or os.path.join(_ROOT, "data")
OUT = os.environ.get("FFCERT_OUT") or os.path.join(_ROOT, "out")
os.makedirs(OUT, exist_ok=True)

CACHE = os.path.join(RES, "cache", "gapfine_raw.npz")
MAN = os.path.join(RES, "gapfine_manifest.json")
FROZEN = os.path.join(RES, "gapfine_result.json")

NBOOT = 4000
SEED = 20260908
LAYOUT_OFFSET_NS = 44          # measured separation = 2 * declared + offset, on this layout


def logit(p):
    return math.log(p / (1 - p))


def main():
    man = json.load(open(MAN, encoding="utf-8"))
    arms = man["arms"]
    meta = {a[0]: a for a in man["arm_meta"]}
    D = np.load(CACHE)["rows"]

    blocks = D[:, 0] * 1000 + D[:, 2]
    arm_of = D[:, 3]
    ublocks = np.unique(blocks)
    print(f"{len(D)} rows, {int(D[:, 4].sum() + D[:, 6].sum()):,} transitions, "
          f"{len(ublocks)} blocks")

    def per_block(name):
        i = arms.index(name)
        s = arm_of == i
        lb, la, bl = [], [], []
        for g in ublocks:
            k = s & (blocks == g)
            if not k.any():
                continue
            b = D[k, 5].sum() / max(D[k, 4].sum(), 1)
            a = D[k, 7].sum() / max(D[k, 6].sum(), 1)
            lb.append(math.log(max(b, 0.5 / max(D[k, 4].sum(), 1))))
            la.append(logit(min(max(a, 1e-6), 1 - 1e-6)))
            bl.append(g)
        return np.array(lb), np.array(la), np.array(bl)

    cache = {a: per_block(a) for a in arms}
    strata = sorted({meta[a][1] for a in arms})
    key_of = {meta[a][1]: a.rsplit("_", 1)[0] for a in arms}

    def theta(gk, sel=None):
        fb, fa, _ = cache[gk + "_full"]
        ab, aa, _ = cache[gk + "_alone"]
        if sel is None:
            return fb.mean() - ab.mean(), fa.mean() - aa.mean()
        return fb[sel].mean() - ab[sel].mean(), fa[sel].mean() - aa[sel].mean()

    frozen = json.load(open(FROZEN, encoding="utf-8"))
    ftb, fta = frozen["theta_b"], frozen["theta_a"]

    def frozen_at(d, g):
        k = next((k for k in d if abs(float(k) - g) < 1e-12), None)
        return None if k is None else d[k]

    rng = np.random.default_rng(SEED)
    rows, bad = [], 0
    print(f"\n{'declared':>9}{'measured':>10}{'theta_b':>10}{'frozen':>10}"
          f"{'theta_a':>10}{'frozen':>10}   90 % interval on theta_b")
    for g in strata:
        k = key_of[g]
        tb, ta = theta(k)
        n = len(cache[k + "_full"][0])
        bs = [theta(k, rng.integers(0, n, n)) for _ in range(NBOOT)]
        cb = np.percentile([x[0] for x in bs], [5, 95]).tolist()
        ca = np.percentile([x[1] for x in bs], [5, 95]).tolist()

        fb_, fa_ = frozen_at(ftb, g), frozen_at(fta, g)
        agree = (fb_ is None or abs(tb - fb_) <= 5e-4) and (fa_ is None or abs(ta - fa_) <= 5e-4)
        bad += 0 if agree else 1
        meas = 2 * g * 1e9 + LAYOUT_OFFSET_NS
        print(f"{g*1e9:>9.0f}{meas:>10.0f}{tb:>10.4f}"
              f"{('--' if fb_ is None else f'{fb_:.4f}'):>10}"
              f"{ta:>10.4f}{('--' if fa_ is None else f'{fa_:.4f}'):>10}"
              f"   [{cb[0]:+.4f}, {cb[1]:+.4f}]{'' if agree else '   MISMATCH'}")
        rows.append({"declared_s": g, "measured_ns": meas, "n_blocks": n,
                     "theta_b": tb, "theta_a": ta, "ci90_theta_b": cb, "ci90_theta_a": ca,
                     "frozen_theta_b": fb_, "frozen_theta_a": fa_,
                     "entry_rate_change_pct": 100 * (math.exp(tb) - 1)})

    out = {"source": "cache/gapfine_raw.npz", "NBOOT": NBOOT, "seed": SEED,
           "layout_offset_ns": LAYOUT_OFFSET_NS,
           "note": "measured separation = 2 * declared + layout offset", "rows": rows}
    json.dump(out, open(os.path.join(OUT, "timing_sweep.json"), "w", encoding="utf-8"), indent=2)

    print(f"\n  {len(strata)} separations recomputed from the raw counts, "
          f"{bad} disagreeing with the frozen artefact")
    print(f"  wrote {os.path.join(OUT, 'timing_sweep.json')}")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
