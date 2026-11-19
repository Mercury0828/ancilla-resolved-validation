"""The operating-point window sweep, by shared-block bootstrap.

The window statistic is the ratio |theta(g)| / |theta(0)| at each swept separation g, and the
question the sweep answers is whether a low-response operating point sits in a flat basin rather than
at a single measured zero. As in the transport ratio, the two sides of this ratio come from arms at
different positions, so pair-units are not shared across separations and the draw must be made on the
block `(job, cycle)`, which every arm occupies.

Two draws are computed and printed side by side. The window sweep additionally uses a STRATIFIED
draw, because the design is not one homogeneous pool: 64 blocks carry all nine gap arms and 16 carry
only the two endpoints, g = 0 and the 4 us arm. Every block carries g = 0, so the reference is never
at risk of being lost; what a pooled draw would distort is the MIXTURE. The denominator of the ratio
is available in all 80 blocks and each numerator in only 64, so a pooled draw builds the two sides
from different mixtures of strata. Each stratum is therefore resampled to its own size, 64 and 16. The 16-block stratum is
printed as `reference-only`, which is the label the frozen run used; it holds both endpoint
arms, g = 0 and 4 us, and no intermediate gap.

`theta` itself stays on pair-units exactly as frozen, so the point estimates do not move.

    python analysis/window_bootstrap.py

Reads `data/cache/window_raw.npz`, `data/window_manifest.json` and `data/window_result.json`.
Writes `out/window_boot_verification.json`.
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

CACHE = os.path.join(RES, "cache", "window_raw.npz")
MAN = os.path.join(RES, "window_manifest.json")
FROZEN = os.path.join(RES, "window_result.json")
NBOOT = 4000
THRESH = 0.20


def logit(p):
    return math.log(p / (1.0 - p))


def main():
    man = json.load(open(MAN, encoding="utf-8"))
    z = np.load(CACHE)
    D = z[z.files[0]]
    arms = man["arms"]
    meta = {a[0]: a for a in man["arm_meta"]}

    unit_of = {(j, p): (j, p // 2) for j, p, c, a, sh in man["schedule"]}
    block_of = {}
    for j, p, c, a, sh in man["schedule"]:
        u = (j, p // 2)
        b = (j, c)
        if u in block_of and block_of[u] != b:
            raise RuntimeError(f"pair-unit {u} spans blocks")
        block_of[u] = b

    def per_unit(arm_name):
        ai = arms.index(arm_name)
        out = {}
        for r in D[D[:, 3] == ai]:
            u = unit_of[(int(r[0]), int(r[1]))]
            b = r[5] / max(r[4], 1)
            a = r[7] / max(r[6], 1)
            out[u] = (math.log(max(b, 0.5 / max(r[4], 1))),
                      logit(min(max(a, 1e-6), 1 - 1e-6)))
        return out

    gaps = sorted({meta[a][1] for a in arms})
    cache = {g: (per_unit(f"w{g}_full"), per_unit(f"w{g}_alone")) for g in gaps}

    def theta(g, sel=None):
        f, a = cache[g]
        us = sorted(set(f) & set(a)) if sel is None else [u for u in sel if u in f and u in a]
        if not us:
            return float("nan"), float("nan")
        return (float(np.mean([f[u][0] - a[u][0] for u in us])),
                float(np.mean([f[u][1] - a[u][1] for u in us])))

    all_units = sorted({u for g in gaps for u in (set(cache[g][0]) & set(cache[g][1]))})
    blocks = sorted({block_of[u] for u in all_units})
    by_block = {b: [u for u in all_units if block_of[u] == b] for b in blocks}

    # The design is not one homogeneous pool. 64 blocks carry all nine gap arms; 16 carry only the
    # g = 0 reference and the 4 us arm. A pooled draw builds the ratio's denominator from a different
    # mixture of strata than its numerator, so the draw is stratified: 64 resampled from the complete
    # stratum and 16 from the reference-only stratum, each with replacement.
    covered = {}
    for g in gaps:
        f_, a_ = cache[g]
        for u in set(f_) & set(a_):
            covered.setdefault(block_of[u], set()).add(g)
    complete = sorted([b for b in blocks if len(covered[b]) == len(gaps)])
    partial = sorted([b for b in blocks if len(covered[b]) < len(gaps)])
    print(f"  {len(all_units)} pair-units in {len(blocks)} blocks "
          f"({len(complete)} complete, {len(partial)} reference-only)")

    frozen = json.load(open(FROZEN, encoding="utf-8"))
    ftab = frozen.get("W5_table") or frozen.get("table") or {}

    out = {"note": "verification only; window_result.json untouched",
           "NBOOT": NBOOT, "n_pair_units": len(all_units), "n_blocks": len(blocks), "rows": []}

    def draw_units(rng):
        return [all_units[i] for i in rng.integers(0, len(all_units), len(all_units))]

    def draw_blocks_fn(rng):
        sel = [blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]
        return [u for b in sel for u in by_block[b]]

    def draw_stratified(rng):
        sel = ([complete[i] for i in rng.integers(0, len(complete), len(complete))]
               + [partial[i] for i in rng.integers(0, len(partial), len(partial))])
        return [u for b in sel for u in by_block[b]]

    for label, drawer in (("draw_pair_units_as_frozen", draw_units),
                          ("draw_blocks", draw_blocks_fn),
                          ("draw_blocks_stratified", draw_stratified)):
        rng = np.random.default_rng(20260913)
        boots = [drawer(rng) for _ in range(NBOOT)]
        b0s = np.array([theta(0, s)[0] for s in boots])
        a0s = np.array([theta(0, s)[1] for s in boots])
        res = {}
        for g in [x for x in gaps if x != 0]:
            rb = np.abs(np.array([theta(g, s)[0] for s in boots])) / np.abs(b0s)
            ra = np.abs(np.array([theta(g, s)[1] for s in boots])) / np.abs(a0s)
            ub = float(np.nanpercentile(rb, 95))
            ua = float(np.nanpercentile(ra, 95))
            # The rule acts on M = max of the two ratios, so the bound it needs is the 95th
            # percentile of that maximum taken replicate by replicate. The larger of the two
            # marginal percentiles is a different quantity and is kept only because it is what the
            # frozen script computed.
            mstar = np.maximum(rb, ra)
            uj = float(np.nanpercentile(mstar, 95))
            res[g] = {"U95_b": ub, "U95_a": ua, "U95_H": max(ub, ua),
                      "U95_joint_M": uj,
                      "under": bool(max(ub, ua) < THRESH),
                      "under_joint": bool(uj < THRESH)}
        out[label] = res

    print(f"\n{'declared':>9}{'true ns':>9}{'theta_b':>10}{'theta_a':>10}"
          f"{'U95 frozen':>12}{'U95 block':>11}{'U95 strat':>11}"
          f"{'joint M':>10}{'<0.20':>8}{'joint<0.20':>12}")
    for g in [x for x in gaps if x != 0]:
        tb, ta = theta(g)
        f_ = out["draw_pair_units_as_frozen"][g]
        b_ = out["draw_blocks"][g]
        t_ = out["draw_blocks_stratified"][g]
        print(f"{g:>9}{2*g+44:>9}{tb:>+10.4f}{ta:>+10.4f}"
              f"{f_['U95_H']:>12.4f}{b_['U95_H']:>11.4f}{t_['U95_H']:>11.4f}"
              f"{t_['U95_joint_M']:>10.4f}{str(t_['under']):>8}"
              f"{str(t_['under_joint']):>12}")
        out["rows"].append({"declared_ns": g, "true_ns": 2 * g + 44, "theta_b": tb, "theta_a": ta,
                            "U95_frozen_draw": f_["U95_H"], "U95_block_draw": b_["U95_H"],
                            "U95_stratified": t_["U95_H"],
                            "U95_joint_M_stratified": t_["U95_joint_M"],
                            "under_0.20_block": b_["under"],
                            "under_0.20_stratified": t_["under"],
                            "under_0.20_joint_M": t_["under_joint"]})

    allq = all(r["under_0.20_stratified"] for r in out["rows"])
    print(f"\n  all swept gaps under 0.20, stratified block draw: {allq}")
    out["all_under_0.20_block"] = bool(all(r["under_0.20_block"] for r in out["rows"]))
    out["all_under_0.20_stratified"] = bool(allq)
    json.dump(out, open(os.path.join(OUT, "window_boot_verification.json"), "w",
                        encoding="utf-8"), indent=2)
    print(f"  wrote {os.path.join(OUT, 'window_boot_verification.json')}")


if __name__ == "__main__":
    main()
