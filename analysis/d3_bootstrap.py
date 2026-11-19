"""The distance-3 logical ratio bounds, recomputed finite.

`data/d3logical_result.json` carries `U95_ratio_b = nan` and `U95_ratio_a = nan`. The cause is the
pair-unit draw described in `block_bootstrap.py`, in its most visible form: the units were built from
the g = 0 arms only, so no replicate contained a unit belonging to the gap arm, `theta(gap, s)`
averaged an empty list, and the ratio was nan in every replicate.

No published claim rests on those two fields. The decision they belong to failed on the one-sided
bounds of `theta(0)`, which are finite and are computed from the same g = 0 units the estimator uses.
They are recomputed here with the shared-block draw so that a published nan is explained by a number
rather than by an argument.

    python analysis/d3_bootstrap.py

Reads `data/cache/d3logical_raw.npz` and `data/d3logical_manifest.json`.
Writes `out/d3_boot_verification.json`.
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

CACHE = os.path.join(RES, "cache", "d3logical_raw.npz")
MAN = os.path.join(RES, "d3logical_manifest.json")
NBOOT = 4000


def logit(p):
    return math.log(p / (1.0 - p))


def main():
    man = json.load(open(MAN, encoding="utf-8"))
    z = np.load(CACHE)
    D = z[z.files[0]]
    arms = man["arms"]
    gaps = sorted({a[1] for a in man["arm_meta"]})
    gap = max(gaps)

    unit_of = {(j, p): (j, p // 2) for j, p, c, a, sh in man["schedule"]}
    block_of = {}
    for j, p, c, a, sh in man["schedule"]:
        block_of[(j, p // 2)] = (j, c)

    def per_unit(nm):
        i = arms.index(nm)
        out = {}
        for r in D[D[:, 3] == i]:
            u = unit_of[(int(r[0]), int(r[1]))]
            b = r[5] / max(r[4], 1)
            a = r[7] / max(r[6], 1)
            out[u] = (math.log(max(b, 0.5 / max(r[4], 1))),
                      logit(min(max(a, 1e-6), 1 - 1e-6)))
        return out

    cache = {g: (per_unit(f"d3_full_g{g}"), per_unit(f"d3_alone_g{g}")) for g in gaps}

    def theta(g, sel=None):
        f, a = cache[g]
        us = sorted(set(f) & set(a)) if sel is None else [u for u in sel if u in f and u in a]
        if not us:
            return float("nan"), float("nan")
        return (float(np.mean([f[u][0] - a[u][0] for u in us])),
                float(np.mean([f[u][1] - a[u][1] for u in us])))

    units0 = sorted(set(cache[0][0]) & set(cache[0][1]))
    all_units = sorted({u for g in gaps for u in (set(cache[g][0]) & set(cache[g][1]))})
    blocks = sorted({block_of[u] for u in all_units})
    by_block = {b: [u for u in all_units if block_of[u] == b] for b in blocks}

    print("=" * 78)
    print("  d = 3 ratio bounds -- frozen draw versus shared-block draw")
    print("=" * 78)
    print(f"  {len(units0)} pair-units in the g = 0 arms, {len(all_units)} across both gaps, "
          f"{len(blocks)} blocks")

    tb0, ta0 = theta(0)
    tbg, tag = theta(gap)
    print(f"  theta_b(0) {tb0:+.4f}   theta_a(0) {ta0:+.4f}")
    print(f"  theta_b({gap}) {tbg:+.4f}   theta_a({gap}) {tag:+.4f}")

    out = {"note": "verification only; d3logical_result.json untouched", "NBOOT": NBOOT,
           "gap_ns_declared": gap, "theta_b_0": tb0, "theta_a_0": ta0,
           "theta_b_gap": tbg, "theta_a_gap": tag}

    for label, drawer in (
            ("draw_g0_units_as_frozen",
             lambda rng: [units0[i] for i in rng.integers(0, len(units0), len(units0))]),
            ("draw_blocks",
             lambda rng: [u for b in [blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]
                          for u in by_block[b]])):
        rng = np.random.default_rng(20260916)
        boots = [drawer(rng) for _ in range(NBOOT)]
        b0s = np.array([theta(0, s)[0] for s in boots])
        a0s = np.array([theta(0, s)[1] for s in boots])
        rb = np.abs(np.array([theta(gap, s)[0] for s in boots])) / np.abs(b0s)
        ra = np.abs(np.array([theta(gap, s)[1] for s in boots])) / np.abs(a0s)
        with np.errstate(invalid="ignore"):
            ub = float(np.nanpercentile(rb, 95)) if np.isfinite(rb).any() else float("nan")
            ua = float(np.nanpercentile(ra, 95)) if np.isfinite(ra).any() else float("nan")
        finite = int(np.isfinite(rb).sum())
        out[label] = {"U95_ratio_b": ub, "U95_ratio_a": ua, "finite_replicates": finite}
        print(f"  {label:<26} U95_ratio_b {ub:>8.4f}   U95_ratio_a {ua:>8.4f}   "
              f"finite replicates {finite}/{NBOOT}")

    print("\n  D0 is unaffected: it tests the one-sided bounds of theta(0), which are finite in both")
    print("  draws and were computed from the same g = 0 units the estimator uses. D0 FAILED, and")
    print("  still fails, on sign as well as magnitude.")

    json.dump(out, open(os.path.join(OUT, "d3_boot_verification.json"), "w",
                        encoding="utf-8"), indent=2)
    print(f"\n  wrote {os.path.join(OUT, 'd3_boot_verification.json')}")


if __name__ == "__main__":
    main()
