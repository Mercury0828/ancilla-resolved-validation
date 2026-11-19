"""Upper bounds on the transport ratio H(low)/H(0), by shared-block bootstrap.

The transport statistic is a ratio of squared displacement magnitudes measured at two separations.
Within one separation the exchangeable unit is the pair-unit `(job, pos // 2)`, which holds a
co-active arm and the isolated arm adjacent to it; the pairing is what removes the drift common to
both. Across separations the pair-unit is NOT shared, because arms at different separations occupy
different positions in the circuit. A bootstrap that draws pair-units and then intersects therefore
builds the numerator and the denominator of the ratio from disjoint halves of each draw, which
destroys the pairing the ratio depends on, and in the most degenerate case returns nan.

The draw used here is the ordinary cluster bootstrap for this design: draw the BLOCK `(job, cycle)`,
which every arm occupies, and carry every pair-unit inside a drawn block along with it.

🔴 The estimator does not move. `theta` stays defined as the mean of within-pair-unit differences,
exactly as frozen. Only the bootstrap draw changes. Both draws are computed and printed side by side,
and the point ratio R is identical under both.

    python analysis/block_bootstrap.py

Reads `data/cache/transport_raw.npz`, `data/cache/translow_raw.npz` and their manifests.
Writes `out/blockboot_verification.json`.
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

CACHE = os.path.join(RES, "cache")
NBOOT = 4000


def logit(p):
    return math.log(p / (1.0 - p))


def load(name):
    z = np.load(os.path.join(CACHE, name))
    return z[z.files[0]] if len(z.files) == 1 else {k: z[k] for k in z.files}


def series(D, arms, arm_name, anc):
    """{pair_unit: (log b, logit a)} -- exactly the frozen estimator's unit."""
    i = arms.index(arm_name)
    sel = (D[:, 3] == i) & (D[:, 4] == anc)
    out = {}
    for r in D[sel]:
        u = (int(r[0]), int(r[1]) // 2)
        b = r[6] / max(r[5], 1)
        a = r[8] / max(r[7], 1)
        out[u] = (math.log(max(b, 0.5 / max(r[5], 1))),
                  logit(min(max(a, 1e-6), 1 - 1e-6)))
    return out


def block_of(D):
    """pair_unit -> block (job, cycle). Verified single-valued before use."""
    m = {}
    for r in D:
        u = (int(r[0]), int(r[1]) // 2)
        b = (int(r[0]), int(r[2]))
        if u in m and m[u] != b:
            raise RuntimeError(f"pair-unit {u} spans blocks {m[u]} and {b}")
        m[u] = b
    return m


def run_campaign(manifest, cache_name, low_gap, seed):
    man = json.load(open(os.path.join(RES, manifest), encoding="utf-8"))
    D = load(cache_name)
    arms, cp, pairs = man["arms"], man["census_patches"], man["frozen_pairs"]
    c_w, eta = float(man["c_weight"]), float(man["eta"])

    cache = {}
    for p in pairs:
        q, tp, _ = p["key"].split("_")
        tgt = int(tp[1:])
        k = [cp[tgt][i] for i in (1, 3, 5, 7)].index(int(q[1:]))
        for g in (0, low_gap):
            cache[(p["key"], g)] = (series(D, arms, f"{p['key']}_full_g{g}", k),
                                    series(D, arms, f"p{tgt}_alone_g{g}", k))

    def H_of(sel=None):
        """sel is a multiset of pair-units; None means all, as in the frozen point estimate."""
        H = {0: 0.0, low_gap: 0.0}
        for p in pairs:
            for g in (0, low_gap):
                f, a = cache[(p["key"], g)]
                us = sorted(set(f) & set(a)) if sel is None else [u for u in sel if u in f and u in a]
                if not us:
                    continue
                tb = float(np.mean([f[u][0] - a[u][0] for u in us]))
                ta = float(np.mean([f[u][1] - a[u][1] for u in us]))
                H[g] += tb * tb + c_w * ta * ta
        return H

    Hpt = H_of()
    R = Hpt[low_gap] / max(Hpt[0], 1e-12)

    all_units = sorted({u for k in cache for i in (0, 1) for u in cache[k][i]})
    b_of = block_of(D)
    blocks = sorted({b_of[u] for u in all_units})
    by_block = {b: [u for u in all_units if b_of[u] == b] for b in blocks}

    out = {"R_point": R, "H0": Hpt[0], "Hlow": Hpt[low_gap], "eta": eta,
           "n_pair_units": len(all_units), "n_blocks": len(blocks)}

    for label, pool, expand in (
            ("draw_pair_units_as_frozen", all_units, lambda s: s),
            ("draw_blocks", blocks, lambda s: [u for b in s for u in by_block[b]])):
        rng = np.random.default_rng(seed)
        n = len(pool)
        boot = []
        for _ in range(NBOOT):
            s = expand([pool[i] for i in rng.integers(0, n, n)])
            h = H_of(s)
            boot.append(h[low_gap] / max(h[0], 1e-12))
        u95 = float(np.percentile(boot, 95))
        out[label] = {"U95": u95, "passes_eta": bool(u95 < eta),
                      "boot_median": float(np.percentile(boot, 50))}
    return out


def main():
    rep = {"note": "verification only; frozen artefacts untouched; estimator unchanged",
           "NBOOT": NBOOT, "campaigns": {}}

    print("=" * 82)
    print("  RESAMPLING-UNIT VERIFICATION")
    print("  theta stays on pair-units exactly as frozen. Only the bootstrap draw changes.")
    print("  eta unchanged. Nothing here relaxes a threshold.")
    print("=" * 82)

    for tag, manifest, cache_name, low, seed, frozen in (
            ("transport, low arm 4 us", "transport_manifest.json",
             "transport_raw.npz", 4000, 20260911, 0.316963),
            ("translow, low arm 624 ns", "translow_manifest.json",
             "translow_raw.npz", 296, 20260913, 0.216738)):
        try:
            r = run_campaign(manifest, cache_name, low, seed)
        except Exception as exc:                                    # noqa: BLE001
            print(f"\n{tag}: COULD NOT RECOMPUTE -- {type(exc).__name__}: {exc}")
            rep["campaigns"][tag] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        r["frozen_published_U95"] = frozen
        rep["campaigns"][tag] = r
        print(f"\n{tag}")
        print(f"  point estimate R {r['R_point']:.5f}  (unchanged by construction)")
        print(f"  {r['n_pair_units']} pair-units in {r['n_blocks']} blocks")
        print(f"  published in the frozen artefact              U95 {frozen:.5f}")
        for label in ("draw_pair_units_as_frozen", "draw_blocks"):
            v = r[label]
            print(f"  {label:<28}  U95 {v['U95']:.5f}   < eta {v['passes_eta']}")

    json.dump(rep, open(os.path.join(OUT, "blockboot_verification.json"), "w",
                        encoding="utf-8"), indent=2)
    print(f"\nwrote {os.path.join(OUT, 'blockboot_verification.json')}")

    # A campaign that could not be recomputed is recorded in the artefact as an error and would
    # otherwise leave this script exiting 0. Fail here instead, so a reader running the scripts on
    # their own sees it without having to reach reproduce.py.
    failed = [t for t, c in rep["campaigns"].items() if "error" in c]
    if failed:
        print(f"  could not recompute: {', '.join(failed)}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
