"""Held-out prediction: does a model calibrated on the isolated run mispredict the concurrent record?

Section marks (§) below refer to the analysis plan, which was fixed before any held-out score was
computed.

Model (§3): per cell, first-order Markov law b = P(1|0), a = P(1|1), Jeffreys pseudocount 0.5.
M_iso is fitted on the isolated arm, M_conc on the concurrent arm of the same cell, both on the
training jobs; both are scored on the concurrent arm of the held-out job (leave one job out).
ΔL = L(M_iso) − L(M_conc) in bits per scored transition, pooled with transition weights.

Log loss is linear in the counts once the model is fixed, so every quantity reduces to
per-(cell, job) count vectors (n0x, n01, n1x, n11); blocks are needed only for the supplementary
fixed-fit bootstrap.

    python analysis/heldout_prediction.py

Writes `out/heldout_prediction.json`.
"""

from __future__ import annotations

import itertools
import json
import os

import numpy as np

DATA = os.environ.get("FFCERT_DATA") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CACHE = os.path.join(DATA, "cache")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
OUT = os.path.join(os.environ.get("FFCERT_OUT") or os.path.join(_ROOT, "out"), "heldout_prediction.json")
NBOOT = 4000
SEED = 20261119
PSEUDO = 0.5
PAIRS = {"q63_p3_a2": (3, 2), "q21_p1_a3": (1, 3), "q1_p1_a0": (1, 0), "q147_p2_a1": (2, 2)}
LN2 = np.log(2.0)


# ---------------------------------------------------------------------------------------------
# cells: each cell = (iso counts per job, conc counts per job, conc counts per block)
# count vector order everywhere: (n0x, n01, n1x, n11)
# ---------------------------------------------------------------------------------------------

def _cell(jobs, iso_rows, conc_rows, iso_job, conc_job, conc_blk):
    J = len(jobs)
    jidx = {j: t for t, j in enumerate(jobs)}
    iso = np.zeros((J, 4)); conc = np.zeros((J, 4))
    for r, j in zip(iso_rows, iso_job):
        iso[jidx[j]] += r
    blocks = {}
    for r, j, b in zip(conc_rows, conc_job, conc_blk):
        conc[jidx[j]] += r
        blocks.setdefault((j, b), np.zeros(4))
        blocks[(j, b)] += r
    blk_keys = sorted(blocks)
    return {"iso": iso, "conc": conc,
            "blk_keys": [(int(k[0]), int(k[1])) for k in blk_keys],
            "blk_job": np.array([jidx[k[0]] for k in blk_keys]),
            "blk_cnt": np.array([blocks[k] for k in blk_keys])}


def cells_census():
    D = np.load(os.path.join(CACHE, "census4_raw.npz"))["rows"]
    cfgs = [tuple(c) for c in json.load(open(os.path.join(DATA, "census4_manifest.json"),
                                             encoding="utf-8"))["configurations"]]
    jobs = sorted(set(D[:, 0].tolist()))
    # census columns: job,pos,cycle,config,patch,k,n_tot,n_one,n1x,n11,n0x,n01
    cnt = np.stack([D[:, 10], D[:, 11], D[:, 8], D[:, 9]], axis=1).astype(float)
    out = {}
    for i in range(4):
        iso_ci = cfgs.index((i,))
        for k in range(4):
            si = (D[:, 3] == iso_ci) & (D[:, 4] == i) & (D[:, 5] == k)
            for ci, c in enumerate(cfgs):
                if i not in c or len(c) < 2:
                    continue
                sc = (D[:, 3] == ci) & (D[:, 4] == i) & (D[:, 5] == k)
                out[f"p{i}_k{k}_cfg{''.join(map(str, c))}"] = _cell(
                    jobs, cnt[si], cnt[sc], D[si, 0], D[sc, 0], D[sc, 2])
    return out, jobs


def cells_pairs(name, gaps):
    D = np.load(os.path.join(CACHE, f"{name}_raw.npz"))["rows"]
    M = json.load(open(os.path.join(DATA, f"{name}_manifest.json"), encoding="utf-8"))
    arms = M["arms"]
    shots = {(int(j), int(p)): int(s) for j, p, c, a, s in M["schedule"]}
    jobs = sorted(set(D[:, 0].tolist()))
    # columns: job,pos,cycle,arm,k,n0x,n01,n1x,n11
    cnt = D[:, 5:9].astype(float)
    row_at = {(int(r[0]), int(r[1]), int(r[4])): t for t, r in enumerate(D)}
    out, validation = {}, {}
    for key, (t, k) in PAIRS.items():
        for g in gaps:
            fa, aa = arms.index(f"{key}_full_g{g}"), arms.index(f"p{t}_alone_g{g}")
            fr = np.where((D[:, 3] == fa) & (D[:, 4] == k))[0]
            ar, ok, bad = [], 0, 0
            for x in fr:
                j, p = int(D[x, 0]), int(D[x, 1])
                partner = p ^ 1                                  # unit = (job, pos // 2)
                y = row_at.get((j, partner, k))
                if (y is not None and int(D[y, 3]) == aa and int(D[y, 2]) == int(D[x, 2])
                        and shots[(j, p)] == shots[(j, partner)]):
                    ar.append(y); ok += 1
                else:
                    bad += 1
            validation[f"{key}_g{g}"] = {"paired_units": ok, "unpaired": bad,
                                          "alone_rows_with_this_name": int(((D[:, 3] == aa) & (D[:, 4] == k)).sum())}
            ar = np.array(ar)
            out[f"{key}_g{g}"] = _cell(jobs, cnt[ar], cnt[fr], D[ar, 0], D[fr, 0], D[fr, 2])
    return out, jobs, validation


def cells_window():
    D = np.load(os.path.join(CACHE, "window_raw.npz"))["rows"]
    arms = json.load(open(os.path.join(DATA, "window_manifest.json"), encoding="utf-8"))["arms"]
    jobs = sorted(set(D[:, 0].tolist()))
    cnt = D[:, 4:8].astype(float)
    out = {}
    for g in (0, 192, 224, 256, 288, 320, 384, 512, 4000):
        si = D[:, 3] == arms.index(f"w{g}_alone")
        sc = D[:, 3] == arms.index(f"w{g}_full")
        out[f"q47_g{g}"] = _cell(jobs, cnt[si], cnt[sc], D[si, 0], D[sc, 0], D[sc, 2])
    return out, jobs


# ---------------------------------------------------------------------------------------------
# fitting and scoring
# ---------------------------------------------------------------------------------------------

def fit(T):
    """T[..., 4] training counts -> (b, a)."""
    b = (T[..., 1] + PSEUDO) / (T[..., 0] + 2 * PSEUDO)
    a = (T[..., 3] + PSEUDO) / (T[..., 2] + 2 * PSEUDO)
    return b, a


def loss_bits(C, b, a):
    """Total log loss in bits of counts C[..., 4] under (b, a)."""
    return -(C[..., 1] * np.log(b) + (C[..., 0] - C[..., 1]) * np.log1p(-b)
             + C[..., 3] * np.log(a) + (C[..., 2] - C[..., 3]) * np.log1p(-a)) / LN2


def ntrans(C):
    return C[..., 0] + C[..., 2]


def per_job_excess(ISO, CONC, m):
    """ISO, CONC: (cells, jobs, 4). m: (jobs,) multiplicities.

    For every held-out job j with m_j > 0: train on sum_k m_k C_k - m_j C_j (all copies of j
    excluded), score job j's concurrent counts. Returns (excess bits, transitions) per (cell, job),
    already multiplied by m_j.
    """
    Si = np.einsum("k,ckx->cx", m, ISO)
    Sc = np.einsum("k,ckx->cx", m, CONC)
    Ti = Si[:, None, :] - m[None, :, None] * ISO
    Tc = Sc[:, None, :] - m[None, :, None] * CONC
    bi, ai = fit(Ti)
    bc, ac = fit(Tc)
    ex = (loss_bits(CONC, bi, ai) - loss_bits(CONC, bc, ac)) * m[None, :]
    n = ntrans(CONC) * m[None, :]
    return ex, n


def stack(cells, names):
    return (np.stack([cells[c]["iso"] for c in names]),
            np.stack([cells[c]["conc"] for c in names]))


def draws(J, rng):
    out = []
    while len(out) < NBOOT:
        m = np.bincount(rng.integers(0, J, J), minlength=J).astype(float)
        if (m > 0).sum() >= 2:
            out.append(m)
    return np.array(out)


def pct(x):
    x = np.asarray(x, dtype=float)
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def panel(cells, jobs, M):
    """Point estimates (leave one job out) and job-cluster bootstrap for one panel."""
    names = sorted(cells)
    ISO, CONC = stack(cells, names)
    ex0, n0 = per_job_excess(ISO, CONC, np.ones(len(jobs)))
    pooled0 = 1000 * ex0.sum() / n0.sum()
    cell0 = 1000 * ex0.sum(1) / n0.sum(1)
    rep_pooled, rep_cell = [], []
    for m in M:
        ex, n = per_job_excess(ISO, CONC, m)
        rep_pooled.append(1000 * ex.sum() / n.sum())
        rep_cell.append(1000 * ex.sum(1) / n.sum(1))
    rep_cell = np.array(rep_cell)
    cells_out = {c: {"dL_mbit": float(cell0[t]), "ci": pct(rep_cell[:, t]),
                     "transitions_scored": float(n0.sum(1)[t])} for t, c in enumerate(names)}
    return {"pooled_dL_mbit": float(pooled0), "pooled_ci_job_bootstrap": pct(rep_pooled),
            "cells": cells_out,
            "n_cells_ci_above_0": int(sum(v["ci"][0] > 0 for v in cells_out.values())),
            "n_cells_ci_below_0": int(sum(v["ci"][1] < 0 for v in cells_out.values()))}, \
        np.array(rep_pooled), (ISO, CONC, names)


def calibration(cells, jobs):
    """§4: transition-weighted mean |ln(r / p̂)| for b and a, per predictor."""
    names = sorted(cells)
    ISO, CONC = stack(cells, names)
    S_i, S_c = ISO.sum(1, keepdims=True), CONC.sum(1, keepdims=True)
    bi, ai = fit(S_i - ISO)
    bc, ac = fit(S_c - CONC)
    rb = (CONC[..., 1] + 0.5) / (CONC[..., 0] + 1)
    ra = (CONC[..., 3] + 0.5) / (CONC[..., 2] + 1)
    wb, wa = CONC[..., 0], CONC[..., 2]
    f = lambda r, p, w: float((w * np.abs(np.log(r / p))).sum() / w.sum())
    return {"b": {"iso": f(rb, bi, wb), "conc": f(rb, bc, wb)},
            "a": {"iso": f(ra, ai, wa), "conc": f(ra, ac, wa)}}


def hand_check(cells):
    """§8: one cell recomputed with plain loops from its per-job counts."""
    c = sorted(cells)[0]
    iso, conc = cells[c]["iso"], cells[c]["conc"]
    J = len(iso)
    tot_ex = tot_n = 0.0
    for j in range(J):
        ti = sum(iso[k] for k in range(J) if k != j)
        tc = sum(conc[k] for k in range(J) if k != j)
        def L(t, C):
            b = (t[1] + 0.5) / (t[0] + 1); a = (t[3] + 0.5) / (t[2] + 1)
            return -(C[1] * np.log2(b) + (C[0] - C[1]) * np.log2(1 - b)
                     + C[3] * np.log2(a) + (C[2] - C[3]) * np.log2(1 - a))
        tot_ex += L(ti, conc[j]) - L(tc, conc[j]); tot_n += conc[j][0] + conc[j][2]
    ex, n = per_job_excess(iso[None], conc[None], np.ones(J))
    return {"cell": c, "loop_mbit": 1000 * tot_ex / tot_n,
            "vectorised_mbit": float(1000 * ex.sum() / n.sum())}


def symmetry_check(cells):
    """§8: swapping which arm trains which predictor flips the sign on the same scored data."""
    names = sorted(cells)
    ISO, CONC = stack(cells, names)
    m = np.ones(ISO.shape[1])
    Si, Sc = np.einsum("k,ckx->cx", m, ISO), np.einsum("k,ckx->cx", m, CONC)
    bi, ai = fit(Si[:, None] - ISO); bc, ac = fit(Sc[:, None] - CONC)
    fwd = (loss_bits(CONC, bi, ai) - loss_bits(CONC, bc, ac)).sum()
    rev = (loss_bits(CONC, bc, ac) - loss_bits(CONC, bi, ai)).sum()
    return {"forward_bits": float(fwd), "swapped_bits": float(rev), "flips": bool(np.isclose(fwd, -rev))}


def block_supplementary(cells, rng):
    """Supplementary §5 interval: fixed fold models, held-out blocks resampled with multiplicity."""
    names = sorted(cells)
    per_block = {}
    for c in names:
        cell = cells[c]
        ISO, CONC = cell["iso"], cell["conc"]
        bi, ai = fit(ISO.sum(0)[None] - ISO); bc, ac = fit(CONC.sum(0)[None] - CONC)
        j = cell["blk_job"]
        ex = loss_bits(cell["blk_cnt"], bi[j], ai[j]) - loss_bits(cell["blk_cnt"], bc[j], ac[j])
        n = ntrans(cell["blk_cnt"])
        for t, key in enumerate(cell["blk_keys"]):          # key = (job, cycle), shared across cells
            acc = per_block.setdefault(key, [0.0, 0.0])
            acc[0] += ex[t]
            acc[1] += n[t]
    arr = np.array(list(per_block.values()))
    nb = len(arr)
    reps = []
    for _ in range(NBOOT):
        w = np.bincount(rng.integers(0, nb, nb), minlength=nb).astype(float)
        reps.append(1000 * (w @ arr[:, 0]) / (w @ arr[:, 1]))
    return {"pooled_ci_block_fixed_fit": pct(reps), "n_blocks": nb}


def timing_contrast(cells, jobs, M, base_suffix, long_suffix):
    """R2: paired ΔL(base), ΔL(long), D and the ratio, with identical job draws."""
    base = {c: v for c, v in cells.items() if c.endswith(base_suffix)}
    long = {c: v for c, v in cells.items() if c.endswith(long_suffix)}
    Ib, Cb = stack(base, sorted(base)); Il, Cl = stack(long, sorted(long))

    def pooled(I, C, m):
        ex, n = per_job_excess(I, C, m)
        return 1000 * ex.sum() / n.sum()

    one = np.ones(len(jobs))
    pb, pl = pooled(Ib, Cb, one), pooled(Il, Cl, one)
    rb, rl = [], []
    for m in M:
        rb.append(pooled(Ib, Cb, m)); rl.append(pooled(Il, Cl, m))
    rb, rl = np.array(rb), np.array(rl)
    ratio = np.where(rb > 0, rl / np.where(rb > 0, rb, 1), np.inf)
    ratio_hi = float(np.percentile(ratio, 97.5)) if np.isfinite(np.percentile(ratio, 97.5)) else float("inf")
    base_ci, long_ci, d_ci = pct(rb), pct(rl), pct(rb - rl)
    return {"dL_base_mbit": pb, "ci_base": base_ci, "dL_long_mbit": pl, "ci_long": long_ci,
            "D_mbit": pb - pl, "ci_D": d_ci,
            "ratio_long_over_base": (pl / pb) if pb > 0 else None, "ratio_97_5": ratio_hi,
            "reading": {
                "reduced_excess_loss (D ci > 0)": d_ci[0] > 0,
                "residual_excess_loss (long ci > 0)": long_ci[0] > 0,
                "long ci contains 0": long_ci[0] <= 0 <= long_ci[1],
                "long favours isolated (long ci < 0)": long_ci[1] < 0,
                "restoration (base ci > 0 and ratio_97.5 < 0.50)": bool(base_ci[0] > 0 and ratio_hi < 0.50)}}


def main():
    rng = np.random.default_rng(SEED)
    res = {"plan": "fixed before any held-out score was computed", "NBOOT": NBOOT, "SEED": SEED}

    # P1 census
    cen, jobs_c = cells_census()
    M_c = draws(len(jobs_c), rng)
    p1, rep1, _ = panel(cen, jobs_c, M_c)
    p1["R1_met (2.5th pct > 0)"] = p1["pooled_ci_job_bootstrap"][0] > 0
    p1["n_cells"] = len(cen)
    p1["calibration"] = calibration(cen, jobs_c)
    p1["supplementary"] = block_supplementary(cen, rng)
    p1["checks"] = {"hand": hand_check(cen), "symmetry": symmetry_check(cen)}
    res["P1_census"] = p1

    # P2 translow, P2b transport
    for tag, name, gaps, longsuf in (("P2_translow", "translow", (0, 296), "_g296"),
                                     ("P2b_transport", "transport", (0, 4000), "_g4000")):
        cells, jobs, val = cells_pairs(name, gaps)
        M = draws(len(jobs), rng)
        allp, _, _ = panel(cells, jobs, M)
        out = {"pairing_validation": val, "cells": allp["cells"],
               "timing": timing_contrast(cells, jobs, M, "_g0", longsuf),
               "calibration_base": calibration({c: v for c, v in cells.items() if c.endswith("_g0")}, jobs),
               "calibration_long": calibration({c: v for c, v in cells.items() if c.endswith(longsuf)}, jobs),
               "checks": {"hand": hand_check(cells), "symmetry": symmetry_check(cells)}}
        res[tag] = out

    # P3 window (descriptive)
    win, jobs_w = cells_window()
    M_w = draws(len(jobs_w), rng)
    p3, _, _ = panel(win, jobs_w, M_w)
    res["P3_window_q47"] = {"cells": p3["cells"], "checks": {"symmetry": symmetry_check(win)}}

    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=2, default=float)

    print("P1 census: pooled dL = %.3f mbit/transition, 95%% job-bootstrap CI [%.3f, %.3f]; R1 met: %s"
          % (p1["pooled_dL_mbit"], *p1["pooled_ci_job_bootstrap"], p1["R1_met (2.5th pct > 0)"]))
    print("   cells %d: CI above 0: %d, below 0: %d" % (p1["n_cells"], p1["n_cells_ci_above_0"], p1["n_cells_ci_below_0"]))
    print("   supplementary block-bootstrap CI", p1["supplementary"])
    print("   calibration", p1["calibration"])
    print("   checks", p1["checks"])
    for tag in ("P2_translow", "P2b_transport"):
        t = res[tag]["timing"]
        print(f"{tag}: base {t['dL_base_mbit']:.3f} {t['ci_base']}  long {t['dL_long_mbit']:.3f} {t['ci_long']}"
              f"  D {t['D_mbit']:.3f} {t['ci_D']}  ratio {t['ratio_long_over_base']} (97.5% {t['ratio_97_5']})")
        print("   reading", t["reading"])
        print("   pairing", res[tag]["pairing_validation"])
        print("   checks", res[tag]["checks"])
    print("P3 window q47:")
    for c, v in sorted(res["P3_window_q47"]["cells"].items(), key=lambda kv: int(kv[0].split("_g")[1])):
        print(f"   {c:<10} {v['dL_mbit']:8.3f}  [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
