"""Higher-order structure of the syndrome record: de-meaned detector correlations and q47 run statistics.

Method notes:

  1. Source attribution. The per-shot registers in `anchor_shots.npz` are the q47 anchor arms carried
     inside the TRANSPORT jobs (declared gaps 0/64/256/4000 ns = 44/172/556/8044 ns scheduled), not
     the fine timing sweep.
  2. De-meaning. With one detector mean pooled over the four target ancillas and all rounds,
     ancillas with different baseline detector rates produce a positive "lag correlation" even
     when each stream is independent. Here every detector
     position (round t, ancilla j) gets its own mean, the covariance is taken between positions, and
     each covariance is normalised by the two positions' own standard deviations. Per-ancilla values
     are the average over rounds; the "mean over ancillas" is the average of those.
  3. Per ancilla, q47 included. q47 is target-ancilla column 2 (path position 5 =
     ancilla index 2).
  4. Direct run statistics on q47, with the ten-round boundary handled explicitly. A run start at
     round s is m_{s-1} = 0 and m_s = 1, so s >= 1 (round 0 has no observed predecessor). A run of at
     least k ones is observable only if s + k - 1 <= 9. The empirical run-start rate of >= 5 ones is
     the number of (shot, s) with s in 1..5 that start a run of >= 5, divided by the number of such
     (shot, s) opportunities. The first-order model's value on the SAME shots is p (1 - a) a^4
     (the formula of `reproduce.py` section 1), with p the occupancy and a = P(1 | prev 1).
     Survival: P(run reaches length k | run starts at s), over starts with s + k - 1 <= 9, against
     the model value a^(k-1).

The pooled quantities in `data/higher_order.json` are recomputed here as well (`pooled_*`) and must
agree with it, which is printed as a check.

Exploratory: no threshold is applied and no verdict depends on these
quantities. Detectors are d_t = m_t xor m_{t-1}, t = 1..9 (consecutive pairs), as in `data/higher_order.json`.

Intervals: 128 contiguous chunks of shots, resampled with replacement (the cache carries no job or
cycle labels, so chunks approximate blocks), NBOOT = 2000, seed 29.

    python analysis/higher_order_detail.py

Writes `out/higher_order_detail.json`.
"""

from __future__ import annotations

import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
RES = os.environ.get("FFCERT_DATA") or os.path.join(_ROOT, "data")
OUTDIR = os.environ.get("FFCERT_OUT") or os.path.join(_ROOT, "out")
CACHE = os.path.join(RES, "cache", "anchor_shots.npz")
OLD = os.path.join(RES, "higher_order.json")
OUT = os.path.join(OUTDIR, "higher_order_detail.json")
NCHUNK = 128
NBOOT = 2000
SEED = 29
GAPS = [("g0", 44), ("g64", 172), ("g256", 556), ("g4000", 8044)]
LAGS = (1, 2, 3)
KMAX = 5
Q47 = 2
NA = 4


def chunk_sums(syn):
    """Per-chunk sufficient statistics. Every reported quantity is a function of these sums."""
    s = syn.astype(np.float64)                         # (shots, 10, 4)
    d = np.abs(s[:, 1:, :] - s[:, :-1, :])            # (shots, 9, 4) detectors
    n, R, _ = s.shape
    T = d.shape[1]
    edges = np.linspace(0, n, NCHUNK + 1).astype(int)

    out = {k: [] for k in ("N", "S1", "SL", "SX", "ones", "prev1", "n11",
                           "start5", "opp5", "surv_num", "surv_den", "start1", "opp1",
                           "n11_prev", "n111")}
    for c in range(NCHUNK):
        sl = slice(edges[c], edges[c + 1])
        dc, sc = d[sl], s[sl]
        out["N"].append(dc.shape[0])
        out["S1"].append(dc.sum(axis=0))                                       # (T, A)
        out["SL"].append(np.stack([np.concatenate(
            [(dc[:, :-k, :] * dc[:, k:, :]).sum(axis=0), np.zeros((k, NA))]) for k in LAGS]))
        out["SX"].append(np.einsum("sti,stj->tij", dc, dc))                    # (T, A, A)
        out["ones"].append(sc.sum(axis=(0, 1)))                                # (A,)
        prev = sc[:, :-1, :]
        out["prev1"].append(prev.sum(axis=(0, 1)))
        out["n11"].append((prev * sc[:, 1:, :]).sum(axis=(0, 1)))
        # run statistics, boundary-aware (see module docstring)
        st5 = np.zeros(NA)
        for s0 in range(1, R - KMAX + 1):              # s = 1..5
            run = (1 - sc[:, s0 - 1, :]) * np.prod(sc[:, s0:s0 + KMAX, :], axis=1)
            st5 += run.sum(axis=0)
        out["start5"].append(st5)
        out["opp5"].append(np.full(NA, dc.shape[0] * (R - KMAX)))
        num = np.zeros((KMAX + 1, NA)); den = np.zeros((KMAX + 1, NA))
        for k in range(2, KMAX + 1):
            for s0 in range(1, R - k + 1):
                start = (1 - sc[:, s0 - 1, :]) * sc[:, s0, :]
                den[k] += start.sum(axis=0)
                num[k] += (start * np.prod(sc[:, s0:s0 + k, :], axis=1)).sum(axis=0)
        out["surv_num"].append(num); out["surv_den"].append(den)
        # run-start rate (any length): (shot, s) with s = 1..9, m_{s-1} = 0 and m_s = 1, per opportunity
        st1 = ((1 - sc[:, :-1, :]) * sc[:, 1:, :]).sum(axis=(0, 1))
        out["start1"].append(st1)
        out["opp1"].append(np.full(NA, dc.shape[0] * (R - 1)))
        # second-order persistence P(m_t = 1 | m_{t-1} = 1, m_{t-2} = 1)
        both = sc[:, :-2, :] * sc[:, 1:-1, :]
        out["n11_prev"].append(both.sum(axis=(0, 1)))
        out["n111"].append((both * sc[:, 2:, :]).sum(axis=(0, 1)))
    return {k: np.array(v, dtype=np.float64) for k, v in out.items()}, T


def evaluate(cs, T, w):
    N = float(cs["N"] @ w)
    S1 = np.tensordot(w, cs["S1"], 1) / N             # (T, A) per-position detector means
    SL = np.tensordot(w, cs["SL"], 1) / N             # (L, T, A)
    SX = np.tensordot(w, cs["SX"], 1) / N             # (T, A, A)
    v = S1 * (1 - S1)
    res = {}

    # de-meaned per position
    lag_anc = np.zeros((len(LAGS), NA))
    for li, k in enumerate(LAGS):
        cov = SL[li, :T - k, :] - S1[:-k, :] * S1[k:, :]
        corr = cov / np.sqrt(v[:-k, :] * v[k:, :])
        lag_anc[li] = corr.mean(axis=0)
        res[f"lag{k}_demeaned_mean"] = float(lag_anc[li].mean())
        res[f"lag{k}_demeaned_q47"] = float(lag_anc[li][Q47])
    cross = []
    for i in range(NA):
        for j in range(i + 1, NA):
            cov = SX[:, i, j] - S1[:, i] * S1[:, j]
            cross.append((cov / np.sqrt(v[:, i] * v[:, j])).mean())
    res["cross_demeaned_mean"] = float(np.mean(cross))
    cq = [np.mean((SX[:, Q47, j] - S1[:, Q47] * S1[:, j]) / np.sqrt(v[:, Q47] * v[:, j]))
          for j in range(NA) if j != Q47]
    res["cross_demeaned_q47"] = float(np.mean(cq))

    # old pooled definition, for the agreement check against higher_order.json
    mu = S1.mean()
    var = mu * (1 - mu)
    res["pooled_detector_rate"] = float(mu)
    for li, k in enumerate(LAGS):
        res[f"pooled_lag{k}"] = float((SL[li, :T - k, :].mean() - mu * mu) / var)
    iu = np.triu_indices(NA, 1)
    res["pooled_cross"] = float((np.mean([SX[:, i, j].mean() for i, j in zip(*iu)]) - mu * mu) / var)

    # q47 run statistics
    ones = cs["ones"].T @ w
    prev1 = cs["prev1"].T @ w
    n11 = cs["n11"].T @ w
    p = ones[Q47] / (N * 10)
    a = n11[Q47] / prev1[Q47]
    res["q47_p"] = float(p)
    res["q47_a"] = float(a)
    res["q47_run5_start_rate_empirical"] = float((cs["start5"].T @ w)[Q47] / (cs["opp5"].T @ w)[Q47])
    res["q47_run5_start_rate_model"] = float(p * (1 - a) * a ** 4)
    res["q47_run_start_rate_empirical"] = float((cs["start1"].T @ w)[Q47] / (cs["opp1"].T @ w)[Q47])
    res["q47_P1_given_11"] = float((cs["n111"].T @ w)[Q47] / (cs["n11_prev"].T @ w)[Q47])
    num = np.tensordot(w, cs["surv_num"], 1)
    den = np.tensordot(w, cs["surv_den"], 1)
    for k in range(2, KMAX + 1):
        res[f"q47_survival{k}_empirical"] = float(num[k, Q47] / den[k, Q47])
        res[f"q47_survival{k}_model"] = float(a ** (k - 1))
    return res


def main():
    z = np.load(CACHE)
    rng = np.random.default_rng(SEED)
    result = {}
    for tag, sep in GAPS:
        row = {}
        draws_arm = {}
        for arm in ("alone", "full"):
            cs, T = chunk_sums(z[f"anchor_{arm}_{tag}__syn"])
            point = evaluate(cs, T, np.ones(NCHUNK))
            dr = {k: np.empty(NBOOT) for k in point}
            for t in range(NBOOT):
                w = np.bincount(rng.integers(0, NCHUNK, NCHUNK), minlength=NCHUNK).astype(float)
                for k, val in evaluate(cs, T, w).items():
                    dr[k][t] = val
            draws_arm[arm] = dr
            row[arm] = {k: {"value": v, "lo": float(np.percentile(dr[k], 2.5)),
                            "hi": float(np.percentile(dr[k], 97.5))} for k, v in point.items()}
        # full/alone ratios of the q47 run-start rate, empirical and model, with intervals from
        # independent resampling of the two arms (they are separate sets of shots)
        ratio = {}
        for kind in ("empirical", "model"):
            key = f"q47_run5_start_rate_{kind}"
            pt = row["full"][key]["value"] / row["alone"][key]["value"]
            d = draws_arm["full"][key] / draws_arm["alone"][key]
            ratio[kind] = {"value": pt, "lo": float(np.percentile(d, 2.5)),
                           "hi": float(np.percentile(d, 97.5))}
        row["q47_run5_ratio_full_over_alone"] = ratio
        result[str(sep)] = row

    old = json.load(open(OLD, encoding="utf-8"))["by_separation"]
    agree = True
    for sep in result:
        for arm in ("alone", "full"):
            for new_k, old_k in (("pooled_detector_rate", "detector_rate"), ("pooled_lag2", "lag2"),
                                 ("pooled_lag3", "lag3"), ("pooled_cross", "cross_ancilla_corr")):
                a_, b_ = result[sep][arm][new_k]["value"], old[sep][arm][old_k]["value"]
                if abs(a_ - b_) > 1e-9:
                    agree = False
                    print(f"  pooled check MISMATCH {sep} {arm} {new_k}: {a_} vs {b_}")
    print(f"pooled quantities reproduce higher_order.json: {agree}")

    out = {"note": "exploratory, no threshold or verdict depends on it",
           "source": "cache/anchor_shots.npz = q47 anchor arms inside the TRANSPORT jobs",
           "q47_column": Q47, "NCHUNK": NCHUNK, "NBOOT": NBOOT, "SEED": SEED,
           "pooled_matches_old_json": agree,
           "separations_ns": [s for _, s in GAPS], "by_separation": result}
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2)

    keys = ["pooled_lag2", "lag2_demeaned_mean", "lag2_demeaned_q47", "lag3_demeaned_mean",
            "cross_demeaned_mean", "cross_demeaned_q47", "q47_run5_start_rate_empirical",
            "q47_run5_start_rate_model", "q47_survival5_empirical", "q47_survival5_model"]
    for sep in result:
        print(f"\n== {sep} ns")
        for k in keys:
            al, fu = result[sep]["alone"][k], result[sep]["full"][k]
            print(f"  {k:<34} alone {al['value']:.5f} [{al['lo']:.5f}, {al['hi']:.5f}]"
                  f"   full {fu['value']:.5f} [{fu['lo']:.5f}, {fu['hi']:.5f}]")
        r = result[sep]["q47_run5_ratio_full_over_alone"]
        print(f"  q47 run>=5 start-rate ratio  empirical {r['empirical']['value']:.3f} "
              f"[{r['empirical']['lo']:.3f}, {r['empirical']['hi']:.3f}]   model "
              f"{r['model']['value']:.3f} [{r['model']['lo']:.3f}, {r['model']['hi']:.3f}]")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
