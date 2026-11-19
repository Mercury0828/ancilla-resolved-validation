"""Reuse of the default-timing isolated calibration at the selected separation.

At 624 ns the manuscript scores the isolated fit made at 624 ns (schedule-matched). A reader may ask
whether the isolated calibration made at the DEFAULT timing (declared gap 0, 32 ns) can be reused
after the separation is changed. For the four test pairs of the translow campaign this script scores,
on held-out jobs (leave one job out, all copies of a resampled job excluded from training):

  matched   L(M_iso fitted at 624 ns)  - L(M_conc fitted at 624 ns)   [the manuscript's number]
  reused    L(M_iso fitted at 32 ns)   - L(M_conc fitted at 624 ns)
  default   L(M_iso fitted at 32 ns)   - L(M_conc fitted at 32 ns)    [the manuscript's number]

all scored on the concurrent record at the separation named by the M_conc term, and reports the
absolute losses L_iso and L_conc as well. Job-cluster bootstrap, NBOOT 4000, as in
`analysis/heldout_prediction.py`. Descriptive; no verdict depends on it.

    python analysis/heldout_reuse.py

Writes `out/heldout_reuse.json`.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import heldout_prediction as W  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.environ.get("FFCERT_OUT") or os.path.join(_ROOT, "out"), "heldout_reuse.json")


def losses(ISO_fit, CONC_fit, CONC_score, m):
    """Leave-one-job-out losses (bits) on CONC_score for fits trained on ISO_fit and CONC_fit."""
    def lojo(T):
        S = np.einsum("k,ckx->cx", m, T)
        return S[:, None, :] - m[None, :, None] * T
    bi, ai = W.fit(lojo(ISO_fit))
    bc, ac = W.fit(lojo(CONC_fit))
    li = (W.loss_bits(CONC_score, bi, ai) * m[None, :]).sum()
    lc = (W.loss_bits(CONC_score, bc, ac) * m[None, :]).sum()
    n = (W.ntrans(CONC_score) * m[None, :]).sum()
    return 1000 * li / n, 1000 * lc / n


def main():
    cells, jobs, _ = W.cells_pairs("translow", (0, 296))
    keys = sorted(W.PAIRS)
    I0 = np.stack([cells[f"{k}_g0"]["iso"] for k in keys])
    C0 = np.stack([cells[f"{k}_g0"]["conc"] for k in keys])
    I1 = np.stack([cells[f"{k}_g296"]["iso"] for k in keys])
    C1 = np.stack([cells[f"{k}_g296"]["conc"] for k in keys])
    J = len(jobs)

    def stats(m):
        li0, lc0 = losses(I0, C0, C0, m)
        li1, lc1 = losses(I1, C1, C1, m)
        lr, _ = losses(I0, C1, C1, m)
        return {"default_32ns": (li0, lc0, li0 - lc0), "matched_624ns": (li1, lc1, li1 - lc1),
                "reused_32ns_iso_at_624ns": (lr, lc1, lr - lc1)}

    point = stats(np.ones(J))
    rng = np.random.default_rng(W.SEED)
    reps = [stats(m) for m in W.draws(J, rng)]
    out = {"note": "descriptive; reuse of the default-timing isolated fit at 624 ns", "jobs": J, "NBOOT": W.NBOOT}
    for name, (li, lc, d) in point.items():
        ds = [r[name][2] for r in reps]
        out[name] = {"L_iso_mbit": li, "L_conc_mbit": lc, "dL_mbit": d, "dL_ci": W.pct(ds)}
        print(f"{name:28s} L_iso {li:9.3f}  L_conc {lc:9.3f}  dL {d:+8.4f}  CI [{W.pct(ds)[0]:+.4f}, {W.pct(ds)[1]:+.4f}] mbit")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
