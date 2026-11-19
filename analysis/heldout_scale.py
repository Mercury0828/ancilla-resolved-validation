"""Descriptive scale for the held-out prediction result: excess loss as a share of the concurrent model's own loss.

Not part of the analysis plan, which fixes ΔL in millibits per transition. This is a unit
conversion that lets a reader judge size: for each panel or cell, the held-out loss of the concurrent
fit L_conc (millibits per transition, leave one job out) and ΔL / L_conc in percent. It reuses the
functions of `analysis/heldout_prediction.py` and changes none of its outputs.

    python analysis/heldout_scale.py

Writes `out/heldout_scale.json`.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import heldout_prediction as W  # noqa: E402

OUT = os.path.join(os.environ.get("FFCERT_OUT") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out"), "heldout_scale.json")


def scale(cells, jobs):
    names = sorted(cells)
    ISO, CONC = W.stack(cells, names)
    S = CONC.sum(1, keepdims=True)
    b, a = W.fit(S - CONC)
    lc = W.loss_bits(CONC, b, a).sum()
    n = W.ntrans(CONC).sum()
    ex, _ = W.per_job_excess(ISO, CONC, np.ones(len(jobs)))
    return {"L_conc_mbit": float(1000 * lc / n), "dL_mbit": float(1000 * ex.sum() / n),
            "dL_share_pct": float(100 * ex.sum() / lc)}


def main():
    out = {"note": "descriptive, not prespecified; unit conversion of the held-out prediction result"}
    cen, jc = W.cells_census()
    out["census_pooled"] = scale(cen, jc)
    for name, gaps in (("translow", (0, 296)), ("transport", (0, 4000))):
        cells, jobs, _ = W.cells_pairs(name, gaps)
        for g in gaps:
            sub = {c: v for c, v in cells.items() if c.endswith(f"_g{g}")}
            out[f"{name}_pairs_g{g}"] = scale(sub, jobs)
    win, jw = W.cells_window()
    for c in sorted(win, key=lambda s: int(s.split("_g")[1])):
        out[f"window_{c}"] = scale({c: win[c]}, jw)
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2)
    for k, v in out.items():
        if isinstance(v, dict):
            print(f"{k:<28} L_conc {v['L_conc_mbit']:8.2f}  dL {v['dL_mbit']:8.3f}  share {v['dL_share_pct']:7.3f} %")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
