"""Check the manuscript's headline numbers against this deposit. Offline, no account.

    python reproduce.py

24 checks. Each prints its value beside the one published in the manuscript, and anything that
disagrees sets a non-zero exit status.

THE THREE PROVENANCES, AND THE DIFFERENCE MATTERS. Of the 24 checks:

   8  are recomputed here every run, arithmetic and all, from the count table
      `data/cache/ablation_raw.npz`.

  14  are read from an artefact under `out/` if you have run the matching script in `analysis/`,
      and otherwise from the copy deposited in `data/`. These are the six timing coordinates and
      the eight bootstrap quantities, each of which takes seconds to minutes to recompute:

          python analysis/timing_sweep.py
          python analysis/census_bootstrap.py
          python analysis/block_bootstrap.py
          python analysis/window_bootstrap.py
          python analysis/d3_bootstrap.py

   2  are read from a deposited campaign result and compared, never recomputed here: the count of
      48 directed effects over 16 ancillas, and the transport anchor coordinate.

This script prefers `out/` when it is present, and says at the end which artefacts it read from
where, so a run always reports whether a number was recomputed on this machine or read as
deposited.

WHAT IS AND IS NOT IN THIS DEPOSIT. The measurement code that submitted these circuits builds a
provider session and cannot run without the account that collected the data, so shipping it would
give a reader something they could read but not execute. What is here instead is the data those
circuits produced, the frozen artefacts, and the analysis a reader can actually run against them.
The analysis behind the campaigns outside these 24 checks is not deposited; see the README.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("FFCERT_DATA") or os.path.join(_HERE, "data")
OUT = os.environ.get("FFCERT_OUT") or os.path.join(_HERE, "out")
CACHE = os.path.join(DATA, "cache")
TOL = 5e-4
EXPECTED_CHECKS = 24     # a skipped section must fail, not quietly shorten the run

# The nine ablation arms, in the order their index is written into column 3 of the cache.
ARMS = ["base", "full", "cx", "mr", "drive", "idle", "pad", "ctrl1", "ctrl3"]

CHECKS = []
RECOMPUTED = set()


def load(name):
    """Prefer a freshly computed artefact in `out/`, else the one frozen in `data/`."""
    p = os.path.join(OUT, name)
    if os.path.exists(p):
        RECOMPUTED.add(name)
    else:
        p = os.path.join(DATA, name)
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def check(label, got, published, tol=TOL, fmt="{:.5f}"):
    ok = abs(got - published) <= tol
    CHECKS.append(ok)
    mark = "ok " if ok else "MISMATCH"
    print(f"  {label:<46} {fmt.format(got):>12}   published {fmt.format(published):>12}   {mark}")
    return ok


def section1():
    """The phenomenon: one ancilla measured alone, then with one neighbouring patch co-active."""
    print("\n§1  the phenomenon -- the exemplar ancilla alone versus one patch co-active")
    D = np.load(os.path.join(CACHE, "ablation_raw.npz"))["rows"]
    out = {}
    for name in ("base", "full"):
        s = D[D[:, 3] == ARMS.index(name)]
        out[name] = {"p1": s[:, 5].sum() / s[:, 4].sum(),
                     "transition": s[:, 7].sum() / s[:, 6].sum(),
                     "a": s[:, 9].sum() / max(s[:, 8].sum(), 1)}
    check("P(m=1) alone", out["base"]["p1"], 0.037210, fmt="{:.6f}")
    check("P(m=1) co-active", out["full"]["p1"], 0.011855, fmt="{:.6f}")
    check("detector rate alone", out["base"]["transition"], 0.061459, fmt="{:.6f}")
    check("detector rate co-active", out["full"]["transition"], 0.014040, fmt="{:.6f}")
    check("P(1|prev=1) alone", out["base"]["a"], 0.176190, fmt="{:.6f}")
    check("P(1|prev=1) co-active", out["full"]["a"], 0.410433, fmt="{:.6f}")
    check("relative change %", 100 * (out["full"]["p1"] / out["base"]["p1"] - 1),
          -68.1, tol=0.1, fmt="{:+.1f}")

    # Under the one-step law a run of k ones has probability p (1-a) a^(k-1); the ratio of the two
    # arms at k = 5 is the quantity the manuscript quotes for run persistence.
    def run5(p, a):
        return p * (1 - a) * a ** 4

    r = run5(out["full"]["p1"], out["full"]["a"]) / run5(out["base"]["p1"], out["base"]["a"])
    check("runs of length >= 5, model-implied ratio", r, 6.71, tol=0.01, fmt="{:.2f}")


def section2():
    """The four-patch census."""
    print("\n§2  the four-patch census")
    r = json.load(open(os.path.join(DATA, "census4_result.json"), encoding="utf-8"))
    eff = r["effects"]
    targets = sorted({k.split("_")[0] for k in eff})
    ok = len(eff) == 48 and len(targets) == 16
    print(f"  {len(eff)} directed effects over {len(targets)} distinct ancillas"
          f"   {'ok ' if ok else 'MISMATCH'}")
    CHECKS.append(ok)
    v = load("census_boot_verification.json")
    check("Var(gamma) / pooled within, multiplicities kept", v["Q1"]["ratio_fixed"], 7.42,
          tol=0.02, fmt="{:.2f}")
    print(f"  Q3 localisation                                {v['Q3']['fixed_of_12']:>12} of 12"
          f"   published            9 of 12   "
          f"{'ok ' if v['Q3']['fixed_of_12'] == 9 else 'MISMATCH'}")
    CHECKS.append(v["Q3"]["fixed_of_12"] == 9)
    print(f"  survivor set identical to the frozen run       "
          f"{str(v['survivor_sets_identical']):>12}")
    CHECKS.append(bool(v["survivor_sets_identical"]))


def section_timing():
    """The timing coordinates the manuscript reports, and which campaign each comes from.

    The body reports the fine sweep. The exemplar anchors carried inside the transport jobs measure
    the same reversal at a different amplitude and are a separate campaign, so both are checked here
    and the two cannot be confused for one another.
    """
    print("\n§4  timing -- the fine sweep, which is what the body and Figure 3 report")
    sweep = os.path.join(OUT, "timing_sweep.json")
    if os.path.exists(sweep):
        # Recomputed from the raw transition counts by analysis/timing_sweep.py.
        RECOMPUTED.add("timing_sweep.json")
        rows = json.load(open(sweep, encoding="utf-8"))["rows"]
        tb = {str(r["declared_s"]): r["theta_b"] for r in rows}
        ta = {str(r["declared_s"]): r["theta_a"] for r in rows}
    else:
        fine = json.load(open(os.path.join(DATA, "gapfine_result.json"), encoding="utf-8"))
        tb, ta = fine["theta_b"], fine["theta_a"]
    # Keys are declared gaps in seconds. The manuscript reports measured separation, 2g + 44 ns.
    at = lambda d, g: d[next(k for k in d if abs(float(k) - g) < 1e-12)]        # noqa: E731
    check("theta_b at 44 ns", at(tb, 0.0), -0.9753, fmt="{:.4f}")
    check("theta_a at 44 ns", at(ta, 0.0), +0.9119, fmt="{:.4f}")
    check("theta_b at 172 ns", at(tb, 6.4e-08), +0.5350, fmt="{:.4f}")
    check("theta_a at 172 ns", at(ta, 6.4e-08), -0.3807, fmt="{:.4f}")
    check("entry-rate change at 44 ns (%)",
          100 * (math.exp(at(tb, 0.0)) - 1), -62.3, tol=0.05, fmt="{:.1f}")
    check("entry-rate change at 172 ns (%)",
          100 * (math.exp(at(tb, 6.4e-08)) - 1), +70.7, tol=0.05, fmt="{:.1f}")

    print("\n§4b timing -- the transport anchors, a separate campaign")
    anc = json.load(open(os.path.join(DATA, "transport_result.json"),
                         encoding="utf-8")).get("anchors") or {}
    row = anc.get("0") or anc.get(0) or {}
    if not isinstance(row, dict) or "theta_b" not in row:
        print("  anchor theta_b at 44 ns                      ABSENT from the artefact   MISMATCH")
        CHECKS.append(False)
    else:
        check("anchor theta_b at 44 ns", row["theta_b"], -1.0769, fmt="{:.4f}")


def section5():
    """Transport of the timing law to pairs frozen in advance."""
    print("\n§5  transport, shared-block bootstrap")
    v = load("blockboot_verification.json")
    for tag, pub in (("transport, low arm 4 us", 0.3221), ("translow, low arm 624 ns", 0.2221)):
        c = v["campaigns"][tag]
        check(tag, c["draw_blocks"]["U95"], pub, tol=5e-4, fmt="{:.4f}")
        print(f"      point ratio R {c['R_point']:.5f} unchanged by the draw; "
              f"{c['n_pair_units']} pair-units in {c['n_blocks']} blocks")


def section6():
    """The operating-point window."""
    print("\n§6  the operating-point window")
    v = load("window_boot_verification.json")
    EXPECTED_GAPS = [192, 224, 256, 288, 320, 384, 512, 4000]   # seven basin gaps + the reference
    got = [r["declared_ns"] for r in v["rows"]]
    if got != EXPECTED_GAPS:
        print(f"  swept gap set {got} is not the frozen set {EXPECTED_GAPS}   MISMATCH")
        CHECKS.append(False)
        return
    basin = [r for r in v["rows"] if r["declared_ns"] != 4000]
    allq = len(basin) == 7 and all(r["under_0.20_stratified"] for r in basin)
    print(f"  all seven basin gaps under 0.20, stratified      {str(allq):>12}")
    CHECKS.append(allq)
    for r in v["rows"]:
        print(f"      {r['declared_ns']:>5} declared = {r['true_ns']:>5} ns true   "
              f"U95 {r['U95_stratified']:.4f} stratified, {r['U95_block_draw']:.4f} pooled")


def section7():
    """The distance-3 ratio bounds, recomputed finite."""
    print("\n§7.1  d = 3, the ratio bounds recomputed finite")
    v = load("d3_boot_verification.json")
    check("U95 ratio b, shared-block draw", v["draw_blocks"]["U95_ratio_b"], 0.3264,
          tol=5e-4, fmt="{:.4f}")
    check("U95 ratio a, shared-block draw", v["draw_blocks"]["U95_ratio_a"], 1.3824,
          tol=5e-4, fmt="{:.4f}")
    print(f"      the frozen draw gave {v['draw_g0_units_as_frozen']['finite_replicates']} finite "
          f"replicates of 4000; the decision is unaffected and failed on sign as well as magnitude")


def main():
    print("=" * 92)
    print("  FF-CERT -- the manuscript's headline numbers checked against this deposit, offline")
    print("=" * 92)
    section1()
    section2()
    section_timing()
    section5()
    section6()
    section7()
    bad = CHECKS.count(False)
    print("\n" + "=" * 92)
    print(f"  {len(CHECKS)} checks, {bad} mismatches")
    if len(CHECKS) != EXPECTED_CHECKS:
        print(f"  {len(CHECKS)} checks ran but {EXPECTED_CHECKS} are expected: a section was "
              f"skipped rather than failing")
        bad += 1
    if RECOMPUTED:
        print(f"  {len(RECOMPUTED)} artefact(s) read from out/, recomputed on this machine: "
              f"{', '.join(sorted(RECOMPUTED))}")
    else:
        print("  every artefact read from data/ as deposited; run the five scripts in "
              "analysis/ to recompute them here")
    print("=" * 92)
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
