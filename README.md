# Concurrent stabilizer extraction: data and analysis

Measurement data and analysis code for the manuscript *Ancilla-Resolved Validation and Timing Control of Concurrent Repetition-Code Memories on a Superconducting Processor*.

```
pip install -r requirements.txt
python reproduce.py
```

`reproduce.py` runs 24 checks covering the manuscript's headline numbers. Each numerical check is
printed beside the value published in the manuscript, the few structural checks print what they
assert, and the script exits non-zero if any disagrees. It needs no
quantum-computing account, no credentials and no network access.

## What is checked, and what is recomputed

The 24 checks do not all have the same standing, and the script says which is which as it runs.

**Recomputed from the raw counts, arithmetic and all**, every time you run it: the phenomenon in §1,
from `data/cache/ablation_raw.npz`.

**Recomputed if you run the analysis scripts first, otherwise read from the deposited artefacts:**
fourteen checks — the six timing coordinates in §4, and the eight bootstrap quantities in §2, §5,
§6 and §7. Each takes seconds to minutes, so the artefacts are deposited and `reproduce.py` reads
them by default. To recompute them instead:

```
python analysis/timing_sweep.py        # the fine timing sweep, from gapfine_raw.npz
python analysis/census_bootstrap.py    # census standard errors, from census4_raw.npz
python analysis/block_bootstrap.py     # transport bounds, from transport_raw and translow_raw
python analysis/window_bootstrap.py    # window sweep, from window_raw.npz
python analysis/d3_bootstrap.py        # distance-3 bounds, from d3logical_raw.npz
python reproduce.py
```

These write into `out/` and never overwrite the deposited copies in `data/`. `reproduce.py` prefers
`out/` when it exists and reports at the end which artefacts it read from where, so you can see
whether a number was recomputed on your machine or read as deposited.

`timing_sweep.py` compares its own coordinates against the deposited result and exits non-zero on
any disagreement. The four bootstrap scripts recompute and print but do not compare; `reproduce.py` is
what compares them, which is why the last step above re-runs it.

**Read from a deposited artefact and compared against the manuscript, not recomputed:** two checks —
the count of 48 directed effects over 16 ancillas in §2, and the transport anchor in §4b.

That is 8 recomputed always, 14 recomputed on request, and 2 read and compared.

Seven of the fifteen deposited caches are consumed here: six by the analysis scripts, plus
`ablation_raw.npz` by `reproduce.py` itself. The other eight are the records behind campaigns the
manuscript reports whose numbers are not among the 24 checks — the level-2 ablation, the coarse
timing sweep, the mitigation screen, the conjugate-basis campaign, the load contrast and its
replication, and the per-shot anchor registers behind the exploratory higher-order analysis. Their
results are deposited alongside them; the analysis that produced those results is not (see *What is
not here*).

## What is here

```
data/            frozen artefacts: design manifests, gate records, campaign results
data/cache/      the count tables and per-shot registers behind the recomputable results
analysis/        the five scripts behind the checks and seven further analyses
reproduce.py     the 24 checks
MANIFEST.sha256  a SHA-256 for every deposited file except itself
```

## The campaigns

Fourteen hardware campaigns on an IBM Heron processor. Each has a design manifest, a gate record and
a result; the campaigns that consumed QPU time also have a decision-gate record.

| prefix | what it measured |
|---|---|
| `ablation`, `ablation2` | which operation class in a co-active neighbour reproduces the effect |
| `census4` | 48 directed pair effects over 16 ancillas in a fresh region |
| `factorial` | the concurrency-penalty factorial that fixed the layout |
| `confirmation`, `validation` | the load contrast and its replication |
| `gapdisc`, `gapfine` | the inter-round timing sweep, coarse then fine |
| `transport`, `translow` | transport of the timing law to pairs frozen before the data |
| `window` | the operating-point window sweep |
| `mitigate` | the reset-context screen |
| `d3logical` | the distance-3 logical bridge |
| `xbasis` | the same measurement in the conjugate basis |

`factorial` carries a structural gate record rather than a decision gate: it fixed the layout and
consumed no QPU seconds under that gate.

Job identifiers in the manifests are the provider's opaque handles for the submitted jobs. They
record that the runs were distinct and when each was made; they resolve only inside the account that
submitted them and carry no account identity.

## The caches

`data/cache/` holds two kinds of file.

**Count tables** (`*_raw.npz`, `*_rows.npy`) are accumulated per job and circuit position. For every
campaign except the load contrast they are the sufficient statistics of the one-step law; the
`confirmation` and `validation` tables instead hold detector and logical aggregates. Shot-level detail beyond these counts is not recoverable
from them. Every table starts with `job index, position, syndrome cycle count`; the columns after
that differ by campaign, and this is the full list:

| file | cols | columns 3 onward |
|---|---|---|
| `gapdisc_raw`, `gapfine_raw`, `window_raw`, `xbasis_raw` | 8 | `arm, n0x, n01, n1x, n11` |
| `d3logical_raw` (`rows`) | 8 | `arm, n0x, n01, n1x, n11` |
| `transport_raw`, `translow_raw` | 9 | `arm, target, n0x, n01, n1x, n11` |
| `ablation2_raw` | 10 | `arm, n_meas, n_ones, n1x, n11, n0x, n01` |
| `confirmation_rows` | 10 | `config, patch, load, n_shots, n_detect, n_detect_slots, n_logical` |
| `mitigate_raw` | 12 | `arm, n0x, n01, n1x, n11` then four guard counts |
| `census4_raw` | 12 | `config, patch, ancilla, n_meas, n_ones, n1x, n11, n0x, n01` |
| `ablation_raw` | 15 | `arm, n_meas, n_ones, n_trans_slots, n_trans, n1x, n11` then three per-ancilla occupancy sums and `n_logical_slots, n_logical` |

`n0x` counts transitions observed starting from syndrome 0 and `n01` how many of those entered 1;
`n1x` and `n11` are the same starting from 1. **Note the order differs**: the tables with an
`n_meas, n_ones` prefix list `n1x, n11` before `n0x, n01`, and the others list `n0x, n01` first.
`arm` indexes the `arms` list in the matching manifest. Each deposited script names the columns it
uses at the point of use.

`validation_rows.npy` has 11 columns and is the earlier replication of the load contrast. No
deposited script consumes it, and its columns are not documented here.

**Per-shot registers** hold unpacked syndrome bits, shot by round by ancilla:
`anchor_shots.npz` (`<arm>__syn` of shape 131072 x 10 x 4, and `<arm>__dat` of shape 131072 x 5, for
each of eight arms), `d3logical_raw.npz` (`syn`, `dat`, `armid`) and `mitigate_p4_baseline.npz`
(`m`). The exploratory higher-order analysis in `data/higher_order.json` was computed from
`anchor_shots.npz`.

## Identifiers in the artefacts

The artefacts are the records written at the time each campaign ran, and are deposited as written
rather than rewritten for publication. Labels you will meet:

- `A12`, `A13` are preregistration amendments. `A12` is the concurrency-penalty factorial that fixed
  the qubit layout; every later campaign reusing it records `"layout": "FROZEN to A12"`, meaning the
  layout was inherited rather than re-selected against live device properties, which would be
  post-hoc layout selection. `A13` covers the campaigns reported here, and `"A13 requires both
  gates"` means a submission needed both a passing simulator gate and a passing decision gate.
- `A13-<campaign>` strings are the campaign labels recorded inside the manifests. Files are joined by
  filename prefix, not by this field. Five artefacts are cross-cutting rather than per-campaign:
  `blockboot_verification`, `census_boot_verification`, `d3_boot_verification`, `higher_order` and
  `logical_consequence`.
- A letter with a number — `D0`-`D2`, `F0`-`F4`, `G0`-`G6`, `M0`-`M8`, `T0`-`T5`, `W0`-`W5` — is a
  labelled decision rule of one campaign, in the order that campaign declares them, and is defined
  with its threshold in the `rules` or `gate` block of that campaign's manifest. Some label families
  live in a result file rather than a manifest: `Q1`-`Q3` are the census summary quantities, `R*` and
  `S*` are ablation result labels, and `A*` is overloaded — `A0`-`A4` in the conjugate-basis analysis
  are unrelated to the `A1`-`A6` gate diagnostics of the load contrast. `C3` names the configuration
  family used for the confirmation run. Search the campaign's own files for a label before assuming
  it means the same thing elsewhere.
- `N` followed by a number is a note in the project's working record. Those notes are not deposited.
  Anything a published number depends on is stated in the manuscript's Methods.

## Analyses added in this version

Seven scripts compute quantities that the manuscript and its supplement report beyond the 24 checks. They read the same
caches in `data/` and write into `out/` (or `$FFCERT_OUT`); each takes under half a minute.

```
python analysis/patch_aggregate.py      # patch aggregates of Fig. 2: pooled odds, RMS, maximum
python analysis/heldout_prediction.py   # held-out prediction (Section V, Table II, Fig. 3)
python analysis/heldout_scale.py        # excess loss as a share of the concurrent model's own loss
python analysis/higher_order_detail.py  # de-meaned detector correlations and q47 run statistics
python analysis/census_job_robustness.py     # census effects with jobs as the unit, and per background
python analysis/transport_job_robustness.py  # four-pair ratio per job left out, per pair
python analysis/heldout_reuse.py             # reusing the default-timing isolated fit at 624 ns
```

`patch_aggregate.py` needs `data/layout2.json`, the patch layout of the census, added in this version.
The held-out prediction fits a first-order syndrome law with a Jeffreys pseudocount, once on the
isolated arm and once on the concurrent arm of each cell, leaves one job out at a time, and scores both
fits on the concurrent arm of the held-out job. Its section marks refer to the analysis plan, fixed
before any held-out score was computed.

## Units and conventions

Separations are reported on the **measured causal axis**. A gap written into the circuit as `g`
produces a measured separation of `2g + δ`, where `δ` is layout-dependent: 44 ns on the incumbent
layout and 32 ns on the census layout. The manuscript and its figures use measured separation in
nanoseconds.

Declared gaps appear in the artefacts in two different units, by campaign. The fine timing sweep
keys its results in **seconds** (`"3.2e-08"`); the window sweep records declared **nanoseconds**
(`"declared_ns": 192`) alongside `"true_ns": 428`. `timing_sweep.py` and `window_bootstrap.py` print
declared and measured side by side so the two cannot be confused.

The one-step syndrome law is `T = [[1-b, b], [1-a, a]]`, with coordinates
`θ_b = log(b_full / b_alone)` and `θ_a = logit(a_full) - logit(a_alone)`, and `H = θ_b² + θ_a²`.

Four artefacts contain bare `NaN` where a frozen bootstrap produced no finite replicate:
`d3logical_result.json`, `d3_boot_verification.json`, `mitigate_result.json` and
`validation_result.json`. Python's `json` reads this; strict RFC-8259 parsers do not.
`analysis/d3_bootstrap.py` explains the two distance-3 cases and recomputes them finite.

## Resampling

Shots arrive in jobs, and each job runs a fixed number of syndrome cycles. The exchangeable unit is
the block `(job, cycle)`. Within a single separation, a co-active arm and the isolated arm adjacent
to it form a pair-unit `(job, pos // 2)`, and the pairing is what removes drift common to both.

For the timing, transport, window and distance-3 campaigns, point estimates are means of
within-pair-unit differences, and confidence bounds are cluster bootstraps that draw whole blocks
with replacement, keeping multiplicities and carrying every pair-unit inside a drawn block. Ratios
taken across separations require the block draw, because arms at different separations occupy
different circuit positions and so do not share pair-units.

The census estimator is different and is described in `analysis/census_bootstrap.py`: it averages a
conditional contrast over every measured background rather than differencing paired units.

Each analysis script computes both the block draw and the draw used in the original frozen run,
prints them side by side, and says which the manuscript reports. Where the two differ, the docstring
states why — in every case because the frozen draw discarded information the corrected draw keeps.

## What is not here

The submission and collection code that built these circuits, dispatched them and pulled their
results opens a provider session and cannot run without the account that collected the data.
Depositing it would give a reader something readable but not executable. Seven manifests — `d3logical`,
`gapdisc`, `gapfine`, `mitigate`, `transport`, `window` and `xbasis` — still record the filename and a
SHA-256 prefix of the analysis program that wrote their result, under `analysis_frozen`.

What is deposited instead is the data those circuits produced and the analysis a reader can actually
run against it.

## Requirements

Python 3.10 or later, NumPy. Nothing else. `requirements.txt` pins the version used, which is what
sets the Python floor.

## Licence

Code under `analysis/` and `reproduce.py`: MIT (`LICENSE-CODE`).
Data under `data/`: CC BY 4.0 (`LICENSE-DATA`).

The archived record carries CC BY 4.0, the licence of the data that makes up almost all of it; the
MIT grant over the analysis code stands alongside it and is the one that governs reuse of the
scripts.

## Citation

This deposit is archived on Zenodo. Version 2.0.0 is <https://doi.org/10.5281/zenodo.24376239>, and the
concept DOI <https://doi.org/10.5281/zenodo.24376238> resolves to the latest version. The code repository is <https://github.com/Mercury0828/ancilla-resolved-validation>.
Machine-readable metadata is in `CITATION.cff`. Please cite the manuscript alongside it.

Version 2.0.0 accompanies the manuscript under its present title. Against the earlier deposit
<https://doi.org/10.5281/zenodo.22307224> (version 1.0.1) it adds the
seven scripts listed under "Analyses added in this version" and `data/layout2.json`, and it
updates the title and metadata. No data file of version 1.0.1 has changed.
