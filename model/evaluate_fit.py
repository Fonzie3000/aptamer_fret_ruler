"""
evaluate_fit.py

Evaluates the orientation-dependent FRET model at the reported parameter
vector and writes three tables to out/.

  1. per_construct.csv: for each of the seven constructs, the predicted
     efficiency, the donor-acceptor separation r_DA (fitted and crystal),
     the orientation factor <kappa^2>, the effective Forster radius, the
     residual in units of the replicate SD and the chi-squared contribution.
  2. phase_gain_profile.csv: the profile of the phase-gain parameter. Each
     row is a parameter vector refitted at a fixed phase gain, read from
     results/phase_profile_fits.csv. Its Delta chi-squared is a paired
     difference against the reported fit over the same seed pairs, so the +4
     row is zero by construction. Values are the data term only; the active
     fraction, its one-sided penalty and the implied true efficiency at 8 bp
     are given in their own columns.
  3. phase_gain_per_construct.csv: how the chi-squared gain between a phase
     gain of +4 and +10 deg per base pair divides across the constructs.

Run it from any directory: every path is resolved from this file.

    python model/evaluate_fit.py            # full: 60000 samples, 20 seed pairs
    python model/evaluate_fit.py --quick    # sanity pass of a few seconds

--quick uses fewer samples and seed pairs, so its numbers are not the
reported ones. QUICK=1 in the environment does the same.

Reads  geometry/reference/ruler_all_data.py, results/phase_profile_fits.csv
       and data/6c63_dssr-pairFrames.txt
Writes out/. Nothing outside the repository is read or written.
"""

import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import fret_model as ff
import ruler_constants as const

# Repository layout. Resolved from this file, never from the working
# directory, so the script behaves the same on Linux, Windows and macOS.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GEOMETRY = ROOT / "geometry" / "reference" / "ruler_all_data.py"
PROFILE_CSV = ROOT / "results" / "phase_profile_fits.csv"
PAIRFRAMES_6C63 = ROOT / "data" / "6c63_dssr-pairFrames.txt"
OUT = ROOT / "out"

# A missing pairFrames file would not raise: prepare_axes falls back to a
# pivot about 16 A away and the fit would change without warning. So the
# inputs are checked before anything is computed.
_missing = [str(p) for p in (GEOMETRY, PROFILE_CSV, PAIRFRAMES_6C63)
            if not p.is_file()]
if _missing:
    raise SystemExit("missing input file(s):\n  " + "\n  ".join(_missing))

QUICK = bool(os.environ.get("QUICK")) or "--quick" in sys.argv[1:]
N_SAMP = 8000 if QUICK else 60000
N_SEEDS = 4 if QUICK else 20
PARAM_NAMES = ["tilt_deg", "hinge_deg", "R0_iso", "dtwist_per_bp",
               "phi0_deg", "active_fraction"]

# The reported five-parameter fit, written out rather than read, so a rerun
# cannot pick up a stale file. results/fit_parameters.json carries the same
# six numbers.
HEADLINE = {"tilt_deg": 13.86147367211555,
            "hinge_deg": 0.0,
            "R0_iso": 55.11540847492566,
            "dtwist_per_bp": 4.0,
            "phi0_deg": -14.58833797900234,
            "active_fraction": 0.803029013518739}

OUT.mkdir(parents=True, exist_ok=True)

DATA = ff.load_geometry(str(GEOMETRY))
BP = np.asarray(const.EXP_BP, float)
E = np.asarray(const.EXP_E, float)
SD = np.asarray(const.EXP_SD, float)
SPEC = [s for s in ff.make_default_specs()
        if s.name == "M1_global_active_fraction"][0]
FLOOR = float(E.max())
FLOOR_SD = float(SD[int(E.argmax())])
BP_REF = int(min(d["bp"] for d in DATA))


def penalty(f):
    return 0.0 if f >= FLOOR else ((FLOOR - f) / FLOOR_SD) ** 2


def make_cfg(cone=45.0):
    c = ff.ModelConfig(verbose=False)
    c.pivot_point = const.JUNCTION.copy()
    c.helical_axis = const.STEM_AXIS / np.linalg.norm(const.STEM_AXIS)
    c = ff.prepare_axes(DATA, c, pairframe_6c63_path=str(PAIRFRAMES_6C63))
    c.acceptor_wobble_deg, c.donor_wobble_deg = cone, 15.0
    c.bounds = dict(ff.DEFAULT_BOUNDS)
    c.bounds["R0_iso"] = (45.0, 60.0)
    c.bounds["dtwist_per_bp"] = (-4.0, 12.0)
    c.n_samples_objective = 8000
    c.n_samples_report = N_SAMP
    c.objective_seeds = (101, 202)
    return c


CFG = make_cfg()

# Continuity guard. Without it the
# accumulated twist advances as a staircase and anything off integer bp,
# or past the +4 bound, is wrong.
_pr = dict(DATA[0])
_gp = {k: 0.0 for k in PARAM_NAMES}
_gp["dtwist_per_bp"] = 10.0
_a, _ = ff.place_acceptor({**_pr, "bp": 11.999}, _gp, CFG, 6)
_b, _ = ff.place_acceptor({**_pr, "bp": 12.001}, _gp, CFG, 6)
if float(np.linalg.norm(np.asarray(_a) - np.asarray(_b))) > 0.05:
    def _place(dr, gp, c, bp_ref):
        tw = gp["phi0_deg"] + gp["dtwist_per_bp"] * (float(dr["bp"]) - float(bp_ref))
        RA = ff.rotate_about_line(np.asarray(dr["R_A"], float),
                                  c.pivot_point, c.helical_axis, tw)
        mA = ff.rotation_matrix(c.helical_axis, tw) @ np.asarray(dr["mu_A"], float)
        RA = ff.rotate_about_line(RA, c.pivot_point, c.tilt_axis, gp["tilt_deg"])
        mA = ff.rotation_matrix(c.tilt_axis, gp["tilt_deg"]) @ mA
        return RA, ff._normalise(mA)
    ff.place_acceptor = _place
    print("continuity guard: repaired the int(bp) staircase")

# Fixed seed stream, so every run draws the same seed pairs.
_rng = np.random.default_rng(20260901)
SEEDS = [tuple(int(x) for x in _rng.integers(1, 10 ** 6, 2)) for _ in range(N_SEEDS)]
print(f"{len(SEEDS)} seed pairs, {N_SAMP} samples per construct"
      f"{'  [QUICK]' if QUICK else ''}")


def chi2_matrix(p):
    """Per-seed, per-construct chi2. Shape (n_seed, 7). Data term only."""
    rows = []
    for sp in SEEDS:
        _, ep, k2 = ff.predict(DATA, p, CFG, SPEC, N_SAMP, sp)
        rows.append(((np.asarray(ep) - E) / SD) ** 2)
    return np.vstack(rows)


def predict_mean(p):
    eps, k2s = [], []
    for sp in SEEDS:
        _, ep, k2 = ff.predict(DATA, p, CFG, SPEC, N_SAMP, sp)
        eps.append(np.asarray(ep))
        k2s.append(np.asarray(k2))
    return np.mean(eps, axis=0), np.mean(k2s, axis=0)


def r_da(p):
    geom = ff._extract_geom_params(p)
    out = []
    for d in sorted(DATA, key=lambda x: int(x["bp"])):
        RA, _ = ff.place_acceptor(d, geom, CFG, BP_REF)
        out.append(float(np.linalg.norm(np.asarray(RA, float)
                                        - np.asarray(d["R_D"], float))))
    return np.array(out)


T0 = time.time()

# ------------------------------------------------- 1. per-construct values
print("\n=== reported fit: per-construct quantities ===")
ep_h, k2_h = predict_mean(HEADLINE)
chi_h = chi2_matrix(HEADLINE)
r_fit = r_da(HEADLINE)
r_cry = np.array([float(np.linalg.norm(np.asarray(d["R_A"], float)
                                       - np.asarray(d["R_D"], float)))
                  for d in sorted(DATA, key=lambda x: int(x["bp"]))])
z_h = (ep_h - E) / SD
R0_eff = HEADLINE["R0_iso"] * (k2_h / (2.0 / 3.0)) ** (1.0 / 6.0)

PC = pd.DataFrame({"bp": BP.astype(int), "E": E, "SD": SD,
                   "E_pred": np.round(ep_h, 4),
                   "r_DA": np.round(r_fit, 1),
                   "r_DA_crystal": np.round(r_cry, 1),
                   "kappa2": np.round(k2_h, 3),
                   "R0_eff": np.round(R0_eff, 1),
                   "residual_sigma": np.round(z_h, 2),
                   "chi2_i": np.round(chi_h.mean(axis=0), 2)})
PC.to_csv(OUT / "per_construct.csv", index=False)
print(PC.to_string(index=False))

steps = np.diff(r_fit)
print(f"\n  data chi2, sum over constructs : {chi_h.mean(axis=0).sum():.2f}")
print(f"  penalty at this f              : {penalty(HEADLINE['active_fraction']):.2f}")
print(f"  orientation factor range       : {k2_h.min():.3f} to {k2_h.max():.3f}")
print(f"  effective Forster radius range : {R0_eff.min():.1f} to {R0_eff.max():.1f} A")
print(f"  largest fitted-crystal r_DA gap: {np.abs(r_fit - r_cry).max():.1f} A")
print(f"  r_DA at the shortest construct : {r_fit[0]:.1f} A")
print(f"  r_DA added per 2 bp step       : {steps.min():.1f} to {steps.max():.1f} A")

# ------------------------------------------------------- 2. phase profile
print("\n=== phase profile, paired and re-anchored on +4 ===")
T = pd.read_csv(PROFILE_CSV)
if QUICK:
    T = T[T.dtwist_per_bp.isin([4.0, 6.0, 10.0])]
T = T.sort_values("dtwist_per_bp").reset_index(drop=True)

mats = {}
sweep4 = None
for _, row in T.iterrows():
    k = float(row.dtwist_per_bp)
    p = {q: float(row[q]) for q in PARAM_NAMES}
    m = chi2_matrix(p)
    print(f"  dtwist {k:+5.1f}  data chi2 {m.sum(axis=1).mean():8.2f}  "
          f"({time.time()-T0:.0f}s)")
    if k == 4.0:
        # The sweep's own +4 solution is beaten by the reported headline vector,
        # which has the same dtwist. Anchor on the headline and keep the sweep's
        # row only as a diagnostic.
        sweep4 = m
        m = chi_h
    mats[k] = m

anchor = mats[4.0].sum(axis=1)
if sweep4 is not None:
    d4 = sweep4.sum(axis=1) - anchor
    print(f"\n  the sweep's own +4 row scores {d4.mean():+.2f} +/- "
          f"{d4.std(ddof=1)/np.sqrt(len(d4)):.2f} against the headline vector")
rows = []
for k in sorted(mats):
    tot = mats[k].sum(axis=1)
    d = tot - anchor
    f = (HEADLINE["active_fraction"] if k == 4.0 else
         float(T.loc[(T.dtwist_per_bp - k).abs().idxmin()].active_fraction))
    rows.append(dict(dtwist=k, eff_twist=round(32.98 - k, 2),
                     data_chi2=round(float(tot.mean()), 2),
                     dchi2=round(float(d.mean()), 2),
                     dchi2_SE=round(float(d.std(ddof=1) / np.sqrt(len(d))), 2)
                     if len(d) > 1 else np.nan,
                     f=round(f, 4), penalty=round(penalty(f), 2),
                     sigma_below_floor=round((FLOOR - f) / FLOOR_SD, 2),
                     E_true_8bp=round(float(E[1] / f), 4)))
PR = pd.DataFrame(rows)
PR.to_csv(OUT / "phase_gain_profile.csv", index=False)
print("\n" + PR.to_string(index=False))

mn = PR.loc[PR.dchi2.idxmin()]
print(f"\n  minimum at dtwist {mn.dtwist:+.0f}, effective twist {mn.eff_twist:.2f} "
      f"deg/bp, depth {mn.dchi2:+.2f} +/- {mn.dchi2_SE:.2f}")
within1 = PR[PR.dchi2 <= mn.dchi2 + 1.0].dtwist.tolist()
print(f"  within one unit of the minimum: {within1}")
below = PR[PR.dtwist < 4.0]
if len(below):
    print(f"  rows below the bound, all positive: "
          f"{dict(zip(below.dtwist, below.dchi2))}")

# --------------------------------------- 3. where the gain lands, +4 vs +10
if 10.0 in mats:
    print("\n=== per-construct decomposition, +4 against +10 ===")
    d = mats[10.0] - mats[4.0]
    G = pd.DataFrame({"bp": BP.astype(int),
                      "chi2_at_4": np.round(mats[4.0].mean(axis=0), 2),
                      "chi2_at_10": np.round(mats[10.0].mean(axis=0), 2),
                      "gain": np.round(d.mean(axis=0), 2),
                      "gain_SE": np.round(d.std(axis=0, ddof=1)
                                          / np.sqrt(d.shape[0]), 2)})
    G["share_pct"] = np.round(100 * G.gain / G.gain.sum(), 1)
    G.to_csv(OUT / "phase_gain_per_construct.csv", index=False)
    print(G.to_string(index=False))
    sel = G.bp.isin([6, 8, 10])
    print(f"\n  total gain {G.gain.sum():+.2f} units; "
          f"6, 8 and 10 bp carry {100*G.loc[sel,'gain'].sum()/G.gain.sum():.0f} "
          f"per cent of it")

print(f"\ndone in {time.time()-T0:.0f}s. Files written under {OUT}.")
