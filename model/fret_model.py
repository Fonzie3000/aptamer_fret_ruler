#!/usr/bin/env python3
"""
fret_model.py
=============

Orientation-dependent FRET modelling framework for the Mango II helical ruler.

This module provides:
1) Geometry loading from ruler_all_data.py
2) Monte Carlo simulation of orientation-dependent FRET with wobble + hinge
3) Nested model definitions for structural hypothesis testing
4) Parameter fitting with reproducible multi-start optimisation
5) Diagnostics, model comparison and leave-one-out predictive checks

Run as a script for a model-comparison report:
    python model/fret_model.py
The script uses the module's default pivot, base pair 1 of 6C63. The reported
fit sets the G-quadruplex pivot instead; see evaluate_fit.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import differential_evolution, minimize


# ---------------------------------------------------------------------------
# Defaults and experimental data
# ---------------------------------------------------------------------------

EXP_BP = np.array([6, 8, 10, 12, 14, 16, 18], dtype=int)
EXP_E = np.array([0.732, 0.814, 0.716, 0.542, 0.452, 0.585, 0.484], dtype=float)
EXP_SD = np.array([0.013, 0.006, 0.007, 0.014, 0.035, 0.023, 0.014], dtype=float)

# Default inputs, resolved from this file so they work from any directory.
_REPO = Path(__file__).resolve().parent.parent
DEFAULT_GEOMETRY_PATH = str(_REPO / "geometry" / "reference" / "ruler_all_data.py")
DEFAULT_6C63_PAIRFRAMES = str(_REPO / "data" / "6c63_dssr-pairFrames.txt")

DEFAULT_BOUNDS = {
    "tilt_deg": (-60.0, 60.0),
    "hinge_deg": (0.0, 60.0),
    "R0_iso": (45.0, 56.0),
    "dtwist_per_bp": (-4.0, 4.0),
    "phi0_deg": (-180.0, 180.0),
    "active_fraction": (0.5, 1.0),
    "active_intercept": (0.5, 1.0),
    "active_slope": (-0.03, 0.03),
}

# Kept for backward compatibility. Prefer cfg.bounds, which is per-run and does
# not leak between fits.
BOUND_MAP = dict(DEFAULT_BOUNDS)


@dataclass
class ModelConfig:
    # data
    exp_bp: np.ndarray = field(default_factory=lambda: EXP_BP.copy())
    exp_E: np.ndarray = field(default_factory=lambda: EXP_E.copy())
    exp_sigma: Optional[np.ndarray] = field(default_factory=lambda: EXP_SD.copy())

    # motion model
    donor_wobble_deg: float = 15.0
    acceptor_wobble_deg: float = 40.0
    averaging: str = "mixed"  # static, dynamic, mixed
    n_hinge_mixed: int = 120

    # Monte Carlo and optimiser
    n_samples_objective: int = 12000
    n_samples_report: int = 50000
    objective_seeds: Tuple[int, ...] = (101, 202, 303)
    report_seeds: Tuple[int, ...] = (11, 29, 47)
    use_global_search: bool = True
    de_maxiter: int = 90
    de_popsize: int = 12
    local_maxiter: int = 700
    n_local_starts: int = 4
    verbose: bool = True

    # geometry controls
    pivot_point: Optional[np.ndarray] = None
    helical_axis: Optional[np.ndarray] = None
    tilt_axis: Optional[np.ndarray] = None

    # penalty
    out_of_range_penalty: float = 1e8

    # per-run bounds. Editing the module-level BOUND_MAP changes every later
    # fit in the session, which is a silent way to get irreproducible results.
    bounds: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_BOUNDS))

    # pivot_mode records WHERE the stem is anchored, because this is a
    # structural claim and not a technical default. See prepare_axes.
    pivot_mode: str = "extension_start"


@dataclass
class ModelSpec:
    name: str
    free_params: Tuple[str, ...]
    scaling_mode: str  # "none", "global_active_fraction", "linear_active_fraction"


@dataclass
class FitResult:
    spec: ModelSpec
    params: Dict[str, float]
    loss: float
    diagnostics: Dict[str, float]
    e_model: np.ndarray
    residuals: np.ndarray
    kappa2: np.ndarray
    metadata: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Geometry loading and setup
# ---------------------------------------------------------------------------

def _normalise(v: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return v / n


def load_geometry(path: str = DEFAULT_GEOMETRY_PATH) -> List[Dict]:
    """Load ALL_DATA from ruler_all_data.py."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"geometry file not found: {path}")

    spec = importlib.util.spec_from_file_location("ruler_all_data_mod", str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    if not hasattr(mod, "ALL_DATA"):
        raise AttributeError(f"{path} has no ALL_DATA variable")

    out = []
    for row in mod.ALL_DATA:
        d = {
            "bp": int(row["bp"]),
            "R_D": np.asarray(row["R_D"], dtype=float),
            "mu_D": _normalise(np.asarray(row["mu_D"], dtype=float)),
            "R_A": np.asarray(row["R_A"], dtype=float),
            "mu_A": _normalise(np.asarray(row["mu_A"], dtype=float)),
        }
        out.append(d)
    out.sort(key=lambda x: x["bp"])
    return out


def parse_pair_frames(path: str) -> Dict[int, Dict[str, np.ndarray]]:
    """Parse DSSR pairFrames text."""
    lines = Path(path).read_text().strip().splitlines()
    frames: Dict[int, Dict[str, np.ndarray]] = {}
    i = 1
    while i + 4 < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        toks = lines[i].split()
        if not toks[0].isdigit():
            i += 1
            continue
        idx = int(toks[0])
        o = np.array([float(v) for v in lines[i + 1].split()[:3]], dtype=float)
        x = _normalise(np.array([float(v) for v in lines[i + 2].split()[:3]], dtype=float))
        y = _normalise(np.array([float(v) for v in lines[i + 3].split()[:3]], dtype=float))
        z = _normalise(np.array([float(v) for v in lines[i + 4].split()[:3]], dtype=float))
        frames[idx] = {"o": o, "x": x, "y": y, "z": z}
        i += 5
    return frames


def estimate_helical_axis(data: Sequence[Dict]) -> np.ndarray:
    """Estimate helical axis from PCA of acceptor positions."""
    P = np.array([d["R_A"] for d in data], dtype=float)
    C = P - P.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(C, full_matrices=False)
    axis = _normalise(vt[0])
    # orient axis from short to long bp
    if np.dot(P[-1] - P[0], axis) < 0:
        axis = -axis
    return axis


def default_pivot_from_6c63(path: str = DEFAULT_6C63_PAIRFRAMES) -> Optional[np.ndarray]:
    """Use pair 1 origin of 6C63 stem as default pivot."""
    p = Path(path)
    if not p.exists():
        return None
    frames = parse_pair_frames(str(p))
    if 1 not in frames:
        return None
    return frames[1]["o"].copy()


def prepare_axes(
    data: Sequence[Dict],
    cfg: ModelConfig,
    pairframe_6c63_path: str = DEFAULT_6C63_PAIRFRAMES,
) -> ModelConfig:
    """Fill missing pivot/helical/tilt axes in cfg."""
    cfg2 = ModelConfig(**{k: getattr(cfg, k) for k in cfg.__dataclass_fields__.keys()})

    # WHICH END IS ANCHORED IS A SCIENTIFIC CHOICE, NOT A DEFAULT.
    #
    #   "extension_start" (used here): pair 1 of 6C63, the open end where the
    #       added stem attaches. The folded aptamer, including its own stem,
    #       is treated as rigid, and only the added extension bends. The lever
    #       arm from pivot to Cy5 then grows with construct length, so a short
    #       stem barely moves and a long one swings widely.
    #
    #   "g_quadruplex" (used for the reported fit): pair 6, next to the G4. The
    #       whole stem including the native part swings as one rod, so even
    #       the 6 bp acceptor sits on a 19 A arm.
    #
    # The two pivots are about 16 A apart and give materially different fits.
    # Whichever is used has to be stated in the Methods.
    if cfg2.pivot_point is None:
        p = default_pivot_from_6c63(pairframe_6c63_path)
        if p is None:
            # fallback to closest acceptor to donor, typically shortest construct
            p = np.array(sorted(data, key=lambda d: d["bp"])[0]["R_A"], dtype=float)
        cfg2.pivot_point = p

    if cfg2.helical_axis is None:
        cfg2.helical_axis = estimate_helical_axis(data)
    cfg2.helical_axis = _normalise(np.asarray(cfg2.helical_axis, dtype=float))

    if cfg2.tilt_axis is None:
        donor = np.asarray(data[0]["R_D"], dtype=float)
        v = donor - np.asarray(cfg2.pivot_point, dtype=float)
        v = v - np.dot(v, cfg2.helical_axis) * cfg2.helical_axis
        if np.linalg.norm(v) < 1e-12:
            # arbitrary orthogonal axis
            trial = np.array([1.0, 0.0, 0.0], dtype=float)
            if abs(np.dot(trial, cfg2.helical_axis)) > 0.8:
                trial = np.array([0.0, 1.0, 0.0], dtype=float)
            v = trial - np.dot(trial, cfg2.helical_axis) * cfg2.helical_axis
        cfg2.tilt_axis = _normalise(v)
    else:
        t = np.asarray(cfg2.tilt_axis, dtype=float)
        t = t - np.dot(t, cfg2.helical_axis) * cfg2.helical_axis
        cfg2.tilt_axis = _normalise(t)

    return cfg2


# ---------------------------------------------------------------------------
# Core geometry / simulation
# ---------------------------------------------------------------------------

def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=float)


def rotation_matrix(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = _normalise(np.asarray(axis, dtype=float))
    th = np.deg2rad(float(angle_deg))
    K = _skew(axis)
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def rotate_about_line(points: np.ndarray, pivot: np.ndarray, axis: np.ndarray, angle_deg: float) -> np.ndarray:
    R = rotation_matrix(axis, angle_deg)
    shifted = np.asarray(points, dtype=float) - np.asarray(pivot, dtype=float)
    if shifted.ndim == 1:
        return R @ shifted + pivot
    return shifted @ R.T + pivot


def cone_samples(base_vec: np.ndarray, angle_deg: float, n: int, rng: np.random.Generator) -> np.ndarray:
    base = _normalise(np.asarray(base_vec, dtype=float))
    a = np.deg2rad(float(angle_deg))
    z = rng.uniform(np.cos(a), 1.0, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    r = np.sqrt(np.maximum(1.0 - z**2, 0.0))
    local = np.column_stack((r * np.cos(phi), r * np.sin(phi), z))
    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    v = np.cross(z_axis, base)
    s = np.linalg.norm(v)
    c = float(np.dot(z_axis, base))
    if s < 1e-12:
        return local if c > 0 else -local
    vx = _skew(v)
    rot = np.eye(3) + vx + (vx @ vx) * ((1.0 - c) / (s**2))
    return local @ rot.T


def kappa2_of(mu_D: np.ndarray, mu_A: np.ndarray, R_hat: np.ndarray) -> np.ndarray:
    return (np.sum(mu_D * mu_A, axis=-1) - 3.0 * np.sum(mu_D * R_hat, axis=-1) * np.sum(mu_A * R_hat, axis=-1)) ** 2


def efficiency_of(R_dist: np.ndarray, k2: np.ndarray, R0_iso: float) -> np.ndarray:
    k2c = np.maximum(k2, 1e-12)
    R0 = float(R0_iso) * (k2c / (2.0 / 3.0)) ** (1.0 / 6.0)
    return 1.0 / (1.0 + (R_dist / np.maximum(R0, 1e-12)) ** 6)


def place_acceptor(data_row: Dict, geom_params: Dict[str, float], cfg: ModelConfig, bp_ref: int) -> Tuple[np.ndarray, np.ndarray]:
    # float(), not int(). The seven measured constructs have integer bp, so this
    # leaves every fitted result bit-identical. But int() truncates fractional
    # bp, which made the accumulated twist advance as a staircase on a fine
    # grid: the display curve stepped by dtwist_per_bp at every integer and
    # looked like a polyline. The model is unchanged; only the interpolation
    # between constructs is now continuous.
    twist_total = (geom_params["phi0_deg"]
                   + geom_params["dtwist_per_bp"] * (float(data_row["bp"]) - float(bp_ref)))
    R_A = rotate_about_line(data_row["R_A"], cfg.pivot_point, cfg.helical_axis, twist_total)
    mu_A = rotation_matrix(cfg.helical_axis, twist_total) @ data_row["mu_A"]

    R_A = rotate_about_line(R_A, cfg.pivot_point, cfg.tilt_axis, geom_params["tilt_deg"])
    mu_A = rotation_matrix(cfg.tilt_axis, geom_params["tilt_deg"]) @ mu_A
    mu_A = _normalise(mu_A)
    return R_A, mu_A


def run_simulation_for_construct(
    data_row: Dict,
    geom_params: Dict[str, float],
    cfg: ModelConfig,
    rng: np.random.Generator,
    n_samples: int,
    bp_ref: int,
) -> Tuple[float, float]:
    """Return (E_true, <kappa2>) for one construct."""
    R_D = np.asarray(data_row["R_D"], dtype=float)
    mu_D_base = _normalise(np.asarray(data_row["mu_D"], dtype=float))
    R_A_base, mu_A_base = place_acceptor(data_row, geom_params, cfg, bp_ref)

    stem_vec = R_A_base - cfg.pivot_point
    stem_len = float(np.linalg.norm(stem_vec))
    if stem_len < 1e-12:
        return float("nan"), float("nan")
    stem_dir = stem_vec / stem_len

    hinge_deg = geom_params["hinge_deg"]

    def hinge_dirs(n: int) -> np.ndarray:
        if hinge_deg < 0.1:
            return np.tile(stem_dir, (n, 1))
        return cone_samples(stem_dir, hinge_deg, n, rng)

    def carry_dipole(mu_local: np.ndarray, bent: np.ndarray) -> np.ndarray:
        # rotate mu_local with minimal rotation that maps stem_dir -> bent
        cross = np.cross(stem_dir, bent)
        sin_t = np.linalg.norm(cross, axis=-1, keepdims=True)
        cos_t = np.sum(bent * stem_dir, axis=-1, keepdims=True)
        k = np.where(sin_t > 1e-12, cross / np.maximum(sin_t, 1e-12), 0.0)
        return (
            mu_local * cos_t
            + np.cross(k, mu_local) * sin_t
            + k * np.sum(k * mu_local, axis=-1, keepdims=True) * (1.0 - cos_t)
        )

    if cfg.averaging in ("static", "dynamic"):
        n = int(n_samples)
        mu_D = cone_samples(mu_D_base, cfg.donor_wobble_deg, n, rng)
        mu_A_local = cone_samples(mu_A_base, cfg.acceptor_wobble_deg, n, rng)
        bent = hinge_dirs(n)
        R_A = cfg.pivot_point + bent * stem_len
        mu_A = carry_dipole(mu_A_local, bent)

        R_vec = R_A - R_D
        R_dist = np.linalg.norm(R_vec, axis=1)
        R_hat = R_vec / np.maximum(R_dist[:, None], 1e-12)
        k2 = kappa2_of(mu_D, mu_A, R_hat)

        if cfg.averaging == "static":
            return float(np.mean(efficiency_of(R_dist, k2, geom_params["R0_iso"]))), float(np.mean(k2))

        return float(efficiency_of(np.array([np.mean(R_dist)]), np.array([np.mean(k2)]), geom_params["R0_iso"])[0]), float(np.mean(k2))

    if cfg.averaging != "mixed":
        raise ValueError("cfg.averaging must be one of: static, dynamic, mixed")

    n_h = int(cfg.n_hinge_mixed)
    n_d = max(int(n_samples // max(n_h, 1)), 20)
    bent = hinge_dirs(n_h)
    R_A = cfg.pivot_point + bent * stem_len
    R_vec = R_A - R_D
    R_dist = np.linalg.norm(R_vec, axis=1)
    R_hat = R_vec / np.maximum(R_dist[:, None], 1e-12)

    mu_D = cone_samples(mu_D_base, cfg.donor_wobble_deg, n_d, rng)
    mu_A_local = cone_samples(mu_A_base, cfg.acceptor_wobble_deg, n_d, rng)
    mu_A = carry_dipole(mu_A_local[None, :, :], bent[:, None, :])
    mu_D_b = np.broadcast_to(mu_D[None, :, :], mu_A.shape)
    R_hat_b = R_hat[:, None, :]

    k2 = kappa2_of(mu_D_b, mu_A, R_hat_b)
    k2_fast = np.mean(k2, axis=1)
    E_per_hinge = efficiency_of(R_dist, k2_fast, geom_params["R0_iso"])
    return float(np.mean(E_per_hinge)), float(np.mean(k2_fast))


# ---------------------------------------------------------------------------
# Model scaling and prediction
# ---------------------------------------------------------------------------

def _extract_geom_params(params: Dict[str, float]) -> Dict[str, float]:
    needed = ("tilt_deg", "hinge_deg", "R0_iso", "dtwist_per_bp", "phi0_deg")
    missing = [k for k in needed if k not in params]
    if missing:
        raise KeyError(f"missing geometric params: {missing}")
    return {k: float(params[k]) for k in needed}


def _active_scale(bp: np.ndarray, params: Dict[str, float], mode: str) -> np.ndarray:
    if mode == "none":
        return np.ones_like(bp, dtype=float)

    if mode == "global_active_fraction":
        f = float(params["active_fraction"])
        return np.full_like(bp, fill_value=f, dtype=float)

    if mode == "linear_active_fraction":
        f0 = float(params["active_intercept"])
        f1 = float(params["active_slope"])
        bp0 = float(np.mean(bp))
        f = f0 + f1 * (bp - bp0)
        return f

    raise ValueError(f"unknown scaling mode: {mode}")


def predict(
    data: Sequence[Dict],
    params: Dict[str, float],
    cfg: ModelConfig,
    spec: ModelSpec,
    n_samples: int,
    seeds: Sequence[int],
    subset_bp: Optional[Iterable[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict measured E for selected constructs.
    Returns (bp, E_pred, k2_mean).
    """
    bp_ref = int(min(d["bp"] for d in data))
    bp_set = set(int(b) for b in subset_bp) if subset_bp is not None else None
    selected = [d for d in data if (bp_set is None or int(d["bp"]) in bp_set)]
    selected.sort(key=lambda x: int(x["bp"]))
    bp = np.array([int(d["bp"]) for d in selected], dtype=int)

    geom = _extract_geom_params(params)
    e_all = []
    k_all = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        e_seed = []
        k_seed = []
        for row in selected:
            e_true, k2 = run_simulation_for_construct(row, geom, cfg, rng, n_samples, bp_ref)
            e_seed.append(e_true)
            k_seed.append(k2)
        e_all.append(np.array(e_seed, dtype=float))
        k_all.append(np.array(k_seed, dtype=float))

    e_true_mean = np.mean(np.vstack(e_all), axis=0)
    k2_mean = np.mean(np.vstack(k_all), axis=0)

    scale = _active_scale(bp.astype(float), params, spec.scaling_mode)
    e_meas = e_true_mean * scale
    return bp, e_meas, k2_mean


# ---------------------------------------------------------------------------
# Objective and fitting
# ---------------------------------------------------------------------------

def _within_bounds(params: Dict[str, float], free_params: Sequence[str],
                   bounds: Optional[Dict[str, Tuple[float, float]]] = None) -> bool:
    bounds = DEFAULT_BOUNDS if bounds is None else bounds
    for name in free_params:
        lo, hi = bounds[name]
        v = float(params[name])
        if v < lo or v > hi:
            return False
    return True


def _vector_to_params(theta: np.ndarray, free_params: Sequence[str], fixed_params: Dict[str, float]) -> Dict[str, float]:
    out = dict(fixed_params)
    for i, name in enumerate(free_params):
        out[name] = float(theta[i])
    return out


def _theta_bounds(free_params: Sequence[str],
                  bounds: Optional[Dict[str, Tuple[float, float]]] = None
                  ) -> List[Tuple[float, float]]:
    bounds = DEFAULT_BOUNDS if bounds is None else bounds
    return [bounds[p] for p in free_params]


def _loss_from_resid(resid: np.ndarray, sigma: Optional[np.ndarray]) -> float:
    if sigma is None:
        return float(np.sum(resid**2))
    return float(np.sum((resid / np.maximum(sigma, 1e-12)) ** 2))


def objective(
    theta: np.ndarray,
    data: Sequence[Dict],
    cfg: ModelConfig,
    spec: ModelSpec,
    free_params: Sequence[str],
    fixed_params: Dict[str, float],
    fit_bp: np.ndarray,
    fit_E: np.ndarray,
    fit_SD: Optional[np.ndarray],
) -> float:
    params = _vector_to_params(theta, free_params, fixed_params)
    if not _within_bounds(params, free_params, cfg.bounds):
        return cfg.out_of_range_penalty

    # explicit constraints for linear active fraction, to avoid invalid >1/<0
    if spec.scaling_mode == "linear_active_fraction":
        bpf = fit_bp.astype(float)
        f = _active_scale(bpf, params, spec.scaling_mode)
        if np.any(f < 0.0) or np.any(f > 1.0):
            return cfg.out_of_range_penalty

    losses = []
    for s in cfg.objective_seeds:
        bp, e_pred, _ = predict(
            data=data,
            params=params,
            cfg=cfg,
            spec=spec,
            n_samples=cfg.n_samples_objective,
            seeds=(int(s),),
            subset_bp=fit_bp,
        )
        if bp.shape[0] != fit_bp.shape[0] or np.any(bp != fit_bp):
            return cfg.out_of_range_penalty
        if np.any(~np.isfinite(e_pred)):
            return cfg.out_of_range_penalty
        resid = e_pred - fit_E
        losses.append(_loss_from_resid(resid, fit_SD))
    return float(np.mean(losses))


def fit_model(
    data: Sequence[Dict],
    cfg: ModelConfig,
    spec: ModelSpec,
    fixed_params: Optional[Dict[str, float]] = None,
    fit_bp: Optional[np.ndarray] = None,
    fit_E: Optional[np.ndarray] = None,
    fit_SD: Optional[np.ndarray] = None,
    warm_start: Optional[Dict[str, float]] = None,
    extra_starts: Optional[Sequence[Dict[str, float]]] = None,
) -> FitResult:
    """
    Fit one model specification.
    Any parameters absent from free_params must be provided in fixed_params.

    warm_start / extra_starts
        Parameter dictionaries used as additional starting points for the local
        optimiser. Supplying a reference vector found by one expensive global
        search makes every later scan comparable: without it, a scan measures
        whether the optimiser happened to find the good basin on that run
        rather than the effect of the quantity being scanned. Values are
        clipped into the current bounds, and keys not in free_params are
        ignored, so one reference vector can seed fits with different free
        parameter sets.
    """
    fixed_params = dict(fixed_params or {})
    free_params = tuple(spec.free_params)

    # defaults for all geometry params if fixed
    defaults = {
        "tilt_deg": 0.0,
        "hinge_deg": 30.0,
        "R0_iso": 52.0,
        "dtwist_per_bp": 0.0,
        "phi0_deg": 0.0,
        "active_fraction": 1.0,
        "active_intercept": 0.8,
        "active_slope": 0.0,
    }
    for k, v in defaults.items():
        if k not in fixed_params and k not in free_params:
            fixed_params[k] = v

    if fit_bp is None:
        fit_bp = cfg.exp_bp.copy()
    if fit_E is None:
        fit_E = cfg.exp_E.copy()
    if fit_SD is None:
        fit_SD = None if cfg.exp_sigma is None else cfg.exp_sigma.copy()

    bounds = _theta_bounds(free_params, cfg.bounds)

    def obj_fn(x: np.ndarray) -> float:
        return objective(
            theta=np.asarray(x, dtype=float),
            data=data,
            cfg=cfg,
            spec=spec,
            free_params=free_params,
            fixed_params=fixed_params,
            fit_bp=fit_bp,
            fit_E=fit_E,
            fit_SD=fit_SD,
        )

    if cfg.verbose:
        print("=" * 78)
        print(f"Fit model: {spec.name}")
        print(f"  free parameters : {list(free_params)}")
        print(f"  fixed parameters: {fixed_params}")
        print(f"  fitting points  : {fit_bp.tolist()}")
        print("=" * 78)

    starts: List[np.ndarray] = []

    if cfg.use_global_search and len(free_params) > 0:
        de = differential_evolution(
            obj_fn,
            bounds=bounds,
            maxiter=cfg.de_maxiter,
            popsize=cfg.de_popsize,
            seed=42,
            polish=False,
            updating="deferred",
            workers=1,
        )
        starts.append(np.asarray(de.x, dtype=float))

    # warm starts first: these are the reference vectors, and sorting by loss
    # below means a random start can only ever replace one by being better.
    for cand in ([warm_start] if warm_start else []) + list(extra_starts or []):
        if not cand:
            continue
        vec = []
        for name, (lo, hi) in zip(free_params, bounds):
            v = float(cand.get(name, np.clip(0.5 * (lo + hi), lo, hi)))
            vec.append(float(np.clip(v, lo, hi)))
        starts.append(np.asarray(vec, dtype=float))

    # deterministic extra starts
    rng = np.random.default_rng(1234)
    while len(starts) < max(cfg.n_local_starts, 1):
        s = np.array([rng.uniform(lo, hi) for (lo, hi) in bounds], dtype=float) if bounds else np.array([], dtype=float)
        starts.append(s)

    results = []
    if len(free_params) == 0:
        theta_best = np.array([], dtype=float)
        best_loss = obj_fn(theta_best)
    else:
        for s in starts:
            res = minimize(
                obj_fn,
                x0=s,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": cfg.local_maxiter},
            )
            results.append(res)
        results.sort(key=lambda r: r.fun)
        theta_best = np.asarray(results[0].x, dtype=float)
        best_loss = float(results[0].fun)
        losses = np.array([r.fun for r in results], dtype=float)
        # fraction of starts landing within 1% of the best loss. A low value
        # means the surface is multimodal and a single run is a coin toss.
        _basin_frac = float(np.mean(losses <= best_loss * 1.01)) if best_loss > 0 else 1.0
        _loss_spread = float(losses.max() - losses.min())

    best_params = _vector_to_params(theta_best, free_params, fixed_params)

    # report predictions with high sampling
    bp, e_model, k2 = predict(
        data=data,
        params=best_params,
        cfg=cfg,
        spec=spec,
        n_samples=cfg.n_samples_report,
        seeds=cfg.report_seeds,
        subset_bp=fit_bp,
    )

    resid = e_model - fit_E
    diag = diagnostics_from_prediction(
        y_true=fit_E,
        y_pred=e_model,
        sigma=fit_SD,
        n_params=len(free_params),
    )

    metadata = {}
    if len(free_params) > 0:
        metadata["n_starts"] = float(len(starts))
        metadata["basin_fraction"] = _basin_frac
        metadata["loss_spread"] = _loss_spread
    for p in free_params:
        lo, hi = cfg.bounds[p]
        span = hi - lo
        d = min((best_params[p] - lo) / span, (hi - best_params[p]) / span) if span > 0 else np.inf
        metadata[f"{p}_bound_margin"] = float(d)

    return FitResult(
        spec=spec,
        params=best_params,
        loss=float(best_loss),
        diagnostics=diag,
        e_model=e_model,
        residuals=resid,
        kappa2=k2,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Diagnostics, model selection, predictive checks
# ---------------------------------------------------------------------------

def diagnostics_from_prediction(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sigma: Optional[np.ndarray],
    n_params: int,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    resid = y_pred - y_true
    sse = float(np.sum(resid**2))
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae = float(np.mean(np.abs(resid)))
    bias = float(np.mean(resid))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 0 else float("nan")

    n = int(y_true.size)
    dof = max(n - n_params, 1)

    out = {
        "n": float(n),
        "k": float(n_params),
        "SSE": sse,
        "RMSE": rmse,
        "MAE": mae,
        "Bias": bias,
        "R2": r2,
    }

    if sigma is not None:
        sigma = np.asarray(sigma, dtype=float)
        chi2 = float(np.sum((resid / np.maximum(sigma, 1e-12)) ** 2))
        out["chi2"] = chi2
        out["chi2_reduced"] = chi2 / dof

        ll = -0.5 * np.sum((resid / sigma) ** 2 + np.log(2.0 * np.pi * sigma**2))
    else:
        var = max(sse / max(n, 1), 1e-12)
        ll = -0.5 * n * (np.log(2.0 * np.pi * var) + 1.0)

    aic = float(2.0 * n_params - 2.0 * ll)
    bic = float(np.log(max(n, 1)) * n_params - 2.0 * ll)
    # AICc needs n - k - 1 > 0. With seven constructs any model carrying six or
    # more free parameters returns inf, so AICc cannot rank the richer models
    # here. Rank on AIC or BIC in that case, and say so.
    if n - n_params - 1 > 0:
        aicc = float(aic + (2.0 * n_params * (n_params + 1)) / (n - n_params - 1))
    else:
        aicc = float("inf")
    out["saturated"] = float(n_params >= n - 1)

    out["log_likelihood"] = float(ll)
    out["AIC"] = aic
    out["AICc"] = aicc
    out["BIC"] = bic
    return out


def bound_report(result: FitResult, threshold: float = 0.02) -> List[str]:
    notes = []
    for p in result.spec.free_params:
        m = result.metadata.get(f"{p}_bound_margin", np.nan)
        if np.isfinite(m) and m < threshold:
            notes.append(f"{p} near bound (margin fraction {m:.4f})")
    return notes


def loo_cross_validation(
    data: Sequence[Dict],
    cfg: ModelConfig,
    spec: ModelSpec,
    fixed_params: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    """Leave-one-out predictive check."""
    bp_full = cfg.exp_bp.copy()
    E_full = cfg.exp_E.copy()
    SD_full = None if cfg.exp_sigma is None else cfg.exp_sigma.copy()

    # With one point held out there are n-1 points left. A model with as many
    # free parameters as that fits them exactly, so its held-out prediction is
    # an extrapolation from a saturated fit and the LOO error is not
    # interpretable. Refuse rather than return a misleading number.
    n_free = len(spec.free_params)
    if n_free >= bp_full.size - 1:
        raise ValueError(
            f"leave-one-out is not meaningful for {spec.name}: {n_free} free "
            f"parameters against {bp_full.size - 1} retained points. Fix some "
            f"parameters first, or use a smaller model.")

    pred = []
    true = []
    held = []
    for i in range(bp_full.size):
        mask = np.ones(bp_full.size, dtype=bool)
        mask[i] = False
        fit_bp = bp_full[mask]
        fit_E = E_full[mask]
        fit_SD = None if SD_full is None else SD_full[mask]
        held_bp = int(bp_full[i])

        local_cfg = ModelConfig(**{k: getattr(cfg, k) for k in cfg.__dataclass_fields__.keys()})
        # speed up LOO
        local_cfg.n_samples_objective = max(3500, cfg.n_samples_objective // 3)
        local_cfg.n_samples_report = max(8000, cfg.n_samples_report // 4)
        local_cfg.objective_seeds = cfg.objective_seeds[:2]
        local_cfg.report_seeds = cfg.report_seeds[:2]
        local_cfg.de_maxiter = max(40, cfg.de_maxiter // 2)
        local_cfg.local_maxiter = max(300, cfg.local_maxiter // 2)
        local_cfg.verbose = False

        fit_res = fit_model(
            data=data,
            cfg=local_cfg,
            spec=spec,
            fixed_params=fixed_params,
            fit_bp=fit_bp,
            fit_E=fit_E,
            fit_SD=fit_SD,
        )
        bp_pred, e_pred, _ = predict(
            data=data,
            params=fit_res.params,
            cfg=local_cfg,
            spec=spec,
            n_samples=local_cfg.n_samples_report,
            seeds=local_cfg.report_seeds,
            subset_bp=[held_bp],
        )
        pred.append(float(e_pred[0]))
        true.append(float(E_full[i]))
        held.append(held_bp)

    held_arr = np.array(held, dtype=int)
    pred_arr = np.array(pred, dtype=float)
    true_arr = np.array(true, dtype=float)
    resid = pred_arr - true_arr
    out = {
        "bp": held_arr,
        "pred": pred_arr,
        "true": true_arr,
        "resid": resid,
        "RMSE_LOO": np.array([float(np.sqrt(np.mean(resid**2)))], dtype=float),
        "MAE_LOO": np.array([float(np.mean(np.abs(resid)))], dtype=float),
    }
    return out


def make_default_specs() -> List[ModelSpec]:
    return [
        ModelSpec(
            name="M0_orientation_only",
            free_params=("tilt_deg", "hinge_deg", "R0_iso", "dtwist_per_bp", "phi0_deg"),
            scaling_mode="none",
        ),
        ModelSpec(
            name="M1_global_active_fraction",
            free_params=("tilt_deg", "hinge_deg", "R0_iso", "dtwist_per_bp", "phi0_deg", "active_fraction"),
            scaling_mode="global_active_fraction",
        ),
        ModelSpec(
            name="M2_linear_active_fraction",
            free_params=("tilt_deg", "hinge_deg", "R0_iso", "dtwist_per_bp", "phi0_deg", "active_intercept", "active_slope"),
            scaling_mode="linear_active_fraction",
        ),
    ]


def compare_models(
    data: Sequence[Dict],
    cfg: ModelConfig,
    specs: Sequence[ModelSpec],
    do_loo: bool = True,
) -> Tuple[List[FitResult], Dict[str, Dict[str, np.ndarray]]]:
    fits: List[FitResult] = []
    loo: Dict[str, Dict[str, np.ndarray]] = {}

    for spec in specs:
        r = fit_model(data=data, cfg=cfg, spec=spec)
        fits.append(r)
        if do_loo:
            loo[spec.name] = loo_cross_validation(data=data, cfg=cfg, spec=spec)

    return fits, loo


def print_fit_summary(fit: FitResult, cfg: ModelConfig) -> None:
    print(f"\nModel: {fit.spec.name}")
    print("-" * 78)
    for k in fit.spec.free_params:
        print(f"  {k:<18} {fit.params[k]:>+10.4f}")
    for b in bound_report(fit):
        print(f"  [warning] {b}")

    d = fit.diagnostics
    keys = ["R2", "RMSE", "MAE", "Bias", "chi2_reduced", "AICc", "BIC"]
    for k in keys:
        if k in d:
            print(f"  {k:<18} {d[k]:>+10.4f}")

    print("  residuals (E_model - E_exp):")
    for bp, e_m, e_t, r in zip(cfg.exp_bp, fit.e_model, cfg.exp_E, fit.residuals):
        print(f"    {int(bp):>2d} bp   model {e_m:.4f}   exp {e_t:.3f}   resid {r:+.4f}")


def print_model_ranking(fits: Sequence[FitResult], loo: Dict[str, Dict[str, np.ndarray]]) -> None:
    print("\n" + "=" * 78)
    print("Model ranking summary")
    print("=" * 78)
    header = f"{'model':<32} {'R2':>8} {'RMSE':>9} {'chi2_red':>10} {'AICc':>10} {'RMSE_LOO':>10}"
    print(header)
    print("-" * len(header))
    ranked = sorted(fits, key=lambda fr: fr.diagnostics.get("AICc", np.inf))
    for fr in ranked:
        name = fr.spec.name
        d = fr.diagnostics
        rmse_loo = float(loo[name]["RMSE_LOO"][0]) if name in loo else float("nan")
        print(
            f"{name:<32} "
            f"{d.get('R2', np.nan):>8.3f} "
            f"{d.get('RMSE', np.nan):>9.4f} "
            f"{d.get('chi2_reduced', np.nan):>10.2f} "
            f"{d.get('AICc', np.nan):>10.2f} "
            f"{rmse_loo:>10.4f}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Orientation-dependent FRET model comparison framework")
    parser.add_argument("--geometry", default=DEFAULT_GEOMETRY_PATH, help="Path to ruler_all_data.py")
    parser.add_argument("--pairframes6c63", default=DEFAULT_6C63_PAIRFRAMES, help="Path to 6c63_dssr-pairFrames.txt")
    parser.add_argument("--no-loo", action="store_true", help="Skip leave-one-out predictive checks")
    parser.add_argument("--quick", action="store_true", help="Run a faster low-precision pass")
    args = parser.parse_args()

    data = load_geometry(args.geometry)
    cfg = ModelConfig(verbose=True)
    cfg = prepare_axes(data, cfg, pairframe_6c63_path=args.pairframes6c63)

    if args.quick:
        cfg.n_samples_objective = 3500
        cfg.n_samples_report = 12000
        cfg.objective_seeds = (101, 202)
        cfg.report_seeds = (11, 29)
        cfg.de_maxiter = 45
        cfg.local_maxiter = 320
        cfg.n_local_starts = 3

    print("Loaded constructs:", [d["bp"] for d in data])
    print("Pivot point:", np.round(cfg.pivot_point, 3))
    print("Helical axis:", np.round(cfg.helical_axis, 4))
    print("Tilt axis:", np.round(cfg.tilt_axis, 4))

    specs = make_default_specs()
    fits, loo = compare_models(data, cfg, specs, do_loo=(not args.no_loo))

    for fr in fits:
        print_fit_summary(fr, cfg)
    print_model_ranking(fits, loo)


if __name__ == "__main__":
    main()
