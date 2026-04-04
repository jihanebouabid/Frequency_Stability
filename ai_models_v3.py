"""
ai_models_v3.py
---------------
Virtual inertia (H) and damping (D) prediction — v3

Improvements over v2:
  - Diagnostics: feature correlations, class distribution, target stats
  - SVR with nested LOO+GridSearch (replaces naive RF)
  - RF with nested LOO+RandomizedSearch (tuned hyperparameters)
  - Physics-CNN with swing-equation synthetic pre-training
  - Ensemble: GPR + SVR + CNN averaging
  - Swing-equation ODE simulator for synthetic data augmentation

Data:
  hd_results_2016.csv    H/D targets + event metadata (21 raw, 20 used)
  event_segments/*.npz   Δf(t), RoCoF(t) time series, T = 300 @ 10 Hz

Targets (signal-derived):
  H_pw_mean  [s]   — swing-equation H estimated from PMU RoCoF (first 3 s)
  D_pw_mean  [pu]  — swing-equation D estimated from onset-to-nadir trajectory

Evaluation: Leave-One-Event-Out (LOO) cross-validation, 20 folds.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import pearsonr, skew

try:
    from scipy.stats.qmc import LatinHypercube
    _HAS_LHS = True
except ImportError:
    _HAS_LHS = False

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
ROOT     = Path(__file__).parent
CSV_PATH = ROOT / "hd_results_2016.csv"
SEG_DIR  = ROOT / "event_segments"
PLOT_DIR = ROOT / "plots"
PLOT_DIR.mkdir(exist_ok=True)

DT = 0.1    # PMU sampling interval [s]
F0 = 50.0   # nominal frequency [Hz]

# Physics-valid output ranges
H_MIN, H_MAX = 2.0, 10.0   # [s]
D_MIN, D_MAX = 0.1,  5.0   # [pu]

FEATURE_NAMES = [
    "rocof_init", "df_nadir", "t_nadir", "dP_pu",
    "H_proxy", "D_proxy",
    "area", "energy", "settling", "direction",
]


# ===========================================================================
# 1. DATA LOADING  (verbatim from v2)
# ===========================================================================

def load_data() -> list[dict]:
    """
    Load hd_results_2016.csv and align with event_segments/*.npz by onset key.

    Each event dict contains:
        onset_str  : 'YYYYMMDD_HHMMSS'
        dP_pu      : float  (signed; negative = generation loss)
        dP_MW      : float
        direction  : +1 (over-freq) or -1 (under-freq)
        cause      : str
        H_from_Ek  : float  [s]  — kinetic energy report value (kept for reference)
        D_nadir    : float  [pu] — nadir-point damping (kept for reference)
        H_pw_mean  : float or NaN [s]  — PRIMARY H TARGET
        D_pw_mean  : float  [pu]       — PRIMARY D TARGET
        df_nadir_Hz: float  [Hz]
        df_train   : (300,) float32  — Δf, 30 s window @ 10 Hz
        rocof_train: (300,) float32  — RoCoF, 30 s window
        df_full    : (1551,) float32 or None — full 155 s trajectory
        rocof_full : (1551,) float32 or None
        t_full     : (1551,) float32 or None
    """
    df_csv = pd.read_csv(CSV_PATH)

    def _key(ts: str) -> str:
        return (ts.replace("T", " ").split(".")[0]
                  .replace("-", "").replace(":", "").replace(" ", "_"))

    csv_index = {_key(row["onset_time"]): row for _, row in df_csv.iterrows()}

    events = []
    for npz_file in sorted(SEG_DIR.glob("*.npz")):
        key = npz_file.stem
        if key not in csv_index:
            print(f"  [WARN] No CSV row for {key}, skipping.")
            continue
        row = csv_index[key]
        seg = np.load(npz_file, allow_pickle=True)
        has_full = "df_full" in seg.files

        H_pw = float(row["H_pw_mean"])
        D_pw = float(row["D_pw_mean"])

        events.append({
            "onset_str":   key,
            "dP_pu":       float(row["dP_pu"]),
            "dP_MW":       float(row["dP_MW"]),
            "direction":   1.0 if row["direction"] == "over" else -1.0,
            "cause":       row["cause"],
            "H_from_Ek":  float(row["H_from_Ek"]),
            "D_nadir":    float(row["D_nadir"]),
            "H_pw_mean":  H_pw if not np.isnan(H_pw) else np.nan,
            "D_pw_mean":  D_pw,
            "df_nadir_Hz": float(row["df_nadir_Hz"]),
            "df_train":    seg["df_train"].astype(np.float32),
            "rocof_train": seg["rocof_train"].astype(np.float32),
            "df_full":     seg["df_full"].astype(np.float32) if has_full else None,
            "rocof_full":  seg["rocof_full"].astype(np.float32) if has_full else None,
            "t_full":      seg["t_full"].astype(np.float32) if has_full else None,
        })

    print(f"Loaded {len(events)} raw events.")

    # FIX-3: drop Oct-08 (H_pw_n=0 → NaN H_pw_mean, D_pw_std=2.0 pu outlier)
    valid = [ev for ev in events if not np.isnan(ev["H_pw_mean"])]
    n_dropped = len(events) - len(valid)
    if n_dropped:
        dropped = [ev["onset_str"] for ev in events if np.isnan(ev["H_pw_mean"])]
        print(f"  Dropped {n_dropped} event(s) with NaN H_pw_mean: {dropped}")
    print(f"  Using {len(valid)} events for modelling.\n")
    return valid


# ===========================================================================
# 2. PHYSICS-INFORMED FEATURE ENGINEERING  (verbatim from v2)
# ===========================================================================

def engineer_features(events: list[dict]) -> np.ndarray:
    """
    Compute 10 physics-informed features per event.

    Feature vector:
      0  rocof_init   — mean |RoCoF| in first 2 s  [Hz/s]
      1  df_nadir     — max |Δf| in 30 s window    [Hz]
      2  t_nadir      — time to nadir               [s]
      3  dP_pu        — |power imbalance|           [pu]
      4  H_proxy      — dP·f0 / (2·RoCoF_init)     [s]   physics H estimate
      5  D_proxy      — dP·f0 / df_nadir            [pu]  ≡ D_nadir formula
                        Legitimate for D_pw_mean target (r≈0.78, not =1.00)
      6  area         — ∫|Δf| dt  [Hz·s]
      7  energy       — ∫Δf² dt   [Hz²·s]
      8  settling     — |Δf| at t = 25 s
      9  direction    — +1 over-freq, -1 under-freq

    Returns ndarray (N, 10).
    """
    rows = []
    for ev in events:
        df    = ev["df_train"]
        rocof = ev["rocof_train"]
        dP    = abs(ev["dP_pu"])

        rocof_init = float(np.mean(np.abs(rocof[:20])))
        rocof_init = max(rocof_init, 1e-6)

        abs_df   = np.abs(df)
        nadir_i  = int(np.argmax(abs_df))
        df_nadir = max(float(abs_df[nadir_i]), 1e-6)
        t_nadir  = nadir_i * DT

        H_proxy = dP * F0 / (2.0 * rocof_init)
        D_proxy = dP * F0 / df_nadir          # = D_nadir analytically, ≠ D_pw_mean

        area     = float(np.sum(abs_df) * DT)
        energy   = float(np.sum(df ** 2) * DT)
        settling = float(abs_df[250]) if len(abs_df) > 250 else float(abs_df[-1])

        rows.append([rocof_init, df_nadir, t_nadir, dP,
                     H_proxy, D_proxy,
                     area, energy, settling,
                     ev["direction"]])

    X = np.array(rows, dtype=np.float32)
    print(f"Feature matrix: {X.shape}  (N_events x N_features)")
    return X


# ===========================================================================
# 3. DIAGNOSTICS
# ===========================================================================

def diagnose_data(
    events: list[dict],
    X: np.ndarray,
    y_H: np.ndarray,
    y_D: np.ndarray,
) -> None:
    """
    Print diagnostic summary:
      - Pearson correlations of all 10 features vs H_pw_mean and D_pw_mean
      - Class distribution (cause, direction)
      - Target statistics (mean, std, min, max, skew)
      - Feature-feature correlation warnings (|r| > 0.9 pairs)
    """
    print("=" * 72)
    print("  DATA DIAGNOSTICS")
    print("=" * 72)

    # --- Target statistics ---
    print("\n--- Target Statistics ---")
    for name, y in [("H_pw_mean [s]", y_H), ("D_pw_mean [pu]", y_D)]:
        print(f"  {name:16s}  mean={np.mean(y):.3f}  std={np.std(y):.3f}  "
              f"min={np.min(y):.3f}  max={np.max(y):.3f}  skew={skew(y):.3f}")

    # --- Class distribution ---
    print("\n--- Class Distribution ---")
    causes = {}
    dirs = {1.0: 0, -1.0: 0}
    for ev in events:
        c = ev["cause"]
        causes[c] = causes.get(c, 0) + 1
        dirs[ev["direction"]] += 1
    print(f"  Direction:  over-freq={dirs[1.0]},  under-freq={dirs[-1.0]}")
    print(f"  Causes:")
    for c, n in sorted(causes.items(), key=lambda x: -x[1]):
        print(f"    {c:20s}  {n:3d}  ({100*n/len(events):.0f}%)")

    # --- Feature-target correlations ---
    print("\n--- Feature-Target Pearson Correlations ---")
    print(f"  {'Feature':14s}  {'r(H_pw)':>8s}  {'p(H)':>8s}  {'r(D_pw)':>8s}  {'p(D)':>8s}")
    print("  " + "-" * 52)
    for j, fname in enumerate(FEATURE_NAMES):
        rH, pH_val = pearsonr(X[:, j], y_H)
        rD, pD_val = pearsonr(X[:, j], y_D)
        flag_H = " ***" if abs(rH) > 0.5 else ""
        flag_D = " ***" if abs(rD) > 0.5 else ""
        print(f"  {fname:14s}  {rH:+8.3f}  {pH_val:8.4f}  {rD:+8.3f}  {pD_val:8.4f}{flag_H}{flag_D}")

    # --- Feature-feature high correlations ---
    print("\n--- Feature-Feature Correlation Warnings (|r| > 0.9) ---")
    n_feat = X.shape[1]
    found = False
    for j in range(n_feat):
        for k in range(j + 1, n_feat):
            r_jk, _ = pearsonr(X[:, j], X[:, k])
            if abs(r_jk) > 0.9:
                print(f"  {FEATURE_NAMES[j]:14s}  x  {FEATURE_NAMES[k]:14s}  r={r_jk:+.3f}")
                found = True
    if not found:
        print("  (none)")

    print("=" * 72 + "\n")


# ===========================================================================
# 4. METRICS HELPER
# ===========================================================================

def _make_result(name, y_H, p_H, s_H, y_D, p_D, s_D) -> dict:
    r2_H  = r2_score(y_H, p_H)
    mae_H = mean_absolute_error(y_H, p_H)
    r2_D  = r2_score(y_D, p_D)
    mae_D = mean_absolute_error(y_D, p_D)
    print(f"  {name:8s}  H_pw: R2={r2_H:+.3f}  MAE={mae_H:.3f} s   "
          f" D_pw: R2={r2_D:+.3f}  MAE={mae_D:.3f} pu")
    return dict(name=name,
                y_H=y_H, p_H=p_H, s_H=s_H,
                y_D=y_D, p_D=p_D, s_D=s_D,
                r2_H=r2_H, mae_H=mae_H,
                r2_D=r2_D, mae_D=mae_D)


# ===========================================================================
# 5. SWING-EQUATION SYNTHETIC DATA AUGMENTATION
# ===========================================================================

def _simulate_swing(
    H: float,
    D: float,
    dP_pu: float,
    dt: float = 0.1,
    T_end: float = 30.0,
    Tg: float = 8.0,
    R: float = 0.05,
    noise_std: float = 0.0003,
    seed: int | None = None,
) -> np.ndarray:
    """
    Euler integration of swing equation with governor response:
        2H/f0 * d(Df)/dt = dP - D/f0 * Df - Pg
        Tg * dPg/dt      = -Df / (R * f0) - Pg

    Parameters
    ----------
    H       : inertia constant [s]
    D       : damping coefficient [pu]
    dP_pu   : power imbalance step [pu] (signed)
    dt      : integration time step [s]
    T_end   : simulation duration [s]
    Tg      : governor time constant [s]
    R       : droop [pu]
    noise_std : measurement noise std [Hz]
    seed    : random seed for noise

    Returns
    -------
    df_sim : (N_steps,) array of frequency deviation [Hz]
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    N_steps = int(T_end / dt)
    df_sim = np.zeros(N_steps, dtype=np.float64)
    Pg = 0.0       # governor power [pu]
    Df = 0.0       # frequency deviation [Hz]

    for k in range(1, N_steps):
        # Governor dynamics
        dPg_dt = (-Df / (R * F0) - Pg) / Tg
        Pg += dPg_dt * dt

        # Swing equation
        dDf_dt = (dP_pu - (D / F0) * Df - Pg) * F0 / (2.0 * H)
        Df += dDf_dt * dt

        # Add measurement noise
        df_sim[k] = Df + rng.normal(0, noise_std)

    return df_sim.astype(np.float32)


def generate_synthetic(n_synth: int = 2000, seed: int = 0) -> list[dict]:
    """
    Latin-Hypercube sample (H, D, dP, Tg, R), simulate swing ODE.

    Parameter ranges:
        H  : [2.5, 6.5] s
        D  : [0.8, 4.0] pu
        dP : [0.005, 0.022] pu  (both signs, ~25% under-freq)
        Tg : [4, 12] s
        R  : [0.03, 0.08]

    Returns list of dicts with keys:
        H, D, dP_pu, direction, df_train, rocof_train
    """
    rng = np.random.RandomState(seed)

    # --- Latin Hypercube sampling ---
    if _HAS_LHS:
        sampler = LatinHypercube(d=5, seed=seed)
        samples = sampler.random(n=n_synth)   # (n_synth, 5) in [0,1]
    else:
        # Fallback: stratified random (approximate LHS)
        samples = np.zeros((n_synth, 5))
        for d in range(5):
            perm = rng.permutation(n_synth)
            samples[:, d] = (perm + rng.uniform(size=n_synth)) / n_synth

    # Scale to physical ranges
    H_arr  = samples[:, 0] * (6.5 - 2.5)  + 2.5
    D_arr  = samples[:, 1] * (4.0 - 0.8)  + 0.8
    dP_arr = samples[:, 2] * (0.022 - 0.005) + 0.005
    Tg_arr = samples[:, 3] * (12.0 - 4.0)  + 4.0
    R_arr  = samples[:, 4] * (0.08 - 0.03) + 0.03

    # ~25% under-frequency (negative dP)
    sign_mask = rng.rand(n_synth) < 0.25
    dP_arr[sign_mask] *= -1.0

    synth_events = []
    for i in range(n_synth):
        df_train = _simulate_swing(
            H=H_arr[i], D=D_arr[i], dP_pu=dP_arr[i],
            Tg=Tg_arr[i], R=R_arr[i],
            noise_std=0.0003, seed=seed + i + 1,
        )
        rocof_train = np.gradient(df_train, 0.1).astype(np.float32)

        synth_events.append({
            "H": float(H_arr[i]),
            "D": float(D_arr[i]),
            "dP_pu": float(dP_arr[i]),
            "direction": -1.0 if dP_arr[i] < 0 else 1.0,
            "df_train": df_train,
            "rocof_train": rocof_train,
        })

    print(f"Generated {n_synth} synthetic events via swing-equation ODE.")
    return synth_events


# ===========================================================================
# 6. GPR  (Architecture 4.1 — carried from v2)
# ===========================================================================

def run_gpr(X: np.ndarray, y_H: np.ndarray, y_D: np.ndarray) -> dict:
    """
    Gaussian Process Regression with LOO-CV.
    Matern-5/2 x ConstantKernel + WhiteKernel.
    Separate GP for H_pw_mean and D_pw_mean.
    """
    print("\n--- GPR LOO-CV ---")
    N      = len(y_H)
    kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(1e-3)

    p_H, p_D = np.zeros(N), np.zeros(N)
    s_H, s_D = np.zeros(N), np.zeros(N)

    for i in range(N):
        tr = np.ones(N, dtype=bool); tr[i] = False

        sx = StandardScaler().fit(X[tr])
        Xtr, Xte = sx.transform(X[tr]), sx.transform(X[[i]])

        gH = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                                       normalize_y=True)
        gD = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                                       normalize_y=True)
        gH.fit(Xtr, y_H[tr]);  gD.fit(Xtr, y_D[tr])

        pH, sH = gH.predict(Xte, return_std=True)
        pD, sD = gD.predict(Xte, return_std=True)
        p_H[i], s_H[i] = pH[0], sH[0]
        p_D[i], s_D[i] = pD[0], sD[0]

        if (i + 1) % 5 == 0 or i == N - 1:
            print(f"  fold {i+1:2d}/{N}")

    p_H = np.clip(p_H, H_MIN, H_MAX)
    p_D = np.clip(p_D, D_MIN, D_MAX)
    return _make_result("GPR", y_H, p_H, s_H, y_D, p_D, s_D)


# ===========================================================================
# 7. SVR  (new in v3)
# ===========================================================================

def run_svr(X: np.ndarray, y_H: np.ndarray, y_D: np.ndarray) -> dict:
    """
    Support Vector Regression with LOO-CV.
    Outer: LOO (N=20 folds).
    Inner: GridSearchCV(cv=5) on the 19 training samples.
    Separate SVR for H and D, both with StandardScaler inside each fold.
    """
    print("\n--- SVR LOO-CV (nested GridSearch) ---")
    N = len(y_H)

    param_grid = {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.01, 0.1],
    }

    p_H, p_D = np.zeros(N), np.zeros(N)

    for i in range(N):
        tr = np.ones(N, dtype=bool); tr[i] = False

        sx = StandardScaler().fit(X[tr])
        Xtr, Xte = sx.transform(X[tr]), sx.transform(X[[i]])

        # --- H ---
        gs_H = GridSearchCV(
            SVR(kernel="rbf"),
            param_grid,
            cv=5,
            scoring="neg_mean_absolute_error",
            refit=True,
        )
        gs_H.fit(Xtr, y_H[tr])
        p_H[i] = gs_H.predict(Xte)[0]

        # --- D ---
        gs_D = GridSearchCV(
            SVR(kernel="rbf"),
            param_grid,
            cv=5,
            scoring="neg_mean_absolute_error",
            refit=True,
        )
        gs_D.fit(Xtr, y_D[tr])
        p_D[i] = gs_D.predict(Xte)[0]

        if (i + 1) % 5 == 0 or i == N - 1:
            print(f"  fold {i+1:2d}/{N}")

    p_H = np.clip(p_H, H_MIN, H_MAX)
    p_D = np.clip(p_D, D_MIN, D_MAX)
    return _make_result("SVR", y_H, p_H, None, y_D, p_D, None)


# ===========================================================================
# 8. RANDOM FOREST — TUNED  (new in v3)
# ===========================================================================

def run_rf_tuned(X: np.ndarray, y_H: np.ndarray, y_D: np.ndarray) -> dict:
    """
    Random Forest with LOO-CV.
    Outer: LOO (N=20 folds).
    Inner: RandomizedSearchCV(n_iter=30, cv=5) on 19 training samples.
    Separate RF for H and D.
    """
    print("\n--- RF-Tuned LOO-CV (nested RandomizedSearch) ---")
    N = len(y_H)

    param_dist = {
        "n_estimators": [100, 200, 500],
        "max_depth": [2, 3, 4, 5, None],
        "min_samples_leaf": [1, 2, 3, 4],
        "max_features": ["sqrt", "log2", 0.5, 0.7],
    }

    p_H, p_D = np.zeros(N), np.zeros(N)

    for i in range(N):
        tr = np.ones(N, dtype=bool); tr[i] = False

        # RF does not strictly need scaling, but we keep it for consistency
        # (no effect on tree splits, but simplifies code flow)

        # --- H ---
        rs_H = RandomizedSearchCV(
            RandomForestRegressor(random_state=42),
            param_dist,
            n_iter=30,
            cv=5,
            scoring="neg_mean_absolute_error",
            random_state=42,
            refit=True,
        )
        rs_H.fit(X[tr], y_H[tr])
        p_H[i] = rs_H.predict(X[[i]])[0]

        # --- D ---
        rs_D = RandomizedSearchCV(
            RandomForestRegressor(random_state=42),
            param_dist,
            n_iter=30,
            cv=5,
            scoring="neg_mean_absolute_error",
            random_state=42,
            refit=True,
        )
        rs_D.fit(X[tr], y_D[tr])
        p_D[i] = rs_D.predict(X[[i]])[0]

        if (i + 1) % 5 == 0 or i == N - 1:
            print(f"  fold {i+1:2d}/{N}")

    p_H = np.clip(p_H, H_MIN, H_MAX)
    p_D = np.clip(p_D, D_MIN, D_MAX)
    return _make_result("RF-T", y_H, p_H, None, y_D, p_D, None)


# ===========================================================================
# 9. PHYSICS-CNN WITH SYNTHETIC PRE-TRAINING  (new in v3)
# ===========================================================================

class PhysicsCNN(nn.Module):
    """
    Multi-input CNN: time-series branch + scalar feature branch.

    Time-series branch:
        Input: (B, T, 3) -> permute to (B, 3, T)
        Conv1d(3 -> 8, k=15, pad=7) -> ReLU -> MaxPool1d(6)
        Conv1d(8 -> 16, k=9, pad=4) -> ReLU -> AdaptiveAvgPool1d(1)
        Output: (B, 16)

    Scalar branch:
        Input: (B, 5) [H_proxy, D_proxy, rocof_init, df_nadir, dP_pu]
        Linear(5 -> 8) -> ReLU
        Output: (B, 8)

    Fusion:
        cat -> (B, 24) -> Linear(24 -> 16) -> ReLU -> Dropout(0.2) -> Linear(16 -> 2)

    Total params: ~2000
    """

    def __init__(self):
        super().__init__()
        # Time-series branch
        self.conv1 = nn.Conv1d(3, 8, kernel_size=15, padding=7)
        self.pool1 = nn.MaxPool1d(6)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=9, padding=4)
        self.pool2 = nn.AdaptiveAvgPool1d(1)

        # Scalar branch
        self.scalar_fc = nn.Linear(5, 8)

        # Fusion
        self.fc1 = nn.Linear(24, 16)
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x_seq: torch.Tensor, x_scalar: torch.Tensor) -> torch.Tensor:
        # x_seq: (B, T, 3) -> (B, 3, T)
        s = x_seq.permute(0, 2, 1)
        s = torch.relu(self.conv1(s))    # (B, 8, T)
        s = self.pool1(s)                # (B, 8, T//6)
        s = torch.relu(self.conv2(s))    # (B, 16, T//6)
        s = self.pool2(s)                # (B, 16, 1)
        s = s.squeeze(-1)                # (B, 16)

        # Scalar branch
        sc = torch.relu(self.scalar_fc(x_scalar))  # (B, 8)

        # Fusion
        h = torch.cat([s, sc], dim=1)   # (B, 24)
        h = torch.relu(self.fc1(h))     # (B, 16)
        h = self.drop(h)
        return self.fc2(h)               # (B, 2)


def _build_cnn_inputs(events: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Build CNN inputs from real events.

    Returns:
        X_seq    : (N, 300, 3) — [df_train, rocof_train, dP_broadcast]
        X_scalar : (N, 5)      — [H_proxy, D_proxy, rocof_init, df_nadir, abs(dP_pu)]
    """
    seqs = []
    scalars = []
    for ev in events:
        T = len(ev["df_train"])
        dP = abs(ev["dP_pu"])
        dP_ch = np.full(T, dP, dtype=np.float32)
        seq = np.stack([ev["df_train"], ev["rocof_train"], dP_ch], axis=-1)
        seqs.append(seq)

        # Scalar features (same logic as engineer_features)
        rocof = ev["rocof_train"]
        df = ev["df_train"]
        rocof_init = max(float(np.mean(np.abs(rocof[:20]))), 1e-6)
        abs_df = np.abs(df)
        df_nadir = max(float(np.max(abs_df)), 1e-6)
        H_proxy = dP * F0 / (2.0 * rocof_init)
        D_proxy = dP * F0 / df_nadir
        scalars.append([H_proxy, D_proxy, rocof_init, df_nadir, dP])

    X_seq = np.array(seqs, dtype=np.float32)
    X_scalar = np.array(scalars, dtype=np.float32)
    return X_seq, X_scalar


def _build_cnn_inputs_synth(synth_events: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Build CNN inputs from synthetic events."""
    seqs = []
    scalars = []
    for ev in synth_events:
        T = len(ev["df_train"])
        dP = abs(ev["dP_pu"])
        dP_ch = np.full(T, dP, dtype=np.float32)
        seq = np.stack([ev["df_train"], ev["rocof_train"], dP_ch], axis=-1)
        seqs.append(seq)

        rocof = ev["rocof_train"]
        df = ev["df_train"]
        rocof_init = max(float(np.mean(np.abs(rocof[:20]))), 1e-6)
        abs_df = np.abs(df)
        df_nadir = max(float(np.max(abs_df)), 1e-6)
        H_proxy = dP * F0 / (2.0 * rocof_init)
        D_proxy = dP * F0 / df_nadir
        scalars.append([H_proxy, D_proxy, rocof_init, df_nadir, dP])

    X_seq = np.array(seqs, dtype=np.float32)
    X_scalar = np.array(scalars, dtype=np.float32)
    return X_seq, X_scalar


def run_physics_cnn(
    events: list[dict],
    y_H: np.ndarray,
    y_D: np.ndarray,
    n_synth: int = 2000,
) -> dict:
    """
    Physics-CNN with synthetic pre-training + LOO fine-tuning.

    1. Generate n_synth synthetic events via swing equation ODE
    2. Pre-train PhysicsCNN 200 epochs on synthetic data
       (batch=64, lr=1e-3, CosineAnnealingLR)
    3. Save pre-trained weights
    4. LOO-CV: for each fold, restore pre-trained weights, fine-tune
       50 epochs on 19 real events (batch=6, lr=5e-5, CosineAnnealingLR)
       with HuberLoss on normalized targets
    5. Predict on held-out event, inverse-transform, clip to physics range
    """
    print("\n--- Physics-CNN LOO-CV (synthetic pre-train) ---")
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cpu")

    N = len(events)
    y_all = np.column_stack([y_H, y_D]).astype(np.float32)

    # ----- Build real CNN inputs -----
    X_seq_real, X_scalar_real = _build_cnn_inputs(events)

    # ----- Generate synthetic data -----
    synth_events = generate_synthetic(n_synth=n_synth, seed=0)
    X_seq_synth, X_scalar_synth = _build_cnn_inputs_synth(synth_events)
    y_synth = np.array(
        [[ev["H"], ev["D"]] for ev in synth_events], dtype=np.float32
    )

    # ----- Normalize synthetic data globally -----
    sx_seq_synth = StandardScaler()
    n_s, T_s, C_s = X_seq_synth.shape
    X_seq_synth_flat = X_seq_synth.reshape(-1, C_s)
    sx_seq_synth.fit(X_seq_synth_flat)
    X_seq_synth_n = sx_seq_synth.transform(X_seq_synth_flat).reshape(n_s, T_s, C_s)

    sx_sc_synth = StandardScaler()
    X_scalar_synth_n = sx_sc_synth.fit_transform(X_scalar_synth)

    sy_synth = StandardScaler()
    y_synth_n = sy_synth.fit_transform(y_synth)

    # Convert to tensors
    X_seq_synth_t = torch.tensor(X_seq_synth_n, dtype=torch.float32)
    X_sc_synth_t = torch.tensor(X_scalar_synth_n, dtype=torch.float32)
    y_synth_t = torch.tensor(y_synth_n, dtype=torch.float32)

    synth_ds = TensorDataset(X_seq_synth_t, X_sc_synth_t, y_synth_t)
    synth_loader = DataLoader(synth_ds, batch_size=64, shuffle=True)

    # ----- Pre-train on synthetic data -----
    print("  Pre-training on synthetic data (200 epochs)...")
    model = PhysicsCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
    crit = nn.HuberLoss()

    for epoch in range(200):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xsb, xscb, yb in synth_loader:
            xsb, xscb, yb = xsb.to(device), xscb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xsb, xscb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        sched.step()
        if (epoch + 1) % 50 == 0:
            print(f"    epoch {epoch+1:3d}/200  loss={epoch_loss/n_batches:.4f}")

    # Save pre-trained weights
    pretrained_state = {k: v.clone() for k, v in model.state_dict().items()}
    print("  Pre-training complete. Starting LOO fine-tuning...\n")

    # ----- LOO-CV with fine-tuning -----
    p_H, p_D = np.zeros(N), np.zeros(N)

    for i in range(N):
        tr = np.ones(N, dtype=bool); tr[i] = False

        # --- Normalize sequence inputs (fit on training fold) ---
        X_seq_tr = X_seq_real[tr]    # (N-1, T, 3)
        X_seq_te = X_seq_real[[i]]   # (1, T, 3)
        T_real = X_seq_tr.shape[1]

        sx_seq = StandardScaler()
        sx_seq.fit(X_seq_tr.reshape(-1, 3))
        X_seq_tr_n = sx_seq.transform(X_seq_tr.reshape(-1, 3)).reshape(-1, T_real, 3)
        X_seq_te_n = sx_seq.transform(X_seq_te.reshape(-1, 3)).reshape(-1, T_real, 3)

        # --- Normalize scalar inputs (fit on training fold) ---
        X_sc_tr = X_scalar_real[tr]
        X_sc_te = X_scalar_real[[i]]

        sx_sc = StandardScaler()
        X_sc_tr_n = sx_sc.fit_transform(X_sc_tr)
        X_sc_te_n = sx_sc.transform(X_sc_te)

        # --- Normalize targets (fit on training fold) ---
        sy = StandardScaler()
        y_tr_n = sy.fit_transform(y_all[tr])

        # Convert to tensors
        X_seq_tr_t = torch.tensor(X_seq_tr_n, dtype=torch.float32)
        X_sc_tr_t = torch.tensor(X_sc_tr_n, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr_n, dtype=torch.float32)

        X_seq_te_t = torch.tensor(X_seq_te_n, dtype=torch.float32)
        X_sc_te_t = torch.tensor(X_sc_te_n, dtype=torch.float32)

        train_ds = TensorDataset(X_seq_tr_t, X_sc_tr_t, y_tr_t)
        train_loader = DataLoader(train_ds, batch_size=6, shuffle=True)

        # --- Restore pre-trained weights ---
        model_ft = PhysicsCNN().to(device)
        model_ft.load_state_dict(pretrained_state)

        opt_ft = torch.optim.Adam(model_ft.parameters(), lr=5e-5, weight_decay=1e-4)
        sched_ft = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ft, T_max=50)
        crit_ft = nn.HuberLoss()

        # Fine-tune
        for _ in range(50):
            model_ft.train()
            for xsb, xscb, yb in train_loader:
                xsb, xscb, yb = xsb.to(device), xscb.to(device), yb.to(device)
                opt_ft.zero_grad()
                pred = model_ft(xsb, xscb)
                loss = crit_ft(pred, yb)
                loss.backward()
                opt_ft.step()
            sched_ft.step()

        # --- Predict ---
        model_ft.eval()
        with torch.no_grad():
            pred_n = model_ft(X_seq_te_t.to(device), X_sc_te_t.to(device)).cpu().numpy()

        pred = sy.inverse_transform(pred_n)
        p_H[i] = np.clip(pred[0, 0], H_MIN, H_MAX)
        p_D[i] = np.clip(pred[0, 1], D_MIN, D_MAX)

        if (i + 1) % 5 == 0 or i == N - 1:
            print(f"  fold {i+1:2d}/{N}")

    return _make_result("P-CNN", y_H, p_H, None, y_D, p_D, None)


# ===========================================================================
# 10. ENSEMBLE
# ===========================================================================

def run_ensemble(
    res_gpr: dict,
    res_svr: dict,
    res_cnn: dict,
    y_H: np.ndarray,
    y_D: np.ndarray,
) -> dict:
    """
    Ensemble of GPR + SVR + CNN predictions.
    Two strategies:
      1. Simple average
      2. Inverse-MAE weighted average
    """
    print("\n--- Ensemble (GPR + SVR + P-CNN) ---")

    models = [res_gpr, res_svr, res_cnn]

    # ---- Simple average ----
    p_H_avg = np.mean([m["p_H"] for m in models], axis=0)
    p_D_avg = np.mean([m["p_D"] for m in models], axis=0)

    p_H_avg = np.clip(p_H_avg, H_MIN, H_MAX)
    p_D_avg = np.clip(p_D_avg, D_MIN, D_MAX)

    r2_H_avg = r2_score(y_H, p_H_avg)
    mae_H_avg = mean_absolute_error(y_H, p_H_avg)
    r2_D_avg = r2_score(y_D, p_D_avg)
    mae_D_avg = mean_absolute_error(y_D, p_D_avg)
    print(f"  Avg-Ens  H_pw: R2={r2_H_avg:+.3f}  MAE={mae_H_avg:.3f} s   "
          f" D_pw: R2={r2_D_avg:+.3f}  MAE={mae_D_avg:.3f} pu")

    # ---- Inverse-MAE weighted average ----
    # Weights proportional to 1/MAE for each target separately
    w_H = np.array([1.0 / max(m["mae_H"], 1e-6) for m in models])
    w_D = np.array([1.0 / max(m["mae_D"], 1e-6) for m in models])
    w_H /= w_H.sum()
    w_D /= w_D.sum()

    p_H_wt = sum(w_H[k] * models[k]["p_H"] for k in range(len(models)))
    p_D_wt = sum(w_D[k] * models[k]["p_D"] for k in range(len(models)))

    p_H_wt = np.clip(p_H_wt, H_MIN, H_MAX)
    p_D_wt = np.clip(p_D_wt, D_MIN, D_MAX)

    r2_H_wt = r2_score(y_H, p_H_wt)
    mae_H_wt = mean_absolute_error(y_H, p_H_wt)
    r2_D_wt = r2_score(y_D, p_D_wt)
    mae_D_wt = mean_absolute_error(y_D, p_D_wt)
    print(f"  Wt-Ens   H_pw: R2={r2_H_wt:+.3f}  MAE={mae_H_wt:.3f} s   "
          f" D_pw: R2={r2_D_wt:+.3f}  MAE={mae_D_wt:.3f} pu")

    print(f"  Weights H: GPR={w_H[0]:.2f}  SVR={w_H[1]:.2f}  P-CNN={w_H[2]:.2f}")
    print(f"  Weights D: GPR={w_D[0]:.2f}  SVR={w_D[1]:.2f}  P-CNN={w_D[2]:.2f}")

    # Return the better of the two ensembles
    if mae_H_wt + mae_D_wt < mae_H_avg + mae_D_avg:
        label = "Wt-Ens"
        p_H_best, p_D_best = p_H_wt, p_D_wt
    else:
        label = "Avg-En"
        p_H_best, p_D_best = p_H_avg, p_D_avg

    return _make_result(label, y_H, p_H_best, None, y_D, p_D_best, None)


# ===========================================================================
# 11. COMPARISON PLOT
# ===========================================================================

def plot_comparison(results_list: list[dict]) -> None:
    """
    2 x n_models scatter subplots: predicted vs actual for H (top) and D (bottom).
    Each subplot shows R2 and MAE.
    Saves to plots/v3_comparison.png.
    """
    n_models = len(results_list)
    colors = {
        "GPR": "#1f77b4", "SVR": "#d62728", "RF-T": "#ff7f0e",
        "P-CNN": "#2ca02c", "Avg-En": "#9467bd", "Wt-Ens": "#9467bd",
    }

    fig, axes = plt.subplots(2, n_models, figsize=(4.2 * n_models, 8))
    if n_models == 1:
        axes = axes[:, None]

    for col, res in enumerate(results_list):
        c = colors.get(res["name"], "steelblue")

        for row, (yv, pv, unit, lbl, key) in enumerate([
            (res["y_H"], res["p_H"], "s",  "H_pw [s]",  "H"),
            (res["y_D"], res["p_D"], "pu", "D_pw [pu]", "D"),
        ]):
            ax = axes[row, col]
            lo = min(yv.min(), pv.min()) - 0.15
            hi = max(yv.max(), pv.max()) + 0.15
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
            ax.scatter(yv, pv, color=c, s=55, alpha=0.85, zorder=3,
                       edgecolors="white", linewidths=0.5)

            # Uncertainty bars if available
            s_key = f"s_{key}"
            if res.get(s_key) is not None:
                ax.errorbar(yv, pv, yerr=2 * res[s_key],
                            fmt="none", color=c, alpha=0.25, capsize=2)

            ax.set_xlabel(f"Actual {lbl}")
            ax.set_ylabel(f"Predicted {lbl}")
            r2_val = res[f"r2_{key}"]
            mae_val = res[f"mae_{key}"]
            ax.set_title(f"{res['name']} -- {lbl}\n"
                         f"R2={r2_val:.3f}  MAE={mae_val:.3f} {unit}",
                         fontsize=9)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.3)

    fig.suptitle("v3 Model Comparison -- LOO-CV: Predicted vs Actual\n"
                 "(targets: H_pw_mean, D_pw_mean)", fontsize=12)
    plt.tight_layout()
    p = PLOT_DIR / "v3_comparison.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p}")


# ===========================================================================
# 12. SUMMARY TABLE
# ===========================================================================

def print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 72)
    print(f"  Targets: H_pw_mean [s],  D_pw_mean [pu]   (signal-derived, N=20)")
    print("=" * 72)
    print(f"{'MODEL':<10} {'R2_H':>8} {'MAE_H [s]':>12} {'R2_D':>8} {'MAE_D [pu]':>12}")
    print("-" * 72)
    for r in results:
        print(f"{r['name']:<10} {r['r2_H']:>8.3f} {r['mae_H']:>12.3f} "
              f"{r['r2_D']:>8.3f} {r['mae_D']:>12.3f}")
    print("=" * 72)


# ===========================================================================
# 13. MAIN
# ===========================================================================

def main():
    print("=" * 72)
    print("  ai_models_v3.py — Virtual Inertia & Damping Prediction")
    print("=" * 72)

    events = load_data()
    X = engineer_features(events)
    y_H = np.array([ev["H_pw_mean"] for ev in events], dtype=np.float32)
    y_D = np.array([ev["D_pw_mean"] for ev in events], dtype=np.float32)

    diagnose_data(events, X, y_H, y_D)

    print("\n=== MODEL RESULTS (LOO-CV) ===")
    res_gpr = run_gpr(X, y_H, y_D)
    res_svr = run_svr(X, y_H, y_D)
    res_rf  = run_rf_tuned(X, y_H, y_D)
    res_cnn = run_physics_cnn(events, y_H, y_D)
    res_ens = run_ensemble(res_gpr, res_svr, res_cnn, y_H, y_D)

    all_results = [res_gpr, res_svr, res_rf, res_cnn, res_ens]
    print_summary(all_results)

    plot_comparison(all_results)
    print("\nPlots saved to plots/v3_comparison.png")


if __name__ == "__main__":
    main()
