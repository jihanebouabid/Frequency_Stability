"""
ai_models.py  (v2 — corrected)
-------------------------------
Fixes applied from analysis_results_concerns.md:

  [FIX-1] Targets changed to H_pw_mean + D_pw_mean (signal-derived, non-leaky).
          H_from_Ek is an external report value, not derivable from the PMU signal.
          D_nadir was trivially reconstructed from df_nadir × dP in the feature set.

  [FIX-2] D_proxy re-added to features as a LEGITIMATE input:
          D_proxy = D_nadir ≠ D_pw_mean (trajectory estimate vs nadir-point estimate).
          Correlation with D_pw_mean is r ≈ 0.78, so it is informative but not leaky.

  [FIX-3] Oct-08 event dropped (H_pw_n = 0 → NaN H_pw_mean, D_pw_std = 2.0 pu).
          N = 20 events used consistently across all models.

  [FIX-4] GRU input extended to (T, 3): [Δf(t), RoCoF(t), dP_broadcast].
          Without dP the signal is physically unidentifiable: the same RoCoF
          magnitude can arise from any (H, dP) combination.

  [FIX-5] GRU training: early stopping on 2 validation samples replaced by
          a fixed 150-epoch budget + CosineAnnealingLR (avoids noisy stopping).

  [NEW]   Virtual inertia augmentation:
          - size_virtual_inertia(): physics-based H_virt/D_virt sizing.
          - augment_trajectory(): VSG signal transformation formula.
          - plot_augmented_trajectories(): original vs augmented Δf comparison
            using LOO-predicted H and D values from the best model.

Models:
  1. GPR  — Gaussian Process Regression on physics-informed features (Arch. 4.1)
  2. RF   — Random Forest on physics-informed features              (Arch. 4.3)
  3. GRU  — Recurrent sequence model, PyTorch, dP context           (Arch. 4.7)

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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

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


# ---------------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 2. PHYSICS-INFORMED FEATURE ENGINEERING  (Tier 1 + 2)
# ---------------------------------------------------------------------------

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
    print(f"Feature matrix: {X.shape}  (N_events × N_features)")
    return X


# ---------------------------------------------------------------------------
# 3. METRICS HELPER
# ---------------------------------------------------------------------------

def _make_result(name, y_H, p_H, s_H, y_D, p_D, s_D) -> dict:
    r2_H  = r2_score(y_H, p_H)
    mae_H = mean_absolute_error(y_H, p_H)
    r2_D  = r2_score(y_D, p_D)
    mae_D = mean_absolute_error(y_D, p_D)
    print(f"  {name:5s}  H_pw: R²={r2_H:+.3f}  MAE={mae_H:.3f} s   "
          f" D_pw: R²={r2_D:+.3f}  MAE={mae_D:.3f} pu")
    return dict(name=name,
                y_H=y_H, p_H=p_H, s_H=s_H,
                y_D=y_D, p_D=p_D, s_D=s_D,
                r2_H=r2_H, mae_H=mae_H,
                r2_D=r2_D, mae_D=mae_D)


# ---------------------------------------------------------------------------
# 4. GPR  (Architecture 4.1)
# ---------------------------------------------------------------------------

def run_gpr(X: np.ndarray, y_H: np.ndarray, y_D: np.ndarray) -> dict:
    """
    Gaussian Process Regression with LOO-CV.
    Matern-5/2 × ConstantKernel + WhiteKernel.
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


# ---------------------------------------------------------------------------
# 5. RANDOM FOREST  (Architecture 4.3)
# ---------------------------------------------------------------------------

def run_rf(X: np.ndarray, y_H: np.ndarray, y_D: np.ndarray) -> dict:
    """Random Forest LOO-CV. Shallow trees (max_depth=3) for N=20."""
    print("\n--- Random Forest LOO-CV ---")
    N = len(y_H)
    p_H, p_D = np.zeros(N), np.zeros(N)

    for i in range(N):
        tr = np.ones(N, dtype=bool); tr[i] = False

        rfH = RandomForestRegressor(n_estimators=300, max_depth=3,
                                    min_samples_leaf=3, random_state=42)
        rfD = RandomForestRegressor(n_estimators=300, max_depth=3,
                                    min_samples_leaf=3, random_state=42)
        rfH.fit(X[tr], y_H[tr]);  rfD.fit(X[tr], y_D[tr])
        p_H[i] = rfH.predict(X[[i]])[0]
        p_D[i] = rfD.predict(X[[i]])[0]

        if (i + 1) % 5 == 0 or i == N - 1:
            print(f"  fold {i+1:2d}/{N}")

    p_H = np.clip(p_H, H_MIN, H_MAX)
    p_D = np.clip(p_D, D_MIN, D_MAX)
    return _make_result("RF", y_H, p_H, None, y_D, p_D, None)


# ---------------------------------------------------------------------------
# 6. GRU  (Architecture 4.7, PyTorch — fixed)
# ---------------------------------------------------------------------------

class GRUModel(nn.Module):
    """
    Two-layer GRU for joint [H_pw, D_pw] regression.

    Input: (batch, T, 3)  — [Δf(t), RoCoF(t), dP_pu_broadcast]
    FIX-4: dP channel resolves the physical ambiguity: without knowing dP,
           the same RoCoF magnitude is consistent with any value of H.

    Architecture:
      GRU(32, seq) → Dropout → GRU(16, last) → Dense(16,ReLU) → Dropout → Dense(2)
    ~9K parameters.
    """

    def __init__(self, input_size: int = 3):
        super().__init__()
        self.gru1  = nn.GRU(input_size, 32, batch_first=True)
        self.drop1 = nn.Dropout(0.2)
        self.gru2  = nn.GRU(32, 16, batch_first=True)
        self.fc1   = nn.Linear(16, 16)
        self.drop2 = nn.Dropout(0.2)
        self.fc2   = nn.Linear(16, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _  = self.gru1(x)       # (B, T, 32)
        out     = self.drop1(out)
        _, h    = self.gru2(out)      # h: (1, B, 16)
        h       = h.squeeze(0)        # (B, 16)
        h       = torch.relu(self.fc1(h))
        h       = self.drop2(h)
        return self.fc2(h)            # (B, 2)


def _build_seq_tensor(events: list[dict]) -> torch.Tensor:
    """
    Build (N, T, 3) tensor: [Δf(t), RoCoF(t), dP_pu_broadcast].
    FIX-4: dP is broadcast as a constant channel over all T timesteps.
    """
    seqs = []
    for ev in events:
        T   = len(ev["df_train"])
        dP_ch = np.full(T, abs(ev["dP_pu"]), dtype=np.float32)
        seq = np.stack([ev["df_train"], ev["rocof_train"], dP_ch], axis=-1)
        seqs.append(seq)
    return torch.tensor(np.array(seqs, dtype=np.float32))  # (N, T, 3)


def run_gru(
    events:     list[dict],
    y_H:        np.ndarray,
    y_D:        np.ndarray,
    epochs:     int   = 150,
    lr:         float = 1e-3,
    batch_size: int   = 6,
    seed:       int   = 42,
) -> dict:
    """
    GRU LOO-CV.

    FIX-4: input (T, 3) includes dP broadcast channel.
    FIX-5: fixed 150-epoch budget + CosineAnnealingLR replaces early stopping
           on 2 validation samples (which was noise-driven, not signal-driven).
    """
    print("\n--- GRU LOO-CV (PyTorch) ---")
    torch.manual_seed(seed)
    device = torch.device("cpu")

    N     = len(events)
    X_raw = _build_seq_tensor(events)           # (N, T, 3)
    T     = X_raw.shape[1]
    y_all = np.column_stack([y_H, y_D]).astype(np.float32)

    p_H, p_D = np.zeros(N), np.zeros(N)

    for i in range(N):
        tr = np.ones(N, dtype=bool); tr[i] = False

        # ---- normalise inputs (fit on train fold only) ----
        X_tr_np = X_raw[tr].numpy()             # (N-1, T, 3)
        X_te_np = X_raw[[i]].numpy()            # (1,   T, 3)

        sx = StandardScaler().fit(X_tr_np.reshape(-1, 3))
        X_tr_n = sx.transform(X_tr_np.reshape(-1, 3)).reshape(-1, T, 3)
        X_te_n = sx.transform(X_te_np.reshape(-1, 3)).reshape(-1, T, 3)

        # ---- normalise targets (fit on train fold only) ----
        sy = StandardScaler().fit(y_all[tr])
        y_tr_n = sy.transform(y_all[tr])

        X_tr_t = torch.tensor(X_tr_n, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr_n, dtype=torch.float32)
        X_te_t = torch.tensor(X_te_n, dtype=torch.float32)

        loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                            batch_size=batch_size, shuffle=True)

        # ---- model + scheduler ----
        model = GRUModel().to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        crit  = nn.HuberLoss()

        # FIX-5: no early stopping — fixed epoch budget with cosine LR decay
        for _ in range(epochs):
            model.train()
            for Xb, yb in loader:
                opt.zero_grad()
                crit(model(Xb.to(device)), yb.to(device)).backward()
                opt.step()
            sched.step()

        # ---- predict ----
        model.eval()
        with torch.no_grad():
            pred_n = model(X_te_t.to(device)).cpu().numpy()  # (1, 2)

        pred = sy.inverse_transform(pred_n)
        p_H[i] = np.clip(pred[0, 0], H_MIN, H_MAX)
        p_D[i] = np.clip(pred[0, 1], D_MIN, D_MAX)

        if (i + 1) % 5 == 0 or i == N - 1:
            print(f"  fold {i+1:2d}/{N}")

    return _make_result("GRU", y_H, p_H, None, y_D, p_D, None)


# ---------------------------------------------------------------------------
# 7. VIRTUAL INERTIA AUGMENTATION
# ---------------------------------------------------------------------------

def size_virtual_inertia(
    H_pred: float,
    D_pred: float,
    dP_pu:  float,
    df_target:    float = 0.4,   # Hz  — stricter than 0.5 Hz UFLS limit
    rocof_target: float = 0.8,   # Hz/s — stricter than 1.0 Hz/s limit
    H_virt_min:   float = 0.3,   # s   — always inject at least this much
    D_virt_min:   float = 0.2,   # pu  — always inject at least this much
) -> tuple[float, float]:
    """
    Physics-based VSG virtual inertia and damping sizing.

    From swing equation at key points:
      H_min = |dP| * f0 / (2 * RoCoF_max)   [RoCoF constraint]
      D_min = |dP| * f0 / df_max             [nadir constraint]

    H_virt and D_virt are the additional virtual contributions needed to
    push the system from (H_pred, D_pred) to at least (H_min, D_min).
    A minimum reserve is always injected even if the system is already stable,
    to improve safety margin.

    Returns (H_virt, D_virt) both >= 0.
    """
    dP = abs(dP_pu)
    H_needed = dP * F0 / (2.0 * rocof_target)
    D_needed = dP * F0 / df_target

    H_virt = max(H_virt_min, H_needed - H_pred)
    D_virt = max(D_virt_min, D_needed - D_pred)
    return float(H_virt), float(D_virt)


def augment_trajectory(
    df_t:    np.ndarray,
    H_pred:  float,
    D_pred:  float,
    H_virt:  float,
    D_virt:  float,
) -> np.ndarray:
    """
    Apply VSG virtual inertia + damping augmentation to a frequency signal.

    Formula (from ai_model_proposal.md §2.3):
        df_aug(t) = df_meas(t / alpha) * beta

    where:
        alpha = (H_pred + H_virt) / H_pred   > 1 → time-stretches the signal
        beta  = D_pred / (D_pred + D_virt)   < 1 → scales amplitude down

    Effect:
        - Nadir occurs at t_nadir * alpha (later → more inertial response time)
        - Nadir depth reduced by factor beta
        - RoCoF reduced by factor beta / alpha

    Assumes delta_f_pre ≈ 0 (disturbance starts from nominal frequency).

    Parameters
    ----------
    df_t   : Δf signal array, length T, uniform DT spacing
    H_pred : estimated base inertia [s]
    D_pred : estimated base damping [pu]
    H_virt : virtual inertia to inject [s]
    D_virt : virtual damping to inject [pu]

    Returns df_aug : same length as df_t
    """
    alpha = (H_pred + H_virt) / H_pred   # time stretch factor  (> 1)
    beta  = D_pred / (D_pred + D_virt)   # amplitude scale      (< 1)

    T   = len(df_t)
    t   = np.arange(T, dtype=np.float32) * DT          # original time axis
    t_q = t / alpha                                     # compressed query times

    # Interpolate original signal at compressed time points
    # np.interp clamps t_q to [t[0], t[-1]] at the boundaries
    df_aug = np.interp(t_q, t, df_t).astype(np.float32) * beta

    return df_aug


# ---------------------------------------------------------------------------
# 8. RESULTS TABLE
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 72)
    print(f"  Targets: H_pw_mean [s],  D_pw_mean [pu]   (signal-derived, N=20)")
    print("=" * 72)
    print(f"{'MODEL':<8} {'R²_H':>8} {'MAE_H [s]':>12} {'R²_D':>8} {'MAE_D [pu]':>12}")
    print("-" * 72)
    for r in results:
        print(f"{r['name']:<8} {r['r2_H']:>8.3f} {r['mae_H']:>12.3f} "
              f"{r['r2_D']:>8.3f} {r['mae_D']:>12.3f}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# 9. PLOTS
# ---------------------------------------------------------------------------

def plot_scatter_residuals(results: list[dict], events: list[dict]) -> None:
    """
    Fig 1 — LOO scatter: predicted vs actual for H and D (one row per model).
    Fig 2 — Residuals over event date.
    """
    labels = [ev["onset_str"][:8] for ev in events]
    colors = {"GPR": "#1f77b4", "RF": "#ff7f0e", "GRU": "#2ca02c"}
    n = len(results)

    # ---- scatter ----
    fig, axes = plt.subplots(n, 2, figsize=(10, 3.2 * n))
    if n == 1:
        axes = axes[None, :]

    for row, res in enumerate(results):
        c = colors.get(res["name"], "steelblue")
        for col, (yv, pv, unit, lbl, key) in enumerate([
            (res["y_H"], res["p_H"], "s",  "H_pw [s]",  "H"),
            (res["y_D"], res["p_D"], "pu", "D_pw [pu]", "D"),
        ]):
            ax  = axes[row, col]
            lo  = min(yv.min(), pv.min()) - 0.1
            hi  = max(yv.max(), pv.max()) + 0.1
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
            ax.scatter(yv, pv, color=c, s=55, alpha=0.85, zorder=3)
            if res.get(f"s_{key}") is not None:
                ax.errorbar(yv, pv, yerr=2 * res[f"s_{key}"],
                            fmt="none", color=c, alpha=0.25, capsize=2)
            ax.set_xlabel(f"Actual {lbl}")
            ax.set_ylabel(f"Predicted {lbl}")
            ax.set_title(f"{res['name']} — {lbl}  "
                         f"R²={res[f'r2_{key}']:.3f}  MAE={res[f'mae_{key}']:.3f} {unit}")
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.grid(True, alpha=0.3)

    fig.suptitle("LOO-CV: Predicted vs Actual  "
                 "(targets: H_pw_mean, D_pw_mean — signal-derived)",
                 fontsize=11)
    plt.tight_layout()
    p = PLOT_DIR / "model_scatter_loo.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")

    # ---- residuals ----
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    x = np.arange(len(events))

    for res in results:
        c = colors.get(res["name"], "steelblue")
        axes[0].plot(x, res["p_H"] - res["y_H"], "o-",
                     color=c, label=res["name"], alpha=0.8, ms=5)
        axes[1].plot(x, res["p_D"] - res["y_D"], "o-",
                     color=c, label=res["name"], alpha=0.8, ms=5)

    for ax in axes:
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("ΔH residual [s]  (pred − actual)")
    axes[1].set_ylabel("ΔD residual [pu] (pred − actual)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[0].legend(framealpha=0.7)
    axes[0].set_title("LOO-CV Residuals by Event Date  "
                      "(targets: H_pw_mean, D_pw_mean)")

    plt.tight_layout()
    p = PLOT_DIR / "model_residuals_loo.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")


def plot_augmented_trajectories(
    events:  list[dict],
    results: list[dict],
) -> None:
    """
    For each model's LOO-predicted H and D, show original Δf vs augmented Δf.
    Uses the 30 s training window (df_train, T=300).

    Layout: N_events rows × N_models columns.
    Each panel annotates nadir improvement and sizing.
    """
    colors = {"GPR": "#1f77b4", "RF": "#ff7f0e", "GRU": "#2ca02c"}
    N = len(events)
    M = len(results)
    t = np.arange(300) * DT   # 0 to 29.9 s

    fig, axes = plt.subplots(N, M, figsize=(5 * M, 2.8 * N),
                              sharex=True, sharey=False)
    if N == 1:
        axes = axes[None, :]
    if M == 1:
        axes = axes[:, None]

    for i, ev in enumerate(events):
        df_orig = ev["df_train"]
        dP_pu   = ev["dP_pu"]
        nadir_o = float(np.max(np.abs(df_orig)))

        for j, res in enumerate(results):
            ax = axes[i, j]
            H_pred = float(res["p_H"][i])
            D_pred = float(res["p_D"][i])
            H_virt, D_virt = size_virtual_inertia(H_pred, D_pred, dP_pu)
            df_aug = augment_trajectory(df_orig, H_pred, D_pred, H_virt, D_virt)

            nadir_a   = float(np.max(np.abs(df_aug)))
            improve   = (nadir_o - nadir_a) / nadir_o * 100.0
            alpha_val = (H_pred + H_virt) / H_pred
            beta_val  = D_pred / (D_pred + D_virt)

            ax.plot(t, df_orig, color="dimgray",  lw=1.2, label="Original", alpha=0.85)
            ax.plot(t, df_aug,  color=colors.get(res["name"], "steelblue"),
                    lw=1.6, label="Augmented", alpha=0.95)

            # mark nadirs
            ni_o = int(np.argmax(np.abs(df_orig)))
            ni_a = int(np.argmax(np.abs(df_aug)))
            ax.axhline(0, color="k", lw=0.4, ls="--", alpha=0.4)
            ax.plot(t[ni_o], df_orig[ni_o], "v", color="tomato",
                    ms=7, zorder=5, label=f"Nadir orig: {df_orig[ni_o]:+.3f} Hz")
            ax.plot(t[ni_a], df_aug[ni_a], "^", color="seagreen",
                    ms=7, zorder=5, label=f"Nadir aug:  {df_aug[ni_a]:+.3f} Hz")

            if i == 0:
                ax.set_title(f"{res['name']}", fontsize=11, fontweight="bold")

            # annotation box
            ann = (f"Ĥ={H_pred:.2f}s  D̂={D_pred:.2f}pu\n"
                   f"+H_v={H_virt:.2f}s +D_v={D_virt:.2f}pu\n"
                   f"α={alpha_val:.2f}  β={beta_val:.2f}\n"
                   f"Nadir ↓{improve:.1f}%")
            ax.text(0.98, 0.03, ann, transform=ax.transAxes,
                    fontsize=6.5, ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75))

            ax.grid(True, alpha=0.2)
            if j == 0:
                month = ev["onset_str"][4:6]
                ax.set_ylabel(f"{ev['onset_str'][:8]}\n(M{month}) Δf [Hz]",
                              fontsize=7)
            if i == N - 1:
                ax.set_xlabel("Time [s]")

    # single legend (bottom)
    handles = [
        plt.Line2D([0], [0], color="dimgray",  lw=1.5, label="Original"),
        plt.Line2D([0], [0], color="steelblue", lw=1.5, label="Augmented"),
        plt.Line2D([0], [0], marker="v", color="tomato",  lw=0,
                   ms=7, label="Nadir original"),
        plt.Line2D([0], [0], marker="^", color="seagreen", lw=0,
                   ms=7, label="Nadir augmented"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=9, framealpha=0.8)
    fig.suptitle("Virtual Inertia & Damping Augmentation  "
                 "(LOO-predicted H & D → physics-sized H_virt, D_virt)",
                 fontsize=12, fontweight="bold")

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    p = PLOT_DIR / "augmented_trajectories.png"
    plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")

    # ---- companion: nadir improvement bar chart ----
    fig, axes = plt.subplots(1, M, figsize=(5 * M, 4), sharey=True)
    if M == 1:
        axes = [axes]

    x      = np.arange(N)
    labels = [ev["onset_str"][:8] for ev in events]

    for j, res in enumerate(results):
        improvements = []
        for i, ev in enumerate(events):
            df_orig   = ev["df_train"]
            H_pred    = float(res["p_H"][i])
            D_pred    = float(res["p_D"][i])
            H_virt, D_virt = size_virtual_inertia(H_pred, D_pred, ev["dP_pu"])
            df_aug    = augment_trajectory(df_orig, H_pred, D_pred, H_virt, D_virt)
            nadir_o   = float(np.max(np.abs(df_orig)))
            nadir_a   = float(np.max(np.abs(df_aug)))
            improvements.append((nadir_o - nadir_a) / nadir_o * 100.0)

        ax = axes[j]
        bars = ax.bar(x, improvements, color=colors.get(res["name"], "steelblue"),
                      alpha=0.8, edgecolor="white", lw=0.5)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{res['name']}", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Nadir depth reduction [%]")
        ax.grid(True, axis="y", alpha=0.3)

        for bar, val in zip(bars, improvements):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=6)

    fig.suptitle("Nadir Depth Reduction per Event  "
                 "(physics-sized virtual inertia + damping)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    p = PLOT_DIR / "nadir_improvement.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")


# ---------------------------------------------------------------------------
# 10. FULL SIGNAL COMPARISON
# ---------------------------------------------------------------------------

def plot_full_signal_comparison(
    events:  list[dict],
    results: list[dict],
) -> None:
    """
    One subplot per event showing the complete recorded frequency trajectory
    (df_full @ 155 s where available, df_train @ 30 s otherwise) overlaid with
    the virtually augmented signal produced by the best LOO model's predictions.

    The augmentation is applied only to the post-onset portion (t >= 0).
    The pre-onset segment (t < 0) is unchanged — it is already at nominal frequency.

    Subplot annotation:
      - Event date, cause, direction
      - Predicted Ĥ / D̂ and sized H_virt / D_virt
      - Original nadir and augmented nadir in Hz
      - Nadir depth reduction in %
      - α (time-stretch) and β (amplitude-scale) factors

    A purple dashed vertical line marks the disturbance onset (t = 0) for
    events that show the pre-onset window.
    """
    best = max(results, key=lambda r: r["r2_H"] + r["r2_D"])
    c_aug = {"GPR": "#1f77b4", "RF": "#ff7f0e", "GRU": "#2ca02c"}.get(
        best["name"], "steelblue"
    )

    N     = len(events)
    ncols = 4
    nrows = (N + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.6 * nrows))
    axes_flat = axes.flatten()

    for i, ev in enumerate(events):
        ax = axes_flat[i]

        H_pred = float(best["p_H"][i])
        D_pred = float(best["p_D"][i])
        H_virt, D_virt = size_virtual_inertia(H_pred, D_pred, ev["dP_pu"])
        alpha = (H_pred + H_virt) / H_pred
        beta  = D_pred / (D_pred + D_virt)

        # ---- choose signal source ----
        if ev["df_full"] is not None:
            t_arr  = ev["t_full"]                     # -5 to 150 s
            df_arr = ev["df_full"]
            onset  = int(np.searchsorted(t_arr, 0.0)) # first index with t >= 0

            df_post_aug = augment_trajectory(
                df_arr[onset:], H_pred, D_pred, H_virt, D_virt
            )
            df_aug = np.concatenate([df_arr[:onset], df_post_aug])
            has_full = True
        else:
            t_arr    = np.arange(len(ev["df_train"])) * DT   # 0 to 29.9 s
            df_arr   = ev["df_train"]
            df_aug   = augment_trajectory(df_arr, H_pred, D_pred, H_virt, D_virt)
            has_full = False

        nadir_o  = float(np.max(np.abs(df_arr)))
        nadir_a  = float(np.max(np.abs(df_aug)))
        improve  = (nadir_o - nadir_a) / nadir_o * 100.0

        # nadir indices
        ni_o = int(np.argmax(np.abs(df_arr)))
        ni_a = int(np.argmax(np.abs(df_aug)))

        # ---- plot ----
        ax.fill_between(t_arr, df_arr, alpha=0.08, color="dimgray")
        ax.plot(t_arr, df_arr, color="dimgray", lw=1.1, alpha=0.9, label="Original")
        ax.plot(t_arr, df_aug, color=c_aug,     lw=1.6, alpha=0.92, label="Augmented")
        ax.axhline(0, color="k", lw=0.4, ls="--", alpha=0.35)

        if has_full:
            ax.axvline(0, color="#8b008b", lw=0.8, ls=":", alpha=0.7,
                       label="Onset (t=0)")

        ax.plot(t_arr[ni_o], df_arr[ni_o], "v",
                color="tomato", ms=6, zorder=5,
                label=f"Nadir orig: {df_arr[ni_o]:+.3f} Hz")
        ax.plot(t_arr[ni_a], df_aug[ni_a], "^",
                color="seagreen", ms=6, zorder=5,
                label=f"Nadir aug:  {df_aug[ni_a]:+.3f} Hz")

        # annotation
        cause_short = ev["cause"][:4]
        dir_sym     = "↑" if ev["direction"] > 0 else "↓"
        ann = (f"Ĥ={H_pred:.2f}s  D̂={D_pred:.2f}pu\n"
               f"+H_v={H_virt:.2f}s  +D_v={D_virt:.2f}pu\n"
               f"α={alpha:.2f}  β={beta:.2f}\n"
               f"Nadir {df_arr[ni_o]:+.3f}→{df_aug[ni_a]:+.3f} Hz  ↓{improve:.1f}%")
        ax.text(0.98, 0.97, ann, transform=ax.transAxes,
                fontsize=6.2, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.82))

        # axes labels / title
        src_tag = "155 s" if has_full else "30 s"
        ax.set_title(
            f"{ev['onset_str'][:8]}  {cause_short}{dir_sym}  [{src_tag}]",
            fontsize=8, fontweight="bold"
        )
        ax.set_xlabel("Time [s]", fontsize=7)
        ax.set_ylabel("Δf [Hz]",  fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.18)

    # hide any spare subplots
    for j in range(N, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # shared legend
    handles = [
        plt.Line2D([0], [0], color="dimgray",  lw=1.5, label="Original Δf"),
        plt.Line2D([0], [0], color=c_aug,       lw=1.5,
                   label=f"Augmented Δf  ({best['name']} LOO predictions)"),
        plt.Line2D([0], [0], marker="v", color="tomato",   lw=0, ms=7,
                   label="Original nadir"),
        plt.Line2D([0], [0], marker="^", color="seagreen", lw=0, ms=7,
                   label="Augmented nadir"),
        plt.Line2D([0], [0], color="#8b008b", lw=0.9, ls=":",
                   label="Disturbance onset (t = 0)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               fontsize=8.5, framealpha=0.85)

    fig.suptitle(
        f"Full Frequency Signal: Original vs Virtually Augmented  "
        f"(best model: {best['name']},  R²_H={best['r2_H']:.3f},  "
        f"R²_D={best['r2_D']:.3f})\n"
        f"df_full (155 s window) where available  |  df_train (30 s) otherwise",
        fontsize=10.5, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.045, 1, 0.96])

    p = PLOT_DIR / "full_signal_comparison.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p}")


# ---------------------------------------------------------------------------
# 11. MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print(" AI Model Pipeline v2 — Nordic Grid Virtual Inertia & Damping")
    print("=" * 72)

    # ---- load & filter ----
    events = load_data()          # N=20 (Oct-08 dropped)
    N      = len(events)

    # ---- targets: signal-derived (FIX-1) ----
    y_H = np.array([ev["H_pw_mean"] for ev in events], dtype=np.float32)
    y_D = np.array([ev["D_pw_mean"] for ev in events], dtype=np.float32)

    print(f"H_pw_mean: {y_H.min():.2f}–{y_H.max():.2f} s  "
          f"(mean={y_H.mean():.2f}  std={y_H.std():.2f})")
    print(f"D_pw_mean: {y_D.min():.2f}–{y_D.max():.2f} pu "
          f"(mean={y_D.mean():.2f}  std={y_D.std():.2f})")

    # ---- features: 10-dim, non-leaky for new targets (FIX-2) ----
    X = engineer_features(events)

    # ---- models ----
    results = []
    results.append(run_gpr(X, y_H, y_D))
    results.append(run_rf (X, y_H, y_D))
    results.append(run_gru(events, y_H, y_D))

    # ---- summary ----
    print_summary(results)

    # ---- standard plots ----
    print("\nGenerating diagnostic plots …")
    plot_scatter_residuals(results, events)

    # ---- augmentation plots (ultimate goal) ----
    print("Generating virtual inertia augmentation plots …")
    plot_augmented_trajectories(events, results)

    # ---- full signal comparison (complete trajectory view) ----
    print("Generating full signal comparison …")
    plot_full_signal_comparison(events, results)

    print("\nDone.  Plots saved to:", PLOT_DIR)


if __name__ == "__main__":
    main()
