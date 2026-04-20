"""
ai_models_v12.py
----------------
Physics-Informed Neural Network for Virtual Inertia and Damping Estimation

Inputs
------
  Time series (2 channels x 1500 time steps, 150 s at 10 Hz):
    ch0 : Delta_f(t)   frequency deviation  [Hz]
    ch1 : RoCoF(t)     rate of change of frequency  [Hz/s]
  Scalars (appended to CNN embedding):
    H_real, D_real     inertia and damping constants from the HD pipeline [CSV]

Architecture
------------
  1D-CNN encodes the two-channel time series into a 16-dim embedding.
  H_real and D_real are appended to that embedding.
  A small MLP maps the combined vector to H_v and D_v.

  Conv1d(2->16, kernel=15) -> ReLU -> GlobalAvgPool -> (16,)
  concat([16, H_real, D_real])      ->               (18,)
  Linear(18->8) -> ReLU -> Linear(8->2) -> Sigmoid
  H_v = sigmoid[0] x 2.0 s
  D_v = sigmoid[1] x 1.0 pu

Differentiable physics layer
-----------------------------
  Transfer-function ODE  G(s) = Δf_stab / Δf_raw, discretised with
  Forward Euler.  ΔP cancels in G(s) — no power measurement needed.

    2·H_eff · dΔf_stab/dt = 2·H_real · dΔf_raw/dt
                            + D_real  · Δf_raw
                            - D_eff   · Δf_stab

  where H_eff = H_real + H_v,  D_eff = D_real + D_v.
  RoCoF (channel 1) provides dΔf_raw/dt directly.

Physics loss
------------
  ISE + injection cost:

    L = mean( sum_t( Δf_stab(t)^2 · dt ) )  +  λ · mean( H_v/H_MAX + D_v/D_MAX )

Evaluation
----------
  Leave-One-Out Cross-Validation (LOO-CV) over 20 events.
  All normalisation is fit on the 19 training events of each fold.
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

from scipy.signal import savgol_filter
from scipy.stats import t as _t_dist
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT     = Path(__file__).parent.parent
CSV_PATH = ROOT / "hd_pipeline" / "hd_results_2016.csv"
SEG_DIR  = ROOT / "event_segments_full"
PLOT_DIR = ROOT / "plots"
RES_DIR  = ROOT / "results"
PLOT_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Physics constants
# ---------------------------------------------------------------------------
DT        = 0.1   # PMU sampling interval [s]
HV_MAX    = 2.0   # maximum virtual inertia injection [s]
DV_MAX    = 1.0   # maximum virtual damping injection [pu]
LAM_H     = 0.1   # inertia injection cost weight
LAM_D     = 0.5   # damping injection cost weight (higher than LAM_H — penalises D_v saturation)
BASELINE_HV = 1.0  # fixed-injection baseline: H_v [s]  (midpoint of [0, HV_MAX])
BASELINE_DV = 0.5  # fixed-injection baseline: D_v [pu] (midpoint of [0, DV_MAX])
SG_WINDOW = 51    # Savitzky-Golay window for RoCoF (5.1 s — matches hd_pipeline.py)
SG_ORDER  = 3     # Savitzky-Golay polynomial order

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
N_EPOCHS  = 500
LR        = 5e-4
GRAD_CLIP = 1.0
SEED      = 42
T_WIN     = 1500  # ODE window: 150 s at 10 Hz  (full event, physics loss)
T_PRED    = 20    # CNN input window: 2 s at 10 Hz  (what the model sees to predict)
PRE_IDX   = 50    # onset index in df_full (5 s pre-onset at 10 Hz)


def _rocof(df: np.ndarray) -> np.ndarray:
    """Savitzky-Golay RoCoF — matches hd_pipeline.py exactly."""
    return savgol_filter(df, window_length=SG_WINDOW, polyorder=SG_ORDER,
                         deriv=1, delta=DT)


# ===========================================================================
# 1. DATA LOADING
# ===========================================================================

def load_data() -> list[dict]:
    df_csv    = pd.read_csv(CSV_PATH)

    def _key(ts: str) -> str:
        return (ts.replace("T", " ").split(".")[0]
                  .replace("-", "").replace(":", "").replace(" ", "_"))

    csv_index = {_key(r["onset_time"]): r for _, r in df_csv.iterrows()}

    events = []
    for npz_file in sorted(SEG_DIR.glob("*.npz")):
        key = npz_file.stem
        if key not in csv_index:
            continue
        row   = csv_index[key]
        seg   = np.load(npz_file, allow_pickle=True)
        hfull = "df_full" in seg.files

        events.append({
            "onset_str": key,
            "direction": 1.0 if row["direction"] == "over" else -1.0,
            "cause":     row["cause"],
            "H_real":    float(row["H_pw_mean"]),
            "D_real":    float(row["D_pw_mean"]),
            "df_train":  seg["df_train"].astype(np.float32),
            "df_full":   seg["df_full"].astype(np.float32)  if hfull else None,
            "t_full":    seg["t_full"].astype(np.float32)   if hfull else None,
        })

    print(f"Loaded {len(events)} raw events.")
    valid   = [ev for ev in events if not np.isnan(ev["H_real"])]
    dropped = [ev["onset_str"] for ev in events if np.isnan(ev["H_real"])]
    if dropped:
        print(f"  Dropped {len(dropped)} event(s): {dropped}")
    print(f"  Using {len(valid)} events.\n")
    return valid


# ===========================================================================
# 2. BUILD INPUT ARRAYS
# ===========================================================================

def build_arrays(events: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    X_ts_cnn : (N, 2, T_PRED)  CNN input — first T_PRED samples, normalised later
    X_ts_ode : (N, 2, T_WIN)   ODE input — full 150 s window, raw physical units
    X_ctx    : (N, 2)           scalars   — [H_real, D_real]

    RoCoF is computed on-the-fly with Savitzky-Golay from the full df window
    so that both arrays use the same smooth derivative — consistent with hd_pipeline.py.
    The first T_PRED values of that rocof go to the CNN; all T_WIN go to the ODE.
    """
    N        = len(events)
    X_ts_cnn = np.zeros((N, 2, T_PRED), dtype=np.float32)
    X_ts_ode = np.zeros((N, 2, T_WIN),  dtype=np.float32)
    X_ctx    = np.zeros((N, 2),          dtype=np.float32)

    for i, ev in enumerate(events):
        df    = ev["df_full"][PRE_IDX:PRE_IDX + T_WIN].astype(np.float64)
        rocof = _rocof(df)
        # ODE gets the full window (raw physical units)
        X_ts_ode[i, 0, :] = df
        X_ts_ode[i, 1, :] = rocof
        # CNN gets only the first T_PRED samples
        X_ts_cnn[i, 0, :] = df[:T_PRED]
        X_ts_cnn[i, 1, :] = rocof[:T_PRED]

        X_ctx[i] = [ev["H_real"], ev["D_real"]]

    print(f"CNN input  : {X_ts_cnn.shape}  (N x 2 x T_PRED  = {T_PRED*DT:.0f} s)")
    print(f"ODE input  : {X_ts_ode.shape}  (N x 2 x T_WIN   = {T_WIN*DT:.0f} s)")
    print(f"Context    : {X_ctx.shape}  (N x [H_real, D_real])")
    return X_ts_cnn, X_ts_ode, X_ctx


# ===========================================================================
# 3. DIFFERENTIABLE VI FILTER  (physics layer)
# ===========================================================================

def vi_filter_torch(
    df_raw:    torch.Tensor,   # (B, T)  Delta_f in physical units [Hz]
    rocof_raw: torch.Tensor,   # (B, T)  d(Delta_f)/dt  [Hz/s]
    H_real:    torch.Tensor,   # (B,)    system inertia constant [s]
    D_real:    torch.Tensor,   # (B,)    system damping constant [pu]
    H_v:       torch.Tensor,   # (B,)    virtual inertia injection [s]
    D_v:       torch.Tensor,   # (B,)    virtual damping injection [pu]
) -> torch.Tensor:
    """
    Differentiable VI filter — Forward Euler discretisation of the
    transfer-function ODE  G(s) = Δf_stab / Δf_raw.

    ΔP cancels in G(s), so no power measurement is needed:

        2·H_eff · dΔf_stab/dt = 2·H_real · dΔf_raw/dt
                                + D_real  · Δf_raw
                                - D_eff   · Δf_stab

    Δf_raw and its derivative (RoCoF, channel 1 of the input) drive the ODE.
    Gradients flow back through every timestep to the CNN weights.
    """
    H_eff  = (H_real + H_v).clamp(min=0.1)
    D_eff  = (D_real + D_v).clamp(min=0.0)
    inv_2H = 1.0 / (2.0 * H_eff)

    stab = [df_raw[:, 0:1]]
    for t in range(1, df_raw.shape[1]):
        rhs  = (2.0 * H_real[:, None] * rocof_raw[:, t:t+1]
                + D_real[:, None] * df_raw[:, t:t+1]
                - D_eff[:, None]  * stab[-1])
        stab.append(stab[-1] + DT * rhs * inv_2H[:, None])

    return torch.cat(stab, dim=1)


# ===========================================================================
# 4. MODEL
# ===========================================================================

class PhysicsPINN(nn.Module):
    """
    Single-branch physics-informed neural network.

    The CNN encodes the shape of the frequency disturbance from the two
    time-series channels (Delta_f and RoCoF).  H_real and D_real are
    appended to the CNN embedding to provide the grid parameter scale
    that the waveform alone cannot determine.
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(2, 16, kernel_size=15, padding=7)
        self.gap  = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(16 + 2, 8),   # 16 CNN embedding + H_real + D_real
            nn.ReLU(),
            nn.Linear(8, 2),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x_ts:  torch.Tensor,   # (B, 2, T)  z-scored [Delta_f, RoCoF]
        x_ctx: torch.Tensor,   # (B, 2)     z-scored [H_real, D_real]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.gap(torch.relu(self.conv(x_ts))).squeeze(-1)  # (B, 16)
        out = self.head(torch.cat([emb, x_ctx], dim=1))          # (B, 2)
        return out[:, 0] * HV_MAX, out[:, 1] * DV_MAX


# ===========================================================================
# 5. PHYSICS LOSS
# ===========================================================================

def physics_loss(
    df_stab: torch.Tensor,   # (B, T)  output of vi_filter_torch
    H_v:     torch.Tensor,   # (B,)
    D_v:     torch.Tensor,   # (B,)
) -> torch.Tensor:
    """
    ISE + injection cost:

    L = mean( sum_t( df_stab[t]^2 * dt ) )
      + mean( LAM_H * H_v/HV_MAX + LAM_D * D_v/DV_MAX )
    """
    ise  = (df_stab ** 2).sum(dim=1).mean() * DT
    cost = (LAM_H * H_v / HV_MAX + LAM_D * D_v / DV_MAX).mean()
    return ise + cost


# ===========================================================================
# 6. LOO-CV TRAINING
# ===========================================================================

def run_loocv(
    events:    list[dict],
    X_ts_cnn:  np.ndarray,   # (N, 2, T_PRED) — CNN input, will be normalised
    X_ts_ode:  np.ndarray,   # (N, 2, T_WIN)  — ODE input, raw physical units
    X_ctx:     np.ndarray,   # (N, 2)          — [H_real, D_real]
) -> dict:
    """
    Leave-One-Out Cross-Validation.

    CNN sees only the first T_PRED samples (T_PRED*DT seconds) of each event.
    ODE runs on the full T_WIN samples (150 s) for the physics loss.
    This trains an oracle: the CNN learns the early fingerprint that predicts
    the optimal injection for the full event trajectory.

    All normalisation is fit on the 19 training events per fold.
    """
    print(f"\n--- LOO-CV  ({N_EPOCHS} epochs/fold) ---")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    N           = len(events)
    p_Hv        = np.zeros(N)
    p_Dv        = np.zeros(N)
    loss_curves = []

    for i in range(N):
        tr = np.ones(N, dtype=bool); tr[i] = False

        # Normalise CNN input (T_PRED window) — z-score per channel, fit on training fold
        ts_tr = X_ts_cnn[tr].copy()
        ts_te = X_ts_cnn[[i]].copy()
        for ch in range(2):
            mu  = ts_tr[:, ch, :].mean()
            sig = ts_tr[:, ch, :].std() + 1e-8
            ts_tr[:, ch, :] = (ts_tr[:, ch, :] - mu) / sig
            ts_te[:, ch, :] = (ts_te[:, ch, :] - mu) / sig

        # Normalise H_real, D_real — fit on training fold
        scaler   = StandardScaler()
        ctx_tr_n = scaler.fit_transform(X_ctx[tr]).astype(np.float32)
        ctx_te_n = scaler.transform(X_ctx[[i]]).astype(np.float32)

        # Tensors for the model (normalised CNN window)
        ts_tr_t  = torch.tensor(ts_tr,    dtype=torch.float32)
        ts_te_t  = torch.tensor(ts_te,    dtype=torch.float32)
        ctx_tr_t = torch.tensor(ctx_tr_n, dtype=torch.float32)
        ctx_te_t = torch.tensor(ctx_te_n, dtype=torch.float32)

        # Raw tensors for the ODE (full 150 s, physical units — NOT normalised)
        df_raw_t    = torch.tensor(X_ts_ode[tr, 0, :], dtype=torch.float32)
        rocof_raw_t = torch.tensor(X_ts_ode[tr, 1, :], dtype=torch.float32)
        H_real_t    = torch.tensor(X_ctx[tr, 0],        dtype=torch.float32)
        D_real_t    = torch.tensor(X_ctx[tr, 1],        dtype=torch.float32)

        model = PhysicsPINN()
        opt   = torch.optim.Adam(model.parameters(), lr=LR)

        fold_losses = []
        for _ in range(N_EPOCHS):
            model.train()
            opt.zero_grad()
            H_v, D_v = model(ts_tr_t, ctx_tr_t)
            df_stab  = vi_filter_torch(df_raw_t, rocof_raw_t, H_real_t, D_real_t, H_v, D_v)
            loss     = physics_loss(df_stab, H_v, D_v)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            fold_losses.append(float(loss))

        loss_curves.append(fold_losses)

        model.eval()
        with torch.no_grad():
            Hv_te, Dv_te = model(ts_te_t, ctx_te_t)
        p_Hv[i] = float(Hv_te[0].clamp(0, HV_MAX))
        p_Dv[i] = float(Dv_te[0].clamp(0, DV_MAX))

        if (i + 1) % 5 == 0 or i == N - 1:
            print(f"  fold {i+1:2d}/{N}  loss={fold_losses[-1]:.5f}"
                  f"  Hv={p_Hv[i]:.3f}s  Dv={p_Dv[i]:.3f}pu")

    return dict(p_Hv=p_Hv, p_Dv=p_Dv, loss_curves=loss_curves)


# ===========================================================================
# 7. EVALUATION  (numpy VI filter with Savitzky-Golay for accuracy)
# ===========================================================================

def apply_vi_filter_np(
    df_raw: np.ndarray, rocof: np.ndarray, t_arr: np.ndarray,
    H_real: float, D_real: float,
    H_v: float, D_v: float,
) -> np.ndarray:
    """
    Numpy VI filter — Forward Euler integration of the G(s) transfer-function ODE.

    ODE:  2·H_eff · dΔf_stab/dt = 2·H_real · dΔf_raw/dt
                                  + D_real  · Δf_raw
                                  - D_eff   · Δf_stab

    No ΔP needed — it cancels in the ratio Δf_stab / Δf_raw.
    """
    H_eff = max(H_real + H_v, 0.1)
    D_eff = max(D_real + D_v, 0.0)
    n     = len(t_arr)
    df_st    = np.zeros(n)
    df_st[0] = df_raw[0]
    for k in range(1, n):
        dt_k     = float(t_arr[k] - t_arr[k - 1])
        rhs      = 2.0*H_real*rocof[k] + D_real*df_raw[k] - D_eff*df_st[k-1]
        df_st[k] = df_st[k-1] + dt_k * rhs / (2.0 * H_eff)
    return df_st.astype(np.float32)


def evaluate_vi(events: list[dict], res: dict) -> pd.DataFrame:
    rows = []
    for i, ev in enumerate(events):
        H_real  = ev["H_real"]
        D_real  = ev["D_real"]
        Hv_pred = float(res["p_Hv"][i])
        Dv_pred = float(res["p_Dv"][i])

        if ev["df_full"] is not None:
            t_arr   = ev["t_full"].astype(np.float64)
            df_raw  = ev["df_full"].astype(np.float64)
            onset   = int(np.searchsorted(t_arr, 0.0))
            t_post  = t_arr[onset:]
            df_post = df_raw[onset:]
        else:
            t_post  = np.arange(len(ev["df_train"])) * DT
            df_post = ev["df_train"].astype(np.float64)

        rocof_post = _rocof(df_post)

        # AI prediction
        df_ai          = apply_vi_filter_np(df_post, rocof_post, t_post, H_real, D_real,
                                            Hv_pred, Dv_pred)
        # Fixed baseline
        df_base        = apply_vi_filter_np(df_post, rocof_post, t_post, H_real, D_real,
                                            BASELINE_HV, BASELINE_DV)

        nadir_mag_raw  = float(np.max(np.abs(df_post)))
        nadir_mag_ai   = float(np.max(np.abs(df_ai)))
        nadir_mag_base = float(np.max(np.abs(df_base)))
        rocof_mag_raw  = float(np.max(np.abs(np.diff(df_post) / DT)))
        rocof_mag_ai   = float(np.max(np.abs(np.diff(df_ai)   / DT)))
        rocof_mag_base = float(np.max(np.abs(np.diff(df_base)  / DT)))

        nadir_imp_ai   = (nadir_mag_raw - nadir_mag_ai)   / nadir_mag_raw * 100
        nadir_imp_base = (nadir_mag_raw - nadir_mag_base) / nadir_mag_raw * 100
        rocof_imp_ai   = (rocof_mag_raw - rocof_mag_ai)   / rocof_mag_raw * 100
        rocof_imp_base = (rocof_mag_raw - rocof_mag_base) / rocof_mag_raw * 100

        rows.append(dict(
            event              = ev["onset_str"],
            H_real             = round(H_real,  3),
            D_real             = round(D_real,  3),
            Hv_pred            = round(Hv_pred, 3),
            Dv_pred            = round(Dv_pred, 3),
            nadir_raw          = round(nadir_mag_raw,   4),
            nadir_stab         = round(nadir_mag_ai,    4),
            nadir_imp_pct      = round(nadir_imp_ai,    1),
            nadir_imp_base_pct = round(nadir_imp_base,  1),
            rocof_raw          = round(rocof_mag_raw,   4),
            rocof_stab         = round(rocof_mag_ai,    4),
            rocof_imp_pct      = round(rocof_imp_ai,    1),
            rocof_imp_base_pct = round(rocof_imp_base,  1),
        ))

    df = pd.DataFrame(rows)

    # ── per-event table ──────────────────────────────────────────────────
    print("\n===== Results =====")
    print(df[["event", "H_real", "D_real", "Hv_pred", "Dv_pred",
              "nadir_imp_pct", "rocof_imp_pct"]].to_string(index=False))

    # ── summary: AI vs Fixed baseline with 95% CI ────────────────────────
    N      = len(df)
    t_crit = float(_t_dist.ppf(0.975, df=N - 1))

    def _ci(col: str) -> str:
        mu  = df[col].mean()
        sem = df[col].std() / np.sqrt(N)
        return f"{mu:.1f}% ± {t_crit * sem:.1f}%"

    print(f"\n  H_v : mean={res['p_Hv'].mean():.3f} s   std={res['p_Hv'].std():.3f}")
    print(f"  D_v : mean={res['p_Dv'].mean():.3f} pu  std={res['p_Dv'].std():.3f}")
    print(f"\n  {'Metric':<22} {'AI (Hv adaptive)':<26} {'Fixed (Hv=1.0,Dv=0.5)':<26} {'Gain'}")
    print(f"  {'-'*22} {'-'*25} {'-'*25} {'-'*10}")

    for label, ai_col, base_col in [
        ("Nadir improvement",  "nadir_imp_pct",  "nadir_imp_base_pct"),
        ("RoCoF improvement",  "rocof_imp_pct",  "rocof_imp_base_pct"),
    ]:
        ai_mean   = df[ai_col].mean()
        base_mean = df[base_col].mean()
        gain      = ai_mean - base_mean
        print(f"  {label:<22} {_ci(ai_col):<26} {_ci(base_col):<26} {gain:+.1f}%")

    return df


# ===========================================================================
# 8. PLOTS
# ===========================================================================

def plot_loss_curves(res: dict) -> None:
    _, ax = plt.subplots(figsize=(8, 4))
    for c in res["loss_curves"]:
        ax.plot(c, alpha=0.35, lw=0.8, color="#2980b9")
    ax.plot(np.mean(res["loss_curves"], axis=0), color="black", lw=2,
            label="Mean across folds")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Physics loss")
    ax.set_title("PhysicsPINN — training loss per LOO fold")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = PLOT_DIR / "v12_150s_loss_curves.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p}")


def plot_vi_stabilisation(events: list[dict], res: dict) -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")

    n     = len(events)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.0 * nrows))
    axes = np.array(axes).flatten()

    for i, ev in enumerate(events):
        ax      = axes[i]
        H_real  = ev["H_real"]
        D_real  = ev["D_real"]
        Hv_pred = float(res["p_Hv"][i])
        Dv_pred = float(res["p_Dv"][i])

        if ev["df_full"] is not None:
            t_arr   = ev["t_full"].astype(np.float64)
            df_raw  = ev["df_full"].astype(np.float64)
            onset   = int(np.searchsorted(t_arr, 0.0))
            t_post  = t_arr[onset:]
            df_post = df_raw[onset:]
        else:
            t_arr   = np.arange(len(ev["df_train"])) * DT
            df_raw  = ev["df_train"].astype(np.float64)
            df_post = df_raw
            t_post  = t_arr

        rocof_post = _rocof(df_post)
        df_filt    = apply_vi_filter_np(df_post, rocof_post, t_post,
                                        H_real, D_real, Hv_pred, Dv_pred)
        df_stab    = np.concatenate([df_raw[:onset], df_filt]) \
                     if ev["df_full"] is not None else df_filt

        nadir_imp = (np.max(np.abs(df_post)) - np.max(np.abs(df_filt))) \
                    / np.max(np.abs(df_post)) * 100
        rocof_imp = (np.max(np.abs(np.diff(df_post) / DT))
                     - np.max(np.abs(np.diff(df_filt) / DT))) \
                    / np.max(np.abs(np.diff(df_post) / DT)) * 100

        ax.set_facecolor("#f0f0f0")
        ax.plot(t_arr, df_raw,  color="#1f77b4", lw=1.4)
        ax.plot(t_arr, df_stab, color="#2980b9", lw=1.2, ls="--")
        ax.axvline(0, color="red", lw=0.8, ls="--", alpha=0.6)
        info = (f"Hv={Hv_pred:.2f}s  Dv={Dv_pred:.2f}pu\n"
                f"nadir -{nadir_imp:.0f}%  RoCoF -{rocof_imp:.0f}%")
        ax.text(0.98, 0.97, info, transform=ax.transAxes, fontsize=6.5,
                ha="right", va="top", family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))
        dir_sym = "up" if ev["direction"] > 0 else "down"
        ax.set_title(f"{ev['onset_str'][:8]}  {ev['cause']} ({dir_sym})",
                     fontsize=8.5, fontweight="bold")
        ax.set_xlabel("Time [s]", fontsize=8)
        ax.set_ylabel("Delta f [Hz]", fontsize=8)
        ax.tick_params(labelsize=7)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.legend(handles=[
        plt.Line2D([0], [0], color="#1f77b4", lw=2,        label="Raw PMU Delta_f"),
        plt.Line2D([0], [0], color="#2980b9", lw=1.5, ls="--", label="Stabilised Delta_f"),
    ], loc="lower right", fontsize=9, ncol=2)
    fig.suptitle(
        "PhysicsPINN — Frequency Stabilisation (LOO-CV)\n"
        "Inputs: Delta_f(t), RoCoF(t), H_real, D_real  |  "
        "Loss: differentiable swing-equation ODE",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    p = PLOT_DIR / "v12_150s_vi_stabilisation.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p}")


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    print("=" * 65)
    print("  ai_models_v12.py — PhysicsPINN")
    print("=" * 65 + "\n")

    events                    = load_data()
    X_ts_cnn, X_ts_ode, X_ctx = build_arrays(events)

    res = run_loocv(events, X_ts_cnn, X_ts_ode, X_ctx)

    df_vi = evaluate_vi(events, res)
    df_vi.to_csv(RES_DIR / "v12_150s_vi_metrics.csv", index=False)
    print(f"\n  Saved: {RES_DIR / 'v12_150s_vi_metrics.csv'}")

    plot_loss_curves(res)
    plot_vi_stabilisation(events, res)

    print("\nDone.")


if __name__ == "__main__":
    main()
