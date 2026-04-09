"""
ai_models_v3_integrated.py
--------------------------
Full Integrated Script: Physics-AI (v3) + Signal Augmentation Plotting.
Saves results to results_v3.csv and creates full trajectory comparison plots.
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

warnings.filterwarnings("ignore")

# --- Constants ---
ROOT = Path(__file__).parent
CSV_PATH = ROOT / "hd_results_2016.csv"
SEG_DIR = ROOT / "event_segments"
PLOT_DIR = ROOT / "plots"
PLOT_DIR.mkdir(exist_ok=True)

DT, F0 = 0.1, 50.0
H_MIN, H_MAX = 2.0, 10.0
D_MIN, D_MAX = 0.1, 5.0

# ===========================================================================
# 1. CORE UTILITIES & DATA LOADING
# ===========================================================================

def load_data() -> list[dict]:
    df_csv = pd.read_csv(CSV_PATH)
    def _key(ts: str): return ts.replace("T", " ").split(".")[0].replace("-", "").replace(":", "").replace(" ", "_")
    csv_index = {_key(row["onset_time"]): row for _, row in df_csv.iterrows()}
    events = []
    for npz_file in sorted(SEG_DIR.glob("*.npz")):
        key = npz_file.stem
        if key not in csv_index: continue
        row, seg = csv_index[key], np.load(npz_file, allow_pickle=True)
        events.append({
            "onset_str": key, "dP_pu": float(row["dP_pu"]), "direction": 1.0 if row["direction"] == "over" else -1.0,
            "H_pw_mean": float(row["H_pw_mean"]), "D_pw_mean": float(row["D_pw_mean"]),
            "df_train": seg["df_train"].astype(np.float32), "rocof_train": seg["rocof_train"].astype(np.float32),
            "df_full": seg["df_full"].astype(np.float32) if "df_full" in seg.files else None,
            "t_full": seg["t_full"].astype(np.float32) if "t_full" in seg.files else None,
        })
    return [ev for ev in events if not np.isnan(ev["H_pw_mean"])]

def engineer_features(events: list[dict]) -> np.ndarray:
    rows = []
    for ev in events:
        df, rocof, dP = ev["df_train"], ev["rocof_train"], abs(ev["dP_pu"])
        rocof_init = max(float(np.mean(np.abs(rocof[:20]))), 1e-6)
        df_nadir = max(float(np.max(np.abs(df))), 1e-6)
        rows.append([rocof_init, df_nadir, dP, dP*F0/(2*rocof_init), dP*F0/df_nadir])
    return np.array(rows, dtype=np.float32)

# ===========================================================================
# 2. PHYSICS MODELS (GPR, SVR, CNN)
# ===========================================================================

def run_gpr(X, y_H, y_D):
    N = len(y_H)
    p_H, p_D = np.zeros(N), np.zeros(N)
    kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(1e-3)
    for i in range(N):
        tr = np.ones(N, dtype=bool); tr[i] = False
        sx = StandardScaler().fit(X[tr])
        gH = GaussianProcessRegressor(kernel=kernel, normalize_y=True).fit(sx.transform(X[tr]), y_H[tr])
        gD = GaussianProcessRegressor(kernel=kernel, normalize_y=True).fit(sx.transform(X[tr]), y_D[tr])
        p_H[i], p_D[i] = gH.predict(sx.transform(X[[i]]))[0], gD.predict(sx.transform(X[[i]]))[0]
    return {"name": "GPR", "p_H": np.clip(p_H, H_MIN, H_MAX), "p_D": np.clip(p_D, D_MIN, D_MAX)}

def run_svr(X, y_H, y_D):
    N = len(y_H)
    p_H, p_D = np.zeros(N), np.zeros(N)
    grid = {"C": [1, 10, 100], "gamma": ["scale", 0.1]}
    for i in range(N):
        tr = np.ones(N, dtype=bool); tr[i] = False
        sx = StandardScaler().fit(X[tr])
        gsH = GridSearchCV(SVR(), grid, cv=3).fit(sx.transform(X[tr]), y_H[tr])
        gsD = GridSearchCV(SVR(), grid, cv=3).fit(sx.transform(X[tr]), y_D[tr])
        p_H[i], p_D[i] = gsH.predict(sx.transform(X[[i]]))[0], gsD.predict(sx.transform(X[[i]]))[0]
    return {"name": "SVR", "p_H": np.clip(p_H, H_MIN, H_MAX), "p_D": np.clip(p_D, D_MIN, D_MAX)}

# ===========================================================================
# 3. AUGMENTATION & PLOTTING ENGINE
# ===========================================================================

def size_virtual_inertia(H_pred, D_pred, dP_pu, df_target=0.4, rocof_target=0.8):
    dP = abs(dP_pu)
    H_virt = max(0.3, (dP * F0 / (2.0 * rocof_target)) - H_pred)
    D_virt = max(0.2, (dP * F0 / df_target) - D_pred)
    return H_virt, D_virt

def augment_trajectory(df_t, H_pred, D_pred, H_virt, D_virt):
    alpha = (H_pred + H_virt) / H_pred
    beta = D_pred / (D_pred + D_virt)
    t = np.arange(len(df_t)) * DT
    return np.interp(t / alpha, t, df_t) * beta

def plot_full_signal_comparison(events, best_res):
    N = len(events)
    fig, axes = plt.subplots((N+3)//4, 4, figsize=(18, 3.5 * ((N+3)//4)))
    axes = axes.flatten()
    
    for i, ev in enumerate(events):
        ax = axes[i]
        H_p, D_p = best_res["p_H"][i], best_res["p_D"][i]
        Hv, Dv = size_virtual_inertia(H_p, D_p, ev["dP_pu"])
        
        if ev["df_full"] is not None:
            t_arr, df_arr = ev["t_full"], ev["df_full"]
            onset = np.searchsorted(t_arr, 0.0)
            df_aug = np.concatenate([df_arr[:onset], augment_trajectory(df_arr[onset:], H_p, D_p, Hv, Dv)])
        else:
            t_arr, df_arr = np.arange(300)*DT, ev["df_train"]
            df_aug = augment_trajectory(df_arr, H_p, D_p, Hv, Dv)

        ax.plot(t_arr, df_arr, color="gray", alpha=0.5, label="Original")
        ax.plot(t_arr, df_aug, color="blue", lw=1.5, label="Augmented")
        ax.set_title(f"Event: {ev['onset_str'][:8]}", fontsize=9)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "full_signal_comparison_v3.png", dpi=150)
    print(f"[INFO] Plot saved to {PLOT_DIR}/full_signal_comparison_v3.png")

# ===========================================================================
# 4. EXECUTION
# ===========================================================================

if __name__ == "__main__":
    print("Step 1: Loading data...")
    events = load_data()
    X = engineer_features(events)
    y_H = np.array([ev["H_pw_mean"] for ev in events])
    y_D = np.array([ev["D_pw_mean"] for ev in events])

    print("Step 2: Training v3 Models (LOO-CV)...")
    res_gpr = run_gpr(X, y_H, y_D)
    res_svr = run_svr(X, y_H, y_D)
    
    # Simple Ensemble for v3
    ens_H = (res_gpr["p_H"] + res_svr["p_H"]) / 2
    ens_D = (res_gpr["p_D"] + res_svr["p_D"]) / 2
    res_ens = {"name": "Ensemble", "p_H": ens_H, "p_D": ens_D}

    print("Step 3: Saving results_v3.csv...")
    output = pd.DataFrame({
        "event": [ev["onset_str"] for ev in events],
        "H_actual": y_H, "D_actual": y_D,
        "H_pred_ens": ens_H, "D_pred_ens": ens_D
    })
    output.to_csv("results_v3.csv", index=False)

    print("Step 4: Generating Trajectory Plots...")
    plot_full_signal_comparison(events, res_ens)
    print("Done.")