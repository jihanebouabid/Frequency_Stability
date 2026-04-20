#!/usr/bin/env python3
"""
hd_pipeline.py
==============
Nordic Grid H and D Estimation Pipeline — full automated run.
Processes all finland_2016_XX.csv monthly PMU files to estimate the Nordic
synchronous grid inertia constant H [s] and load damping coefficient D [pu]
per disturbance event using the swing equation:
2H · d(Δf)/dt + D · Δf = ΔP_pu · f0
Data source: Kangasala 400 kV PMU (Fingrid, 2016, 10 Hz).

Usage
-----
python hd_pipeline.py                 # incremental (skip done months)
python hd_pipeline.py --force         # delete CSV and regenerate all
python hd_pipeline.py --data-dir /path/to/ # override data directory
python hd_pipeline.py --output results.csv # override output filename
python hd_pipeline.py --plots plots_dir    # override plots directory
"""

# ══════════════════════════════════════════════════════════════════════════════
# Section 1 — Imports and constants
# ══════════════════════════════════════════════════════════════════════════════
import os
import glob
import argparse
import sys
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import linregress 
import matplotlib
matplotlib.use('Agg') # non-interactive backend — saves figures to files
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Sampling ──────────────────────────────────────────────────────────────────
DT = 0.1 # sampling interval [s] — Kangasala PMU @ 10 Hz
FS = 10.0 # sampling rate [Hz]
POLYORDER = 3 # SG polynomial order (main filter)
WINDOW_S = 5.0 # SG window length [s]; actual window = 5.1 s (51 samples)

# ── Derived SG window sizes ───────────────────────────────────────────────────
_HALF = round(WINDOW_S * FS / 2) # 25 half-samples
WLEN = 2 * _HALF + 1 # 51 samples = 5.1 s (main SG)
SG_HALF_S = (WLEN // 2) * DT # 2.5 s half-window
_HALF_SH = round(0.55 * FS) # 5 half-samples
WLEN_SH = 2 * _HALF_SH + 1 # 11 samples = 1.1 s (short SG for RoCoF)
POLY_SH = 5 # polynomial order for short SG

# ── Gap handling ──────────────────────────────────────────────────────────────
EXCL_BUFFER = 120 # [s] exclusion zone on each side of moderate/major gaps

# ── Event detection thresholds ────────────────────────────────────────────────
FREQ_THRESH = 0.2 # [Hz] FCR-D activation boundary (±0.2 Hz)
MIN_DURATION_S = 5.0 # [s] FCR-D Full Activation Time; threshold must sustain
MIN_EVENT_GAP_S = 1200 # [s] minimum separation between distinct events (20 min)
PRE_SEARCH_S = 30 # [s] look-back window before crossing for onset refinement
POST_WINDOW_S = 300 # [s] look-forward window for nadir and recovery detection
ROCOF_LAG_MAX_S = 10 # [s] maximum lag from onset to peak RoCoF
MAX_RECOVERY_S = 60 # [s] max time from nadir to 50 % recovery (standard)
SIGMA_CHOICE = 3 # σ multiplier for RoCoF onset threshold

# ── January special case ──────────────────────────────────────────────────────
JAN_MAX_RECOVERY_S = 90 # [s] relaxed recovery window for January only

# ── Physics and system base ───────────────────────────────────────────────────
S_BASE_MW = 55000.0 # [MW] Nordic synchronous load — S_base = 55 000 MW 
F0 = 50.0 # [Hz] nominal grid frequency

# ── Estimation windows ────────────────────────────────────────────────────────
PRE_ONSET_S = 30.0 # [s] pre-onset window for Δf_pre baseline
D_PW_SKIP_S = 2.5 # [s] skip SG half-window artefact (= SG_HALF_S)
D_PW_MIN_DF_HZ = 0.05 # [Hz] minimum |Δf| for a D_pointwise sample
H_PW_END_S = 3.0 # [s] H_pointwise window length after onset
H_PW_MIN_ROCOF = 0.030 # [Hz/s] minimum |RoCoF| for an H_pointwise sample

# ── Matching ──────────────────────────────────────────────────────────────────
MATCH_TOL_S = 120 # [s] tolerance for report-event ↔ detected-segment matching

# ── Validation expected ranges ────────────────────────────────────────────────
VALID_H_EK_RANGE = (2.0, 6.0) # [s] Nordic H_from_Ek
VALID_D_NAD_RANGE = (1.0, 4.0) # [pu] D_nadir
VALID_D_PW_RANGE = (0.5, 3.5) # [pu] D_pw_mean
VALID_H_PW_RANGE = (1.5, 8.0) # [s] H_pw_mean

# ── Report events (ENTSO-E / Fingrid 2016 FCR disturbance reports) ────────────
_REPORT_EVENTS_RAW = [
    {"date_str": "2016-01-12 10:22:53", "delta_f_Hz": -0.335, "delta_P_MW": 1167, "delta_t_s": 9.0, "Ek_GWs": 274, "cause": "Nuclear"},
    {"date_str": "2016-02-09 22:05:28", "delta_f_Hz": 0.214, "delta_P_MW": 880, "delta_t_s": 8.3, "Ek_GWs": 252, "cause": "HVDC"},
    {"date_str": "2016-02-20 10:45:31", "delta_f_Hz": -0.348, "delta_P_MW": 1000, "delta_t_s": 10.7, "Ek_GWs": 227, "cause": "Nuclear"},
    {"date_str": "2016-02-24 12:30:29", "delta_f_Hz": 0.228, "delta_P_MW": 700, "delta_t_s": 9.0, "Ek_GWs": 251, "cause": "HVDC"},
    {"date_str": "2016-03-21 11:59:50", "delta_f_Hz": 0.269, "delta_P_MW": 723, "delta_t_s": 7.3, "Ek_GWs": 248, "cause": "HVDC"},
    {"date_str": "2016-04-06 13:49:55", "delta_f_Hz": 0.230, "delta_P_MW": 700, "delta_t_s": 6.7, "Ek_GWs": 221, "cause": "HVDC"},
    {"date_str": "2016-04-14 10:07:07", "delta_f_Hz": 0.234, "delta_P_MW": 700, "delta_t_s": 6.7, "Ek_GWs": 240, "cause": "HVDC"},
    {"date_str": "2016-04-20 09:31:18", "delta_f_Hz": 0.249, "delta_P_MW": 700, "delta_t_s": 7.1, "Ek_GWs": 224, "cause": "HVDC"},
    {"date_str": "2016-04-21 11:43:42", "delta_f_Hz": 0.283, "delta_P_MW": 700, "delta_t_s": 7.9, "Ek_GWs": 222, "cause": "HVDC"},
    {"date_str": "2016-05-15 01:56:13", "delta_f_Hz": 0.339, "delta_P_MW": 720, "delta_t_s": 7.0, "Ek_GWs": 162, "cause": "HVDC"},
    {"date_str": "2016-05-18 22:08:27", "delta_f_Hz": 0.266, "delta_P_MW": 700, "delta_t_s": 6.3, "Ek_GWs": 205, "cause": "HVDC"},
    {"date_str": "2016-06-20 17:10:09", "delta_f_Hz": -0.389, "delta_P_MW": 700, "delta_t_s": 8.8, "Ek_GWs": 175, "cause": "AC-line"},
    {"date_str": "2016-06-25 22:28:59", "delta_f_Hz": 0.347, "delta_P_MW": 600, "delta_t_s": 7.1, "Ek_GWs": 154, "cause": "HVDC"},
    {"date_str": "2016-06-27 05:01:25", "delta_f_Hz": -0.263, "delta_P_MW": 400, "delta_t_s": 8.9, "Ek_GWs": 131, "cause": "Industry"},
    {"date_str": "2016-07-04 14:41:01", "delta_f_Hz": -0.454, "delta_P_MW": 1026, "delta_t_s": 9.0, "Ek_GWs": 192, "cause": "AC-line"},
    {"date_str": "2016-07-12 14:42:29", "delta_f_Hz": 0.322, "delta_P_MW": 670, "delta_t_s": 8.0, "Ek_GWs": 189, "cause": "HVDC"},
    {"date_str": "2016-07-26 22:52:43", "delta_f_Hz": 0.337, "delta_P_MW": 700, "delta_t_s": 7.9, "Ek_GWs": 191, "cause": "HVDC"},
    {"date_str": "2016-08-05 11:08:09", "delta_f_Hz": 0.230, "delta_P_MW": 500, "delta_t_s": 8.4, "Ek_GWs": 188, "cause": "HVDC"},
    {"date_str": "2016-08-12 21:39:46", "delta_f_Hz": -0.313, "delta_P_MW": 500, "delta_t_s": 10.8, "Ek_GWs": 174, "cause": "Nuclear"},
    {"date_str": "2016-08-28 23:06:07", "delta_f_Hz": -0.244, "delta_P_MW": 590, "delta_t_s": 8.4, "Ek_GWs": 180, "cause": "HVDC"},
    {"date_str": "2016-10-03 21:21:17", "delta_f_Hz": 0.290, "delta_P_MW": 700, "delta_t_s": 7.9, "Ek_GWs": 205, "cause": "HVDC"},
    {"date_str": "2016-10-08 23:14:51", "delta_f_Hz": -0.358, "delta_P_MW": 880, "delta_t_s": 9.4, "Ek_GWs": 196, "cause": "Nuclear"},
]

def _build_report_events():
    df = pd.DataFrame(_REPORT_EVENTS_RAW)
    df['report_time'] = pd.to_datetime(df['date_str'])
    df['direction'] = df['delta_f_Hz'].apply(lambda x: 'under' if x < 0 else 'over')
    df['month'] = df['report_time'].dt.to_period('M')
    return df

REPORT_EVENTS = _build_report_events()

# ══════════════════════════════════════════════════════════════════════════════
# Section 2 — Signal loading and gap handling
# ══════════════════════════════════════════════════════════════════════════════
def load_month(filepath):
    df_raw = pd.read_csv(filepath, header=0)
    cols = list(df_raw.columns)
    df = pd.DataFrame()
    df.index = pd.to_datetime(df_raw[cols[0]])
    df.index.name = 'timestamp'
    raw_vals = df_raw[cols[1]].values.astype(float)
    
    col_name = cols[1].lower()
    if 'mhz' in col_name or (np.nanmax(np.abs(raw_vals)) > 10):
        df['delta_f_Hz'] = raw_vals / 1000.0
    else:
        df['delta_f_Hz'] = raw_vals

    def classify_gap(n):
        if n <= 10: return 'minor' 
        elif n <= 100: return 'moderate'
        else: return 'major'

    nan_mask = df['delta_f_Hz'].isna()
    run_id = (nan_mask != nan_mask.shift()).cumsum()
    
    if nan_mask.any():
        nan_runs_info = (
            df[nan_mask]
            .groupby(run_id[nan_mask])
            .agg(
                gap_start=('delta_f_Hz', lambda x: x.index.min()),
                gap_end  =('delta_f_Hz', lambda x: x.index.max()),
                gap_len  =('delta_f_Hz', 'size'),
            )
            .reset_index(drop=True)
        )
        nan_runs_info['severity'] = nan_runs_info['gap_len'].apply(classify_gap)
    else:
        nan_runs_info = pd.DataFrame(columns=['gap_start', 'gap_end', 'gap_len', 'severity'])

    minor_mod_mask = nan_mask.copy()
    for _, row in nan_runs_info[nan_runs_info['severity'] == 'major'].iterrows():
        minor_mod_mask.loc[row['gap_start']:row['gap_end']] = False
        
    df['delta_f_Hz'] = (
        df['delta_f_Hz']
        .where(~minor_mod_mask, other=np.nan)
        .interpolate(method='linear', limit_area='inside')
    )

    df['reliable'] = True
    for _, row in nan_runs_info[nan_runs_info['severity'].isin(['moderate', 'major'])].iterrows():
        buf = pd.Timedelta(seconds=EXCL_BUFFER)
        df.loc[row['gap_start'] - buf : row['gap_end'] + buf, 'reliable'] = False
        
    return df, nan_runs_info

# ══════════════════════════════════════════════════════════════════════════════
# Section 3 — SG filtering
# ══════════════════════════════════════════════════════════════════════════════
def apply_sg_filters(df, nan_runs_info):
    arr = (df['delta_f_Hz']
           .interpolate(method='linear', limit_area='inside')
           .ffill()
           .bfill()
           .values.astype(float))

    df['delta_f_smooth'] = savgol_filter(arr, window_length=WLEN, polyorder=POLYORDER, deriv=0, delta=DT)
    df['rocof_Hz_per_s'] = savgol_filter(arr, window_length=WLEN, polyorder=POLYORDER, deriv=1, delta=DT)
    df['rocof_short_Hz_per_s'] = savgol_filter(arr, window_length=WLEN_SH, polyorder=POLY_SH, deriv=1, delta=DT)

    major_gaps = nan_runs_info[nan_runs_info['severity'] == 'major']
    for _, row in major_gaps.iterrows():
        df.loc[row['gap_start']:row['gap_end'],
               ['delta_f_smooth', 'rocof_Hz_per_s', 'rocof_short_Hz_per_s']] = np.nan
               
    return df

# ══════════════════════════════════════════════════════════════════════════════
# Section 4 — Event detection
# ══════════════════════════════════════════════════════════════════════════════
def detect_events(df_rel, freq_thresh, rocof_onset_thresh,
                  min_gap_s, min_duration_s, pre_search_s,
                  post_window_s, rocof_lag_max_s, max_recovery_s,
                  rocof_short_std=None):
    outside = df_rel['delta_f_Hz'].abs() > freq_thresh
    state_change = outside & ~outside.shift(fill_value=False)
    raw_crossings = df_rel[state_change].index.tolist()
    
    if len(raw_crossings) == 0:
        return pd.DataFrame(), 0

    merged = [raw_crossings[0]]
    for t in raw_crossings[1:]:
        if (t - merged[-1]).total_seconds() > min_gap_s:
            merged.append(t)
            
    seg_records = []
    for t in merged:
        pre_start = t - pd.Timedelta(seconds=pre_search_s)
        post_end = t + pd.Timedelta(seconds=post_window_s)
        window = df_rel.loc[pre_start:post_end]
        if len(window) < 10:
            continue

        crossing_window = df_rel.loc[t : t + pd.Timedelta(seconds=min_duration_s)]
        if not (crossing_window['delta_f_Hz'].abs() > freq_thresh).all():
            continue

        pre_window = df_rel.loc[pre_start:t]
        rocof_above = pre_window['rocof_Hz_per_s'].abs() > rocof_onset_thresh
        onset_time = pre_window[rocof_above].index[0] if rocof_above.any() else t
        post_onset = window.loc[onset_time:]
        if len(post_onset) == 0:
            continue

        peak_idx = post_onset['delta_f_Hz'].abs().idxmax()
        peak_df = float(post_onset.loc[peak_idx, 'delta_f_Hz'])
        direction = 'under' if peak_df < 0 else 'over'

        peak_rocof = float(window['rocof_short_Hz_per_s'].abs().max())
        peak_rocof_time = window['rocof_short_Hz_per_s'].abs().idxmax()
        rocof_lag = (peak_rocof_time - onset_time).total_seconds()
        peak_abs = abs(peak_df)
        
        post_peak = post_onset.loc[peak_idx:]
        recovery_window = post_peak.iloc[:int(post_window_s / DT)]
        fast_window = post_peak.iloc[:int(max_recovery_s / DT)]
        
        recovery_50_fast = fast_window[fast_window['delta_f_Hz'].abs() < peak_abs * 0.5]
        recovery_80_fast = fast_window[fast_window['delta_f_Hz'].abs() < peak_abs * 0.8]
        recovery_50_slow = recovery_window[recovery_window['delta_f_Hz'].abs() < peak_abs * 0.5]
        
        if len(recovery_50_fast) > 0:
            end_time = recovery_50_fast.index[0]
            recovery_type = '50%_fast'
        elif len(recovery_80_fast) > 0:
            end_time = recovery_80_fast.index[0]
            recovery_type = '80%_fast'
        elif len(recovery_50_slow) > 0:
            end_time = recovery_50_slow.index[0]
            recovery_type = '50%_slow'
        else:
            end_time = recovery_window.index[-1]
            recovery_type = 'none'
            
        duration = (end_time - onset_time).total_seconds()
        
        seg_records.append({
            'onset_time'      : onset_time,
            'peak_time'       : peak_idx,
            'peak_rocof_time' : peak_rocof_time,
            'end_time'        : end_time,
            'direction'       : direction,
            'peak_df_Hz'      : round(peak_df, 4),
            'max_rocof_Hz_s'  : round(peak_rocof, 5),
            'rocof_lag_s'     : round(rocof_lag, 1),
            'duration_s'      : round(duration, 1),
            'recovery_type'   : recovery_type,
        })
        
    segs = pd.DataFrame(seg_records).reset_index(drop=True)
    if len(segs) == 0:
        return segs, len(raw_crossings)

    if rocof_short_std is not None:
        strong_thresh = 8 * rocof_short_std
        moderate_thresh = 4 * rocof_short_std
        def classify(row):
            lag_ok = row['rocof_lag_s'] < rocof_lag_max_s
            if row['max_rocof_Hz_s'] > strong_thresh and lag_ok: return 'STRONG'
            if row['max_rocof_Hz_s'] > moderate_thresh and lag_ok: return 'MODERATE'
            return 'WEAK'
        segs['classification'] = segs.apply(classify, axis=1)
    else:
        segs['classification'] = 'UNKNOWN'
        
    return segs, len(raw_crossings)

def _match_report_to_segment(rrow, segs, tol=MATCH_TOL_S):
    r_time = rrow['report_time']
    t_lo = r_time - pd.Timedelta(seconds=float(rrow['delta_t_s']) + tol)
    t_hi = r_time + pd.Timedelta(seconds=30)
    
    cands = segs[
        (segs['onset_time'] >= t_lo) &
        (segs['onset_time'] <= t_hi) &
        (segs['direction'] == rrow['direction'])
    ].copy()
    
    if cands.empty:
        return None, None
        
    expected = r_time - pd.Timedelta(seconds=float(rrow['delta_t_s']))
    cands['dt'] = (cands['onset_time'] - expected).abs()
    best = cands.nsmallest(1, 'dt')
    return best.index[0], best.iloc[0]

# ══════════════════════════════════════════════════════════════════════════════
# Section 5 — H and D estimation
# ══════════════════════════════════════════════════════════════════════════════
def estimate_hd(df_reliable, seg, report_row):
    onset = seg['onset_time']
    nadir_t = seg['peak_time']
    direction = seg['direction']
    
    sign = 1.0 if direction == 'over' else -1.0
    dP_MW = float(report_row['delta_P_MW'])
    dP_pu = sign * dP_MW / S_BASE_MW
    
    H_from_Ek = float(report_row['Ek_GWs']) * 1000.0 / S_BASE_MW
    H_est = H_from_Ek
    
    try:
        df_nadir_Hz = float(df_reliable.loc[nadir_t, 'delta_f_Hz'])
    except KeyError:
        idx = df_reliable.index.get_indexer([nadir_t], method='nearest')[0]
        df_nadir_Hz = float(df_reliable.iloc[idx]['delta_f_Hz'])
        
    if abs(df_nadir_Hz) < 1e-6:
        D_nadir = np.nan
    else:
        D_nadir = abs((dP_pu * F0) / df_nadir_Hz)

    pw_start = onset + pd.Timedelta(seconds=D_PW_SKIP_S)
    d_pw_win = df_reliable.loc[pw_start:nadir_t].copy()
    D_pw_mean = D_pw_median = D_pw_std = np.nan
    D_pw_n = 0
    
    if len(d_pw_win) >= 5:
        df_pw = d_pw_win['delta_f_Hz'].values
        rocof_pw = d_pw_win['rocof_Hz_per_s'].values
        numer_pw = dP_pu * F0 - 2.0 * H_est * rocof_pw
        valid_pw = np.abs(df_pw) > D_PW_MIN_DF_HZ
        D_samples = numer_pw[valid_pw] / df_pw[valid_pw]
        D_samples = D_samples[D_samples > 0.3]
        if len(D_samples) >= 3:
            D_pw_mean = float(np.nanmean(D_samples))
            D_pw_median = float(np.nanmedian(D_samples))
            D_pw_std = float(np.nanstd(D_samples))
            D_pw_n = int(len(D_samples))

    h_pw_win = df_reliable.loc[onset : onset + pd.Timedelta(seconds=H_PW_END_S)].copy()
    H_pw_mean = H_pw_median = H_pw_std = np.nan
    H_pw_n = 0
    
    if len(h_pw_win) >= 5:
        df_h_arr = h_pw_win['delta_f_Hz'].values
        rocof_h_arr = h_pw_win['rocof_Hz_per_s'].values
        expected_rocof_sign = 1.0 if direction == 'over' else -1.0
        valid_h = (
            (np.abs(rocof_h_arr) > H_PW_MIN_ROCOF) &
            (np.sign(rocof_h_arr) == expected_rocof_sign)
        )
        if valid_h.any():
            numer_h = dP_pu * F0 - D_nadir * df_h_arr[valid_h]
            denom_h = 2.0 * rocof_h_arr[valid_h]
            with np.errstate(divide='ignore', invalid='ignore'):
                H_raw = np.where(np.abs(denom_h) > 1e-8, numer_h / denom_h, np.nan)
            H_samples = H_raw[(H_raw > 1.0) & (H_raw < 8.0)]
            if len(H_samples) >= 3:
                H_pw_mean = float(np.nanmean(H_samples))
                H_pw_median = float(np.nanmedian(H_samples))
                H_pw_std = float(np.nanstd(H_samples))
                H_pw_n = int(len(H_samples))

    def _r(v, d=4):
        return round(float(v), d) if (v is not None and not np.isnan(v)) else np.nan

    return {
        'onset_time'  : onset,
        'cause'       : report_row['cause'],
        'direction'   : direction,
        'dP_MW'       : dP_MW,
        'dP_pu'       : _r(dP_pu, 6),
        'df_nadir_Hz' : _r(df_nadir_Hz),
        'H_from_Ek'   : _r(H_from_Ek),
        'D_nadir'     : _r(D_nadir),
        'D_pw_mean'   : _r(D_pw_mean),
        'D_pw_median' : _r(D_pw_median),
        'D_pw_std'    : _r(D_pw_std),
        'D_pw_n'      : D_pw_n,
        'H_pw_mean'   : _r(H_pw_mean),
        'H_pw_median' : _r(H_pw_median),
        'H_pw_std'    : _r(H_pw_std),
        'H_pw_n'      : H_pw_n,
    }

# ══════════════════════════════════════════════════════════════════════════════
# Section 6 — Plotting Functions
# ══════════════════════════════════════════════════════════════════════════════
def plot_event_dynamics(df_reliable, seg, result_dict, plot_dir='plots'):
    """
    Generates a two-panel time-series plot (Frequency and RoCoF) for a single event.
    """
    if plot_dir and not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    onset = seg['onset_time']
    nadir = seg['peak_time']
    recovery = seg['end_time']
    peak_rocof_t = seg['peak_rocof_time']
    
    plot_start = onset - pd.Timedelta(seconds=30)
    plot_end = recovery + pd.Timedelta(seconds=30)
    
    win = df_reliable.loc[plot_start:plot_end].copy()
    if win.empty:
        return
    
    t_rel = (win.index - onset).total_seconds()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Top Panel
    ax1.plot(t_rel, win['delta_f_Hz'], color='lightgray', label='Raw Δf', lw=1)
    ax1.plot(t_rel, win['delta_f_smooth'], color='navy', label='Smooth Δf (Main SG)', lw=2)
    ax1.axvline(0, color='black', ls='--', lw=1.5, alpha=0.7)
    ax1.scatter(0, win.loc[onset, 'delta_f_smooth'], color='orange', zorder=5, s=80, label='Onset')
    
    t_nadir_rel = (nadir - onset).total_seconds()
    ax1.scatter(t_nadir_rel, win.loc[nadir, 'delta_f_smooth'], color='red', zorder=5, s=80, label='Nadir')
    
    t_rec_rel = (recovery - onset).total_seconds()
    ax1.scatter(t_rec_rel, win.loc[recovery, 'delta_f_smooth'], color='green', zorder=5, s=80, label=f'Recovery ({seg["recovery_type"]})')

    ax1.set_ylabel('Frequency Deviation Δf (Hz)', fontweight='bold')
    ax1.set_title(f"Event Dynamics: {onset.strftime('%Y-%m-%d %H:%M:%S')} | Cause: {result_dict['cause']} | ΔP: {result_dict['dP_MW']} MW", fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right')

    # Bottom Panel
    ax2.plot(t_rel, win['rocof_short_Hz_per_s'], color='lightblue', label='Short SG RoCoF', lw=1.5)
    ax2.plot(t_rel, win['rocof_Hz_per_s'], color='darkred', label='Main SG RoCoF', lw=2)
    
    t_peak_rocof_rel = (peak_rocof_t - onset).total_seconds()
    ax2.scatter(t_peak_rocof_rel, win.loc[peak_rocof_t, 'rocof_short_Hz_per_s'], color='purple', zorder=5, s=80, label='Peak RoCoF')
    
    ax2.axvline(0, color='black', ls='--', lw=1.5, alpha=0.7)
    ax2.axvspan(0, 3.0, color='gray', alpha=0.15, label='H Pointwise Window (0-3s)')

    ax2.set_ylabel('RoCoF (Hz/s)', fontweight='bold')
    ax2.set_xlabel('Time relative to event onset (seconds)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right')
    
    info_text = (
        f"H_from_Ek: {result_dict['H_from_Ek']} s\n"
        f"D_nadir: {result_dict['D_nadir']} pu\n"
        f"Direction: {result_dict['direction']}"
    )
    fig.text(0.02, 0.02, info_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.tight_layout()
    filename = f"event_{onset.strftime('%Y%m%d_%H%M%S')}_{result_dict['cause']}.png"
    filepath = os.path.join(plot_dir, filename)
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()


def plot_summary_results(results_df, plot_dir):
    """
    Two-panel bar chart: H_from_Ek vs H_pw_mean (left) and D_nadir vs D_pw_mean (right).
    NaN H_pw / D_pw rows are skipped — they do not render as spurious zero bars.
    Y-axes are fixed to physical Nordic bounds so out-of-range values are immediately visible.
    """
    if len(results_df) < 2:
        print("[PLOT] Not enough events to generate a summary plot.")
        return

    df = results_df.copy()
    df['onset_time'] = pd.to_datetime(df['onset_time'])
    df = df.sort_values('onset_time').reset_index(drop=True)

    dir_sym = df['direction'].map({'under': '▼', 'over': '▲'}).fillna('')
    labels  = [f"{r['onset_time'].strftime('%d-%b')}\n{r['cause']} {dir_sym[i]}"
               for i, (_, r) in enumerate(df.iterrows())]

    x_ev  = np.arange(len(df))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── H panel ───────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.axhspan(2.0, 6.0, color='steelblue', alpha=0.08, label='Expected range 2–6 s')
    ax.bar(x_ev - width/2, df['H_from_Ek'], width,
           color='navy', alpha=0.85, label='H_from_Ek  [PRIMARY]')

    mask_h = df['H_pw_mean'].notna()
    if mask_h.any():
        h_pw  = df.loc[mask_h, 'H_pw_mean'].values
        h_std = df.loc[mask_h, 'H_pw_std'].fillna(0).values
        ax.bar(x_ev[mask_h] + width/2, h_pw, width,
               color='steelblue', alpha=0.75, label='H_pw_mean  [diagnostic]')
        ax.errorbar(x_ev[mask_h] + width/2, h_pw, yerr=h_std,
                    fmt='none', color='black', capsize=4, lw=1.2, alpha=0.7)
        ax.legend(fontsize=9, loc='upper right')
    else:
        ax.legend(fontsize=9, loc='upper right')

    h_mean = df['H_from_Ek'].mean()
    ax.axhline(h_mean, color='navy', ls='--', lw=1.4,
               label=f'mean H_Ek = {h_mean:.2f} s')
    ax.set_xticks(x_ev)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('H  [s]', fontweight='bold')
    ax.set_ylim(0, 8.0)   # hard upper = physical Nordic maximum
    ax.set_title('Inertia constant H per event', fontweight='bold', pad=8)
    ax.grid(True, alpha=0.35, axis='y', ls=':')

    # ── D panel ───────────────────────────────────────────────────────────────
    ax = axes[1]
    ax.axhspan(1.0, 4.0, color='red', alpha=0.06, label='Expected range 1–4 pu')
    ax.bar(x_ev - width/2, df['D_nadir'], width,
           color='red', alpha=0.80, label='D_nadir  [PRIMARY]')

    mask_d = df['D_pw_mean'].notna()
    if mask_d.any():
        d_pw  = df.loc[mask_d, 'D_pw_mean'].values
        d_std = df.loc[mask_d, 'D_pw_std'].fillna(0).values
        ax.bar(x_ev[mask_d] + width/2, d_pw, width,
               color='darkorange', alpha=0.80, label='D_pw_mean  [cross-check]')
        ax.errorbar(x_ev[mask_d] + width/2, d_pw, yerr=d_std,
                    fmt='none', color='black', capsize=4, lw=1.2, alpha=0.7)

    d_mean    = df['D_nadir'].mean()
    d_pw_mean = float(df['D_pw_mean'].mean()) if mask_d.any() else None
    ax.axhline(d_mean, color='red', ls='--', lw=1.4,
               label=f'mean D_nadir = {d_mean:.2f} pu')
    if d_pw_mean is not None:
        ax.axhline(d_pw_mean, color='darkorange', ls='--', lw=1.4,
                   label=f'mean D_pw = {d_pw_mean:.2f} pu')
    ax.set_xticks(x_ev)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('D  [pu]', fontweight='bold')
    ax.set_ylim(0, 5.0)
    ax.set_title('Damping coefficient D per event', fontweight='bold', pad=8)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.35, axis='y', ls=':')

    plt.suptitle(
        f'Nordic Grid 2016 — H and D Estimation  |  '
        f'S_base = {S_BASE_MW:.0f} MW  |  {len(df)} events',
        fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()

    os.makedirs(plot_dir, exist_ok=True)
    fpath = os.path.join(plot_dir, 'hd_summary_2016.png')
    plt.savefig(fpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Summary saved → {fpath}")

# ══════════════════════════════════════════════════════════════════════════════
# Section 7 — Automated pipeline
# ══════════════════════════════════════════════════════════════════════════════
def run_pipeline(data_dir='initial_data', output_csv='hd_pipeline/hd_results_2016.csv',
                 force_rerun=False, plot_dir=None):
    if force_rerun and os.path.exists(output_csv):
        os.remove(output_csv)
        print(f"[FORCE] Deleted {output_csv} — reprocessing all months.")

    if os.path.exists(output_csv):
        try:
            existing = pd.read_csv(output_csv, parse_dates=['onset_time'])
            done_periods = set(existing['onset_time'].dt.to_period('M').astype(str).tolist())
        except Exception:
            existing = pd.DataFrame()
            done_periods = set()
    else:
        existing = pd.DataFrame()
        done_periods = set()

    pattern = os.path.join(data_dir, 'finland_2016_*.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[ERROR] No files matching '{pattern}' found.")
        sys.exit(1)

    all_results = []
    
    for fpath in files:
        fname = os.path.basename(fpath)
        parts = fname.replace('.csv', '').split('_')
        try:
            year = int(parts[1])
            month_num = int(parts[2])
            month_label = f"{year}-{month_num:02d}"
        except (IndexError, ValueError):
            print(f"[WARNING] Cannot parse month from {fname}, skipping.")
            continue

        if month_label in done_periods:
            print(f"[SKIP] {month_label} (already in {output_csv})")
            continue

        print(f"\n[PROCESSING] {month_label} ({fname})")

        df, nan_runs = load_month(fpath)
        df = apply_sg_filters(df, nan_runs)
        df_reliable = df[df['reliable']].copy()
        
        if len(df_reliable) < 100:
            print(f"   [WARNING] {month_label} — insufficient reliable data, skipping.")
            continue

        rocof_std = float(df_reliable['rocof_Hz_per_s'].std())
        rocof_short_std = float(df_reliable['rocof_short_Hz_per_s'].std())
        rocof_onset_thr = SIGMA_CHOICE * rocof_std

        segments, n_raw = detect_events(
            df_reliable, freq_thresh=FREQ_THRESH, rocof_onset_thresh=rocof_onset_thr,
            min_gap_s=MIN_EVENT_GAP_S, min_duration_s=MIN_DURATION_S, pre_search_s=PRE_SEARCH_S,
            post_window_s=POST_WINDOW_S, rocof_lag_max_s=ROCOF_LAG_MAX_S,
            max_recovery_s=MAX_RECOVERY_S, rocof_short_std=rocof_short_std
        )

        segments_est = segments[
            (segments['classification'] != 'WEAK') &
            (segments['recovery_type'].isin(['50%_fast', '80%_fast']))
        ].copy().reset_index(drop=True)

        if month_label == '2016-01' and len(segments_est) == 0 and len(segments) > 0:
            print(f"   [INFO] Retrying January with relaxed recovery window (90s).")
            segments, n_raw = detect_events(
                df_reliable, freq_thresh=FREQ_THRESH, rocof_onset_thresh=rocof_onset_thr,
                min_gap_s=MIN_EVENT_GAP_S, min_duration_s=MIN_DURATION_S, pre_search_s=PRE_SEARCH_S,
                post_window_s=POST_WINDOW_S, rocof_lag_max_s=ROCOF_LAG_MAX_S,
                max_recovery_s=JAN_MAX_RECOVERY_S, 
                rocof_short_std=rocof_short_std
            )
            segments_est = segments[
                (segments['classification'] != 'WEAK') &
                (segments['recovery_type'].isin(['50%_fast', '80%_fast', '50%_slow']))
            ].copy().reset_index(drop=True)

        if len(segments_est) == 0:
            print(f"   [WARNING] No estimation-grade events found.")
            continue

        month_reports = REPORT_EVENTS[REPORT_EVENTS['month'] == month_label]
        
        for _, rrow in month_reports.iterrows():
            best_idx, best_seg = _match_report_to_segment(rrow, segments_est)
            
            if best_seg is not None:
                result_dict = estimate_hd(df_reliable, best_seg, rrow)
                all_results.append(result_dict)
                
                print(f"   [MATCH] {rrow['date_str']} | H: {result_dict['H_from_Ek']}s | D: {result_dict['D_nadir']}pu")
                
                if plot_dir:
                    plot_event_dynamics(df_reliable, best_seg, result_dict, plot_dir)
            else:
                print(f"   [MISS] No PMU segment matched report event at {rrow['date_str']}")

    if all_results:
        new_df = pd.DataFrame(all_results)
        combined_df = pd.concat([existing, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['onset_time']).reset_index(drop=True)
        combined_df.to_csv(output_csv, index=False)
        print(f"\n[DONE] Saved {len(combined_df)} total events to {output_csv}")
    else:
        combined_df = existing
        print("\n[DONE] No new events processed.")

    if plot_dir and len(combined_df) >= 2:
        plot_summary_results(combined_df, plot_dir)

    return len(combined_df)

# ══════════════════════════════════════════════════════════════════════════════
# Section 8 — Execution
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nordic Grid H and D Estimation Pipeline")
    parser.add_argument('--data-dir', type=str, default='initial_data', help='Directory containing the monthly CSV files')
    parser.add_argument('--output', type=str, default='hd_pipeline/hd_results_2016.csv', help='Output CSV filename')
    parser.add_argument('--force', action='store_true', help='Force rerun (deletes existing output CSV)')
    parser.add_argument('--plots', type=str, default='plots', help='Directory to save generated plots')
    
    args = parser.parse_args()
    
    run_pipeline(
        data_dir=args.data_dir, 
        output_csv=args.output, 
        force_rerun=args.force, 
        plot_dir=args.plots
    )