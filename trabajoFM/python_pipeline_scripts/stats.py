from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# -----------------------------
# Helpers: safe math
# -----------------------------

def _finite_pairs(y: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    m = np.asarray(m, dtype=float)
    mask = np.isfinite(y) & np.isfinite(m)
    return y[mask], m[mask]


def _pearson_r(y: np.ndarray, m: np.ndarray) -> float:
    y, m = _finite_pairs(y, m)
    if y.size < 2:
        return float("nan")
    if np.std(y) == 0 or np.std(m) == 0:
        return float("nan")
    with np.errstate(invalid="ignore"):
        r = np.corrcoef(y, m)[0, 1]
    return float(r)


def _nse(y: np.ndarray, m: np.ndarray) -> float:
    y, m = _finite_pairs(y, m)
    if y.size == 0:
        return float("nan")
    denom = np.nansum((y - np.nanmean(y)) ** 2)
    if not np.isfinite(denom) or denom == 0:
        return float("nan")
    num = np.nansum((y - m) ** 2)
    return float(1.0 - num / denom)


def _kge(y: np.ndarray, m: np.ndarray) -> float:
    y, m = _finite_pairs(y, m)
    if y.size < 2:
        return float("nan")
    r = _pearson_r(y, m)
    if not np.isfinite(r):
        return float("nan")
    mu_y = np.nanmean(y)
    mu_m = np.nanmean(m)
    sd_y = np.nanstd(y)
    sd_m = np.nanstd(m)
    if not all(np.isfinite(v) for v in (mu_y, mu_m, sd_y, sd_m)) or sd_y == 0:
        return float("nan")
    alpha = sd_m / sd_y
    beta = mu_m / mu_y if mu_y != 0 else np.nan
    if not np.isfinite(beta):
        return float("nan")
    kge = 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    return float(kge)


def _pbias(y: np.ndarray, m: np.ndarray) -> float:
    y, m = _finite_pairs(y, m)
    if y.size == 0:
        return float("nan")
    denom = np.nansum(y)
    if not np.isfinite(denom) or denom == 0:
        return float("nan")
    num = np.nansum(y - m)
    return float(100.0 * num / denom)


def _rmse(y: np.ndarray, m: np.ndarray) -> float:
    y, m = _finite_pairs(y, m)
    if y.size == 0:
        return float("nan")
    return float(np.sqrt(np.nanmean((y - m) ** 2)))


def _mae(y: np.ndarray, m: np.ndarray) -> float:
    y, m = _finite_pairs(y, m)
    if y.size == 0:
        return float("nan")
    return float(np.nanmean(np.abs(y - m)))


def _medae(y: np.ndarray, m: np.ndarray) -> float:
    y, m = _finite_pairs(y, m)
    if y.size == 0:
        return float("nan")
    return float(np.nanmedian(np.abs(y - m)))


# -----------------------------
# New band deviation metrics
# -----------------------------

def _band_deviation_stats(mean_series: np.ndarray, min_series: np.ndarray, max_series: np.ndarray) -> Dict[str, float]:
    """Compute statistics describing how much the min/max bands deviate around the mean.
    
    Returns metrics quantifying simulation uncertainty/spread.
    """
    mean_arr = np.asarray(mean_series, dtype=float)
    min_arr = np.asarray(min_series, dtype=float) 
    max_arr = np.asarray(max_series, dtype=float)
    
    # Only use points where all three are finite
    mask = np.isfinite(mean_arr) & np.isfinite(min_arr) & np.isfinite(max_arr)
    if not np.any(mask):
        return {
            "band_width_mean": float("nan"),
            "band_width_rel%": float("nan"),
            "band_asymmetry": float("nan"),
            "band_rmse_vs_mean": float("nan"),
        }
    
    mean_vals = mean_arr[mask]
    min_vals = min_arr[mask]  
    max_vals = max_arr[mask]
    
    # Band width (max - min)
    band_widths = max_vals - min_vals
    band_width_mean = float(np.nanmean(band_widths))
    
    # Relative band width as percentage of mean
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_widths = (band_widths / np.abs(mean_vals)) * 100.0
        band_width_rel = float(np.nanmean(rel_widths[np.isfinite(rel_widths)]))
    
    # Band asymmetry: how much closer is mean to min vs max (0 = centered, +1 = closer to min, -1 = closer to max)
    with np.errstate(divide='ignore', invalid='ignore'):
        asymmetry = (2 * (mean_vals - min_vals) / band_widths) - 1.0
        band_asymmetry = float(np.nanmean(asymmetry[np.isfinite(asymmetry)]))
    
    # RMSE of band edges vs mean (measure of overall spread)
    min_dev = (min_vals - mean_vals) ** 2
    max_dev = (max_vals - mean_vals) ** 2
    band_rmse = float(np.sqrt(np.nanmean(np.concatenate([min_dev, max_dev]))))
    
    return {
        "band_width_mean": band_width_mean,
        "band_width_rel%": band_width_rel,
        "band_asymmetry": band_asymmetry, 
        "band_rmse_vs_mean": band_rmse,
    }


def _percentile_band_deviation_stats(mean_series: np.ndarray, p25_series: np.ndarray, p75_series: np.ndarray, 
                                   p05_series: np.ndarray = None, p95_series: np.ndarray = None) -> Dict[str, float]:
    """Compute statistics for percentile band deviations around the mean."""
    stats = {}
    
    # 50% band (p25-p75) 
    if p25_series is not None and p75_series is not None:
        band_stats_50 = _band_deviation_stats(mean_series, p25_series, p75_series)
        for key, val in band_stats_50.items():
            stats[f"p50_{key}"] = val
    
    # 90% band (p05-p95)
    if p05_series is not None and p95_series is not None:
        band_stats_90 = _band_deviation_stats(mean_series, p05_series, p95_series)
        for key, val in band_stats_90.items():
            stats[f"p90_{key}"] = val
            
    return stats


def _measured_vs_series_stats(measured: np.ndarray, target_series: np.ndarray, series_name: str) -> Dict[str, float]:
    """Compute comprehensive stats comparing measured data against any target time series."""
    y, m = _finite_pairs(measured, target_series)
    if y.size < 2:
        return {f"vs_{series_name}_r": float("nan"), f"vs_{series_name}_rmse": float("nan"), 
                f"vs_{series_name}_mae": float("nan"), f"vs_{series_name}_nse": float("nan"),
                f"vs_{series_name}_bias": float("nan")}
    
    return {
        f"vs_{series_name}_r": _pearson_r(y, m),
        f"vs_{series_name}_rmse": _rmse(y, m), 
        f"vs_{series_name}_mae": _mae(y, m),
        f"vs_{series_name}_nse": _nse(y, m),
        f"vs_{series_name}_bias": float(np.nanmean(y - m)),  # obs - pred
    }


# -----------------------------
# Core assemblers
# -----------------------------

def _collect_pairs(
    q_df: pd.DataFrame,
    measured: Sequence[pd.Series],
    *,
    window: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Return flattened arrays (Y, M) and dict of quantile arrays for coverage.

    - q_df: columns include 'p50', optionally 'p25','p75','p05','p95'.
    - measured: list of series. Each will be sliced by window and reindexed to q_df.
    """
    ys: List[np.ndarray] = []
    ms: List[np.ndarray] = []
    lo25s: List[np.ndarray] = []
    hi75s: List[np.ndarray] = []
    lo05s: List[np.ndarray] = []
    hi95s: List[np.ndarray] = []

    if q_df is None or q_df.empty or not measured:
        return np.array([]), np.array([]), {}

    for s in measured:
        if s is None or s.empty:
            continue
        ss = s
        if window is not None:
            x0, x1 = window
            ss = ss.loc[(ss.index >= x0) & (ss.index <= x1)]
        if ss.empty:
            continue
        q_win = q_df.reindex(ss.index)
        if "p50" not in q_win.columns:
            continue
        mask = q_win["p50"].notna() & ss.notna()
        if not mask.any():
            continue
        ys.append(ss.loc[mask].to_numpy(dtype=float))
        ms.append(q_win.loc[mask, "p50"].to_numpy(dtype=float))
        if "p25" in q_win.columns and "p75" in q_win.columns:
            lo25s.append(q_win.loc[mask, "p25"].to_numpy(dtype=float))
            hi75s.append(q_win.loc[mask, "p75"].to_numpy(dtype=float))
        if "p05" in q_win.columns and "p95" in q_win.columns:
            lo05s.append(q_win.loc[mask, "p05"].to_numpy(dtype=float))
            hi95s.append(q_win.loc[mask, "p95"].to_numpy(dtype=float))

    if not ys:
        return np.array([]), np.array([]), {}
    Y = np.concatenate(ys)
    M = np.concatenate(ms)
    qdict: Dict[str, np.ndarray] = {}
    if lo25s and hi75s:
        qdict["p25"] = np.concatenate(lo25s)
        qdict["p75"] = np.concatenate(hi75s)
    if lo05s and hi95s:
        qdict["p05"] = np.concatenate(lo05s)
        qdict["p95"] = np.concatenate(hi95s)
    return Y, M, qdict


def _coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    y, lo = _finite_pairs(y, lo)
    y, hi = _finite_pairs(y, hi)
    n = min(y.size, lo.size, hi.size)
    if n == 0:
        return float("nan")
    lo = lo[:n]
    hi = hi[:n]
    y = y[:n]
    with np.errstate(invalid="ignore"):
        cov = np.nanmean((y >= lo) & (y <= hi))
    return float(cov)


def _extras_correlations(
    measured: Sequence[pd.Series], extras: Optional[Dict[str, pd.Series]], *, window: Optional[Tuple[pd.Timestamp, pd.Timestamp]]
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    if not extras:
        return out
    for name, ex_s in extras.items():
        if ex_s is None or ex_s.empty:
            continue
        yy_list: List[np.ndarray] = []
        ee_list: List[np.ndarray] = []
        for s in measured:
            if s is None or s.empty:
                continue
            ss = s
            if window is not None:
                x0, x1 = window
                ss = ss.loc[(ss.index >= x0) & (ss.index <= x1)]
            if ss.empty:
                continue
            ex_win = ex_s.reindex(ss.index)
            mask = ss.notna() & ex_win.notna()
            if mask.any():
                yy_list.append(ss.loc[mask].to_numpy(dtype=float))
                ee_list.append(ex_win.loc[mask].to_numpy(dtype=float))
        if yy_list and ee_list:
            Y = np.concatenate(yy_list)
            E = np.concatenate(ee_list)
            r = _pearson_r(Y, E)
            out[str(name)] = {"r": r, "n": float(len(Y))}
    return out


def _global_best_lag(
    measured: Sequence[pd.Series], median_series: pd.Series, *, max_lag: int = 2, window: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    choose_by: str = "r",
) -> Optional[Dict[str, float]]:
    if median_series is None or median_series.empty or not measured:
        return None
    lags = list(range(int(-max_lag), int(max_lag) + 1))
    results: List[Tuple[int, float, float, float, float]] = []  # (lag, r, rmse, mae, nse)
    for L in lags:
        m_shift = median_series.shift(L)
        yy_list: List[np.ndarray] = []
        mm_list: List[np.ndarray] = []
        for s in measured:
            if s is None or s.empty:
                continue
            ss = s
            if window is not None:
                x0, x1 = window
                ss = ss.loc[(ss.index >= x0) & (ss.index <= x1)]
            if ss.empty:
                continue
            ms = m_shift.reindex(ss.index)
            mask = ss.notna() & ms.notna()
            if mask.any():
                yy_list.append(ss.loc[mask].to_numpy(dtype=float))
                mm_list.append(ms.loc[mask].to_numpy(dtype=float))
        if not yy_list:
            continue
        Y = np.concatenate(yy_list)
        M = np.concatenate(mm_list)
        r = _pearson_r(Y, M)
        rmse = _rmse(Y, M)
        mae = _mae(Y, M)
        nse = _nse(Y, M)
        results.append((L, r, rmse, mae, nse))
    if not results:
        return None
    if choose_by.lower() == "nse":
        best = max(results, key=lambda x: (np.nan_to_num(x[4], nan=-np.inf)))
    else:
        best = max(results, key=lambda x: (np.nan_to_num(x[1], nan=-np.inf)))
    L, r, rmse, mae, nse = best
    out = {
        "best_lag_days": float(L),
        "metric": "NSE" if choose_by.lower() == "nse" else "r",
        "r": float(r),
        "R2": float(r ** 2) if np.isfinite(r) else float("nan"),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "NSE": float(nse),
    }
    return out


def _local_window_match(
    measured: Sequence[pd.Series], median_series: pd.Series, *, K: int = 1, window: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    strategy: str = "nearest",
) -> Optional[Dict[str, Union[float, int]]]:
    if median_series is None or median_series.empty or not measured or K is None or K < 0:
        return None
    pairs: List[Tuple[pd.Timestamp, float, float, int]] = []  # (time, obs, mod_best, k_days)
    for s in measured:
        if s is None or s.empty:
            continue
        ss = s
        if window is not None:
            x0, x1 = window
            ss = ss.loc[(ss.index >= x0) & (ss.index <= x1)]
        if ss.empty:
            continue
        for t, o in ss.dropna().items():
            lo = t - pd.Timedelta(days=K)
            hi = t + pd.Timedelta(days=K)
            window_vals = median_series.loc[lo:hi]
            window_vals = window_vals.dropna()
            if window_vals.empty:
                continue
            if strategy == "mean":
                m_best = float(window_vals.mean())
                # choose k closest day to center for reporting
                k_days = 0
            else:
                # nearest in absolute error
                diffs = (window_vals - float(o)).abs()
                t_star = diffs.idxmin()
                m_best = float(window_vals.loc[t_star])
                k_days = int((t_star - t) / pd.Timedelta(days=1))
            pairs.append((t, float(o), m_best, k_days))
    if not pairs:
        return None
    df = pd.DataFrame(pairs, columns=["time", "obs", "mod_best", "k_days"]).set_index("time")
    y = df["obs"].to_numpy(dtype=float)
    m = df["mod_best"].to_numpy(dtype=float)
    rmse = _rmse(y, m)
    mae = _mae(y, m)
    bias = float(np.nanmean(y - m)) if y.size else float("nan")
    medlag = float(np.nanmedian(df["k_days"]))
    q75 = float(np.nanpercentile(df["k_days"], 75))
    q25 = float(np.nanpercentile(df["k_days"], 25))
    iqr = float(q75 - q25)
    frac0 = float(np.mean(df["k_days"] == 0))
    out = {
        "K": float(K),
        "strategy": strategy,
        "RMSE": float(rmse),
        "MAE": float(mae),
        "Bias(obs-pred)": float(bias),
        "median_lag_days": medlag,
        "IQR_lag_days": iqr,
        "fraction_zero_lag": float(frac0),
        "n": float(len(df)),
    }
    return out


def compute_stats_for_view(
    q_df: pd.DataFrame,
    measured_series: Sequence[pd.Series],
    *,
    window: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    extras: Optional[Dict[str, pd.Series]] = None,
    compute_log: bool = True,
    max_global_lag: int = 2,
    local_window_ks: Sequence[int] = (1,),
    local_strategy: str = "nearest",
    choose_best_lag_by: str = "r",
    # New parameters for band deviation analysis
    band_data: Optional[Dict[str, pd.Series]] = None,
) -> Dict[str, object]:
    """Compute an extensible suite of stats for the dashboard view.

    New parameters:
    - band_data: Optional dict with keys like 'min', 'max', 'mean', 'p25', 'p75', 'p05', 'p95' 
                 containing time series for band deviation analysis

    Returns a nested dict with sections: n, same_day, log_space, global_lag, local_window_K*, extras, 
                                        band_stats, measured_vs_series.
    """
    stats: Dict[str, object] = {}

    Y, M, qdict = _collect_pairs(q_df, measured_series, window=window)
    n = int(Y.size)
    stats["n"] = n
    if n == 0:
        stats["same_day"] = {}
        return stats

    # Same-day metrics
    r = _pearson_r(Y, M)
    R2 = float(r ** 2) if np.isfinite(r) else float("nan")
    rmse = _rmse(Y, M)
    mae = _mae(Y, M)
    bias = float(np.nanmean(Y - M)) if n else float("nan")  # obs - pred
    pbias = _pbias(Y, M)
    nse = _nse(Y, M)
    kge = _kge(Y, M)
    medae = _medae(Y, M)
    same_day: Dict[str, float] = {
        "r": r,
        "R2": R2,
        "MAE": mae,
        "RMSE": rmse,
        "Bias(obs-pred)": bias,
        "PBIAS%": pbias,
        "NSE": nse,
        "KGE": kge,
        "MedAE": medae,
    }
    if "p25" in qdict and "p75" in qdict:
        same_day["coverage50"] = _coverage(Y, qdict["p25"], qdict["p75"])  # fraction
    if "p05" in qdict and "p95" in qdict:
        same_day["coverage90"] = _coverage(Y, qdict["p05"], qdict["p95"])  # fraction
    stats["same_day"] = same_day

    # Log-space metrics (base 10) for positive values
    if compute_log:
        mask_pos = (Y > 0) & (M > 0) & np.isfinite(Y) & np.isfinite(M)
        if np.any(mask_pos):
            ylog = np.log10(Y[mask_pos])
            mlog = np.log10(M[mask_pos])
            stats["log_space"] = {
                "r_log": _pearson_r(ylog, mlog),
                "R2_log": float(_pearson_r(ylog, mlog) ** 2) if np.isfinite(_pearson_r(ylog, mlog)) else float("nan"),
                "RMSElog10": _rmse(ylog, mlog),
                "MAElog10": _mae(ylog, mlog),
                "NSElog10": _nse(ylog, mlog),
                "n_pos": int(mask_pos.sum()),
            }

    # Global best lag (against median p50)
    median_series = q_df["p50"] if (q_df is not None and "p50" in q_df.columns) else None
    if median_series is not None and not median_series.empty and max_global_lag is not None and max_global_lag > 0:
        gl = _global_best_lag(measured_series, median_series, max_lag=int(max_global_lag), window=window, choose_by=str(choose_best_lag_by))
        if gl is not None:
            stats["global_lag"] = gl

    # Local window matching for one or more K values
    if median_series is not None and not median_series.empty:
        for K in local_window_ks:
            lw = _local_window_match(measured_series, median_series, K=int(K), window=window, strategy=local_strategy)
            if lw is not None:
                # Calculate NSE for local window matching
                try:
                    y = np.array(lw.get("obs")) if "obs" in lw else None
                    m = np.array(lw.get("mod_best")) if "mod_best" in lw else None
                    # If obs and mod_best are not present, reconstruct from measured_series and median_series
                    if y is None or m is None:
                        # Recompute pairs for this K
                        pairs = []
                        for s in measured_series:
                            if s is None or s.empty:
                                continue
                            ss = s
                            if window is not None:
                                x0, x1 = window
                                ss = ss.loc[(ss.index >= x0) & (ss.index <= x1)]
                            if ss.empty:
                                continue
                            for t, o in ss.dropna().items():
                                lo = t - pd.Timedelta(days=K)
                                hi = t + pd.Timedelta(days=K)
                                window_vals = median_series.loc[lo:hi].dropna()
                                if window_vals.empty:
                                    continue
                                if local_strategy == "mean":
                                    m_best = float(window_vals.mean())
                                else:
                                    diffs = (window_vals - float(o)).abs()
                                    t_star = diffs.idxmin()
                                    m_best = float(window_vals.loc[t_star])
                                pairs.append((float(o), m_best))
                        if pairs:
                            y = np.array([p[0] for p in pairs], dtype=float)
                            m = np.array([p[1] for p in pairs], dtype=float)
                    if y is not None and m is not None and y.size > 0:
                        lw["NSE"] = _nse(y, m)
                except Exception:
                    lw["NSE"] = float("nan")
                stats[f"local_window_K{int(K)}"] = lw

    # Extras correlations
    ex = _extras_correlations(measured_series, extras, window=window)
    if ex:
        stats["extras"] = ex

    # NEW: Band deviation analysis
    if band_data is not None and isinstance(band_data, dict):
        band_stats = {}
        
        # Apply window filtering to band data
        def _apply_window_to_series(s: pd.Series) -> pd.Series:
            if s is None or s.empty:
                return s
            if window is not None:
                x0, x1 = window
                return s.loc[(s.index >= x0) & (s.index <= x1)]
            return s
        
        # Get windowed band series
        windowed_bands = {k: _apply_window_to_series(v) for k, v in band_data.items()}
        
        # Min/max band analysis
        if all(k in windowed_bands for k in ['mean', 'min', 'max']):
            mean_vals = windowed_bands['mean'].dropna().values
            min_vals = windowed_bands['min'].reindex(windowed_bands['mean'].index).dropna().values
            max_vals = windowed_bands['max'].reindex(windowed_bands['mean'].index).dropna().values
            if len(mean_vals) > 0 and len(mean_vals) == len(min_vals) == len(max_vals):
                band_stats.update(_band_deviation_stats(mean_vals, min_vals, max_vals))
        
        # Percentile band analysis  
        perc_keys = ['mean', 'p25', 'p75', 'p05', 'p95']
        if all(k in windowed_bands for k in perc_keys[:3]):  # At least mean, p25, p75
            mean_vals = windowed_bands['mean'].dropna().values
            p25_vals = windowed_bands['p25'].reindex(windowed_bands['mean'].index).dropna().values
            p75_vals = windowed_bands['p75'].reindex(windowed_bands['mean'].index).dropna().values
            p05_vals = windowed_bands.get('p05')
            p95_vals = windowed_bands.get('p95')
            
            if p05_vals is not None and p95_vals is not None:
                p05_vals = p05_vals.reindex(windowed_bands['mean'].index).dropna().values
                p95_vals = p95_vals.reindex(windowed_bands['mean'].index).dropna().values
            else:
                p05_vals = p95_vals = None
            
            if len(mean_vals) > 0:
                perc_stats = _percentile_band_deviation_stats(mean_vals, p25_vals, p75_vals, p05_vals, p95_vals)
                band_stats.update(perc_stats)
        
        if band_stats:
            stats["band_deviation"] = band_stats

    # NEW: Measured vs all available series  
    if measured_series and band_data is not None:
        measured_vs_series = {}
        
        # Flatten all measured data for comparison
        all_measured_values = []
        all_measured_times = []
        for s in measured_series:
            if s is None or s.empty:
                continue
            ss = s
            if window is not None:
                x0, x1 = window
                ss = ss.loc[(ss.index >= x0) & (ss.index <= x1)]
            for t, val in ss.dropna().items():
                all_measured_values.append(val)
                all_measured_times.append(t)
        
        if all_measured_values:
            measured_times = pd.Index(all_measured_times)
            measured_vals = np.array(all_measured_values)
            
            # Compare against each band series
            for series_name, series_data in band_data.items():
                if series_data is None or series_data.empty:
                    continue
                
                # Apply window filter
                series_windowed = _apply_window_to_series(series_data)
                if series_windowed.empty:
                    continue
                
                # Align measured times with series data
                try:
                    aligned_series = series_windowed.reindex(measured_times, method='nearest', tolerance=pd.Timedelta(days=1))
                    series_vals = aligned_series.dropna().values
                    
                    # Only keep pairs where both are available
                    valid_indices = ~pd.isna(aligned_series.values)
                    if np.any(valid_indices):
                        measured_subset = measured_vals[valid_indices]
                        series_subset = series_vals
                        
                        if len(measured_subset) >= 2 and len(series_subset) >= 2:
                            series_stats = _measured_vs_series_stats(measured_subset, series_subset, series_name)
                            measured_vs_series.update(series_stats)
                except Exception:
                    continue
            
            # Also compare against extras if available
            if extras is not None:
                for extra_name, extra_series in extras.items():
                    if extra_series is None or extra_series.empty:
                        continue
                    
                    extra_windowed = _apply_window_to_series(extra_series)
                    if extra_windowed.empty:
                        continue
                    
                    try:
                        aligned_extra = extra_windowed.reindex(measured_times, method='nearest', tolerance=pd.Timedelta(days=1))
                        extra_vals = aligned_extra.dropna().values
                        
                        valid_indices = ~pd.isna(aligned_extra.values)
                        if np.any(valid_indices):
                            measured_subset = measured_vals[valid_indices]
                            
                            if len(measured_subset) >= 2 and len(extra_vals) >= 2:
                                extra_stats = _measured_vs_series_stats(measured_subset, extra_vals, f"extra_{extra_name}")
                                measured_vs_series.update(extra_stats)
                    except Exception:
                        continue
            
            if measured_vs_series:
                stats["measured_vs_series"] = measured_vs_series

    return stats


def format_stats_text(stats: Dict[str, object]) -> str:
    """Format a stats dict (from compute_stats_for_view) into HTML for the annotation box.

    Designed to be resilient to additional/unknown keys.
    """
    if not isinstance(stats, dict):
        return "No stats"
    n = int(stats.get("n", 0) or 0)
    lines: List[str] = []
    lines.append("<b>Stats (view)</b>")
    lines.append(f"n = {n}")

    same = stats.get("same_day", {}) or {}
    if same:
        def fmt_pair(k: str, v: float) -> Optional[str]:
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                return None
            if k.lower() in {"r", "r_log", "r (meas vs median)", "nse", "kge", "r2", "r2_log"}:
                return f"{k} = {float(v):.3f}"
            if "coverage" in k.lower():
                return f"{k} = {float(v) * 100:.1f}%"
            # Error metrics and bias
            return f"{k} = {float(v):.3g}"

        lines.append("<b>Same-day</b>")
        order = ["r", "R2", "MAE", "RMSE", "Bias(obs-pred)", "PBIAS%", "NSE", "KGE", "MedAE", "coverage50", "coverage90"]
        for key in order:
            if key in same:
                s = fmt_pair(key, same[key])
                if s:
                    lines.append(s)
        # Include any extra keys
        for key, val in same.items():
            if key in order:
                continue
            s = fmt_pair(str(key), val)
            if s:
                lines.append(s)

    logd = stats.get("log_space", {}) or {}
    if logd:
        lines.append("<b>Log-space</b>")
        for key in ["r_log", "R2_log", "MAElog10", "RMSElog10", "NSElog10", "n_pos"]:
            if key in logd:
                val = logd[key]
                if key == "n_pos":
                    lines.append(f"n(+/+) = {int(val)}")
                else:
                    if key.lower().startswith("r") or key.lower().endswith("log") or key.lower().startswith("nse"):
                        if isinstance(val, (int, float)) and np.isfinite(val):
                            lines.append(f"{key} = {float(val):.3f}")
                    else:
                        if isinstance(val, (int, float)) and np.isfinite(val):
                            lines.append(f"{key} = {float(val):.3g}")

    gl = stats.get("global_lag", {}) or {}
    if gl:
        lines.append("<b>Global lag</b>")
        metric = gl.get("metric", "r")
        if "best_lag_days" in gl:
            lines.append(f"best lag = {int(gl['best_lag_days'])} d (by {metric})")
        for key in ["r", "R2", "MAE", "RMSE", "NSE"]:
            if key in gl and isinstance(gl[key], (int, float)) and np.isfinite(gl[key]):
                fmt = ".3f" if key in {"r", "R2", "NSE"} else ".3g"
                lines.append(f"{key} = {gl[key]:{fmt}}")

    # Show one or more local-window blocks sorted by K
    for key in sorted([k for k in stats.keys() if str(k).startswith("local_window_K")]):
        lw = stats.get(key) or {}
        if not lw:
            continue
        lines.append(f"<b>Local window K={int(lw.get('K', 0))}</b>")
        for kk in ["RMSE", "MAE", "Bias(obs-pred)", "NSE","median_lag_days", "IQR_lag_days", "fraction_zero_lag", "n"]:
            if kk in lw and isinstance(lw[kk], (int, float)) and np.isfinite(lw[kk]):
                if kk in {"fraction_zero_lag"}:
                    lines.append(f"{kk} = {lw[kk] * 100:.1f}%")
                elif kk in {"median_lag_days", "IQR_lag_days", "n"}:
                    if kk == "n":
                        lines.append(f"{kk} = {int(lw[kk])}")
                    else:
                        lines.append(f"{kk} = {lw[kk]:.3g}")
                elif kk in {"NSE"}:
                    lines.append(f"{kk} = {lw[kk]:.3f}")
                else:
                    lines.append(f"{kk} = {lw[kk]:.3g}")

    ex = stats.get("extras", {}) or {}
    if ex:
        lines.append("<b>Extras</b>")
        for name, ed in ex.items():
            r = ed.get("r")
            if isinstance(r, (int, float)) and np.isfinite(r):
                lines.append(f"r (obs vs {name}) = {r:.3f}")

    # Band deviation analysis
    band_dev = stats.get("band_deviation", {}) or {}
    if band_dev:
        lines.append("<b>Band Analysis</b>")
        
        # Min-max band stats (stored with keys like "band_width_mean", "band_width_rel%")
        if "band_width_mean" in band_dev:
            width = band_dev["band_width_mean"]
            if isinstance(width, (int, float)) and np.isfinite(width):
                lines.append(f"Min-max width = {width:.3g}")
        if "band_width_rel%" in band_dev:
            rel_width = band_dev["band_width_rel%"]
            if isinstance(rel_width, (int, float)) and np.isfinite(rel_width):
                lines.append(f"Min-max width (%) = {rel_width:.1f}%")
        if "band_asymmetry" in band_dev:
            asym = band_dev["band_asymmetry"]
            if isinstance(asym, (int, float)) and np.isfinite(asym):
                lines.append(f"Min-max asymmetry = {asym:.3f}")
        if "band_rmse_vs_mean" in band_dev:
            rmse = band_dev["band_rmse_vs_mean"]
            if isinstance(rmse, (int, float)) and np.isfinite(rmse):
                lines.append(f"Min-max RMSE vs mean = {rmse:.3g}")
                
        # Percentile band stats (stored with keys like "p50_band_width_mean", "p90_band_width_mean")
        if "p50_band_width_mean" in band_dev:
            width = band_dev["p50_band_width_mean"]
            if isinstance(width, (int, float)) and np.isfinite(width):
                lines.append(f"50% band width = {width:.3g}")
        if "p90_band_width_mean" in band_dev:
            width = band_dev["p90_band_width_mean"]
            if isinstance(width, (int, float)) and np.isfinite(width):
                lines.append(f"90% band width = {width:.3g}")
        if "p50_band_width_rel%" in band_dev:
            rel_width = band_dev["p50_band_width_rel%"]
            if isinstance(rel_width, (int, float)) and np.isfinite(rel_width):
                lines.append(f"50% band width (%) = {rel_width:.1f}%")
        if "p90_band_width_rel%" in band_dev:
            rel_width = band_dev["p90_band_width_rel%"]
            if isinstance(rel_width, (int, float)) and np.isfinite(rel_width):
                lines.append(f"90% band width (%) = {rel_width:.1f}%")

    # Measured vs series comparisons
    meas_vs = stats.get("measured_vs_series", {}) or {}
    if meas_vs:
        lines.append("<b>Measured vs All Series</b>")
        
        # Parse flat keys like "vs_min_r", "vs_p05_rmse", "vs_extra_external_flow_nse"
        # Group by series name
        series_groups = {}
        for key, val in meas_vs.items():
            if not key.startswith("vs_"):
                continue
            # Extract series name and metric
            # Examples: vs_min_r -> min, r; vs_p05_rmse -> p05, rmse; vs_extra_external_flow_nse -> extra_external_flow, nse
            parts = key[3:].split("_")  # Remove "vs_" prefix
            if len(parts) >= 2:
                metric = parts[-1]  # Last part is always the metric
                series_name = "_".join(parts[:-1])  # Everything before last part is series name
                if series_name not in series_groups:
                    series_groups[series_name] = {}
                series_groups[series_name][metric] = val
        
        # Define display order for series types
        series_order = ["mean", "min", "max", "p05", "p25", "p50", "p75", "p95"]
        
        # Add external series at the end (they have "extra_" prefix)
        for series_name in series_groups.keys():
            if series_name not in series_order:
                series_order.append(series_name)
        
        for series_name in series_order:
            if series_name not in series_groups:
                continue
            series_stats = series_groups[series_name]
            if not series_stats:
                continue
                
            # Format series name for display
            display_name = series_name
            if series_name.startswith("p") and series_name[1:].isdigit():
                display_name = f"{series_name[1:]}{series_name[0]}"  # p05 -> 05p
            elif series_name.startswith("extra_"):
                display_name = series_name[6:]  # Remove "extra_" prefix
            
            lines.append(f"<i>vs {display_name}:</i>")
            
            # Show key metrics
            for metric in ["r", "rmse", "mae", "nse", "bias"]:
                if metric in series_stats:
                    val = series_stats[metric]
                    if isinstance(val, (int, float)) and np.isfinite(val):
                        if metric in {"r", "nse"}:
                            lines.append(f"  {metric} = {val:.3f}")
                        else:
                            lines.append(f"  {metric} = {val:.3g}")

    # Join with HTML <br>
    return "<br>".join(lines)


# -----------------------------
# Public diagnostics: figures for exploration inside/outside dashboard
# -----------------------------

def local_window_matching_detail(
    measured: Sequence[pd.Series], median_series: pd.Series, *, K: int = 1,
    window: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None, strategy: str = "nearest",
) -> Optional[pd.DataFrame]:
    """Return per-observation local matching detail used for diagnostics.

    Columns: ['obs', 'mod_best', 'k_days'] indexed by observation time.
    """
    if median_series is None or median_series.empty or not measured or K is None or K < 0:
        return None
    rows: List[Tuple[pd.Timestamp, float, float, int]] = []
    for s in measured:
        if s is None or s.empty:
            continue
        ss = s
        if window is not None:
            x0, x1 = window
            ss = ss.loc[(ss.index >= x0) & (ss.index <= x1)]
        if ss.empty:
            continue
        for t, o in ss.dropna().items():
            lo = t - pd.Timedelta(days=int(K))
            hi = t + pd.Timedelta(days=int(K))
            win = median_series.loc[lo:hi].dropna()
            if win.empty:
                continue
            if strategy == "mean":
                m_best = float(win.mean())
                k_days = 0
            else:
                diffs = (win - float(o)).abs()
                t_star = diffs.idxmin()
                m_best = float(win.loc[t_star])
                k_days = int((t_star - t) / pd.Timedelta(days=1))
            rows.append((t, float(o), m_best, k_days))
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["time", "obs", "mod_best", "k_days"]).set_index("time").sort_index()
    return df


def build_fit_diagnostics(
    q_df: pd.DataFrame,
    measured_series: Sequence[pd.Series],
    *,
    window: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    template: str = "plotly_white",
    title: Optional[str] = None,
    lag_hist_K: int = 1,
) -> Dict[str, go.Figure]:
    """Build a small set of diagnostic figures to evaluate fit and error structure.

    Returns a dict of Plotly figures: keys include 'obs_vs_pred', 'resid_hist', 'resid_vs_pred', 'lag_hist'.
    """
    Y, M, qdict = _collect_pairs(q_df, measured_series, window=window)
    figs: Dict[str, go.Figure] = {}
    if Y.size == 0:
        return figs

    # Observed vs Predicted (median) scatter
    fig_sc = go.Figure(layout=dict(template=template))
    fig_sc.add_trace(go.Scatter(x=M, y=Y, mode="markers", name="pred vs obs",
                                marker=dict(color="#1f77b4", size=6, opacity=0.8)))
    # 1:1 line
    lo = float(np.nanmin([np.nanmin(Y), np.nanmin(M)]))
    hi = float(np.nanmax([np.nanmax(Y), np.nanmax(M)]))
    if lo == hi:
        hi = lo + 1.0
    fig_sc.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="1:1",
                                line=dict(color="#444", dash="dash")))
    # OLS regression on the filtered pairs (Y vs M)
    try:
        x = np.asarray(M, dtype=float)
        y = np.asarray(Y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.any() and np.sum(mask) >= 2:
            b1, b0 = np.polyfit(x[mask], y[mask], deg=1)
            xx = np.linspace(lo, hi, 100)
            yy = b1 * xx + b0
            fig_sc.add_trace(go.Scatter(x=xx, y=yy, mode="lines", name=f"OLS fit (y={b1:.3g}x+{b0:.3g})",
                                        line=dict(color="#ff7f0e", width=2)))
    except Exception:
        pass
    fig_sc.update_layout(title=(title or "Observed vs Predicted (median)"), xaxis_title="Predicted (median)", yaxis_title="Observed")
    figs["obs_vs_pred"] = fig_sc

    # Residuals histogram (obs - pred)
    resid = Y - M
    fig_hist = go.Figure(layout=dict(template=template))
    fig_hist.add_trace(go.Histogram(x=resid, nbinsx=30, marker=dict(color="#2ca02c"), name="residuals"))
    fig_hist.add_shape(type="line", x0=0, x1=0, y0=0, y1=1, xref="x", yref="paper", line=dict(color="#444", dash="dash"))
    mu = float(np.nanmean(resid)) if resid.size else float("nan")
    sd = float(np.nanstd(resid)) if resid.size else float("nan")
    sk = float(np.nanmean(((resid - mu) / sd) ** 3)) if resid.size and sd not in (0, float("nan")) else float("nan")
    fig_hist.update_layout(title=f"Residuals (obs - pred): mean={mu:.3g}, sd={sd:.3g}, skew={sk:.3g}", xaxis_title="Residual", yaxis_title="Count")
    figs["resid_hist"] = fig_hist

    # Residual vs Predicted scatter
    fig_rv = go.Figure(layout=dict(template=template))
    fig_rv.add_trace(go.Scatter(x=M, y=resid, mode="markers", name="resid vs pred",
                                marker=dict(color="#9467bd", size=6, opacity=0.8)))
    fig_rv.add_shape(type="line", x0=float(np.nanmin(M)), x1=float(np.nanmax(M)), y0=0, y1=0, xref="x", yref="y",
                     line=dict(color="#444", dash="dash"))
    fig_rv.update_layout(title="Residual vs Predicted (median)", xaxis_title="Predicted (median)", yaxis_title="Residual (obs - pred)")
    figs["resid_vs_pred"] = fig_rv

    # Local window lag histogram
    median_series = q_df["p50"] if (q_df is not None and "p50" in q_df.columns) else None
    if median_series is not None and not median_series.empty and lag_hist_K is not None and lag_hist_K > 0:
        df_lag = local_window_matching_detail(measured_series, median_series, K=int(lag_hist_K), window=window, strategy="nearest")
        if df_lag is not None and not df_lag.empty:
            fig_lag = go.Figure(layout=dict(template=template))
            fig_lag.add_trace(go.Histogram(x=df_lag["k_days"], nbinsx=2 * int(lag_hist_K) + 1, marker=dict(color="#e377c2")))
            fig_lag.update_layout(title=f"Local matching lags (K={int(lag_hist_K)})", xaxis_title="Lag (days)", yaxis_title="Count")
            figs["lag_hist"] = fig_lag

    return figs


# -----------------------------
# Convenience: compute stats and diagnostics from raw inputs (outside dashboard)
# -----------------------------

def evaluate_fit(
    sim_dfs: Dict[str, pd.DataFrame],
    var: str,
    reach: Union[int, str],
    *,
    freq: str = "1D",
    method: str = "mean",
    date_col: str = "date",
    reach_col: str = "RCH",
    flow_col: Optional[str] = None,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    season_months: Optional[Sequence[int]] = None,
    measured_df: Optional[pd.DataFrame] = None,
    measured_name: Optional[Union[str, Sequence[str]]] = None,
    measured_name_col: str = "NOMBRE",
    measured_date_col: str = "F_MUESTREO",
    measured_value_col: Optional[str] = None,
    measured_station_col: str = "est_estaci",
    selected_stations: Optional[Sequence[str]] = None,
    template: str = "plotly_white",
    extras: Optional[Dict[str, pd.Series]] = None,
    compute_log: bool = True,
    max_global_lag: int = 2,
    local_window_ks: Sequence[int] = (1, 2),
    local_strategy: str = "nearest",
    choose_best_lag_by: str = "r",
) -> Dict[str, object]:
    """Build quantiles and measured series, then compute stats and diagnostics.

    Returns dict with keys: 'q_df', 'measured_series', 'stats', 'figs'.
    """
    # Helpers
    def _ensure_dt_index(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col).sort_index()
        return df

    def _slice_time(df: pd.DataFrame) -> pd.DataFrame:
        if start is None and end is None:
            return df
        s = pd.to_datetime(start) if start is not None else None
        e = pd.to_datetime(end) if end is not None else None
        if s is not None and e is not None:
            return df.loc[(df.index >= s) & (df.index <= e)]
        elif s is not None:
            return df.loc[df.index >= s]
        elif e is not None:
            return df.loc[df.index <= e]
        return df

    def _filter_season(df: pd.DataFrame) -> pd.DataFrame:
        if not season_months:
            return df
        months = set(int(m) for m in season_months)
        return df.loc[df.index.month.isin(months)]

    def _resample_series(df: pd.DataFrame, value_col: str, *, freq: str, how: str, flow_col: Optional[str]) -> pd.Series:
        how = str(how)
        if how == "sum":
            return df[value_col].resample(freq).sum(min_count=1)
        elif how == "flow_weighted_mean":
            if flow_col is None or flow_col not in df.columns:
                return df[value_col].resample(freq).mean()
            num = (df[value_col] * df[flow_col]).resample(freq).sum(min_count=1)
            den = df[flow_col].resample(freq).sum(min_count=1)
            with np.errstate(invalid='ignore', divide='ignore'):
                out = num / den
            return out
        else:
            return df[value_col].resample(freq).mean()

    # Build per-simulation series
    per_sim: Dict[str, pd.Series] = {}
    for name, df in sim_dfs.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        if any(c not in df.columns for c in (reach_col, date_col, var)):
            continue
        sub = df[df[reach_col] == reach][[date_col, var] + ([flow_col] if flow_col and flow_col in df.columns else [])].copy()
        if sub.empty:
            continue
        sub = _ensure_dt_index(sub, date_col)
        sub = _slice_time(sub)
        sub = _filter_season(sub)
        if sub.empty:
            continue
        s = _resample_series(sub, var, freq=freq, how=method, flow_col=flow_col if flow_col in sub.columns else None)
        s.name = str(name)
        per_sim[str(name)] = s

    if not per_sim:
        q_df = pd.DataFrame()
    else:
        aligned_df = pd.concat(per_sim.values(), axis=1).sort_index()
        arr = aligned_df.to_numpy(dtype=float)
        percs = [5, 10, 25, 50, 75, 90, 95]
        if arr.shape[1] == 0:
            q_df = pd.DataFrame()
        else:
            qs = np.nanpercentile(arr, percs, axis=1)
            q = {p: qs[i, :] for i, p in enumerate(percs)}
            q_df = pd.DataFrame({
                "p05": q[5], "p10": q[10], "p25": q[25], "p50": q[50], "p75": q[75], "p90": q[90], "p95": q[95]
            }, index=aligned_df.index)

    # Build measured series list (per station), if provided
    measured_series: List[pd.Series] = []
    if isinstance(measured_df, pd.DataFrame) and not measured_df.empty and measured_value_col and measured_station_col in measured_df.columns and measured_date_col in measured_df.columns:
        df = measured_df.copy()
        if measured_name is not None:
            if isinstance(measured_name, (list, tuple, set)):
                names = [str(x) for x in measured_name]
                if measured_name_col in df.columns:
                    df = df[df[measured_name_col].astype(str).isin(names)]
            else:
                nm = str(measured_name)
                if measured_name_col in df.columns:
                    df = df[df[measured_name_col].astype(str) == nm]
        df[measured_date_col] = pd.to_datetime(df[measured_date_col])
        if selected_stations:
            sel = set(str(s) for s in selected_stations)
            df = df[df[measured_station_col].astype(str).isin(sel)]
        if start is not None:
            df = df[df[measured_date_col] >= pd.to_datetime(start)]
        if end is not None:
            df = df[df[measured_date_col] <= pd.to_datetime(end)]
        if season_months:
            months = set(int(m) for m in season_months)
            df = df[df[measured_date_col].dt.month.isin(months)]
        if not df.empty:
            # If multiple names provided, keep them separate in grouping
            if isinstance(measured_name, (list, tuple, set)) and measured_name_col in df.columns:
                grp = (df.groupby([df[measured_station_col].astype(str), df[measured_name_col].astype(str), df[measured_date_col].dt.floor('D')])[measured_value_col]
                         .mean().sort_index())
                for (station, _nm) in grp.index.droplevel(2).unique():
                    s = grp.xs((station, _nm))
                    s.index.name = None
                    s = s.resample(freq).mean().dropna()
                    measured_series.append(s)
            else:
                grp = (df.groupby([df[measured_station_col].astype(str), df[measured_date_col].dt.floor('D')])[measured_value_col]
                         .mean().sort_index())
                for station in grp.index.get_level_values(0).unique():
                    s = grp.xs(station, level=0)
                    s.index.name = None
                    s = s.resample(freq).mean().dropna()
                    measured_series.append(s)

    # Compute stats and diagnostics in the given (start,end) window
    window = None
    if start is not None or end is not None:
        x0 = pd.to_datetime(start) if start is not None else (q_df.index.min() if not q_df.empty else None)
        x1 = pd.to_datetime(end) if end is not None else (q_df.index.max() if not q_df.empty else None)
        if x0 is not None and x1 is not None:
            window = (x0, x1)

    stats = compute_stats_for_view(
        q_df,
        measured_series,
        window=window,
        extras=extras,
        compute_log=compute_log,
        max_global_lag=max_global_lag,
        local_window_ks=local_window_ks,
        local_strategy=local_strategy,
        choose_best_lag_by=choose_best_lag_by,
    )

    figs = build_fit_diagnostics(
        q_df,
        measured_series,
        window=window,
        template=template,
        title=f"Diagnostics: {var} (Reach {reach})",
        lag_hist_K=int(local_window_ks[0]) if local_window_ks else 1,
    )

    return {
        "q_df": q_df,
        "measured_series": measured_series,
        "stats": stats,
        "figs": figs,
    }
