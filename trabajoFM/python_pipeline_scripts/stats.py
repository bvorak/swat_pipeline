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

def _index_of_agreement(y: np.ndarray, m: np.ndarray) -> float:
    "Willmott index of agreement (d)."
    y, m = _finite_pairs(y, m)
    if y.size == 0:
        return float('nan')
    o_bar = np.nanmean(y)
    denom = np.nansum((np.abs(m - o_bar) + np.abs(y - o_bar)) ** 2)
    if not np.isfinite(denom) or denom == 0:
        return float('nan')
    num = np.nansum((y - m) ** 2)
    return float(1.0 - num / denom)


def _index_of_agreement_from_series(y: np.ndarray, m: np.ndarray, *, scale: Optional[float] = None) -> float:
    y, m = _finite_pairs(y, m)
    if y.size == 0:
        return float('nan')
    if scale is not None:
        if not np.isfinite(scale) or scale == 0:
            return float('nan')
        y = y / scale
        m = m / scale
    return _index_of_agreement(y, m)


def _index_of_agreement_log(y: np.ndarray, m: np.ndarray) -> float:
    y, m = _finite_pairs(y, m)
    mask = (y > 0) & (m > 0) & np.isfinite(y) & np.isfinite(m)
    if not np.any(mask):
        return float('nan')
    y_log = np.log10(y[mask])
    m_log = np.log10(m[mask])
    return _index_of_agreement(y_log, m_log)


def _index_of_agreement_relative(y: np.ndarray, m: np.ndarray) -> float:
    "Relative form using absolute-mean scaling of observations."
    y, m = _finite_pairs(y, m)
    if y.size == 0:
        return float('nan')
    mean_abs = np.nanmean(np.abs(y))
    if not np.isfinite(mean_abs) or mean_abs == 0:
        return float('nan')
    return _index_of_agreement_from_series(y, m, scale=mean_abs)


def _nse_relative(y: np.ndarray, m: np.ndarray) -> float:
    y, m = _finite_pairs(y, m)
    if y.size == 0:
        return float('nan')
    mean_abs = np.nanmean(np.abs(y))
    if not np.isfinite(mean_abs) or mean_abs == 0:
        return float('nan')
    y_norm = y / mean_abs
    m_norm = m / mean_abs
    return _nse(y_norm, m_norm)


def _nse_log(y: np.ndarray, m: np.ndarray) -> float:
    y, m = _finite_pairs(y, m)
    mask = (y > 0) & (m > 0) & np.isfinite(y) & np.isfinite(m)
    if not np.any(mask):
        return float('nan')
    y_log = np.log10(y[mask])
    m_log = np.log10(m[mask])
    return _nse(y_log, m_log)


def _rsr(y: np.ndarray, m: np.ndarray) -> float:
    "RSR = RMSE / SD_obs (population form consistent with Moriasi eq. 3)."
    y, m = _finite_pairs(y, m)
    if y.size == 0:
        return float('nan')
    sse = np.nansum((y - m) ** 2)
    o_bar = np.nanmean(y)
    sso = np.nansum((y - o_bar) ** 2)
    if not np.isfinite(sso) or sso == 0:
        return float('nan')
    return float(np.sqrt(sse) / np.sqrt(sso))


def _distribution_summary(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "mean": float('nan'),
            "median": float('nan'),
            "p05": float('nan'),
            "p25": float('nan'),
            "p50": float('nan'),
            "p75": float('nan'),
            "p95": float('nan'),
            "sd": float('nan'),
            "var": float('nan'),
            "n": 0.0,
        }
    out = {
        "mean": float(np.nanmean(arr)),
        "median": float(np.nanmedian(arr)),
        "p05": float(np.nanpercentile(arr, 5)),
        "p25": float(np.nanpercentile(arr, 25)),
        "p50": float(np.nanpercentile(arr, 50)),
        "p75": float(np.nanpercentile(arr, 75)),
        "p95": float(np.nanpercentile(arr, 95)),
        "sd": float(np.nanstd(arr)),
        "var": float(np.nanvar(arr)),
        "n": float(arr.size),
    }
    return out



def _duration_curve_from_series(series: pd.Series, levels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if not isinstance(series, pd.Series):
        return levels, np.full_like(levels, np.nan, dtype=float)
    arr = series.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return levels, np.full_like(levels, np.nan, dtype=float)
    values = np.percentile(arr, 100 - levels)
    return levels, values


def _empirical_exceedance(values: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.array([]), np.array([])
    arr_sorted = np.sort(arr)[::-1]
    exceed = (np.arange(1, arr_sorted.size + 1) / (arr_sorted.size + 1)) * 100.0
    return exceed, arr_sorted
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


def _normalize_band_groups(band_data: Optional[Dict[str, object]]) -> Dict[str, Dict[str, pd.Series]]:
    """Normalize band data into a dict[group -> dict[band_name -> series]]."""
    if not isinstance(band_data, dict) or not band_data:
        return {}
    # If already flat (series only), treat as single ensemble group
    if all(isinstance(v, pd.Series) for v in band_data.values()):
        return {"ensemble": {str(k): v for k, v in band_data.items() if isinstance(v, pd.Series)}}
    normalized: Dict[str, Dict[str, pd.Series]] = {}
    for group_key, value in band_data.items():
        if isinstance(value, dict):
            inner = {str(k): v for k, v in value.items() if isinstance(v, pd.Series)}
            if inner:
                normalized[str(group_key)] = inner
        elif isinstance(value, pd.Series):
            normalized.setdefault("ensemble", {})[str(group_key)] = value
    return {g: bands for g, bands in normalized.items() if bands}



def _measured_vs_series_stats(measured: np.ndarray, target_series: np.ndarray, series_name: str) -> Dict[str, float]:
    """Compute comprehensive stats comparing measured data against any target time series."""
    y, m = _finite_pairs(measured, target_series)
    if y.size < 2:
        return {
            f"vs_{series_name}_r": float("nan"),
            f"vs_{series_name}_rmse": float("nan"),
            f"vs_{series_name}_mae": float("nan"),
            f"vs_{series_name}_nse": float("nan"),
            f"vs_{series_name}_NSE_rel": float("nan"),
            f"vs_{series_name}_d": float("nan"),
            f"vs_{series_name}_d_rel": float("nan"),
            f"vs_{series_name}_RSR": float("nan"),
            f"vs_{series_name}_bias": float("nan"),
        }

    return {
        f"vs_{series_name}_r": _pearson_r(y, m),
        f"vs_{series_name}_rmse": _rmse(y, m),
        f"vs_{series_name}_mae": _mae(y, m),
        f"vs_{series_name}_nse": _nse(y, m),
        f"vs_{series_name}_NSE_rel": _nse_relative(y, m),
        f"vs_{series_name}_d": _index_of_agreement(y, m),
        f"vs_{series_name}_d_rel": _index_of_agreement_relative(y, m),
        f"vs_{series_name}_RSR": _rsr(y, m),
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


def _calculate_data_usage(
    q_df: pd.DataFrame,
    measured_series: Sequence[pd.Series],
    *,
    window: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
) -> Dict[str, Union[int, float]]:
    """Calculate statistics on data usage/filtering for transparency.
    
    Returns percentage of available days actually used in calculations.
    """
    usage_stats = {}
    
    if q_df is None or q_df.empty or not measured_series:
        return {
            "sim_days_total": 0,
            "sim_days_used": 0,
            "sim_usage_percent": 0.0,
            "measured_days_total": 0,
            "measured_days_used": 0,
            "measured_usage_percent": 0.0,
            "paired_days_used": 0,
            "paired_usage_percent": 0.0,
        }
    
    # Calculate simulation data availability
    sim_total_days = len(q_df.index)
    sim_finite_days = q_df.notna().any(axis=1).sum()  # Days with any finite simulation data
    
    # Apply window filtering to simulation data
    if window is not None:
        x0, x1 = window
        q_windowed = q_df.loc[(q_df.index >= x0) & (q_df.index <= x1)]
        sim_windowed_days = len(q_windowed.index)
        sim_windowed_finite = q_windowed.notna().any(axis=1).sum()
    else:
        sim_windowed_days = sim_total_days
        sim_windowed_finite = sim_finite_days
    
    # Calculate measured data availability
    measured_total_days = 0
    measured_finite_days = 0
    measured_windowed_days = 0
    measured_windowed_finite = 0
    
    for s in measured_series:
        if s is None or s.empty:
            continue
        measured_total_days = max(measured_total_days, len(s.index))
        measured_finite_days = max(measured_finite_days, s.notna().sum())
        
        # Apply window filtering
        if window is not None:
            x0, x1 = window
            s_windowed = s.loc[(s.index >= x0) & (s.index <= x1)]
            measured_windowed_days = max(measured_windowed_days, len(s_windowed.index))
            measured_windowed_finite = max(measured_windowed_finite, s_windowed.notna().sum())
        else:
            measured_windowed_days = measured_total_days
            measured_windowed_finite = measured_finite_days
    
    # Calculate paired data (used in actual statistics)
    Y, M, _ = _collect_pairs(q_df, measured_series, window=window)
    paired_days_used = int(Y.size)
    
    return {
        "sim_days_total": int(sim_total_days),
        "sim_days_finite": int(sim_finite_days),
        "sim_days_windowed": int(sim_windowed_days),
        "sim_days_windowed_finite": int(sim_windowed_finite),
        "sim_usage_percent": float(sim_windowed_finite / sim_total_days * 100) if sim_total_days > 0 else 0.0,
        
        "measured_days_total": int(measured_total_days),
        "measured_days_finite": int(measured_finite_days),
        "measured_days_windowed": int(measured_windowed_days),
        "measured_days_windowed_finite": int(measured_windowed_finite),
        "measured_usage_percent": float(measured_windowed_finite / measured_total_days * 100) if measured_total_days > 0 else 0.0,
        
        "paired_days_used": paired_days_used,
        "paired_usage_percent": float(paired_days_used / max(sim_total_days, measured_total_days) * 100) if max(sim_total_days, measured_total_days) > 0 else 0.0,
    }


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
    band_groups = _normalize_band_groups(band_data)
    windowed_groups: Dict[str, Dict[str, pd.Series]] = {}

    def _apply_window_to_series(s: pd.Series) -> pd.Series:
        if not isinstance(s, pd.Series) or s.empty:
            return s
        if window is not None:
            x0, x1 = window
            return s.loc[(s.index >= x0) & (s.index <= x1)]
        return s

    Y, M, qdict = _collect_pairs(q_df, measured_series, window=window)
    n = int(Y.size)
    stats["n"] = n
    
    # Calculate data usage statistics
    data_usage = _calculate_data_usage(q_df, measured_series, window=window)
    stats["data_usage"] = data_usage
    
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
        "NSE_rel": _nse_relative(Y, M),
        "KGE": kge,
        "MedAE": medae,
        "d": _index_of_agreement(Y, M),
        "d_rel": _index_of_agreement_relative(Y, M),
        "RSR": _rsr(Y, M),
    }
    if "p25" in qdict and "p75" in qdict:
        same_day["coverage50"] = _coverage(Y, qdict["p25"], qdict["p75"])  # fraction
    if "p05" in qdict and "p95" in qdict:
        same_day["coverage90"] = _coverage(Y, qdict["p05"], qdict["p95"])  # fraction
    stats["same_day"] = same_day

    paired_summary = {
        "observed": _distribution_summary(Y),
        "predicted": _distribution_summary(M),
    }
    full_pred_series = None
    if q_df is not None and "p50" in q_df.columns:
        full_pred_series = q_df["p50"].copy()
        if window is not None:
            x0, x1 = window
            full_pred_series = full_pred_series.loc[(full_pred_series.index >= x0) & (full_pred_series.index <= x1)]
        full_pred_series = full_pred_series.dropna()
    observed_full_values: list[float] = []
    for _s in measured_series:
        if _s is None or _s.empty:
            continue
        ss = _s
        if window is not None:
            x0, x1 = window
            ss = ss.loc[(ss.index >= x0) & (ss.index <= x1)]
        observed_full_values.extend(ss.dropna().to_numpy(dtype=float).tolist())
    distribution_block = {
        "paired": paired_summary,
        "observed_full": _distribution_summary(np.array(observed_full_values, dtype=float)),
        "predicted_full": _distribution_summary(full_pred_series.to_numpy(dtype=float)) if isinstance(full_pred_series, pd.Series) and not full_pred_series.empty else _distribution_summary(np.array([], dtype=float)),
    }
    stats["distribution_summary"] = distribution_block


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
                "NSE_log": _nse_log(Y, M),
                "d_log": _index_of_agreement_log(Y, M),
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
    if band_groups:
        band_stats = {}
        windowed_groups = {
            str(group_key): {
                str(band_key): _apply_window_to_series(series)
                for band_key, series in bands.items()
                if isinstance(series, pd.Series)
            }
            for group_key, bands in band_groups.items()
        }
        ensemble_bands = windowed_groups.get("ensemble", {})
        if all(k in ensemble_bands for k in ["mean", "min", "max"]):
            mean_series = ensemble_bands["mean"]
            min_series = ensemble_bands["min"]
            max_series = ensemble_bands["max"]
            if all(isinstance(s, pd.Series) and not s.empty for s in (mean_series, min_series, max_series)):
                idx_common = mean_series.dropna().index
                idx_common = idx_common.intersection(min_series.dropna().index)
                idx_common = idx_common.intersection(max_series.dropna().index)
                if len(idx_common) > 0:
                    mean_vals = mean_series.reindex(idx_common).to_numpy(dtype=float)
                    min_vals = min_series.reindex(idx_common).to_numpy(dtype=float)
                    max_vals = max_series.reindex(idx_common).to_numpy(dtype=float)
                    band_stats.update(_band_deviation_stats(mean_vals, min_vals, max_vals))
        if all(k in ensemble_bands for k in ["mean", "p25", "p75"]):
            mean_series = ensemble_bands["mean"]
            p25_series = ensemble_bands["p25"]
            p75_series = ensemble_bands["p75"]
            if all(isinstance(s, pd.Series) and not s.empty for s in (mean_series, p25_series, p75_series)):
                idx_common = mean_series.dropna().index
                idx_common = idx_common.intersection(p25_series.dropna().index)
                idx_common = idx_common.intersection(p75_series.dropna().index)
                if len(idx_common) > 0:
                    mean_vals = mean_series.reindex(idx_common).to_numpy(dtype=float)
                    p25_vals = p25_series.reindex(idx_common).to_numpy(dtype=float)
                    p75_vals = p75_series.reindex(idx_common).to_numpy(dtype=float)
                    p05_vals = p95_vals = None
                    if "p05" in ensemble_bands and "p95" in ensemble_bands:
                        p05_series = ensemble_bands["p05"]
                        p95_series = ensemble_bands["p95"]
                        if isinstance(p05_series, pd.Series) and isinstance(p95_series, pd.Series):
                            p05_vals = p05_series.reindex(idx_common).to_numpy(dtype=float)
                            p95_vals = p95_series.reindex(idx_common).to_numpy(dtype=float)
                    perc_stats = _percentile_band_deviation_stats(mean_vals, p25_vals, p75_vals, p05_vals, p95_vals)
                    band_stats.update(perc_stats)
        if band_stats:
            stats["band_deviation"] = band_stats


    # NEW: Measured vs all available series
    measured_vs_series: Dict[str, float] = {}
    coverage_entries: Dict[str, Dict[str, float]] = {}
    coverage_debug: Dict[str, List[str]] = {}

    def _record_cov_debug(label: object, message: str) -> None:
        coverage_debug.setdefault(str(label), []).append(message)

    if measured_series:
        all_measured_values: List[float] = []
        all_measured_times: List[pd.Timestamp] = []
        for s in measured_series:
            if s is None or s.empty:
                continue
            ss = s
            if window is not None:
                x0, x1 = window
                ss = ss.loc[(ss.index >= x0) & (ss.index <= x1)]
            for t, val in ss.dropna().items():
                all_measured_values.append(float(val))
                all_measured_times.append(t)
        if not all_measured_values:
            _record_cov_debug('global', 'no measured data available after filtering for coverage analysis')
        else:
            measured_times = pd.Index(all_measured_times)
            measured_vals = np.array(all_measured_values, dtype=float)
            tol = pd.Timedelta(days=1)
            if band_groups and windowed_groups:
                fraction_lookup = {"p05": 0.05, "p25": 0.25, "p50": 0.50, "p75": 0.75, "p95": 0.95}
                def _resolve_band_series(label_hint: str, bands_dict: Dict[str, pd.Series], key: str) -> Optional[pd.Series]:
                    series = bands_dict.get(key)
                    if isinstance(series, pd.Series) and not series.empty:
                        return series
                    frac = fraction_lookup.get(key)
                    if frac is None:
                        return None
                    min_series = bands_dict.get("min")
                    max_series = bands_dict.get("max")
                    if not isinstance(min_series, pd.Series) or min_series.empty or not isinstance(max_series, pd.Series) or max_series.empty:
                        return None
                    idx = min_series.index.union(max_series.index)
                    min_aligned = min_series.reindex(idx)
                    max_aligned = max_series.reindex(idx)
                    approx = min_aligned + (max_aligned - min_aligned) * frac
                    bands_dict[key] = approx
                    _record_cov_debug(label_hint, f"approximated {key} using min/max (frac={frac:.2f})")
                    return approx
                for group_key, bands in windowed_groups.items():
                    if not isinstance(bands, dict) or not bands:
                        _record_cov_debug(group_key, 'no band series present')
                        continue
                    center_key = 'mean' if 'mean' in bands else ('p50' if 'p50' in bands else None)
                    if center_key is None:
                        _record_cov_debug(group_key, 'no mean/p50 series available')
                        continue
                    center_series = bands.get(center_key)
                    if not isinstance(center_series, pd.Series) or center_series.empty:
                        _record_cov_debug(group_key, f'{center_key} series empty')
                        continue
                    aligned_center = center_series.reindex(measured_times, method='nearest', tolerance=tol)
                    center_vals = aligned_center.to_numpy(dtype=float)
                    valid_mask = np.isfinite(center_vals) & np.isfinite(measured_vals)
                    if not np.any(valid_mask):
                        _record_cov_debug(group_key, 'no overlapping measured points with center series')
                        continue
                    label = center_key if group_key == 'ensemble' else f"{group_key}_{center_key}"
                    def _add_coverage(lo_key: str, hi_key: str, suffix: str) -> None:
                        lo_series = bands.get(lo_key)
                        if not isinstance(lo_series, pd.Series) or lo_series.empty:
                            lo_series = _resolve_band_series(label, bands, lo_key)
                        hi_series = bands.get(hi_key)
                        if not isinstance(hi_series, pd.Series) or hi_series.empty:
                            hi_series = _resolve_band_series(label, bands, hi_key)
                        if lo_series is None or hi_series is None or not isinstance(lo_series, pd.Series) or not isinstance(hi_series, pd.Series) or lo_series.empty or hi_series.empty:
                            _record_cov_debug(label, f'skip {suffix}: band series {lo_key}/{hi_key} unavailable after fallback')
                            return
                        aligned_lo = lo_series.reindex(measured_times, method='nearest', tolerance=tol)
                        aligned_hi = hi_series.reindex(measured_times, method='nearest', tolerance=tol)
                        lo_vals = aligned_lo.to_numpy(dtype=float)
                        hi_vals = aligned_hi.to_numpy(dtype=float)
                        mask = valid_mask & np.isfinite(lo_vals) & np.isfinite(hi_vals)
                        count = int(np.count_nonzero(mask))
                        if count == 0:
                            _record_cov_debug(label, f'skip {suffix}: no finite band values aligned with measured points')
                            return
                        cov = _coverage(measured_vals[mask], lo_vals[mask], hi_vals[mask])
                        if np.isfinite(cov):
                            coverage_entries.setdefault(label, {})[suffix] = float(cov)
                            _record_cov_debug(label, f'{suffix}: coverage={cov:.3f} over {count} points')
                        else:
                            _record_cov_debug(label, f'skip {suffix}: coverage returned non-finite value')
                    _add_coverage('p25', 'p75', 'coverage50')
                    _add_coverage('p05', 'p95', 'coverage90')
                    _add_coverage('min', 'max', 'coverage_minmax')
            else:
                _record_cov_debug('global', 'no band data available for coverage analysis')
            if coverage_entries:
                for label, metrics in coverage_entries.items():
                    for suffix, value in metrics.items():
                        measured_vs_series[f"vs_{label}_{suffix}"] = value
            else:
                _record_cov_debug('global', 'coverage metrics were not computed for any series')
    else:
        _record_cov_debug('global', 'no measured series provided for coverage analysis')

    if measured_vs_series:
        stats["measured_vs_series"] = measured_vs_series
    if coverage_debug:
        stats["coverage_debug"] = coverage_debug


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
    
    # Add data usage information
    data_usage = stats.get("data_usage", {}) or {}
    if data_usage:
        paired_pct = data_usage.get("paired_usage_percent", 0.0)
        sim_pct = data_usage.get("sim_usage_percent", 0.0)
        meas_pct = data_usage.get("measured_usage_percent", 0.0)
        
        lines.append(f"data usage = {paired_pct:.1f}% of available days")
        if data_usage.get("sim_days_total", 0) > 0:
            lines.append(f"sim coverage = {sim_pct:.1f}% ({data_usage.get('sim_days_windowed_finite', 0)}/{data_usage.get('sim_days_total', 0)} days)")
        if data_usage.get("measured_days_total", 0) > 0:
            lines.append(f"measured coverage = {meas_pct:.1f}% ({data_usage.get('measured_days_windowed_finite', 0)}/{data_usage.get('measured_days_total', 0)} days)")

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
        for key in ["r_log", "R2_log", "MAElog10", "RMSElog10", "NSElog10", "NSE_log", "d_log", "n_pos"]:
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

    dist_summary = stats.get("distribution_summary", {}) or {}
    if dist_summary:
        lines.append("<b>Distribution Summary</b>")
        paired = dist_summary.get("paired", {}) or {}
        for label, summary in [("Observed (paired)", paired.get("observed")), ("Predicted (paired)", paired.get("predicted"))]:
            if isinstance(summary, dict):
                lines.append(f"{label}:")
                for key in ["mean", "median", "p05", "p25", "p50", "p75", "p95", "sd", "var", "n"]:
                    val = summary.get(key)
                    if isinstance(val, (int, float)) and np.isfinite(val):
                        if key == "n":
                            lines.append(f"  {key} = {int(val)}")
                        else:
                            lines.append(f"  {key} = {val:.3g}")
        obs_full = dist_summary.get("observed_full")
        if isinstance(obs_full, dict):
            lines.append("Observed (full series):")
            for key in ["mean", "median", "sd", "var"]:
                val = obs_full.get(key)
                if isinstance(val, (int, float)) and np.isfinite(val):
                    lines.append(f"  {key} = {val:.3g}")
        pred_full = dist_summary.get("predicted_full")
        if isinstance(pred_full, dict):
            lines.append("Predicted (full series):")
            for key in ["mean", "median", "sd", "var"]:
                val = pred_full.get(key)
                if isinstance(val, (int, float)) and np.isfinite(val):
                    lines.append(f"  {key} = {val:.3g}")

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
        
        # Parse flat keys like "vs_mean_coverage50"
        series_groups = {}
        for key, val in meas_vs.items():
            if not key.startswith("vs_"):
                continue
            parts = key[3:].split("_")  # Remove "vs_" prefix
            if len(parts) >= 2:
                metric = parts[-1]
                series_name = "_".join(parts[:-1])
                series_groups.setdefault(series_name, {})[metric] = val
        
        series_order = ["mean", "min", "max", "p05", "p25", "p50", "p75", "p95"]
        for series_name in series_groups.keys():
            if series_name not in series_order:
                series_order.append(series_name)
        
        metric_defs_numeric = [
            ("r", "r"),
            ("nse", "NSE"),
            ("NSE_rel", "NSE (relative)"),
            ("d", "Index of agreement"),
            ("d_rel", "Index of agreement (relative)"),
            ("RSR", "RSR"),
            ("rmse", "RMSE"),
            ("mae", "MAE"),
            ("bias", "Bias (obs-pred)"),
        ]
        metric_defs_coverage = [
            ("coverage50", "within 50% band"),
            ("coverage90", "within 90% band"),
            ("coverage_minmax", "within min-max envelope"),
        ]
        for series_name in series_order:
            if series_name not in series_groups:
                continue
            series_stats = series_groups[series_name]
            if not series_stats:
                continue
            display_name = series_name
            if series_name.startswith("p") and series_name[1:].isdigit():
                display_name = f"{series_name[1:]}{series_name[0]}"  # p05 -> 05p
            elif series_name.startswith("extra_"):
                display_name = series_name[6:]
            lines.append(f"<i>vs {display_name}:</i>")
            for metric, label_text in metric_defs_numeric:
                val = series_stats.get(metric)
                if isinstance(val, (int, float)) and np.isfinite(val):
                    lines.append(f"  {label_text} = {val:.3g}")
            for metric, label_text in metric_defs_coverage:
                val = series_stats.get(metric)
                if isinstance(val, (int, float)) and np.isfinite(val):
                    lines.append(f"  {label_text} = {val * 100:.1f}%")

    coverage_debug = stats.get("coverage_debug", {}) or {}
    if coverage_debug:
        lines.append("<b>Coverage Debug</b>")
        for label, messages in coverage_debug.items():
            lines.append(f"{label}:")
            for msg in messages:
                lines.append(f"  {msg}")

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
    compare_mode: str = "load",
) -> Dict[str, go.Figure]:
    """Build a small set of diagnostic figures to evaluate fit and error structure.

    Returns a dict of Plotly figures: keys include 'obs_vs_pred', 'resid_hist', 'resid_vs_pred', 'lag_hist', 'load_duration_curve'.

    compare_mode controls whether load-oriented diagnostics are included (set to "load" to add the load duration curve).
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

    if compare_mode == "load":
        try:
            levels = np.linspace(1.0, 99.0, 99)
            x0 = x1 = None
            if window is not None:
                x0, x1 = window
            median_series = None
            if q_df is not None and "p50" in q_df.columns:
                median_series = q_df["p50"].copy()
                if window is not None:
                    median_series = median_series.loc[(median_series.index >= x0) & (median_series.index <= x1)]
            if isinstance(median_series, pd.Series):
                median_series = median_series.dropna()
            if isinstance(median_series, pd.Series) and not median_series.empty:
                fig_ldc = go.Figure(layout=dict(template=template))
                x_median, y_median = _duration_curve_from_series(median_series, levels)
                band_added = False
                if q_df is not None and all(col in q_df.columns for col in ("p05", "p95")):
                    p95_series = q_df["p95"].copy()
                    p05_series = q_df["p05"].copy()
                    if window is not None:
                        p95_series = p95_series.loc[(p95_series.index >= x0) & (p95_series.index <= x1)]
                        p05_series = p05_series.loc[(p05_series.index >= x0) & (p05_series.index <= x1)]
                    x_band, y_p95 = _duration_curve_from_series(p95_series, levels)
                    _, y_p05 = _duration_curve_from_series(p05_series, levels)
                    if np.any(np.isfinite(y_p95)) and np.any(np.isfinite(y_p05)):
                        fig_ldc.add_trace(go.Scatter(
                            x=x_band,
                            y=y_p95,
                            mode="lines",
                            line=dict(color="rgba(31,119,180,0.35)", width=0.5),
                            name="Model p95",
                            showlegend=False,
                            hoverinfo="skip",
                        ))
                        fig_ldc.add_trace(go.Scatter(
                            x=x_band,
                            y=y_p05,
                            mode="lines",
                            line=dict(color="rgba(31,119,180,0.35)", width=0.5),
                            fill="tonexty",
                            fillcolor="rgba(31,119,180,0.2)",
                            name="Model 90% band",
                            hoverinfo="skip",
                        ))
                        band_added = True
                if np.any(np.isfinite(y_median)):
                    fig_ldc.add_trace(go.Scatter(
                        x=x_median,
                        y=y_median,
                        mode="lines",
                        line=dict(color="black", width=2),
                        name="Model median",
                    ))
                observed_values: list[float] = []
                for s in measured_series:
                    if s is None or not isinstance(s, pd.Series) or s.empty:
                        continue
                    ss = s
                    if window is not None:
                        ss = ss.loc[(ss.index >= x0) & (ss.index <= x1)]
                    observed_values.extend(ss.dropna().to_numpy(dtype=float).tolist())
                obs_exceed, obs_vals = _empirical_exceedance(observed_values)
                if obs_exceed.size:
                    fig_ldc.add_trace(go.Scatter(
                        x=obs_exceed,
                        y=obs_vals,
                        mode="markers",
                        name="Observed",
                        marker=dict(color="#d62728", size=8, opacity=0.85, line=dict(width=0.5, color="#333")),
                    ))
                if np.any(np.isfinite(y_median)) or band_added or obs_exceed.size:
                    ldc_title = "Load duration curve"
                    if title:
                        ldc_title = f"{title} - Load duration curve"
                    fig_ldc.update_layout(
                        title=ldc_title,
                        xaxis=dict(title="Exceedance probability (%)", autorange="reversed", range=[0, 100]),
                        yaxis=dict(title="Load"),
                        legend=dict(orientation="h", y=-0.15),
                        margin=dict(l=60, r=20, t=60, b=80),
                    )
                    figs["load_duration_curve"] = fig_ldc
        except Exception:
            pass
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
    compare_mode: str = "load",
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
        compare_mode=compare_mode,
    )

    return {
        "q_df": q_df,
        "measured_series": measured_series,
        "stats": stats,
        "figs": figs,
    }
