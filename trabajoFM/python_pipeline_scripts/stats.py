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

    band_series: Optional[Tuple[Optional[pd.Series], Optional[pd.Series]]] = None,

) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:

    """Return flattened arrays (Y, M) and dict of quantile arrays for coverage.



    - q_df: columns include 'p50', optionally 'p25','p75','p05','p95'.

    - measured: list of series. Each will be sliced by window and reindexed to q_df.

    - band_series: optional tuple of (min_series, max_series) for full-span coverage.

    """

    ys: List[np.ndarray] = []

    ms: List[np.ndarray] = []

    lo25s: List[np.ndarray] = []

    hi75s: List[np.ndarray] = []

    lo05s: List[np.ndarray] = []

    hi95s: List[np.ndarray] = []

    mins: List[np.ndarray] = []

    maxs: List[np.ndarray] = []



    if q_df is None or q_df.empty or not measured:

        return np.array([]), np.array([]), {}



    min_series: Optional[pd.Series] = None

    max_series: Optional[pd.Series] = None

    if isinstance(band_series, tuple) and len(band_series) == 2:

        min_series, max_series = band_series

        if not isinstance(min_series, pd.Series):

            min_series = None

        if not isinstance(max_series, pd.Series):

            max_series = None



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

        if isinstance(min_series, pd.Series) and isinstance(max_series, pd.Series):

            min_aligned = min_series.reindex(ss.index)

            max_aligned = max_series.reindex(ss.index)

            if min_aligned is not None and max_aligned is not None:

                mins.append(min_aligned.loc[mask].to_numpy(dtype=float))

                maxs.append(max_aligned.loc[mask].to_numpy(dtype=float))



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

    if mins and maxs:

        qdict["min"] = np.concatenate(mins)

        qdict["max"] = np.concatenate(maxs)

    return Y, M, qdict








def _pairwise_metrics(
    y: np.ndarray,
    m: np.ndarray,
    qdict: Optional[Dict[str, np.ndarray]] = None,
    *,
    include_minmax: bool = False,
) -> Dict[str, float]:
    """Compute core pairwise metrics for two aligned series."""
    y_arr = np.asarray(y, dtype=float)
    m_arr = np.asarray(m, dtype=float)

    r = _pearson_r(y_arr, m_arr)
    rmse = _rmse(y_arr, m_arr)
    mae = _mae(y_arr, m_arr)
    bias = float(np.nanmean(y_arr - m_arr)) if y_arr.size else float("nan")

    metrics = {
        "r": r,
        "R2": float(r ** 2) if np.isfinite(r) else float("nan"),
        "MAE": mae,
        "RMSE": rmse,
        "Bias(obs-pred)": bias,
        "PBIAS%": _pbias(y_arr, m_arr),
        "NSE": _nse(y_arr, m_arr),
        "NSE_rel": _nse_relative(y_arr, m_arr),
        "KGE": _kge(y_arr, m_arr),
        "MedAE": _medae(y_arr, m_arr),
        "d": _index_of_agreement(y_arr, m_arr),
        "d_rel": _index_of_agreement_relative(y_arr, m_arr),
        "RSR": _rsr(y_arr, m_arr),
    }

    bands = qdict or {}
    if isinstance(bands, dict):
        if "p25" in bands and "p75" in bands:
            metrics["coverage50"] = _coverage(y_arr, bands["p25"], bands["p75"])
        if "p05" in bands and "p95" in bands:
            metrics["coverage90"] = _coverage(y_arr, bands["p05"], bands["p95"])
        if "min" in bands and "max" in bands:
            cov_full = _coverage(y_arr, bands["min"], bands["max"])
            metrics["coverage100"] = cov_full
            if include_minmax:
                metrics["coverage_minmax"] = cov_full

    return metrics



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

    event_context: Optional[Dict[str, object]] = None,

) -> Dict[str, object]:

    """Compute an extensible suite of stats for the dashboard view.



    New parameters:

    - band_data: Optional dict with keys like 'min', 'max', 'mean', 'p25', 'p75', 'p05', 'p95' 

                 containing time series for band deviation analysis

    - event_context: Optional dict capturing event/non-event day indices and current view mode



    Returns a nested dict with sections: n, same_day, log_space, global_lag, local_window_K*, extras, 

                                        band_stats, measured_vs_series.

    """

    # -----------------------------
    # Global shift aggregation switch
    # Set SHIFT_AGG to 'median' or 'mean' to control how (baseline - overlay) shifts are summarized.
    # This affects: overall paired shift, percent shift, event shift, non-event shift.
    # -----------------------------
    global SHIFT_AGG  # allow user to modify externally if desired
    try:
        SHIFT_AGG  # type: ignore  # noqa: F821
    except NameError:
        SHIFT_AGG = 'median'  # fallback default if not already defined elsewhere

    def _shift_agg(arr: np.ndarray) -> float:
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            return float('nan')
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float('nan')
        if SHIFT_AGG == 'mean':
            return float(np.nanmean(arr))
        # default median (for 'median' or fallback when not both)
        return float(np.nanmedian(arr))

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



    ensemble_min = None

    ensemble_max = None

    if isinstance(band_groups, dict):

        ensemble = band_groups.get("ensemble", {})

        if isinstance(ensemble, dict):

            ensemble_min = ensemble.get("min")

            ensemble_max = ensemble.get("max")

    Y, M, qdict = _collect_pairs(q_df, measured_series, window=window, band_series=(ensemble_min, ensemble_max))

    n = int(Y.size)

    stats["n"] = n

    

    # Calculate data usage statistics

    data_usage = _calculate_data_usage(q_df, measured_series, window=window)

    stats["data_usage"] = data_usage

    

    same_day_metrics = _pairwise_metrics(Y, M, qdict) if n else {}
    stats["same_day"] = same_day_metrics



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

                    _add_coverage('min', 'max', 'coverage100')

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



    overlay_comparison: Dict[str, Dict[str, float]] = {}
    overlay_full_stats: Dict[str, Dict[str, float]] = {}
    raw_ensemble = band_groups.get("ensemble_raw", {}) if isinstance(band_groups, dict) else {}
    baseline_series_for_overlay: Optional[pd.Series] = None
    baseline_summary_series: Optional[pd.Series] = None

    def _window_series(series: Optional[pd.Series]) -> Optional[pd.Series]:
        return _apply_window_to_series(series) if isinstance(series, pd.Series) else None

    raw_min_series = _window_series(raw_ensemble.get("min")) if isinstance(raw_ensemble, dict) else None
    raw_max_series = _window_series(raw_ensemble.get("max")) if isinstance(raw_ensemble, dict) else None
    raw_width_series: Optional[pd.Series] = None
    if isinstance(raw_min_series, pd.Series) and isinstance(raw_max_series, pd.Series):
        raw_width_series = (raw_max_series - raw_min_series).abs()
        raw_width_series = raw_width_series.dropna()

    def _width_stats_for_index(idx: Optional[pd.Index]) -> Tuple[float, float]:
        if not isinstance(idx, pd.Index) or idx.empty:
            return float("nan"), float("nan")
        if not isinstance(raw_width_series, pd.Series) or raw_width_series.empty:
            return float("nan"), float("nan")
        width_subset = raw_width_series.reindex(idx)
        if not isinstance(width_subset, pd.Series):
            return float("nan"), float("nan")
        width_values = width_subset.to_numpy(dtype=float)
        width_values = width_values[np.isfinite(width_values)]
        if width_values.size == 0:
            return float("nan"), float("nan")
        return float(np.nanmedian(width_values)), float(np.nanpercentile(width_values, 90))

    def _coverage_fraction_for_series(series: Optional[pd.Series], idx: Optional[pd.Index]) -> float:
        if not isinstance(series, pd.Series) or series.empty:
            return float("nan")
        if not isinstance(idx, pd.Index) or idx.empty:
            return float("nan")
        if not isinstance(raw_min_series, pd.Series) or not isinstance(raw_max_series, pd.Series):
            return float("nan")
        min_subset = raw_min_series.reindex(idx)
        max_subset = raw_max_series.reindex(idx)
        value_subset = series.reindex(idx)
        mask = value_subset.notna() & min_subset.notna() & max_subset.notna()
        if not mask.any():
            return float("nan")
        val = value_subset.loc[mask].to_numpy(dtype=float)
        min_vals = min_subset.loc[mask].to_numpy(dtype=float)
        max_vals = max_subset.loc[mask].to_numpy(dtype=float)
        if val.size == 0:
            return float("nan")
        safe_min = np.minimum(min_vals, max_vals)
        safe_max = np.maximum(min_vals, max_vals)
        coverage_mask = (val >= safe_min) & (val <= safe_max)
        if coverage_mask.size == 0:
            return float("nan")
        return float(np.mean(coverage_mask))

    def _normalize_day_collection(values: Optional[Iterable[object]]) -> Optional[pd.DatetimeIndex]:
        if values is None:
            return None
        if isinstance(values, pd.DatetimeIndex):
            idx = values
        else:
            try:
                idx = pd.DatetimeIndex(pd.to_datetime(list(values)))
            except Exception:
                return None
        if idx.size == 0:
            return None
        idx = idx.floor('D')
        idx = idx.unique()
        try:
            idx = idx.sort_values()
        except Exception:
            idx = idx.sort_values()
        return idx

    baseline_raw_series = _window_series(raw_ensemble.get("p50")) if isinstance(raw_ensemble, dict) else None
    if baseline_raw_series is not None:
        baseline_series_for_overlay = baseline_raw_series
    elif q_df is not None and "p50" in q_df.columns:
        baseline_series_for_overlay = _apply_window_to_series(q_df["p50"].copy())

    baseline_summary: Dict[str, float] = {}
    baseline_median_value = float("nan")
    baseline_median_abs = float("nan")

    baseline_total_days = int(baseline_series_for_overlay.index.size) if isinstance(baseline_series_for_overlay, pd.Series) else 0
    baseline_finite_days = int(baseline_series_for_overlay.dropna().size) if isinstance(baseline_series_for_overlay, pd.Series) else 0

    baseline_summary_series = baseline_series_for_overlay.dropna() if isinstance(baseline_series_for_overlay, pd.Series) else None
    if isinstance(baseline_summary_series, pd.Series) and not baseline_summary_series.empty:
        baseline_median_value = float(np.nanmedian(baseline_summary_series.values))
        baseline_median_abs = float(np.nanmedian(np.abs(baseline_summary_series.values)))
        baseline_summary = {
            "median": baseline_median_value,
            "mean": float(np.nanmean(baseline_summary_series.values)),
            "sd": float(np.nanstd(baseline_summary_series.values)),
        }

    raw_baseline_total = int(baseline_raw_series.index.size) if isinstance(baseline_raw_series, pd.Series) else 0
    raw_baseline_finite = int(baseline_raw_series.dropna().size) if isinstance(baseline_raw_series, pd.Series) else 0

    baseline_quantiles: Dict[str, float] = {}
    if isinstance(baseline_series_for_overlay, pd.Series):
        baseline_nonnull_for_quantiles = baseline_series_for_overlay.dropna()
        if not baseline_nonnull_for_quantiles.empty:
            baseline_values_for_quantiles = baseline_nonnull_for_quantiles.to_numpy(dtype=float)
            baseline_quantiles = {
                "q10": float(np.nanpercentile(baseline_values_for_quantiles, 10)),
                "q90": float(np.nanpercentile(baseline_values_for_quantiles, 90)),
            }

    baseline_median_width = float("nan")
    baseline_p90_width = float("nan")
    baseline_mean_width = float("nan")
    baseline_coverage_fraction = float("nan")
    if isinstance(baseline_series_for_overlay, pd.Series):
        baseline_width_index = baseline_series_for_overlay.dropna().index
        if isinstance(baseline_width_index, pd.Index) and baseline_width_index.size:
            baseline_median_width, baseline_p90_width = _width_stats_for_index(baseline_width_index)
            # Compute mean raw width directly (not just median/p90) for user-requested display
            if isinstance(raw_width_series, pd.Series) and not raw_width_series.empty:
                try:
                    baseline_mean_width = float(np.nanmean(raw_width_series.reindex(baseline_width_index).to_numpy(dtype=float)))
                except Exception:
                    baseline_mean_width = float("nan")
        baseline_coverage_fraction = _coverage_fraction_for_series(baseline_series_for_overlay, baseline_series_for_overlay.index)

    event_mode_label = "all"
    event_idx_raw: Optional[pd.DatetimeIndex] = None
    event_idx_buffered: Optional[pd.DatetimeIndex] = None
    event_idx_non: Optional[pd.DatetimeIndex] = None
    event_idx_selected: Optional[pd.DatetimeIndex] = None
    event_idx_all: Optional[pd.DatetimeIndex] = None
    if isinstance(event_context, dict):
        mode_val = event_context.get("mode") or event_context.get("view")
        if isinstance(mode_val, str) and mode_val.strip():
            event_mode_label = mode_val.strip()
        event_idx_raw = _normalize_day_collection(event_context.get("events"))
        event_idx_buffered = _normalize_day_collection(event_context.get("buffered_events"))
        if event_idx_buffered is None:
            event_idx_buffered = event_idx_raw
        event_idx_non = _normalize_day_collection(event_context.get("non_events"))
        event_idx_selected = _normalize_day_collection(event_context.get("selected"))
        event_idx_all = _normalize_day_collection(event_context.get("all_days"))
        if event_idx_all is not None and event_idx_buffered is not None and event_idx_non is None:
            tmp_non = event_idx_all.difference(event_idx_buffered)
            if tmp_non.size:
                event_idx_non = tmp_non
        if event_idx_non is not None and event_idx_non.size == 0:
            event_idx_non = None
    event_stats_summary = {
        "view": event_mode_label,
        "event_days": int(event_idx_raw.size) if event_idx_raw is not None else 0,
        "buffered_days": int(event_idx_buffered.size) if event_idx_buffered is not None else 0,
        "non_event_days": int(event_idx_non.size) if event_idx_non is not None else 0,
        "selected_days": int(event_idx_selected.size) if event_idx_selected is not None else 0,
        "all_days": int(event_idx_all.size) if event_idx_all is not None else 0,
    }

    baseline_relative_width = float("nan")  # median-based relative width (min-max/median)
    baseline_mean_relative_width = float("nan")  # mean-based relative width (min-max/mean)
    p05_series_full = _window_series(raw_ensemble.get("p05")) if isinstance(raw_ensemble, dict) else None
    p95_series_full = _window_series(raw_ensemble.get("p95")) if isinstance(raw_ensemble, dict) else None
    if p05_series_full is None and q_df is not None and "p05" in q_df.columns:
        p05_series_full = _apply_window_to_series(q_df["p05"].copy())
    if p95_series_full is None and q_df is not None and "p95" in q_df.columns:
        p95_series_full = _apply_window_to_series(q_df["p95"].copy())
    if (
        isinstance(baseline_summary_series, pd.Series)
        and not baseline_summary_series.empty
        and isinstance(p05_series_full, pd.Series)
        and isinstance(p95_series_full, pd.Series)
    ):
        idx = baseline_summary_series.index.intersection(p05_series_full.index).intersection(p95_series_full.index)
        if len(idx) > 0:
            base_subset = baseline_summary_series.reindex(idx).to_numpy(dtype=float)
            width_vals = (
                p95_series_full.reindex(idx).to_numpy(dtype=float)
                - p05_series_full.reindex(idx).to_numpy(dtype=float)
            )
            # Median-based relative width
            denom_median = np.where(np.abs(base_subset) > 0, np.abs(base_subset), np.nan)
            rel_vals_median = width_vals / denom_median
            rel_vals_median = rel_vals_median[np.isfinite(rel_vals_median)]
            if rel_vals_median.size:
                baseline_relative_width = float(np.nanmedian(rel_vals_median))
            # Mean-based relative width (uses mean of base_subset for each day equivalently -> width/base)
            denom_mean = np.where(np.abs(base_subset) > 0, np.abs(base_subset), np.nan)
            rel_vals_mean = width_vals / denom_mean
            rel_vals_mean = rel_vals_mean[np.isfinite(rel_vals_mean)]
            if rel_vals_mean.size:
                baseline_mean_relative_width = float(np.nanmean(rel_vals_mean))

    allowed_overlay_metrics = {"r", "R2", "MAE", "RMSE", "Bias(obs-pred)", "MedAE"}

    if extras and isinstance(extras, dict) and isinstance(baseline_series_for_overlay, pd.Series):
        for name, series in extras.items():
            if not isinstance(series, pd.Series) or series.empty:
                continue
            overlay_series = series
            if window is not None:
                x0, x1 = window
                overlay_series = overlay_series.loc[(overlay_series.index >= x0) & (overlay_series.index <= x1)]
            overlay_total_days = int(overlay_series.index.size)
            overlay_series = overlay_series.dropna()
            overlay_finite_days = int(overlay_series.index.size)
            if overlay_series.empty:
                continue
            aligned_base = baseline_series_for_overlay.reindex(overlay_series.index)
            mask = aligned_base.notna() & overlay_series.notna()
            if not mask.any():
                continue
            paired_base = aligned_base.loc[mask]
            paired_overlay = overlay_series.loc[mask]
            if paired_base.empty or paired_overlay.empty:
                continue
            diff_series = paired_base - paired_overlay
            base_vals = paired_base.to_numpy(dtype=float)
            overlay_vals = paired_overlay.to_numpy(dtype=float)
            diff = base_vals - overlay_vals
            # SHIFT AGGREGATION POINT: Change np.nanmedian -> np.nanmean below to switch to mean shifts
            # Primary shift (baseline - overlay) summary. Supports SHIFT_AGG in {'median','mean','both'}
            median_delta = float('nan')
            mean_delta = float('nan')
            if diff.size:
                if SHIFT_AGG == 'both':
                    with np.errstate(all='ignore'):
                        median_delta = float(np.nanmedian(diff))
                        mean_delta = float(np.nanmean(diff))
                else:
                    median_delta = _shift_agg(diff)
            denom_series = paired_base.abs().replace(0, np.nan)
            rel_series = (diff_series.abs() / denom_series).replace([np.inf, -np.inf], np.nan)
            rel_vals = rel_series.dropna().to_numpy(dtype=float)
            relative_width = float(np.nanmedian(rel_vals)) if rel_vals.size else float("nan")
            percent_series = (diff_series / denom_series) * 100.0
            percent_series = percent_series.replace([np.inf, -np.inf], np.nan)
            percent_vals = percent_series.dropna().to_numpy(dtype=float)
            # SHIFT AGGREGATION POINT (%): Change np.nanmedian -> np.nanmean to use mean percent shift
            if percent_vals.size:
                if SHIFT_AGG == 'both':
                    with np.errstate(all='ignore'):
                        median_delta_pct = float(np.nanmedian(percent_vals))
                        mean_delta_pct = float(np.nanmean(percent_vals))
                else:
                    median_delta_pct = _shift_agg(percent_vals)
            else:
                median_delta_pct = float('nan')
                mean_delta_pct = float('nan') if SHIFT_AGG == 'both' else None
            rmse_overlay = _rmse(base_vals, overlay_vals)
            r_overlay = _pearson_r(base_vals, overlay_vals)
            paired_index = paired_base.index
            median_w, p90_w = _width_stats_for_index(paired_index)
            coverage_fraction = _coverage_fraction_for_series(paired_overlay, paired_index)
            relative_width_low = float("nan")
            relative_width_high = float("nan")
            relative_width_event_contrast = float("nan")
            lower_thresh = baseline_quantiles.get("q10") if baseline_quantiles else None
            upper_thresh = baseline_quantiles.get("q90") if baseline_quantiles else None
            if lower_thresh is not None and upper_thresh is not None and not np.isnan(lower_thresh) and not np.isnan(upper_thresh):
                low_rel = rel_series.loc[paired_base <= lower_thresh].dropna()
                high_rel = rel_series.loc[paired_base >= upper_thresh].dropna()
                if not low_rel.empty:
                    low_vals = low_rel.to_numpy(dtype=float)
                    low_vals = low_vals[np.isfinite(low_vals)]
                    if low_vals.size:
                        relative_width_low = float(np.nanmedian(low_vals))
                if not high_rel.empty:
                    high_vals = high_rel.to_numpy(dtype=float)
                    high_vals = high_vals[np.isfinite(high_vals)]
                    if high_vals.size:
                        relative_width_high = float(np.nanmedian(high_vals))
                if np.isfinite(relative_width_low) and np.isfinite(relative_width_high) and relative_width_low != 0:
                    relative_width_event_contrast = float(relative_width_high / relative_width_low)

            relative_width_event = float("nan")
            relative_width_nonevent = float("nan")
            event_ratio = float("nan")
            event_pairs_count = 0
            nonevent_pairs_count = 0
            paired_days_floor = None
            if isinstance(paired_index, pd.DatetimeIndex):
                paired_days_floor = paired_index.floor('D')
            else:
                try:
                    paired_days_floor = pd.DatetimeIndex(paired_index).floor('D')
                except Exception:
                    paired_days_floor = None
            event_mask = None
            nonevent_mask = None
            if paired_days_floor is not None and event_idx_buffered is not None:
                event_mask = paired_days_floor.isin(event_idx_buffered)
                if event_idx_non is not None:
                    nonevent_mask = paired_days_floor.isin(event_idx_non)
                else:
                    nonevent_mask = ~event_mask
            elif paired_days_floor is not None and event_idx_non is not None:
                nonevent_mask = paired_days_floor.isin(event_idx_non)
            if event_mask is not None and np.any(event_mask):
                rel_event_series = rel_series.loc[event_mask].dropna()
                if not rel_event_series.empty:
                    event_pairs_count = int(rel_event_series.size)
                    event_vals = rel_event_series.to_numpy(dtype=float)
                    event_vals = event_vals[np.isfinite(event_vals)]
                    if event_vals.size:
                        relative_width_event = float(np.nanmedian(event_vals))
            if nonevent_mask is not None and np.any(nonevent_mask):
                rel_nonevent_series = rel_series.loc[nonevent_mask].dropna()
                if not rel_nonevent_series.empty:
                    nonevent_pairs_count = int(rel_nonevent_series.size)
                    nonevent_vals = rel_nonevent_series.to_numpy(dtype=float)
                    nonevent_vals = nonevent_vals[np.isfinite(nonevent_vals)]
                    if nonevent_vals.size:
                        relative_width_nonevent = float(np.nanmedian(nonevent_vals))
            if np.isfinite(relative_width_event) and np.isfinite(relative_width_nonevent) and relative_width_nonevent != 0:
                event_ratio = float(relative_width_event / relative_width_nonevent)

            # Compute event / non-event specific shift (baseline - overlay) using same masks
            # Controlled by global SHIFT_AGG (median/mean)
            delta_event_pairs = float("nan")
            delta_non_event_pairs = float("nan")
            delta_event_pairs_mean = float("nan")
            delta_non_event_pairs_mean = float("nan")
            try:
                if event_mask is not None and np.any(event_mask):
                    ev_base = paired_base.loc[event_mask].to_numpy(dtype=float)
                    ev_overlay = paired_overlay.loc[event_mask].to_numpy(dtype=float)
                    if ev_base.size and ev_overlay.size:
                        ev_diff = ev_base - ev_overlay
                        if ev_diff.size:
                            if SHIFT_AGG == 'both':
                                with np.errstate(all='ignore'):
                                    delta_event_pairs = float(np.nanmedian(ev_diff))
                                    delta_event_pairs_mean = float(np.nanmean(ev_diff))
                            else:
                                delta_event_pairs = _shift_agg(ev_diff)
            except Exception:
                pass
            try:
                if nonevent_mask is not None and np.any(nonevent_mask):
                    nev_base = paired_base.loc[nonevent_mask].to_numpy(dtype=float)
                    nev_overlay = paired_overlay.loc[nonevent_mask].to_numpy(dtype=float)
                    if nev_base.size and nev_overlay.size:
                        nev_diff = nev_base - nev_overlay
                        if nev_diff.size:
                            if SHIFT_AGG == 'both':
                                with np.errstate(all='ignore'):
                                    delta_non_event_pairs = float(np.nanmedian(nev_diff))
                                    delta_non_event_pairs_mean = float(np.nanmean(nev_diff))
                            else:
                                delta_non_event_pairs = _shift_agg(nev_diff)
            except Exception:
                pass

            def _safe_stat(arr, func):
                if arr.size == 0:
                    return float("nan")
                val = func(arr)
                return float(val) if np.isfinite(val) else float("nan")

            base_min = _safe_stat(base_vals, np.nanmin)
            base_max = _safe_stat(base_vals, np.nanmax)
            base_mean = _safe_stat(base_vals, np.nanmean)
            base_p50 = _safe_stat(base_vals, lambda x: np.nanpercentile(x, 50))
            base_p90 = _safe_stat(base_vals, lambda x: np.nanpercentile(x, 90))

            overlay_min = _safe_stat(overlay_vals, np.nanmin)
            overlay_max = _safe_stat(overlay_vals, np.nanmax)
            overlay_mean = _safe_stat(overlay_vals, np.nanmean)
            overlay_p50 = _safe_stat(overlay_vals, lambda x: np.nanpercentile(x, 50))
            overlay_p90 = _safe_stat(overlay_vals, lambda x: np.nanpercentile(x, 90))

            delta_min = base_min - overlay_min if np.isfinite(base_min) and np.isfinite(overlay_min) else float("nan")
            delta_max = base_max - overlay_max if np.isfinite(base_max) and np.isfinite(overlay_max) else float("nan")
            delta_mean = base_mean - overlay_mean if np.isfinite(base_mean) and np.isfinite(overlay_mean) else float("nan")
            delta_p50 = base_p50 - overlay_p50 if np.isfinite(base_p50) and np.isfinite(overlay_p50) else float("nan")
            delta_p90 = base_p90 - overlay_p90 if np.isfinite(base_p90) and np.isfinite(overlay_p90) else float("nan")

            # Normalized shift metrics (using baseline central tendency).
            normalized_median_delta = float('nan')
            normalized_mean_delta = float('nan')
            if np.isfinite(base_p50) and base_p50 != 0 and np.isfinite(median_delta):
                normalized_median_delta = float(median_delta / base_p50)
            if SHIFT_AGG in ('mean','both') and np.isfinite(base_mean) and base_mean != 0:
                # if using mean-only mode, median_delta holds mean but we compute explicit mean_delta where both
                ref_delta = mean_delta if SHIFT_AGG == 'both' else median_delta
                if np.isfinite(ref_delta):
                    normalized_mean_delta = float(ref_delta / base_mean)

            entry = {
                "median_delta": median_delta,
                "median_delta_pct": median_delta_pct,
                "relative_width": relative_width,
                "rmse": rmse_overlay,
                "r": r_overlay,
                "median_W": median_w,
                "p90_W": p90_w,
                "coverage_fraction": coverage_fraction,
                "relative_width_low": relative_width_low,
                "relative_width_high": relative_width_high,
                "relative_width_event_contrast": relative_width_event_contrast,
                "relative_width_event": relative_width_event,
                "relative_width_nonevent": relative_width_nonevent,
                "event_ratio": event_ratio,
                "event_pairs": int(event_pairs_count),
                "non_event_pairs": int(nonevent_pairs_count),
                "baseline_min": base_min,
                "baseline_max": base_max,
                "baseline_mean": base_mean,
                "baseline_p50": base_p50,
                "baseline_p90": base_p90,
                "overlay_min": overlay_min,
                "overlay_max": overlay_max,
                "overlay_mean": overlay_mean,
                "overlay_p50": overlay_p50,
                "overlay_p90": overlay_p90,
                "delta_min": delta_min,
                "delta_max": delta_max,
                "delta_mean": delta_mean,
                "delta_p50": delta_p50,
                "delta_p90": delta_p90,
                "delta_event_pairs": delta_event_pairs,
                "delta_non_event_pairs": delta_non_event_pairs,
                "normalized_median_delta": normalized_median_delta,
            }
            if SHIFT_AGG == 'both':
                entry.update({
                    "mean_delta": mean_delta,
                    "mean_delta_pct": mean_delta_pct,
                    "delta_event_pairs_mean": delta_event_pairs_mean,
                    "delta_non_event_pairs_mean": delta_non_event_pairs_mean,
                    "normalized_mean_delta": normalized_mean_delta,
                })
            elif SHIFT_AGG == 'mean':
                # For clarity when using mean-only mode, provide alias keys.
                entry.update({
                    "mean_delta": median_delta,  # median_delta holds mean shift value
                    "mean_delta_pct": median_delta_pct,
                    "normalized_mean_delta": normalized_mean_delta,
                })
            overlay_comparison[str(name)] = entry

            overlay_metrics = _pairwise_metrics(overlay_vals, base_vals)
            filtered_metrics = {k: overlay_metrics[k] for k in allowed_overlay_metrics if k in overlay_metrics and np.isfinite(overlay_metrics[k])}
            filtered_metrics.update({
                "n_pairs": int(mask.sum()),
            })
            overlay_full_stats[str(name)] = filtered_metrics

    stats["event_context"] = event_stats_summary

    baseline_entry = overlay_comparison.setdefault("__baseline__", {})
    if np.isfinite(baseline_relative_width):
        baseline_entry["relative_width"] = baseline_relative_width
    # Mean-based relative width (if computed)
    if 'baseline_mean_relative_width' not in locals():
        baseline_mean_relative_width = float('nan')  # safety
    if np.isfinite(baseline_mean_relative_width):
        baseline_entry["relative_width_mean"] = baseline_mean_relative_width
    # Normalized widths (dimensionless) if possible
    try:
        if np.isfinite(baseline_relative_width) and baseline_summary.get('median') not in (None, 0):
            median_val = baseline_summary.get('median')
            if isinstance(median_val, (int, float)) and np.isfinite(median_val) and median_val != 0:
                baseline_entry["relative_width_norm_median"] = float(baseline_relative_width)
        if np.isfinite(baseline_mean_relative_width) and baseline_summary.get('mean') not in (None, 0):
            mean_val = baseline_summary.get('mean')
            if isinstance(mean_val, (int, float)) and np.isfinite(mean_val) and mean_val != 0:
                baseline_entry["relative_width_norm_mean"] = float(baseline_mean_relative_width)
    except Exception:
        pass
    if np.isfinite(baseline_median_width):
        baseline_entry["median_W"] = baseline_median_width
    if np.isfinite(baseline_mean_width):
        baseline_entry["mean_W"] = baseline_mean_width
    if np.isfinite(baseline_p90_width):
        baseline_entry["p90_W"] = baseline_p90_width
    if np.isfinite(baseline_coverage_fraction):
        baseline_entry["coverage_fraction"] = baseline_coverage_fraction
    if baseline_summary:
        baseline_entry.setdefault("median", baseline_summary.get("median"))
        baseline_entry.setdefault("mean", baseline_summary.get("mean"))
        baseline_entry.setdefault("sd", baseline_summary.get("sd"))
        stats["baseline_summary"] = baseline_summary
    stats["overlay_comparison"] = overlay_comparison
    if overlay_full_stats:
        stats["overlay_full_series"] = overlay_full_stats

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

    def _format_metric(key: str, value: object) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            if isinstance(value, float) and not np.isfinite(value):
                return None
            low_key = key.lower()
            if low_key in {"r", "r_log", "r (meas vs median)", "nse", "kge", "r2", "r2_log"}:
                return f"{key} = {float(value):.3f}"
            if "coverage" in low_key:
                return f"{key} = {float(value) * 100:.1f}%"
            if isinstance(value, int) and not isinstance(value, bool):
                return f"{key} = {int(value)}"
            return f"{key} = {float(value):.3g}"
        if isinstance(value, str):
            return f"{key} = {value}"
        return f"{key} = {value}"

    # Add data usage information
    data_usage = stats.get("data_usage", {}) or {}
    if data_usage:
        paired_pct = data_usage.get("paired_usage_percent", 0.0)
        sim_pct = data_usage.get("sim_usage_percent", 0.0)
        meas_pct = data_usage.get("measured_usage_percent", 0.0)
        lines.append(f"data usage = {paired_pct:.1f}% of available days")
        if data_usage.get("sim_days_total", 0) > 0:
            lines.append(
                f"sim coverage = {sim_pct:.1f}% ("
                f"{data_usage.get('sim_days_windowed_finite', 0)}/"
                f"{data_usage.get('sim_days_total', 0)} days)"
            )
        if data_usage.get("measured_days_total", 0) > 0:
            lines.append(
                f"measured coverage = {meas_pct:.1f}% ("
                f"{data_usage.get('measured_days_windowed_finite', 0)}/"
                f"{data_usage.get('measured_days_total', 0)} days)"
            )

    same = stats.get("same_day", {}) or {}
    if same:
        lines.append("<b>Same-day</b>")
        order = [
            "r",
            "R2",
            "MAE",
            "RMSE",
            "Bias(obs-pred)",
            "PBIAS%",
            "NSE",
            "KGE",
            "MedAE",
            "coverage50",
            "coverage90",
            "coverage100",
        ]
        for key in order:
            if key in same:
                formatted = _format_metric(key, same[key])
                if formatted:
                    lines.append(formatted)
        for key, val in same.items():
            if key in order:
                continue
            formatted = _format_metric(str(key), val)
            if formatted:
                lines.append(formatted)

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
        for label, summary in [
            ("Observed (paired)", paired.get("observed")),
            ("Predicted (paired)", paired.get("predicted")),
        ]:
            if isinstance(summary, dict):
                lines.append(f"{label}:")
                for key in ["mean", "median", "p05", "p25", "p50", "p75", "p95", "sd", "var", "n"]:
                    val = summary.get(key)
                    if isinstance(val, (int, float)) and np.isfinite(val):
                        if key == "n":
                            lines.append(f"  {key} = {int(val)}")
                        else:
                            lines.append(f"  {key} = {val:.3g}")
        pred_full = dist_summary.get("predicted_full")
        if isinstance(pred_full, dict):
            lines.append("Predicted (full series):")
            for key in ["mean", "median", "p05", "p25", "p50", "p75", "p95", "sd", "var", "n"]:
                val = pred_full.get(key)
                if isinstance(val, (int, float)) and np.isfinite(val):
                    if key == "n":
                        lines.append(f"  {key} = {int(val)}")
                    else:
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

    for key in sorted([k for k in stats.keys() if str(k).startswith("local_window_K")]):
        lw = stats.get(key) or {}
        if not lw:
            continue
        lines.append(f"<b>Local window K={int(lw.get('K', 0))}</b>")
        for kk in [
            "RMSE",
            "MAE",
            "Bias(obs-pred)",
            "NSE",
            "median_lag_days",
            "IQR_lag_days",
            "fraction_zero_lag",
            "n",
        ]:
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

    overlay_cmp = stats.get("overlay_comparison", {}) or {}
    if overlay_cmp:
        lines.append("<b>Overlay Comparison</b>")
        cmp_dict = dict(overlay_cmp)
        baseline_entry = cmp_dict.pop("__baseline__", None)
        if isinstance(baseline_entry, dict):
            rel = baseline_entry.get("relative_width")
            if isinstance(rel, (int, float)) and np.isfinite(rel):
                lines.append(f"Model relative width (min-max / median) = {rel:.3g}")
            rel_mean = baseline_entry.get("relative_width_mean")
            if isinstance(rel_mean, (int, float)) and np.isfinite(rel_mean):
                lines.append(f"Model relative width (min-max / mean) = {rel_mean:.3g}")
            rel_norm_med = baseline_entry.get("relative_width_norm_median")
            if isinstance(rel_norm_med, (int, float)) and np.isfinite(rel_norm_med):
                lines.append(f"Model normalized relative width (using median) = {rel_norm_med:.3g}")
            rel_norm_mean = baseline_entry.get("relative_width_norm_mean")
            if isinstance(rel_norm_mean, (int, float)) and np.isfinite(rel_norm_mean):
                lines.append(f"Model normalized relative width (using mean) = {rel_norm_mean:.3g}")
            median_w = baseline_entry.get("median_W")
            if isinstance(median_w, (int, float)) and np.isfinite(median_w):
                lines.append(f"Model raw width median (W) = {median_w:.3g}")
            mean_w = baseline_entry.get("mean_W")
            if isinstance(mean_w, (int, float)) and np.isfinite(mean_w):
                lines.append(f"Model raw width mean (W) = {mean_w:.3g}")
            p90_w = baseline_entry.get("p90_W")
            if isinstance(p90_w, (int, float)) and np.isfinite(p90_w):
                lines.append(f"Model raw width p90 (W) = {p90_w:.3g}")
            cov_frac = baseline_entry.get("coverage_fraction")
            if isinstance(cov_frac, (int, float)) and np.isfinite(cov_frac):
                lines.append(f"Model coverage fraction (baseline inside envelope) = {cov_frac:.1%}")
            for label_key in ("median", "mean", "sd"):
                val = baseline_entry.get(label_key)
                if isinstance(val, (int, float)) and np.isfinite(val):
                    lines.append(f"Model {label_key} = {val:.3g}")
        event_ctx = stats.get("event_context", {}) or {}
        raw_view_label = str(event_ctx.get("view") or "all").strip() or "all"
        view_key = raw_view_label.lower()
        summary_bits = []
        for key, label in (("selected_days", "selected"), ("buffered_days", "buffered"), ("event_days", "events"), ("non_event_days", "non-events")):
            val = event_ctx.get(key)
            if isinstance(val, (int, float)) and val > 0:
                summary_bits.append(f"{label}={int(val)}")
        total_days_val = event_ctx.get("all_days")
        if isinstance(total_days_val, (int, float)) and total_days_val > 0:
            summary_bits.append(f"total={int(total_days_val)}")
        if summary_bits or view_key != "all":
            summary_txt = ", ".join(summary_bits) if summary_bits else "no event counts available"
            lines.append(f"Event filter view: {raw_view_label} ({summary_txt})")

    general_metrics = [
        ("median_W", "Raw width median (W)", False),
        ("p90_W", "Raw width p90 (W)", False),
        ("coverage_fraction", "Coverage fraction (overlay inside envelope)", True),
    ]
    quantile_metrics = [
        ("relative_width_low", "Relative width low decile"),
        ("relative_width_high", "Relative width high decile"),
        ("relative_width_event_contrast", "Event contrast (high / low)"),
    ]
    event_metric_map = {
        "events": [("relative_width_event", "Relative width (event days)")],
        "non_events": [("relative_width_nonevent", "Relative width (non-event days)")],
        "all": [
            ("relative_width_event", "Relative width (event days)"),
            ("relative_width_nonevent", "Relative width (non-event days)"),
            ("event_ratio", "Event ratio (event / non-event)"),
        ],
    }
    baseline_stat_metrics = [
        ("baseline_min", "Baseline min"),
        ("baseline_max", "Baseline max"),
        ("baseline_mean", "Baseline mean"),
        ("baseline_p50", "Baseline p50"),
        ("baseline_p90", "Baseline p90"),
    ]
    overlay_stat_metrics = [
        ("overlay_min", "Overlay min"),
        ("overlay_max", "Overlay max"),
        ("overlay_mean", "Overlay mean"),
        ("overlay_p50", "Overlay p50"),
        ("overlay_p90", "Overlay p90"),
    ]
    delta_stat_metrics = [
        ("delta_min", "Shift in min (baseline - overlay)"),
        ("delta_max", "Shift in max (baseline - overlay)"),
        ("delta_mean", "Shift in mean (baseline - overlay)"),
        ("delta_p50", "Shift in p50 (baseline - overlay)"),
        ("delta_p90", "Shift in p90 (baseline - overlay)"),
        ("delta_event_pairs", "Shift event pairs (baseline - overlay)"),
        ("delta_non_event_pairs", "Shift non-event pairs (baseline - overlay)"),
    ]
    event_metrics = event_metric_map.get(view_key, event_metric_map["all"])
    for name, comp in cmp_dict.items():
        if not isinstance(comp, dict):
            continue
        lines.append("")  # Blank line between entries
        lines.append(f"<i>{name}:</i>")
        printed_keys = {"median_delta", "median_delta_pct", "relative_width", "rmse", "r"}
        # Determine aggregation mode availability
        has_mean = "mean_delta" in comp and np.isfinite(comp.get("mean_delta", float("nan")))
        has_both = has_mean and ("median_delta" in comp) and np.isfinite(comp.get("median_delta", float("nan"))) and comp.get("mean_delta") != comp.get("median_delta")
        median_delta_val = comp.get("median_delta")
        mean_delta_val = comp.get("mean_delta") if has_mean else None
        if has_both:
            if isinstance(median_delta_val, (int, float)) and np.isfinite(median_delta_val):
                lines.append(f"  Median shift (baseline - overlay) = {median_delta_val:.3g}")
            if isinstance(mean_delta_val, (int, float)) and np.isfinite(mean_delta_val):
                lines.append(f"  Mean shift (baseline - overlay) = {mean_delta_val:.3g}")
        else:
            # Single mode (median or mean); label generically as 'Shift'
            single_label = "Shift" if has_mean and not ("median_delta" in comp) else ("Median shift" if not has_mean else "Mean shift")
            val = mean_delta_val if has_mean and not has_both else median_delta_val
            if isinstance(val, (int, float)) and np.isfinite(val):
                lines.append(f"  {single_label} (baseline - overlay) = {val:.3g}")
        # Percent shift
        if has_both:
            mdp = comp.get("median_delta_pct")
            mnp = comp.get("mean_delta_pct")
            if isinstance(mdp, (int, float)) and np.isfinite(mdp):
                lines.append(f"  Median shift (%) = {mdp:.2f}%")
            if isinstance(mnp, (int, float)) and np.isfinite(mnp):
                lines.append(f"  Mean shift (%) = {mnp:.2f}%")
        else:
            mdp = comp.get("median_delta_pct") if not has_mean or not has_both else comp.get("mean_delta_pct")
            if isinstance(mdp, (int, float)) and np.isfinite(mdp):
                lines.append(f"  Shift (%) = {mdp:.2f}%")
        # Normalized shifts
        nmed = comp.get("normalized_median_delta")
        if isinstance(nmed, (int, float)) and np.isfinite(nmed):
            lines.append(f"  Normalized median shift (delta / baseline p50) = {nmed:.3g}")
        nmean = comp.get("normalized_mean_delta")
        if isinstance(nmean, (int, float)) and np.isfinite(nmean):
            lines.append(f"  Normalized mean shift (delta / baseline mean) = {nmean:.3g}")
        rel = comp.get("relative_width")
        #if isinstance(rel, (int, float)) and np.isfinite(rel):
         #   lines.append(f"  Relative width = {rel:.3g}")
        rmse_val = comp.get("rmse")
        if isinstance(rmse_val, (int, float)) and np.isfinite(rmse_val):
            lines.append(f"  RMSE = {rmse_val:.3g}")
        r_val = comp.get("r")
        if isinstance(r_val, (int, float)) and np.isfinite(r_val):
            lines.append(f"  r = {r_val:.3f}")
        for key, label, is_percent in general_metrics:
            value = comp.get(key)
            if isinstance(value, (int, float)) and np.isfinite(value):
                if is_percent:
                    lines.append(f"  {label} = {value:.1%}")
                else:
                    lines.append(f"  {label} = {value:.3g}")
                printed_keys.add(key)
        for key, label in quantile_metrics:
            value = comp.get(key)
            if isinstance(value, (int, float)) and np.isfinite(value):
                lines.append(f"  {label} = {value:.3g}")
                printed_keys.add(key)
        for key, label in event_metrics:
            value = comp.get(key)
            if isinstance(value, (int, float)) and np.isfinite(value):
                lines.append(f"  {label} = {value:.3g}")
                printed_keys.add(key)
        for key, label in baseline_stat_metrics:
            value = comp.get(key)
            if isinstance(value, (int, float)) and np.isfinite(value):
                lines.append(f"  {label} = {value:.3g}")
                printed_keys.add(key)
        for key, label in overlay_stat_metrics:
            value = comp.get(key)
            if isinstance(value, (int, float)) and np.isfinite(value):
                lines.append(f"  {label} = {value:.3g}")
                printed_keys.add(key)
        for key, label in delta_stat_metrics:
            value = comp.get(key)
            if isinstance(value, (int, float)) and np.isfinite(value):
                # Avoid re-printing median/mean shifts already shown
                if key in {"delta_mean", "delta_p50"} and has_both:
                    pass
                else:
                    lines.append(f"  {label} = {value:.3g}")
                printed_keys.add(key)
        # If both modes, also expose event/non-event mean variant shifts if present
        if has_both:
            evm = comp.get("delta_event_pairs_mean")
            if isinstance(evm, (int, float)) and np.isfinite(evm):
                lines.append(f"  Shift event pairs (mean) = {evm:.3g}")
            nvm = comp.get("delta_non_event_pairs_mean")
            if isinstance(nvm, (int, float)) and np.isfinite(nvm):
                lines.append(f"  Shift non-event pairs (mean) = {nvm:.3g}")

        event_pairs = comp.get("event_pairs")
        if isinstance(event_pairs, (int, float)) and event_pairs > 0:
            lines.append(f"  Event pairs = {int(event_pairs)}")
            printed_keys.add("event_pairs")

        nonevent_pairs = comp.get("non_event_pairs")
        if isinstance(nonevent_pairs, (int, float)) and nonevent_pairs > 0:
            lines.append(f"  Non-event pairs = {int(nonevent_pairs)}")
            printed_keys.add("non_event_pairs")
        for extra_key, extra_val in comp.items():
            if extra_key in printed_keys:
                continue
            if isinstance(extra_val, (int, float)) and np.isfinite(extra_val):
                label = str(extra_key).replace("_", " ")
                lines.append(f"  {label} = {extra_val:.3g}")
    overlay_full = stats.get("overlay_full_series", {}) or {}
    if overlay_full:
        lines.append("<b>Overlay Full-Series</b>")
        metric_order = [
            "r",
            "R2",
            "MAE",
            "RMSE",
            "Bias(obs-pred)",
            "MedAE",
        ]
        for name, metrics in overlay_full.items():
            if not isinstance(metrics, dict):
                continue
            lines.append("")  # Blank line between entries
            lines.append(f"<i>{name}:</i>")
            n_pairs = metrics.get("n_pairs")
            if isinstance(n_pairs, (int, float)) and np.isfinite(n_pairs):
                lines.append(f"  n = {int(n_pairs)}")
            for key in metric_order:
                if key in metrics:
                    formatted = _format_metric(key, metrics[key])
                    if formatted:
                        lines.append(f"  {formatted}")
            for key, val in metrics.items():
                if key in metric_order or key == "n_pairs":
                    continue
                formatted = _format_metric(str(key), val)
                if formatted:
                    lines.append(f"  {formatted}")

    coverage_debug = stats.get("coverage_debug", {}) or {}
    if coverage_debug:
        lines.append("<b>Coverage Debug</b>")
        for label, messages in coverage_debug.items():
            lines.append(f"{label}:")
            for msg in messages:
                lines.append(f"  {msg}")

    return "<br>".join(lines)

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



    if compare_mode == "load":
        try:
            levels = np.linspace(1.0, 99.0, 99)
            x0 = x1 = None
            if window is not None:
                x0, x1 = window
            median_series_load: Optional[pd.Series] = None
            if q_df is not None and "p50" in q_df.columns:
                median_series_load = q_df["p50"].copy()
                if window is not None:
                    median_series_load = median_series_load.loc[(median_series_load.index >= x0) & (median_series_load.index <= x1)]
            if isinstance(median_series_load, pd.Series):
                median_series_load = median_series_load.dropna()
            if isinstance(median_series_load, pd.Series) and not median_series_load.empty:
                fig_ldc = go.Figure(layout=dict(template=template))
                x_levels, y_median = _duration_curve_from_series(median_series_load, levels)
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
                        x=x_levels,
                        y=y_median,
                        mode="lines",
                        line=dict(color="black", width=2),
                        name="Model median",
                    ))
                observed_values: List[float] = []
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
        band_data = {}

    else:

        aligned_df = pd.concat(per_sim.values(), axis=1).sort_index()

        arr = aligned_df.to_numpy(dtype=float)

        percs = [5, 10, 25, 50, 75, 90, 95]

        if arr.shape[1] == 0:
            q_df = pd.DataFrame()
            band_data = {}
        else:
            qs = np.nanpercentile(arr, percs, axis=1)
            q = {p: qs[i, :] for i, p in enumerate(percs)}
            q_df = pd.DataFrame({
                "p05": q[5],
                "p10": q[10],
                "p25": q[25],
                "p50": q[50],
                "p75": q[75],
                "p90": q[90],
                "p95": q[95],
            }, index=aligned_df.index)
            try:
                min_series = pd.Series(np.nanmin(arr, axis=1), index=aligned_df.index, name="min")
                max_series = pd.Series(np.nanmax(arr, axis=1), index=aligned_df.index, name="max")
            except Exception:
                band_data = {}
            else:
                band_data = {
                    "ensemble": {
                        "min": min_series,
                        "max": max_series,
                        "median": q_df.get("p50"),
                        "p05": q_df.get("p05"),
                        "p25": q_df.get("p25"),
                        "p75": q_df.get("p75"),
                        "p95": q_df.get("p95"),
                    }
                }
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
        band_data=band_data,
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

