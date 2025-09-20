from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Union, Sequence, Tuple
from datetime import datetime, date

import numpy as np
import pandas as pd
import re


# -----------------------------
# Time + resampling helpers
# -----------------------------

def _ensure_dt_index(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
    return df


def _make_freq_string(freq_code: str, bin_size: int) -> str:
    freq_code = str(freq_code).upper()
    if freq_code not in {"D", "W", "M", "A"}:
        freq_code = "D"
    bin_size = max(int(bin_size), 1)
    return f"{bin_size}{freq_code}"


def _slice_time(df: pd.DataFrame,
                start: Optional[Union[str, datetime, date]] = None,
                end: Optional[Union[str, datetime, date]] = None) -> pd.DataFrame:
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


def _filter_season(df: pd.DataFrame, season_months: Optional[List[int]]) -> pd.DataFrame:
    if not season_months:
        return df
    months = set(int(m) for m in season_months)
    return df.loc[df.index.month.isin(months)]


def _resample_series(df: pd.DataFrame,
                     value_col: str,
                     *,
                     freq: str,
                     how: str = "mean",
                     flow_col: Optional[str] = None) -> pd.Series:
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


# -----------------------------
# Measured data mapping
# -----------------------------

def _detect_value_col(measured_df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "kg_per_day",  # project-specific
        "VALOR", "valor", "VALUE", "value", "Resultado", "RESULTADO", "CONCENTRACION", "concentracion",
    ]
    for c in candidates:
        if c in measured_df.columns:
            try:
                print(f"Detected measured value column: '{c}'")
            except Exception:
                pass
            return c
    exclude = {"F_MUESTREO", "est_estaci", "NOMBRE", "EST_ESTACI", "fecha", "Fecha"}
    for c in measured_df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(measured_df[c]):
            return c
    return None


def _normalize_meas_map_for_var(measured_var_map: Dict[str, object], var: str) -> Dict[int, List[str]]:
    if measured_var_map is None:
        return {}
    raw = measured_var_map.get(var, None)
    if raw is None:
        # try case-insensitive lookup
        try:
            low = {str(k).strip().lower(): v for k, v in measured_var_map.items()}
            raw = low.get(str(var).strip().lower(), None)
        except Exception:
            raw = None
    if raw is None:
        # try loose substring match (either direction)
        try:
            vlow = str(var).strip().lower()
            for k, v in measured_var_map.items():
                klow = str(k).strip().lower()
                if klow in vlow or vlow in klow:
                    raw = v
                    break
        except Exception:
            raw = None
    if raw is None:
        return {}
    norm: Dict[int, List[str]] = {}
    if isinstance(raw, dict):
        keys_int: List[int] = []
        convertible = True
        for k in raw.keys():
            try:
                keys_int.append(int(k))
            except Exception:
                convertible = False
                break
        if convertible and all(ki in (1, 2, 3) for ki in keys_int):
            for k, v in raw.items():
                ki = int(k)
                if isinstance(v, (list, tuple, set)):
                    vv: List[str] = []
                    for item in v:
                        if isinstance(item, (list, tuple)) and len(item) >= 1:
                            if str(item[0]).strip():
                                vv.append(str(item[0]))
                        elif isinstance(item, str) and item.strip():
                            vv.append(item)
                    if vv:
                        norm[ki] = vv
                elif isinstance(v, (list, tuple)) and len(v) >= 1:
                    name = v[0]
                    if isinstance(name, str) and name.strip():
                        norm[ki] = [name]
                elif isinstance(v, str) and v.strip():
                    norm[ki] = [v]
        else:
            items: List[Tuple[int, List[str]]] = []
            for k, v in raw.items():
                try:
                    weight = int(k)
                except Exception:
                    continue
                names: List[str] = []
                if isinstance(v, (list, tuple)):
                    if len(v) >= 1 and isinstance(v[0], str):
                        first = v[0].strip()
                        if first:
                            names.append(first)
                    else:
                        for item in v:
                            if isinstance(item, (list, tuple)) and len(item) >= 1 and isinstance(item[0], str):
                                nm = item[0].strip()
                                if nm:
                                    names.append(nm)
                            elif isinstance(item, str) and item.strip():
                                names.append(item)
                elif isinstance(v, str) and v.strip():
                    names.append(v)
                if names:
                    items.append((weight, names))
            items.sort(key=lambda x: x[0], reverse=True)
            for idx, (_w, names) in enumerate(items[:3], start=1):
                norm[idx] = names
    elif isinstance(raw, (list, tuple)):
        for i, v in enumerate(raw[:3], start=1):
            if isinstance(v, (list, tuple, set)):
                norm[i] = [str(x) for x in v]
            elif isinstance(v, str):
                norm[i] = [v]
    return norm


def _measured_options_for_category(measured_df: pd.DataFrame,
                                   name_col: str,
                                   allowed_names: Optional[Sequence[str]]) -> List[str]:
    names = measured_df[name_col].dropna().astype(str)
    uniq = pd.Index(names.unique())
    if not allowed_names:
        return []
    allowed = pd.Index([str(x) for x in allowed_names])
    present = uniq.intersection(allowed)
    return sorted(present.tolist())


def _chem_options_with_placeholder(names: Sequence[str], extra: Optional[str] = None) -> List[Tuple[str, Optional[str]]]:
    opts: List[Tuple[str, Optional[str]]] = [("- select -", None)]
    seen = set()
    if extra is not None and str(extra) not in names:
        opts.append((str(extra), str(extra)))
        seen.add(str(extra))
    for n in names:
        sn = str(n)
        if sn in seen:
            continue
        opts.append((sn, sn))
        seen.add(sn)
    return opts


def _aggregate_measured(
    measured_df: pd.DataFrame,
    *,
    date_col: str,
    station_col: str,
    name_col: str,
    value_col: str,
    selected_name: str,
    selected_stations: Sequence[str],
    start: Optional[Union[str, datetime, date]] = None,
    end: Optional[Union[str, datetime, date]] = None,
    season_months: Optional[List[int]] = None,
) -> Dict[str, pd.Series]:
    if measured_df.empty or not selected_name:
        return {}
    df = measured_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df[df[name_col].astype(str) == str(selected_name)]
    if start is not None:
        df = df[df[date_col] >= pd.to_datetime(start)]
    if end is not None:
        df = df[df[date_col] <= pd.to_datetime(end)]
    if season_months:
        months = set(int(m) for m in season_months)
        df = df[df[date_col].dt.month.isin(months)]
    if selected_stations:
        sel = set(str(s) for s in selected_stations)
        df = df[df[station_col].astype(str).isin(sel)]

    if df.empty:
        return {}

    grp = (df.groupby([df[station_col].astype(str), df[date_col].dt.floor('D')])[value_col]
             .mean().sort_index())

    result: Dict[str, pd.Series] = {}
    for station in grp.index.get_level_values(0).unique():
        s = grp.xs(station, level=0)
        s.index.name = None
        s.name = station
        result[station] = s
    return result


def _period_day_counts(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    *,
    freq: str,
    season_months: Optional[List[int]] = None,
) -> pd.Series:
    if start_date > end_date:
        start_date, end_date = end_date, start_date
    days = pd.date_range(start_date.normalize(), end_date.normalize(), freq='D')
    s = pd.Series(1.0, index=days)
    if season_months:
        months = set(int(m) for m in season_months)
        s = s[s.index.month.isin(months)]
    counts = s.resample(freq).sum()
    return counts


# -----------------------------
# Measured cleaning + conversion helpers
# -----------------------------

def convert_measured_mgL_to_kg_per_day(
    df_samples: pd.DataFrame,
    df_flow: pd.DataFrame,
    *,
    sample_date_col: str = "F_MUESTREO",
    sample_value_col: str = "RESULTADO",
    flow_date_col: str = "date",
    flow_value_col: str = "water_flow_m3_d_cubillas",
    kg_col: str = "kg_per_day",
    nonnum_policy: str = "as_na",   # "as_na" | "drop" | "zero"
    negative_policy: str = "zero",   # "keep" | "drop" | "zero"
) -> pd.DataFrame:
    """
    Create/overwrite a kg/day load column on measured samples from mg/L concentrations
    joined with daily water flow (m^3/day).

    - kg/day = (mg/L) * (m^3/day) * 0.001
    - Dates are floored to day before joining.

    Policies:
    - nonnum_policy: how to handle values that cannot be coerced to numeric
        * "as_na": keep the row; the numeric column becomes NaN
        * "drop":  drop rows where coercion produced NaN (for sample or flow)
        * "zero": set non-numeric values to 0
        * "half_MDL": set non-numeric sample values to half the the typical Method Detection Limit
    - negative_policy: how to handle negative mg/L sample values
        * "keep": keep negatives as-is
        * "drop": drop rows with negative sample values
        * "zero": set negatives to 0
    """
    s = df_samples.copy()
    f = df_flow.copy()

    # Normalize dates to daily
    s[sample_date_col] = pd.to_datetime(s[sample_date_col], errors="coerce").dt.floor("D")
    f[flow_date_col] = pd.to_datetime(f[flow_date_col], errors="coerce").dt.floor("D")

    # Coerce numeric
    s[sample_value_col] = pd.to_numeric(s[sample_value_col], errors="coerce").astype(float)
    f[flow_value_col] = pd.to_numeric(f[flow_value_col], errors="coerce").astype(float)

    # Negative policy on sample values
    if negative_policy == "drop":
        s = s[s[sample_value_col] >= 0]
    elif negative_policy == "zero":
        s.loc[s[sample_value_col] < 0, sample_value_col] = 0.0
    # else: keep

    # Reduce flow to daily (mean across duplicates)
    f_reduced = (
        f[[flow_date_col, flow_value_col]]
        .groupby(flow_date_col, as_index=False)
        .mean(numeric_only=True)
    )

    # Join by day
    merged = s.merge(
        f_reduced,
        left_on=sample_date_col,
        right_on=flow_date_col,
        how="left",
        suffixes=("", "_flow"),
    )
    # Remove right join key to keep original schema tidy
    if flow_date_col in merged.columns:
        merged = merged.drop(columns=[flow_date_col])

    # Handle non-numeric coercion results
    # Identify non-numeric sample values (including NaN and strings like "LC")
    nonnum_mask = ~pd.to_numeric(merged[sample_value_col], errors="coerce").notna()
    nonnum_count = nonnum_mask.sum()
    print(f"[INFO] Found {nonnum_count} non-numeric sample values (including NaN and strings like 'LC').")
    if nonnum_policy == "drop":
        before = len(merged)
        merged = merged[~nonnum_mask & merged[flow_value_col].notna()]
        after = len(merged)
        print(f"[INFO] Dropped {before - after} rows with non-numeric sample or missing flow values.")
    elif nonnum_policy == "zero":
        merged.loc[nonnum_mask, sample_value_col] = 0.0
        print(f"[INFO] Set {nonnum_count} non-numeric sample values to 0.")
    elif nonnum_policy == "half_MDL":
        typical_mdl = 0.2  # mg/L
        half_mdl = typical_mdl * 0.5
        merged.loc[nonnum_mask, sample_value_col] = half_mdl
        print(f"[INFO] Set {nonnum_count} non-numeric sample values to half MDL ({half_mdl}).")
    # else keep as NaN
    # else keep as NaN


    # Compute kg/day
    with np.errstate(invalid='ignore'):
        merged[kg_col] = merged[sample_value_col] * merged[flow_value_col] * 0.001

    return merged


# -----------------------------
# Event detection helpers
# -----------------------------

def add_event_flags(df,
                    thresholds: dict,
                    intervals: dict,
                    time_col: str = None,
                    flow_col: str = 'Q'):
    """
    Add boolean event-flag columns to a dataframe for multiple threshold/interval defs.

    - thresholds: dict[name] = numeric_value OR string:
        * 'q1','q2','q3'        -> 25th,50th,75th percentiles
        * 'q25','q50','q75'    -> same as above
        * 'pNN' or 'NNpct'     -> NN percentile, e.g. 'p90' or '90pct'
      (values are interpreted in same units as flow_col)
    - intervals: dict[name] = etmin_days (minimum event length in days)
    - time_col: optional column name containing datetimes (if provided, it will be set as index)
    - flow_col: name of the flow column used for detection

    Returns a copy of df with added boolean columns: f'{name}_event' (True inside event).
    Prints diagnostic steps to console.
    """
    df = df.copy()
    # set index to datetime
    if time_col is not None:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)
    df.index = pd.to_datetime(df.index)
    if flow_col not in df.columns:
        raise KeyError(f'flow_col "{flow_col}" not found in dataframe')

    q_series = df[flow_col]
    # compute median sampling interval in days
    diffs = q_series.index.to_series().diff().dropna()
    if diffs.empty:
        raise ValueError("Not enough timestamps to compute sampling interval.")
    dt_days = diffs.median().total_seconds() / 86400.0
    try:
        print(f"[info] median sampling interval = {dt_days:.6f} days ({dt_days*24:.3f} h)")
    except Exception:
        pass

    def resolve_threshold(v):
        """Return numeric threshold given v (numeric or string token)."""
        if isinstance(v, (int, float, np.floating, np.integer)) and not np.isnan(v):
            return float(v)
        if not isinstance(v, str):
            raise ValueError("threshold must be numeric or string token like 'q1' or 'p90'")
        s = v.strip().lower()
        # q1/q2/q3 shortcuts
        if s in ('q1', 'q25'):
            pct = 25.0
        elif s in ('q2', 'q50'):
            pct = 50.0
        elif s in ('q3', 'q75'):
            pct = 75.0
        else:
            m = re.match(r'^(p|)(\d{1,2}|100)(pct|)$', s)
            if m:
                pct = float(m.group(2))
            else:
                # try pattern like '90' or '90pct'
                m2 = re.match(r'^(\d{1,2}|100)$', s)
                if m2:
                    pct = float(m2.group(1))
                else:
                    raise ValueError(f"unrecognized threshold token '{v}'")
        return float(np.nanpercentile(q_series.values, pct))

    def runs_from_bool(mask):
        """List of (start_idx, end_idx inclusive) for True runs."""
        mask = mask.astype(bool)
        if not mask.any():
            return []
        d = np.diff(np.r_[0, mask.view(np.int8), 0])
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0] - 1
        return list(zip(starts, ends))

    n = len(q_series)
    q = q_series.values

    for name, thr_token in thresholds.items():
        if name not in intervals:
            raise KeyError(f'intervals must contain key "{name}"')
        etmin_days = float(intervals[name])
        min_samples = int(np.ceil(etmin_days / dt_days))
        thr_value = resolve_threshold(thr_token)

        try:
            print(f"[proc] '{name}': threshold resolved = {thr_value} (units same as '{flow_col}'), etmin = {etmin_days} days -> min_samples = {min_samples}")
        except Exception:
            pass

        mask = (q >= thr_value) & (~np.isnan(q))
        runs = runs_from_bool(mask)
        # prune short runs, produce final boolean mask
        final_mask = np.zeros(n, dtype=bool)
        n_events = 0
        for s_idx, e_idx in runs:
            length = e_idx - s_idx + 1
            if length >= min_samples:
                n_events += 1
                final_mask[s_idx:e_idx+1] = True

        colname = f'{name}_event'
        df[colname] = final_mask
        try:
            print(f"[result] '{name}': runs found = {len(runs)}, kept (>= {min_samples} samples) = {n_events}")
        except Exception:
            pass

    return df
