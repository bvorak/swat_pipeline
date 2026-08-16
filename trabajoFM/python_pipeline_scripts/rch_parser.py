from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Literal, Optional, Union, Any

import pickle

import os
import pandas as pd
import numpy as np
import re

DEFAULT_RCH_COLUMNS = [
    "object type","RCH","GIS","MON","AREAkm2","FLOW_INcms","FLOW_OUTcmscms","EVAPcms","TLOSScms",
    "SED_INtons","SED_OUTtons","SEDCONCmg/L","ORGN_INkg","ORGN_OUTkg","ORGP_INkg","ORGP_OUTkg",
    "NO3_INkg","NO3_OUTkg","NH4_INkg","NH4_OUTkg","NO2_INkg","NO2_OUTkg","MINP_INkg","MINP_OUTkg",
    "CHLA_INkg","CHLA_OUTkg","CBOD_INkg","CBOD_OUTkg","DISOX_INkg","DISOX_OUTkg","SOLPST_INmg",
    "SOLPST_OUTmg","SORPST_INmg","SORPST_OUTmg","REACTPSTmg","VOLPSTmg","SETTLPSTmg","RESUSP_PSTmg",
    "DIFFUSEPSTmg","REACBEDPSTmg","BURYPSTmg","BED_PSTmg","BACTP_OUTct","BACTLP_OUTct","CMETAL#1kg",
    "CMETAL#2kg","CMETAL#3kg","TOT_Nkg","TOT_Pkg","NO3ConcMg/l","WTMPdegc","Salt1","Salt2","Salt3",
    "Salt4","Salt5","Salt6","Salt7","Salt8","Salt9","Salt10","SAR","EC"
]

DEFAULT_DROP_COLS = ["object type","total_days","GIS","MON","AREAkm2","YEAR"] # To have a clean dataframe, adjust as wanted

def load_output_rch(
    file_path: str,
    cio_file: str,  
    *,
    columns: list[str] = None,
    skiprows: int = 9,
    group_size: int = 17,        
    add_area_ha: bool = True,
    hectare_per_km2: float = 100.0,
    drop_cols: list[str] = None,
    reorder_date_cols: bool = True
) -> pd.DataFrame:
    """
    Loads SWAT output.rch, attach datetime based on file.cio, compute area_ha, and tidy columns.

    Parameters
    ----------
    file_path : str
        Path to output.rch
    cio_file : str
        Path to file.cio (used to derive simulation start date and possibly other metadata)
    columns : list[str], optional
        Column names for output.rch; defaults to DEFAULT_RCH_COLUMNS
    skiprows : int, optional
        Lines to skip before header/data (default 9 for SWAT outputs)
    group_size : int, optional
        Number of rows per simulated day in output.rch (default 17)
    add_area_ha : bool, optional
        If True, adds area_ha = AREAkm2 * 100
    drop_cols : list[str], optional
        Columns to drop at the end (default DEFAULT_DROP_COLS)
    reorder_date_cols : bool, optional
        If True, moves 'date' to col 3 and 'YEAR' to col 4 (0-based)

    Returns
    -------
    pd.DataFrame
        Tidy DataFrame with date, optional area_ha, and dropped columns.
    """

    # ---- helpers to read cio ----
    def _getModelParameter(param: str, parameterfile: str) -> str | None:
        with open(parameterfile, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if param in line:
                    # value expected before first '|'
                    return line.partition("|")[0].strip()
        return None

    def _getStartDate(swatiofile: str) -> date:
        skip_year = int(_getModelParameter("NYSKIP", swatiofile))
        sim_year = int(_getModelParameter("NBYR", swatiofile))
        start_year = int(_getModelParameter("IYR", swatiofile))
        start_day = int(_getModelParameter("IDAF", swatiofile))
        # first day of (start_year + skip_year) plus (start_day - 1)
        return date(start_year + skip_year, 1, 1) + timedelta(days=start_day - 1)

    # ---- defaults ----
    if columns is None:
        columns = DEFAULT_RCH_COLUMNS
    if drop_cols is None:
        drop_cols = DEFAULT_DROP_COLS

    # ---- read output.rch ----
    df = pd.read_csv(file_path, sep=r"\s+", skiprows=skiprows, header=None, names=columns, engine="python")

    # ---- create total_days from row index / group_size ----
    df.index.name = "total_days"
    df.reset_index(drop=False, inplace=True)
    if group_size <= 0:
        raise ValueError("group_size must be > 0")
    df["total_days"] = df["total_days"] // group_size

    # ---- compute date & YEAR from cio ----
    start_date = _getStartDate(cio_file)
    df["date"] = df["total_days"].apply(lambda d: start_date + timedelta(days=int(d)))
    df["date"] = pd.to_datetime(df["date"])
    df["YEAR"] = df["date"].dt.year

    # ---- reorder date/YEAR columns if desired ----
    if reorder_date_cols:
        # insert 'date' at position 3 and 'YEAR' at position 4 (0-based)
        df.insert(3, "date", df.pop("date"))
        df.insert(4, "YEAR", df.pop("YEAR"))

    # ---- add area_ha if requested ----
    if add_area_ha:
        if "AREAkm2" not in df.columns:
            raise KeyError("AREAkm2 column not found; cannot compute area_ha.")
        df["area_ha"] = df["AREAkm2"] * 100
        # place area_ha after AREAkm2 (which is index 6 after inserts; but robustly reinsert)
        # insert at 7 like your original
        df.insert(7, "area_ha", df.pop("area_ha"))

    # ---- final tidy: drop columns ----
    to_drop = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=to_drop)

    return df


def load_multiple_rch_from_folders(base_folders: list[str], **kwargs) -> dict[str, pd.DataFrame]:
    """
    Loads output.rch and file.cio from multiple SWAT TxtInOut base folders.

    Parameters
    ----------
    base_folders : list[str]
        List of paths to TxtInOut folders.
    **kwargs :
        Additional keyword arguments to pass to load_output_rch().

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping parent folder names to loaded DataFrames.
    """
    results = {}

    for folder in base_folders:
        # Normalize path
        folder = os.path.abspath(folder)
        parent_folder = os.path.dirname(folder)
        if "txtinout" in os.path.basename(folder).lower():
            realization_name = f"rch_{os.path.basename(parent_folder)}"
        else:
            realization_name = f"rch_{os.path.basename(folder)}"
        print(f"Loading from folder: {folder} as {realization_name}")

        # Expected files
        output_rch_path = os.path.join(folder, "output.rch")
        cio_path = os.path.join(folder, "file.cio")

        # Safety check
        if not os.path.isfile(output_rch_path):
            raise FileNotFoundError(f"output.rch not found in {folder}")
        if not os.path.isfile(cio_path):
            raise FileNotFoundError(f"file.cio not found in {folder}")

        # Load using the previous function
        df = load_output_rch(file_path=output_rch_path, cio_file=cio_path, **kwargs)

        results[realization_name] = df

    return results


# -----------------------------
# Find folders of RUN number
# -----------------------------


def find_run_folders(run_number, path=r"C:\SWAT\RSWAT\cubillas\mc_results"):
    """
    Find folders in the given path that match the pattern:
    runXXXXXX_realYYYYYY_*
    
    Parameters
    ----------
    run_number : int
        The run number to match (will be zero-padded to 6 digits).
    path : str, optional
        The folder path to search in. Defaults to 
        'C:\\SWAT\\RSWAT\\cubillas\\mc_results'.
    
    Returns
    -------
    list of str
        A list of matching folder names.
    """
    # Format the number as 6 digits with leading zeros
    run_str = f"run{run_number:06d}_"
    
    try:
        # List all entries in the path
        all_entries = os.listdir(path)
    except FileNotFoundError:
        print(f"Error: Path '{path}' does not exist.")
        return []
    
    # Keep only directories that start with the correct run string
    matching_folders = [
        os.path.join(path, folder) for folder in all_entries
        if os.path.isdir(os.path.join(path, folder)) and folder.startswith(run_str)
    ]
    
    return matching_folders


# -----------------------------
# converts a run number to a dict of .rch parsed dfs within that run's folders
# -----------------------------


def load_or_build_dfs_for_runs(
    runs: int | Iterable[int],
    *,
    pickle_name_fmt: str = "all_dfs_mc_run_{run}.pkl",
    force_rebuild: bool = False,
    save_pickle: bool = True,
) -> Dict[str, Any]:
    """
    For each run:
      - Look in first folder from find_run_folders(run) for a pickle.
      - Load it unless force_rebuild; otherwise build with load_multiple_rch_from_folders.
      - Coerce result to a dict (compat with old pickles returning lists), then merge.
    Returns ONE merged dict across all runs.
    """
    run_list = [runs] if isinstance(runs, int) else list(runs)
    merged: Dict[str, Any] = {}

    for run in run_list:
        folders = find_run_folders(run)
        if not folders:
            continue
        first_folder = Path(folders[0])
        pkl_path = first_folder / pickle_name_fmt.format(run=run)

        run_obj = None
        if not force_rebuild and pkl_path.exists():
            try:
                with open(pkl_path, "rb") as f:
                    run_obj = pickle.load(f)
            except Exception:
                run_obj = None

        if run_obj is None:
            run_obj = load_multiple_rch_from_folders(folders)
            if save_pickle:
                try:
                    with open(pkl_path, "wb") as f:
                        pickle.dump(run_obj, f)
                except Exception:
                    pass

        run_dict = _coerce_to_dict(run_obj, run)
        # if keys collide across runs, later runs overwrite earlier ones
        merged.update(run_dict)

    return merged







def _coerce_to_dict(obj: Any, run: int) -> Dict[str, Any]:
    """Make sure we hand back a dict no matter what was stored."""
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list):
        # common patterns: list of (name, df) or list of dfs
        try:
            if all(isinstance(x, tuple) and len(x) == 2 for x in obj):
                return dict(obj)  # [(key, df), ...]
        except Exception:
            pass
        # fallback: enumerate
        return {f"run{run}_sim{i}": x for i, x in enumerate(obj)}
    # last resort: wrap single object
    return {f"run{run}": obj}



def load_or_build_dfs_for_runs(
    runs: int | Iterable[int],
    *,
    pickle_name_fmt: str = "all_dfs_mc_run_{run}.pkl",
    force_rebuild: bool = False,
    save_pickle: bool = True,
) -> Dict[str, Any]:
    """
    For each run:
      - Look in first folder from find_run_folders(run) for a pickle.
      - Load it unless force_rebuild; otherwise build with load_multiple_rch_from_folders.
      - Coerce result to a dict (compat with old pickles returning lists), then merge.
    Returns ONE merged dict across all runs.
    """
    run_list = [runs] if isinstance(runs, int) else list(runs)
    merged: Dict[str, Any] = {}

    for run in run_list:
        folders = find_run_folders(run)
        if not folders:
            continue
        first_folder = Path(folders[0])
        pkl_path = first_folder / pickle_name_fmt.format(run=run)

        run_obj = None
        if not force_rebuild and pkl_path.exists():
            try:
                with open(pkl_path, "rb") as f:
                    run_obj = pickle.load(f)
            except Exception:
                run_obj = None

        if run_obj is None:
            run_obj = load_multiple_rch_from_folders(folders)
            if save_pickle:
                try:
                    with open(pkl_path, "wb") as f:
                        pickle.dump(run_obj, f)
                except Exception:
                    pass

        run_dict = _coerce_to_dict(run_obj, run)
        # if keys collide across runs, later runs overwrite earlier ones
        merged.update(run_dict)

    return merged





def compare_dfs(df1: pd.DataFrame, df2: pd.DataFrame):
    # Only compare numeric columns
    df1_numeric = df1.select_dtypes(include=[np.number])
    df2_numeric = df2.select_dtypes(include=[np.number])

    # Handle column name differences (e.g., FLOW_OUTcms vs FLOW_OUTcmscms)
    col_map = {
        "FLOW_OUTcms": "FLOW_OUTcmscms",
        "FLOW_OUTcmscms": "FLOW_OUTcms"
    }
    df1_cols = set(df1_numeric.columns)
    df2_cols = set(df2_numeric.columns)

    # Prepare lists of columns to compare
    common_cols = [c for c in df1_numeric.columns if c in df2_numeric.columns and c not in col_map]
    mapped_comparisons = []
    for c1, c2 in col_map.items():
        if c1 in df1_cols and c2 in df2_cols:
            mapped_comparisons.append((c1, c2))
        elif c2 in df1_cols and c1 in df2_cols:
            mapped_comparisons.append((c2, c1))

    # Align DataFrames by columns
    df1_common = df1_numeric[common_cols]
    df2_common = df2_numeric[common_cols]

    # Compare common columns directly
    if df1_common.shape != df2_common.shape:
        raise ValueError("DataFrames must have the same shape for direct comparison.")

    diff_mask = df1_common != df2_common
    any_diff = diff_mask.any().any()

    # Metrics for common columns
    abs_diff = (df1_common - df2_common).abs()
    mean_abs_diff = abs_diff.mean()
    max_abs_diff = abs_diff.max()
    mean_rel_diff = ((abs_diff / (df1_common.replace(0, np.nan))).mean()).replace([np.inf, -np.inf], np.nan)

    # Compare mapped columns
    mapped_diff_columns = []
    mapped_diff_counts = {}
    mapped_mean_abs_diff = {}
    mapped_max_abs_diff = {}
    mapped_mean_rel_diff = {}
    for c1, c2 in mapped_comparisons:
        col_diff_mask = df1_numeric[c1] != df2_numeric[c2]
        if col_diff_mask.any():
            mapped_diff_columns.append(f"{c1} vs {c2}")
            mapped_diff_counts[f"{c1} vs {c2}"] = int(col_diff_mask.sum())
            abs_diff_col = (df1_numeric[c1] - df2_numeric[c2]).abs()
            mapped_mean_abs_diff[f"{c1} vs {c2}"] = abs_diff_col.mean()
            mapped_max_abs_diff[f"{c1} vs {c2}"] = abs_diff_col.max()
            mapped_mean_rel_diff[f"{c1} vs {c2}"] = (abs_diff_col / df1_numeric[c1].replace(0, np.nan)).mean()
            any_diff = True

    # Columns with at least one difference
    diff_columns = diff_mask.any(axis=0)
    diff_counts = diff_mask.sum(axis=0)[diff_columns]

    # Collect metrics for columns with differences
    metrics = {}
    for col in diff_columns.index[diff_columns]:
        metrics[col] = {
            "count": int(diff_counts[col]),
            "mean_abs_diff": float(mean_abs_diff[col]),
            "max_abs_diff": float(max_abs_diff[col]),
            "mean_rel_diff": float(mean_rel_diff[col]) if col in mean_rel_diff else None
        }
    for col in mapped_diff_columns:
        metrics[col] = {
            "count": mapped_diff_counts[col],
            "mean_abs_diff": mapped_mean_abs_diff[col],
            "max_abs_diff": mapped_max_abs_diff[col],
            "mean_rel_diff": mapped_mean_rel_diff[col]
        }

    # Print easy to read stats
    print("Comparison Summary")
    print("==================")
    print(f"Any difference: {any_diff}")
    print(f"Number of numeric columns with differences: {int(diff_columns.sum()) + len(mapped_diff_columns)}")
    print("Numeric columns with differences:")
    for col in diff_columns.index[diff_columns]:
        m = metrics[col]
        print(f"  {col}: count={m['count']}, mean_abs_diff={m['mean_abs_diff']:.4g}, max_abs_diff={m['max_abs_diff']:.4g}, mean_rel_diff={m['mean_rel_diff']:.4g}")
    for col in mapped_diff_columns:
        m = metrics[col]
        print(f"  {col}: count={m['count']}, mean_abs_diff={m['mean_abs_diff']:.4g}, max_abs_diff={m['max_abs_diff']:.4g}, mean_rel_diff={m['mean_rel_diff']:.4g}")

    result = {
        "any_difference": any_diff,
        "n_diff_columns": int(diff_columns.sum()) + len(mapped_diff_columns),
        "diff_columns": diff_columns.index[diff_columns].tolist() + mapped_diff_columns,
        "diff_counts": {**diff_counts.to_dict(), **mapped_diff_counts},
        "metrics": metrics
    }
    return result
