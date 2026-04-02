from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Generic .chm reader  (mirrors sol_reader.read_sol_file)
# ---------------------------------------------------------------------------

def read_chm_file(path: Union[str, Path]) -> Dict[str, List[float]]:
    """Parse a SWAT ``.chm`` file and return every numeric key-value row.

    Returns a dict mapping the label (stripped) to a list of floats
    (one per soil layer).  Non-numeric rows (header, Pesticide table, etc.)
    are silently skipped.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    result: Dict[str, List[float]] = {}
    for line in lines:
        if ":" not in line:
            continue
        colon_idx = line.index(":")
        label = line[:colon_idx].strip()
        after_colon = line[colon_idx + 1:]

        tokens = after_colon.split()
        if not tokens:
            continue

        values: List[float] = []
        for tok in tokens:
            try:
                values.append(float(tok))
            except ValueError:
                break
        else:
            if values:
                result[label] = values

    return result


def read_chm_value(
    path: Union[str, Path],
    key: str,
    *,
    layer: int = 1,
) -> float:
    """Return a single numeric value from a ``.chm`` file.

    Parameters
    ----------
    key : label to match exactly (e.g. ``"Soil NO3 [mg/kg]"``).
    layer : 1-based layer index.
    """
    data = read_chm_file(path)
    if key not in data:
        raise KeyError(f"Key {key!r} not found in {path}")
    values = data[key]
    if layer < 1 or layer > len(values):
        raise IndexError(
            f"Layer {layer} out of range (file has {len(values)} value(s) for {key!r})"
        )
    return values[layer - 1]


# ---------------------------------------------------------------------------
# Discover run folders for a given run number
# ---------------------------------------------------------------------------

def _find_run_folders(
    run_number: int,
    mc_dir: Union[str, Path],
) -> Tuple[Path, Path]:
    """Return ``(folder_1, folder_2)`` inside *mc_dir* for a run number.

    The naming convention is ``run<NNN>_real<MMM>_1`` / ``run<NNN>_real<MMM+1>_2``
    where NNN and MMM are zero-padded to 6 digits and the realization numbers
    may differ between the two suffixes.  We glob for the pattern and match
    exactly one folder per suffix.
    """
    mc_dir = Path(mc_dir)
    prefix = f"run{run_number:06d}_real"

    folder_1 = folder_2 = None
    for d in mc_dir.iterdir():
        if not d.is_dir() or not d.name.startswith(prefix):
            continue
        if d.name.endswith("_1"):
            folder_1 = d
        elif d.name.endswith("_2"):
            folder_2 = d

    if folder_1 is None:
        raise FileNotFoundError(f"No *_1 folder found for run {run_number} in {mc_dir}")
    if folder_2 is None:
        raise FileNotFoundError(f"No *_2 folder found for run {run_number} in {mc_dir}")
    return folder_1, folder_2


# ---------------------------------------------------------------------------
# Average Layer-1 CHM values across _1 and _2 folders for a run
# ---------------------------------------------------------------------------

def get_run_chm_averages(
    run_number: int,
    mc_dir: Union[str, Path] = r"C:\SWAT\RSWAT\cubillas\mc_realizations",
    *,
    keys: Optional[Sequence[str]] = None,
    layer: int = 1,
) -> Dict[int, Dict[str, float]]:
    """Read matching ``.chm`` files from both run folders and return the
    element-wise average of their Layer-*layer* values.

    Parameters
    ----------
    run_number : e.g. 186
    mc_dir : parent folder containing ``run<NNN>_real<MMM>_{1,2}`` subfolders.
    keys : if given, only average these labels; otherwise average all numeric
           labels found in both files.
    layer : 1-based layer index (default: Layer 1).

    Returns
    -------
    ``{hru_id: {label: averaged_value, ...}, ...}``
    """
    folder_1, folder_2 = _find_run_folders(run_number, mc_dir)

    # Index CHM files in folder_2 by numeric stem
    index_2: Dict[int, Path] = {}
    for p in folder_2.glob("*.chm"):
        try:
            index_2[int(p.stem)] = p
        except ValueError:
            continue

    results: Dict[int, Dict[str, float]] = {}

    for p1 in sorted(folder_1.glob("*.chm")):
        try:
            hru_id = int(p1.stem)
        except ValueError:
            continue
        p2 = index_2.get(hru_id)
        if p2 is None:
            continue

        data1 = read_chm_file(p1)
        data2 = read_chm_file(p2)

        use_keys = keys if keys is not None else sorted(set(data1) & set(data2))

        row: Dict[str, float] = {}
        for k in use_keys:
            vals1 = data1.get(k)
            vals2 = data2.get(k)
            if vals1 is None or vals2 is None:
                continue
            if layer < 1 or layer > min(len(vals1), len(vals2)):
                continue
            row[k] = (vals1[layer - 1] + vals2[layer - 1]) / 2.0

        if row:
            results[hru_id] = row

    return results


# ---------------------------------------------------------------------------
# Correlation analysis between CHM averages and SOL properties
# ---------------------------------------------------------------------------

def correlate_chm_sol(
    chm_averages: Dict[int, Dict[str, float]],
    sol_dir: Union[str, Path] = (
        r"C:\SWAT\RSWAT\cubillas\cubillas_set_219_ruben"
        r"\cubillas_BASE_set-219"
        r"\BASE recreated from arcswat default\TxtInOut_1"
    ),
    *,
    chm_keys: Optional[Sequence[str]] = None,
    sol_keys: Optional[Sequence[str]] = None,
    sol_layer: int = 1,
    method: str = "pearson",
) -> "pd.DataFrame":
    """Compute pair-wise correlations between CHM averages and SOL properties.

    Parameters
    ----------
    chm_averages : output of :func:`get_run_chm_averages`.
    sol_dir : BASE TxtInOut folder containing ``.sol`` files.
    chm_keys : CHM labels to include (default: all found in *chm_averages*).
    sol_keys : SOL labels to include (default: all numeric keys).
    sol_layer : 1-based layer index for SOL values.
    method : ``"pearson"`` (default), ``"spearman"``, or ``"kendall"``.

    Returns
    -------
    A ``pandas.DataFrame`` with CHM keys as rows, SOL keys as columns,
    and correlation coefficients as values.  An extra ``_pvalue`` DataFrame
    is attached as the ``.pvalues`` attribute.
    """
    import pandas as pd
    from scipy import stats as sp_stats

    from .sol_reader import read_sol_file

    sol_dir = Path(sol_dir)

    # Collect HRU ids present in chm_averages
    hru_ids = sorted(chm_averages.keys())

    # Determine CHM keys to use
    if chm_keys is None:
        all_chm = set()
        for row in chm_averages.values():
            all_chm.update(row.keys())
        chm_keys = sorted(all_chm)

    # Read SOL data for matching HRUs
    sol_data: Dict[int, Dict[str, float]] = {}
    for hru_id in hru_ids:
        for candidate in (sol_dir / f"{hru_id}.sol", sol_dir / f"{hru_id:09d}.sol"):
            if candidate.exists():
                parsed = read_sol_file(candidate)
                row: Dict[str, float] = {}
                for k, vals in parsed.items():
                    if sol_layer <= len(vals):
                        row[k] = vals[sol_layer - 1]
                sol_data[hru_id] = row
                break

    # Determine SOL keys to use
    if sol_keys is None:
        all_sol = set()
        for row in sol_data.values():
            all_sol.update(row.keys())
        sol_keys = sorted(all_sol)

    # Intersect HRUs present in both datasets
    common_hrus = sorted(set(hru_ids) & set(sol_data.keys()))
    if not common_hrus:
        raise ValueError("No HRUs found in both CHM averages and SOL directory")

    # Build aligned arrays
    corr_values: Dict[Tuple[str, str], float] = {}
    p_values: Dict[Tuple[str, str], float] = {}

    for ck in chm_keys:
        for sk in sol_keys:
            x_vals, y_vals = [], []
            for hru in common_hrus:
                chm_row = chm_averages.get(hru, {})
                sol_row = sol_data.get(hru, {})
                if ck in chm_row and sk in sol_row:
                    x_vals.append(chm_row[ck])
                    y_vals.append(sol_row[sk])

            if len(x_vals) < 3:
                corr_values[(ck, sk)] = np.nan
                p_values[(ck, sk)] = np.nan
                continue

            x = np.array(x_vals)
            y = np.array(y_vals)

            # Skip if either series is constant
            if np.ptp(x) == 0 or np.ptp(y) == 0:
                corr_values[(ck, sk)] = np.nan
                p_values[(ck, sk)] = np.nan
                continue

            if method == "spearman":
                r, p = sp_stats.spearmanr(x, y)
            elif method == "kendall":
                r, p = sp_stats.kendalltau(x, y)
            else:
                r, p = sp_stats.pearsonr(x, y)

            corr_values[(ck, sk)] = r
            p_values[(ck, sk)] = p

    # Reshape into DataFrames
    corr_df = pd.DataFrame(
        {sk: {ck: corr_values.get((ck, sk), np.nan) for ck in chm_keys} for sk in sol_keys}
    )
    pval_df = pd.DataFrame(
        {sk: {ck: p_values.get((ck, sk), np.nan) for ck in chm_keys} for sk in sol_keys}
    )
    corr_df.attrs["pvalues"] = pval_df
    corr_df.pvalues = pval_df  # convenience alias (may warn on newer pandas)
    corr_df.attrs["n_hrus"] = len(common_hrus)
    corr_df.attrs["method"] = method

    return corr_df


# ---------------------------------------------------------------------------
# Build a joined DataFrame: CHM averages + raster-interpolated CSV inputs
# ---------------------------------------------------------------------------

def build_chm_vs_csv_df(
    chm_averages: Dict[int, Dict[str, float]],
    csv_path: Union[str, Path] = (
        r"C:\Users\Usuario\OneDrive - UNIVERSIDAD DE HUELVA"
        r"\Granada\TrabajoFM\Genil GEO_INFO_POOL\Input Data"
        r"\Diffuse loads\Soil chemical composition"
        r"\python calculated hru stats\hru_chem_stats.csv"
    ),
    *,
    csv_sep: str = ";",
    csv_id_col: str = "HRU_GIS",
) -> "pd.DataFrame":
    """Join CHM run-averages with the raster-interpolation CSV on HRU id.

    *chm_averages* keys are integers derived from ``int(stem)`` of the
    zero-padded ``.chm`` filenames (e.g. 10001).  The CSV's *csv_id_col*
    is also an integer HRU identifier.

    Returns a tidy DataFrame with one row per HRU and columns for every
    CHM key (prefixed ``chm_``) and every CSV column (prefixed ``csv_``).
    """
    import pandas as pd

    # --- CHM side ---
    chm_rows = []
    for hru_id, vals in chm_averages.items():
        row = {"hru_id": hru_id}
        for k, v in vals.items():
            row[f"chm_{k}"] = v
        chm_rows.append(row)
    df_chm = pd.DataFrame(chm_rows).set_index("hru_id")

    # --- CSV side ---
    df_csv = pd.read_csv(csv_path, sep=csv_sep)
    # Rename columns with csv_ prefix (except the id column)
    rename = {c: f"csv_{c}" for c in df_csv.columns if c != csv_id_col}
    df_csv = df_csv.rename(columns=rename)
    df_csv = df_csv.set_index(csv_id_col)
    df_csv.index.name = "hru_id"

    # --- Join ---
    df = df_chm.join(df_csv, how="inner")
    return df
