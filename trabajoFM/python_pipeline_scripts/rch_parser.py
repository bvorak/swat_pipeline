from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Literal, Optional, Union

import os
import pandas as pd
import re

DEFAULT_RCH_COLUMNS = [
    "object type","RCH","GIS","MON","AREAkm2","FLOW_INcms","FLOW_OUTcms","EVAPcms","TLOSScms",
    "SED_INtons","SED_OUTtons","SEDCONCmg/L","ORGN_INkg","ORGN_OUTkg","ORGP_INkg","ORGP_OUTkg",
    "NO3_INkg","NO3_OUTkg","NH4_INkg","NH4_OUTkg","NO2_INkg","NO2_OUTkg","MINP_INkg","MINP_OUTkg",
    "CHLA_INkg","CHLA_OUTkg","CBOD_INkg","CBOD_OUTkg","DISOX_INkg","DISOX_OUTkg","SOLPST_INmg",
    "SOLPST_OUTmg","SORPST_INmg","SORPST_OUTmg","REACTPSTmg","VOLPSTmg","SETTLPSTmg","RESUSP_PSTmg",
    "DIFFUSEPSTmg","REACBEDPSTmg","BURYPSTmg","BED_PSTmg","BACTP_OUTct","BACTLP_OUTct","CMETAL#1kg",
    "CMETAL#2kg","CMETAL#3kg","TOT_Nkg","TOT_Pkg","NO3ConcMg/l","WTMPdegc","Salt1","Salt2","Salt3",
    "Salt4","Salt5","Salt6","Salt7","Salt8","Salt9","Salt10","SAR","EC"
]

DEFAULT_DROP_COLS = ["object type","total_days","GIS","MON","AREAkm2","YEAR"]

def load_output_rch(
    file_path: Union[str, Path],
    cio_file: Union[str, Path],
    *,
    columns: Optional[list[str]] = None,
    skiprows: int = 9,
    group_size: int = 17,        # rows per day in output.rch (typical)
    add_area_ha: bool = True,
    hectare_per_km2: float = 100.0,
    drop_cols: Optional[list[str]] = None,
    reorder_date_cols: bool = True
) -> pd.DataFrame:
    """
    Load SWAT output.rch, attach datetime based on file.cio, compute area_ha, and tidy columns.

    Parameters
    ----------
    file_path : str
        Path to output.rch
    cio_file : str
        Path to file.cio (used to derive simulation start date)
    columns : list[str], optional
        Column names for output.rch; defaults to DEFAULT_RCH_COLUMNS
    skiprows : int, optional
        Lines to skip before header/data (default 9 for SWAT outputs)
    group_size : int, optional
        Number of rows per simulated day in output.rch (default 17)
    add_area_ha : bool, optional
        If True, adds area_ha = AREAkm2 * hectare_per_km2
    hectare_per_km2 : float, optional
        Conversion factor (default 100 ha per km²)
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
    def _getModelParameter(param: str, parameterfile: Union[str, Path]) -> Optional[str]:
        with open(parameterfile, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if param in line:
                    # value expected before first '|'
                    return line.partition("|")[0].strip()
        return None

    def _getStartDate(swatiofile: Union[str, Path]) -> date:
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
    file_path = str(file_path)
    cio_file = str(cio_file)
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
        df["area_ha"] = df["AREAkm2"] * hectare_per_km2
        # place area_ha after AREAkm2 (which is index 6 after inserts; but robustly reinsert)
        # insert at 7 like your original
        df.insert(7, "area_ha", df.pop("area_ha"))

    # ---- final tidy: drop columns ----
    to_drop = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=to_drop)

    return df



def load_multiple_rch_from_folders(
    base_folders: Iterable[Union[str, Path]],
    *,
    return_dict: bool = False,
    name_prefix: str = "rch_",
    name_from: Literal["parent", "folder"] = "folder",
    **kwargs,
) -> Union[list[pd.DataFrame], dict[str, pd.DataFrame]]:
    """
    Load output.rch and file.cio from multiple SWAT TxtInOut base folders.

    Parameters
    ----------
    base_folders : Iterable[Union[str, Path]]
        Paths to output folders ('out_' needs to be in their names).
    return_dict : bool, optional
        If True, returns a dict[name->DataFrame]; otherwise returns a list of DataFrames (default).
    name_prefix : str, optional
        Prefix to add to generated names when return_dict=True (default 'rch_').
    name_from : {'parent','folder'}
        How to derive the dict key name when return_dict=True:
        - 'parent': one folder above TxtInOut (default)
        - 'folder': the TxtInOut folder name itself
    **kwargs :
        Extra args passed to load_output_rch().

    Returns
    -------
    list[pd.DataFrame] or dict[str, pd.DataFrame]
        Parsed DataFrames in the same order as base_folders, or a mapping when return_dict=True.
    """
    dfs: list[pd.DataFrame] = []
    mapping: dict[str, pd.DataFrame] = {}

    for folder in base_folders:
        # Normalize and resolve TxtInOut path
        folder = os.path.abspath(str(folder))
        txt = Path(folder)
        if "out_" not in txt.name.lower():
            raise FileNotFoundError(f"Expected a folder with 'out_' in its name but folder name is: {folder}")

        # Parent folder name (one above TxtInOut)
        if name_from == "parent":
            base_name = os.path.basename(os.path.dirname(str(txt)))
        else:
            base_name = os.path.basename(str(txt))
        parent_name = f"{name_prefix}{base_name}"

        # Expected files
        output_rch_path = os.path.join(str(txt), "output.rch")
        cio_path = os.path.join(str(txt), "file.cio")

        # Safety check
        if not os.path.isfile(output_rch_path):
            raise FileNotFoundError(f"output.rch not found in {txt}")
        if not os.path.isfile(cio_path):
            raise FileNotFoundError(f"file.cio not found in {txt}")

        # Load using the previous function
        df = load_output_rch(file_path=output_rch_path, cio_file=cio_path, **kwargs)

        if return_dict:
            mapping[parent_name] = df
        else:
            dfs.append(df)

    return mapping if return_dict else dfs

