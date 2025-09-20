from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

from ..provenance import RealizationProvenance
from ..utils import get_logger


def read_population_by_subbasin_csv_to_df(
    csv_path: Path | str,
    *,
    id_col: str = "GRIDCODE",
    sep: str = ";",
    decimal_comma_for_last_n: int = 0,
    rp: RealizationProvenance | None = None,
) -> pd.DataFrame:
    """Load a CSV of subbasin population by year.

    - Optionally converts last N columns from '1,23' to float 1.23
    - Ensures id_col is int when possible
    - Returns a DataFrame with id_col and year columns (int names if possible)
    """
    log = get_logger(__name__)
    df = pd.read_csv(csv_path, sep=sep)
    if decimal_comma_for_last_n > 0:
        for col in df.columns[-decimal_comma_for_last_n:]:
            df[col] = df[col].astype(str).str.replace(",", ".", regex=False).astype(float)
    # id to int if '123,0'
    try:
        df[id_col] = df[id_col].apply(lambda x: int(str(x).split(",")[0]))
    except Exception:
        pass
    # rename year-like columns to ints if purely digits
    new_cols = {}
    for c in df.columns:
        if isinstance(c, str) and c.isdigit():
            new_cols[c] = int(c)
    if new_cols:
        df = df.rename(columns=new_cols)
    if rp:
        rp.record_input(Path(csv_path), compute_hash=False, kind="csv")
    log.info("Loaded subbasin CSV: %s | rows=%s", csv_path, len(df))
    return df

# Add after your imports (needs re and pandas)
import re
import pandas as pd

def normalize_point_year_columns(df: pd.DataFrame, *, id_col: str = "GRIDCODE", debug: bool = True) -> pd.DataFrame:
    """
    Normalize population CSV to have integer year columns (e.g., 1970, 1981, ...).
    - Keeps only population-sum-like columns (skips pixel counters etc.).
    - Renames trailing 4-digit year suffixes to int.
    - Deduplicates year columns and coerces values to numeric.
    """
    # Patterns for column names that end with a 4-digit year
    patterns = [
        r"(?:popul(?:ation)?_)?sum_(\d{4})$",  # popul_sum_1970 / population_sum_1970 / sum_1970
        r".*_(\d{4})$",                        # any_suffix_1970 (fallback)
        r"(\d{4})$",                           # bare 1970 (fallback)
    ]

    def match_year(col: str) -> int | None:
        s = str(col)
        for pat in patterns:
            m = re.search(pat, s, flags=re.IGNORECASE)
            if m:
                return int(m.group(1))
        return None

    # Pick columns that match the population pattern (skip pixel counters)
    keep_cols: list[str] = []
    year_map: dict[str, int] = {}
    for c in df.columns:
        #print(c)
        y = match_year(c)
        if y is not None:
            if "pixel" in str(c).lower():
                continue
            keep_cols.append(c)
            year_map[c] = y

    # Fallback: if none matched, take any 4-digit numeric-looking columns
    if not keep_cols:
        keep_cols = [c for c in df.columns if str(c).isdigit() and len(str(c)) == 4]
        year_map = {c: int(c) for c in keep_cols}

    cols = [id_col] + keep_cols if id_col in df.columns else keep_cols
    out = df[cols].rename(columns=year_map)

    # Deduplicate year columns by keeping the first occurrence
    out = out.loc[:, ~out.columns.duplicated()]

    # Coerce id to int where possible
    if id_col in out.columns:
        try:
            out[id_col] = out[id_col].apply(lambda x: int(str(x).split(",")[0]))
        except Exception:
            pass

    # Ensure year columns are numeric values
    for c in out.columns:
        if isinstance(c, int):
            out[c] = pd.to_numeric(out[c], errors="coerce")
            #print(f"Coerced column '{c}' to numeric.")

    # Optional debug
    yrs = [c for c in out.columns if isinstance(c, int)]
    _debug = debug if debug is not None else globals().get("DEBUG", False)
    if _debug:
        print(f"[point] normalized years found: {yrs[:10]}{'...' if len(yrs) > 10 else ''}")
    return out


def transform_interpolate_years_wide(
    data: pd.DataFrame,
    dest_dir: Path,
    rng: object,
    params: Dict,
    rp: RealizationProvenance,
    debug: bool = True,
) -> tuple[pd.DataFrame, list[Path]]:
    """Interpolate wide per-subbasin population columns across a full year range.

    Input: wide DataFrame with id_col and some integer year columns (e.g., decades).
    Output: wide DataFrame with id_col and all integer years from year_start..year_end
            linearly interpolated between available columns.

    params:
      id_col: str
      year_start: int
      year_end: int
      keep_existing: bool = True (keep original values at known years)
    """


    log = get_logger(__name__)
    df = data.copy()
    df = normalize_point_year_columns(df, id_col="GRIDCODE", debug=debug)
    # Compute and print source-wide totals across all subbasins for each available year
    source_year_cols = sorted(c for c in df.columns if isinstance(c, int))
    population_totals_source: dict[int, float] = {}
    if source_year_cols:
        try:
            totals_series = df[source_year_cols].sum(numeric_only=True)
            # Ensure keys are int, values float
            population_totals_source = {int(k): float(v) for k, v in totals_series.to_dict().items()}
            # Print a concise report
            preview_keys = sorted(population_totals_source.keys())
            head = preview_keys[:5]
            tail = preview_keys[-5:] if len(preview_keys) > 5 else []
            print("[point] Total inhabitants across all subbasins (source years):")
            for y in head:
                print(f"  year {y}: {population_totals_source[y]:,.0f}")
            if tail and tail[0] not in head:
                if len(preview_keys) > 10:
                    print("  …")
                for y in tail:
                    if y not in head:
                        print(f"  year {y}: {population_totals_source[y]:,.0f}")
        except Exception:
            population_totals_source = {}
    #print(df.columns)
    id_col = params.get("id_col", "GRIDCODE")
    y0 = int(params["year_start"]) if "year_start" in params else None
    y1 = int(params["year_end"]) if "year_end" in params else None
    keep_existing = bool(params.get("keep_existing", True))
    if y0 is None or y1 is None:
        raise ValueError("transform_interpolate_years_wide requires 'year_start' and 'year_end'")

    year_cols = sorted(c for c in df.columns if isinstance(c, int))
    if not year_cols:
        raise ValueError("No integer year columns found to interpolate")

    full_years = list(range(y0, y1 + 1))
    out_records = []
    for _, row in df.iterrows():
        # known points
        xs = [int(y) for y in year_cols]
        ys = [float(row[y]) for y in year_cols]

        # linear interpolation across full range
        import numpy as _np
        interp = _np.interp(full_years, xs, ys)

        rec = {id_col: int(row[id_col])}
        for y, v in zip(full_years, interp):
            rec[int(y)] = float(v)
        # if keeping originals, overwrite with exact values
        if keep_existing:
            for y, v in zip(xs, ys):
                rec[int(y)] = float(v)
        out_records.append(rec)

    wide = pd.DataFrame.from_records(out_records)
    wide = wide[[id_col] + full_years]
    # Compute totals for interpolated full-year range
    population_totals_interpolated: dict[int, float] = {}
    try:
        totals_interp = wide[full_years].sum(numeric_only=True)
        population_totals_interpolated = {int(k): float(v) for k, v in totals_interp.to_dict().items()}
        # Print concise report for interpolated totals (first/last 5 years)
        preview_keys = sorted(population_totals_interpolated.keys())
        head = preview_keys[:5]
        tail = preview_keys[-5:] if len(preview_keys) > 5 else []
        print("[point] Total inhabitants across all subbasins (interpolated years):")
        for y in head:
            print(f"  year {y}: {population_totals_interpolated[y]:,.0f}")
        if tail and tail[0] not in head:
            if len(preview_keys) > 10:
                print("  …")
            for y in tail:
                if y not in head:
                    print(f"  year {y}: {population_totals_interpolated[y]:,.0f}")
    except Exception:
        population_totals_interpolated = {}
    # Stash totals in DataFrame attrs without altering function signature
    try:
        wide.attrs["population_totals"] = {
            "source": population_totals_source,
            "interpolated": population_totals_interpolated,
        }
    except Exception:
        pass

    if rp:
        with rp.step(
            name="transform_interpolate_years_wide",
            module=__name__,
            args={
                "id_col": id_col,
                "year_start": y0,
                "year_end": y1,
                "keep_existing": keep_existing,
                # Include totals in provenance for reporting
                "population_totals_source": population_totals_source,
                "population_totals_interpolated": population_totals_interpolated,
            },
        ):
            pass
    log.info("Interpolated population to yearly wide table: %s..%s", y0, y1)
    return wide, []


def transform_build_point_load_timeseries(
    data: pd.DataFrame,
    dest_dir: Path,
    rng: object,
    params: Dict,
    rp: RealizationProvenance | None = None,
):
    """Compute SWAT point-source timeseries per subbasin from population by year.

    params:
      id_col: str = 'GRIDCODE'
      years: Optional[list[int]] (auto-detect numeric columns if None)
      wastewater_lppd: float = 150.0
      mgL_values: Dict[str, Dict] mapping pollutant var -> { mean, lower, upper, mgL (override), source }
      out_columns: Optional[list[str]]: SWAT variable order; default standard order
      round_to: int = 6
    Output: returns a long DataFrame with columns: [id_col, YEAR, FLOYR, pollutants...]
    """
    log = get_logger(__name__)
    df = data.copy()
    id_col = params.get("id_col", "GRIDCODE")
    # Accept scalar or dict for wastewater_lppd with bounds
    def _resolve_lppd(spec) -> tuple[float, Dict[str, float | str | None]]:
        meta = {"value": None, "standard": None, "lower": None, "upper": None, "source": None}
        try:
            if isinstance(spec, dict):
                meta["standard"] = float(spec.get("standard", spec.get("mean"))) if spec.get("standard", spec.get("mean")) is not None else None
                meta["lower"] = float(spec.get("lower")) if spec.get("lower") is not None else None
                meta["upper"] = float(spec.get("upper")) if spec.get("upper") is not None else None
                if spec.get("value") is not None:
                    meta["value"] = float(spec.get("value"))
                    meta["source"] = str(spec.get("source", "value"))
                else:
                    # Default to standard/mean if provided
                    if meta["standard"] is not None:
                        meta["value"] = float(meta["standard"])
                        meta["source"] = "standard"
                    else:
                        # Fallback to any numeric-like key
                        for k in ("lppd", "val", "v"):
                            if k in spec and spec[k] is not None:
                                meta["value"] = float(spec[k])
                                meta["source"] = "given"
                                break
                if meta["value"] is None:
                    meta["value"] = 150.0
                    meta["source"] = meta["source"] or "default"
                return float(meta["value"]), meta
            else:
                v = 150.0 if spec is None else float(spec)
                meta.update({"value": v, "source": "fixed"})
                return v, meta
        except Exception:
            return 150.0, {"value": 150.0, "standard": None, "lower": None, "upper": None, "source": "default"}

    wastewater_lppd_value, wastewater_lppd_meta = _resolve_lppd(params.get("wastewater_lppd", 150.0))
    mgL_values: Dict[str, Dict] = params.get("mgL_values", {})
    out_columns = params.get(
        "out_columns",
        [
            "YEAR",
            "FLOYR",
            "SEDYR",
            "ORGNYR",
            "ORGPYR",
            "NO3YR",
            "NH3YR",
            "NO2YR",
            "MINPYR",
            "CBODYR",
            "DISOXYR",
            "CHLAYR",
            "SOLPSTYR",
            "SRBPSTYR",
            "BACTPYR",
            "BACTLPYR",
            "CMTL1YR",
            "CMTL2YR",
            "CMTL3YR",
        ],
    )
    round_to = int(params.get("round_to", 6))

    # Detect years
    years = params.get("years")
    if not years:
        years = [c for c in df.columns if isinstance(c, int)]
        years = sorted(y for y in years if 1800 <= y <= 2200)
    if not years:
        raise ValueError("No year columns found in input data. Provide years in params or rename columns to ints.")

    # Compute and print totals (inhabitants) across all subbasins per year from the provided data
    population_totals_years: dict[int, float] = {}
    try:
        totals_series = df[years].sum(numeric_only=True)
        population_totals_years = {int(k): float(v) for k, v in totals_series.to_dict().items()}
        # Print concise report
        preview_keys = sorted(population_totals_years.keys())
        head = preview_keys[:5]
        tail = preview_keys[-5:] if len(preview_keys) > 5 else []
        print("[point] Total inhabitants across all subbasins (from timeseries input):")
        for y in head:
            print(f"  year {y}: {population_totals_years[y]:,.0f}")
        if tail and tail[0] not in head:
            if len(preview_keys) > 10:
                print("  …")
            for y in tail:
                if y not in head:
                    print(f"  year {y}: {population_totals_years[y]:,.0f}")
    except Exception:
        population_totals_years = {}

    # Build output rows
    records = []
    used_mgL = {}
    for var, meta in mgL_values.items():
        # Accept 'standard' as a fallback for convenience
        mg = meta.get("mgL", meta.get("mean", meta.get("standard")))
        used_mgL[var] = {
            "mgL": float(mg) if mg is not None else None,
            "mean": meta.get("mean"),
            "lower": meta.get("lower"),
            "upper": meta.get("upper"),
            "source": meta.get("source", "n/a"),
        }

    for _, row in df.iterrows():
        sid = row[id_col]
        for y in years:
            pop = float(row[y])
            fl_yr = round(pop * wastewater_lppd_value / 1000.0, round_to)  # m3/day
            rec = {id_col: int(sid), "YEAR": int(y), "FLOYR": fl_yr}
            # pollutants to kg/day
            for var, meta in used_mgL.items():
                mg = meta.get("mgL")
                if mg is None:
                    val = 0.0
                else:
                    val = mg * wastewater_lppd_value * pop / 1_000_000.0
                rec[var] = round(float(val), round_to)
            records.append(rec)

    long_df = pd.DataFrame.from_records(records)

    # ensure columns order
    cols = [id_col] + [c for c in out_columns if c in long_df.columns]
    long_df = long_df[cols]

    if rp:
        with rp.step(
            name="point_mgL_choices",
            module=__name__,
            args={
                "wastewater_lppd": wastewater_lppd_value,
                "wastewater_lppd_meta": wastewater_lppd_meta,
                "mgL_values": used_mgL,
                "id_col": id_col,
                "years": years,
                # Add population totals to provenance
                "population_totals_years": population_totals_years,
            },
        ):
            pass
    log.info("Computed point-source timeseries for %s subbasins across %s years", df[id_col].nunique(), len(years))
    return long_df, []


def transform_write_point_dat_from_df(
    data: pd.DataFrame,
    dest_dir: Path,
    rng: object,
    params: Dict,
    copy_ready_fig_fig,
    rp: RealizationProvenance | None = None,
):
    """Write rcyr_XX.dat files per subbasin from a long dataframe.

    params:
      id_col: str
      columns_order: list[str] (YEAR first is enforced)
      start_year: Optional[int]
      end_year: Optional[int]
      areas_df: Optional[pd.DataFrame] with columns [id_col, 'Area'] in ha
    """

    # pause this for
    copy_ready_fig_fig = True
    if copy_ready_fig_fig:
        # Copy ready made fig.fig to output directory if fig.fig exists at scr_fig, overwriting any existing one
        # later we should generate fig.fig properly like here: "C:\Users\Usuario\OneDrive - UNIVERSIDAD DE HUELVA\Granada\TrabajoFM\scripts\script POINT loads - input .dat\swat_ready_recyear_files\.fig files\visualize_fig_configs.ipynb"
        import shutil
        src_fig = "C:\\Users\\Usuario\\OneDrive - UNIVERSIDAD DE HUELVA\\Granada\\TrabajoFM\\scripts\\script POINT loads - input .dat\\swat_ready_recyear_files\\.fig files\\modified .fig\\fig.fig"
        dest_fig = dest_dir / "fig.fig"
        if Path(src_fig).exists():
            shutil.copy2(src_fig, dest_fig)
            print(f"WARNING: For simplicity, copied ready-made fig.fig to realization directory from this path: {src_fig}")
        else:
            print(f"ERROR: Source fig.fig not found at {src_fig}, skipping copy.")
        

    log = get_logger(__name__)
    df = data.copy()
    id_col = params.get("id_col", "GRIDCODE")
    columns_order = params.get("columns_order") or [
        "YEAR",
        "FLOYR",
        "SEDYR",
        "ORGNYR",
        "ORGPYR",
        "NO3YR",
        "NH3YR",
        "NO2YR",
        "MINPYR",
        "CBODYR",
        "DISOXYR",
        "CHLAYR",
        "SOLPSTYR",
        "SRBPSTYR",
        "BACTPYR",
        "BACTLPYR",
        "CMTL1YR",
        "CMTL2YR",
        "CMTL3YR",
    ]
    start_year = params.get("start_year")
    end_year = params.get("end_year")
    areas_df = params.get("areas_df")

    def format_float_16(value: float) -> str:
        s = f"{value:.10E}"
        if len(s) > 16:
            for p in range(9, -1, -1):
                s = f"{value:.{p}E}"
                if len(s) <= 16:
                    break
        else:
            s = s.rjust(16)
        return s

    written: list[Path] = []
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    by_id = df.groupby(id_col)
    for sid, grp in by_id:
        grp_sorted = grp.sort_values("YEAR")
        if start_year is not None:
            grp_sorted = grp_sorted[grp_sorted["YEAR"] >= int(start_year)]
        if end_year is not None:
            grp_sorted = grp_sorted[grp_sorted["YEAR"] <= int(end_year)]

        present_cols = [c for c in columns_order if c in grp_sorted.columns]
        # ensure YEAR first
        if present_cols[0] != "YEAR":
            present_cols = ["YEAR"] + [c for c in present_cols if c != "YEAR"]

        filename = f"rcyr_{int(sid)}.dat"
        file_path = dest_dir / filename

        drainage_area = None
        if isinstance(areas_df, pd.DataFrame) and id_col in areas_df.columns and "Area" in areas_df.columns:
            row = areas_df[areas_df[id_col] == sid]
            if not row.empty:
                area_ha = float(row.iloc[0]["Area"])
                drainage_area = area_ha / 100.0

        with file_path.open("w", encoding="utf-8") as f:
            area_str = f"{drainage_area:.3f}" if drainage_area is not None else "0.000"
            y0 = int(grp_sorted["YEAR"].min()) if not grp_sorted.empty else (start_year or 0)
            y1 = int(grp_sorted["YEAR"].max()) if not grp_sorted.empty else (end_year or 0)
            f.write(f" TITLE LINE 1 - Subbasin ID {int(sid)} | Simulation Years: {y0}-{y1} | DRAINAGE_AREA (km²): {area_str}\n")
            f.write(" TITLE LINE 2 - Source: TrabajoFM model\n")
            f.write(" TITLE LINE 3 - Units: kg/day\n")
            f.write(f" TITLE LINE 4 - Period: {y0}-{y1}\n")
            f.write(" TITLE LINE 5 - \n")

            header_line = f"{'YEAR':>5}"
            for col in present_cols:
                if col != "YEAR":
                    header_line += f"{col:>17}"
            f.write(header_line + "\n")

            for _, row in grp_sorted[present_cols].iterrows():
                line = f"{int(row['YEAR']):>5}"
                for v in row.iloc[1:]:
                    line += " " + format_float_16(float(v))
                f.write(line + "\n")

        written.append(file_path)

    if rp and written:
        rp.record_outputs(written, kind="dat")
        with rp.step(
            name="transform_write_point_dat_from_df",
            module=__name__,
            args={
                "id_col": id_col,
                "columns_order": columns_order,
                "start_year": start_year,
                "end_year": end_year,
                "count": len(written),
            },
        ):
            pass
    log.info("Wrote %s rcyr_*.dat files to %s", len(written), dest_dir)
    return df, written
