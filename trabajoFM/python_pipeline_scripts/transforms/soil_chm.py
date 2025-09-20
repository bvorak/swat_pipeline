from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple, Union, Literal

import numpy as np
import pandas as pd

from ..provenance import RealizationProvenance
from ..utils import get_logger
from ..writers.chm_writer import apply_replacements_bulk
import numpy as np

try:
    import rasterio
    from rasterio.features import rasterize
except ImportError:
    rasterio = None

try:
    import geopandas as gpd
except ImportError:
    gpd = None


def read_n_p_means_from_csv_to_df(
    csv_path: Path | str,
    *,
    id_col: str = "HRU_GIS",
    n_col: str = "mean_Nitrogeno_total_porcent_resample_Rediam",
    p_col: str = "mean_Fosforo_mg_100g_P205_rediam",
    sep: str = ";",
    decimal_comma_for_last_n: int = 3,
    rp: RealizationProvenance | None = None,
) -> pd.DataFrame:
    """Load a CSV of HRU stats and coerce types similar to previous notebook code.

    - Converts the last `decimal_comma_for_last_n` columns from '1,23' to float 1.23
    - Ensures id_col is int
    - Returns a DataFrame with at least [id_col, n_col, p_col]
    """
    log = get_logger(__name__)
    df = pd.read_csv(csv_path, sep=sep)
    if decimal_comma_for_last_n > 0:
        for col in df.columns[-decimal_comma_for_last_n:]:
            df[col] = df[col].astype(str).str.replace(",", ".", regex=False).astype(float)
    # Make id int even if '123,0'
    df[id_col] = df[id_col].apply(lambda x: int(str(x).split(",")[0]))
    # Basic sanity
    for col in (id_col, n_col, p_col):
        if col not in df.columns:
            raise KeyError(f"Column not found in CSV: {col}")
    if rp:
        rp.record_input(Path(csv_path), compute_hash=False, kind="csv")
    log.info("Loaded HRU CSV: %s | rows=%s", csv_path, len(df))
    return df[[c for c in df.columns]]


# New modular transforms for sequential pipelines

def transform_compute_base_soil_vars(
    data: pd.DataFrame,
    dest_dir: Path,
    rng: np.random.Generator,
    params: Dict,
    rp: RealizationProvenance | None = None,
) -> tuple[pd.DataFrame, list[Path]]:
    """Compute base N_total_mg_kg and P_element_mg_kg from source columns using generic scaling ops.

    params:
      id_col, n_col, p_col
    Notes:
      This is a convenience wrapper that applies two deterministic scale transforms
      (no bounds) so the provenance clearly records these conversions.
    """
    log = get_logger(__name__)
    n_col = params.get("n_col", "mean_Nitrogeno_total_porcent_resample_Rediam")
    p_col = params.get("p_col", "mean_Fosforo_mg_100g_P205_rediam")
    input_source = params.get("input_source")

    df = data.copy()

    # Apply generic scaling for N (percent -> mg/kg)
    scale_n_params = {
        "src": n_col,
        "out": "N_total_mg_kg",
        "mode": "relative",  # multiply
        "factor": 10_000.0,
        "mean": 1.0,
        "lower": None,
        "upper": None,
        "source": "deterministic",
        "input_source": input_source,
        "debug": params.get("debug", False),
    }
    print("reset n factor to: " + str(scale_n_params["factor"]))
    df, _ = transform_scale_variable(df, dest_dir, rng, scale_n_params, rp)

    # Apply generic scaling for P (mg/100g P2O5 -> mg/kg elemental P)
    scale_p_params = {
        "src": p_col,
        "out": "P_element_mg_kg",
        "mode": "relative",  # multiply
        "factor": 10.0 * 0.4364,
        "mean": 1.0,
        "lower": None,
        "upper": None,
        "source": "deterministic",
        "input_source": input_source,
        "debug": params.get("debug", False),
    }
    print("reset p factor to: " + str(scale_p_params["factor"]))
    df, _ = transform_scale_variable(df, dest_dir, rng, scale_p_params, rp)

    log.info("Computed base soil vars via generic scaling: N_total_mg_kg, P_element_mg_kg")
    return df, []



def transform_write_chm_from_df(
    data: pd.DataFrame,
    dest_dir: Path,
    rng: np.random.Generator,
    params: Dict,
    rp: RealizationProvenance,
) -> tuple[pd.DataFrame, list[Path]]:
    """Write CHM files from dataframe columns using column-to-label mapping.

    params:
      id_col: str
      label_map: { 'Soil NO3 [mg/kg]': 'Soil NO3 [mg/kg]', ... } # df_col -> CHM label (often same)
      pperco_val: optional float
    """
    log = get_logger(__name__)
    df = data
    id_col = params.get("id_col", "HRU_GIS")
    label_map: Dict[str, str] = params.get("label_map", {
        "Soil NO3 [mg/kg]": "Soil NO3 [mg/kg]",
        "Soil organic N [mg/kg]": "Soil organic N [mg/kg]",
        "Soil labile P [mg/kg]": "Soil labile P [mg/kg]",
        "Soil organic P [mg/kg]": "Soil organic P [mg/kg]",
    })
    pperco_val = params.get("pperco_val", None)

    mapping: Dict[int, Dict[str, float]] = {}
    for _, row in df.iterrows():
        hru_id = int(row[id_col])
        repl: Dict[str, float] = {}
        for df_col, chm_label in label_map.items():
            repl[chm_label] = float(row[df_col])
        mapping[hru_id] = repl

    src_txtinout = Path(rp.base_txtinout)
    written = apply_replacements_bulk(
        base_txtinout=src_txtinout,
        dest_txtinout=Path(dest_dir),
        hru_replacements=mapping,
        pperco_val=pperco_val,
        overwrite=True,
    )
    rp.record_outputs(written, kind="chm")
    log.info("Wrote %s CHM files to %s", len(written), dest_dir)
    return df, written




def transform_perturb_relative(
    data: pd.DataFrame,
    dest_dir: Path,
    rng: np.random.Generator,
    params: Dict,
    rp: RealizationProvenance | None = None,
) -> tuple[pd.DataFrame, list[Path]]:
    """Apply relative/absolute perturbations to specified columns.

    params:
      targets: [{ name: 'N_total_mg_kg', mode: 'relative', bound: 0.2, delta: optional }, ...]
    """
    log = get_logger(__name__)
    df = data.copy()
    targets = params.get("targets", [])
    applied: list[Dict[str, float]] = []
    for t in targets:
        col = t["name"]
        mode = t.get("mode", "relative")
        bound = float(t.get("bound", 0.0))
        delta = t.get("delta")
        source = "override"
        if delta is None and bound > 0:
            delta = float(rng.uniform(-bound, +bound))
            source = "random"
        elif delta is None:
            delta = 0.0
            source = "none"
        if mode == "relative":
            df[col] = df[col].astype(float) * (1.0 + float(delta))
        else:
            df[col] = df[col].astype(float) + float(delta)
        log.info("Perturbed %s | mode=%s | delta=%s | bound=%s", col, mode, delta, bound)
        applied.append({"name": col, "mode": mode, "bound": bound, "delta": float(delta), "source": source})
    # Record the actual applied deltas as their own step for clarity
    if rp and applied:
        with rp.step(
            name="perturbation_choices",
            module=__name__,
            args={"targets": applied},
        ):
            pass
    return df, []



def transform_scale_variable(
    data: pd.DataFrame,
    dest_dir: Path,
    rng: np.random.Generator,
    params: Dict,
    rp: RealizationProvenance | None = None,
) -> tuple[pd.DataFrame, list[Path]]:
    """Create an output variable by scaling a source column with a provided factor.

    Simplified bounds-only approach:
      - params defines mean/lower/upper for logging (no random here)
      - per_realization_params should provide 'factor' (e.g., 0.8 or 1.2)
    params:
      src, out, mode ('relative'|'absolute'), mean, lower, upper, factor (override), input_source, debug
    """
    log = get_logger(__name__)
    df = data.copy()
    src = params.get("src")
    out = params.get("out")
    mode = params.get("mode", "relative")
    mean = float(params.get("mean", 1.0))
    lower = params.get("lower")
    upper = params.get("upper")
    factor = params.get("factor", mean)
    input_source = params.get("input_source")
    source = params.get("source", "n/a")

    if mode == "relative":
        df[out] = df[src].astype(float) * float(factor)
    else:
        df[out] = df[src].astype(float) + float(factor)

    if params.get("debug"):
        log.info("[debug] scale %s -> %s | mode=%s | factor=%.6f | mean=%.6f lower=%s upper=%s", src, out, mode, factor, mean, lower, upper)
    if rp:
        with rp.step(
            name="scale_choices",
            module=__name__,
            args={
                "src": src,
                "out": out,
                "mode": mode,
                "mean": mean,
                "lower": lower,
                "upper": upper,
                "factor": float(factor),
                "source": source,
                "input_source": input_source,
            },
        ):
            pass
    return df, []


def transform_apply_ops(
    data: pd.DataFrame,
    dest_dir: Path,
    rng: np.random.Generator,
    params: Dict,
    rp: RealizationProvenance | None = None,
) -> tuple[pd.DataFrame, list[Path]]:
    """Apply one or more generic operations on columns with optional bounds metadata.

    Supports simple per-column operations with transparent provenance:
      - op: 'mul' (multiply), 'add' (add), 'set' (constant or from another column)
      - Provide 'factor' for 'mul', 'delta' for 'add', 'value' for 'set'
      - For logging/reporting, you may specify mean/lower/upper bounds (no sampling here)

    params:
      ops: [
        { src:'A', out:'B', op:'mul', factor:2.0, mean:1.0, lower:None, upper:None, source:'deterministic' },
        { src:'B', out:'C', op:'add', delta:1.5, mean:0.0, lower:-2.0, upper:2.0, source:'override' },
        { out:'D', op:'set', value:42 }
      ]
      input_source: optional string to annotate provenance
      debug: bool
    """
    log = get_logger(__name__)
    df = data.copy()
    ops = params.get("ops", []) or []
    input_source = params.get("input_source")
    applied = []

    for spec in ops:
        op = spec.get("op") or spec.get("mode")  # allow 'mode' synonyms
        src = spec.get("src")
        out = spec.get("out") or src
        mean = spec.get("mean")
        lower = spec.get("lower")
        upper = spec.get("upper")
        source = spec.get("source", "n/a")

        if op in ("mul", "relative"):
            # Do NOT default to 1.0 silently; prefer explicit factor/mean/standard, else skip
            factor = spec.get("factor", None)
            if factor is None:
                if mean is not None:
                    factor = float(mean)
                elif spec.get("standard") is not None:
                    factor = float(spec.get("standard"))
            if factor is None:
                if params.get("debug"):
                    log.warning("[debug] skipping mul op due to missing factor: src=%s out=%s", src, out)
                continue
            factor = float(factor)
            df[out] = df[src].astype(float) * factor
            val = factor
            opname = "mul"
        elif op in ("add", "absolute"):
            # Do NOT default to 0.0 silently; prefer explicit delta/mean/standard, else skip
            delta = spec.get("delta", None)
            if delta is None:
                if mean is not None:
                    delta = float(mean)
                elif spec.get("standard") is not None:
                    delta = float(spec.get("standard"))
            if delta is None:
                if params.get("debug"):
                    log.warning("[debug] skipping add op due to missing delta: src=%s out=%s", src, out)
                continue
            delta = float(delta)
            df[out] = df[src].astype(float) + delta
            val = delta
            opname = "add"
        elif op == "set":
            # Prefer explicit 'value'; if not present but 'mean'/'standard' provided, use those; else require src
            if "value" in spec and spec.get("value") is not None:
                df[out] = float(spec.get("value"))
                val = float(spec.get("value"))
            elif mean is not None:
                df[out] = float(mean)
                val = float(mean)
            elif spec.get("standard") is not None:
                df[out] = float(spec.get("standard"))
                val = float(spec.get("standard"))
            elif src is not None:
                df[out] = df[src]
                val = "from_src"
            else:
                if params.get("debug"):
                    log.warning("[debug] skipping set op due to missing value and src: out=%s", out)
                continue
            opname = "set"
        else:
            raise ValueError(f"Unsupported op: {op}")

        if params.get("debug"):
            log.info("[debug] op %s: %s -> %s | value=%s | mean=%s lower=%s upper=%s", opname, src, out, val, mean, lower, upper)
        applied.append({
            "src": src,
            "out": out,
            "op": opname,
            "value": val,
            "mean": mean,
            "lower": lower,
            "upper": upper,
            "source": source,
            "input_source": input_source,
        })

    if rp and applied:
        with rp.step(
            name="ops_choices",
            module=__name__,
            args={"ops": applied},
        ):
            pass

    return df, []





def transform_split_with_bounds(
    data: pd.DataFrame,
    dest_dir: Path,
    rng: np.random.Generator,
    params: Dict,
    rp: RealizationProvenance | None = None,
) -> tuple[pd.DataFrame, list[Path]]:
    """Split a source column into outputs using provided ratios (already chosen within bounds).

    Simplified bounds-only approach: per_realization_params should include
      outputs: [{name, ratio}], and transforms_params carries mean/lower/upper for logging.
    params: src, outputs(list), renormalize, input_source, debug
    """
    log = get_logger(__name__)
    df = data.copy()
    src = params.get("src")
    outs = params.get("outputs", [])  # each item may include name, ratio (chosen), mean/lower/upper for logging
    renorm = bool(params.get("renormalize", True))
    input_source = params.get("input_source")

    # choose ratios from overrides; if not present, fall back to 'mean'
    chosen = []
    for o in outs:
        ratio = o.get("ratio")
        if ratio is None:
            ratio = o.get("mean")
        chosen.append(float(ratio))

    final = chosen
    if renorm and sum(final) != 0:
        total = sum(final)
        final = [r / total for r in final]

    # write outputs
    for o, frac in zip(outs, final):
        name = o.get("name")
        df[name] = df[src].astype(float) * float(frac)
        if params.get("debug"):
            log.info("[debug] split %s -> %s | ratio=%.6f (mean=%s lower=%s upper=%s) renorm=%s", src, name, frac, o.get("mean"), o.get("lower"), o.get("upper"), renorm)

    if rp:
        # record detailed choices (mean/lower/upper and used ratio)
        details = []
        for o, frac in zip(outs, final):
            details.append({
                "name": o.get("name"),
                "mean": o.get("mean"),
                "lower": o.get("lower"),
                "upper": o.get("upper"),
                "ratio": frac,
                "source": o.get("source", "n/a"),
                "input_source": input_source,
            })
        with rp.step(
            name="split_choices",
            module=__name__,
            args={"src": src, "renormalize": renorm, "outputs": details},
        ):
            pass

    return df, []



########################## Soil CHM GLOBAL uncertainty ##########################

ArrayLike = np.ndarray
PathOrArray = Union[str, ArrayLike]

# ---------------------------
# Low-level helpers
# ---------------------------

def _load_raster(x: PathOrArray) -> ArrayLike:
    """Load a single-band raster or accept a NumPy array."""
    if isinstance(x, np.ndarray):
        return x.astype(float)
    if rasterio is None:
        raise ImportError("rasterio not installed; pass arrays instead of file paths.")
    with rasterio.open(x) as ds:
        arr = ds.read(1).astype(float)
        nodata = ds.nodatavals[0]
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        transform = ds.transform
        crs = ds.crs
        return arr, transform, crs

def _ensure_raster_tuple(x: PathOrArray):
    """Return (arr, transform, crs). If array supplied, transform/crs are None."""
    if isinstance(x, np.ndarray):
        return x.astype(float), None, None
    arr, transform, crs = _load_raster(x)
    # when _load_raster gets filepath, it returns tuple; here we normalize
    if isinstance(arr, tuple):
        # Already normalized by caller
        return arr
    else:
        # _load_raster returned only array when given array path? (compat)
        return arr, transform, crs

def _load_raster_any(x: PathOrArray):
    """Return arr; if path, returns arr and keeps transform/crs elsewhere."""
    if isinstance(x, np.ndarray):
        return x.astype(float)
    if rasterio is None:
        raise ImportError("rasterio not installed; pass arrays instead of file paths.")
    with rasterio.open(x) as ds:
        arr = ds.read(1).astype(float)
        nodata = ds.nodatavals[0]
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        return arr

def _center_from_mode(pred: ArrayLike, q50: Optional[ArrayLike], center_mode: Literal["prediction","median"]) -> ArrayLike:
    if center_mode == "median" and q50 is not None:
        return q50
    return pred

def _valid_mask(*arrays: Optional[ArrayLike]) -> ArrayLike:
    m = None
    for a in arrays:
        if a is None:
            continue
        cur = np.isfinite(a)
        m = cur if m is None else (m & cur)
    if m is None:
        raise ValueError("No arrays provided for masking.")
    return m

def _safe_rel(num: ArrayLike, den: ArrayLike, eps: float = 1e-12) -> ArrayLike:
    return num / np.maximum(np.abs(den), eps)

# ---------------------------
# Main function
# ---------------------------

def derive_global_uncertainty(
    prediction: PathOrArray,
    se: PathOrArray,
    q50: Optional[PathOrArray] = None,
    prob: Optional[PathOrArray] = None,  # reserved for future checks
    method: Literal[
        "p95_rel_se",          # global: 95th pct of |SE/center|
        "rmse_rel_center",     # global: sqrt(mean(SE^2))/mean(center)
        "median_se_zscore",    # global: z * median(SE)/median(center)
        "fixed_rel",           # global: user constant
        "hru_support"          # HRU-aware: per-HRU variance scaling (can collapse to one number)
    ] = "p95_rel_se",
    center_mode: Literal["prediction", "median"] = "prediction",
    z: float = 1.96,                  # for median_se_zscore
    fixed_rel: Optional[float] = None,  # for fixed_rel
    # HRU options:
    hru_path: Optional[str] = None,   # path to polygons (any format geopandas can read)
    hru_id_field: Optional[str] = None, # column to use as ID; if None, index is used
    correlation_range_m: Optional[float] = None,  # r (meters) required if method="hru_support"
    collapse: Optional[Literal["median","p90","max"]] = "p90",  # how to collapse per-HRU to one number
    clip_rel_range: Tuple[float, float] = (0.0, 2.0)  # clamp half-width to [0, 200%]
) -> Dict[str, object]:
    """
    Returns:
      {
        'half_width_rel': float,             # single global number (± relative)
        'lower_factor': float,               # 1 - half_width_rel
        'upper_factor': float,               # 1 + half_width_rel
        'per_hru' : pandas.DataFrame or None # optional table with per-HRU half_width_rel if method='hru_support'
      }
    Notes:
      - If method != 'hru_support', 'per_hru' is None.
      - If method == 'hru_support' and collapse is not None, the top-level number
        is the chosen summary (median/p90/max) across HRUs.
    """
    # --- Load rasters ---
    if isinstance(prediction, np.ndarray):
        pred = prediction.astype(float)
        transform = None; crs = None
    else:
        if rasterio is None:
            raise ImportError("rasterio required when passing file paths.")
        with rasterio.open(prediction) as ds:
            pred = ds.read(1).astype(float)
            nodata = ds.nodatavals[0]
            if nodata is not None:
                pred = np.where(pred == nodata, np.nan, pred)
            transform = ds.transform
            crs = ds.crs

    se_r = _load_raster_any(se)
    q50_r = _load_raster_any(q50) if q50 is not None else None
    center = _center_from_mode(pred, q50_r, center_mode=center_mode)

    m = _valid_mask(pred, se_r, center)
    if not np.any(m):
        raise ValueError("No valid pixels after masking NaNs.")

    # ------------- Global methods -------------
    if method in ("p95_rel_se", "rmse_rel_center", "median_se_zscore", "fixed_rel"):
        if method == "p95_rel_se":
            rel = _safe_rel(se_r[m], center[m])
            half_width_rel = np.nanpercentile(np.abs(rel), 95)

        elif method == "rmse_rel_center":
            rmse = np.sqrt(np.nanmean(se_r[m] ** 2))
            center_mean = np.nanmean(np.abs(center[m]))
            half_width_rel = _safe_rel(rmse, center_mean)

        elif method == "median_se_zscore":
            med_se = np.nanmedian(np.abs(se_r[m]))
            med_center = np.nanmedian(np.abs(center[m]))
            half_width_rel = _safe_rel(z * med_se, med_center)

        elif method == "fixed_rel":
            if fixed_rel is None:
                raise ValueError("fixed_rel must be provided when method='fixed_rel'.")
            half_width_rel = float(fixed_rel)

        half_width_rel = float(np.clip(half_width_rel, *clip_rel_range))
        out = {
            "half_width_rel": half_width_rel,
            "lower_factor": max(0.0, 1.0 - half_width_rel),
            "upper_factor": 1.0 + half_width_rel,
            "per_hru": None
        }
        return out

    # ------------- HRU-aware method -------------
    if method == "hru_support":
        if gpd is None:
            raise ImportError("geopandas is required for method='hru_support'.")
        if rasterio is None:
            raise ImportError("rasterio is required for method='hru_support'.")
        if hru_path is None:
            raise ValueError("hru_path is required for method='hru_support'.")
        if correlation_range_m is None:
            raise ValueError("correlation_range_m (r) is required for method='hru_support' (meters, same CRS).")

        # Load polygons
        gdf = gpd.read_file(hru_path)
        if gdf.empty:
            raise ValueError("HRU polygon file is empty.")
        if hru_id_field is None:
            gdf = gdf.reset_index().rename(columns={"index":"HRU_ID"})
            hru_id_field = "HRU_ID"
        if transform is None or crs is None:
            raise ValueError("When using file rasters, we require prediction's transform/crs. "
                             "If you passed arrays, please also pass file paths so CRS/transform are known.")

        # Ensure projected CRS in meters
        if gdf.crs is None:
            warnings.warn("HRU polygons have no CRS; assuming it matches the raster CRS.")
            gdf = gdf.set_crs(crs)
        elif gdf.crs != crs:
            gdf = gdf.to_crs(crs)

        # Build integer ID raster for polygons
        ids = gdf[hru_id_field].values
        shapes = [(geom, int(i)) for geom, i in zip(gdf.geometry, ids)]

        hru_id_raster = rasterize(
            shapes=shapes,
            out_shape=pred.shape,
            transform=transform,
            fill=0,
            dtype="int32",
            all_touched=False
        )
        unique_ids = np.unique(hru_id_raster)
        unique_ids = unique_ids[unique_ids != 0]  # zero is background

        # Pixel area (assumes square-ish pixels; from affine)
        pixel_area = abs(transform.a) * abs(transform.e)  # meters^2

        se2 = se_r**2
        per_hru = []

        for hid in unique_ids:
            mask = (hru_id_raster == hid) & m
            n_pix = int(np.nansum(mask))
            if n_pix == 0:
                continue

            # Mean pixel variance inside HRU
            mean_var = float(np.nanmean(se2[mask]))  # (units^2)

            # HRU area (geometry area)
            area_m2 = float(gdf.loc[gdf[hru_id_field] == hid, "geometry"].area.values[0])

            # Effective sample size using correlation range r
            r = float(correlation_range_m)
            n_eff = max(1.0, min(n_pix, area_m2 / (np.pi * r**2)))

            # HRU SD (areal mean)
            sd_hru = np.sqrt(mean_var / n_eff)

            # HRU center (mean prediction or median) for relative half-width
            center_mean = float(np.nanmean(center[mask]))
            # Guard against near-zero center (choose absolute half-width fallback)
            if abs(center_mean) < 1e-12:
                half_rel = np.nan  # will handle later
                half_abs = sd_hru * 1.96  # 95% approx if desired; or use sd_hru as half-width
            else:
                # Pick a simple, defensible 95% relative half-width using z=1.96
                half_rel = (1.96 * sd_hru) / abs(center_mean)
                half_abs = 1.96 * sd_hru

            per_hru.append({
                "HRU_ID": hid,
                "n_pixels": n_pix,
                "area_m2": area_m2,
                "mean_pixel_var": mean_var,
                "n_eff": n_eff,
                "sd_hru": sd_hru,
                "center_mean": center_mean,
                "half_width_rel": half_rel,
                "half_width_abs": half_abs
            })

        import pandas as pd
        df = pd.DataFrame(per_hru)
        if df.empty:
            raise ValueError("No HRU had valid overlap with the raster.")

        # For any HRU with undefined relative (center~0), fallback:
        if df["half_width_rel"].isna().any():
            # Use absolute half-width divided by global |center| median as a crude fallback
            global_center_med = float(np.nanmedian(np.abs(center[m])))
            df.loc[df["half_width_rel"].isna(), "half_width_rel"] = df.loc[
                df["half_width_rel"].isna(), "half_width_abs"
            ] / max(global_center_med, 1e-12)

        # Clamp and ensure non-negative
        df["half_width_rel"] = df["half_width_rel"].clip(clip_rel_range[0], clip_rel_range[1])

        if collapse is None:
            # No collapse: we still must return a single number per your design—use median
            summary_val = float(df["half_width_rel"].median())
        elif collapse == "median":
            summary_val = float(df["half_width_rel"].median())
        elif collapse == "p90":
            summary_val = float(np.nanpercentile(df["half_width_rel"].values, 90))
        elif collapse == "max":
            summary_val = float(df["half_width_rel"].max())
        else:
            raise ValueError("collapse must be one of None, 'median', 'p90', 'max'.")

        summary_val = float(np.clip(summary_val, *clip_rel_range))
        out = {
            "half_width_rel": summary_val,
            "lower_factor": max(0.0, 1.0 - summary_val),
            "upper_factor": 1.0 + summary_val,
            "per_hru": df
        }
        return out

    raise ValueError(f"Unknown method: {method}")

"""

def transform_split_fixed_ratios(
    data: pd.DataFrame,
    dest_dir: Path,
    rng: np.random.Generator,
    params: Dict,
    rp: RealizationProvenance | None = None,
) -> tuple[pd.DataFrame, list[Path]]:
#    Split source columns into outputs using fixed ratios.
#
#    params:
#      splits: [
#        { src: 'N_total_mg_kg', outputs: [ {name:'Soil NO3 [mg/kg]', ratio:0.02}, {name:'Soil organic N [mg/kg]', ratio:0.98} ] },
#        { src: 'P_element_mg_kg', outputs: [ {name:'Soil labile P [mg/kg]', ratio:1.0}, {name:'Soil organic P [mg/kg]', ratio:0.0} ] }
#      ]

    log = get_logger(__name__)
    df = data.copy()
    for spec in params.get("splits", []):
        src = spec["src"]
        outs = spec.get("outputs", [])
        for o in outs:
            name = o["name"]
            ratio = float(o.get("ratio", 0.0))
            df[name] = df[src].astype(float) * ratio
            log.info("Split %s -> %s ratio=%.4f", src, name, ratio)
    return df, []


def convert_soil_nutrients(
    N_total_percent: float,
    P_mg_100g_P2O5: float,
    *,
    nitrate_frac: float = 0.02,
    organicN_frac: float = 0.98,
    labileP_frac: float = 1.0,
    organicP_frac: float = 0.0,
    round_to: int = 2,
) -> Dict[str, float]:
    #Deterministic conversion from total N% and P2O5 mg/100g to SWAT-ready pools.
#
#    Returns a replacements dict keyed by CHM labels.
    
    N_total_mg_kg = float(N_total_percent) * 10_000.0
    P2O5_mg_kg = float(P_mg_100g_P2O5) * 10.0
    available_P_mg_kg = P2O5_mg_kg * 0.4364

    soil_NO3 = N_total_mg_kg * float(nitrate_frac)
    soil_orgN = N_total_mg_kg * float(organicN_frac)
    soil_labP = available_P_mg_kg * float(labileP_frac)
    soil_orgP = available_P_mg_kg * float(organicP_frac)

    r = round_to
    return {
        "Soil NO3 [mg/kg]": round(soil_NO3, r),
        "Soil organic N [mg/kg]": round(soil_orgN, r),
        "Soil labile P [mg/kg]": round(soil_labP, r),
        "Soil organic P [mg/kg]": round(soil_orgP, r),
    }



def replacements_from_dataframe(
    df: pd.DataFrame,
    *,
    id_col: str,
    n_col: str,
    p_col: str,
    nitrate_frac: float = 0.02,
    organicN_frac: float = 0.98,
    labileP_frac: float = 1.0,
    organicP_frac: float = 0.0,
) -> Dict[int, Dict[str, float]]:
    # Builds a mapping HRU id -> replacements dict from dataframe columns.
    mapping: Dict[int, Dict[str, float]] = {}
    for _, row in df.iterrows():
        hru_id = int(row[id_col])
        repl = convert_soil_nutrients(
            row[n_col],
            row[p_col],
            nitrate_frac=nitrate_frac,
            organicN_frac=organicN_frac,
            labileP_frac=labileP_frac,
            organicP_frac=organicP_frac,
        )
        mapping[hru_id] = repl
    return mapping








def mc_transform_write_chm( ############## I think not in use anymore
    base_data: pd.DataFrame,
    dest_dir: Path,
    rng: np.random.Generator,
    params: Dict,
    rp: RealizationProvenance,
) -> list[Path]:
    #Monte Carlo transform that perturbs N and P inputs within bounds and writes CHM files.
#
#    Expected params keys (all optional with sensible defaults):
#    - id_col, n_col, p_col: dataframe column names
#    - bounds: { 'N_total_pct': 0.10, 'P2O5_mg100g': 0.10 }  # ± fraction
#    - fractions: { 'nitrate_frac': 0.02, 'organicN_frac': 0.98, 'labileP_frac': 1.0, 'organicP_frac': 0.0 }
#    - pperco_val: float or None
#   
    log = get_logger(__name__)
    id_col = params.get("id_col", "HRU_GIS")
    n_col = params.get("n_col", "mean_Nitrogeno_total_porcent_resample_Rediam")
    p_col = params.get("p_col", "mean_Fosforo_mg_100g_P205_rediam")
    bounds = params.get("bounds", {"N_total_pct": 0.0, "P2O5_mg100g": 0.0})
    deltas = params.get("deltas")  # optional deterministic deltas dict with same keys as bounds
    fr = params.get(
        "fractions",
        {
            "nitrate_frac": 0.02,
            "organicN_frac": 0.98,
            "labileP_frac": 1.0,
            "organicP_frac": 0.0,
        },
    )
    pperco_val = params.get("pperco_val", None)

    # Build perturbed dataframe
    df = base_data.copy()
    if bounds.get("N_total_pct", 0.0) > 0:
        alpha = float(bounds["N_total_pct"])  # e.g., 0.10 => ±10%
        if deltas and "N_total_pct" in deltas:
            df[n_col] = df[n_col] * (1.0 + float(deltas["N_total_pct"]))
        else:
            noise = rng.uniform(-alpha, +alpha, size=len(df))
            df[n_col] = df[n_col] * (1.0 + noise)
    if bounds.get("P2O5_mg100g", 0.0) > 0:
        alpha = float(bounds["P2O5_mg100g"])  # ± frac
        if deltas and "P2O5_mg100g" in deltas:
            df[p_col] = df[p_col] * (1.0 + float(deltas["P2O5_mg100g"]))
        else:
            noise = rng.uniform(-alpha, +alpha, size=len(df))
            df[p_col] = df[p_col] * (1.0 + noise)

    # Create replacements mapping and write CHMs
    mapping = replacements_from_dataframe(
        df,
        id_col=id_col,
        n_col=n_col,
        p_col=p_col,
        nitrate_frac=fr.get("nitrate_frac", 0.02),
        organicN_frac=fr.get("organicN_frac", 0.98),
        labileP_frac=fr.get("labileP_frac", 1.0),
        organicP_frac=fr.get("organicP_frac", 0.0),
    )

    # Source TxtInOut comes from provenance base path
    src_txtinout = Path(rp.base_txtinout)
    written = apply_replacements_bulk(
        base_txtinout=src_txtinout,
        dest_txtinout=Path(dest_dir),
        hru_replacements=mapping,
        pperco_val=pperco_val,
        overwrite=True,
    )
    rp.record_outputs(written, kind="chm")
    log.info("Wrote %s CHM files to %s", len(written), dest_dir)
    return written 
"""
