#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

# Ensure repo-local modules are importable when running this script directly.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_PACKAGE_ROOT = SCRIPT_DIR.parent  # .../trabajoFM
sys.path.insert(0, str(REPO_PACKAGE_ROOT))

from python_pipeline_scripts import utils
from python_pipeline_scripts.runner import _resolve_txtinout
from python_pipeline_scripts.provenance_report import build_upstream_inputs, summarize_run, realization_report
from python_pipeline_scripts.raster_agg import rasterZonalAggregationToGPKG
from python_pipeline_scripts.spec_runner import run_from_spec
from python_pipeline_scripts.transforms.point_dat import read_population_by_subbasin_csv_to_df
from python_pipeline_scripts.transforms.point_utils import normalize_point_year_columns
from python_pipeline_scripts.transforms.soil_chm import derive_global_uncertainty, read_n_p_means_from_csv_to_df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Monte Carlo SWAT workflow converted from notebooks/01_run_swat_monte_carlo.ipynb"
    )
    p.add_argument("--base-dir", type=Path, default=REPO_PACKAGE_ROOT, help="Base directory used to resolve relative paths")
    p.add_argument(
        "--config",
        type=Path,
        default=REPO_PACKAGE_ROOT / "config" / "config.yaml",
        help="Path to config YAML",
    )

    p.add_argument("--mode", default="minmax", choices=["mean", "random", "extreme", "minmax"], help="MC mode")
    p.add_argument("--draws", type=int, default=10, help="Number of draws used only when --mode random")
    p.add_argument("--seed", type=int, default=0, help="Random seed")

    p.add_argument(
        "--uncertainty-method",
        default="csv_p75_absolute_relative_error",
        help="Method passed to derive_global_uncertainty",
    )
    p.add_argument("--filter-zeros", action="store_true", default=True, help="Filter rows with Measured == 0 in CV CSV")
    p.add_argument("--no-filter-zeros", action="store_false", dest="filter_zeros", help="Disable zero filtering in CV CSV")
    p.add_argument(
        "--min-measured-value",
        type=float,
        default=0.001,
        help="If set, rows with Measured <= threshold are removed before uncertainty stats",
    )

    p.add_argument("--rebuild-chm-csv", action="store_true", help="Force rebuild CHM intermediate CSV")
    p.add_argument("--rebuild-point-csv", action="store_true", help="Force rebuild point intermediate CSV")
    p.add_argument(
        "--enable-plotting",
        action="store_true",
        help="Enable interactive raster plots during CHM aggregation (disabled by default for terminal runs)",
    )

    p.add_argument("--run-model", action="store_true", default=True, help="Execute SWAT runs")
    p.add_argument("--no-run-model", action="store_false", dest="run_model", help="Prepare realizations only, skip SWAT")

    p.add_argument("--log-level", default="INFO", help="Logging level written into loaded config")
    p.add_argument(
        "--mc-debug",
        action="store_true",
        help="Enable verbose transform debug logs in MC spec (can produce large output)",
    )
    p.add_argument(
        "--show-realization-report",
        action="store_true",
        help="Print full single-realization provenance report at the end (very verbose)",
    )
    return p.parse_args()


def _resolve(path_like: str | Path, base_dir: Path) -> Path:
    pp = Path(path_like)
    return pp if pp.is_absolute() else (base_dir / pp).resolve()


def _require_existing(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required {label}: {path}")
    return path


def _find_files_with_string(
    base_path: Path,
    search_str: str,
    *,
    remove_substring: str = "",
    file_extension_filter: str | None = None,
) -> List[str]:
    if not base_path.is_dir():
        raise NotADirectoryError(f"Not a valid directory: {base_path}")

    ext_filter = file_extension_filter.lower().strip() if file_extension_filter else None
    if ext_filter and not ext_filter.startswith("."):
        ext_filter = "." + ext_filter

    matches: List[str] = []
    for entry in sorted(base_path.iterdir(), key=lambda p: p.name.lower()):
        if not entry.is_file():
            continue
        if search_str not in entry.name:
            continue
        if ext_filter and not entry.name.lower().endswith(ext_filter):
            continue

        name = entry.name
        if remove_substring:
            name = name.replace(remove_substring, "")
        matches.append(name)

    if not matches:
        ext_msg = f" and extension '{ext_filter}'" if ext_filter else ""
        raise FileNotFoundError(f"No files found in {base_path} containing '{search_str}'{ext_msg}")

    return matches


def _read_csv_columns(csv_path: Path, sep: str = ";") -> List[str]:
    with csv_path.open("r", encoding="utf-8") as f:
        header = f.readline().strip()
    if not header:
        raise ValueError(f"CSV appears empty: {csv_path}")
    return [c.strip() for c in header.split(sep)]


def _resolve_chm_column(csv_columns: List[str], raster_stem: str, kind: str) -> str:
    candidates = [f"mean_{raster_stem}", raster_stem]
    for col in candidates:
        if col in csv_columns:
            return col

    similar = [c for c in csv_columns if raster_stem in c or c.startswith("mean_")]
    short_cols = ", ".join(csv_columns[:12])
    similar_msg = ", ".join(similar[:12]) if similar else short_cols
    raise KeyError(
        f"Could not resolve {kind} column for raster stem '{raster_stem}'. "
        f"Tried {candidates}. CSV sample columns: {similar_msg}"
    )


def ensure_chm_csv(
    *,
    chm_csv_path: Path,
    chm_raster_folder: Path,
    hru_zones: Path,
    output_gpkg: Path,
    rebuild: bool,
    cfg: dict,
    enable_plotting: bool,
) -> Path:
    if rebuild or not chm_csv_path.exists():
        log = utils.get_logger("scripts.run_mc_from_notebook_spec", config=cfg)
        log.info("CHM CSV missing or --rebuild-chm-csv set; running raster zonal aggregation")
        _ = rasterZonalAggregationToGPKG(
            raster_folder=chm_raster_folder,
            zones_fp=hru_zones,
            zone_field="HRU_GIS",
            label_field="OBJECTID",
            output_gpkg=output_gpkg,
            files_end_with="_resample.tif",
            stat_operation="mean",
            raster_alias="full_name",
            zone_meaning="HRU",
            overwrite_cache=rebuild,
            enable_plotting=enable_plotting,
        )
    return _require_existing(chm_csv_path, "CHM CSV")


def ensure_point_csv(
    *,
    point_csv_path: Path,
    pop_raster_folder: Path,
    subbasins_fp: Path,
    pop_output_gpkg: Path,
    rebuild: bool,
    cfg: dict,
) -> Path:
    if rebuild or not point_csv_path.exists():
        from python_pipeline_scripts.point_dat_builder import subbasinPopulationAggregationToGPKG

        log = utils.get_logger("scripts.run_mc_from_notebook_spec", config=cfg)
        log.info("Point CSV missing or --rebuild-point-csv set; building point GPKG/CSV")
        _ = subbasinPopulationAggregationToGPKG(
            raster_folder=str(pop_raster_folder),
            sub_basin_fp=str(subbasins_fp),
            zone_field="GRIDCODE",
            output_gpkg=str(pop_output_gpkg),
            overwrite_cache=rebuild,
            enable_plotting=False,
        )
    return _require_existing(point_csv_path, "Point CSV")


def main() -> int:
    args = parse_args()

    base_dir = args.base_dir.resolve()
    cfg = utils.load_config(args.config)
    cfg.setdefault("logging", {})["level"] = str(args.log_level).upper()
    log = utils.get_logger("scripts.run_mc_from_notebook_spec", config=cfg)

    # Initialize module loggers early so cached/rebuild actions are visible.
    utils.get_logger("python_pipeline_scripts.raster_agg", config=cfg)
    utils.get_logger("python_pipeline_scripts.transforms.soil_chm", config=cfg)

    hru_zones = _resolve(r"input_data\Basin info\hru\hru1.shp", base_dir)
    subbasin_zones = _resolve(r"input_data\Basin info\subbasins\Sub_basin.shp", base_dir)
    modified_fig_path = _resolve(
        r"python_pipeline_scripts\script POINT loads - input .dat\.fig files\modified .fig\fig.fig",
        base_dir,
    )

    chm_raster_folder = _resolve(r"input_data\Diffuse loads\soil interpolations", base_dir)
    chm_cross_validation_folder = _resolve(
        r"input_data\Diffuse loads\soil interpolations\soil_interpolation_cross_validation_arcgis",
        base_dir,
    )
    output_gpkg = _resolve(r"input_data\Diffuse loads\python_intermediate_outputs\hru_chem_stats.gpkg", base_dir)

    pop_raster_folder = _resolve(r"input_data\Point loads\fgoerlich-HIPGDAC-ES-cd11f21\HIPGDAC-ES", base_dir)
    pop_output_gpkg = _resolve(r"input_data\Point loads\python_intermediate_outputs\cubillas_population.gpkg", base_dir)

    swat_exe_path = _resolve(r"input_data\swat\arcswat\swat_64rel.exe", base_dir)
    base_txtinout = _resolve(
        r"input_data\swat\cubillas\cubillas_BASE_set-219\BASE recreated from arcswat default\TxtInOut_1",
        base_dir,
    )
    realizations_root = _resolve(r"outputs\realizations", base_dir)
    results_root = _resolve(r"outputs\results", base_dir)

    _require_existing(hru_zones, "HRU shapefile")
    _require_existing(subbasin_zones, "Subbasin shapefile")
    _require_existing(modified_fig_path, "modified fig.fig template")
    _require_existing(chm_raster_folder, "CHM raster folder")
    _require_existing(chm_cross_validation_folder, "CHM cross-validation folder")
    _require_existing(pop_raster_folder, "Population raster folder")
    _require_existing(base_txtinout, "SWAT TxtInOut folder")
    if args.run_model:
        _require_existing(swat_exe_path, "SWAT executable")

    # Resolve nested/messy TxtInOut layouts to the effective deepest valid folder.
    base_txtinout = _resolve_txtinout(base_txtinout)
    log.info("Resolved base TxtInOut: %s", base_txtinout)

    output_gpkg.parent.mkdir(parents=True, exist_ok=True)
    pop_output_gpkg.parent.mkdir(parents=True, exist_ok=True)
    realizations_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    manifest_file = Path(str(output_gpkg) + ".manifest.json")
    chm_csv_path = Path(str(output_gpkg).replace(".gpkg", ".csv"))
    point_csv_path = Path(str(pop_output_gpkg).replace(".gpkg", ".csv"))

    chm_csv_path = ensure_chm_csv(
        chm_csv_path=chm_csv_path,
        chm_raster_folder=chm_raster_folder,
        hru_zones=hru_zones,
        output_gpkg=output_gpkg,
        rebuild=args.rebuild_chm_csv,
        cfg=cfg,
        enable_plotting=args.enable_plotting,
    )
    point_csv_path = ensure_point_csv(
        point_csv_path=point_csv_path,
        pop_raster_folder=pop_raster_folder,
        subbasins_fp=subbasin_zones,
        pop_output_gpkg=pop_output_gpkg,
        rebuild=args.rebuild_point_csv,
        cfg=cfg,
    )

    n_raster_stem = _find_files_with_string(chm_raster_folder, "N_", remove_substring=".tif", file_extension_filter=".tif")[0]
    p_raster_stem = _find_files_with_string(chm_raster_folder, "P_", remove_substring=".tif", file_extension_filter=".tif")[0]
    csv_columns = _read_csv_columns(chm_csv_path, sep=";")
    n_interp = _resolve_chm_column(csv_columns, n_raster_stem, kind="N")
    p_interp = _resolve_chm_column(csv_columns, p_raster_stem, kind="P")

    se_dir = chm_raster_folder / "standard error"
    cv_dir = chm_raster_folder / "soil_interpolation_cross_validation_arcgis"
    _require_existing(se_dir, "standard error folder")
    _require_existing(cv_dir, "cross-validation folder")

    n_pred = chm_raster_folder / _find_files_with_string(chm_raster_folder, "N_", file_extension_filter=".tif")[0]
    n_se = se_dir / _find_files_with_string(se_dir, "N_", file_extension_filter=".tif")[0]
    n_cv_csv = cv_dir / _find_files_with_string(cv_dir, "N_", file_extension_filter=".csv")[0]

    p_pred = chm_raster_folder / _find_files_with_string(chm_raster_folder, "P_", file_extension_filter=".tif")[0]
    p_se = se_dir / _find_files_with_string(se_dir, "P_", file_extension_filter=".tif")[0]
    p_cv_csv = cv_dir / _find_files_with_string(cv_dir, "P_", file_extension_filter=".csv")[0]

    out_n = derive_global_uncertainty(
        prediction=str(n_pred),
        se=str(n_se),
        method=args.uncertainty_method,
        csv_path=str(n_cv_csv),
        csv_filter_zeros=args.filter_zeros,
        csv_min_measured_value=args.min_measured_value,
    )
    out_p = derive_global_uncertainty(
        prediction=str(p_pred),
        se=str(p_se),
        method=args.uncertainty_method,
        csv_path=str(p_cv_csv),
        csv_filter_zeros=args.filter_zeros,
        csv_min_measured_value=args.min_measured_value,
    )

    upstream_chm = build_upstream_inputs(
        raster_folder=chm_raster_folder,
        pattern="*_rediam.tif",
        zones_fp=hru_zones,
        gpkg_path=output_gpkg,
        csv_path=chm_csv_path,
    )
    upstream_point = [point_csv_path]
    upstream = list({Path(p) for p in (upstream_chm + upstream_point)})

    aggregator = lambda: {
        "chm": read_n_p_means_from_csv_to_df(
            str(chm_csv_path),
            id_col="HRU_GIS",
            n_col=n_interp,
            p_col=p_interp,
        ),
        "point": normalize_point_year_columns(
            read_population_by_subbasin_csv_to_df(point_csv_path, id_col="GRIDCODE", sep=";"),
            id_col="GRIDCODE",
            debug=False,
        ),
    }

    mc_spec = {
        "mode": args.mode,
        "draws": int(args.draws),
        "seed": int(args.seed),
        "debug": bool(args.mc_debug),
        "transforms": [
            {
                "type": "ops",
                "target": "chm",
                "name": "compute_base_ops",
                "ops": [
                    {
                        "src": n_interp,
                        "out": "TKN_total_mg_kg",
                        "op": "mul",
                        "lower": 10_000.0 * out_n["lower_factor"],
                        "upper": 10_000.0 * out_n["upper_factor"],
                    },
                    {
                        "src": p_interp,
                        "out": "Olsen_P_mg_kg",
                        "op": "mul",
                        "lower": (10.0 * 0.4364) * out_p["lower_factor"],
                        "upper": (10.0 * 0.4364) * out_p["upper_factor"],
                    },
                ],
            },
            {
                "type": "ops",
                "target": "chm",
                "name": "derive_n_pools_from_tkn",
                "ops": [
                    {"src": "TKN_total_mg_kg", "out": "TN_total_mg_kg", "op": "mul", "lower": 1.0 / 0.97, "upper": 1.0 / 0.97},
                    {"src": "TN_total_mg_kg", "out": "Soil organic N [mg/kg]", "op": "mul", "lower": 0.97, "upper": 0.97},
                    {"src": "TN_total_mg_kg", "out": "Soil NO3 [mg/kg]", "op": "mul", "lower": 0.03, "upper": 0.03},
                ],
            },
            {
                "type": "ops",
                "target": "chm",
                "name": "derive_p_org_from_n",
                "ops": [
                    {"src": "TKN_total_mg_kg", "out": "Soil organic P [mg/kg]", "op": "mul", "lower": 0.07, "upper": 0.2}
                ],
            },
            {
                "type": "split",
                "target": "chm",
                "src": "Olsen_P_mg_kg",
                "renormalize": False,
                "outputs": [
                    {"name": "Soil labile P [mg/kg]", "lower": 1, "upper": 5}
                ],
            },
            {
                "type": "write_chm",
                "target": "chm",
                "id_col": "HRU_GIS",
                "label_map": {
                    "Soil NO3 [mg/kg]": "Soil NO3 [mg/kg]",
                    "Soil organic N [mg/kg]": "Soil organic N [mg/kg]",
                    "Soil labile P [mg/kg]": "Soil labile P [mg/kg]",
                    "Soil organic P [mg/kg]": "Soil organic P [mg/kg]",
                },
                "require_all_source_chm": True,
            },
            {
                "type": "interpolate_years_wide",
                "target": "point",
                "id_col": "GRIDCODE",
                "year_start": 1981,
                "year_end": 2020,
            },
            {
                "type": "build_point",
                "target": "point",
                "id_col": "GRIDCODE",
                "wastewater_lppd": {"lower": 100, "upper": 400},
                "mgL_values": {
                    "ORGNYR": {"lower": 8, "upper": 35},
                    "ORGPYR": {"lower": 1, "upper": 5},
                    "NO3YR": {"lower": 0, "upper": 0},
                    "NH3YR": {"lower": 12, "upper": 50},
                    "NO2YR": {"lower": 0, "upper": 0},
                    "MINPYR": {"lower": 3, "upper": 10},
                    "SEDYR": {"lower": 350, "upper": 1200},
                    "CBODYR": {"lower": 110, "upper": 400},
                    "DISOXYR": {"lower": 0, "upper": 2},
                    "CHLAYR": {"lower": 0.0, "upper": 0.002},
                    "SOLPSTYR": {"lower": 0, "upper": 0},
                    "SRBPSTYR": {"lower": 0, "upper": 0},
                    "BACTPYR": {"lower": 0, "upper": 0},
                    "BACTLPYR": {"lower": 0, "upper": 0},
                    "CMTL1YR": {"lower": 0, "upper": 0},
                    "CMTL2YR": {"lower": 0, "upper": 0},
                    "CMTL3YR": {"lower": 0, "upper": 0},
                },
                "out_columns": [
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
            },
            {
                "type": "write_point_dat",
                "target": "point",
                "id_col": "GRIDCODE",
                "columns_order": [
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
                "start_year": 1981,
                "end_year": 2020,
                "fig_fig_path": str(modified_fig_path),
            },
        ],
    }

    results = run_from_spec(
        spec=mc_spec,
        aggregator=aggregator,
        exe_path=swat_exe_path,
        base_txtinout=base_txtinout,
        realization_root=realizations_root,
        results_root=results_root,
        link_file_regexes=[r"^[0-9]+\.chm$", r"^rcyr_.*\.dat$", r"fig.fig"],
        outputs_to_copy=["output.std", "*.rch", "output.sub", "file.cio"],
        config=cfg,
        manifest_file=manifest_file,
        run_model=args.run_model,
        upstream_inputs=upstream,
    )

    ok = sum(1 for r in results if r.success)
    run_id = results[0].run_id if results else -1
    print(f"Created {len(results)} realizations; {ok} succeeded. run_id={run_id}")
    for r in results:
        print(f"- {r.name}: id={r.realization_id} run_id={r.run_id} success={r.success} folder={r.folder}")

    if results:
        print("\n=== Monte Carlo run summary ===")
        print(summarize_run(run_id))
        if args.show_realization_report:
            print(f"\n=== Single realization report (id={results[0].realization_id}) ===")
            print(realization_report(results[0].realization_id))
        else:
            print("\nSkipping full realization report. Use --show-realization-report to print it.")

    log.info("Finished run_mc_from_notebook_spec")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
