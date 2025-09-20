#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
from itertools import product
import sys
import geopandas as gpd

# Ensure the repo-local modules are importable when running this script directly
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_PACKAGE_ROOT = SCRIPT_DIR.parent  # .../trabajoFM
sys.path.insert(0, str(REPO_PACKAGE_ROOT))

from python_pipeline_scripts.raster_agg import raster_zonal_aggregation_to_gpkg
from python_pipeline_scripts.transforms.soil_chm import (
    read_n_p_means_from_csv_to_df,
    mc_transform_write_chm,
)
from python_pipeline_scripts.mc_engine import run_monte_carlo
from python_pipeline_scripts import utils


def _resolve(p: str, base: Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (base / pp).resolve()


def main() -> int:
    # Base dir = folder containing this script; used to resolve relative paths
    script_dir = Path(__file__).resolve().parent

    # 1) Zonal aggregation → GPKG and CSV
    output_gpkg = r"..\..\Genil GEO_INFO_POOL\Input Data\Diffuse loads\Soil chemical composition\python calculated hru stats\hru_chem_stats.gpkg"
    raster_folder = r"..\..\Genil_ArcGIS_Ruben\ArcGIS _ base de suelo quimico _ ruben _ 30-05-25\raster"
    zones_fp = r"C:\Users\Usuario\OneDrive - UNIVERSIDAD DE HUELVA\Archivos de Cesar Ruben Fernandez De Villaran San Juan - swat_cubillas\cubillas_hru\Watershed\Shapes\hru1.shp"

    output_gpkg_p = _resolve(output_gpkg, script_dir)
    raster_folder_p = _resolve(raster_folder, script_dir)
    zones_fp_p = Path(zones_fp)

    _ = raster_zonal_aggregation_to_gpkg(
        raster_folder=raster_folder_p,
        zones_fp=zones_fp_p,
        zone_field="HRU_GIS",
        label_field="OBJECTID",
        output_gpkg=output_gpkg_p,
        files_end_with="_rediam.tif",
        stat_operation="mean",
        raster_alias="full_name",
        zone_meaning="HRU",
        overwrite_cache=False,
        write_manifest=True,
    )

    layer_name = "values_by_hru"
    gdf = gpd.read_file(output_gpkg_p, layer=layer_name)
    csv_output = str(output_gpkg_p).replace(".gpkg", ".csv")
    gdf.drop(columns="geometry").to_csv(csv_output, sep=";", index=False)
    print(f"Exported to CSV: {csv_output}")

    # 2) Monte Carlo — extreme-bound combinations (±20% for N and P)
    base_txtinout = r"C:\Users\Usuario\OneDrive - UNIVERSIDAD DE HUELVA\Archivos de Cesar Ruben Fernandez De Villaran San Juan - swat_cubillas\cubillas_hru\Scenarios\Default\TxtInOut"
    realizations_root = r"C:\SWAT\RSWAT\cubillas\mc_realizations"
    results_root = r"C:\SWAT\RSWAT\cubillas\mc_results"

    base_txtinout_p = Path(base_txtinout)
    realizations_root_p = Path(realizations_root)
    results_root_p = Path(results_root)

    # Build 4 param sets for corners
    alpha_n = 0.20
    alpha_p = 0.20
    per_params = []
    for dn, dp in product([-alpha_n, +alpha_n], [-alpha_p, +alpha_p]):
        per_params.append(
            {
                "bounds": {"N_total_pct": alpha_n, "P2O5_mg100g": alpha_p},
                "deltas": {"N_total_pct": dn, "P2O5_mg100g": dp},
                "id_col": "HRU_GIS",
                "n_col": "mean_Nitrogeno_total_porcent_resample_Rediam",
                "p_col": "mean_Fosforo_mg_100g_P205_rediam",
                # Optional: "pperco_val": 15,
            }
        )

    cfg = utils.load_config(Path("trabajoFM/config/config.yaml"))

    from python_pipeline_scripts.provenance_report import build_upstream_inputs
    manifest_file = Path(str(output_gpkg_p) + ".manifest.json")
    upstream = build_upstream_inputs(
        raster_folder=raster_folder_p,
        pattern="*_rediam.tif",
        zones_fp=zones_fp_p,
        gpkg_path=output_gpkg_p,
        csv_path=Path(csv_output),
    )

    results = run_monte_carlo(
        N=len(per_params),
        base_txtinout=base_txtinout_p,
        realization_root=realizations_root_p,
        results_root=results_root_p,
        link_file_regexes=[r"^.*\.chm$"],
        outputs_to_copy=["output.std", "*.rch"],
        aggregator=lambda: read_n_p_means_from_csv_to_df(
            csv_output,
            id_col="HRU_GIS",
            n_col="mean_Nitrogeno_total_porcent_resample_Rediam",
            p_col="mean_Fosforo_mg_100g_P205_rediam",
        ),
        transforms=[mc_transform_write_chm],
        exe_path=None,  # relies on config.paths.swat_executable
        seed=0,
        expect_plus=False,
        config=cfg,
        include_base_run=True,
        create_workspace_copy=True,
        force_recreate_workspace=True,
        per_realization_params=per_params,
        upstream_inputs=upstream,
        manifest_file=manifest_file if manifest_file.exists() else None,
        auto_attach_manifest=True,
    )

    ok = sum(1 for r in results if r.success)
    print(f"Completed {len(results)} realizations; {ok} succeeded.")
    for r in results:
        print(f"- {r.name}: success={r.success} code={r.returncode} outputs_dir={r.outputs_dir}")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
