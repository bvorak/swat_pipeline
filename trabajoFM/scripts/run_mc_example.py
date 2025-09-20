#!/usr/bin/env python
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
from typing import List, Dict
import sys

# Ensure the repo-local modules are importable when running this script directly
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_PACKAGE_ROOT = SCRIPT_DIR.parent  # .../trabajoFM
sys.path.insert(0, str(REPO_PACKAGE_ROOT))

from python_pipeline_scripts.mc_engine import run_monte_carlo
from python_pipeline_scripts.transforms.soil_chm import (
    read_n_p_means_from_csv_to_df,
    mc_transform_write_chm,
)


def build_corner_params(alpha_n: float, alpha_p: float) -> List[Dict]:
    deltas = list(product([-alpha_n, +alpha_n], [-alpha_p, +alpha_p]))
    params_list: List[Dict] = []
    for dn, dp in deltas:
        params_list.append(
            {
                "bounds": {"N_total_pct": alpha_n, "P2O5_mg100g": alpha_p},
                "deltas": {"N_total_pct": dn, "P2O5_mg100g": dp},
                # optional extras: pperco_val, fractions, columns override can be added here
            }
        )
    return params_list


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a tiny MC example using only extreme-bound combinations")
    p.add_argument("--csv", required=True, help="Path to HRU CSV (contains N and P columns)")
    p.add_argument("--base-txtinout", required=True, help="Base TxtInOut folder")
    p.add_argument("--realizations-root", required=True, help="Folder to create per-realization inputs")
    p.add_argument("--results-root", required=True, help="Folder to collect SWAT outputs")
    p.add_argument("--exe", required=True, help="Path to swat2012.exe or swatplus-rel.exe")
    p.add_argument("--alpha-n", type=float, default=0.10, help="± fraction for N total percent (default 0.10)")
    p.add_argument("--alpha-p", type=float, default=0.10, help="± fraction for P2O5 mg/100g (default 0.10)")
    p.add_argument("--id-col", default="HRU_GIS")
    p.add_argument("--n-col", default="mean_Nitrogeno_total_porcent_resample_Rediam")
    p.add_argument("--p-col", default="mean_Fosforo_mg_100g_P205_rediam")
    p.add_argument("--outputs", nargs="*", default=["output.std"], help="Outputs to collect")
    p.add_argument("--include-base-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Load base data once
    base_df = read_n_p_means_from_csv_to_df(
        args.csv,
        id_col=args.id_col,
        n_col=args.n_col,
        p_col=args.p_col,
    )

    # Build per-realization parameter list for extreme corners
    per_params = build_corner_params(args.alpha_n, args.alpha_p)
    # Attach column names to each param dict so the transform can read them
    for d in per_params:
        d.update({
            "id_col": args.id_col,
            "n_col": args.n_col,
            "p_col": args.p_col,
        })

  results = run_monte_carlo(
    N=len(per_params),
        base_txtinout=Path(args.base_txtinout),
        realization_root=Path(args.realizations_root),
        results_root=Path(args.results_root),
        link_file_regexes=[r"^.*\.chm$"],
        outputs_to_copy=args.outputs,
        aggregator=lambda: base_df,
        transforms=[mc_transform_write_chm],
        exe_path=Path(args.exe),
        seed=0,
        expect_plus=False,
        config=None,
        include_base_run=args.include_base_run,
        create_workspace_copy=True,
        force_recreate_workspace=True,
        per_realization_params=per_params,
        # Optionally attach upstream inputs if you have them available in your context
        # upstream_inputs=[Path('.../rasters/..'), Path('.../zones.shp'), Path('.../stats.gpkg'), Path('.../stats.csv')],
        # manifest_file=Path('.../stats.gpkg.manifest.json'),
        auto_attach_manifest=True,
  )

    # Print a short summary
    ok = sum(1 for r in results if r.success)
    print(f"Completed {len(results)} realizations; {ok} succeeded.")
    for r in results:
        print(f"- {r.name}: success={r.success} code={r.returncode} outputs_dir={r.outputs_dir}")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
