# trabajoFM SWAT Pipeline

Lightweight but reproducible SWAT pipeline: modular transforms, Monte Carlo orchestration, and provenance logging. Designed for humans and LLMs to understand, run, and reproduce results.

- `trabajoFM/python_pipeline_scripts/`: importable modules (transforms, writers, MC engine, provenance, runner)
- `trabajoFM/scripts/`: examples and setup scripts
- `trabajoFM/config/`: YAML config, logs, outputs, provenance ledger
- `trabajoFM/notebooks/`: exploratory work that can import modules from the repo

## Quick Start

- Create env + kernel (Windows): `trabajoFM\scripts\setup_venv.ps1`
- Create env + kernel (macOS/Linux): `bash trabajoFM/scripts/setup_venv.sh`
- Configure paths in `trabajoFM/config/config.yaml`:
  - `paths.base_txtinout`: absolute path to your model’s `TxtInOut`
  - `paths.swat_executable`: absolute path to `swat2012.exe` or `swatplus-rel.exe`
- Run the example (extreme bounds): `python trabajoFM\scripts\run_mc_extreme_bounds.py`

## Modular Transforms + MC Engine

- Typical sequence (soil chemistry example):
  1) `transform_compute_base_soil_vars`: derive N_total_mg_kg and P_element_mg_kg from CSV
  2) `transform_perturb_relative`: apply relative/absolute perturbations (bounds or fixed deltas)
  3) `transform_split_fixed_ratios`: split base values into CHM variables by ratios
  4) `transform_write_chm_from_df`: write CHM files for each HRU
- Orchestrate with `python_pipeline_scripts.mc_engine.run_monte_carlo`:
  - `transforms`: list of callables(data, dest_dir, rng, params, rp) → (data, written_paths)
  - `transforms_params`: defaults per transform; `per_realization_params`: overrides per realization
  - Logs to console + `trabajoFM/config/logs/pipeline.log`
  - Appends provenance to `trabajoFM/config/provenance/realizations.jsonl`

## Runner (direct SWAT)

- `from python_pipeline_scripts.runner import run_swat`
- `result = run_swat('C:/.../TxtInOut', exe_path='C:/.../swat2012.exe')`
- With config fallbacks: set `paths.base_txtinout` and `paths.swat_executable`, then `run_swat()`
- Captures `_swat_stdout.txt` and `_swat_stderr.txt` in TxtInOut

## Raster Utilities

- `raster_zonal_aggregation_to_gpkg(...)` aggregates rasters to HRUs and writes a GPKG layer; cached reprojection & clipping with `overwrite_cache=False` to accelerate repeated runs.

## Provenance (JSONL Ledger)

- Location: `trabajoFM/config/provenance/realizations.jsonl` (one JSON per record)
- Record fields (abridged):
  - `id`, `run_id`, `name`, `created_at`, `schema_version`
  - `base_txtinout`, `realization_folder`, `results_dir`
  - `parameters` (high-level), `engine` (module, version, seed, N, run_id)
  - `inputs` (file refs), `steps` (ordered transforms with `args` and timing)
  - `status`, `error`, `outputs_summary`

Interpretation & Reproduction
- Inputs: which source files are required
- Steps: transform sequence and exact args
- Reproduce: execute the same transforms (same order + args) on the same inputs, then run SWAT with the same exe; outputs appear under `results_dir`
- Group into one MC run by `run_id` (shared across all realizations of the same orchestration)

## Provenance Helpers

Functions in `trabajoFM/python_pipeline_scripts/provenance_report.py` help summarize and replay:

- `read_ledger(ledger_path=None) -> list[dict]`: load all records
- `summarize_provenance(ledger_path=None) -> dict`: counts, ids, engine versions, time ranges, run_ids, unique inputs, estimated durations
- `format_summary(summary) -> str`: pretty, human/LLM friendly report
- `realization_report(realization_id, ledger_path=None) -> str`: show transforms/parameters for a specific realization id
- `summarize_run(run_id, ledger_path=None) -> str`: list realizations for a run and time span
- `reconstruct_mc_from_ledger(ids=None, base_txtinout=..., realization_root=..., results_root=..., exe_path=None, config=None)`: rebuilds the transform sequence and replays matching realizations exactly (you supply the aggregator)

Example (report)

```python
from python_pipeline_scripts.provenance_report import read_ledger, summarize_provenance, format_summary
recs = read_ledger()
suminfo = summarize_provenance()
print(format_summary(suminfo))
```

Example (single realization)

```python
from python_pipeline_scripts.provenance_report import realization_report
print(realization_report(123))
```

Example (run summary)

```python
from python_pipeline_scripts.provenance_report import summarize_run
print(summarize_run(5))
```

Example (replay exact runs)

```python
from python_pipeline_scripts.provenance_report import reconstruct_mc_from_ledger
from pathlib import Path
from python_pipeline_scripts import utils
cfg = utils.load_config(Path('trabajoFM/config/config.yaml'))
reconstruct_mc_from_ledger(
    ids=[123, 124],
    base_txtinout=Path(cfg['paths']['base_txtinout']),
    realization_root=Path('C:/replay/realizations'),
    results_root=Path('C:/replay/results'),
    exe_path=None,
    config=cfg,
)
```

## Environment & Setup

- Python target: 3.12
- Windows: `trabajoFM\scripts\setup_venv.ps1` (installs deps, registers Jupyter kernel, writes `.pth` so `python_pipeline_scripts` is importable anywhere in the venv)
- macOS/Linux: `bash trabajoFM/scripts/setup_venv.sh`
- Logging configured via `trabajoFM/config/config.yaml` → `logging`

## Troubleshooting

- “Executable not found”: set `paths.swat_executable` to an absolute exe path
- “TxtInOut not provided”: pass `base_txtinout` explicitly or set `paths.base_txtinout`
- CHMs not written: ensure base CHMs exist; zero‑padded names like `000123.chm` are supported
- Raster step slow: ensure `overwrite_cache=False` to reuse cached reprojection/clip (logs show “(cached)”) 

```text
This README is designed to be parsed by humans and LLMs. Filepaths are absolute or repo‑relative; APIs are importable from `python_pipeline_scripts`. For full context, inspect the ledger at `trabajoFM/config/provenance/realizations.jsonl`.
```
