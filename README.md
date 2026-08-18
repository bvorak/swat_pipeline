# trabajoFM SWAT Pipeline

A lightweight but reproducible pipeline for running SWAT models: modular input transforms, Monte Carlo orchestration, and full provenance logging so every run can be traced and replayed.

**Project layout:**

| Folder | What's in it |
|---|---|
| [`trabajoFM/python_pipeline_scripts/`](https://github.com/bvorak/swat_pipeline/tree/main/trabajoFM/python_pipeline_scripts) | Importable Python modules: transforms, the Monte Carlo engine, the SWAT runner, provenance tools, output parsers |
| [`trabajoFM/notebooks/`](https://github.com/bvorak/swat_pipeline/tree/main/trabajoFM/notebooks) | Exploratory/build notebooks — can import everything from `python_pipeline_scripts` |
| [`trabajoFM/scripts/`](https://github.com/bvorak/swat_pipeline/tree/main/trabajoFM/scripts) | Command-line entry points that wire the config to the pipeline (setup, running, freezing dependencies) |
| [`trabajoFM/input_data/`](https://github.com/bvorak/swat_pipeline/tree/main/trabajoFM/input_data) | Large input data — see [Data Setup](#data-setup-input_data) |
| [`trabajoFM/config/`](https://github.com/bvorak/swat_pipeline/tree/main/trabajoFM/config) | YAML config, logs, and the provenance ledger |

New here? Jump straight to **[Quickstart](#quickstart)** below — it walks through the whole setup using nothing but a browser download.

## Table of Contents

- [Quickstart](#quickstart)
- [Data Setup (input_data)](#data-setup-input_data)
- [How the Pipeline Works](#how-the-pipeline-works)
- [Running SWAT](#running-swat)
- [Reading Model Outputs (RCH files)](#reading-model-outputs-rch-files)
- [Raster Utilities](#raster-utilities)
- [Provenance & Reproducibility (the ledger)](#provenance--reproducibility-the-ledger)
- [Environment & Setup](#environment--setup)
- [Scripts & Notebooks Quick Reference](#scripts--notebooks-quick-reference)
- [Troubleshooting](#troubleshooting)

---

## Quickstart

Here is how to set up the environment needed to execute all input_data transformations + uncerteinty + monte carlo run logic with one script. 

For the Github-ZIP + manual input_data copy workflow, you only need:

1. Python 3.12 installed
2. around 34 GB of free hard disk space

The easiest way (without git and it´s LFS big file extension) is to:
1) downlaod the Code.zip from the [Github](https://github.com/bvorak/swat_pipeline). Unzip to `YOUR-LOCATION` . This is your main project folder.
2) downlaod the [input_data.zip](https://universidadhuelva-my.sharepoint.com/:u:/g/personal/konstantinpascal_walter_alu_uhu_es/IQAETGkg6A3TTKUdO301mQQYAcXqXc4CineAIbyJ7eE3rH8?e=I1TD2O). Unzip. Replace ``YOUR-LOCATION\swat_pipeline-main\trabajoFM\input_data`` 
3) If you dont have the 219 BASE calibration TxtInOut on your machine already: Extract the [´TxtInOut_1_work.zip´ from within this .zip](https://universidadhuelva-my.sharepoint.com/:u:/g/personal/konstantinpascal_walter_alu_uhu_es/IQDc6mH7Ze32T72QL3IVCtXGAUkdDRGfvjCtFDV3PqO6-JA?e=sjuBd6). Unzip twice. 
	1) Move into `YOUR-LOCATION\swat_pipeline-main\trabajoFM\input_data\swat\cubillas\cubillas_BASE_set-219\BASE recreated from arcswat default\TxtInOut_1\`


(Git and Git LFS are not required for this specific workflow if you manually provide input_data. But if we want to keep updating this codebase it will be easier in the long run to use git and LFS)


EXECUTE: Step by step (Windows, PowerShell)



1. Open PowerShell in `trabajoFM`.

- Example path should end with `YOUR-LOCATION/swat_pipeline-main/trabajoFM.`

4. Run environment setup.

- If script execution is blocked, run first:  
    ``Set-ExecutionPolicy -Scope Process Bypass``


- Command:  
    ``.\scripts\setup_venv.ps1``

- This installs dependencies from `requirements.txt` or lock file if present.

5. Activate the virtual environment.

- Command:  
    `.venv/scripts/Activate.ps1`

6. Quick sanity check.

- Command:  
    ``python -V`` -> should be 3.12.XX

7. Test the converted workflow without running SWAT.

- Script: `run_mc_from_notebook_spec.py`
- Command:  
    python .\scripts\run_mc_from_notebook_spec.py --no-run-model
- This validates paths, data transforms, and realization generation flow.

8. Run the full workflow (includes SWAT execution).

- Command:  
    python .\scripts\run_mc_from_notebook_spec.py --mode minmax --rebuild-chm-csv --rebuild-point-csv
- This requires SWAT executable and base TxtInOut under your input_data tree as configured in `config.yaml`.

9. If full run fails, check first these paths:

- SWAT exe path in `config.yaml`
- Base TxtInOut path in `config.yaml`
- Required fig template path expected by `run_mc_from_notebook_spec.py`

NEXT:
- the workflow can be easiest understood and debugged using the Jupyter Notebook that wraps this pipeline `trabajoFM\notebooks\01_run_swat_monte_carlo.ipynb`
- For command line runs. all paths, monte carlo adjustments and so on can best be adjusted in `run_mc_from_notebook_spec.py` (see MC_SPEC variable)
- The second part of the pipeline: dashboard + automated analysis jupyter notebooks (03_---.ipynb and 04_---.ipynb) still has some absolute paths, that need to be adjusted before working on a new machine.

---

## Data Setup (`input_data`)

Everything under `trabajoFM/input_data/` counts as **large data** — treat it separately from the code.

> The Quickstart above already walks through the simplest option (a ready-made ZIP). This section covers both supported ways in more detail, plus what to do if something looks off.

- **Path A — Git clone + Git LFS** (best if you'll keep pulling updates): clone the repo with Git, then run `git lfs install` and `git lfs pull`.
- **Path B — external data package** (no Git LFS needed, and the path the Quickstart uses): download the externally provided ZIP and replace the whole `trabajoFM/input_data/` folder with the extracted one.

⚠️ Don't rely on GitHub's "Download ZIP" button for this data on its own — Git LFS files can download as small pointer files instead of the real content.

If data looks like it's missing or suspiciously small, run `git lfs ls-files` to check, then `git lfs pull` again.

---

## How the Pipeline Works

Two building blocks sit behind every run: **modular transforms** that prepare SWAT input files, and a **Monte Carlo engine** that turns them into many reproducible realizations.

### Modular transforms

A "transform" is just a function that takes some data, applies a change, and writes SWAT-ready files. A typical soil-chemistry sequence looks like this:

1. `transform_compute_base_soil_vars` — derive `N_total_mg_kg` and `P_element_mg_kg` from a CSV
2. `transform_perturb_relative` — apply relative/absolute perturbations (bounds or fixed deltas)
3. `transform_split_fixed_ratios` — split base values into CHM variables (SWAT's soil chemistry input files) by ratio
4. `transform_write_chm_from_df` — write the CHM file for each HRU (Hydrologic Response Unit)

### The Monte Carlo engine

Orchestrate any sequence of transforms with [`python_pipeline_scripts.mc_engine.run_monte_carlo`](https://github.com/bvorak/swat_pipeline/blob/main/trabajoFM/python_pipeline_scripts/mc_engine.py):

- **Aggregator** (runs once): builds the base data your transforms will perturb — e.g. from rasters/HRUs.
- **Transforms** (run once per realization): draw random factors within bounds, write the SWAT-ready files into a per-realization folder, and return the paths written.
- **Batch run**: the engine hands everything to the [realizations runner](#batch-realizations) to link files into a workspace, run SWAT, and collect outputs.

Key arguments: `transforms` (a list of `callable(data, dest_dir, rng, params, rp) -> (data, written_paths)`), `transforms_params` (defaults per transform), `per_realization_params` (overrides per realization). Every run logs to `trabajoFM/config/logs/pipeline.log` and appends provenance to `trabajoFM/config/provenance/realizations.jsonl`.

Minimal sketch:

```python
from python_pipeline_scripts.mc_engine import run_monte_carlo

def my_aggregator():
    ...
    return base_data

def my_transform(base, dest, rng, params, rp):
    ...
    return [list_of_paths_written]

run_monte_carlo(
    N=1000,
    base_txtinout=...,
    realization_root=...,
    results_root=...,
    link_file_regexes=[...],
    outputs_to_copy=[...],
    aggregator=my_aggregator,
    transforms=[my_transform],
    exe_path=...,
)
```

Two ready-made entry points wrap this — see the [quick reference table](#scripts--notebooks-quick-reference) for links.

---

## Running SWAT

### A single, direct run

```python
from python_pipeline_scripts.runner import run_swat
result = run_swat('C:/.../TxtInOut', exe_path='C:/.../swat2012.exe')
```

Check `result.success`, `result.returncode`, `result.stdout_path`, `result.stderr_path`. Pass `expect_plus=True` if you're using SWAT+ (`swatplus-rel.exe`). If you omit `exe_path`, the runner looks in `TxtInOut/`, then the project root, then your system `PATH`.

Prefer config over arguments? Set `paths.base_txtinout` and `paths.swat_executable` in `trabajoFM/config/config.yaml`, then just call `run_swat()`.

From the command line, using named config groups:

```
python trabajoFM/scripts/run_pipeline.py --group baseline
```

(configure `paths.swat_executable` and `input_groups.<name>.folder` in `config.yaml` first)

Either way, SWAT's stdout/stderr are captured to `_swat_stdout.txt` / `_swat_stderr.txt` inside the `TxtInOut` folder, and logs also go to `trabajoFM/config/logs/pipeline.log` (and the console).

### Batch realizations

Running the same base model dozens or hundreds of times doesn't mean duplicating the whole `TxtInOut` folder each time. [`python_pipeline_scripts.realizations`](https://github.com/bvorak/swat_pipeline/blob/main/trabajoFM/python_pipeline_scripts/realizations.py) links (symlink → hardlink → copy, in that order of preference) only the files that differ into one shared working copy, then runs the model and collects the requested outputs.

```python
from python_pipeline_scripts.realizations import RealizationSpec, run_realizations_batch

base = r"C:/SWATProjects/MyProj/TxtInOut"
realizations = [
    RealizationSpec("urban_2030", r"C:/studies/realizations/urban_2030"),
    RealizationSpec("rcp45", r"C:/studies/realizations/rcp45"),
]
patterns = [r"^weather/.*\.txt$", r"^climate/.*", r"^inputs/landuse\.dat$"]
outputs = ["output.std", "*.hru", "*.rch"]

res = run_realizations_batch(
    base, realizations, patterns, outputs,
    exe_path=r"C:/Tools/SWAT/swat2012.exe",
    expect_plus=False,
    results_root=r"C:/results/sim_runs",
    include_base_run=True,
    create_workspace_copy=True,
    force_recreate_workspace=True,
)
```

Each result comes back as a `RealizationRunResult` with `name`, `success`, `message`, `returncode`, `outputs_dir`, `linked_count`, `skipped_missing`, `matched_count`, `missing_in_realization`, `extra_in_realization`, and `elapsed_seconds`.

A few things worth knowing:
- If `exe_path` is omitted, it falls back to `paths.swat_executable` in the config.
- On Windows, creating symlinks may need Developer Mode or admin rights — the function falls back to hardlinks, then to copying, automatically.
- **Workspace strategy**: by default a clean copy is created at `<base>_work/TxtInOut` (or `workspace_dir`, if you pass one); with `force_recreate_workspace=True` (the default) any existing copy is deleted and rebuilt first. Set `create_workspace_copy=False` to modify the original base `TxtInOut` in place instead — only do this if you mean to.
- `include_base_run=True` also runs the copied base model once, storing its outputs under `base_<parent-of-TxtInOut>`.
- Each realization's summary logs how many files matched, linked, were missing, or had no counterpart — check `pipeline.log` if something looks off. Extra files found in a realization folder are linked into the workspace too (with a warning logged).

---

## Reading Model Outputs (RCH files)

[`python_pipeline_scripts.rch_parser`](https://github.com/bvorak/swat_pipeline/blob/main/trabajoFM/python_pipeline_scripts/rch_parser.py) turns SWAT's `output.rch` (the reach/routing output file) into a pandas DataFrame.

Single file:
```python
from python_pipeline_scripts.rch_parser import load_output_rch
df = load_output_rch('C:/.../TxtInOut/output.rch', 'C:/.../TxtInOut/file.cio')
```
Optional arguments mirror the code's defaults: `skiprows=9`, `group_size=17`, `add_area_ha=True`, `hectare_per_km2=100.0`, `drop_cols=[...]`.

Multiple folders at once:
```python
from python_pipeline_scripts.rch_parser import load_multiple_rch_from_folders
dfs = load_multiple_rch_from_folders([r'C:/.../TxtInOut_1', r'C:/.../TxtInOut_2'])
# or, as a name -> DataFrame mapping:
name_to_df = load_multiple_rch_from_folders([...], return_dict=True, name_from='parent', name_prefix='rch_')
```

---

## Raster Utilities

`raster_zonal_aggregation_to_gpkg(...)` aggregates raster data to HRUs and writes the result as a GeoPackage (`.gpkg`) layer. Reprojection and clipping are cached — pass `overwrite_cache=False` to reuse that cache and speed up repeated runs (you'll see `(cached)` in the logs when it kicks in).

---

## Provenance & Reproducibility (the ledger)

Every realization writes a record to a JSONL ledger at `trabajoFM/config/provenance/realizations.jsonl` — one JSON object per line, so you can always trace, and replay, exactly what produced a given result.

### What's in a record

| Field group | Contents |
|---|---|
| Identity | `id`, `run_id`, `name`, `created_at`, `schema_version` |
| Paths | `base_txtinout`, `realization_folder`, `results_dir` |
| Setup | `parameters` (high-level), `engine` (module, version, seed, N, `run_id`) |
| History | `inputs` (file references), `steps` (ordered transforms with their `args` and timing) |
| Result | `status`, `error`, `outputs_summary` |

Realizations sharing a `run_id` belong to the same Monte Carlo orchestration. To reproduce a result: run the same transforms, in the same order, with the same args, on the same inputs, then run SWAT with the same executable — outputs land under the same `results_dir`.

### Helper functions

[`python_pipeline_scripts/provenance_report.py`](https://github.com/bvorak/swat_pipeline/blob/main/trabajoFM/python_pipeline_scripts/provenance_report.py) has everything you need to summarize or replay past runs:

- `read_ledger(ledger_path=None) -> list[dict]` — load all records
- `summarize_provenance(ledger_path=None) -> dict` — counts, ids, engine versions, time ranges, run_ids, unique inputs, estimated durations
- `format_summary(summary) -> str` — a pretty, human/LLM-friendly report
- `realization_report(realization_id, ledger_path=None) -> str` — transforms/parameters for one realization
- `summarize_run(run_id, ledger_path=None) -> str` — all realizations in a run, and their time span
- `reconstruct_mc_from_ledger(ids=None, base_txtinout=..., realization_root=..., results_root=..., exe_path=None, config=None)` — rebuilds the transform sequence and replays matching realizations exactly (you supply the aggregator)

```python
# Overall summary
from python_pipeline_scripts.provenance_report import read_ledger, summarize_provenance, format_summary
recs = read_ledger()
print(format_summary(summarize_provenance()))
```
```python
# One realization
from python_pipeline_scripts.provenance_report import realization_report
print(realization_report(123))
```
```python
# One run
from python_pipeline_scripts.provenance_report import summarize_run
print(summarize_run(5))
```
```python
# Replay exact runs
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

---

## Environment & Setup

- **Python version**: this repo targets Python 3.12 (pinned in `.python-version`).
- **Set up the venv + Jupyter kernel:**
  - Windows: [`trabajoFM\scripts\setup_venv.ps1`](https://github.com/bvorak/swat_pipeline/blob/main/trabajoFM/scripts/setup_venv.ps1)
  - macOS/Linux: `bash` [`trabajoFM/scripts/setup_venv.sh`](https://github.com/bvorak/swat_pipeline/blob/main/trabajoFM/scripts/setup_venv.sh)

  Either script installs dependencies (from `requirements.txt`, or the lock file if present), registers a Jupyter kernel named **"Python (TrabajoFM SWAT)"**, writes a `.pth` file so `python_pipeline_scripts` is importable anywhere in the venv, and runs `git lfs install --local` when Git LFS is available.
- **Freeze a lock file** once you have a known-good environment, so others can reproduce it exactly:
  - Windows: [`trabajoFM/scripts/freeze_lock.ps1`](https://github.com/bvorak/swat_pipeline/blob/main/trabajoFM/scripts/freeze_lock.ps1)
  - macOS/Linux: `bash` [`trabajoFM/scripts/freeze_lock.sh`](https://github.com/bvorak/swat_pipeline/blob/main/trabajoFM/scripts/freeze_lock.sh)

  Commit the resulting `trabajoFM/requirements.lock.txt`; others then run `pip install -r trabajoFM/requirements.lock.txt`.
- **Logging** is configured via the `logging` section of `trabajoFM/config/config.yaml`.
- **External tools**: this template doesn't install external SWAT wrappers by default — add them to `trabajoFM/requirements.txt`, or install manually after creating the venv.
- If you switch Python versions across machines, re-freeze the lock file from that version to avoid resolver conflicts.

---

## Scripts & Notebooks Quick Reference

| File | Folder | What it does |
|---|---|---|
| [`01_run_swat_monte_carlo.ipynb`](https://github.com/bvorak/swat_pipeline/blob/main/trabajoFM/notebooks/01_run_swat_monte_carlo.ipynb) | [`notebooks/`](https://github.com/bvorak/swat_pipeline/tree/main/trabajoFM/notebooks) | The easiest way to understand and debug the pipeline — wraps the whole Monte Carlo workflow. Start here if you like working interactively. |
| `03_...ipynb`, `04_...ipynb` | [`notebooks/`](https://github.com/bvorak/swat_pipeline/tree/main/trabajoFM/notebooks) | Dashboard + automated analysis notebooks (second half of the pipeline). Still contain some absolute paths that need adjusting on a new machine. |
| [`run_mc_from_notebook_spec.py`](https://github.com/bvorak/swat_pipeline/blob/main/trabajoFM/scripts/run_mc_from_notebook_spec.py) | [`scripts/`](https://github.com/bvorak/swat_pipeline/tree/main/trabajoFM/scripts) | CLI version of the notebook's Monte Carlo spec (see the `MC_SPEC` variable) — the script used in the Quickstart. |
| [`run_mc_example.py`](https://github.com/bvorak/swat_pipeline/blob/main/trabajoFM/scripts/run_mc_extreme_bounds.py) *(now `run_mc_extreme_bounds.py`)* | [`scripts/`](https://github.com/bvorak/swat_pipeline/tree/main/trabajoFM/scripts) | Corner-sampling example: 4 realizations across every ± combination of N/P bounds, e.g. `--alpha-n 0.10 --alpha-p 0.10`. |
| [`run_pipeline.py`](https://github.com/bvorak/swat_pipeline/blob/main/trabajoFM/scripts/run_pipeline.py) | [`scripts/`](https://github.com/bvorak/swat_pipeline/tree/main/trabajoFM/scripts) | CLI entry point that runs a named `input_groups` config against the runner. |
| [`setup_venv.ps1`](https://github.com/bvorak/swat_pipeline/blob/main/trabajoFM/scripts/setup_venv.ps1) / [`setup_venv.sh`](https://github.com/bvorak/swat_pipeline/blob/main/trabajoFM/scripts/setup_venv.sh) | [`scripts/`](https://github.com/bvorak/swat_pipeline/tree/main/trabajoFM/scripts) | Creates the venv, installs deps, registers the Jupyter kernel. |
| [`freeze_lock.ps1`](https://github.com/bvorak/swat_pipeline/blob/main/trabajoFM/scripts/freeze_lock.ps1) / [`freeze_lock.sh`](https://github.com/bvorak/swat_pipeline/blob/main/trabajoFM/scripts/freeze_lock.sh) | [`scripts/`](https://github.com/bvorak/swat_pipeline/tree/main/trabajoFM/scripts) | Freezes the current environment into `requirements.lock.txt`. |
| [`mc_engine.py`](https://github.com/bvorak/swat_pipeline/blob/main/trabajoFM/python_pipeline_scripts/mc_engine.py) | [`python_pipeline_scripts/`](https://github.com/bvorak/swat_pipeline/tree/main/trabajoFM/python_pipeline_scripts) | The Monte Carlo orchestrator (`run_monte_carlo`). |
| [`runner.py`](https://github.com/bvorak/swat_pipeline/blob/main/trabajoFM/python_pipeline_scripts/runner.py) | [`python_pipeline_scripts/`](https://github.com/bvorak/swat_pipeline/tree/main/trabajoFM/python_pipeline_scripts) | Direct SWAT execution (`run_swat`). |
| [`realizations.py`](https://github.com/bvorak/swat_pipeline/blob/main/trabajoFM/python_pipeline_scripts/realizations.py) | [`python_pipeline_scripts/`](https://github.com/bvorak/swat_pipeline/tree/main/trabajoFM/python_pipeline_scripts) | Batch realization runner (`RealizationSpec`, `run_realizations_batch`). |
| [`provenance_report.py`](https://github.com/bvorak/swat_pipeline/blob/main/trabajoFM/python_pipeline_scripts/provenance_report.py) | [`python_pipeline_scripts/`](https://github.com/bvorak/swat_pipeline/tree/main/trabajoFM/python_pipeline_scripts) | Summarize and replay runs from the provenance ledger. |
| [`rch_parser.py`](https://github.com/bvorak/swat_pipeline/blob/main/trabajoFM/python_pipeline_scripts/rch_parser.py) | [`python_pipeline_scripts/`](https://github.com/bvorak/swat_pipeline/tree/main/trabajoFM/python_pipeline_scripts) | Parses `output.rch` into DataFrames. |
| [`utils.py`](https://github.com/bvorak/swat_pipeline/blob/main/trabajoFM/python_pipeline_scripts/utils.py) | [`python_pipeline_scripts/`](https://github.com/bvorak/swat_pipeline/tree/main/trabajoFM/python_pipeline_scripts) | Shared helpers, including `load_config`. |

*Note: `03_...ipynb`/`04_...ipynb` and the exact current filename of the extreme-bounds example script weren't fully pinned down in the source notes — check the [`notebooks/`](https://github.com/bvorak/swat_pipeline/tree/main/trabajoFM/notebooks) and [`scripts/`](https://github.com/bvorak/swat_pipeline/tree/main/trabajoFM/scripts) folders directly if a link above looks stale.*

---

## Troubleshooting

- **"Executable not found"** → set `paths.swat_executable` to an absolute path in `config.yaml`.
- **"TxtInOut not provided"** → pass `base_txtinout` explicitly, or set `paths.base_txtinout` in `config.yaml`.
- **CHMs not written** → make sure the base CHM files exist; zero-padded names like `000123.chm` are supported.
- **Raster step is slow** → make sure `overwrite_cache=False` so cached reprojection/clipping gets reused (you'll see `(cached)` in the logs).

---

> This README is written to be read by both humans and LLMs. Paths are given as absolute or repo-relative, and everything under `python_pipeline_scripts` is importable directly. For full run-by-run detail, the source of truth is always the ledger at `trabajoFM/config/provenance/realizations.jsonl`.
