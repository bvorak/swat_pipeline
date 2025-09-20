# trabajoFM

Lightweight structure to keep Python modules and Jupyter notebooks side‑by‑side while remaining easy to run as a small pipeline.

- `python_pipeline_scripts/`: importable helpers and the main runner
- `notebooks/`: exploratory and build notebooks (can import the scripts)
- `scripts/`: CLI entry point that wires config to the runner
- `config/`: YAML config with paths, logging, and input groups for reproducible changes

Quick start:

- Install requirements: `pip install -r trabajoFM/requirements.txt`
- Run pipeline: `python trabajoFM/scripts/run_pipeline.py --config trabajoFM/config/config.yaml`
- In notebooks, ensure the parent folder is on `sys.path` (examples provided inside the notebook stubs).

Monte Carlo & Provenance

- Purpose: generate many SWAT input realizations (e.g., perturbations within bounds), run the model, and capture full provenance for reproducibility.
- Components:
  - `provenance`: JSONL ledger writer + `RealizationProvenance` step logger stored at `trabajoFM/config/provenance/realizations.jsonl`.
  - `realization_id`: monotonic ID allocator (IDs 000001..999999).
  - `mc_engine`: orchestrates N realizations with user-provided hooks (aggregator + transforms), then runs the batch via the realizations runner.
- Basic flow (hooks to implement in your code):
  - Aggregator (once): produces base data from rasters/HRUs to feed transforms.
  - Transforms (per realization): draw random factors within bounds, write SWAT-ready files into a per-realization folder, and return paths written. The engine logs steps and assigns IDs.
  - Batch run: the engine calls the existing realizations runner to link files into a workspace, run SWAT, and collect outputs.
- Example sketch:
  - `from python_pipeline_scripts.mc_engine import run_monte_carlo`
  - `def my_aggregator(): ... return base_data`
  - `def my_transform(base, dest, rng, params, rp): ... return [list_of_paths_written]`
  - `run_monte_carlo(N=1000, base_txtinout=..., realization_root=..., results_root=..., link_file_regexes=[...], outputs_to_copy=[...], aggregator=my_aggregator, transforms=[my_transform], exe_path=...)`

- Example script (corner sampling of bounds):
  - `python trabajoFM/scripts/run_mc_example.py --csv <hru.csv> --base-txtinout <TxtInOut> --realizations-root <dir> --results-root <dir> --exe <swat.exe> --alpha-n 0.10 --alpha-p 0.10 --outputs output.std`
  - Runs 4 realizations using all combinations of ±alpha for N and P bounds, writes CHM files via the provided transform, runs SWAT, and records provenance.

RCH parsing

- Single file: `from python_pipeline_scripts.rch_parser import load_output_rch`
  - `df = load_output_rch('C:/.../TxtInOut/output.rch', 'C:/.../TxtInOut/file.cio')`
  - Optional args mirror defaults in the code: `skiprows=9`, `group_size=17`, `add_area_ha=True`, `hectare_per_km2=100.0`, `drop_cols=[...]`.
- Multiple folders: `from python_pipeline_scripts.rch_parser import load_multiple_rch_from_folders`
  - `dfs = load_multiple_rch_from_folders([r'C:/.../TxtInOut_1', r'C:/.../TxtInOut_2'])`  (returns list of DataFrames)
  - Or mapping: `name_to_df = load_multiple_rch_from_folders([...], return_dict=True, name_from='parent', name_prefix='rch_')`

Runner usage

- Run SWAT/SWAT+ for a given TxtInOut folder (Python):
  - `from python_pipeline_scripts.runner import run_swat`
  - `result = run_swat('path/to/TxtInOut', exe_path='C:/path/to/swat2012.exe')`
  - Check: `result.success`, `result.returncode`, `result.stdout_path`, `result.stderr_path`.
  - Optional: `expect_plus=True` if using SWAT+ (`swatplus-rel.exe`). If `exe_path` is omitted, the runner looks in `TxtInOut/`, project root, then system PATH.
- Omit arguments and use config fallbacks:
  - Set in `trabajoFM/config/config.yaml` under `paths`:
    - `base_txtinout`: path to the base model’s `TxtInOut` (or the project root containing it)
    - `swat_executable`: full path to `swat2012.exe` or `swatplus-rel.exe`
  - Then call: `result = run_swat()` and it will use those config values.
- From the pipeline CLI using config groups:
  - Configure `paths.swat_executable` and `input_groups.<name>.folder` in `trabajoFM/config/config.yaml`.
  - `python trabajoFM/scripts/run_pipeline.py --group baseline`
- Logging:
  - Logs go to `trabajoFM/config/logs/pipeline.log` (and console) based on the `logging` section of `trabajoFM/config/config.yaml`.
  - SWAT stdout/stderr are captured to `_swat_stdout.txt` and `_swat_stderr.txt` inside the TxtInOut directory.

Batch realizations

- Purpose: run the same base model for multiple realizations without duplicating the whole TxtInOut. Files matching regex patterns are linked (symlink/hardlink, falling back to copy) from each realization folder into a single working copy; then the model runs and requested outputs are collected.
- API:
  - `from python_pipeline_scripts.realizations import RealizationSpec, run_realizations_batch`
  - Example:
    - `base = r"C:/SWATProjects/MyProj/TxtInOut"`
    - `realizations = [RealizationSpec("urban_2030", r"C:/studies/realizations/urban_2030"), RealizationSpec("rcp45", r"C:/studies/realizations/rcp45")]`
    - `patterns = [r"^weather/.*\\.txt$", r"^climate/.*", r"^inputs/landuse\\.dat$"]`
    - `outputs = ["output.std", "*.hru", "*.rch"]`
    - `res = run_realizations_batch(base, realizations, patterns, outputs, exe_path=r"C:/Tools/SWAT/swat2012.exe", expect_plus=False, results_root=r"C:/results/sim_runs", include_base_run=True, create_workspace_copy=True, force_recreate_workspace=True)`
  - Results: list of `RealizationRunResult` with fields: `name`, `success`, `message`, `returncode`, `outputs_dir`, `linked_count`, `skipped_missing`, `matched_count`, `missing_in_realization`, `extra_in_realization`, `elapsed_seconds`.
- Notes:
  - If you omit `exe_path`, it falls back to `paths.swat_executable` in the config via the shared runner.
  - Logging is verbose; see `pipeline.log` and per-run SWAT stdout/stderr files under the working TxtInOut.
  - On Windows, creating symlinks may require Developer Mode or admin. The function automatically falls back to hardlinks, and then copying if linking is not permitted.
  - Workspace strategy:
    - Default: creates a clean full copy next to the base at `<base>_work/TxtInOut` (or `workspace_dir` if provided). With `force_recreate_workspace=True` (default), any existing workspace copy is deleted and recreated.
    - In-place mode: set `create_workspace_copy=False` to manipulate the original base `TxtInOut` directly (danger: it will be modified). Useful when you explicitly want to run on the original.
  - Output files are overwritten in the results destination if they already exist (previous runs). The runner also unlinks matching outputs in the workspace before each run so they are regenerated fresh.
  - Optional: `results_root` saves all outputs under that folder with subfolders per realization name; otherwise outputs are saved under each realization folder in `<realization>/_results/<name>`.
  - Optional: `include_base_run=True` runs the copied base model once before any realization and stores outputs under `base_<parent-of-TxtInOut>` in the `results_root` (or default location).
  - Summary stats are logged per realization: total matched files, linked count, how many matched files were missing in the realization, and how many files matched in the realization but had no counterpart in the working copy. Extra files from the realization are also linked into the workspace before running (a warning is logged along with this action).

Integration: external tools

- This template does not install external SWAT wrappers or tools by default. You can add them to `trabajoFM/requirements.txt` or install them manually after creating the venv.
- For reproducibility, freeze a lock file after you have a known‑good setup (see the freeze scripts in `trabajoFM/scripts/`).

Reproducible environment (team-friendly)

- Pinned Python: this repo targets Python 3.12 (see `.python-version`).
- Create venv + install + Jupyter kernel (Windows PowerShell): `trabajoFM/scripts/setup_venv.ps1` (prefers Python 3.12)
- Create venv + install + Jupyter kernel (macOS/Linux): `bash trabajoFM/scripts/setup_venv.sh` (prefers Python 3.12)
- Prefer installing from a lock file for reproducibility. After you have a known-good env, freeze it:
  - Windows: `trabajoFM/scripts/freeze_lock.ps1`
  - macOS/Linux: `bash trabajoFM/scripts/freeze_lock.sh`
  Commit `trabajoFM/requirements.lock.txt` and others can do: `pip install -r trabajoFM/requirements.lock.txt`.

Notes

- Ensure your Jupyter kernel is the venv you created (named "Python (TrabajoFM SWAT)" by the setup scripts).
- If using a different Python version across machines, update the lock file from that version to avoid resolver conflicts.
