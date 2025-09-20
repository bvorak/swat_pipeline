from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np

from .provenance import RealizationProvenance
from .realization_id import next_id, next_run_id, format_id
from .realizations import RealizationSpec, run_realizations_batch
from .utils import ensure_dir, get_logger


# Type aliases for user hooks
Aggregator = Callable[[], object]
Transform = Callable[[object, Path, np.random.Generator, Dict, RealizationProvenance], tuple[object, List[Path]]]


@dataclass
class MonteCarloResult:
    realization_id: int
    run_id: int
    name: str
    folder: Path
    success: bool
    outputs_dir: Optional[Path]
    returncode: int


def run_monte_carlo(
    *,
    N: int,
    base_txtinout: Path,
    realization_root: Path,
    results_root: Path,
    link_file_regexes: List[str],
    outputs_to_copy: List[str],
    aggregator: Aggregator,
    transforms: Iterable[Transform],
    transforms_params: Optional[List[Dict]] = None,
    exe_path: Optional[Path] = None,
    seed: int = 0,
    expect_plus: bool = False,
    config: Optional[Dict] = None,
    include_base_run: bool = False,
    create_workspace_copy: bool = True,
    force_recreate_workspace: bool = True,
    per_realization_params: Optional[List[Dict]] = None,
    report: bool = True,
    run_model: bool = True,
    upstream_inputs: Optional[List[Path]] = None,
    manifest_file: Optional[Path] = None,
    auto_attach_manifest: bool = False,
    # New: ensure aggressive replacement of inputs prior to run
    preclean_input_globs: Optional[List[str]] = None,
    preclean_linked_inputs: bool = False,
) -> List[MonteCarloResult]:
    """Run N Monte Carlo realizations using provided hooks and capture provenance.

    - aggregator: called once to produce base_data (e.g., aggregated HRU values)
    - transforms: sequence of callables; each writes files under a per-realization folder and returns paths written
    """
    log = get_logger(__name__, config)
    base_txtinout = Path(base_txtinout).resolve()
    realization_root = Path(realization_root).resolve()
    results_root = Path(results_root).resolve()
    ensure_dir(realization_root)
    ensure_dir(results_root)

    # Prepare base_data once
    rng0 = np.random.default_rng(seed)
    log.info("MC start | N=%s | seed=%s", N, seed)
    base_data = aggregator()

    # Prepare provenance/engine metadata
    ledger_path = Path(__file__).resolve().parent.parent / "config" / "provenance" / "realizations.jsonl"
    run_id = next_run_id()
    run_id_str = format_id(run_id)
    engine_meta = {
        "module": __name__,
        "version": "0.1.0",
        "seed": seed,
        "N": N,
        "run_id": run_id,
    }

    # Create transforms for each realization, record provenance, then batch-run
    specs: List[RealizationSpec] = []
    rps: Dict[str, RealizationProvenance] = {}
    results: List[MonteCarloResult] = []

    for i in range(N):
        rid = next_id()
        rid_str = format_id(rid)
        # Use a distinct variable name to avoid accidental shadowing
        realization_name = f"run{run_id_str}_real{rid_str}_{i+1}"
        r_folder = realization_root / realization_name
        ensure_dir(r_folder)

        # Per-realization RNG
        rng = np.random.default_rng(rng0.bit_generator.random_raw() ^ (rid + i + seed))

        rp = RealizationProvenance(
            ledger_path=ledger_path,
            realization_id=rid,
            name=realization_name,
            base_txtinout=base_txtinout,
            realization_folder=r_folder,
            results_dir=results_root / realization_name,
            parameters={
                "link_file_regexes": link_file_regexes,
                "outputs_to_copy": outputs_to_copy,
            },
            engine=engine_meta,
            run_id=run_id,
        )
        # Record upstream inputs (e.g., rasters, zones, GPKG/manifest, CSV)
        # Optionally parse manifest JSON and attach its file paths
        merged_inputs: List[Path] = []
        if auto_attach_manifest and manifest_file and Path(manifest_file).exists():
            try:
                import json
                m = json.loads(Path(manifest_file).read_text(encoding="utf-8"))
                for key in ("rasters",):
                    for p in m.get(key, []) or []:
                        merged_inputs.append(Path(p))
                for key in ("zones", "gpkg"):
                    q = m.get(key)
                    if q:
                        merged_inputs.append(Path(q))
                # also attach the manifest itself
                merged_inputs.append(Path(manifest_file))
            except Exception:
                pass
        if upstream_inputs:
            merged_inputs.extend(list(upstream_inputs))
        if merged_inputs:
            for pth in upstream_inputs:
                try:
                    rp.record_input(Path(pth), compute_hash=False)
                except Exception:
                    pass
            for pth in merged_inputs:
                try:
                    rp.record_input(Path(pth), compute_hash=False)
                except Exception:
                    pass

        # Apply transforms in order, passing data along
        written: List[Path] = []
        data_obj = base_data
        for ti, t in enumerate(transforms):
            base_params = (transforms_params[ti] if transforms_params and ti < len(transforms_params) else {})
            override = None
            if per_realization_params and i < len(per_realization_params):
                pr_i = per_realization_params[i]
                if isinstance(pr_i, list) and ti < len(pr_i):
                    override = pr_i[ti]
                elif isinstance(pr_i, dict):
                    override = pr_i.get(t.__name__)

            # Deep-merge for 'outputs' lists: keep base meta (mean/lower/upper), overlay chosen values (ratio/source)
            call_params = dict(base_params)
            if override:
                for k, v in override.items():
                    if k == "outputs" and isinstance(v, list) and isinstance(base_params.get("outputs"), list):
                        merged = []
                        base_by_name = {o.get("name"): dict(o) for o in base_params.get("outputs")}
                        for ov in v:
                            out_name = ov.get("name")
                            if out_name in base_by_name:
                                merged.append({**base_by_name[out_name], **ov})
                            else:
                                merged.append(dict(ov))
                        call_params["outputs"] = merged
                    else:
                        call_params[k] = v

            with rp.step(name=t.__name__, module=t.__module__, args=call_params):
                data_obj, out_paths = t(data_obj, r_folder, rng, call_params, rp)
                if out_paths:
                    written.extend(out_paths)
        rp.record_outputs(written)

        spec = RealizationSpec(name=realization_name, folder=r_folder)
        specs.append(spec)
        rps[realization_name] = rp

        # In CHM-only mode, finalize each realization immediately to avoid any
        # chance of losing records due to later interruptions and to make
        # provenance visible incrementally in the ledger.
        if not run_model:
            rp.finalize(
                success=True,
                error=None,
                outputs_summary={"note": "model run skipped"},
                additional_fields={"returncode": 0},
                write_copy_to=Path(r_folder) / "provenance.json",
            )
            log.info("Provenance appended | id=%s | name=%s | ledger=%s", rp.id, realization_name, ledger_path)
            results.append(
                MonteCarloResult(
                    realization_id=rp.id,
                    run_id=run_id,
                    name=realization_name,
                    folder=Path(rp.realization_folder),
                    success=True,
                    outputs_dir=None,
                    returncode=0,
                )
            )

    # If run_model is True, we'll overwrite results with batch run outputs below.
    # If False, results already contains per-realization entries and we keep them.
    if run_model:
        results = []
    if run_model:
        # Batch run the model across all realizations
        batch_results = run_realizations_batch(
            base_txtinout=base_txtinout,
            realizations=specs,
            link_file_regexes=link_file_regexes,
            outputs_to_copy=outputs_to_copy,
            exe_path=exe_path,
            expect_plus=expect_plus,
            config=config,
            results_parent_name="_results",
            results_root=results_root,
            include_base_run=include_base_run,
            clean_outputs_before_run=False,
            workspace_dir=None,
            create_workspace_copy=create_workspace_copy,
            force_recreate_workspace=force_recreate_workspace,
            preclean_input_globs=preclean_input_globs,
            preclean_linked_inputs=preclean_linked_inputs,
        )

        for br in batch_results:
            rp = rps.get(br.name)
            if rp:
                rp.finalize(
                    success=br.success,
                    error=None if br.success else br.message,
                    outputs_summary={"outputs_dir": str(br.outputs_dir) if br.outputs_dir else None},
                    additional_fields={"returncode": br.returncode},
                    write_copy_to=Path(rp.realization_folder) / "provenance.json",
                )
                log.info("Provenance appended | id=%s | name=%s | ledger=%s", rp.id, br.name, ledger_path)
            results.append(
                MonteCarloResult(
                    realization_id=rp.id if rp else -1,
                    run_id=run_id,
                    name=br.name,
                    folder=Path(rp.realization_folder) if rp else Path("."),
                    success=br.success,
                    outputs_dir=br.outputs_dir,
                    returncode=br.returncode,
                )
            )
    else:
        # Already finalized and appended results in the loop above
        pass

    if report:
        ok = sum(1 for r in results if r.success)
        ids = [r.realization_id for r in results]
        log.info("MC finished | run_id=%s | %s/%s succeeded | ledger=%s | ids=%s", run_id, ok, len(results), ledger_path, ids)
        log.info("Reproduce: find records by 'id' in the ledger and replay transforms with the recorded parameters and seeds.")

    return results
