from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .mc_engine import run_monte_carlo
from .provenance_report import realization_report as _realization_report, write_provenance_reports
from .transforms.soil_chm import (
    transform_apply_ops,
    transform_split_with_bounds,
    transform_write_chm_from_df,
)
from .transforms.point_dat import (
    transform_interpolate_years_wide,
    transform_build_point_load_timeseries,
    transform_write_point_dat_from_df,
)
from .transforms.point_utils import normalize_point_year_columns


def _wrap_on(key: str, inner: Callable):
    def fn(data, dest_dir, rng, params, rp):
        df = data[key]
        out_df, paths = inner(df, dest_dir, rng, params, rp)
        data[key] = out_df
        return data, paths
    return fn


def _coerce_ops_for_apply(ops_list: List[Dict]) -> List[Dict]:
    out = []
    for spec in ops_list:
        spec = dict(spec)
        op = spec.get("op") or spec.get("mode")
        if op in ("mul", "relative"):
            if "factor" not in spec and "standard" in spec:
                spec["factor"] = float(spec["standard"])
            if "factor" not in spec and "mean" in spec:
                spec["factor"] = float(spec["mean"])
        elif op in ("add", "absolute"):
            if "delta" not in spec and "standard" in spec:
                spec["delta"] = float(spec["standard"])
            if "delta" not in spec and "mean" in spec:
                spec["delta"] = float(spec["mean"])
        elif op == "set":
            if "value" not in spec and "standard" in spec:
                spec["value"] = float(spec["standard"])
            if "value" not in spec and "mean" in spec:
                spec["value"] = float(spec["mean"])
        out.append(spec)
    return out


def _op_param_name(kind: str) -> str:
    return {"mul": "factor", "relative": "factor", "add": "delta", "absolute": "delta", "set": "value"}.get(
        kind, "factor"
    )


def _has_bounds(lo, hi) -> bool:
    try:
        return lo is not None and hi is not None and float(hi) != float(lo)
    except:  # noqa: E722
        return False


def _build_transforms_and_params(spec: Dict[str, Any], *, manifest_file: Optional[Path], debug: bool):
    transforms = []
    transforms_params = []

    def transform_init_copy(data, dest_dir, rng, params, rp):
        out = {}
        import copy as _copy

        for k, v in data.items():
            try:
                out[k] = v.copy(deep=True)
            except Exception:
                out[k] = _copy.deepcopy(v)
        return out, []

    transforms.append(transform_init_copy)
    transforms_params.append({})

    for t in spec.get("transforms", []):
        tgt = t.get("target", "chm")
        ttype = t["type"]

        if ttype == "ops":
            transforms.append(_wrap_on(tgt, transform_apply_ops))
            coerced_ops = _coerce_ops_for_apply(t.get("ops", []))
            transforms_params.append(
                {
                    "ops": coerced_ops,
                    "input_source": str(manifest_file) if manifest_file and Path(manifest_file).exists() else None,
                    "debug": debug,
                }
            )
        elif ttype == "split":
            transforms.append(_wrap_on(tgt, transform_split_with_bounds))
            outputs = []
            for o in t.get("outputs", []):
                outputs.append(
                    {
                        "name": o["name"],
                        "mean": o.get("standard", o.get("mean")),
                        "lower": o.get("lower"),
                        "upper": o.get("upper"),
                        "source": spec.get("mode"),
                    }
                )
            transforms_params.append(
                {
                    "src": t["src"],
                    "renormalize": bool(t.get("renormalize", True)),
                    "outputs": outputs,
                    "input_source": str(manifest_file) if manifest_file and Path(manifest_file).exists() else None,
                    "debug": debug,
                }
            )
        elif ttype == "write_chm":
            transforms.append(_wrap_on(tgt, transform_write_chm_from_df))
            transforms_params.append({"id_col": t["id_col"], "label_map": t["label_map"]})
        elif ttype == "interpolate_years_wide":
            def _normalize_then_interpolate(df, dest_dir, rng, params, rp):
                df2 = normalize_point_year_columns(df, id_col=params.get("id_col", "GRIDCODE"), debug=debug)
                return transform_interpolate_years_wide(df2, dest_dir, rng, params, rp)

            transforms.append(_wrap_on("point", _normalize_then_interpolate))
            transforms_params.append(
                {
                    "id_col": t.get("id_col", "GRIDCODE"),
                    "year_start": int(t["year_start"]),
                    "year_end": int(t["year_end"]),
                    "keep_existing": bool(t.get("keep_existing", True)),
                }
            )
        elif ttype == "build_point":
            transforms.append(_wrap_on("point", transform_build_point_load_timeseries))
            transforms_params.append(
                {
                    "id_col": t.get("id_col", "GRIDCODE"),
                    # Accept scalar or bounds dict {standard/lower/upper}
                    "wastewater_lppd": t.get("wastewater_lppd", 500.0),
                    "mgL_values": t.get("mgL_values", {}),
                    "out_columns": t.get("out_columns"),
                    "round_to": int(t.get("round_to", 6)),
                }
            )
        elif ttype == "write_point_dat":
            transforms.append(_wrap_on("point", transform_write_point_dat_from_df))
            transforms_params.append(
                {
                    "id_col": t.get("id_col", "GRIDCODE"),
                    "columns_order": t.get("columns_order"),
                    "start_year": t.get("start_year"),
                    "end_year": t.get("end_year"),
                }
            )
        else:
            raise ValueError(f"Unknown transform type: {ttype}")

    return transforms, transforms_params


def _build_overrides(spec: Dict[str, Any], *, mode: str, draws: int, seed: Optional[int]):
    from itertools import product

    def extreme():
        per_tf_opts = []
        for t in spec.get("transforms", []):
            tt = t["type"]
            if tt == "ops":
                bundles, any_b = [], False
                for op in t.get("ops", []):
                    kind = op.get("op", "mul"); key = _op_param_name(kind)
                    lo, hi = op.get("lower"), op.get("upper")
                    if _has_bounds(lo, hi):
                        any_b = True
                        bundles.append([{**op, key: float(lo), "source": "extreme_lower"}, {**op, key: float(hi), "source": "extreme_upper"}])
                    else:
                        bundles.append([{**op}])
                per_tf_opts.append([{"ops": [dict(b) for b in combo]} for combo in product(*bundles)] if any_b else [{}])
            elif tt == "split":
                outs = []
                for o in t.get("outputs", []):
                    lo, hi = o.get("lower"), o.get("upper"); name = o.get("name")
                    if _has_bounds(lo, hi):
                        outs.append([{"name": name, "ratio": float(lo), "source": "extreme_lower"}, {"name": name, "ratio": float(hi), "source": "extreme_upper"}])
                    else:
                        outs.append([{"name": name}])
                per_tf_opts.append([{"outputs": list(combo)} for combo in product(*outs)])
            elif tt == "build_point":
                # Build combinations for mg/L values (per pollutant)
                mg_sets = []
                for var, meta in t.get("mgL_values", {}).items():
                    lo, hi = meta.get("lower"), meta.get("upper")
                    if _has_bounds(lo, hi):
                        mg_sets.append([
                            {var: {**meta, "mgL": float(lo), "source": "extreme_lower"}},
                            {var: {**meta, "mgL": float(hi), "source": "extreme_upper"}},
                        ])
                    else:
                        std = meta.get("mgL", meta.get("standard", meta.get("mean", 0.0)))
                        mg_sets.append([{var: {**meta, "mgL": float(std), "source": meta.get("source", "standard")}}])

                mg_combos = []
                for combo in product(*mg_sets):
                    merged = {}
                    for d in combo:
                        merged.update(d)
                    mg_combos.append({"mgL_values": merged})

                # Build combinations for wastewater_lppd bounds if provided
                lppd_spec = t.get("wastewater_lppd")
                lppd_combos = [{ }]
                try:
                    if isinstance(lppd_spec, dict):
                        lo, hi = lppd_spec.get("lower"), lppd_spec.get("upper")
                        std = lppd_spec.get("standard", lppd_spec.get("mean"))
                        if _has_bounds(lo, hi):
                            lppd_combos = [
                                {"wastewater_lppd": {"value": float(lo), "source": "extreme_lower"}},
                                {"wastewater_lppd": {"value": float(hi), "source": "extreme_upper"}},
                            ]
                        elif std is not None:
                            lppd_combos = [{"wastewater_lppd": {"value": float(std), "source": "standard"}}]
                    elif lppd_spec is not None:
                        lppd_combos = [{"wastewater_lppd": {"value": float(lppd_spec), "source": "fixed"}}]
                except Exception:
                    pass

                combos = []
                for a, b in product(mg_combos, lppd_combos):
                    m = {}
                    m.update(a)
                    m.update(b)
                    combos.append(m)
                per_tf_opts.append(combos)
            else:
                per_tf_opts.append([{}])
        return [list(c) for c in product(*per_tf_opts)]

    def lower_upper():
        lowers = []
        uppers = []
        for t in spec.get("transforms", []):
            tt = t["type"]
            if tt == "ops":
                low_ops, up_ops = [], []
                for op in t.get("ops", []):
                    op = dict(op); key = _op_param_name(op.get("op", "mul"))
                    lo, hi = op.get("lower"), op.get("upper")
                    low_ops.append({**op, key: float(lo), "source": "all_lower"} if _has_bounds(lo, hi) else dict(op))
                    up_ops.append({**op, key: float(hi), "source": "all_upper"} if _has_bounds(lo, hi) else dict(op))
                lowers.append({"ops": low_ops}); uppers.append({"ops": up_ops})
            elif tt == "split":
                low_outs, up_outs = [], []
                for o in t.get("outputs", []):
                    o = dict(o); lo, hi = o.get("lower"), o.get("upper"); name = o.get("name")
                    l = {"name": name}; u = {"name": name}
                    if _has_bounds(lo, hi):
                        l.update({"ratio": float(lo), "source": "all_lower"})
                        u.update({"ratio": float(hi), "source": "all_upper"})
                    low_outs.append(l); up_outs.append(u)
                lowers.append({"outputs": low_outs}); uppers.append({"outputs": up_outs})
            elif tt == "build_point":
                low_map, up_map = {}, {}
                for var, meta in t.get("mgL_values", {}).items():
                    meta = dict(meta); lo, hi = meta.get("lower"), meta.get("upper")
                    if _has_bounds(lo, hi):
                        low_map[var] = {**meta, "mgL": float(lo), "source": "all_lower"}
                        up_map[var] = {**meta, "mgL": float(hi), "source": "all_upper"}
                    else:
                        std = meta.get("mgL", meta.get("standard", meta.get("mean", 0.0)))
                        low_map[var] = {**meta, "mgL": float(std), "source": meta.get("source", "standard")}
                        up_map[var] = {**meta, "mgL": float(std), "source": meta.get("source", "standard")}
                low_tf = {"mgL_values": low_map}
                up_tf = {"mgL_values": up_map}
                # wastewater_lppd bounds
                lppd_spec = t.get("wastewater_lppd")
                try:
                    if isinstance(lppd_spec, dict):
                        lo, hi = lppd_spec.get("lower"), lppd_spec.get("upper")
                        std = lppd_spec.get("standard", lppd_spec.get("mean"))
                        if _has_bounds(lo, hi):
                            low_tf["wastewater_lppd"] = {"value": float(lo), "source": "all_lower"}
                            up_tf["wastewater_lppd"] = {"value": float(hi), "source": "all_upper"}
                        elif std is not None:
                            low_tf["wastewater_lppd"] = {"value": float(std), "source": "standard"}
                            up_tf["wastewater_lppd"] = {"value": float(std), "source": "standard"}
                    elif lppd_spec is not None:
                        v = float(lppd_spec)
                        low_tf["wastewater_lppd"] = {"value": v, "source": "fixed"}
                        up_tf["wastewater_lppd"] = {"value": v, "source": "fixed"}
                except Exception:
                    pass
                lowers.append(low_tf); uppers.append(up_tf)
            else:
                lowers.append({}); uppers.append({})
        return [lowers, uppers]

    def random(draws: int, seed: Optional[int]):
        import random as _random

        rnd = _random.Random(seed)
        runs: List[List[Dict[str, Any]]] = []
        for _ in range(draws):
            per_tf: List[Dict[str, Any]] = []
            for t in spec.get("transforms", []):
                tt = t["type"]
                if tt == "ops":
                    bundle = []
                    for op in t.get("ops", []):
                        op = dict(op); key = _op_param_name(op.get("op", "mul"))
                        lo, hi = op.get("lower"), op.get("upper")
                        if _has_bounds(lo, hi):
                            op[key] = rnd.uniform(float(lo), float(hi)); op["source"] = "random"
                        bundle.append(op)
                    per_tf.append({"ops": bundle})
                elif tt == "split":
                    outs = []
                    for o in t.get("outputs", []):
                        lo, hi = o.get("lower"), o.get("upper"); name = o.get("name")
                        if _has_bounds(lo, hi):
                            outs.append({"name": name, "ratio": rnd.uniform(float(lo), float(hi)), "source": "random"})
                        else:
                            outs.append({"name": name})
                    per_tf.append({"outputs": outs})
                elif tt == "build_point":
                    mg = {}
                    for var, meta in t.get("mgL_values", {}).items():
                        lo, hi = meta.get("lower"), meta.get("upper")
                        if _has_bounds(lo, hi):
                            mg[var] = {**meta, "mgL": rnd.uniform(float(lo), float(hi)), "source": "random"}
                        else:
                            std = meta.get("mgL", meta.get("standard", meta.get("mean", 0.0)))
                            mg[var] = {**meta, "mgL": float(std), "source": meta.get("source", "standard")}
                    step = {"mgL_values": mg}
                    # wastewater_lppd random draw if bounds exist
                    lppd_spec = t.get("wastewater_lppd")
                    try:
                        if isinstance(lppd_spec, dict):
                            lo, hi = lppd_spec.get("lower"), lppd_spec.get("upper")
                            std = lppd_spec.get("standard", lppd_spec.get("mean"))
                            if _has_bounds(lo, hi):
                                step["wastewater_lppd"] = {"value": rnd.uniform(float(lo), float(hi)), "source": "random"}
                            elif std is not None:
                                step["wastewater_lppd"] = {"value": float(std), "source": "standard"}
                        elif lppd_spec is not None:
                            step["wastewater_lppd"] = {"value": float(lppd_spec), "source": "fixed"}
                    except Exception:
                        pass
                    per_tf.append(step)
                else:
                    per_tf.append({})
            runs.append(per_tf)
        return runs

    def mean():
        per_tf: List[Dict[str, Any]] = []
        for t in spec.get("transforms", []):
            tt = t["type"]
            if tt == "ops":
                bundle = []
                for op in t.get("ops", []):
                    op = dict(op); key = _op_param_name(op.get("op", "mul"))
                    std = op.get("standard", op.get("mean", op.get(key)))
                    op[key] = float(std); op["source"] = "standard"
                    bundle.append(op)
                per_tf.append({"ops": bundle})
            elif tt == "split":
                outs = []
                for o in t.get("outputs", []):
                    outs.append({"name": o["name"], "ratio": float(o.get("standard", o.get("mean"))), "source": "standard"})
                per_tf.append({"outputs": outs})
            elif tt == "build_point":
                mg = {}
                for var, meta in t.get("mgL_values", {}).items():
                    std = float(meta.get("standard", meta.get("mean", meta.get("mgL", 0.0))))
                    mg[var] = {**meta, "mgL": std, "source": "standard"}
                step = {"mgL_values": mg}
                # wastewater_lppd mean/standard
                lppd_spec = t.get("wastewater_lppd")
                try:
                    if isinstance(lppd_spec, dict):
                        std = lppd_spec.get("standard", lppd_spec.get("mean"))
                        if std is not None:
                            step["wastewater_lppd"] = {"value": float(std), "source": "standard"}
                    elif lppd_spec is not None:
                        step["wastewater_lppd"] = {"value": float(lppd_spec), "source": "fixed"}
                except Exception:
                    pass
                per_tf.append(step)
            else:
                per_tf.append({})
        return [per_tf]

    mode = mode.strip().lower()
    if mode in ("minmax", "lower_upper"):
        runs = lower_upper()
    elif mode == "extreme":
        runs = extreme()
    elif mode == "random":
        runs = random(draws, seed)
    elif mode == "mean":
        runs = mean()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Pad for initial copy step so overrides align with transform list
    if runs and isinstance(runs[0], list):
        runs = [[{}] + step_list for step_list in runs]
    return runs


def run_from_spec(
    *,
    spec: Dict[str, Any],
    aggregator: Callable[[], Dict[str, Any]],
    base_txtinout: Path,
    realization_root: Path,
    results_root: Path,
    link_file_regexes: List[str],
    outputs_to_copy: List[str],
    config: Optional[Dict[str, Any]] = None,
    manifest_file: Optional[Path] = None,
    run_model: bool = True,
    upstream_inputs: Optional[List[Path]] = None,
    # New: aggressively remove stale inputs before linking to ensure overwrite
    preclean_input_globs: Optional[List[str]] = None,
    preclean_linked_inputs: bool = False,
):
    transforms, transforms_params = _build_transforms_and_params(
        spec, manifest_file=manifest_file, debug=bool(spec.get("debug", False))
    )
    overrides = _build_overrides(
        spec,
        mode=str(spec.get("mode", "mean")),
        draws=int(spec.get("draws", 1)),
        seed=spec.get("seed"),
    )

    results = run_monte_carlo(
        N=len(overrides),
        base_txtinout=Path(base_txtinout),
        realization_root=Path(realization_root),
        results_root=Path(results_root),
        link_file_regexes=link_file_regexes,
        outputs_to_copy=outputs_to_copy,
        aggregator=aggregator,
        transforms=transforms,
        transforms_params=transforms_params,
        per_realization_params=overrides,
        exe_path=None,
        seed=int(spec.get("seed", 0)),
        expect_plus=bool(spec.get("expect_plus", False)),
        config=config,
        include_base_run=bool(spec.get("include_base_run", False)),
        create_workspace_copy=bool(spec.get("create_workspace_copy", True)),
        force_recreate_workspace=bool(spec.get("force_recreate_workspace", True)),
        report=True,
        run_model=run_model,
        upstream_inputs=upstream_inputs,
        manifest_file=manifest_file if manifest_file and Path(manifest_file).exists() else None,
        auto_attach_manifest=True,
        preclean_input_globs=preclean_input_globs,
        preclean_linked_inputs=preclean_linked_inputs,
    )

    # Write provenance reports (report.txt, summary_table.md, report.docx) in r.folder
    write_provenance_reports(results)
    
    return results
