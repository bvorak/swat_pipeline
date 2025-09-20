from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from ..provenance import RealizationProvenance
from ..utils import get_logger


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
        log.info(
            "[debug] scale %s -> %s | mode=%s | factor=%.6f | mean=%.6f lower=%s upper=%s",
            src,
            out,
            mode,
            factor,
            mean,
            lower,
            upper,
        )
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
    """
    log = get_logger(__name__)
    df = data.copy()
    ops = params.get("ops", []) or []
    input_source = params.get("input_source")
    applied = []

    for spec in ops:
        op = spec.get("op") or spec.get("mode")
        src = spec.get("src")
        out = spec.get("out") or src
        mean = spec.get("mean")
        lower = spec.get("lower")
        upper = spec.get("upper")
        source = spec.get("source", "n/a")

        if op in ("mul", "relative"):
            factor = float(spec.get("factor", 1.0))
            df[out] = df[src].astype(float) * factor
            val = factor
            opname = "mul"
        elif op in ("add", "absolute"):
            delta = float(spec.get("delta", 0.0))
            df[out] = df[src].astype(float) + delta
            val = delta
            opname = "add"
        elif op == "set":
            if "value" in spec:
                df[out] = float(spec.get("value"))
                val = float(spec.get("value"))
            elif src is not None:
                df[out] = df[src]
                val = "from_src"
            else:
                raise ValueError("'set' op requires 'value' or 'src'")
            opname = "set"
        else:
            raise ValueError(f"Unsupported op: {op}")

        if params.get("debug"):
            log.info(
                "[debug] op %s: %s -> %s | value=%s | mean=%s lower=%s upper=%s",
                opname,
                src,
                out,
                val,
                mean,
                lower,
                upper,
            )
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
    outs = params.get("outputs", [])
    renorm = bool(params.get("renormalize", True))
    input_source = params.get("input_source")

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

    for o, frac in zip(outs, final):
        name = o.get("name")
        df[name] = df[src].astype(float) * float(frac)
        if params.get("debug"):
            log.info(
                "[debug] split %s -> %s | ratio=%.6f (mean=%s lower=%s upper=%s) renorm=%s",
                src,
                name,
                frac,
                o.get("mean"),
                o.get("lower"),
                o.get("upper"),
                renorm,
            )

    if rp:
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

